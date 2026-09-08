#!/usr/bin/env python3
"""Tests for Kache executable-cache report classification."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kache_executable_cache_regression import (
    ReportFailure,
    checkout_target,
    contains_bytes,
    isolated_environment,
    kache_store,
    restored_executable_candidates,
    validate_legacy_namespace_report,
    validate_debug_uuid_outputs,
    validate_warm_report,
    write_isolated_kache_config,
)


class KacheWarmReportTests(unittest.TestCase):
    def test_each_clean_clone_owns_its_target_directory(self) -> None:
        checkout = Path("/tmp/probe/warm")
        self.assertEqual(checkout_target(checkout), checkout / "target")

    def test_cold_and_warm_runs_share_only_a_fresh_probe_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            cache_dir = root / "kache-cache"
            config = root / "probe-kache.toml"
            write_isolated_kache_config(
                "[cache]\nignore_env = true\ncache_executables = true\n",
                config,
                cache_dir,
            )
            first = isolated_environment(
                {"KACHE_CONFIG": "/host/config"},
                target=root / "target-cold",
                config=config,
            )
            second = isolated_environment(
                {"KACHE_CONFIG": "/other/host/config"},
                target=root / "target-warm",
                config=config,
            )
            self.assertEqual(first["KACHE_CONFIG"], str(config))
            self.assertEqual(second["KACHE_CONFIG"], str(config))
            self.assertEqual(first["KACHE_LOG_FILE"], "kache=debug")
            self.assertEqual(
                first["KACHE_LOG_FILE_PATH"], str(root / "kache-wrapper.log")
            )
            self.assertNotEqual(first["CARGO_TARGET_DIR"], second["CARGO_TARGET_DIR"])
            self.assertIn(f"local_store = {json.dumps(str(cache_dir))}", config.read_text())
            self.assertEqual(kache_store(cache_dir), cache_dir / "store")

    def test_isolated_config_rejects_an_existing_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            with self.assertRaisesRegex(ReportFailure, "already sets local_store"):
                write_isolated_kache_config(
                    '[cache]\nlocal_store = "/host/cache"\n',
                    root / "probe-kache.toml",
                    root / "cache",
                )

    def test_warm_report_requires_hits_and_restored_bytes(self) -> None:
        report = {
            "summary": {"local_hits": 4},
            "storage": {"restored_bytes": 1024},
            "bypass": {"reasons": []},
        }
        validate_warm_report(report)

    def test_uncached_executable_reason_is_rejected(self) -> None:
        report = {
            "summary": {"local_hits": 4},
            "storage": {"restored_bytes": 1024},
            "bypass": {
                "reasons": [
                    {
                        "reason": "user-facing executable (cache_executables=false)",
                        "count": 2,
                    }
                ]
            },
        }
        with self.assertRaisesRegex(ReportFailure, "executable bypass"):
            validate_warm_report(report)

    def test_zero_hit_report_is_rejected(self) -> None:
        report = {
            "summary": {"local_hits": 0},
            "storage": {"restored_bytes": 0},
            "bypass": {"reasons": []},
        }
        with self.assertRaisesRegex(ReportFailure, "no local hits"):
            validate_warm_report(report)

    def test_legacy_schema_entry_must_not_be_a_current_local_hit(self) -> None:
        report = {
            "summary": {"local_hits": 1, "misses": 0},
            "all_events": [
                {"crate_name": "schema_probe", "result": "local_hit"}
            ],
        }
        with self.assertRaisesRegex(ReportFailure, "reused a schema-27 entry"):
            validate_legacy_namespace_report(report)

    def test_legacy_schema_probe_accepts_only_current_misses(self) -> None:
        report = {
            "summary": {"local_hits": 0, "misses": 2},
            "all_events": [
                {"crate_name": "schema_probe", "result": "miss"},
                {"crate_name": "schema_probe", "result": "miss"},
            ],
        }
        validate_legacy_namespace_report(report)

    def test_local_hit_cache_metadata_identifies_the_exact_restored_executable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            store = Path(raw_directory) / "store"
            key = "a" * 64
            entry = store / key
            entry.mkdir(parents=True)
            (entry / "meta.json").write_text(
                json.dumps(
                    {
                        "cache_key": key,
                        "key_schema": 28,
                        "crate_name": "phase3_stamped_ingress",
                        "files": [
                            {
                                "name": "phase3_stamped_ingress-a1b2",
                                "executable": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidates = [
                Path("phase3_stamped_ingress-a1b2"),
                Path("phase3_stamped_ingress-c3d4"),
            ]
            report = {
                "all_events": [
                    {
                        "cache_key": key,
                        "crate_name": "phase3_stamped_ingress",
                        "result": "local_hit",
                    }
                ]
            }
            self.assertEqual(
                restored_executable_candidates(candidates, report, store),
                candidates[:1],
            )

    def test_local_hit_with_missing_metadata_fails_closed(self) -> None:
        report = {
            "all_events": [
                {
                    "cache_key": "b" * 64,
                    "crate_name": "suite",
                    "result": "local_hit",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            with self.assertRaisesRegex(ReportFailure, "metadata is absent"):
                restored_executable_candidates(
                    [Path("suite-a1b2")], report, Path(raw_directory)
                )

    def test_crate_prefix_without_an_exact_executable_metadata_row_is_not_restored(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            store = Path(raw_directory) / "store"
            key = "c" * 64
            entry = store / key
            entry.mkdir(parents=True)
            (entry / "meta.json").write_text(
                json.dumps(
                    {
                        "cache_key": key,
                        "key_schema": 28,
                        "crate_name": "suite",
                        "files": [{"name": "suite-other", "executable": True}],
                    }
                ),
                encoding="utf-8",
            )
            report = {
                "all_events": [
                    {"cache_key": key, "crate_name": "suite", "result": "local_hit"}
                ]
            }
            with self.assertRaisesRegex(ReportFailure, "exact executable metadata"):
                restored_executable_candidates([Path("suite-a1b2")], report, store)

    def test_local_hit_with_a_metadata_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            store = Path(raw_directory) / "store"
            key = "d" * 64
            entry = store / key
            entry.mkdir(parents=True)
            (entry / "meta.json").write_text(
                json.dumps(
                    {
                        "cache_key": "e" * 64,
                        "key_schema": 28,
                        "crate_name": "suite",
                        "files": [{"name": "suite-a1b2", "executable": True}],
                    }
                ),
                encoding="utf-8",
            )
            report = {
                "all_events": [
                    {"cache_key": key, "crate_name": "suite", "result": "local_hit"}
                ]
            }
            with self.assertRaisesRegex(ReportFailure, "key mismatch"):
                restored_executable_candidates([Path("suite-a1b2")], report, store)

    def test_legacy_executable_cache_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            store = Path(raw_directory) / "store"
            key = "f" * 64
            entry = store / key
            entry.mkdir(parents=True)
            (entry / "meta.json").write_text(
                json.dumps(
                    {
                        "cache_key": key,
                        "key_schema": 27,
                        "crate_name": "suite",
                        "files": [{"name": "suite-a1b2", "executable": True}],
                    }
                ),
                encoding="utf-8",
            )
            report = {
                "all_events": [
                    {"cache_key": key, "crate_name": "suite", "result": "local_hit"}
                ]
            }
            with self.assertRaisesRegex(ReportFailure, "schema mismatch"):
                restored_executable_candidates([Path("suite-a1b2")], report, store)

    def test_dsym_uuid_check_rejects_empty_bundle_output(self) -> None:
        binary = "UUID: 01234567-89AB-CDEF-0123-456789ABCDEF (arm64) binary\n"
        with self.assertRaisesRegex(ReportFailure, "bundle has no UUID"):
            validate_debug_uuid_outputs(binary, "")

    def test_dsym_uuid_check_rejects_malformed_bundle_output(self) -> None:
        binary = "UUID: 01234567-89AB-CDEF-0123-456789ABCDEF (arm64) binary\n"
        malformed = "UUID: not-a-mach-o-uuid (arm64) bundle\n"
        with self.assertRaisesRegex(ReportFailure, "bundle has no UUID"):
            validate_debug_uuid_outputs(binary, malformed)

    def test_dsym_uuid_check_rejects_mismatched_bundle(self) -> None:
        binary = "UUID: 01234567-89AB-CDEF-0123-456789ABCDEF (arm64) binary\n"
        bundle = "UUID: FEDCBA98-7654-3210-FEDC-BA9876543210 (arm64) bundle\n"
        with self.assertRaisesRegex(ReportFailure, "UUID mismatch"):
            validate_debug_uuid_outputs(binary, bundle)

    def test_dsym_uuid_check_accepts_an_exact_multi_arch_match(self) -> None:
        uuids = (
            "UUID: 01234567-89AB-CDEF-0123-456789ABCDEF (arm64) artifact\n"
            "UUID: FEDCBA98-7654-3210-FEDC-BA9876543210 (x86_64) artifact\n"
        )
        self.assertEqual(
            validate_debug_uuid_outputs(uuids, uuids),
            frozenset(
                {
                    "01234567-89AB-CDEF-0123-456789ABCDEF",
                    "FEDCBA98-7654-3210-FEDC-BA9876543210",
                }
            ),
        )

    def test_binary_scan_finds_a_path_split_across_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            artifact = Path(raw_directory) / "artifact"
            artifact.write_bytes(b"abc/cold/checkout/xyz")
            self.assertTrue(contains_bytes(artifact, b"/cold/checkout", chunk_size=8))
            self.assertTrue(contains_bytes(artifact, b"x", chunk_size=1))
            self.assertFalse(contains_bytes(artifact, b"/other/checkout", chunk_size=8))
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                contains_bytes(artifact, b"")


if __name__ == "__main__":
    unittest.main()
