#!/usr/bin/env python3
"""Unit tests for the typed wire and PyO3 capacity enumerators."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event
import unittest
from pathlib import Path
from unittest.mock import patch

from capacity_census_typed import (
    WIRE_RUSTDOC_TARGET_DIR,
    CensusError,
    binding_rows,
    build_parser,
    resolve_target_dir,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CENSUS_CALL_SITES = (
    REPO_ROOT / "tests/support/capacity_census_wire_verifier.rs",
    REPO_ROOT / "tests/support/capacity_census_compiler_json.rs",
)
# Flags that let a call site choose build inputs. Both exist for ad-hoc local
# runs; neither belongs in the source-bound Rust census call sites.
FORBIDDEN_CALL_SITE_FLAGS = ("--target-dir", "--rustdoc-json")


def binding_document(input_type: dict) -> dict:
    return {
        "index": {
            "7": {
                "name": "reviewer_raw_dtype_probe",
                "inner": {
                    "function": {
                        "sig": {
                            "inputs": [["dtype", input_type]],
                            "output": {"primitive": "i32"},
                        }
                    }
                },
            }
        },
        "paths": {
            "7": {
                "path": ["chelis_python", "reviewer_raw_dtype_probe"],
                "kind": "function",
            }
        },
    }


class BindingEnumerator(unittest.TestCase):
    def test_registered_dtype_i32_is_raw_dtype_and_unregistered_is_absent(self) -> None:
        document = binding_document({"primitive": "i32"})
        self.assertEqual(binding_rows(document, []), [])
        rows = binding_rows(document, ["reviewer_raw_dtype_probe"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["flags"], ["raw-dtype-int"])

    def test_registered_callable_without_rustdoc_signature_fails_closed(self) -> None:
        with self.assertRaisesRegex(CensusError, "no top-level rustdoc JSON signature"):
            binding_rows(binding_document({"primitive": "i32"}), ["not_in_rustdoc"])

    def test_registered_method_without_rustdoc_signature_fails_closed(self) -> None:
        with self.assertRaisesRegex(CensusError, "registered PyO3 method"):
            binding_rows(
                binding_document({"primitive": "i32"}),
                [],
                ["NativeTensor::reviewer_dtype"],
            )


class RustdocTargetDirs(unittest.TestCase):
    def test_wire_and_binding_use_disjoint_worktree_targets(self) -> None:
        from capacity_census_typed import BINDING_RUSTDOC_TARGET_DIR

        root = Path("/workspace")
        self.assertEqual(
            resolve_target_dir(root, None, "wire"),
            root / WIRE_RUSTDOC_TARGET_DIR,
        )
        self.assertEqual(
            resolve_target_dir(root, None, "bindings-discovery"),
            root / BINDING_RUSTDOC_TARGET_DIR,
        )
        self.assertNotEqual(WIRE_RUSTDOC_TARGET_DIR, BINDING_RUSTDOC_TARGET_DIR)

    def test_directories_are_isolated_from_the_ambient_target(self) -> None:
        # A nested cargo pointed at the outer `cargo nextest` build's target
        # directory would contend with that build's lock.
        from capacity_census_typed import BINDING_RUSTDOC_TARGET_DIR

        for path in (WIRE_RUSTDOC_TARGET_DIR, BINDING_RUSTDOC_TARGET_DIR):
            parts = path.parts
            self.assertEqual(parts[:2], ("target", "agents"))
            self.assertGreater(len(parts), 2)

    def test_explicit_relative_target_dir_is_anchored_to_the_root(self) -> None:
        self.assertEqual(
            resolve_target_dir(Path("/workspace"), Path("target/agents/ad-hoc"), "bindings"),
            Path("/workspace/target/agents/ad-hoc"),
        )

    def test_explicit_absolute_target_dir_is_honored_verbatim(self) -> None:
        self.assertEqual(
            resolve_target_dir(Path("/workspace"), Path("/elsewhere/rustdoc"), "wire"),
            Path("/elsewhere/rustdoc"),
        )

    def test_no_census_call_site_supplies_target_or_document(self) -> None:
        # The enumerator chooses its targets; a supplied rustdoc document
        # would skip live Cargo execution at the Rust census call site.
        for flag in FORBIDDEN_CALL_SITE_FLAGS:
            for path in CENSUS_CALL_SITES:
                source = path.read_text()
                invocations = [
                    line
                    for line in source.splitlines()
                    if flag in line and not re.match(r"\s*//", line)
                ]
                self.assertEqual(
                    invocations,
                    [],
                    f"{path.name} passes {flag}; it must let the enumerator "
                    "choose its own worktree target and census live documents",
                )

    def test_target_dir_has_exactly_one_spelling(self) -> None:
        # argparse's default prefix abbreviation would accept `--t`, which
        # evades the literal call-site guard above. Locked with
        # allow_abbrev=False.
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["wire", "--t", "/tmp/abbreviated"])
        self.assertEqual(
            parser.parse_args(["wire", "--target-dir", "/tmp/explicit"]).target_dir,
            Path("/tmp/explicit"),
        )

    def test_live_wire_guard_uses_the_private_verification_bridge(self) -> None:
        path = REPO_ROOT / "crates/chelis-compiler-api/tests/capacity_census_wire.rs"
        source = path.read_text()
        self.assertIn('tests/support/capacity_census_wire_verifier.rs', source)
        self.assertIn('capacity_census_wire_verifier::discover()', source)

    def test_live_binding_guard_uses_the_private_execution_bridge(self) -> None:
        path = REPO_ROOT / "crates/chelis-python/tests/capacity_census_bindings.rs"
        source = path.read_text()
        self.assertIn('tests/support/capacity_census_compiler_json.rs', source)
        self.assertIn('capacity_census_compiler_json::discover(', source)
        bridge = (REPO_ROOT / "tests/support/capacity_census_compiler_json.rs").read_text()
        self.assertIn('"bindings-discovery"', bridge)
        self.assertNotIn('pub fn from_', bridge)

    def test_call_site_guard_reads_the_real_files(self) -> None:
        # Guard the guard: a renamed or moved census test must fail loudly
        # here rather than silently checking nothing.
        for path in CENSUS_CALL_SITES:
            self.assertTrue(path.is_file(), f"census call site missing: {path}")
            self.assertIn("capacity_census_typed.py", path.read_text())


class ConcurrentBindingProof(unittest.TestCase):
    class Wire:
        source_sha256 = "current"

        def validate(self):
            pass

    class Native:
        source_sha256 = "current"

        def validate(self):
            pass

    class Work:
        source_sha256 = "current"

    class CompilerJson:
        def validate(self):
            pass

    def mocked_proof(self, collect=None, wire=None, native=None, source=None):
        stack = ExitStack()
        stack.enter_context(patch("capacity_census_wire_adapters.source_identity",
                                  side_effect=source or (lambda root: "current")))
        stack.enter_context(patch("capacity_census_wire_verifier.VerifiedWireCensus", self.Wire))
        stack.enter_context(patch("capacity_census_native_authority.VerifiedNativeBindings", self.Native))
        stack.enter_context(patch("capacity_census_compiler_json._CompilerJsonWork", self.Work))
        stack.enter_context(patch("capacity_census_compiler_json.collect_compiler_json_bindings",
                                  side_effect=collect or (lambda root, target: self.Work())))
        stack.enter_context(patch("capacity_census_native_authority.verify_native_bindings",
                                  side_effect=native or (lambda root, target: self.Native())))
        stack.enter_context(patch("capacity_census_wire_verifier.verify_wire_census",
                                  side_effect=wire or (lambda root, target: self.Wire())))
        finalizer = stack.enter_context(patch(
            "capacity_census_compiler_json.finalize_compiler_json_bindings",
            return_value=self.CompilerJson(),
        ))
        return stack, finalizer

    def run_proof(self, wire_target=None, binding_target=None):
        from capacity_census_typed import verify_binding_proofs

        root = Path("/workspace")
        return verify_binding_proofs(
            root,
            wire_target or root / WIRE_RUSTDOC_TARGET_DIR,
            binding_target or root / "target/agents/729-capacity-binding-proof",
        )

    def test_binding_work_overlaps_wire_and_no_success_precedes_both(self):
        binding_started, wire_started, wire_finished = Event(), Event(), Event()
        release_binding, release_wire = Event(), Event()
        targets = {}

        def collect(root, target):
            targets["binding"] = target
            binding_started.set()
            self.assertTrue(release_binding.wait(5))
            return self.Work()

        def wire(root, target):
            targets["wire"] = target
            wire_started.set()
            self.assertTrue(release_wire.wait(5))
            wire_finished.set()
            return self.Wire()

        def native(root, target):
            self.assertTrue(release_binding.is_set())
            targets["native"] = target
            return self.Native()

        stack, finalizer = self.mocked_proof(collect=collect, wire=wire, native=native)
        with stack, ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(self.run_proof)
            try:
                self.assertTrue(binding_started.wait(5))
                self.assertTrue(wire_started.wait(5))
                self.assertFalse(result.done())
                self.assertFalse(finalizer.called)
                release_wire.set()
                self.assertTrue(wire_finished.wait(5))
                self.assertFalse(result.done())
                self.assertFalse(finalizer.called)
                release_binding.set()
                self.assertIsInstance(result.result(timeout=5)[0], self.CompilerJson)
            finally:
                release_wire.set()
                release_binding.set()
        self.assertTrue(finalizer.called)
        self.assertEqual(targets["wire"], Path("/workspace") / WIRE_RUSTDOC_TARGET_DIR)
        self.assertEqual(targets["binding"], Path("/workspace/target/agents/729-capacity-binding-proof"))
        self.assertEqual(targets["native"], targets["binding"])

    def test_wire_failure_still_waits_for_binding_worker(self):
        worker_started, wire_failed, release_worker = Event(), Event(), Event()

        def collect(root, target):
            worker_started.set()
            self.assertTrue(release_worker.wait(5))
            return self.Work()

        def wire(root, target):
            self.assertTrue(worker_started.wait(5))
            wire_failed.set()
            raise RuntimeError("wire failed")

        stack, finalizer = self.mocked_proof(collect=collect, wire=wire)
        with stack, ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(self.run_proof)
            try:
                self.assertTrue(worker_started.wait(5))
                self.assertTrue(wire_failed.wait(5))
                self.assertFalse(result.done())
                release_worker.set()
                with self.assertRaisesRegex(RuntimeError, "wire failed"):
                    result.result(timeout=5)
            finally:
                release_worker.set()
        finalizer.assert_not_called()

    def test_binding_failure_rejects_without_finalization(self):
        def collect(root, target):
            raise RuntimeError("binding failed")

        stack, finalizer = self.mocked_proof(collect=collect)
        with stack, self.assertRaisesRegex(RuntimeError, "binding failed"):
            self.run_proof()
        finalizer.assert_not_called()

    def test_two_failures_report_both_without_finalization(self):
        def collect(root, target):
            raise RuntimeError("binding failed")

        def wire(root, target):
            raise RuntimeError("wire failed")

        stack, finalizer = self.mocked_proof(collect=collect, wire=wire)
        with stack, self.assertRaisesRegex(CensusError, "wire failed.*binding failed"):
            self.run_proof()
        finalizer.assert_not_called()

    def test_changed_source_rejects_at_join(self):
        values = iter(("current", "changed"))
        stack, finalizer = self.mocked_proof(source=lambda root: next(values))
        with stack, self.assertRaisesRegex(Exception, "source"):
            self.run_proof()
        finalizer.assert_not_called()

    def test_saved_wire_receipt_cannot_replace_live_witness(self):
        stack, finalizer = self.mocked_proof(
            wire=lambda root, target: {"source_sha256": "current", "report": "saved"}
        )
        with stack, self.assertRaisesRegex(Exception, "wire"):
            self.run_proof()
        finalizer.assert_not_called()

    def test_overlapping_or_foreign_targets_reject_before_work(self):
        for wire, binding in (
            ("target/agents/shared", "target/agents/shared"),
            ("target/agents/shared", "target/agents/shared/nested"),
            ("target/agents/shared", "../foreign"),
        ):
            stack, finalizer = self.mocked_proof()
            with stack, self.subTest(wire=wire, binding=binding), self.assertRaises(Exception):
                self.run_proof(Path("/workspace") / wire, Path("/workspace") / binding)
            finalizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
