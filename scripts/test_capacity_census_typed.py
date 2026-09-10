#!/usr/bin/env python3
"""Unit tests for the typed wire and PyO3 capacity enumerators."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from capacity_census_typed import (
    SHARED_RUSTDOC_TARGET_DIR,
    CensusError,
    binding_rows,
    build_parser,
    resolve_target_dir,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CENSUS_CALL_SITES = (
    REPO_ROOT / "tests/support/capacity_census_wire_verifier.rs",
    REPO_ROOT / "crates/chelis-python/tests/capacity_census_bindings.rs",
)
# Flags that let a call site decide what gets built or censused. Both exist
# for ad-hoc local runs; neither belongs in a test that guards a frozen
# baseline.
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


class SharedRustdocTargetDir(unittest.TestCase):
    """Both legs must land in ONE cargo target directory.

    This is a performance contract with a correctness-shaped failure mode: if
    the legs drift back onto separate directories, nothing fails, they just
    each recompile the shared chelis dependency graph and quietly become the
    slowest tests in the repository again. Only an executable check notices.
    """

    def test_both_legs_resolve_to_the_same_directory(self) -> None:
        root = Path("/workspace")
        self.assertEqual(
            resolve_target_dir(root, None),
            root / SHARED_RUSTDOC_TARGET_DIR,
        )

    def test_shared_directory_is_isolated_from_the_ambient_target(self) -> None:
        # A nested cargo pointed at the outer `cargo nextest` build's target
        # directory would contend with that build's lock.
        parts = SHARED_RUSTDOC_TARGET_DIR.parts
        self.assertEqual(parts[:2], ("target", "agents"))
        self.assertGreater(len(parts), 2)

    def test_explicit_relative_target_dir_is_anchored_to_the_root(self) -> None:
        self.assertEqual(
            resolve_target_dir(Path("/workspace"), Path("target/agents/ad-hoc")),
            Path("/workspace/target/agents/ad-hoc"),
        )

    def test_explicit_absolute_target_dir_is_honored_verbatim(self) -> None:
        self.assertEqual(
            resolve_target_dir(Path("/workspace"), Path("/elsewhere/rustdoc")),
            Path("/elsewhere/rustdoc"),
        )

    def test_no_census_call_site_opts_out_of_the_shared_build(self) -> None:
        # The drift guard. `--target-dir` opts a leg back out of the shared
        # build. `--rustdoc-json` is worse: it skips the cargo invocation
        # entirely and censuses whatever document it is handed, so a call site
        # spelled that way turns this guard artifact green in ~90ms having
        # built nothing. No caller passes either; the enumerator chooses.
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
                    f"build into {SHARED_RUSTDOC_TARGET_DIR} so both legs share "
                    f"one build and neither can census a supplied document",
                )

    def test_target_dir_has_exactly_one_spelling(self) -> None:
        # argparse's default prefix abbreviation would accept `--t`, which
        # splits the legs back apart in a spelling the call-site guard above
        # cannot see. Locked with allow_abbrev=False.
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

    def test_call_site_guard_reads_the_real_files(self) -> None:
        # Guard the guard: a renamed or moved census test must fail loudly
        # here rather than silently checking nothing.
        for path in CENSUS_CALL_SITES:
            self.assertTrue(path.is_file(), f"census call site missing: {path}")
            self.assertIn("capacity_census_typed.py", path.read_text())


if __name__ == "__main__":
    unittest.main()
