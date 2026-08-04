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
    wire_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CENSUS_CALL_SITES = (
    REPO_ROOT / "crates/chelis-compiler-api/tests/capacity_census_wire.rs",
    REPO_ROOT / "crates/chelis-python/tests/capacity_census_bindings.rs",
)


def serialized_struct_document(field_type: dict | None) -> dict:
    fields = [3] if field_type is not None else []
    index = {
        "1": {
            "id": 1,
            "name": "ReviewerWireNumericProbe",
            "span": {"filename": "crates/chelis-compiler-api/src/schema.rs"},
            "visibility": "public",
            "inner": {
                "struct": {
                    "kind": {"plain": {"fields": fields, "has_stripped_fields": False}},
                    "impls": [2],
                }
            },
        },
        "2": {"inner": {"impl": {"trait": {"path": "Serialize"}}}},
    }
    if field_type is not None:
        index["3"] = {
            "name": "value",
            "inner": {"struct_field": field_type},
        }
    return {
        "index": index,
        "paths": {
            "1": {
                "path": [
                    "chelis_compiler_api",
                    "schema",
                    "ReviewerWireNumericProbe",
                ],
                "kind": "struct",
            }
        },
    }


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


class WireEnumerator(unittest.TestCase):
    def test_public_serialized_f64_addition_and_removal_change_rows(self) -> None:
        before = wire_rows(serialized_struct_document(None))
        after = wire_rows(serialized_struct_document({"primitive": "f64"}))
        self.assertEqual(before, [])
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]["flags"], ["float-carrier"])
        self.assertIn("ReviewerWireNumericProbe.value: f64", after[0]["id"])
        self.assertEqual(wire_rows(serialized_struct_document(None)), before)

    def test_nonserialized_public_type_is_not_a_wire_carrier(self) -> None:
        document = serialized_struct_document({"primitive": "f64"})
        document["index"]["1"]["inner"]["struct"]["impls"] = []
        self.assertEqual(wire_rows(document), [])


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

    def test_no_census_call_site_passes_its_own_target_dir(self) -> None:
        # The drift guard. A call site that reintroduces `--target-dir` opts
        # its leg back out of the shared build without any other signal.
        for path in CENSUS_CALL_SITES:
            source = path.read_text()
            invocations = [
                line
                for line in source.splitlines()
                if "--target-dir" in line and not re.match(r"\s*//", line)
            ]
            self.assertEqual(
                invocations,
                [],
                f"{path.name} passes --target-dir; it must let the enumerator "
                f"choose {SHARED_RUSTDOC_TARGET_DIR} so both legs share one build",
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

    def test_call_site_guard_reads_the_real_files(self) -> None:
        # Guard the guard: a renamed or moved census test must fail loudly
        # here rather than silently checking nothing.
        for path in CENSUS_CALL_SITES:
            self.assertTrue(path.is_file(), f"census call site missing: {path}")
            self.assertIn("capacity_census_typed.py", path.read_text())


if __name__ == "__main__":
    unittest.main()
