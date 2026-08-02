#!/usr/bin/env python3
"""Unit tests for the typed wire and PyO3 capacity enumerators."""

from __future__ import annotations

import unittest

from capacity_census_typed import CensusError, binding_rows, wire_rows


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


if __name__ == "__main__":
    unittest.main()
