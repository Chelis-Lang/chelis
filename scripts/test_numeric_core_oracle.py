#!/usr/bin/env python3
"""Failure controls for complete numeric acceptance evidence."""

import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numeric_core_oracle as oracle
from no_std_core_support import OracleFailure


def costs():
    return [{"operation": op, "elements": n, "allocations": 1,
        "allocated_bytes": n * 8, "nanoseconds": n}
        for n in oracle.SCALES for op in oracle.COST_OPERATIONS]


class NumericCoreOracleTests(unittest.TestCase):
    def test_complete_feature_coverage(self):
        oracle.require_features(set(oracle.FEATURES))
        for feature in oracle.FEATURES:
            with self.subTest(feature=feature):
                with self.assertRaisesRegex(OracleFailure, "feature coverage"):
                    oracle.require_features(set(oracle.FEATURES) - {feature})

    def test_each_evidence_group_is_required(self):
        oracle.require_evidence(set(oracle.EVIDENCE))
        for group in oracle.EVIDENCE:
            with self.subTest(group=group):
                with self.assertRaisesRegex(OracleFailure, "missing evidence"):
                    oracle.require_evidence(set(oracle.EVIDENCE) - {group})

    def test_empty_and_duplicate_records_fail(self):
        for text in ("", "NUMERIC {}\nNUMERIC {}\n", "NUMERIC invalid\n", "NUMERIC []\n"):
            with self.subTest(text=text):
                with self.assertRaises(OracleFailure):
                    oracle.record(text, "NUMERIC", {"value"})

    def test_missing_record_field_fails(self):
        with self.assertRaisesRegex(OracleFailure, "fields"):
            oracle.record('NUMERIC {"value":1}\n', "NUMERIC", {"value", "wire"})
        self.assertEqual(oracle.record('NUMERIC {"value":1}\n', "NUMERIC", {"value"}), {"value": 1})

    def test_exact_comparison_rejects_changed_or_absent_fixtures(self):
        oracle.compare_exact({"integer": [1, 2]}, {"integer": [1, 2]}, "wire")
        for candidate in ({}, {"integer": [1, 3]}):
            with self.assertRaisesRegex(OracleFailure, "wire"):
                oracle.compare_exact({"integer": [1, 2]}, candidate, "wire")

    def test_cost_measurements_are_required(self):
        valid = costs()
        oracle.inspect_costs(valid)
        for field in ("operation", "allocations", "allocated_bytes", "nanoseconds"):
            invalid = [dict(row) for row in valid]
            invalid[0].pop(field)
            with self.assertRaises(OracleFailure):
                oracle.inspect_costs(invalid)
        with self.assertRaisesRegex(OracleFailure, "cost coverage"):
            oracle.inspect_costs(valid[:-1])
        for invalid in (None, [1, 2, 3], [dict(row, nanoseconds=0) for row in valid],
            [dict(row, allocations=True) for row in valid]):
            with self.assertRaises(OracleFailure):
                oracle.inspect_costs(invalid)

    def test_per_element_allocation_regression_fails(self):
        baseline = costs()
        candidate = [dict(row, allocations=row["elements"]) for row in baseline]
        with self.assertRaisesRegex(OracleFailure, "allocation"):
            oracle.compare_costs(baseline, candidate)

    def test_requested_byte_regression_fails(self):
        baseline = costs()
        candidate = [dict(row, allocated_bytes=row["allocated_bytes"] + 1) for row in baseline]
        with self.assertRaisesRegex(OracleFailure, "allocation"):
            oracle.compare_costs(baseline, candidate)

    def test_each_tensor_operation_is_required(self):
        for operation in oracle.COST_OPERATIONS:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(OracleFailure, "coverage"):
                    oracle.inspect_costs([row for row in costs() if row["operation"] != operation])

    def test_real_metadata_control_must_reject(self):
        with patch.object(oracle.support, "_normal_packages", return_value={oracle.CRATE, "chelis-vocab"}):
            with self.assertRaisesRegex(OracleFailure, "unexpectedly passed"):
                oracle.expect_graph_rejection({}, "std")

    def test_metadata_rejection_must_have_correct_reason(self):
        with patch.object(oracle.support, "_normal_packages", side_effect=OracleFailure("missing graph")):
            with self.assertRaisesRegex(OracleFailure, "another reason"):
                oracle.expect_graph_rejection({}, "std")

    def test_crmath_is_the_only_additional_numeric_target_owner(self):
        accepted = {oracle.CRATE, "chelis-vocab", "chelis-crmath", "chelis-no-std-consumer"}
        with patch.object(oracle.support, "_normal_packages", return_value=accepted):
            self.assertEqual(oracle.inspect_graph({}), accepted)
        for name in ("chelis-abi", "chelis-types", "chelis-new-host-service", "reef-runtime"):
            with self.subTest(name=name):
                with patch.object(oracle.support, "_normal_packages", return_value=accepted | {name}):
                    with self.assertRaisesRegex(OracleFailure, "forbidden"):
                        oracle.inspect_graph({})

    def test_numeric_owner_and_vocabulary_are_required(self):
        for names in ({oracle.CRATE}, {"chelis-vocab"}):
            with patch.object(oracle.support, "_normal_packages", return_value=names):
                with self.assertRaisesRegex(OracleFailure, "omits"):
                    oracle.inspect_graph({})

    def test_runtime_midpoint_mutation_fails(self):
        with self.assertRaisesRegex(OracleFailure, "runtime"):
            oracle.compare_exact({"rounding": [0x7a73, 0x3f81]},
                {"rounding": [0x7a74, 0x3f80]}, "runtime")

    def test_runtime_mutation_requires_numeric_failure(self):
        with patch.object(oracle.support, "_package"), patch.object(oracle.support, "_execute") as execute:
            execute.return_value = subprocess.CompletedProcess([], 101, "", "direct runtime rounding")
            oracle.runtime_mutation_rejected()
            execute.return_value = subprocess.CompletedProcess([], 0, "", "")
            with self.assertRaisesRegex(OracleFailure, "unexpectedly passed"):
                oracle.runtime_mutation_rejected()
            execute.return_value = subprocess.CompletedProcess([], 101, "", "cannot compile dependency")
            with self.assertRaisesRegex(OracleFailure, "another reason"):
                oracle.runtime_mutation_rejected()

    def test_baseline_has_exact_immutable_identity(self):
        with patch.object(oracle, "run") as run, patch.object(oracle.support, "_execute") as execute:
            execute.return_value = subprocess.CompletedProcess([], 0, "", "")
            run.return_value = subprocess.CompletedProcess([], 0, "wrong-revision\n", "")
            with self.assertRaisesRegex(OracleFailure, "identity"):
                oracle.baseline_clone(Path("/isolated"))
            run.return_value = subprocess.CompletedProcess([], 0, oracle.BASELINE_REVISION + "\n", "")
            self.assertEqual(oracle.baseline_clone(Path("/isolated")), Path("/isolated/baseline"))

    def test_missing_baseline_fetch_is_mandatory(self):
        with patch.object(oracle, "run") as run, patch.object(oracle.support, "_execute") as execute:
            execute.return_value = subprocess.CompletedProcess([], 1, "", "absent")
            run.side_effect = [subprocess.CompletedProcess([], 0, "", ""), OracleFailure("fetch failed")]
            with self.assertRaisesRegex(OracleFailure, "fetch failed"):
                oracle.baseline_clone(Path("/isolated"))

    def test_missing_surface_receipt_fails(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(oracle, "run", return_value=result):
            with self.assertRaisesRegex(OracleFailure, "surface"):
                oracle.surface_controls([])

    def test_python_evidence_needs_every_named_case(self):
        required = ("module.Tests.mandatory",)
        for value in (
            {"selected": [], "passed": [], "skipped": []},
            {"selected": [required[0]], "passed": [], "skipped": []},
            {"selected": [required[0]], "passed": [required[0]], "skipped": [required[0]]},
            {"selected": ["other"], "passed": ["other"], "skipped": []},
            {"selected": [required[0]], "passed": [required[0], required[0]], "skipped": []},
        ):
            with self.subTest(value=value):
                result = subprocess.CompletedProcess([], 0, "PYTHON " + json.dumps(value), "")
                with patch.object(oracle, "run", return_value=result):
                    with self.assertRaisesRegex(OracleFailure, "Python"):
                        oracle.python_tests(("module",), required)

    def test_failed_command_sets_nonzero_exit(self):
        with patch.object(oracle, "run_oracle", side_effect=OracleFailure("missing evidence")):
            self.assertEqual(oracle.main(), 1)


if __name__ == "__main__":
    unittest.main()
