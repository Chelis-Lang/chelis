"""Failure controls for syntax parity and resource evidence."""

from __future__ import annotations

from copy import deepcopy
import unittest

from no_std_core_support import OracleFailure
from syntax_predicate_core_oracle import CASES, OPERATIONS, compare_snapshots


def evidence() -> dict:
    semantic = {name: {"wire": name} for name in CASES}
    semantic.update({"predicate": {}, "errors": [], "span": {"offset": 2, "len": 5}, "path_error": "mismatch"})
    return {
        "semantic": semantic,
        "costs": {
            case: {
                operation: {"iterations": 8, "allocations": 8,
                            "allocated_bytes": 800, "nanoseconds": 1000}
                for operation in OPERATIONS
            }
            for case in CASES
        },
    }


class SyntaxEvidenceTests(unittest.TestCase):
    def test_wire_mismatch_rejects_other_matching_evidence(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["semantic"]["identity"]["wire"] = "different"
        with self.assertRaisesRegex(OracleFailure, "wire parity differs"):
            compare_snapshots(baseline, candidate)

    def test_two_empty_semantic_records_do_not_prove_parity(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        baseline["semantic"] = {}
        candidate["semantic"] = {}
        with self.assertRaisesRegex(OracleFailure, "incomplete syntax semantic"):
            compare_snapshots(baseline, candidate)

    def test_missing_case_rejects_other_measurements(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        del candidate["costs"]["tensor"]
        with self.assertRaisesRegex(OracleFailure, "incomplete syntax cost cases"):
            compare_snapshots(baseline, candidate)

    def test_missing_operation_rejects_other_measurements(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        del candidate["costs"]["macro"]["expand"]
        with self.assertRaisesRegex(OracleFailure, "incomplete syntax cost operations"):
            compare_snapshots(baseline, candidate)

    def test_allocation_growth_cannot_hide_in_another_operation(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["costs"]["macro"]["expand"]["allocations"] += 1
        candidate["costs"]["macro"]["parse"]["allocations"] -= 1
        with self.assertRaisesRegex(OracleFailure, "allocations regression: macro/expand"):
            compare_snapshots(baseline, candidate)

    def test_byte_growth_rejects_equal_allocation_counts(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["costs"]["tensor"]["resugar"]["allocated_bytes"] += 1
        with self.assertRaisesRegex(OracleFailure, "allocated_bytes regression: tensor/resugar"):
            compare_snapshots(baseline, candidate)

    def test_reduced_iteration_count_does_not_prove_lower_cost(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["costs"]["identity"]["parse"]["iterations"] = 4
        with self.assertRaisesRegex(OracleFailure, "unequal syntax cost work"):
            compare_snapshots(baseline, candidate)

    def test_invalid_measurements_cannot_prove_acceptance(self) -> None:
        for value in (-1, "8", True, None):
            with self.subTest(value=value):
                baseline = evidence()
                candidate = deepcopy(baseline)
                candidate["costs"]["integer-float"]["format"]["allocations"] = value
                with self.assertRaisesRegex(OracleFailure, "invalid syntax cost measurement"):
                    compare_snapshots(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
