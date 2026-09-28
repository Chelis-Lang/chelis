"""Selection and evidence-integrity tests; these are not compiler receipts."""

import copy
from dataclasses import replace
import unittest

from capacity_census_graph import GraphError
from capacity_census_wire_adapters import check_observations
from capacity_census_wire_materialization import materialization_cases


class MaterializationCases(unittest.TestCase):
    def test_each_source_field_has_its_own_transport_and_admission_evidence(self):
        from capacity_census_wire_adapters import ExecutionOutcome
        from capacity_census_wire_fixed_roles import compiler_fixed_field_contracts
        from capacity_census_wire_materialization import (
            source_field_evidence,
            validate_materialization_execution,
        )

        expected = {
            c.field
            for c in compiler_fixed_field_contracts()
            if c.role in {"source-integer", "source-float"}
        }
        self.assertEqual(set(source_field_evidence()), expected)
        cases = materialization_cases()
        # Synthetic outcomes exercise reconciliation only, not the compiler.
        outcomes = tuple(
            ExecutionOutcome(c.identity, True, True, "passed", "a" * 64) for c in cases
        )
        selected = validate_materialization_execution(cases, outcomes)
        self.assertTrue(selected)
        for field, pairs in source_field_evidence().items():
            self.assertGreaterEqual(len(pairs), 2, field)
            for positive, negative in pairs:
                for identity in (positive, negative):
                    with self.subTest(field=field, missing=identity):
                        with self.assertRaisesRegex(
                            GraphError, "materialization execution"
                        ):
                            validate_materialization_execution(
                                cases,
                                tuple(o for o in outcomes if o.identity != identity),
                            )
        first = next(o for o in outcomes if o.identity in selected)
        for changed in (
            replace(first, selected=False),
            replace(first, executed=False),
            replace(first, outcome="skipped"),
        ):
            with self.assertRaisesRegex(GraphError, "materialization execution"):
                validate_materialization_execution(
                    cases,
                    tuple(
                        changed if o.identity == first.identity else o for o in outcomes
                    ),
                )

    def test_selection_is_nonempty_unique_and_pairs_every_route(self):
        cases = materialization_cases()
        self.assertTrue(cases)
        self.assertEqual(len(cases), len({case.identity for case in cases}))
        routes = {(case.carrier, case.codec) for case in cases}
        for route in routes:
            selected = [c for c in cases if (c.carrier, c.codec) == route]
            self.assertTrue(any(c.expected is None for c in selected), route)
            # Opaque data has no valid runtime-stamping counterpart by design;
            # RawSourceAdmission/runtime supplies the actual expression control.
            if route != ("ExtensionData", "runtime"):
                self.assertTrue(any(c.expected is not None for c in selected), route)

    def test_independent_expectations_keep_wide_integer_and_negative_zero(self):
        cases = {case.identity: case for case in materialization_cases()}
        wide = cases["SourceProgram/eval-deep/int64-wide"]
        zero = cases["SourceProgram/eval-deep/f64-negative-zero"]
        self.assertEqual(
            wide.expected["roots"][0]["value"]["value"]["value"], 9007199254740993
        )
        self.assertEqual(
            zero.expected["roots"][0]["value"]["value"]["bits"], "8000000000000000"
        )
        tampered = copy.deepcopy(wide.expected)
        tampered["roots"][0]["value"]["value"]["value"] = 9007199254740992
        with self.assertRaisesRegex(GraphError, "wrong codec observation"):
            check_observations([wide], [{"id": wide.identity, "observation": tampered}])
        with self.assertRaisesRegex(GraphError, "not all executed"):
            check_observations([wide], [])

    def test_rejections_need_actual_matching_diagnostics(self):
        case = next(c for c in materialization_cases() if c.rejection_contains)
        with self.assertRaisesRegex(GraphError, "missing actual decode rejection"):
            check_observations([case], [{"id": case.identity, "observation": None}])
        with self.assertRaisesRegex(GraphError, "wrong decode rejection reason"):
            check_observations(
                [case],
                [
                    {
                        "id": case.identity,
                        "observation": None,
                        "decode_error": "unrelated",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
