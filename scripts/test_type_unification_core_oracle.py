#!/usr/bin/env python3
"""Check mandatory failure channels for the type core oracle."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import type_unification_core_oracle as oracle


def snapshot_fixture() -> dict:
    accepted = {"status": "accepted", "typed_nodes": 1, "total_nodes": 1,
                "checked_bytes": [1], "tree": ["unit"], "types": {"unit": "unit"}}
    error = {"kind": "TypeMismatch", "identifier": None, "message": "control diagnostic", "suggestions": [],
             "severity_bits": 0, "expected": None, "got": None,
             "span_offset": None, "span_id": None}
    rejected = {"status": "rejected", "typed_nodes": 0, "total_nodes": 1,
                "diagnostics": [error]}
    rejected_cases = {"unbound", "precision", "branch", "bad_nominal"}
    return {
        "cases": {name: deepcopy(rejected if name in rejected_cases or name.startswith("rejected_") else accepted)
                  for name in oracle.HOST_CASES},
        "context": {"bytes": [1], "threads": 8,
                    "checks": [deepcopy(accepted), deepcopy(accepted), deepcopy(rejected)]},
    }


class TypeUnificationCoreOracleTests(unittest.TestCase):
    def test_each_required_failure_prevents_success(self) -> None:
        stages = (
            "portable_graphs", "core_feature_isolation", "target_consumer", "solver_consumer",
            "authority_fixtures", "immutable_parity", "host_parity", "python_controls",
        )
        for failed_stage in stages:
            with self.subTest(stage=failed_stage), ExitStack() as stack:
                output = stack.enter_context(redirect_stdout(io.StringIO()))
                errors = stack.enter_context(redirect_stderr(io.StringIO()))
                stack.enter_context(patch.object(oracle.sys, "platform", "linux"))
                for stage in stages:
                    failure = oracle.OracleFailure(f"{stage} rejected") if stage == failed_stage else None
                    stack.enter_context(patch.object(oracle, stage, side_effect=failure))
                self.assertEqual(oracle.main(), 1)
                self.assertNotIn("PASS", output.getvalue())
                self.assertIn(f"{failed_stage} rejected", errors.getvalue())

    def test_missing_target_stops_host_execution(self) -> None:
        with (
            patch.object(oracle, "portable", side_effect=oracle.OracleFailure("target unavailable")),
            patch.object(oracle, "target_consumer") as target,
            patch.object(oracle, "solver_consumer") as solver,
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "target unavailable"):
                oracle.run_oracle()
        target.assert_not_called()
        solver.assert_not_called()

    def test_serde_graph_failure_is_mandatory(self) -> None:
        with (
            patch.object(oracle, "portable", side_effect=[set(), oracle.OracleFailure("serde enables std")]),
            patch.object(oracle, "target_consumer") as target,
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "serde enables std"):
                oracle.run_oracle()
        target.assert_not_called()

    def test_unexpected_compile_success_cannot_reach_host_parity(self) -> None:
        with (
            patch.object(oracle, "portable_graphs"),
            patch.object(oracle, "target_consumer"),
            patch.object(oracle, "solver_consumer"),
            patch.object(oracle, "compile_rejected", side_effect=oracle.OracleFailure("fixture unexpectedly passed")),
            patch.object(oracle, "host_parity") as parity,
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "unexpectedly passed"):
                oracle.run_oracle()
        parity.assert_not_called()

    def test_allocator_build_failure_is_not_ignored(self) -> None:
        with patch.object(oracle, "run", side_effect=oracle.OracleFailure("allocator target failed")):
            with self.assertRaisesRegex(oracle.OracleFailure, "allocator target failed"):
                oracle.target_consumer()

    def test_solver_assertion_failure_is_not_ignored(self) -> None:
        with patch.object(oracle, "consumer", side_effect=oracle.OracleFailure("compression assertion failed")):
            with self.assertRaisesRegex(oracle.OracleFailure, "compression assertion failed"):
                oracle.solver_consumer()

    def test_each_host_suite_failure_is_mandatory(self) -> None:
        for index in range(7):
            with (
                self.subTest(suite=index),
                patch.object(oracle, "run", side_effect=[
                    *([None] * index), oracle.OracleFailure("host parity failed"),
                ]),
            ):
                with self.assertRaisesRegex(oracle.OracleFailure, "host parity failed"):
                    oracle.host_parity()

    def test_each_supported_host_records_its_platform(self) -> None:
        for platform in ("linux", "darwin"):
            with (
                self.subTest(platform=platform),
                patch.object(oracle.sys, "platform", platform),
                patch.object(oracle, "run_oracle"),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(oracle.main(), 0)
                self.assertIn(platform, output.getvalue())
                self.assertIn("No device execution", output.getvalue())

    def test_unsupported_host_cannot_claim_acceptance(self) -> None:
        with (
            patch.object(oracle.sys, "platform", "win32"),
            patch.object(oracle, "run_oracle") as checks,
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(oracle.main(), 1)
        checks.assert_not_called()
        self.assertNotIn("PASS", output.getvalue())
        self.assertIn("Linux or macOS", errors.getvalue())

    def test_empty_matching_snapshots_cannot_prove_parity(self) -> None:
        with self.assertRaisesRegex(oracle.OracleFailure, "incomplete"):
            oracle.compare_snapshots({}, {})

    def test_every_host_case_is_mandatory_on_both_sides(self) -> None:
        for name in oracle.HOST_CASES:
            for side in (False, True):
                before, after = snapshot_fixture(), snapshot_fixture()
                del (after if side else before)["cases"][name]
                with self.subTest(case=name, candidate=side):
                    with self.assertRaisesRegex(oracle.OracleFailure, "inventory"):
                        oracle.compare_snapshots(before, after)

    def test_each_exact_output_field_controls_parity(self) -> None:
        changes = (
            ("generic", "checked_bytes", [2]),
            ("generic", "tree", ["different"]),
            ("generic", "types", {"unit": "different"}),
            ("generic", "typed_nodes", 0),
            ("unbound", "diagnostics", [{
                "kind": "TypeMismatch", "identifier": None, "message": "different", "suggestions": [],
                "severity_bits": 0, "expected": None, "got": None,
                "span_offset": 1, "span_id": "different",
            }]),
        )
        for case, field, value in changes:
            before, after = snapshot_fixture(), snapshot_fixture()
            after["cases"][case][field] = value
            with self.subTest(case=case, field=field):
                with self.assertRaisesRegex(oracle.OracleFailure, "differ"):
                    oracle.compare_snapshots(before, after)

    def test_missing_context_and_empty_diagnostics_fail(self) -> None:
        for section in ("context", "diagnostics"):
            before, after = snapshot_fixture(), snapshot_fixture()
            if section == "context":
                del after["context"]["checks"]
            else:
                after["cases"]["unbound"]["diagnostics"] = []
            with self.subTest(section=section):
                with self.assertRaises(oracle.OracleFailure):
                    oracle.compare_snapshots(before, after)

    def test_core_serde_leak_stops_positive_consumer(self) -> None:
        with (
            patch.object(oracle, "compile_rejected", side_effect=oracle.OracleFailure("core serde leaked")),
            patch.object(oracle, "consumer") as positive,
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "core serde leaked"):
                oracle.core_feature_isolation()
        positive.assert_not_called()

    def test_core_serde_positive_consumer_is_mandatory(self) -> None:
        with (
            patch.object(oracle, "compile_rejected"),
            patch.object(oracle, "consumer", side_effect=oracle.OracleFailure("core serde failed")),
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "core serde failed"):
                oracle.core_feature_isolation()


if __name__ == "__main__":
    unittest.main()
