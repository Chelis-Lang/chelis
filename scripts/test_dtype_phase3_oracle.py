#!/usr/bin/env python3

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_phase3_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_every_phase3_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs(sys.executable)],
            [
                "inherited Phase 2 contract",
                "inherited observation Phase 3 contract",
                "compiled C dtype matrix",
                "compiled C structural locks",
                "numeric surface censuses",
            ],
        )

    def test_compiled_matrix_leg_names_every_phase3_issue_fixture(self) -> None:
        leg = next(
            leg
            for leg in oracle.oracle_legs(sys.executable)
            if leg.name == "compiled C dtype matrix"
        )
        tests = {
            leg.argv[index + 1]
            for index, arg in enumerate(leg.argv[:-1])
            if arg == "--test"
        }
        self.assertEqual(
            tests,
            {
                "narrow_dtype_matrix",
                "int_width_lane_matrix",
                "precision_matrix",
                "scalar_stub_matrix",
                "reduction_and_bitwise_matrix",
                "fold_static_cond_matrix",
                "issue_759_checked_cast_default",
                "issue_761_subnormal_ingress",
                "issue_734_tostring_placeholder",
                "observation_roundtrip_harness",
            },
        )

    def test_oracle_never_runs_ignored_or_non_c_backends(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("--ignored", leg.argv, leg.name)
            self.assertNotIn("chelis-backend-hip", leg.argv, leg.name)
            self.assertNotIn("chelis-backend-metal", leg.argv, leg.name)

    def test_inherited_leg_uses_the_oracle_interpreter(self) -> None:
        inherited = oracle.oracle_legs("/chosen/python")[0]
        self.assertEqual(inherited.argv[0], "/chosen/python")
        self.assertEqual(inherited.argv[1], "scripts/dtype_phase2_oracle.py")


class RunnerTests(unittest.TestCase):
    def _run_with_target(self, target: str) -> None:
        with mock.patch.dict(oracle.os.environ, {"CARGO_TARGET_DIR": target}):
            oracle.run_oracle(sys.executable)

    @mock.patch.object(oracle.faithful, "receipt_violations", return_value=[])
    @mock.patch.object(oracle.faithful, "preflight_violations", return_value=[])
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_executes_one_nextest_union_then_each_non_test_leg(
        self,
        run: mock.Mock,
        _preflight: mock.Mock,
        _receipts: mock.Mock,
    ) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as target:
            self._run_with_target(target)
        calls = [tuple(call.args[0]) for call in run.call_args_list]
        self.assertEqual(
            sum(command[:3] == ("cargo", "nextest", "run") for command in calls),
            1,
        )
        self.assertEqual(
            calls[1:],
            [leg.argv for leg in oracle.manifest.non_test_legs(sys.executable)],
        )
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])

    @mock.patch.object(oracle.faithful, "preflight_violations", return_value=[])
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_when_the_flattened_nextest_union_fails(
        self, run: mock.Mock, _preflight: mock.Mock
    ) -> None:
        run.side_effect = subprocess.CalledProcessError(7, ["cargo"])
        with tempfile.TemporaryDirectory() as target:
            with self.assertRaisesRegex(SystemExit, "flattened nextest union"):
                self._run_with_target(target)
        self.assertEqual(run.call_count, 1)

    @mock.patch.object(
        oracle.faithful,
        "preflight_violations",
        return_value=["static contract changed"],
    )
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_before_execution_on_preflight_failure(
        self, run: mock.Mock, _preflight: mock.Mock
    ) -> None:
        with self.assertRaisesRegex(SystemExit, "static contract changed"):
            oracle.run_oracle(sys.executable)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
