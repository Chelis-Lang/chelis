#!/usr/bin/env python3

import subprocess
import sys
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


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(7, []),
        ]
        with self.assertRaisesRegex(SystemExit, "observation Phase 3"):
            oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, 2)

    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_executes_every_leg_from_repository_root(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, len(oracle.oracle_legs(sys.executable)))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
