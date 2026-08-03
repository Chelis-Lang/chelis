#!/usr/bin/env python3

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_phase0_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_every_phase0_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs()],
            [
                "domain detector and active dtype matrices",
                "byte-exact executable-corpus parity",
                "legacy floating agreement controls",
            ],
        )

    def test_detector_leg_is_the_frozen_phase0_surface(self) -> None:
        leg = oracle.oracle_legs()[0]
        tests = {
            leg.argv[index + 1]
            for index, arg in enumerate(leg.argv[:-1])
            if arg == "--test"
        }
        self.assertEqual(
            tests,
            {
                "domain_checker",
                "eval_tensor_narrowing_matrix",
                "narrow_dtype_matrix",
                "precision_matrix",
                "int_width_lane_matrix",
                "reduction_and_bitwise_matrix",
                "scalar_stub_matrix",
                "issue_680_int_exactness",
            },
        )

    def test_oracle_never_runs_ignored_rows(self) -> None:
        for leg in oracle.oracle_legs():
            self.assertNotIn("--ignored", leg.argv, leg.name)


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(7, []),
        ]
        with self.assertRaisesRegex(SystemExit, "byte-exact executable-corpus parity"):
            oracle.run_oracle()
        self.assertEqual(run.call_count, 2)

    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_executes_every_leg_from_repository_root(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        oracle.run_oracle()
        self.assertEqual(run.call_count, len(oracle.oracle_legs()))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
