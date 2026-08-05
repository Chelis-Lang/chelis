#!/usr/bin/env python3

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import loud_unsupported_phase3_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_each_phase3_obligation_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs(sys.executable)],
            [
                "shared gate and rejected-cell contract",
                "sealed diagnostic-kind contract",
                "typed rejection-authority tests",
                "rejection-authority boundary",
                "live issue-authority manifest",
                "root-realizability integration",
            ],
        )

    def test_gate_leg_covers_inventory_parity_and_rejected_cells(self) -> None:
        leg = oracle.oracle_legs(sys.executable)[0]
        rendered = " ".join(leg.argv)
        self.assertIn("phase3_gate_contract", rendered)
        self.assertIn("phase3_gate_inventory", rendered)
        self.assertIn("issue_687_rejected_cells_corpus", rendered)

    def test_root_realizability_is_the_explicit_final_interlock(self) -> None:
        leg = oracle.oracle_legs(sys.executable)[-1]
        self.assertIn("issue_912_root_boundary", leg.argv)
        self.assertIn("--run-ignored", leg.argv)
        self.assertEqual(leg.argv[-1], "all")

    def test_python_legs_use_the_oracle_interpreter(self) -> None:
        legs = oracle.oracle_legs("/chosen/python")
        for index in [1, 3, 4]:
            self.assertEqual(legs[index].argv[0], "/chosen/python")


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(7, []),
        ]
        with self.assertRaisesRegex(SystemExit, "sealed diagnostic-kind contract"):
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
