#!/usr/bin/env python3

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_phase1_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_every_phase1_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs(sys.executable)],
            [
                "inherited Phase 0 contract",
                "covered-family Rust and C tripwire",
                "typed execution-wire census",
                "registered PyO3 signature census",
                "sealed dtype semantics",
                "typed Load ingress",
                "execution-wire exactness",
                "Phase 1 payload and checker contracts",
                "Python dtype ingress",
                "Hull tagged-value reader",
            ],
        )

    def test_phase1_leg_does_not_repeat_phase0_matrices(self) -> None:
        leg = next(
            leg
            for leg in oracle.oracle_legs(sys.executable)
            if leg.name == "Phase 1 payload and checker contracts"
        )
        tests = {
            leg.argv[index + 1]
            for index, arg in enumerate(leg.argv[:-1])
            if arg == "--test"
        }
        self.assertEqual(
            tests,
            {
                "issue_729_payload_census",
                "issue_759_checked_cast_default",
                "issue_860_checker_chokepoint",
            },
        )

    def test_inherited_leg_uses_the_oracle_interpreter(self) -> None:
        inherited = oracle.oracle_legs("/chosen/python")[0]
        self.assertEqual(
            inherited.argv,
            ("/chosen/python", "scripts/dtype_phase0_oracle.py"),
        )

    def test_no_leg_runs_ignored_or_c_backend_phase_rows(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("--ignored", leg.argv, leg.name)
            self.assertNotIn("chelis-backend-c", leg.argv, leg.name)

    def test_python_legs_use_the_oracle_interpreter(self) -> None:
        python_legs = oracle.oracle_legs("/chosen/python")[-2:]
        self.assertTrue(all(leg.argv[0] == "/chosen/python" for leg in python_legs))


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 7),
        ]
        with self.assertRaisesRegex(SystemExit, "covered-family Rust and C tripwire"):
            oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, 2)

    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_executes_every_leg_from_the_repository_root(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, len(oracle.oracle_legs(sys.executable)))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
