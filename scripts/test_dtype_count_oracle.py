#!/usr/bin/env python3

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_count_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_every_count_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs(sys.executable)],
            [
                "checker axis and dtype contract",
                "dedicated Count IR and evaluator",
                "exact WireDag v6 and registered wire capacity",
                "compiled C exact multi-axis and empty execution",
                "loud HIP and Metal issue receipts",
                "semantic registration and executable example parity",
                "generated rejection registry agreement",
            ],
        )

    def test_manifest_pins_dedicated_ir_wire_c_and_example_oracles(self) -> None:
        commands = [" ".join(leg.argv) for leg in oracle.oracle_legs(sys.executable)]
        joined = "\n".join(commands)
        for required in (
            "issue_1287_count_checker",
            "issue_1287_count",
            "adjacent_pair_fold_preserves_canonical_tree",
            "count_add_traps_int64_overflow",
            "wire_dag_v6_count",
            "capacity_census_wire",
            "exec_count_",
            "count_with_issue_1291_receipt",
            "issue_1287_count_device_reject",
            "count_is_registered_against_its_exact_authority_atom",
            "parity_count_bool_axes",
            "parity_corpus_is_complete",
            "generate_rejection_registries.py --check",
        ):
            self.assertIn(required, joined)

    def test_device_work_is_rejection_only_in_this_oracle(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("chelis-backend-hip", leg.argv, leg.name)
            self.assertNotIn("chelis-backend-metal", leg.argv, leg.name)
            self.assertNotIn("--ignored", leg.argv, leg.name)


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(7, []),
        ]
        with self.assertRaisesRegex(SystemExit, "dedicated Count IR"):
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
