#!/usr/bin/env python3
"""Unit and mutation tests for the chelis#1281 completion oracle."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_exact_reductions_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_each_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs("/chosen/python")],
            [
                "oracle self-tests",
                "typed reduction kernels",
                "IR evaluation and adjoints",
                "compiled C exact reductions",
                "checker reduction domain",
                "downstream exhaustive consumers",
            ],
        )

    def test_self_test_uses_the_oracle_interpreter(self) -> None:
        self.assertEqual(
            oracle.oracle_legs("/chosen/python")[0].argv,
            ("/chosen/python", "scripts/test_dtype_exact_reductions_oracle.py"),
        )

    def test_ir_and_c_legs_run_dedicated_issue_targets(self) -> None:
        legs = {leg.name: leg.argv for leg in oracle.oracle_legs(sys.executable)}
        self.assertIn("issue_1281_exact_reductions", legs["IR evaluation and adjoints"])
        self.assertIn(
            "issue_1281_exact_reductions",
            legs["compiled C exact reductions"],
        )

    def test_manifest_does_not_claim_ignored_device_execution(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("--ignored", leg.argv, leg.name)
            self.assertNotIn("scripts/hip_test.py", leg.argv, leg.name)


class SourceContractTests(unittest.TestCase):
    def test_current_tree_satisfies_declared_source_contracts(self) -> None:
        oracle.validate_source_contracts(oracle.REPO_ROOT)

    def test_obsolete_c_window_extrema_contract_is_rejected(self) -> None:
        source = (
            "existing tie/NaN arithmetic remains tracked by #1298\n"
            "fmaxf(acc, value)\n"
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "compiled C window extrema"):
            oracle.validate_source_text(
                "compiled C window extrema",
                source,
                required=("first NaN",),
                forbidden=(
                    "existing tie/NaN arithmetic remains tracked by #1298",
                    "fmaxf(acc, value)",
                ),
            )


if __name__ == "__main__":
    unittest.main()
