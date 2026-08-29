#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_oracle_manifest as manifest  # noqa: E402


class FlattenedManifestTests(unittest.TestCase):
    def test_every_inherited_nextest_leg_has_one_phase_owner(self) -> None:
        legs = manifest.owned_nextest_legs(sys.executable)
        self.assertEqual(len(legs), 21)
        self.assertEqual(
            {owner: sum(leg.owner == owner for leg in legs) for owner in manifest.OWNERS},
            {
                "dtype-phase0": 3,
                "dtype-phase1": 7,
                "dtype-phase2": 6,
                "observation-phase3": 3,
                "dtype-phase3": 2,
            },
        )
        self.assertTrue(
            all(leg.argv[:3] == ("cargo", "nextest", "run") for leg in legs)
        )

    def test_recursive_nextest_commands_flatten_to_one_exact_union(self) -> None:
        command = manifest.flattened_nextest_command(sys.executable)
        self.assertEqual(command[:4], ("cargo", "nextest", "run", "--workspace"))
        self.assertIn("--ignore-default-filter", command)
        self.assertIn("--no-fail-fast", command)
        self.assertEqual(command.count("-E"), 1)
        expression = command[command.index("-E") + 1]
        for selector in (
            "package(=chelis-cli)",
            "binary(=parity)",
            "package(=chelis-e2e)",
            "binary(=eval_agreement)",
            "package(=chelis-backend-c)",
            "binary(=exec_compile)",
        ):
            self.assertIn(selector, expression)

    def test_only_non_test_obligations_remain_as_separate_processes(self) -> None:
        legs = manifest.non_test_legs(sys.executable)
        self.assertEqual(
            [leg.name for leg in legs],
            [
                "Python dtype ingress",
                "Hull tagged-value reader",
                "numeric surface censuses",
            ],
        )
        for leg in legs:
            self.assertNotEqual(leg.argv[:3], ("cargo", "nextest", "run"))
            self.assertFalse(
                any(
                    arg.endswith(
                        (
                            "dtype_phase0_oracle.py",
                            "dtype_phase1_oracle.py",
                            "dtype_phase2_oracle.py",
                            "faithful_observation_phase3_oracle.py",
                        )
                    )
                    for arg in leg.argv
                ),
                leg,
            )

    def test_ownership_receipt_records_the_before_and_after_counts(self) -> None:
        receipt = manifest.ownership_receipt(sys.executable)
        self.assertEqual(receipt["schema_version"], 1)
        self.assertEqual(receipt["nextest_invocations_before_flattening"], 21)
        self.assertEqual(receipt["nextest_invocations_after_flattening"], 1)
        self.assertEqual(set(receipt["owners"]), set(manifest.OWNERS))
        self.assertEqual(len(receipt["filter_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
