#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from unittest import mock


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
        self.assertEqual(
            command[command.index("--profile") + 1],
            "ci-full",
            "the flattened CI oracle must publish JUnit telemetry without "
            "changing its explicit filterset",
        )
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

    def test_selector_translation_rejects_unsafe_or_ambiguous_inputs(self) -> None:
        cases = (
            (
                ("cargo", "test", "--workspace"),
                "not a nextest run command",
            ),
            (
                ("cargo", "nextest", "run"),
                "selects no tests",
            ),
            (
                ("cargo", "nextest", "run", "--workspace"),
                "unrecognized dtype-oracle nextest selector",
            ),
            (
                (
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-cli) | all()",
                ),
                "unsafe nextest package name",
            ),
            (
                (
                    "cargo",
                    "nextest",
                    "run",
                    "-E",
                    "package(=chelis-cli)",
                    "-E",
                    "all()",
                ),
                "multiple filtersets",
            ),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, message):
                    manifest.command_filter(argv)

    def test_owner_filter_rejects_an_unknown_owner(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown dtype phase owner"):
            manifest.owner_filter("not-a-phase", sys.executable)

    def test_recursive_manifest_inheritance_cycle_fails_closed(self) -> None:
        script = "scripts/injected_dtype_phase3_cycle.py"
        cycle = manifest.OracleLeg(
            "injected recursive inheritance",
            (sys.executable, script),
        )
        with mock.patch.dict(
            manifest._INHERITED_SCRIPTS,
            {script: "dtype-phase3"},
        ), mock.patch.object(
            manifest,
            "phase3_legacy_legs",
            return_value=(cycle,),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "dtype oracle inheritance cycle: dtype-phase3 -> dtype-phase3",
            ):
                manifest.owned_leaf_legs(sys.executable)

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
        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(receipt["nextest_invocations_before_flattening"], 21)
        self.assertEqual(receipt["nextest_invocations_after_flattening"], 1)
        self.assertEqual(receipt["non_test_invocations_before_flattening"], 3)
        self.assertEqual(receipt["non_test_invocations_after_flattening"], 3)
        self.assertEqual(set(receipt["owners"]), set(manifest.OWNERS))
        self.assertEqual(set(receipt["non_test_owners"]), set(manifest.OWNERS))
        self.assertEqual(len(receipt["filter_sha256"]), 64)

    def test_future_inherited_non_test_leg_is_executed_and_receipted(self) -> None:
        original = manifest.dtype_phase0_oracle.oracle_legs()
        injected = manifest.dtype_phase0_oracle.OracleLeg(
            "injected inherited non-test obligation",
            (sys.executable, "scripts/injected_dtype_obligation.py"),
        )
        with mock.patch.object(
            manifest.dtype_phase0_oracle,
            "oracle_legs",
            return_value=original + (injected,),
        ):
            legs = manifest.non_test_legs(sys.executable)
            receipt = manifest.ownership_receipt(sys.executable)

        self.assertIn(tuple(injected.argv), [leg.argv for leg in legs])
        self.assertIn(
            {
                "name": injected.name,
                "argv": list(injected.argv),
            },
            receipt["non_test_owners"]["dtype-phase0"],
        )


if __name__ == "__main__":
    unittest.main()
