"""Spec-derived controls for the runtime-representation Phase 2 receipt."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import runtime_representation_phase2 as oracle


class ContractTests(unittest.TestCase):
    def test_exact_phase_two_leg_inventory_is_bounded_to_the_deliverable(self):
        self.assertEqual(
            [name for name, _ in oracle.phase2_legs()],
            [
                "canonical ABI metadata and generated layouts",
                "opaque C metadata-plan adapter",
                "HIP descriptor contracts",
                "HIP published owner header contract",
                "HIP pinned-runtime CPU execution contracts",
                "Python host device and DLPack boundaries",
                "backend runtime-header capacity census",
            ],
        )
        hip = dict(oracle.phase2_legs())["HIP descriptor contracts"]
        self.assertEqual(
            [hip[index + 1] for index, value in enumerate(hip) if value == "--test"],
            [
                "codegen_structure",
                "device_entry_contract",
            ],
        )
        runtime = dict(oracle.phase2_legs())[
            "HIP pinned-runtime CPU execution contracts"
        ]
        self.assertEqual(
            [runtime[index + 1] for index, value in enumerate(runtime) if value == "--test"],
            [
                "device_entry_execution",
                "device_owner_contract",
            ],
        )
        self.assertEqual(runtime[-2:], ("--run-ignored", "only"))

    def test_phase_two_rejects_development_shortcuts(self):
        for argv in (
            ["--phase", "2", "--skip-mutations"],
            ["--phase", "2", "--regenerate"],
        ):
            with self.subTest(argv=argv), self.assertRaises(oracle.OracleFailure):
                oracle.check_options(argv)
        oracle.check_options(["--phase", "2"])

    def test_python_leg_is_platform_invariant_and_bounded_to_phase_two(self):
        args = dict(oracle.phase2_legs())[
            "Python host device and DLPack boundaries"
        ]
        self.assertEqual(
            args,
            (
                "-p",
                "chelis-python",
                "-E",
                oracle.PYTHON_BOUNDARY_FILTER,
            ),
        )
        self.assertNotIn("compiled_host_outputs_have_no_runtime_metadata_leaks", args[-1])
        self.assertNotIn("manual_", args[-1])

    def test_phase_two_execute_leg_reports_and_executes_an_added_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            junit = root / "junit.xml"
            required = ["p::contract::negative"]
            addition = "p::contract::new_negative"
            selected = sorted([*required, addition])
            xml = (
                '<testsuites><testsuite name="p::contract">'
                '<testcase name="negative"/><testcase name="new_negative"/>'
                '</testsuite></testsuites>'
            )

            def command(argv, command_root, command_evidence, label):
                self.assertEqual(command_root, root)
                command_evidence.mkdir(parents=True, exist_ok=True)
                if label == "run":
                    junit.write_text(xml)
                return "{}"

            with (
                mock.patch.object(oracle, "ROOT", root),
                mock.patch.object(oracle.phase1, "command", side_effect=command),
                mock.patch.object(
                    oracle.phase1, "selection", return_value=(selected, {})
                ),
                mock.patch.object(oracle.phase1, "junit_path", return_value=junit),
            ):
                receipt = oracle.execute_leg(
                    "fixture",
                    (),
                    required,
                    evidence,
                )

            self.assertEqual(receipt["required"], required)
            self.assertEqual(receipt["selected"], selected)
            self.assertEqual(receipt["additions"], [addition])
            self.assertEqual(
                receipt["executed"],
                [{"id": identity, "outcome": "passed"} for identity in selected],
            )

    def test_phase_two_added_test_must_appear_in_framework_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            junit = root / "junit.xml"
            required = ["p::contract::negative"]
            selected = [*required, "p::contract::new_negative"]
            xml = (
                '<testsuites><testsuite name="p::contract">'
                '<testcase name="negative"/>'
                '</testsuite></testsuites>'
            )

            def command(argv, command_root, command_evidence, label):
                command_evidence.mkdir(parents=True, exist_ok=True)
                if label == "run":
                    junit.write_text(xml)
                return "{}"

            with (
                mock.patch.object(oracle, "ROOT", root),
                mock.patch.object(oracle.phase1, "command", side_effect=command),
                mock.patch.object(
                    oracle.phase1, "selection", return_value=(selected, {})
                ),
                mock.patch.object(oracle.phase1, "junit_path", return_value=junit),
            ):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    "missing or unexpected framework execution",
                ):
                    oracle.execute_leg("fixture", (), required, evidence)


class ReceiptTests(unittest.TestCase):
    def test_phase_one_precedes_every_phase_two_execution_and_receipt_is_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            legs = oracle.phase2_legs()
            packet = {
                "schema": 2,
                "python_required": ["phase2.contract"],
                "legs": [
                    {
                        "name": name,
                        "args": list(args),
                        "required": [name],
                    }
                    for name, args in legs
                ],
                "manual_exclusions": list(oracle.manual_exclusions()),
            }
            events: list[str] = []

            def phase_one():
                events.append("phase1")
                return root / "phase1.json"

            def execute(name, args, selected, evidence):
                self.assertEqual(selected, [name])
                events.append(name)
                return {
                    "name": name,
                    "required": [name],
                    "selected": [name],
                    "additions": [],
                    "executed": [{"id": name, "outcome": "passed"}],
                    "artifacts": {},
                }

            with (
                mock.patch.object(oracle, "ROOT", root),
                mock.patch.object(oracle, "MANIFEST", manifest),
                mock.patch.object(oracle, "MANIFEST_SHA256", "reviewed"),
                mock.patch.object(oracle, "frozen_manifest", return_value=packet),
                mock.patch.object(
                    oracle, "source_identity", return_value=("head", "source")
                ),
                mock.patch.object(
                    oracle.phase1,
                    "python_execution",
                    return_value={
                        "required": ["phase2.contract"],
                        "selected": ["phase2.contract"],
                        "additions": [],
                        "executed": ["phase2.contract"],
                    },
                ),
                mock.patch.object(oracle.phase1, "run", side_effect=phase_one),
                mock.patch.object(oracle, "execute_leg", side_effect=execute),
                mock.patch.object(oracle.uuid, "uuid4", return_value="run-id"),
            ):
                receipt = oracle.run()

            self.assertEqual(events, ["phase1", *[name for name, _ in legs]])
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["phase1_receipt"], str(root / "phase1.json"))
            self.assertEqual(payload["head"], "head")
            self.assertEqual(payload["source_digest"], "source")
            self.assertEqual(payload["manual_exclusions"], list(oracle.manual_exclusions()))
            self.assertEqual(payload["schema"], 2)
            self.assertEqual(payload["python"]["required"], ["phase2.contract"])
            self.assertEqual(payload["python"]["selected"], ["phase2.contract"])
            self.assertEqual(payload["python"]["additions"], [])
            self.assertEqual(
                payload["python"]["executed"],
                [{"id": "phase2.contract", "outcome": "passed"}],
            )

    def test_manifest_requires_exact_commands_identity_lists_and_manual_boundary(self):
        legs = oracle.phase2_legs()
        packet = {
            "schema": 2,
            "python_required": ["phase2.contract"],
            "legs": [
                {
                    "name": name,
                    "args": list(args),
                    "required": [name],
                }
                for name, args in legs
            ],
            "manual_exclusions": list(oracle.manual_exclusions()),
        }
        oracle.validate_manifest(packet)
        for mutation in (
            "command",
            "selection",
            "legacy",
            "unsorted",
            "nonstring",
            "manual",
        ):
            changed = json.loads(json.dumps(packet))
            if mutation == "command":
                changed["legs"][0]["args"].append("--changed")
            elif mutation == "selection":
                changed["legs"][0]["required"] = []
            elif mutation == "legacy":
                del changed["legs"][0]["required"]
                changed["legs"][0]["selected_count"] = 1
                changed["legs"][0]["selected_sha256"] = "a" * 64
            elif mutation == "unsorted":
                changed["legs"][0]["required"] = ["z", "a"]
            elif mutation == "nonstring":
                changed["legs"][0]["required"] = [1]
            else:
                changed["manual_exclusions"] = []
            with self.subTest(mutation=mutation), self.assertRaises(oracle.OracleFailure):
                oracle.validate_manifest(changed)


if __name__ == "__main__":
    unittest.main()
