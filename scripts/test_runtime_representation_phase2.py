"""Spec-derived controls for the runtime-representation Phase 2 receipt."""

from __future__ import annotations

import json
from contextlib import nullcontext
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
                "HIP descriptor and device-owner contracts",
                "Python host device and DLPack boundaries",
                "backend runtime-header capacity census",
            ],
        )
        hip = dict(oracle.phase2_legs())["HIP descriptor and device-owner contracts"]
        self.assertEqual(
            [hip[index + 1] for index, value in enumerate(hip) if value == "--test"],
            [
                "codegen_structure",
                "device_entry_contract",
                "device_entry_execution",
                "device_owner_contract",
            ],
        )

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

    def test_selection_digest_binds_every_exact_test_identity(self):
        selected = ["package::binary::negative", "package::binary::positive"]
        digest = oracle.selection_digest(selected)
        oracle.require_frozen_selection(selected, len(selected), digest)
        for changed in (selected[:1], [*selected, selected[0]], list(reversed(selected))):
            with self.subTest(changed=changed), self.assertRaises(oracle.OracleFailure):
                oracle.require_frozen_selection(changed, len(selected), digest)


class ReceiptTests(unittest.TestCase):
    def test_phase_one_precedes_every_phase_two_execution_and_receipt_is_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            legs = oracle.phase2_legs()
            packet = {
                "schema": 1,
                "python_selected": ["phase2.contract"],
                "legs": [
                    {
                        "name": name,
                        "args": list(args),
                        "selected_count": 1,
                        "selected_sha256": oracle.selection_digest([name]),
                    }
                    for name, args in legs
                ],
                "manual_exclusions": list(oracle.manual_exclusions()),
            }
            runtime_receipt = {
                "pinned_artifact": {str(root / "runtime/libchelis_runtime.a"): "sha256"}
            }
            events: list[str] = []

            def phase_one():
                events.append("phase1")
                return root / "phase1.json"

            def execute(name, args, count, digest, evidence):
                self.assertEqual(count, 1)
                self.assertEqual(digest, oracle.selection_digest([name]))
                events.append(name)
                return {
                    "name": name,
                    "selected": [name],
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
                    oracle.phase1, "python_execution", return_value=["phase2.contract"]
                ),
                mock.patch.object(oracle.phase1, "run", side_effect=phase_one),
                mock.patch.object(
                    oracle.phase1,
                    "runtime_pin",
                    return_value=nullcontext(runtime_receipt),
                ) as runtime_pin,
                mock.patch.object(oracle, "execute_leg", side_effect=execute),
                mock.patch.object(oracle.uuid, "uuid4", return_value="run-id"),
            ):
                receipt = oracle.run()

            runtime_pin.assert_called_once_with(root / "target/runtime-representation-phase2/run-id/runtime-build")
            self.assertEqual(events, ["phase1", *[name for name, _ in legs]])
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["phase1_receipt"], str(root / "phase1.json"))
            self.assertEqual(payload["runtime"], runtime_receipt)
            self.assertEqual(payload["head"], "head")
            self.assertEqual(payload["source_digest"], "source")
            self.assertEqual(payload["manual_exclusions"], list(oracle.manual_exclusions()))

    def test_manifest_requires_exact_commands_selection_hashes_and_manual_boundary(self):
        legs = oracle.phase2_legs()
        packet = {
            "schema": 1,
            "python_selected": ["phase2.contract"],
            "legs": [
                {
                    "name": name,
                    "args": list(args),
                    "selected_count": 1,
                    "selected_sha256": "a" * 64,
                }
                for name, args in legs
            ],
            "manual_exclusions": list(oracle.manual_exclusions()),
        }
        oracle.validate_manifest(packet)
        for mutation in ("command", "count", "manual"):
            changed = json.loads(json.dumps(packet))
            if mutation == "command":
                changed["legs"][0]["args"].append("--changed")
            elif mutation == "count":
                changed["legs"][0]["selected_count"] = 0
            else:
                changed["manual_exclusions"] = []
            with self.subTest(mutation=mutation), self.assertRaises(oracle.OracleFailure):
                oracle.validate_manifest(changed)


if __name__ == "__main__":
    unittest.main()
