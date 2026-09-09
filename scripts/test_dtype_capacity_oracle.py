#!/usr/bin/env python3
"""Specification-derived controls for the #1288 composite receipt adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import dtype_capacity_oracle as oracle
import dtype_pre_phase4c_oracle as composite


class ContractTests(unittest.TestCase):
    def test_exact_four_surface_groups_are_required(self):
        self.assertEqual(
            [group.name for group in oracle.GROUPS],
            ["primary", "stdlib", "wire", "bindings"],
        )
        self.assertEqual(
            {group.package for group in oracle.GROUPS},
            {"chelis-cli", "chelis-compiler-api", "chelis-python"},
        )

    def test_binding_requires_a_real_final_zero_legacy_gate(self):
        bindings = next(group for group in oracle.GROUPS if group.name == "bindings")
        self.assertIn("binding_census_has_zero_legacy_rows", bindings.selected)
        self.assertNotIn("registered_pyfunctions_match_the_reviewed_rustdoc_signatures", bindings.positive)

    def test_every_group_has_executed_positive_negative_and_mutation_controls(self):
        for group in oracle.GROUPS:
            with self.subTest(group=group.name):
                self.assertTrue(group.positive)
                self.assertTrue(group.negative)
                self.assertTrue(group.mutation)
                self.assertTrue(set(group.positive) <= set(group.selected))
                self.assertTrue(set(group.negative) <= set(group.selected))
                self.assertTrue(set(group.mutation) <= set(group.selected))

    def test_primary_and_stdlib_selection_covers_final_authority_requirement_classes(self):
        groups = {group.name: group for group in oracle.GROUPS}
        primary = set(groups["primary"].selected)
        stdlib = set(groups["stdlib"].selected)
        self.assertTrue({
            "migrated_primary_rows_have_exact_final_authority_and_no_transition_disposition",
            "primary_baseline_has_zero_legacy_rows",
            "duplicate_primary_descriptors_fail_before_map_collapse",
            "every_primary_final_registration_binds_kind_identity_and_flags",
            "maintainer_override_is_not_a_final_authority_class",
        } <= primary)
        self.assertTrue({
            "capacity_census_stdlib_tests::import_failures_and_ambiguity_do_not_erase_numeric_capacity",
            "capacity_census_stdlib_tests::unresolved_nominals_and_wrong_nominal_arguments_fail_closed",
            "capacity_census_stdlib_tests::nested_alias_fields_and_precision_variables_are_capacity",
            "capacity_census_stdlib_tests::forward_and_recursive_aliases_reach_a_finite_numeric_fixed_point",
            "capacity_census_stdlib_tests::generic_summaries_substitute_used_parameter_positions",
            "capacity_census_stdlib_tests::alias_body_changes_capacity_without_rewriting_the_authored_identity",
        } <= stdlib)

    def test_composite_keeps_capacity_missing_until_the_final_binding_gate_exists(self):
        child = next(item for item in composite.prerequisites("python") if item.name == "capacity")
        self.assertIsInstance(child, composite.MissingOracle)
        self.assertIn("final PyO3 zero-legacy gate", child.reason)


class ReceiptTests(unittest.TestCase):
    def test_receipt_requires_every_group_and_prefixes_test_identities(self):
        groups = oracle.GROUPS
        executions = {
            group.name: tuple(group.selected)
            for group in groups
        }
        packet = oracle.receipt_packet("head", "digest", "run", executions)
        self.assertEqual(packet["schema"], 1)
        self.assertEqual(packet["hosts"], {})
        self.assertEqual(packet["devices"], [])
        self.assertTrue(all(row["outcome"] == "passed" for row in packet["executed"]))
        self.assertTrue(all("::" in row["id"] for row in packet["executed"]))
        self.assertEqual(
            {row["id"] for row in packet["executed"]}, set(packet["selected"])
        )

    def test_receipt_rejects_missing_duplicate_and_unexecuted_controls(self):
        complete = {group.name: tuple(group.selected) for group in oracle.GROUPS}
        missing = dict(complete)
        missing.pop("wire")
        with self.assertRaises(oracle.CapacityOracleError):
            oracle.receipt_packet("head", "digest", "run", missing)
        duplicate = dict(complete)
        duplicate["primary"] = duplicate["primary"] + duplicate["primary"][:1]
        with self.assertRaises(oracle.CapacityOracleError):
            oracle.receipt_packet("head", "digest", "run", duplicate)
        absent = dict(complete)
        absent["bindings"] = absent["bindings"][1:]
        with self.assertRaises(oracle.CapacityOracleError):
            oracle.receipt_packet("head", "digest", "run", absent)


class ExecutionTests(unittest.TestCase):
    def test_actual_artifact_command_is_locked_and_never_accepts_a_saved_receipt(self):
        group = oracle.GROUPS[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "crates/chelis-cli/tests").mkdir(parents=True)
            (root / "crates/chelis-cli/tests/capacity_census_tripwire.rs").write_text("// fixture\n")
            artifact = root / "target/dtype-capacity/debug/deps/capacity_census_tripwire-test"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("")
            completed = mock.Mock(returncode=0, stdout=json.dumps({
                "reason": "compiler-artifact",
                "target": {"name": "capacity_census_tripwire", "kind": ["test"], "src_path": str(root / "crates/chelis-cli/tests/capacity_census_tripwire.rs")},
                "profile": {"test": True},
                "executable": str(artifact),
            }) + "\n", stderr="")
            evidence = root / "target/dtype-capacity/evidence/primary"
            with mock.patch("dtype_capacity_oracle.subprocess.run", return_value=completed) as run, \
                    mock.patch("dtype_capacity_oracle.run_libtest", return_value=tuple(group.selected)):
                self.assertEqual(
                    oracle.execute_group(root, root / "target/dtype-capacity", group, evidence),
                    tuple(group.selected),
                )
            command = run.call_args.args[0]
            self.assertIn("--locked", command)
            self.assertIn("--no-run", command)
            self.assertIn("--message-format=json", command)
            self.assertEqual(run.call_args.kwargs["env"]["RUSTC_BOOTSTRAP"], "1")
            self.assertEqual(run.call_args.kwargs["cwd"], root)
            self.assertEqual((evidence / "cargo.stdout").read_text(), completed.stdout)
            self.assertEqual((evidence / "cargo.artifacts.jsonl").read_text(), completed.stdout)
            self.assertEqual((evidence / "cargo.stderr").read_text(), "")
            cargo = json.loads((evidence / "cargo.process.json").read_text())
            self.assertEqual(cargo["argv"], list(command))
            self.assertEqual(cargo["returncode"], 0)

    def test_missing_final_binding_test_cannot_turn_a_zero_test_run_into_success(self):
        with self.assertRaises(oracle.CapacityOracleError):
            oracle.validate_libtest_events(
                ("binding_census_has_zero_legacy_rows",),
                [{"type": "suite", "event": "started", "test_count": 0}],
            )

    def test_lifecycle_rejects_ignored_failed_and_trailing_events(self):
        good = [
            {"type": "suite", "event": "started", "test_count": 1},
            {"type": "test", "event": "started", "name": "selected"},
            {"type": "test", "event": "ok", "name": "selected"},
            {"type": "suite", "event": "ok", "passed": 1, "failed": 0,
             "ignored": 0, "measured": 0, "filtered_out": 2},
        ]
        self.assertEqual(oracle.validate_libtest_events(("selected",), good), ("selected",))
        for changed in (
            good[:2] + [{"type": "test", "event": "ignored", "name": "selected"}],
            good[:2] + [{"type": "test", "event": "failed", "name": "selected"}],
            good[:3] + [{"type": "test", "event": "ok", "name": "selected"}],
            good + [{"type": "test", "event": "stdout", "name": "selected"}],
        ):
            with self.subTest(changed=changed[-1]):
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle.validate_libtest_events(("selected",), changed)

    def test_nonobject_and_foreign_artifacts_are_typed_failures(self):
        group = oracle.GROUPS[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "crates/chelis-cli/tests/capacity_census_tripwire.rs"
            source.parent.mkdir(parents=True)
            source.write_text("// fixture\n")
            target = root / "target/dtype-capacity"
            target.mkdir(parents=True)
            owned = target / "owned-test"
            owned.write_text("")
            foreign = root / "foreign-test"
            foreign.write_text("")
            record = {
                "reason": "compiler-artifact",
                "target": {"name": group.binary, "kind": ["test"], "src_path": str(source)},
                "profile": {"test": True},
                "executable": str(owned),
            }
            self.assertEqual(
                oracle._test_artifact(root, target, group, json.dumps(record)), owned
            )
            malformed = "[]\n" + json.dumps(record)
            foreign_record = {**record, "executable": str(foreign)}
            for payload in (malformed, json.dumps(foreign_record)):
                with self.subTest(payload=payload):
                    with self.assertRaises(oracle.CapacityOracleError):
                        oracle._test_artifact(root, target, group, payload)

    def test_capacity_environment_rejects_stale_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt = root / "target" / "capacity.json"
            identity = composite.SourceIdentity("a" * 40, "b" * 64)
            environment = {
                "CHELIS_ORACLE_HEAD": identity.head,
                "CHELIS_ORACLE_SOURCE_DIGEST": identity.digest,
                "CHELIS_ORACLE_RUN_ID": "fresh",
                "CHELIS_ORACLE_RECEIPT": str(receipt),
            }
            with mock.patch.dict(os.environ, environment), \
                    mock.patch("dtype_capacity_oracle.source_identity", return_value=identity):
                self.assertEqual(oracle._required_environment(root), identity)
            stale = composite.SourceIdentity("c" * 40, identity.digest)
            with mock.patch.dict(os.environ, environment), \
                    mock.patch("dtype_capacity_oracle.source_identity", return_value=stale):
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle._required_environment(root)

    def test_main_places_fresh_group_evidence_beside_framework_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt = root / "target" / "receipts" / "execution.json"
            receipt.parent.mkdir(parents=True)
            identity = composite.SourceIdentity("a" * 40, "b" * 64)
            environment = {
                "CHELIS_ORACLE_HEAD": identity.head,
                "CHELIS_ORACLE_SOURCE_DIGEST": identity.digest,
                "CHELIS_ORACLE_RUN_ID": "fresh",
                "CHELIS_ORACLE_RECEIPT": str(receipt),
            }
            seen = []

            def execute(_root, _target, group, evidence):
                seen.append((group.name, evidence))
                return group.selected

            with mock.patch.dict(os.environ, environment), \
                    mock.patch("dtype_capacity_oracle.REPO_ROOT", root), \
                    mock.patch("dtype_capacity_oracle.source_identity", return_value=identity), \
                    mock.patch("dtype_capacity_oracle.execute_group", side_effect=execute):
                self.assertEqual(oracle.main(), 0)
            evidence_root = receipt.parent / "capacity"
            self.assertEqual([path for _, path in seen], [
                evidence_root / "bindings", evidence_root / "primary",
                evidence_root / "stdlib", evidence_root / "wire",
            ])
            self.assertTrue(receipt.is_file())
            evidence_root.mkdir()
            with self.assertRaises(oracle.CapacityOracleError):
                oracle._receipt_evidence_dir(root, receipt)

    def test_real_pinned_libtest_json_executes_one_selected_case(self):
        Path("target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="capacity-libtest-", dir="target") as tmp:
            root = Path(tmp).resolve()
            source = root / "fixture.rs"
            binary = root / "fixture-test"
            source.write_text(
                "#[test] fn selected() { println!(\"fixture\"); }\n"
                "#[test] fn second_selected() {}\n"
            )
            subprocess.run(
                ("rustc", "--test", str(source), "-o", str(binary)),
                cwd=root,
                env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
            (root / "evidence").mkdir()
            self.assertEqual(
                oracle.run_libtest(root, binary, ("selected", "second_selected"), root / "evidence"),
                ("selected", "second_selected"),
            )

    def test_libtest_uses_managed_environment_and_root_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            binary = root / "test"
            binary.write_bytes(b"fixture")
            events = "\n".join(json.dumps(event) for event in [
                {"type": "suite", "event": "started", "test_count": 1},
                {"type": "test", "event": "started", "name": "selected"},
                {"type": "test", "event": "ok", "name": "selected"},
                {"type": "suite", "event": "ok", "passed": 1, "failed": 0,
                 "ignored": 0, "measured": 0, "filtered_out": 0},
            ])
            result = mock.Mock(returncode=0, stdout=events, stderr="")
            (root / "evidence").mkdir()
            with mock.patch("dtype_capacity_oracle.subprocess.run", return_value=result) as run:
                self.assertEqual(oracle.run_libtest(root, binary, ("selected",), root / "evidence"), ("selected",))
            command = run.call_args.args[0]
            self.assertIn("-Zunstable-options", command)
            self.assertIn("--format=json", command)
            self.assertIn("--test-threads=1", command)
            self.assertEqual(run.call_args.kwargs["cwd"], root)
            self.assertEqual(run.call_args.kwargs["env"]["RUSTC_BOOTSTRAP"], "1")
            self.assertEqual(run.call_args.kwargs["env"]["PYO3_PYTHON"], sys.executable)
            self.assertEqual(run.call_args.kwargs["env"]["VIRTUAL_ENV"], sys.prefix)

    def test_libtest_binary_mutation_after_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            binary = root / "test"
            binary.write_bytes(b"before")
            events = "\n".join(json.dumps(event) for event in [
                {"type": "suite", "event": "started", "test_count": 1},
                {"type": "test", "event": "started", "name": "selected"},
                {"type": "test", "event": "ok", "name": "selected"},
                {"type": "suite", "event": "ok", "passed": 1, "failed": 0,
                 "ignored": 0, "measured": 0, "filtered_out": 0},
            ])
            def mutate(*args, **kwargs):
                binary.write_bytes(b"after")
                return mock.Mock(returncode=0, stdout=events, stderr="")
            (root / "evidence").mkdir()
            with mock.patch("dtype_capacity_oracle.subprocess.run", side_effect=mutate):
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle.run_libtest(root, binary, ("selected",), root / "evidence")

    def test_libtest_retains_lifecycle_output_and_exact_process_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            binary = root / "test"
            binary.write_bytes(b"fixture")
            evidence = root / "evidence"
            events = "\n".join(json.dumps(event) for event in [
                {"type": "suite", "event": "started", "test_count": 1},
                {"type": "test", "event": "started", "name": "selected"},
                {"type": "test", "event": "ok", "name": "selected"},
                {"type": "suite", "event": "ok", "passed": 1, "failed": 0,
                 "ignored": 0, "measured": 0, "filtered_out": 0},
            ])
            result = mock.Mock(returncode=0, stdout=events, stderr="libtest diagnostic\n")
            evidence.mkdir()
            with mock.patch("dtype_capacity_oracle.subprocess.run", return_value=result) as run:
                self.assertEqual(oracle.run_libtest(root, binary, ("selected",), evidence), ("selected",))
            self.assertEqual((evidence / "libtest.stdout").read_text(), events)
            self.assertEqual((evidence / "libtest.stderr").read_text(), "libtest diagnostic\n")
            process = json.loads((evidence / "libtest.process.json").read_text())
            self.assertEqual(process["argv"], list(run.call_args.args[0]))
            self.assertEqual(process["returncode"], 0)

    def test_failed_process_retains_output_and_existing_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            binary = root / "test"
            binary.write_bytes(b"fixture")
            evidence = root / "evidence"
            events = "\n".join(json.dumps(event) for event in [
                {"type": "suite", "event": "started", "test_count": 1},
                {"type": "test", "event": "started", "name": "selected"},
                {"type": "test", "event": "ok", "name": "selected"},
                {"type": "suite", "event": "ok", "passed": 1, "failed": 0,
                 "ignored": 0, "measured": 0, "filtered_out": 0},
            ])
            result = mock.Mock(returncode=9, stdout=events, stderr="failed after output\n")
            evidence.mkdir()
            with mock.patch("dtype_capacity_oracle.subprocess.run", return_value=result):
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle.run_libtest(root, binary, ("selected",), evidence)
            self.assertEqual((evidence / "libtest.stdout").read_text(), events)
            self.assertEqual((evidence / "libtest.stderr").read_text(), "failed after output\n")
            with mock.patch("dtype_capacity_oracle.subprocess.run", return_value=result):
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle.run_libtest(root, binary, ("selected",), evidence)

    def test_existing_group_evidence_directory_is_rejected_before_cargo_runs(self):
        group = oracle.GROUPS[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "crates/chelis-cli/tests/capacity_census_tripwire.rs"
            source.parent.mkdir(parents=True)
            source.write_text("// fixture\n")
            target = root / "target/dtype-capacity"
            evidence = target / "evidence/primary"
            evidence.mkdir(parents=True)
            with mock.patch("dtype_capacity_oracle.subprocess.run") as run:
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle.execute_group(root, target, group, evidence)
            run.assert_not_called()

    def test_failed_cargo_process_retains_outputs_without_running_libtest(self):
        group = oracle.GROUPS[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "crates/chelis-cli/tests/capacity_census_tripwire.rs"
            source.parent.mkdir(parents=True)
            source.write_text("// fixture\n")
            target = root / "target/dtype-capacity"
            evidence = target / "evidence/primary"
            result = mock.Mock(returncode=101, stdout='{"reason":"build-finished"}\n',
                               stderr="locked build failed\n")
            with mock.patch("dtype_capacity_oracle.subprocess.run", return_value=result), \
                    mock.patch("dtype_capacity_oracle.run_libtest") as run_libtest:
                with self.assertRaises(oracle.CapacityOracleError):
                    oracle.execute_group(root, target, group, evidence)
            self.assertEqual((evidence / "cargo.stdout").read_text(), result.stdout)
            self.assertEqual((evidence / "cargo.artifacts.jsonl").read_text(), result.stdout)
            self.assertEqual((evidence / "cargo.stderr").read_text(), result.stderr)
            self.assertEqual(json.loads((evidence / "cargo.process.json").read_text())["returncode"], 101)
            run_libtest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
