#!/usr/bin/env python3
"""Check failures in checker oracle evidence and comparison boundaries."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import checker_core_oracle as oracle


def inventory(*, ignored=False, duplicate=False):
    suite = {"package-name": "checker", "binary-name": "checker",
        "testcases": {"required": {"kind": "test", "ignored": ignored,
            "filter-match": {"status": "matches"}}}}
    return {"rust-suites": {"one": suite, **({"two": deepcopy(suite)} if duplicate else {})}}


def event(name="checker::checker$required", result="ok"):
    return (json.dumps({"type": "test", "event": "started", "name": name}) + "\n"
        + json.dumps({"type": "test", "event": result, "name": name}) + "\n")


def lane(*identities):
    return {"selected": sorted(identities), "executed": sorted(identities)}


def host_snapshot():
    schema = oracle.load(oracle.FIXTURES / "schema.json")
    corpus = oracle.load(oracle.FIXTURES / "corpus.json")
    checked = {name: {} for name in schema["checked_fields"]}
    checked.update(bytes=[1], tree=[{"owner": 1}], typed_nodes=1, total_nodes=1)
    checked["payload"] = {name: {} for name in schema["checked_payload_fields"]}
    checked["local_ascriptions"] = [{"id": 1, "origin": "authored", "outstanding_claims": [2]}]
    report = {"typed_nodes": 1, "total_nodes": 1, "errors": []}
    accepted = {"status": "accepted", "checked": checked,
        "effects": {"status": "accepted", "checked": deepcopy(checked),
            "linearity": {"status": "accepted", "checked": deepcopy(checked), "copy_repairs": "Ok([])"}}}
    context = {"bytes": [1], "payload": {"variables": [1]}, "library_def_count": 1, "matches_program": True}
    case = {"status": "admitted", "source_tree": [{"source": 1}],
        "inference": deepcopy(report), "ir_inference": deepcopy(report),
        "ir": deepcopy(accepted), "typed": deepcopy(accepted), "declared_surface": "Ok(surface)"}
    extension = {name: deepcopy(accepted) for name in (
        "contextual", "restored", "layered", "restored_layered", "monolithic")}
    extension.update(composition=deepcopy(checked), foreign_composition=False)
    return {"cases": {item["id"]: deepcopy(case) for item in corpus["cases"]},
        "extensions": {item["id"]: deepcopy(extension) for item in corpus["extensions"]},
        "library": deepcopy(checked), "layer": deepcopy(checked),
        "context": deepcopy(context), "layered_context": deepcopy(context),
        "isolation": [deepcopy(accepted) for _ in range(8)]}


class BaselineAcquisitionTests(unittest.TestCase):
    def test_shallow_candidate_retains_baseline_objects_after_scratch_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote"
            remote.mkdir()

            def git(*arguments, cwd=remote):
                return subprocess.run(
                    ["git", "-c", "user.name=CheckerOracleTest",
                        "-c", "user.email=checker@example.invalid", *arguments],
                    cwd=cwd, check=True, text=True, capture_output=True,
                ).stdout

            git("init", "--quiet", "--initial-branch=main")
            (remote / "input.txt").write_text("baseline\n")
            git("add", "input.txt")
            git("commit", "--quiet", "-m", "baseline")
            baseline = git("rev-parse", "HEAD").strip()
            identity = oracle.digest(remote / "input.txt")
            (remote / "input.txt").write_text("candidate\n")
            git("commit", "--quiet", "-am", "candidate")
            candidate = root / "candidate"
            git("clone", "--quiet", "--depth=1", remote.as_uri(), str(candidate), cwd=root)
            head = git("rev-parse", "HEAD", cwd=candidate).strip()
            missing = subprocess.run(
                ["git", "cat-file", "-e", f"{baseline}^{{commit}}"],
                cwd=candidate, capture_output=True, check=False,
            )
            self.assertNotEqual(missing.returncode, 0)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            (fixtures / "immutable_source.json").write_text(json.dumps({
                "revision": baseline, "files": {"input.txt": identity},
            }))
            with (
                patch.object(oracle, "ROOT", candidate),
                patch.object(oracle, "BASELINE", baseline),
                patch.object(oracle, "REPOSITORY", remote.as_uri()),
                patch.object(oracle, "FIXTURES", fixtures),
                tempfile.TemporaryDirectory(dir=root) as scratch,
            ):
                checkout = oracle.checkout_baseline(Path(scratch) / "source")
                self.assertEqual((checkout / "input.txt").read_text(), "baseline\n")
            self.assertEqual(git("show", f"{baseline}:input.txt", cwd=candidate), "baseline\n")
            self.assertEqual(git("rev-parse", "HEAD", cwd=candidate).strip(), head)
            self.assertEqual(git("status", "--porcelain=v1", cwd=candidate), "")
            identity["sha256"] = "0" * 64
            (fixtures / "immutable_source.json").write_text(json.dumps({
                "revision": baseline, "files": {"input.txt": identity},
            }))
            with (
                patch.object(oracle, "ROOT", candidate),
                patch.object(oracle, "BASELINE", baseline),
                patch.object(oracle, "REPOSITORY", remote.as_uri()),
                patch.object(oracle, "FIXTURES", fixtures),
                tempfile.TemporaryDirectory(dir=root) as scratch,
                self.assertRaisesRegex(oracle.OracleFailure, "Immutable source file differs"),
            ):
                oracle.checkout_baseline(Path(scratch) / "source")

    def test_only_the_pinned_tree_probe_can_fail_without_compiler_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            subprocess.run(
                ["git", "init", "--quiet", str(repository)],
                check=True, capture_output=True,
            )
            evidence = oracle.Evidence(root / "evidence", "implementation")
            evidence.execute(
                ("git", "cat-file", "-e", f"{oracle.BASELINE}^{{tree}}"),
                cwd=repository,
            )
            document = {
                "status": "passed", "scope": "implementation", "platform": sys.platform,
                "machine": oracle.platform.machine(), "python": sys.executable,
                "python_version": oracle.platform.python_version(),
                "invocation": [sys.executable, __file__],
                "script": oracle.digest(Path(oracle.__file__)),
                "command_count": 1, "receipts": evidence.receipts,
            }
            evidence.save("receipts.json", document)
            self.assertEqual(oracle.validate_receipts(evidence.directory)[0]["exit"], 128)
            evidence.execute(
                ("git", "cat-file", "-e", f"{'0' * 40}^{{tree}}"),
                cwd=repository,
            )
            document["command_count"] = 2
            evidence.save("receipts.json", document)
            with self.assertRaisesRegex(oracle.OracleFailure, "Failed command lacks intended negative evidence"):
                oracle.validate_receipts(evidence.directory)


class ExecutionEvidenceTests(unittest.TestCase):
    def test_selected_identity(self):
        self.assertEqual(oracle.selected_tests(inventory()), {"checker::checker$required"})

    def test_ignored_mandatory_test_fails(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.selected_tests(inventory(ignored=True))

    def test_duplicate_selected_identity_fails(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.selected_tests(inventory(duplicate=True))

    def test_empty_selection_fails(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.selected_tests({"rust-suites": {}})

    def test_each_execution_identity_occurs_once(self):
        selected = {"checker::checker$required"}
        self.assertEqual(oracle.check_execution(event(), selected), selected)
        for output in ("", event() + event(), event("foreign"), event(result="failed"),
                event(result="ignored"), "not JSON"):
            with self.subTest(output=output), self.assertRaises(oracle.OracleFailure):
                oracle.check_execution(output, selected)

    def test_mock_runner_cannot_create_acceptance_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(oracle.support, "_execute"):
            with self.assertRaises(oracle.OracleFailure):
                oracle.Evidence(Path(temporary), "implementation")

    def test_existing_destination_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "previous").write_text("old evidence")
            with self.assertRaises(oracle.OracleFailure):
                oracle.Evidence(path, "implementation")

    def test_receipts_need_raw_files_locks_and_real_tool_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            tool = Path(sys.executable).resolve()
            files = {}
            for name in ("stdout", "stderr", "Cargo.toml", "Cargo.lock", "src/lib.rs"):
                raw = path / name.replace("/", "_")
                raw.write_text(name)
                files[name] = {"path": raw.name, **oracle.digest(raw)}
            record = {"identity": 0, "command": [str(tool), "--version"], "exit": 0,
                "elapsed_ns": 1, "consumer": True, "files": files,
                "tool": {"path": str(tool), **oracle.digest(tool)}}
            document = {"status": "passed", "scope": "implementation", "platform": sys.platform,
                "machine": oracle.platform.machine(), "python": sys.executable,
                "python_version": oracle.platform.python_version(), "invocation": [sys.executable, __file__],
                "script": oracle.digest(Path(oracle.__file__)), "command_count": 1, "receipts": [record]}
            manifest = path / "receipts.json"
            manifest.write_text(json.dumps(document))
            self.assertEqual(len(oracle.validate_receipts(path)), 1)
            for mutation in ("missing-lock", "failed-positive", "duplicate", "wrong-hash", "foreign-path", "missing-tool"):
                changed = deepcopy(document)
                item = changed["receipts"][0]
                if mutation == "missing-lock":
                    del item["files"]["Cargo.lock"]
                elif mutation == "failed-positive":
                    item["exit"] = 1
                elif mutation == "duplicate":
                    changed["receipts"].append(deepcopy(item))
                    changed["command_count"] = 2
                elif mutation == "wrong-hash":
                    item["files"]["stdout"]["sha256"] = "0" * 64
                elif mutation == "foreign-path":
                    item["files"]["stdout"]["path"] = "../foreign"
                else:
                    del item["tool"]
                manifest.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises(oracle.OracleFailure):
                    oracle.validate_receipts(path)

    def test_negative_receipt_needs_the_intended_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = oracle.Evidence(Path(temporary), "implementation")
            evidence.receipts = [{"exit": 1}]
            with self.assertRaises(oracle.OracleFailure):
                evidence.accept_negative(["E0423", "private fields"], "missing crate")
            evidence.accept_negative(["E0423", "private fields"], "E0423 private fields")
            evidence.receipts[-1]["exit"] = 0
            with self.assertRaises(oracle.OracleFailure):
                evidence.accept_negative(["E0423"], "E0423")


class SemanticComparisonTests(unittest.TestCase):
    def test_complete_snapshot_round_trip(self):
        original = host_snapshot()
        oracle.compare_snapshots(original, deepcopy(original))

    def test_semantic_fields_never_disappear(self):
        original = host_snapshot()
        for name in oracle.load(oracle.FIXTURES / "schema.json")["checked_fields"]:
            changed = deepcopy(original)
            del changed["library"][name]
            with self.subTest(name=name), self.assertRaises(oracle.OracleFailure):
                oracle.compare_snapshots(original, changed)

    def test_every_checked_payload_field_remains_compared(self):
        original = host_snapshot()
        for name in oracle.load(oracle.FIXTURES / "schema.json")["checked_payload_fields"]:
            changed = deepcopy(original)
            changed["library"]["payload"][name] = {"changed": True}
            with self.subTest(name=name), self.assertRaises(oracle.OracleFailure):
                oracle.compare_snapshots(original, changed)

    def test_owner_ascription_context_bytes_and_counters_remain_compared(self):
        original = host_snapshot()
        for name in ("owner", "ascription", "context", "bytes", "counter", "ingress", "diagnostic"):
            changed = deepcopy(original)
            case = next(iter(changed["cases"].values()))
            if name == "owner":
                changed["library"]["tree"][0]["owner"] = 2
            elif name == "ascription":
                changed["library"]["local_ascriptions"][0]["outstanding_claims"] = []
            elif name == "context":
                changed["context"]["payload"]["variables"] = [2]
            elif name == "bytes":
                changed["library"]["bytes"] = [2]
            elif name == "counter":
                case["inference"]["typed_nodes"] = 0
            elif name == "ingress":
                case["typed"]["checked"]["bytes"] = [3]
            else:
                error = {field: None for field in oracle.load(oracle.FIXTURES / "schema.json")["error_fields"]}
                error["message"] = "changed diagnostic"
                case["inference"]["errors"] = [error]
            with self.subTest(name=name), self.assertRaises(oracle.OracleFailure):
                oracle.compare_snapshots(original, changed)

    def test_missing_case_extension_or_thread_fails(self):
        for name in ("cases", "extensions", "isolation"):
            changed = host_snapshot()
            if name == "isolation":
                changed[name].pop()
            else:
                changed[name].pop(next(iter(changed[name])))
            with self.subTest(name=name), self.assertRaises(oracle.OracleFailure):
                oracle.validate_snapshot(changed)

    def test_foreign_composition_fails(self):
        changed = host_snapshot()
        next(iter(changed["extensions"].values()))["foreign_composition"] = True
        with self.assertRaises(oracle.OracleFailure):
            oracle.validate_snapshot(changed)

    def test_nonfinite_comparison_fails(self):
        with self.assertRaises(ValueError):
            oracle.canonical({"invalid": float("nan")})


class MigrationEvidenceTests(unittest.TestCase):
    def test_unchanged_name_has_one_concrete_owner(self):
        before = {"host": lane("old::old$algorithm")}
        after = {"core": lane("new::new$algorithm")}
        mapping = oracle.reconcile_inventories(before, after, {"renamed": {}, "retired": {}})
        self.assertEqual(mapping, {"old::old$algorithm": "new::new$algorithm"})

    def test_missing_ambiguous_duplicate_and_unexecuted_tests_fail(self):
        before = {"host": lane("old::old$algorithm")}
        cases = ({"core": lane("new::new$other")},
            {"core": lane("new::new$algorithm", "other::other$algorithm")},
            {"core": lane("new::new$algorithm"), "duplicate": lane("new::new$algorithm")},
            {"core": {"selected": ["new::new$algorithm"], "executed": []}})
        for after in cases:
            with self.subTest(after=after), self.assertRaises(oracle.OracleFailure):
                oracle.reconcile_inventories(before, after, {"renamed": {}, "retired": {}})

    def test_retirement_needs_executed_behavior(self):
        retired = "chelis-types::chelis_types$source_arch::tests::source_assertion"
        before = {"host": lane(retired)}
        after = {"core": lane("new::new$behavior")}
        migrations = {"renamed": {}, "retired": {retired: {
            "reason": "Private source spelling retires.", "behavior_identities": ["new::new$behavior"]}}}
        self.assertEqual(oracle.reconcile_inventories(before, after, migrations), {})
        migrations["retired"][retired]["behavior_identities"] = []
        with self.assertRaises(oracle.OracleFailure):
            oracle.reconcile_inventories(before, after, migrations)

    def test_timing_migration_preserves_the_exact_value(self):
        mapping = {"old::old$algorithm": "new::new$algorithm"}
        oracle.check_timing({"old::algorithm": 1.234}, {"new::algorithm": 1.234}, mapping, {})
        for current in ({}, {"new::algorithm": 0}, {"new::algorithm": 1.234, "old::algorithm": 1.234}):
            with self.subTest(current=current), self.assertRaises(oracle.OracleFailure):
                oracle.check_timing({"old::algorithm": 1.234}, current, mapping, {})

    def test_integration_timing_migration_preserves_binary_and_value(self):
        mapping = {"old::first$algorithm": "new::second$algorithm"}
        previous = {"old::first::algorithm": 1.234}
        oracle.check_timing(previous, {"new::second::algorithm": 1.234}, mapping, {})
        for current in ({"new::algorithm": 1.234}, {"new::second::algorithm": 0},
                {"new::second::algorithm": 1.234, "old::first::algorithm": 1.234}):
            with self.subTest(current=current), self.assertRaises(oracle.OracleFailure):
                oracle.check_timing(previous, current, mapping, {})

    def test_retired_integration_timing_requires_its_exact_key_to_be_absent(self):
        previous = {"old::consumer::source_assertion": 0.009}
        retired = {"old::consumer$source_assertion": {}}
        oracle.check_timing(previous, {}, {}, retired)
        with self.assertRaisesRegex(oracle.OracleFailure, "stale timing owner"):
            oracle.check_timing(previous, previous, {}, retired)


class AdditionalFailureControls(unittest.TestCase):
    def test_new_behavior_identities_must_execute_once(self):
        before = {"host": lane("old::old$existing")}
        after = {"core": lane("new::new$existing", "new::new$movement")}
        migrations = {"renamed": {}, "retired": {}, "new_tests": ["new::new$movement"]}
        oracle.reconcile_inventories(before, after, migrations)
        for identities in (["new::new$missing"], ["new::new$movement", "new::new$movement"],
                ["new::new$existing"]):
            changed = deepcopy(migrations)
            changed["new_tests"] = identities
            if identities == ["new::new$existing"]:
                changed["renamed"] = {"old::old$existing": "new::new$existing"}
            with self.subTest(identities=identities), self.assertRaises(oracle.OracleFailure):
                oracle.reconcile_inventories(before, after, changed)

    def test_missing_lane_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            for scope in ("baseline-capture", "implementation", "mock"):
                with self.subTest(scope=scope), self.assertRaises(oracle.OracleFailure):
                    oracle.validate_artifacts(Path(temporary), scope)

    def test_duplicate_json_fields_fail(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.decode('{"identity": 0, "identity": 1}')

    def test_junit_retains_successful_output_and_exact_timing(self):
        metadata = {"rust-suites": {"checker::checker": inventory()["rust-suites"]["one"]}}
        raw = ('<testsuites><testsuite name="checker::checker">'
            '<testcase name="required" time="0.001234">'
            '<system-out>raw stdout</system-out><system-err>cost 123</system-err>'
            '</testcase></testsuite></testsuites>')
        result = oracle.junit_evidence(raw, metadata)
        self.assertEqual(result["raw_xml"], raw)
        self.assertEqual(result["observations"]["checker::checker$required"]["attributes"]["time"], "0.001234")
        self.assertEqual(result["observations"]["checker::checker$required"]["stderr"], "cost 123")
        for child in ("failure", "error", "skipped", "flakyFailure", "rerunFailure"):
            changed = raw.replace("</testcase>", f"<{child}/></testcase>")
            with self.subTest(child=child), self.assertRaises(oracle.OracleFailure):
                oracle.junit_evidence(changed, metadata)
        for changed in (raw.replace('time="0.001234"', ""), raw.replace('name="required"', 'name="foreign"'),
                raw.replace("</testsuite>", '<testcase name="required" time="1"/></testsuite>')):
            with self.subTest(changed=changed), self.assertRaises(oracle.OracleFailure):
                oracle.junit_evidence(changed, metadata)

    def test_junit_resolves_library_and_integration_binary_identities(self):
        library = deepcopy(inventory()["rust-suites"]["one"])
        library.update({"package-name": "checker-lib", "binary-name": "checker_lib"})
        integration = deepcopy(library)
        integration["binary-name"] = "context"
        metadata = {"rust-suites": {"checker-lib": library, "checker-lib::context": integration}}
        raw = ('<testsuites><testsuite name="checker-lib"><testcase name="required" time="1"/>'
            '</testsuite><testsuite name="checker-lib::context">'
            '<testcase name="required" time="2"/></testsuite></testsuites>')
        result = oracle.junit_evidence(raw, metadata)
        self.assertEqual(set(result["observations"]), {
            "checker-lib::checker_lib$required", "checker-lib::context$required"})
        for changed in (raw.replace('name="checker-lib"', 'name="checker_lib"'),
                raw.replace('name="checker-lib::context"', 'name="checker-lib"')):
            with self.subTest(changed=changed), self.assertRaises(oracle.OracleFailure):
                oracle.junit_evidence(changed, metadata)

    def test_negative_compilation_must_fail_at_the_actual_fixture_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = "pub fn forge() {}"
            fixture = directory / "src/lib.rs"
            record = {"reason": "compiler-message", "target": {"src_path": str(fixture)}, "message": {
                "level": "error", "message": "cannot initialize a tuple struct which contains private fields",
                "code": {"code": "E0423"}, "children": [], "spans": [{
                    "is_primary": True, "file_name": str(fixture), "line_start": 1,
                    "text": [{"text": source}], "label": "private fields"}]}}
            expected = ["E0423", "private fields"]
            oracle.negative_diagnostics(json.dumps(record), directory, source, expected)
            for mutation in ("dependency", "wrong-code", "wrong-line", "missing-private", "missing-error"):
                changed = deepcopy(record)
                if mutation == "dependency":
                    changed["target"]["src_path"] = str(directory / "dependency.rs")
                elif mutation == "wrong-code":
                    changed["message"]["code"]["code"] = "E0432"
                elif mutation == "wrong-line":
                    changed["message"]["spans"][0]["text"][0]["text"] = "foreign source"
                elif mutation == "missing-private":
                    changed["message"]["message"] = "unrelated failure"
                    changed["message"]["spans"][0]["label"] = None
                else:
                    changed["message"]["level"] = "warning"
                with self.subTest(mutation=mutation), self.assertRaises(oracle.OracleFailure):
                    oracle.negative_diagnostics(json.dumps(changed), directory, source, expected)

    def test_semantic_tests_cannot_retire_as_source_assertions(self):
        identity = "chelis-types::tuple_nesting_cost$checking_a_literal_tuple_chain_allocates_linearly"
        migrations = {"renamed": {}, "retired": {identity: {
            "reason": "A source spelling retires.", "behavior_identities": ["core::core$replacement"]}}}
        with self.assertRaises(oracle.OracleFailure):
            oracle.reconcile_inventories({"host": lane(identity)},
                {"core": lane("core::core$replacement")}, migrations)

    def test_only_a_fully_retired_source_target_leaves_the_candidate_selection(self):
        retired = "chelis-types::stack_guard_coverage$source_assertion"
        arguments = ("-p", "chelis-types", "--test", "stack_guard_coverage", "--test", "context")
        migrations = {"renamed": {}, "retired": {retired: {}}}
        self.assertEqual(oracle.candidate_arguments(arguments, lane(retired), migrations),
            ("-p", "chelis-types", "--test", "context"))
        self.assertEqual(oracle.candidate_arguments(arguments,
            lane(retired, "chelis-types::stack_guard_coverage$another"), migrations), arguments)

    def test_previous_core_baselines_and_required_obligations_cannot_change(self):
        source = 'BASELINE = "fixed"\nREQUIRED = {"semantic"}\n'
        self.assertNotEqual(oracle.prior_contracts(source),
            oracle.prior_contracts(source.replace('"fixed"', '"candidate"')))
        self.assertNotEqual(oracle.prior_contracts(source),
            oracle.prior_contracts(source.replace('{"semantic"}', 'set()')))
        with self.assertRaises(oracle.OracleFailure):
            oracle.prior_contracts('REQUIRED = {"semantic"}\n')

    def test_missing_cost_observation_fails(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.cost_observations({"full-host": {"junit": {"observations": {}}}})


if __name__ == "__main__":
    unittest.main()
