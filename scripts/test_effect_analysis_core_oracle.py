#!/usr/bin/env python3
"""Exercise effect acceptance evidence directly, without compiler substitutes."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import effect_analysis_core_oracle as oracle
from no_std_core_support import OracleFailure


def event(name: str, outcome: str = "ok") -> str:
    return json.dumps({"type": "test", "event": outcome, "name": name})


def inventory() -> dict:
    return {"rust-suites": {"suite": {"package-name": "owner", "binary-name": "binary",
        "testcases": {"case": {"kind": "test", "ignored": False,
            "filter-match": {"status": "matches"}}}}}}


def graph(name: str = "portable", *, std: bool = False, kind=None) -> dict:
    return {"resolve": {"root": "root", "nodes": [
        {"id": "root", "features": [], "deps": [{"pkg": "dependency", "dep_kinds": [{"kind": kind}]}]},
        {"id": "dependency", "features": ["std"] if std else [], "deps": []}]},
        "packages": [{"id": "root", "name": "consumer", "targets": [{"kind": ["lib"]}]},
            {"id": "dependency", "name": name, "targets": [{"kind": ["lib"]}]}]}


def checked() -> dict:
    return {"bytes": [1], "payload": {"owner_sequence": [1, 2], "proof": 7},
        "tree": [{"metadata": {"effects": [], "span": "source:1..2"}}],
        "type_env": {"header": "retained"}, "signatures": {}, "linearity": {},
        "typed_nodes": 1, "total_nodes": 1}


def result(accepted: bool) -> dict:
    value = {"status": "accepted" if accepted else "rejected", "input": checked(), "rows": {"pure": []}}
    if accepted:
        value.update({"output": checked(), "targets": {target: {"status": "accepted"}
            for target in ("c", "hip", "metal")}})
    else:
        value.update({"stage": "effects", "errors": [{"kind": "UnhandledEffect", "message": "missing IO",
            "suggestions": ["declare IO"]}]})
    return value


def snapshot() -> dict:
    corpus = json.loads((oracle.FIXTURES / "corpus.json").read_text())
    return {"cases": {name: result(accepted) for name, _, accepted, _ in corpus["cases"]},
        "extensions": {name: {"contextual": result(accepted), "monolithic": result(accepted)}
            for name, accepted, _ in corpus["extensions"]},
        "library": checked(), "context": {"bytes": [1], "payload": {"binding_owner": 1}}}


class ExecutionControls(unittest.TestCase):
    def test_complete_execution(self):
        selected = {"owner::binary$case"}
        text = event(next(iter(selected)), "started") + "\n" + event(next(iter(selected)))
        self.assertEqual(oracle.check_execution(text, selected), selected)

    def test_missing_execution(self):
        with self.assertRaisesRegex(OracleFailure, "missing"):
            oracle.check_execution("", {"owner::binary$case"})

    def test_duplicate_execution(self):
        text = event("owner::binary$case")
        with self.assertRaisesRegex(OracleFailure, "duplicate"):
            oracle.check_execution(text + "\n" + text, {"owner::binary$case"})

    def test_unexpected_execution(self):
        with self.assertRaisesRegex(OracleFailure, "unexpected"):
            oracle.check_execution(event("foreign"), {"owner::binary$case"})

    def test_skipped_and_failed_execution(self):
        for outcome in ("ignored", "failed"):
            with self.subTest(outcome=outcome), self.assertRaisesRegex(OracleFailure, "did not pass"):
                oracle.check_execution(event("required", outcome), {"required"})

    def test_invalid_execution_json(self):
        with self.assertRaisesRegex(OracleFailure, "invalid"):
            oracle.check_execution("not json", {"required"})

    def test_selected_identity(self):
        self.assertEqual(oracle.selected_tests(inventory()), {"owner::binary$case"})

    def test_duplicate_selected_identity(self):
        value = inventory()
        value["rust-suites"]["duplicate"] = copy.deepcopy(value["rust-suites"]["suite"])
        with self.assertRaisesRegex(OracleFailure, "duplicate"):
            oracle.selected_tests(value)

    def test_ignored_selection(self):
        value = inventory()
        value["rust-suites"]["suite"]["testcases"]["case"]["ignored"] = True
        with self.assertRaisesRegex(OracleFailure, "ignored"):
            oracle.selected_tests(value)

    def test_empty_selection(self):
        with self.assertRaisesRegex(OracleFailure, "empty"):
            oracle.selected_tests({"rust-suites": {}})


class ReceiptControls(unittest.TestCase):
    def create(self, directory: Path) -> dict:
        receipts = []
        for index in range(2):
            files = {}
            for stream in ("stdout", "stderr"):
                name = f"command-{index}.{stream}"
                path = directory / name
                path.write_text("actual retained output\n")
                files[stream] = {"path": name, **oracle.digest(path)}
            receipts.append({"identity": index, "command": ["compiler", "--version"], "cwd": "/fixture",
                "exit": 0, "files": files})
        return {"status": "passed", "command_count": 2, "receipts": receipts}

    def save(self, directory: Path, value: dict):
        (directory / "receipts.json").write_text(json.dumps(value))

    def test_complete_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            self.save(directory, value)
            self.assertEqual(oracle.validate_receipts(directory), value["receipts"])

    def test_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(OracleFailure, "missing"):
                oracle.validate_receipts(Path(temporary))

    def test_missing_last_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"].pop()
            self.save(directory, value)
            with self.assertRaisesRegex(OracleFailure, "missing"):
                oracle.validate_receipts(directory)

    def test_duplicate_receipt_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"][1]["identity"] = 0
            self.save(directory, value)
            with self.assertRaisesRegex(OracleFailure, "duplicate"):
                oracle.validate_receipts(directory)

    def test_missing_raw_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            self.save(directory, value)
            (directory / "command-0.stderr").unlink()
            with self.assertRaisesRegex(OracleFailure, "missing"):
                oracle.validate_receipts(directory)

    def test_changed_raw_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            self.save(directory, value)
            (directory / "command-0.stdout").write_text("substituted output")
            with self.assertRaisesRegex(OracleFailure, "differs"):
                oracle.validate_receipts(directory)

    def test_receipt_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"][0]["files"]["stdout"]["path"] = "../foreign"
            self.save(directory, value)
            with self.assertRaisesRegex(OracleFailure, "differs"):
                oracle.validate_receipts(directory)

    def test_duplicate_raw_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"][1]["files"]["stdout"] = copy.deepcopy(value["receipts"][0]["files"]["stdout"])
            self.save(directory, value)
            with self.assertRaisesRegex(OracleFailure, "duplicate raw"):
                oracle.validate_receipts(directory)

    def test_missing_independent_lock_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"][0]["cwd"] = "/tmp/chelis-no-std-consumer-control"
            self.save(directory, value)
            with self.assertRaisesRegex(OracleFailure, "independent consumer lock"):
                oracle.validate_receipts(directory)

    def test_complete_independent_lock_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"][0]["cwd"] = "/tmp/chelis-no-std-consumer-control"
            path = directory / "consumer.Cargo.lock"
            path.write_text("version = 4\n")
            value["receipts"][0]["files"]["consumer_lock"] = {"path": path.name, **oracle.digest(path)}
            self.save(directory, value)
            self.assertEqual(oracle.validate_receipts(directory), value["receipts"])

    def test_boolean_receipt_identity_cannot_substitute_for_integer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = self.create(directory)
            value["receipts"][1]["identity"] = True
            self.save(directory, value)
            with self.assertRaisesRegex(OracleFailure, "receipt identity"):
                oracle.validate_receipts(directory)

class BaselineCheckoutControls(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ, {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.git(self.remote, "init", "--quiet")
        source = self.remote / "state"
        source.write_text("baseline\n")
        self.git(self.remote, "add", "state")
        self.git(self.remote, "commit", "--quiet", "-m", "Baseline")
        self.baseline = self.git(self.remote, "rev-parse", "HEAD").stdout.strip()
        source.write_text("candidate\n")
        self.git(self.remote, "add", "state")
        self.git(self.remote, "commit", "--quiet", "-m", "Candidate")
        self.shallow = self.root / "shallow"
        self.git(self.root, "clone", "--quiet", "--depth=1",
                 self.remote.as_uri(), str(self.shallow))
        self.assertNotEqual(self.git(
            self.shallow, "cat-file", "-e", f"{self.baseline}^{{commit}}",
            check=False).returncode, 0)

    def git(self, directory, *arguments, check=True):
        return subprocess.run((
            "git", "-c", "user.name=Oracle Fixture",
            "-c", "user.email=oracle@example.invalid", *arguments),
            cwd=directory, check=check, text=True, capture_output=True)

    def checkout(self, repository):
        with patch.multiple(oracle, ROOT=self.shallow, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=str(repository)):
            return oracle.checkout_baseline(self.root / "restored")

    def test_shallow_source_fetches_the_exact_immutable_commit(self):
        restored = self.checkout(self.remote)
        self.assertEqual((restored / "state").read_text(), "baseline\n")
        self.assertEqual(self.git(restored, "rev-parse", "HEAD").stdout.strip(),
                         self.baseline)

    def test_unavailable_immutable_source_rejects_acceptance(self):
        with self.assertRaises(OracleFailure):
            self.checkout(self.root / "missing.git")



class GraphControls(unittest.TestCase):
    def test_portable_graph(self):
        self.assertEqual(oracle.support._normal_packages(graph(), oracle.FORBIDDEN), {"consumer", "portable"})

    def test_every_forbidden_owner(self):
        for name in oracle.FORBIDDEN:
            with self.subTest(name=name), self.assertRaisesRegex(OracleFailure, "forbidden target dependency"):
                oracle.support._normal_packages(graph(name), oracle.FORBIDDEN)

    def test_dependency_std_feature(self):
        with self.assertRaisesRegex(OracleFailure, "target std feature"):
            oracle.support._normal_packages(graph(std=True), oracle.FORBIDDEN)

    def test_transitive_forbidden_owner(self):
        value = graph()
        value["resolve"]["nodes"][1]["deps"] = [{"pkg": "deep", "dep_kinds": [{"kind": None}]}]
        value["resolve"]["nodes"].append({"id": "deep", "features": [], "deps": []})
        value["packages"].append({"id": "deep", "name": "stacker", "targets": [{"kind": ["lib"]}]})
        with self.assertRaisesRegex(OracleFailure, "forbidden target dependency"):
            oracle.support._normal_packages(value, oracle.FORBIDDEN)

    def test_dev_edge_does_not_contaminate_normal_target(self):
        self.assertEqual(oracle.support._normal_packages(graph("stacker", kind="dev"), oracle.FORBIDDEN), {"consumer"})

    def test_incomplete_graph(self):
        value = graph()
        value["packages"].pop()
        with self.assertRaisesRegex(OracleFailure, "incomplete"):
            oracle.support._normal_packages(value, oracle.FORBIDDEN)


class SnapshotControls(unittest.TestCase):
    def test_exact_semantic_snapshot(self):
        value = snapshot()
        oracle.compare_snapshots(value, copy.deepcopy(value))

    def test_no_owner_context_or_annotation_normalization(self):
        previous = snapshot()
        mutations = [
            lambda value: value["library"]["payload"]["owner_sequence"].reverse(),
            lambda value: value["context"]["payload"].update(binding_owner=2),
            lambda value: value["cases"]["pure"]["output"]["tree"][0]["metadata"].update(effects=["io"]),
            lambda value: value["cases"]["declared_empty"]["errors"][0]["suggestions"].append("different repair"),
            lambda value: value["cases"]["pure"]["output"]["bytes"].append(2),
            lambda value: value["extensions"]["import_io"]["contextual"]["rows"].update(pure=["Io"]),
            lambda value: value["library"]["payload"]["owner_sequence"].__setitem__(0, True),
            lambda value: value["cases"]["pure"]["output"].update(typed_nodes=1.0),
        ]
        for index, mutation in enumerate(mutations):
            current = copy.deepcopy(previous)
            mutation(current)
            with self.subTest(index=index), self.assertRaisesRegex(OracleFailure, "differ"):
                oracle.compare_snapshots(previous, current)

    def test_missing_case(self):
        value = snapshot()
        value["cases"].pop("pure")
        with self.assertRaisesRegex(OracleFailure, "inventory"):
            oracle.validate_snapshot(value)

    def test_missing_checked_field(self):
        value = snapshot()
        del value["cases"]["pure"]["output"]["linearity"]
        with self.assertRaisesRegex(OracleFailure, "owner or context"):
            oracle.validate_snapshot(value)

    def test_missing_diagnostic_field(self):
        value = snapshot()
        del value["cases"]["declared_empty"]["errors"][0]["suggestions"]
        with self.assertRaisesRegex(OracleFailure, "diagnostic"):
            oracle.validate_snapshot(value)

    def test_wrong_disposition(self):
        value = snapshot()
        value["cases"]["pure"]["status"] = "rejected"
        with self.assertRaisesRegex(OracleFailure, "disposition"):
            oracle.validate_snapshot(value)

    def test_missing_context_path(self):
        value = snapshot()
        del value["extensions"]["import_io"]["monolithic"]
        with self.assertRaisesRegex(OracleFailure, "context path"):
            oracle.validate_snapshot(value)


class DiagnosticControls(unittest.TestCase):
    def value(self):
        conversions = []
        for kind in ("UnhandledEffect", "InvalidHandler", "BuildTargetMismatch", "TypeTotality"):
            for severity in (0.0, 0.25, 0.8, 1.0):
                bits = oracle.struct.unpack(">Q", oracle.struct.pack(">d", severity))[0]
                diagnostic = {"wire": {"kind": kind}, "kind": kind, "message": "detail",
                    "severity_bits": bits, "expected": None, "got": None,
                    "suggestions": ["first", "second"], "span": None,
                    "deep_path": None, "span_id": None, "unsupported_identity": "None"}
                conversions.append({"kind": kind, "severity_bits": bits, "diagnostic": diagnostic})
        corpus = json.loads((oracle.FIXTURES / "corpus.json").read_text())
        return {"conversions": conversions, "outcomes": {entry[0]: {
            "status": "rejected", "stage": "effects", "transcript": [],
            "diagnostics": [copy.deepcopy(conversions[0]["diagnostic"])]} for entry in corpus["cases"]}}

    def test_complete_host_diagnostic_fields(self):
        oracle.validate_diagnostics(self.value())

    def test_missing_host_diagnostic_details(self):
        value = self.value()
        del value["conversions"][0]["diagnostic"]["unsupported_identity"]
        with self.assertRaisesRegex(OracleFailure, "fields"):
            oracle.validate_diagnostics(value)

    def test_missing_conversion(self):
        value = self.value()
        value["conversions"].pop()
        with self.assertRaisesRegex(OracleFailure, "inventory"):
            oracle.validate_diagnostics(value)

    def test_duplicate_conversion(self):
        value = self.value()
        value["conversions"][1] = copy.deepcopy(value["conversions"][0])
        with self.assertRaisesRegex(OracleFailure, "duplicate"):
            oracle.validate_diagnostics(value)

    def test_wrong_supplied_severity(self):
        value = self.value()
        value["conversions"][0]["severity_bits"] = 1
        with self.assertRaisesRegex(OracleFailure, "supplied severities"):
            oracle.validate_diagnostics(value)

    def test_missing_compiler_detail(self):
        value = self.value()
        del value["outcomes"]["pure"]["diagnostics"][0]["span_id"]
        with self.assertRaisesRegex(OracleFailure, "compiler diagnostic"):
            oracle.validate_diagnostics(value)


class MigrationControls(unittest.TestCase):
    def setUp(self):
        self.moves = {"old_package": "old", "old_binary": "old_binary",
            "new_package": "new", "new_binary": "new_binary", "same_test_names": ["tests::moved"],
            "retired_tests": {}}
        self.before = "old::old_binary$tests::moved"
        self.after = "new::new_binary$tests::moved"

    def lane(self, identities):
        return {"lane": {"selected": identities, "executed": identities}}

    def test_moved_execution_identity(self):
        oracle.compare_inventories(self.lane([self.before]), self.lane([self.after]), self.moves)

    def test_missing_moved_execution(self):
        with self.assertRaisesRegex(OracleFailure, "absent"):
            oracle.compare_inventories(self.lane([self.before]), self.lane([]), self.moves)

    def test_duplicate_execution_obligation(self):
        current = self.lane([self.after])
        current["duplicate"] = copy.deepcopy(current["lane"])
        with self.assertRaisesRegex(OracleFailure, "duplicate"):
            oracle.compare_inventories(self.lane([self.before]), current, self.moves)

    def test_obsolete_identity_cannot_remain(self):
        with self.assertRaisesRegex(OracleFailure, "obsolete"):
            oracle.compare_inventories(self.lane([self.before]), self.lane([self.before, self.after]), self.moves)

    def test_frozen_inventory_requires_execution(self):
        current = self.lane([self.after])
        current["lane"]["executed"] = []
        with self.assertRaisesRegex(OracleFailure, "incomplete"):
            oracle.compare_inventories(self.lane([self.before]), current, self.moves)

    def test_explicit_source_wording_retirement(self):
        self.moves["retired_tests"] = {oracle.RETIRED_SOURCE_TEST: "Main explicitly retires this source-wording test."}
        previous = self.lane([self.before, oracle.RETIRED_SOURCE_TEST])
        oracle.compare_inventories(previous, self.lane([self.after]), self.moves)
        self.assertIn(oracle.RETIRED_SOURCE_TEST, previous["lane"]["executed"])

    def test_retired_test_cannot_move_to_another_binary(self):
        self.moves["retired_tests"] = {oracle.RETIRED_SOURCE_TEST: "Explicit source-wording retirement."}
        retired_name = oracle.RETIRED_SOURCE_TEST.split("$", 1)[1]
        current = self.lane([self.after, "new::new_binary$" + retired_name])
        with self.assertRaisesRegex(OracleFailure, "retired source-wording"):
            oracle.compare_inventories(self.lane([self.before, oracle.RETIRED_SOURCE_TEST]), current, self.moves)

    def test_retirement_needs_baseline_execution(self):
        self.moves["retired_tests"] = {oracle.RETIRED_SOURCE_TEST: "Explicit source-wording retirement."}
        with self.assertRaisesRegex(OracleFailure, "original baseline execution"):
            oracle.compare_inventories(self.lane([self.before]), self.lane([self.after]), self.moves)

    def test_behavior_test_cannot_retire(self):
        self.moves["retired_tests"] = {self.before: "Remove a behavior test."}
        with self.assertRaisesRegex(OracleFailure, "unapproved test retirement"):
            oracle.compare_inventories(self.lane([self.before]), self.lane([]), self.moves)

    def test_timing_value_moves_without_reset(self):
        oracle.check_timing({"old::tests::moved": 1.932}, {"new::tests::moved": 1.932}, self.moves)

    def test_timing_reset_rejects(self):
        with self.assertRaisesRegex(OracleFailure, "measured baseline"):
            oracle.check_timing({"old::tests::moved": 1.932}, {"new::tests::moved": 0.1}, self.moves)

    def test_historical_timing_name_stays(self):
        oracle.check_timing({"old::historical": 0.008}, {"old::historical": 0.008}, self.moves)


if __name__ == "__main__":
    unittest.main()
