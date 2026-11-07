#!/usr/bin/env python3
"""Exercise IR evidence rejection through data and real fixture contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import ir_core_oracle as oracle
from no_std_core_support import OracleFailure


def metadata(cases=(('passes', False), ('manual', True))):
    return {"rust-suites": {"suite": {"package-name": "owner", "binary-name": "binary",
        "testcases": {name: {"kind": "test", "ignored": ignored,
            "filter-match": {"status": "matches"}} for name, ignored in cases}}}}


def events(*values):
    return "\n".join(json.dumps({"type": "test", "name": name, "event": event}) for name, event in values)


def lane(*names):
    return {"selected": sorted(names), "executed": sorted(names)}


def migrations(**values):
    return {"baseline": oracle.BASELINE, "renamed": {}, "retired": {}, "new_tests": [], **values}


class InventoryControls(unittest.TestCase):
    def test_complete_inventory_keeps_manual_obligations(self):
        self.assertEqual(oracle.inventory_cases(metadata()), {"owner::binary$passes": False,
            "owner::binary$manual": True})

    def test_empty_and_duplicate_discovery_fail(self):
        with self.assertRaises(OracleFailure):
            oracle.inventory_cases(metadata(()))
        value = metadata()
        value["rust-suites"]["duplicate"] = copy.deepcopy(value["rust-suites"]["suite"])
        with self.assertRaises(OracleFailure):
            oracle.inventory_cases(value)

    def test_invalid_filter_and_kind_fail(self):
        for key, bad in (("kind", "benchmark"), ("filter-match", {"status": "unknown"})):
            value = metadata()
            value["rust-suites"]["suite"]["testcases"]["passes"][key] = bad
            with self.assertRaises(OracleFailure):
                oracle.inventory_cases(value)

    def test_original_ignored_start_parser_failure_stays_repaired(self):
        selected, ignored = {"owner::binary$passes"}, {"owner::binary$manual"}
        output = events(("owner::binary$manual", "started"), ("owner::binary$manual", "ignored"),
            ("owner::binary$passes", "started"), ("owner::binary$passes", "ok"))
        self.assertEqual(oracle.execution(output, selected, ignored), selected)

    def test_missing_duplicate_failed_and_substitute_events_fail(self):
        name = "owner::binary$passes"
        for output in ("", events((name, "ok")), events((name, "started"), (name, "ok"), (name, "ok")),
                events((name, "started"), (name, "failed")),
                events(("foreign::binary$passes", "started"), ("foreign::binary$passes", "ok"))):
            with self.assertRaises(OracleFailure):
                oracle.execution(output, {name}, set())

    def test_unselected_manual_execution_fails(self):
        with self.assertRaises(OracleFailure):
            oracle.execution(events(("owner::binary$manual", "ok")), {"owner::binary$passes"},
                {"owner::binary$manual"})

    def test_junit_requires_exact_success_and_time(self):
        start = '<testsuites><testsuite name="suite">'
        end = '</testsuite></testsuites>'
        valid = '<testcase name="passes" time="1"><system-out>work=4</system-out></testcase>'
        result = oracle.junit(start + valid + end, metadata(), {"owner::binary$passes"}, {"owner::binary$manual"})
        self.assertEqual(result["observations"]["owner::binary$passes"]["stdout"], "work=4")
        for body in ("", valid + valid, '<testcase name="passes"/>',
                '<testcase name="passes" time="1"><skipped/></testcase>',
                '<testcase name="passes" time="1"><failure/></testcase>',
                '<testcase name="foreign" time="1"/>'):
            with self.assertRaises(OracleFailure):
                oracle.junit(start + body + end, metadata(), {"owner::binary$passes"}, {"owner::binary$manual"})


class MigrationControls(unittest.TestCase):
    def test_explicit_owner_move_succeeds(self):
        before, after = "old::binary$case", "new::binary$case"
        self.assertEqual(oracle.reconcile({"lane": lane(before)}, {"lane": lane(after)},
            migrations(renamed={before: after})), {before: after})

    def test_name_only_owner_inference_fails(self):
        with self.assertRaises(OracleFailure):
            oracle.reconcile({"lane": lane("old::binary$case")}, {"lane": lane("new::binary$case")}, migrations())

    def test_missing_execution_and_duplicate_lanes_fail(self):
        identity = "owner::binary$case"
        for value in ({"lane": {"selected": [identity], "executed": []}},
                {"one": lane(identity), "two": lane(identity)}):
            with self.assertRaises(OracleFailure):
                oracle.reconcile({"lane": lane(identity)}, value, migrations())

    def test_manual_omission_and_shared_destination_fail(self):
        before = {"lane": lane("old::binary$a", "old::binary$b")}
        after = {"lane": lane("new::binary$a")}
        with self.assertRaises(OracleFailure):
            oracle.reconcile(before, after, migrations(renamed={"old::binary$a": "new::binary$a",
                "old::binary$b": "new::binary$a"}))
        with self.assertRaises(OracleFailure):
            oracle.reconcile({"lane": lane("owner::binary$a")}, {"lane": lane("owner::binary$a")},
                migrations(), manual_before=["owner::binary$manual"])

    def test_retired_assertion_needs_executed_behavior(self):
        original, behavior = "owner::binary$spelling", "owner::binary$behavior"
        retired = {original: {"reason": "An incidental source assertion retires.", "behavior_identities": [behavior]}}
        with self.assertRaises(OracleFailure):
            oracle.reconcile({"lane": lane(original)}, {"lane": lane("owner::binary$other")},
                migrations(retired=retired))
        self.assertEqual(oracle.reconcile({"lane": lane(original)}, {"lane": lane(behavior)},
            migrations(retired=retired)), {})

    def test_timing_moves_preserve_values_and_samples(self):
        oracle.validate_timing({"old": 9.5}, {"new": 9.5}, {"old": "new"})
        for value in ({"new": 0}, {"new": 9.5, "old": 9.5}, {}):
            with self.assertRaises(OracleFailure):
                oracle.validate_timing({"old": 9.5}, value, {"old": "new"})
        old = {"version": 1, "targets": {"old": {"milliseconds": 9, "samples": 3}}}
        new = {"version": 1, "targets": {"new": {"milliseconds": 9, "samples": 2}}}
        with self.assertRaises(OracleFailure):
            oracle.validate_timing(old, new, {"old": "new"}, durations=True)


class SemanticControls(unittest.TestCase):
    def test_wire_diagnostic_trap_and_cost_changes_fail(self):
        original = {"binary": [0, 1, 255], "diagnostic": {"span": 9, "fatal": True},
            "trap": "numeric trap: overflow in add at i64", "work": 8}
        for name, value in (("binary", [0, 1, 254]), ("diagnostic", {"span": 9, "fatal": False}),
                ("trap", "accepted"), ("work", 9)):
            current = {**original, name: value}
            with self.assertRaises(OracleFailure):
                oracle.compare_exact(original, current, name)

    def test_real_work_counter_and_workload_scale_changes_fail(self):
        before = "eval-performance planner contractions=4096 nodes=16386 wall=1ms PASS"
        after = "eval-performance planner contractions=4096 nodes=16386 wall=9ms PASS"
        self.assertEqual(oracle.measured_work(before), oracle.measured_work(after))
        for changed in (after.replace("4096", "256"), after.replace("16386", "20000")):
            with self.assertRaises(OracleFailure):
                oracle.compare_exact(oracle.measured_work(before), oracle.measured_work(changed), "work contract")

    def test_duplicate_and_nonfinite_json_fail(self):
        for value in ('{"same":1,"same":2}', '{"value":NaN}'):
            with self.assertRaises(OracleFailure):
                oracle.decode(value)

    def test_all_frozen_case_and_context_identities_remain_mandatory(self):
        schema = oracle.load(oracle.FIXTURES / "schema.json")
        value = {"version": 1, "cases": [{key: identity if key == "id" else {}
            for key in schema["record_fields"] if key != "check_error"} for identity in schema["case_ids"]],
            "contexts": [{key: identity if key == "id" else {} for key in schema["context_fields"]}
                for identity in schema["context_ids"]]}
        oracle.validate_snapshot(value)
        for section in ("cases", "contexts"):
            missing = copy.deepcopy(value)
            missing[section].pop()
            with self.assertRaises(OracleFailure):
                oracle.validate_snapshot(missing)
            duplicate = copy.deepcopy(value)
            duplicate[section].append(copy.deepcopy(duplicate[section][0]))
            with self.assertRaises(OracleFailure):
                oracle.validate_snapshot(duplicate)

    def test_missing_measured_cost_output_fails(self):
        names = ["owner::eval_performance$full", "owner::callable_snapshot_allocations$full"]
        old = {"lane": {"junit": {"observations": {name: {"stdout": "work=8", "stderr": ""} for name in names}}}}
        current = copy.deepcopy(old)
        current["lane"]["junit"]["observations"][names[0]]["stdout"] = ""
        with self.assertRaises(OracleFailure):
            oracle.cost_observations(old, current, {})


class DependencyControls(unittest.TestCase):
    def graph(self, owner="chelis-ir-core", features=(), kind=None):
        return {"resolve": {"root": "consumer", "nodes": [
            {"id": "consumer", "features": [], "deps": [
                {"pkg": "dependency", "dep_kinds": [{"kind": kind}]}]},
            {"id": "dependency", "features": list(features), "deps": []}]},
            "packages": [{"id": "consumer", "name": "consumer", "targets": [{"kind": ["lib"]}]},
                {"id": "dependency", "name": owner, "targets": [{"kind": ["lib"]}]}]}

    def test_transitive_native_and_std_paths_fail(self):
        for value in (self.graph("chelis-ir"), self.graph("ordinary", ("std",))):
            with self.assertRaisesRegex(OracleFailure, "Dependency paths"):
                oracle.inspect_graph(value)

    def test_dev_only_owner_does_not_pollute_normal_graph(self):
        self.assertEqual(oracle.inspect_graph(self.graph("chelis-ir", kind="dev"))["packages"], ["consumer"])


class ReceiptControls(unittest.TestCase):
    def document(self, directory):
        path = directory / "stdout"
        path.write_text("evidence\n")
        stderr = directory / "stderr"
        stderr.write_text("")
        tool = Path(sys.executable).resolve()
        receipt = {"identity": 0, "command": [str(tool), "--version"], "exit": 0, "elapsed_ns": 1,
            "consumer": False, "tool": {"path": str(tool), **oracle.digest(tool)},
            "files": {"stdout": {"path": path.name, **oracle.digest(path)},
                "stderr": {"path": stderr.name, **oracle.digest(stderr)}}}
        inputs = directory / "inputs.json"
        inputs.write_text(json.dumps({"command": receipt["command"], "tool": receipt["tool"],
            "environment_tools": {}}))
        receipt["files"]["inputs"] = {"path": inputs.name, **oracle.digest(inputs)}
        return {"status": "passed", "scope": "implementation", "platform": sys.platform,
            "machine": "test", "python": sys.executable, "python_version": "test",
            "script": oracle.digest(Path(oracle.__file__)), "invocation": ["behavioral-control"],
            "command_count": 1, "receipts": [receipt]}

    def test_real_file_hashes_and_tool_identity_are_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document = self.document(directory)
            (directory / "receipts.json").write_text(json.dumps(document))
            self.assertEqual(len(oracle.validate_receipts(directory)), 1)
            (directory / "stdout").write_text("altered evidence\n")
            with self.assertRaises(OracleFailure):
                oracle.validate_receipts(directory)

    def test_missing_lock_unrelated_error_and_duplicate_command_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for mutation in ("lock", "error", "duplicate", "tool"):
                document = self.document(directory)
                receipt = document["receipts"][0]
                if mutation == "lock":
                    receipt["consumer"] = True
                elif mutation == "error":
                    receipt["exit"] = 1
                    receipt["expected_failure"] = {"fragments": ["expected private field"]}
                elif mutation == "tool":
                    receipt["tool"]["sha256"] = "0" * 64
                else:
                    document["receipts"].append(copy.deepcopy(receipt))
                    document["command_count"] = 2
                (directory / "receipts.json").write_text(json.dumps(document))
                with self.assertRaises(OracleFailure):
                    oracle.validate_receipts(directory)


if __name__ == "__main__":
    unittest.main()
