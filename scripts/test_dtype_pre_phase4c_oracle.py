#!/usr/bin/env python3
"""Executed positive/negative controls for the pre-4C composite framework."""

from __future__ import annotations

from dataclasses import replace
import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import dtype_pre_phase4c_oracle as oracle


class ManifestTests(unittest.TestCase):
    def test_complete_input_set_includes_host_direct_arithmetic_not_device_count(self):
        inputs = oracle.prerequisites(sys.executable)
        actual = {issue for child in inputs for issue in child.issues}
        self.assertEqual(actual, oracle.REQUIRED_ISSUES)
        self.assertIn(1306, actual)
        self.assertIn(1313, actual)
        self.assertNotIn(1291, actual)
        self.assertNotIn(1296, actual)
        oracle.validate_manifest(inputs)

    def test_missing_duplicate_and_replacement_owners_fail(self):
        inputs = oracle.prerequisites(sys.executable)
        for changed in (inputs[1:], inputs + (inputs[0],),
                        (replace(inputs[0], issues=(9999,)),) + inputs[1:]):
            with self.subTest(changed=changed):
                with self.assertRaises(oracle.OracleFailure):
                    oracle.validate_manifest(changed)

    def test_oracles_are_explicitly_missing_not_replaced_by_neighboring_suites(self):
        inputs = oracle.prerequisites(sys.executable)
        unresolved = {issue for child in inputs if isinstance(child, oracle.MissingOracle)
                      for issue in child.issues}
        self.assertTrue({893, 1288, 1294}.issubset(unresolved))
        with self.assertRaisesRegex(oracle.OracleFailure, "#1288"):
            oracle.require_available(inputs)

    def test_known_exact_commands_and_markers(self):
        available = {child.name: child for child in oracle.prerequisites(sys.executable)
                     if isinstance(child, oracle.ChildOracle)}
        self.assertEqual(available["count"].argv,
                         (sys.executable, "scripts/dtype_count_oracle.py"))
        self.assertEqual(available["count"].success_line, "DTYPE COUNT ORACLE: PASS")
        self.assertEqual(available["direct-arithmetic"].issues, (1306,))
        self.assertEqual(available["relu"].issues, (1313,))


def packet(*, run_id="fresh", head="a" * 40, digest="b" * 64):
    return {
        "schema": 1, "run_id": run_id, "head": head, "source_digest": digest,
        "oracle": "example", "argv": [sys.executable, "child.py"],
        "selected": ["positive", "negative", "mutation"],
        "executed": [{"id": name, "outcome": "passed"}
                     for name in ("positive", "negative", "mutation")],
        "obligations": {"positive": ["positive"], "negative": ["negative"],
                        "mutation": ["mutation"]},
        "hosts": {"eval": ["positive"], "c-host": ["positive"], "c-dag": ["positive"]},
        "devices": [],
    }


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.child = oracle.ChildOracle("example", (1287,),
                                        (sys.executable, "child.py"), "EXAMPLE: PASS")
        self.identity = oracle.SourceIdentity("a" * 40, "b" * 64)

    def validate(self, payload):
        return oracle.validate_receipt(payload, self.child, self.identity, "fresh")

    def test_exact_selected_and_executed_cases_pass(self):
        self.assertEqual(self.validate(packet()), 3)

    def test_missing_extra_duplicate_skipped_failed_and_empty_cases_fail(self):
        mutations = []
        p = packet(); p["executed"].pop(); mutations.append(p)
        p = packet(); p["executed"].append({"id": "extra", "outcome": "passed"}); mutations.append(p)
        p = packet(); p["executed"].append(p["executed"][0]); mutations.append(p)
        for outcome in ("skipped", "ignored", "failed", "waived", "unknown"):
            p = packet(); p["executed"][0]["outcome"] = outcome; mutations.append(p)
        p = packet(); p["selected"] = []; p["executed"] = []; mutations.append(p)
        p = packet(); p["selected"].append("positive"); mutations.append(p)
        for p in mutations:
            with self.subTest(packet=p), self.assertRaises(oracle.OracleFailure):
                self.validate(p)

    def test_stale_head_artifact_nonce_identity_or_command_fails(self):
        for key, value in (("head", "c" * 40), ("source_digest", "d" * 64),
                           ("run_id", "old"), ("oracle", "other"),
                           ("argv", [sys.executable, "other.py"]), ("schema", 0)):
            p = packet(); p[key] = value
            with self.subTest(key=key), self.assertRaises(oracle.OracleFailure):
                self.validate(p)

    def test_positive_negative_and_mutation_obligations_are_mandatory(self):
        for kind in ("positive", "negative", "mutation"):
            p = packet(); p["obligations"][kind] = []
            with self.subTest(kind=kind), self.assertRaises(oracle.OracleFailure):
                self.validate(p)
        p = packet(); p["obligations"]["mutation"] = ["never-ran"]
        with self.assertRaises(oracle.OracleFailure):
            self.validate(p)

    def test_each_required_host_lane_needs_executed_cases(self):
        for value in ({}, {"eval": ["positive"]},
                      {"eval": [], "c-host": ["positive"], "c-dag": ["positive"]},
                      {"eval": ["unexecuted"], "c-host": ["positive"], "c-dag": ["positive"]}):
            p = packet(); p["hosts"] = value
            with self.subTest(hosts=value), self.assertRaises(oracle.OracleFailure):
                self.validate(p)

    def test_structural_prerequisites_do_not_invent_host_execution(self):
        for issues in ((), (1288,), (1294,), (1288, 1294)):
            child = replace(self.child, issues=issues)
            p = packet(); p["hosts"] = {}
            with self.subTest(issues=issues):
                self.assertEqual(oracle.validate_receipt(p, child, self.identity, "fresh"), 3)
                with self.assertRaises(oracle.OracleFailure):
                    oracle.validate_receipt(packet(), child, self.identity, "fresh")
        # Combining a structural obligation with behavior cannot waive a lane.
        p = packet(); p["hosts"] = {}
        with self.assertRaises(oracle.OracleFailure):
            oracle.validate_receipt(p, replace(self.child, issues=(1288, 1287)),
                                    self.identity, "fresh")

    def test_unimplemented_device_receipt_requires_lane_cell_and_issue_identity(self):
        p = packet()
        p["devices"] = [{"lane": "hip", "cell": "count/bool/tensor",
                          "kind": "Unimplemented", "issue": 1291}]
        self.assertEqual(self.validate(p), 3)
        for change in ({"kind": "waived"}, {"issue": 0}, {"lane": "c-host"}):
            bad = packet(); bad["devices"] = [{**p["devices"][0], **change}]
            with self.subTest(change=change), self.assertRaises(oracle.OracleFailure):
                self.validate(bad)

    def test_unknown_fields_cannot_smuggle_waivers(self):
        p = packet(); p["waiver"] = "manual"
        with self.assertRaises(oracle.OracleFailure):
            self.validate(p)

    def test_duplicate_json_fields_cannot_replace_failed_outcomes(self):
        text = '{"executed": [{"outcome": "failed", "outcome": "passed"}]}'
        with self.assertRaisesRegex(oracle.OracleFailure, "duplicate JSON"):
            json.loads(text, object_pairs_hook=oracle._unique_json_fields)

    def test_device_authority_rejects_closed_missing_and_pull_request_owners(self):
        from capacity_census_liveness import IssueKind, IssueRecord, IssueState
        p = packet()
        p["devices"] = [{"lane": "hip", "cell": "count/bool/tensor",
                          "kind": "Unimplemented", "issue": 1291}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "execution.json").write_text(json.dumps(p))
            receipts = [{"receipt": "execution.json"}]
            records = [None, IssueRecord(IssueKind.ISSUE, IssueState.CLOSED),
                       IssueRecord(IssueKind.PULL_REQUEST, IssueState.OPEN)]
            with mock.patch("generate_rejection_registries.load_issue_manifest", return_value=[1291]):
                for record in records:
                    with self.subTest(record=record), mock.patch(
                            "capacity_census_liveness.fetch_issue", return_value=record):
                        with self.assertRaises(oracle.OracleFailure):
                            oracle.validate_device_authorities(receipts, root)
                with mock.patch("capacity_census_liveness.fetch_issue",
                                return_value=IssueRecord(IssueKind.ISSUE, IssueState.OPEN)):
                    oracle.validate_device_authorities(receipts, root)
            with mock.patch("generate_rejection_registries.load_issue_manifest", return_value=[]):
                with self.assertRaises(oracle.OracleFailure):
                    oracle.validate_device_authorities(receipts, root)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.identity = oracle.SourceIdentity("a" * 40, "b" * 64)
        self.child = oracle.ChildOracle("example", (1287,),
                                        (sys.executable, "child.py"), "EXAMPLE: PASS")

    def write_child(self, behavior="pass"):
        # The fixture executes the same protocol boundary as a real child.
        source = '''import json, os, sys
from pathlib import Path
p = json.loads(os.environ["FIXTURE_PACKET"])
p["run_id"] = os.environ["CHELIS_ORACLE_RUN_ID"]
if BEHAVIOR == "stale": p["run_id"] = "stale"
if BEHAVIOR != "missing":
    Path(os.environ["CHELIS_ORACLE_RECEIPT"]).write_text(json.dumps(p))
print("EXAMPLE: PASS" if BEHAVIOR != "marker" else "looks green")
if BEHAVIOR == "nonzero": sys.exit(1)
'''.replace("BEHAVIOR", repr(behavior))
        (self.root / "child.py").write_text(source)

    def execute(self, output):
        return oracle.run_child(self.child, self.root, self.identity, "fresh", output,
                                extra_env={"FIXTURE_PACKET": json.dumps(packet())})

    def test_real_child_process_produces_distinct_retained_evidence(self):
        self.write_child()
        result = self.execute(self.root / "receipts" / "example")
        self.assertEqual(result["executed_tests"], 3)
        self.assertTrue((self.root / "receipts/example/stdout.log").is_file())
        self.assertTrue((self.root / "receipts/example/execution.json").is_file())

    def test_real_child_nonzero_missing_marker_receipt_and_stale_receipt_fail(self):
        for behavior in ("nonzero", "marker", "missing", "stale"):
            with self.subTest(behavior=behavior):
                self.write_child(behavior)
                with self.assertRaises(oracle.OracleFailure):
                    self.execute(self.root / behavior)

    def test_preexisting_receipt_directory_is_not_reused(self):
        self.write_child()
        output = self.root / "old"
        output.mkdir()
        (output / "execution.json").write_text(json.dumps(packet()))
        with self.assertRaises(oracle.OracleFailure):
            self.execute(output)

    def test_actual_cli_fails_for_current_missing_prerequisites_without_pass(self):
        completed = subprocess.run([sys.executable, str(Path(oracle.__file__))],
                                   capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn(oracle.PASS_LINE, completed.stdout.splitlines())
        self.assertIn("#1288", completed.stdout + completed.stderr)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git("init", "--quiet")
        (self.root / "source.txt").write_text("source\n")
        (self.root / ".gitignore").write_text("target/\n")
        self.git("add", ".")
        self.commit()

    def git(self, *args):
        return subprocess.run(("git", *args), cwd=self.root, check=True,
                              capture_output=True, text=True)

    def commit(self):
        self.git("-c", "user.name=Oracle Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")

    def test_clean_source_identity_is_stable_and_committed_change_is_distinct(self):
        first = oracle.source_identity(self.root)
        self.assertEqual(first, oracle.source_identity(self.root))
        (self.root / "source.txt").write_text("changed\n")
        self.git("add", ".")
        self.commit()
        self.assertNotEqual(first, oracle.source_identity(self.root))

    def test_modified_staged_or_untracked_source_cannot_claim_exact_head(self):
        source = self.root / "source.txt"
        source.write_text("changed\n")
        with self.assertRaises(oracle.OracleFailure):
            oracle.source_identity(self.root)
        self.git("add", ".")
        with self.assertRaises(oracle.OracleFailure):
            oracle.source_identity(self.root)
        self.commit()
        (self.root / "new.txt").write_text("new\n")
        with self.assertRaises(oracle.OracleFailure):
            oracle.source_identity(self.root)

    def test_nonrepository_is_not_an_exact_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(oracle.OracleFailure):
                oracle.source_identity(Path(tmp))

    def test_git_visibility_flags_cannot_hide_changed_or_missing_tracked_bytes(self):
        source = self.root / "source.txt"
        for flag, reset in (("--assume-unchanged", "--no-assume-unchanged"),
                            ("--skip-worktree", "--no-skip-worktree")):
            self.git("update-index", flag, "source.txt")
            clean = oracle.source_identity(self.root)
            source.write_text("hidden source change\n")
            self.assertEqual(self.git("status", "--porcelain").stdout, "")
            with self.subTest(flag=flag), self.assertRaises(oracle.OracleFailure):
                oracle.source_identity(self.root)
            source.unlink()
            with self.subTest(flag=flag, missing=True), self.assertRaises(oracle.OracleFailure):
                oracle.source_identity(self.root)
            source.write_text("source\n")
            self.assertEqual(clean, oracle.source_identity(self.root))
            self.git("update-index", reset, "source.txt")

    def test_committed_symlink_is_hashed_as_link_not_followed(self):
        link = self.root / "link"
        link.symlink_to("source.txt")
        self.git("add", ".")
        self.commit()
        first = oracle.source_identity(self.root)
        self.git("update-index", "--assume-unchanged", "link")
        link.unlink()
        link.symlink_to("missing.txt")
        with self.assertRaises(oracle.OracleFailure):
            oracle.source_identity(self.root)
        link.unlink()
        link.symlink_to("source.txt")
        self.assertEqual(first, oracle.source_identity(self.root))

    def test_complete_fixture_composite_writes_receipt_only_after_all_children_pass(self):
        children = (
            oracle.ChildOracle("freeze", (), (sys.executable, "freeze.py"), "FREEZE: PASS"),
            oracle.ChildOracle("example", tuple(sorted(oracle.REQUIRED_ISSUES)),
                               (sys.executable, "example.py"), "EXAMPLE: PASS"),
        )
        for child in children:
            p = packet()
            p["oracle"] = child.name
            p["argv"] = list(child.argv)
            if not child.issues:
                p["hosts"] = {}
            source = f'''import json, os
from pathlib import Path
p = json.loads({json.dumps(p)!r})
p["run_id"] = os.environ["CHELIS_ORACLE_RUN_ID"]
p["head"] = os.environ["CHELIS_ORACLE_HEAD"]
p["source_digest"] = os.environ["CHELIS_ORACLE_SOURCE_DIGEST"]
Path(os.environ["CHELIS_ORACLE_RECEIPT"]).write_text(json.dumps(p))
print({child.success_line!r})
'''
            (self.root / child.argv[1]).write_text(source)
        self.git("add", ".")
        self.commit()
        with mock.patch.object(oracle, "REPO_ROOT", self.root), \
                mock.patch.object(oracle, "prerequisites", return_value=children), \
                mock.patch("generate_rejection_registries.load_issue_manifest", return_value=[]):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(oracle.main(["--receipt-dir", str(self.root / "target/pass")]), 0)
            self.assertEqual(output.getvalue().splitlines()[-1], oracle.PASS_LINE)
            summary = json.loads((self.root / "target/pass/receipt.json").read_text())
            self.assertEqual([row["oracle"] for row in summary["children"]], ["freeze", "example"])
            self.assertNotEqual(summary["children"][0]["receipt"], summary["children"][1]["receipt"])

            script = self.root / "example.py"
            script.write_text(script.read_text() + '\nPath("source.txt").write_text("hidden changed source\\n")\n')
            self.git("add", ".")
            self.commit()
            for flag, reset in (("--assume-unchanged", "--no-assume-unchanged"),
                                ("--skip-worktree", "--no-skip-worktree")):
                self.git("update-index", flag, "source.txt")
                output, errors = io.StringIO(), io.StringIO()
                receipt_dir = self.root / "target" / flag.removeprefix("--")
                with self.subTest(flag=flag), redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(oracle.main(["--receipt-dir", str(receipt_dir)]), 1)
                self.assertNotIn(oracle.PASS_LINE, output.getvalue())
                self.assertFalse((receipt_dir / "receipt.json").exists())
                self.assertIn("tracked source bytes differ", errors.getvalue())
                (self.root / "source.txt").write_text("source\n")
                self.git("update-index", reset, "source.txt")

            (self.root / "example.py").write_text("print('EXAMPLE: PASS')\n")
            self.git("add", ".")
            self.commit()
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(oracle.main(["--receipt-dir", str(self.root / "target/fail")]), 1)
            self.assertNotIn(oracle.PASS_LINE, output.getvalue())
            self.assertFalse((self.root / "target/fail/receipt.json").exists())
            self.assertIn("receipt adapter required", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
