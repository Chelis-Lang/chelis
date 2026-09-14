"""Changed-test review cues from committed Phase 3 inventories and definitions."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import unittest
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

import yaml

import phase3_test_change_report as report
import faithful_observation_phase3_oracle as oracle


ORACLE = "scripts/faithful_observation_phase3_oracle.py"
SOURCE = "crates/example/tests/parity.rs"
BODY = '#[test]\nfn parity_example() { assert_eq!(1, 1); }\n'


def inventory(names: tuple[str, ...] = ("parity_example",), path: str = SOURCE) -> str:
    return (
        "from pathlib import Path\n"
        f"PARITY_SOURCE = Path({path!r})\n"
        f"REQUIRED_TESTS = {{PARITY_SOURCE: {{{', '.join(repr(n) for n in names)}}}}}\n"
    )


class InventoryTests(unittest.TestCase):
    def test_reads_literal_inventory_without_executing_python(self):
        source = "raise RuntimeError('must not execute')\n" + inventory()
        self.assertEqual(report.read_required_tests(source), {SOURCE: {"parity_example"}})

    def test_refuses_missing_dynamic_duplicate_or_escaping_inventory(self):
        for source in [
            "# no inventory\n",
            "REQUIRED_TESTS = load_from_somewhere()\n",
            inventory() + "REQUIRED_TESTS = {}\n",
            inventory() + "REQUIRED_TESTS.update({PARITY_SOURCE: {'another'}})\n",
            inventory() + "alias = REQUIRED_TESTS\nalias.clear()\n",
            inventory(path="../outside.rs"),
            inventory(names=("parity_example", "parity_example")),
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                report.read_required_tests(source)


class CommittedChanges(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "--initial-branch=fixture")
        self.git("config", "user.name", "Phase 3 fixture")
        self.git("config", "user.email", "fixture@example.invalid")

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", "maintenance.auto=false", "-C", str(self.root), *args],
            text=True, stderr=subprocess.PIPE,
        ).strip()

    def commit(self, source=BODY, oracle=None):
        for path, text in [(ORACLE, inventory() if oracle is None else oracle), (SOURCE, source)]:
            file = self.root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
        self.git("add", ".")
        self.git("commit", "--allow-empty", "-m", "fixture")
        return self.git("rev-parse", "HEAD")

    def compare(self, base, candidate="HEAD"):
        return report.changed_tests(self.root, base, candidate)

    def test_unchanged_commit_reports_no_changes(self):
        base = self.commit()
        result = self.compare(base)
        self.assertEqual(result["base"], base)
        self.assertEqual(result["candidate"], base)
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["problems"], [])

    def test_body_edit_is_named_with_source_locations(self):
        base = self.commit()
        head = self.commit(BODY.replace("1, 1", "2, 2"))
        result = self.compare(base)
        self.assertEqual(result["candidate"], head)
        self.assertEqual(len(result["changes"]), 1)
        row = result["changes"][0]
        self.assertEqual((row["path"], row["test"]), (SOURCE, "parity_example"))
        self.assertEqual(row["changes"], ["definition changed"])
        self.assertEqual(row["before_lines"], [1, 2])
        self.assertEqual(row["after_lines"], [1, 2])

    def test_empty_body_is_reported_and_original_digest_control_remains_separate(self):
        base = self.commit()
        self.commit('#[test]\nfn parity_example() {}\n')
        result = self.compare(base)
        self.assertEqual(result["changes"][0]["changes"], ["definition changed"])
        self.assertEqual(result["problems"], [])

    def test_removed_required_definition_is_named_and_invalid(self):
        base = self.commit()
        self.commit("// deleted test\n")
        result = self.compare(base)
        self.assertEqual(result["changes"][0]["changes"], ["definition removed"])
        self.assertIn("parity_example", "\n".join(result["problems"]))

    def test_commented_required_definition_is_named_and_invalid(self):
        base = self.commit()
        self.commit('/*\n' + BODY + '*/\n')
        result = self.compare(base)
        self.assertEqual(result["changes"][0]["changes"], ["definition removed"])
        self.assertIn("parity_example", "\n".join(result["problems"]))

    def test_commented_duplicate_is_not_an_ambiguous_active_definition(self):
        base = self.commit()
        self.commit(BODY + '/*\n' + BODY + '*/\n')
        self.assertEqual(self.compare(base)["changes"], [])

    def test_restoring_a_missing_required_definition_can_be_reported(self):
        base = self.commit("// missing test in the base\n")
        self.commit()
        result = self.compare(base)
        self.assertEqual(result["changes"][0]["changes"], ["definition added"])
        self.assertEqual(result["problems"], [])

    def test_old_required_identity_is_not_hidden_by_removing_its_membership(self):
        other = '#[test]\nfn another() {}\n'
        base = self.commit(BODY + other)
        self.commit(BODY + other, inventory(names=("another",)))
        result = self.compare(base)
        rows = {row["test"]: row["changes"] for row in result["changes"]}
        self.assertEqual(rows["parity_example"], ["no longer required"])
        self.assertEqual(rows["another"], ["newly required"])

    def test_new_required_definition_and_membership_are_both_named(self):
        base = self.commit()
        self.commit(BODY + '#[test]\nfn another() {}\n', inventory(names=("parity_example", "another")))
        row = self.compare(base)["changes"][0]
        self.assertEqual(row["test"], "another")
        self.assertEqual(row["changes"], ["newly required", "definition added"])

    def test_uses_merge_base_and_does_not_attribute_sibling_changes(self):
        base = self.commit()
        self.git("checkout", "-b", "sibling")
        sibling = self.commit(BODY.replace("1, 1", "2, 2"))
        self.git("checkout", "fixture")
        candidate = self.commit()
        result = self.compare(sibling, candidate)
        self.assertEqual(result["base"], base)
        self.assertEqual(result["comparison_ref"], sibling)
        self.assertEqual(result["changes"], [])

    def test_regenerated_pr_merge_compares_its_actual_first_parent(self):
        event_base = self.commit()
        self.git("checkout", "-b", "candidate")
        head = self.commit()
        self.git("checkout", "fixture")
        advanced = self.commit(BODY.replace("1, 1", "2, 2"))
        self.git("merge", "--no-ff", "--no-edit", "candidate")
        merge = self.git("rev-parse", "HEAD")
        base, candidate = report.resolve_comparison(self.root, "HEAD", "", head)
        self.assertEqual((base, candidate), (advanced, merge))
        self.assertEqual(self.compare(base, candidate)["changes"], [])
        self.assertTrue(self.compare(event_base, candidate)["changes"])
        for candidate, wrong_head in [(head, head), (merge, event_base), (merge, "bad")]:
            with self.subTest(candidate=candidate, head=wrong_head), self.assertRaises(ValueError):
                report.resolve_comparison(self.root, candidate, "", wrong_head)
        for base_ref, pr_head in [(advanced, head), ("", "")]:
            with self.subTest(base=base_ref, head=pr_head), self.assertRaises(ValueError):
                report.resolve_comparison(self.root, merge, base_ref, pr_head)
        output = self.root / "report.json"
        with mock.patch.object(report, "ROOT", self.root), redirect_stdout(io.StringIO()):
            self.assertEqual(report.main(["--pr-head", head, "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["base"], advanced)

    def test_ignores_uncommitted_working_tree_edits(self):
        base = self.commit()
        (self.root / SOURCE).write_text("broken uncommitted source")
        self.assertEqual(self.compare(base)["changes"], [])

    def test_ambiguous_or_unterminated_definition_fails_closed(self):
        base = self.commit()
        for source in [BODY + BODY, '#[test]\nfn parity_example() {']:
            with self.subTest(source=source):
                self.commit(source)
                with self.assertRaises(ValueError):
                    self.compare(base)

    def test_unprotected_module_local_names_do_not_widen_the_report_scope(self):
        source = BODY + 'mod one { #[test] fn helper() {} }\nmod two { #[test] fn helper() {} }\n'
        base = self.commit(source)
        self.commit(source.replace('mod two { #[test] fn helper() {}', 'mod two { #[test] fn helper() { assert!(true); }'))
        self.assertEqual(self.compare(base)["changes"], [])

    def test_missing_commit_or_manifest_fails_closed(self):
        base = self.commit()
        with self.assertRaises(ValueError):
            self.compare("0" * 40)
        self.git("rm", ORACLE)
        self.git("commit", "-m", "missing oracle")
        with self.assertRaises(ValueError):
            self.compare(base)

    def test_cli_publishes_review_cue_and_clears_stale_output_on_bad_evidence(self):
        base = self.commit()
        self.commit(BODY.replace("1, 1", "2, 2"))
        output = self.root / "report.json"
        with mock.patch.object(report, "ROOT", self.root), redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(report.main(["--base", base, "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["changes"][0]["test"], "parity_example")
        self.assertIn("not a repair", stdout.getvalue())
        with mock.patch.object(report, "ROOT", self.root), redirect_stderr(io.StringIO()):
            self.assertEqual(report.main(["--base", "0" * 40, "--output", str(output)]), 1)
        self.assertFalse(output.exists())

    def test_real_guard_mutations_are_named_beside_the_retained_rejection(self):
        original = oracle.shipped_sources()
        for path, source in original.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        baseline = self.commit(oracle=(oracle.REPO_ROOT / ORACLE).read_text())
        cases = [
            (oracle.PARITY_SOURCE, "parity_tensor_structural_ops", "{}"),
            (oracle.PARITY_SOURCE, "parity_corpus_is_complete", None),
            (oracle.EVAL_AGREEMENT_SOURCE, "agreement_sqrt_is_exact", '{ record_phase3_receipt("sqrt(4)", "forged"); }'),
        ]
        for path, name, replacement in cases:
            with self.subTest(name=name):
                self.git("checkout", "--detach", baseline)
                sources = dict(original)
                if replacement is None:
                    start, end, _, _ = oracle.test_definition_spans(sources[path])[name]
                    sources[path] = sources[path][:start] + sources[path][end:]
                else:
                    sources[path] = oracle.replace_test_body(sources[path], name, replacement)
                retained_failures = oracle.source_violations(sources) + oracle.definition_digest_violations(sources)
                self.assertTrue(any(name in failure for failure in retained_failures))
                (self.root / path).write_text(sources[path])
                self.commit(oracle=(oracle.REPO_ROOT / ORACLE).read_text())
                rows = self.compare(baseline)["changes"]
                self.assertEqual([(r["path"], r["test"]) for r in rows], [(str(path), name)])


class WorkflowTests(unittest.TestCase):
    command = '.venv/bin/python scripts/phase3_test_change_report.py --base "$BASE_REF" --pr-head "$PR_HEAD" --output target/phase3-test-changes.json'

    def workflow(self):
        root = Path(__file__).resolve().parents[1]
        return yaml.safe_load((root / ".github/workflows/ci.yml").read_text())

    def assert_contract(self, workflow):
        job = workflow["jobs"]["docs"]
        steps = job["steps"]
        checkouts = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
        self.assertEqual(checkouts[0]["with"]["fetch-depth"], 0)
        reports = [s for s in steps if "phase3_test_change_report.py" in s.get("run", "")]
        self.assertEqual(len(reports), 1)
        step = reports[0]
        self.assertEqual(step["run"], self.command)
        self.assertEqual(step["env"]["BASE_REF"], "${{ github.event.before }}")
        self.assertEqual(step["env"]["PR_HEAD"], "${{ github.event.pull_request.head.sha }}")
        self.assertNotIn("if", step)
        self.assertNotIn("continue-on-error", step)
        self.assertNotIn("shell", step)
        self.assertNotIn("continue-on-error", job)
        artifacts = [s for s in steps if s.get("with", {}).get("name") == "phase3-test-changes"]
        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0]["uses"].startswith("actions/upload-artifact@"))
        self.assertEqual(artifacts[0]["with"]["path"], "target/phase3-test-changes.json")
        self.assertEqual(artifacts[0]["with"]["if-no-files-found"], "error")

    def test_docs_job_executes_and_publishes_report(self):
        self.assert_contract(self.workflow())

    def test_noop_skipped_or_suppressed_report_is_rejected(self):
        for mutation in ["noop", "skip", "suppress"]:
            workflow = self.workflow()
            candidates = [s for s in workflow["jobs"]["docs"]["steps"] if "phase3_test_change_report.py" in s.get("run", "")]
            if candidates:
                step = candidates[0]
                if mutation == "noop":
                    step["run"] = "true # " + self.command
                elif mutation == "skip":
                    step["if"] = False
                else:
                    step["continue-on-error"] = True
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.assert_contract(workflow)


if __name__ == "__main__":
    unittest.main()
