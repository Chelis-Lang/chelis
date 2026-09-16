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

    def test_empty_body_is_reported_and_behavior_control_remains_separate(self):
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

    def test_real_guard_mutations_require_exact_review_acknowledgements(self):
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
                (self.root / path).write_text(sources[path])
                self.commit(oracle=(oracle.REPO_ROOT / ORACLE).read_text())
                result = self.compare(baseline)
                rows = result["changes"]
                self.assertEqual([(r["path"], r["test"]) for r in rows], [(str(path), name)])
                line = f"Protected-test-change: {path}::{name}"
                self.assertEqual(result["required_acknowledgements"], [line])
                self.assertTrue(report.acknowledgement_violations(result, ""))
                self.assertEqual(report.acknowledgement_violations(result, line), [])
                # The Rust body audit independently rejects all three named
                # mutations; acknowledgement is a review cue, not that oracle.
                if replacement is None:
                    self.assertTrue(any(name in failure for failure in oracle.source_violations(sources)))


class AcknowledgementTests(unittest.TestCase):
    setUp = CommittedChanges.setUp
    git = CommittedChanges.git
    commit = CommittedChanges.commit
    compare = CommittedChanges.compare

    def changed(self):
        base = self.commit()
        self.commit(BODY.replace("1, 1", "2, 2"))
        return base, self.compare(base)

    def test_report_names_the_exact_line_and_requires_it(self):
        _, result = self.changed()
        line = f"Protected-test-change: {SOURCE}::parity_example"
        self.assertEqual(result["required_acknowledgements"], [line])
        self.assertEqual(report.acknowledgement_violations(result, line), [])
        self.assertTrue(report.acknowledgement_violations(result, ""))
        self.assertTrue(report.acknowledgement_violations(result, line + "\n" + line))

    def test_stale_unknown_malformed_quoted_or_hidden_lines_do_not_count(self):
        _, result = self.changed()
        line = f"Protected-test-change: {SOURCE}::parity_example"
        for body in [
            line.replace("parity_example", "unknown"),
            line.lower(), " " + line, "- " + line, "> " + line,
            line.replace(": ", ":  "), line + " trailing explanation",
            "```text\n" + line + "\n```", "~~~\n" + line + "\n~~~",
            "````\n```\n" + line + "\n````",
            "```\n~~~\n" + line + "\n```",
            "```\n```still code\n" + line + "\n```",
            "> Example only:\n" + line,
            "<details>\n\n" + line + "\n\n</details>",
            "`Example only\n" + line + "\n`",
            "Prose before the acknowledgement.\n\n" + line,
            "<!--\n" + line + "\n-->",
            "<!-- closed --><!--\n" + line + "\n-->",
            "<!--\n--><!--\n" + line + "\n-->",
            "<pre>\n</pre><pre>\n" + line + "\n</pre>",
            "<?xml\n\n" + line + "\n?>",
            "<![CDATA[\n\n" + line + "\n]]>",
            "<!DOCTYPE\n\n" + line + "\n>",
            "<details>\n" + line + "\n</details>",
            "<pre>\n\n" + line + "\n</pre>",
        ]:
            with self.subTest(body=body):
                self.assertTrue(report.acknowledgement_violations(result, body))
        unchanged = {**result, "required_acknowledgements": []}
        self.assertTrue(report.acknowledgement_violations(unchanged, line))
        self.assertTrue(report.acknowledgement_violations(result, line + "\nProtected_test_change: invalid"))

    def test_opening_block_is_the_only_acknowledgement_authority(self):
        _, result = self.changed()
        line = result["required_acknowledgements"][0]
        for example in [
            "Review evidence follows.",
            "```example\n" + line + "\n```",
            "<!-- example -->",
            "<details>\n\n" + line + "\n\n</details>",
            "> Example only:\n" + line,
            "`Example\n" + line + "\n`",
            line.replace("parity_example", "not_an_acknowledgement_here"),
        ]:
            with self.subTest(example=example):
                self.assertEqual(report.acknowledgement_violations(result, line + "\n\n" + example), [])
                self.assertTrue(report.acknowledgement_violations(result, example + "\n\n" + line))
        self.assertEqual(report.acknowledgement_violations(result, "\n  \n" + line), [])
        self.assertTrue(report.acknowledgement_violations(result, line + "\nProse needs a separating blank line."))

    def run_cli(self, base, body, *, enforce=True, env=False):
        output = self.root / "ack-report.json"
        args = ["--base", base, "--output", str(output)]
        if enforce:
            args += ["--require-acknowledgement"]
        if env:
            args += ["--acknowledgements-env", "TEST_PR_BODY"]
            context = mock.patch.dict("os.environ", {"TEST_PR_BODY": body})
        else:
            path = self.root / "body.txt"
            path.write_text(body)
            args += ["--acknowledgements-file", str(path)]
            context = mock.patch.dict("os.environ", {})
        with context, mock.patch.object(report, "ROOT", self.root), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = report.main(args)
        return status, json.loads(output.read_text()) if output.exists() else None

    def test_cli_enforces_file_and_environment_and_preserves_failure_artifact(self):
        base, result = self.changed()
        line = result["required_acknowledgements"][0]
        for env in (False, True):
            with self.subTest(env=env):
                self.assertEqual(self.run_cli(base, line, env=env)[0], 0)
                status, failed = self.run_cli(base, "", env=env)
                self.assertEqual(status, 1)
                self.assertEqual(failed["required_acknowledgements"], [line])
                self.assertTrue(failed["acknowledgement_problems"])
        self.assertEqual(self.run_cli(base, "", enforce=False)[0], 0)

    def test_removed_requirement_still_requires_its_historical_identity(self):
        other = '#[test]\nfn another() {}\n'
        base = self.commit(BODY + other)
        self.commit(other, inventory(names=("another",)))
        result = self.compare(base)
        lines = result["required_acknowledgements"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(report.acknowledgement_violations(result, "\n".join(lines)), [])
        for omitted in lines:
            self.assertTrue(report.acknowledgement_violations(result, "\n".join(x for x in lines if x != omitted)))

    def test_acknowledgement_cannot_admit_a_missing_definition(self):
        base = self.commit()
        self.commit("// missing\n")
        line = f"Protected-test-change: {SOURCE}::parity_example"
        status, result = self.run_cli(base, line)
        self.assertEqual(status, 1)
        self.assertTrue(result["problems"])
        self.assertEqual(result["acknowledgement_problems"], [])

    def test_missing_body_source_and_invalid_comparison_fail_closed(self):
        base, _ = self.changed()
        output = self.root / "missing.json"
        for extra in [[], ["--acknowledgements-file", str(self.root / "absent")], ["--acknowledgements-env", "MISSING_PR_BODY"]]:
            with self.subTest(extra=extra), mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(report, "ROOT", self.root), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(report.main(["--base", base, "--output", str(output), "--require-acknowledgement", *extra]), 1)
        self.assertEqual(self.run_cli("0" * 40, "")[0], 1)


class WorkflowTests(unittest.TestCase):
    command = '.venv/bin/python scripts/phase3_test_change_report.py --base "$BASE_REF" --pr-head "$PR_HEAD" --output target/phase3-test-changes.json'

    def workflow(self):
        root = Path(__file__).resolve().parents[1]
        return yaml.safe_load((root / ".github/workflows/ci.yml").read_text())

    def acknowledgement_workflow(self):
        root = Path(__file__).resolve().parents[1]
        return yaml.safe_load(
            (root / ".github/workflows/pr-contract-acknowledgements.yml").read_text()
        )

    def assert_contract(self, workflow):
        events = workflow.get("on", workflow.get(True))
        self.assertEqual(
            set(events["pull_request"]["types"]),
            {"opened", "synchronize", "reopened"},
        )
        job = workflow["jobs"]["docs"]
        steps = job["steps"]
        checkouts = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
        self.assertEqual(checkouts[0]["with"]["fetch-depth"], 0)
        reports = [s for s in steps if "phase3_test_change_report.py" in s.get("run", "") and "--require-acknowledgement" not in s.get("run", "")]
        self.assertEqual(len(reports), 1)
        step = reports[0]
        self.assertEqual(step["run"], self.command)
        self.assertEqual(step["env"]["BASE_REF"], "${{ github.event_name == 'push' && github.event.before || '' }}")
        self.assertEqual(
            step["env"]["PR_HEAD"],
            "${{ inputs.expected_head_sha || github.event.pull_request.head.sha }}",
        )
        self.assertNotIn("if", step)
        self.assertNotIn("continue-on-error", step)
        self.assertNotIn("shell", step)
        self.assertNotIn("continue-on-error", job)
        artifact = next(
            s
            for s in steps
            if s.get("with", {}).get("name") == "phase3-test-changes"
        )
        self.assertEqual(
            artifact["if"],
            "${{ always() && hashFiles('target/phase3-test-changes.json') != '' }}",
        )
        self.assertEqual(
            artifact["with"]["if-no-files-found"],
            "error",
        )
        enforcing = [
            s
            for s in steps
            if "phase3_test_change_report.py" in s.get("run", "")
            and "--require-acknowledgement" in s.get("run", "")
        ]
        self.assertEqual(enforcing, [])
        ack = self.acknowledgement_workflow()
        ack_events = ack.get("on", ack.get(True))
        self.assertEqual(
            set(ack_events["pull_request"]["types"]),
            {"opened", "synchronize", "reopened", "edited"},
        )
        enforce = next(
            s
            for s in ack["jobs"]["acknowledgements"]["steps"]
            if "phase3_test_change_report.py" in s.get("run", "")
        )
        self.assertEqual(enforce["run"], '.venv/bin/python scripts/phase3_test_change_report.py --pr-head "$PR_HEAD" --require-acknowledgement --acknowledgements-env PR_BODY --output target/phase3-test-changes.json')
        self.assertEqual(enforce["env"], {"PR_HEAD": "${{ github.event.pull_request.head.sha }}", "PR_BODY": "${{ github.event.pull_request.body }}"})
        self.assertNotIn("continue-on-error", enforce)
        self.assertNotIn("shell", enforce)
        ack_steps = ack["jobs"]["acknowledgements"]["steps"]
        self.assertLess(
            ack_steps.index(enforce),
            ack_steps.index(
                next(
                    s
                    for s in ack_steps
                    if s.get("with", {}).get("name") == "phase3-test-changes"
                )
            ),
        )
        artifacts = [s for s in ack_steps if s.get("with", {}).get("name") == "phase3-test-changes"]
        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0]["uses"].startswith("actions/upload-artifact@"))
        self.assertEqual(
            artifacts[0]["if"],
            "always() && hashFiles('target/phase3-test-changes.json') != ''",
        )
        self.assertEqual(artifacts[0]["with"]["path"], "target/phase3-test-changes.json")
        self.assertEqual(artifacts[0]["with"]["if-no-files-found"], "error")

    def test_missing_skipped_suppressed_or_untrusted_enforcement_fails(self):
        for mutation in ("remove", "suppress", "body", "head", "interpolate", "edited"):
            workflow = self.acknowledgement_workflow()
            steps = workflow["jobs"]["acknowledgements"]["steps"]
            step = next(
                (
                    s
                    for s in steps
                    if "phase3_test_change_report.py" in s.get("run", "")
                ),
                None,
            )
            if step is not None:
                if mutation == "edited":
                    workflow.get("on", workflow.get(True))["pull_request"] = None
                elif mutation == "remove": steps.remove(step)
                elif mutation == "suppress": step["continue-on-error"] = True
                elif mutation == "body": step["env"].pop("PR_BODY")
                elif mutation == "head": step["env"]["PR_HEAD"] = "${{ github.event.before }}"
                else: step["run"] += " ${{ github.event.pull_request.body }}"
            with self.subTest(mutation=mutation), self.assertRaises(
                (AssertionError, KeyError, StopIteration, TypeError)
            ):
                original = self.acknowledgement_workflow
                self.acknowledgement_workflow = lambda: workflow
                try:
                    self.assert_contract(self.workflow())
                finally:
                    self.acknowledgement_workflow = original

    def test_docs_job_executes_and_publishes_report(self):
        self.assert_contract(self.workflow())

    def test_synchronize_before_is_not_allowed_to_select_push_comparison(self):
        # pull_request.synchronize includes `before`, the previous PR head.
        # Presence alone cannot select the push mode; actual hosted synchronize
        # execution exercises this event expression with both payload fields.
        workflow = self.workflow()
        step = next(s for s in workflow["jobs"]["docs"]["steps"]
                    if "phase3_test_change_report.py" in s.get("run", ""))
        step["env"]["BASE_REF"] = "${{ github.event.before }}"
        with self.assertRaises(AssertionError):
            self.assert_contract(workflow)

    def test_noop_skipped_or_suppressed_report_is_rejected(self):
        for mutation in ["noop", "skip", "suppress"]:
            workflow = self.workflow()
            candidates = [s for s in workflow["jobs"]["docs"]["steps"] if "phase3_test_change_report.py" in s.get("run", "") and "--require-acknowledgement" not in s.get("run", "")]
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
