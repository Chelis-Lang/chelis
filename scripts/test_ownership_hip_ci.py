"""CI receipts must preserve the hardware oracle's failure and commit identity."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import ownership_hip_ci as ci


class ReceiptTests(unittest.TestCase):
    def test_exact_clean_commit_is_accepted(self):
        head = "a" * 40
        with patch.object(ci, "git", side_effect=[head, ""]):
            self.assertEqual(ci.verify_head(Path("."), head), head)

    def test_branch_names_and_moving_or_dirty_heads_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            ci.verify_head(Path("."), "main")
        for responses, message in [(["b" * 40], "does not match"),
                                   (["a" * 40, " M tracked.rs"], "dirty")]:
            with self.subTest(message=message), patch.object(ci, "git", side_effect=responses):
                with self.assertRaisesRegex(ValueError, message):
                    ci.verify_head(Path("."), "a" * 40)

    def test_pass_requires_both_zero_exit_and_final_marker(self):
        for output, code, expected in [(ci.PASS_MARKER, 0, 0),
                                       (ci.PASS_MARKER, 7, 7),
                                       ("no selected hardware tests", 0, 1),
                                       (ci.PASS_MARKER + "\nlater failure", 0, 1),
                                       ("BLOCKED: hipcc=None", 1, 1)]:
            with self.subTest(output=output, code=code), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                command = [sys.executable, "-c", f"print({output!r}); raise SystemExit({code})"]
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ci.run_logged(command, directory, directory / "oracle.log")
                self.assertEqual(result, (expected, code))
                self.assertEqual((directory / "oracle.log").read_text(), output + "\n")

    def test_failed_oracle_still_writes_identifiable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(ci, "verify_head", return_value="a" * 40), \
                 patch.object(ci, "run_logged", return_value=(7, 7)), \
                 patch.object(ci, "ROOT", root):
                self.assertEqual(ci.main(["--expected-head", "a" * 40]), 7)
            receipt = json.loads((root / "target/ownership-hip-ci/receipt.json").read_text())
            self.assertEqual(receipt["head"], "a" * 40)
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["oracle_exit_code"], 7)
            self.assertEqual(receipt["command"][-3:], ["--phase", "3", "--require-hip"])

    def test_spawn_error_is_a_failed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(ci, "verify_head", return_value="a" * 40), \
                 patch.object(ci, "run_logged", side_effect=OSError("cannot execute")), \
                 patch.object(ci, "ROOT", root), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ci.main(["--expected-head", "a" * 40]), 1)
            receipt = json.loads((root / "target/ownership-hip-ci/receipt.json").read_text())
            self.assertFalse(receipt["passed"])
            self.assertIn("cannot execute", receipt["error"])


class WorkflowTests(unittest.TestCase):
    def workflow(self):
        # This security-sensitive .yml uses JSON's unambiguous YAML subset.
        # Reject other spellings, aliases, and duplicate keys instead of trying
        # to approximate YAML event semantics with substring checks.
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate workflow key: {key}")
                result[key] = value
            return result

        source = (ci.ROOT / ".github/workflows/ownership-hip.yml").read_text()
        return json.loads(source, object_pairs_hook=unique_object)

    def test_manual_dispatch_keeps_untrusted_pr_events_off_the_gpu_host(self):
        workflow = self.workflow()
        self.assertIsInstance(workflow["on"], dict)
        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertEqual(set(workflow["jobs"]), {"phase3"})
        job = workflow["jobs"]["phase3"]
        self.assertEqual(job["runs-on"], ["self-hosted", "linux", "x64", "chelis-hip-gfx1151"])
        self.assertEqual(job["steps"][0], {
            "uses": "actions/checkout@v6",
            "with": {"ref": "${{ inputs.commit }}", "persist-credentials": False},
        })

    def test_event_policy_rejects_every_automatic_event_and_alternate_shape(self):
        original = self.workflow()
        for event in ["pull_request", "pull_request_target", "push", "workflow_run",
                      "schedule", "repository_dispatch", "issue_comment"]:
            mutated = json.loads(json.dumps(original))
            mutated["on"][event] = {}
            with self.subTest(event=event), patch.object(Path, "read_text", return_value=json.dumps(mutated)):
                with self.assertRaises(AssertionError):
                    self.test_manual_dispatch_keeps_untrusted_pr_events_off_the_gpu_host()
        for events in ["workflow_dispatch", ["workflow_dispatch"], {}]:
            mutated = dict(original, on=events)
            with self.subTest(events=events), patch.object(Path, "read_text", return_value=json.dumps(mutated)):
                with self.assertRaises(AssertionError):
                    self.test_manual_dispatch_keeps_untrusted_pr_events_off_the_gpu_host()

    def test_quoted_escaped_duplicate_and_yaml_only_event_forms_fail_closed(self):
        original = json.dumps(self.workflow())
        sources = [
            original.replace('"workflow_dispatch":', '"pull_request": {}, "workflow_dispatch":', 1),
            original.replace('"workflow_dispatch":', '"pull_\\u0072equest": {}, "workflow_dispatch":', 1),
            original.replace('"workflow_dispatch":', '"workflow_dispatch": {}, "workflow_dispatch":', 1),
            original.replace('"on":', '"on": {"pull_request": {}}, "on":', 1),
            "on:\n  workflow_dispatch:\n  'pull_request':\n",
            'on: {workflow_dispatch: {}, "pull_request": {}}\n',
            "events: &events {pull_request: {}}\non: *events\n",
        ]
        for source in sources:
            with self.subTest(source=source), patch.object(Path, "read_text", return_value=source):
                with self.assertRaises((AssertionError, ValueError)):
                    self.test_manual_dispatch_keeps_untrusted_pr_events_off_the_gpu_host()

    def test_oracle_failure_cannot_be_hidden_by_the_artifact_step(self):
        workflow = self.workflow()
        job = workflow["jobs"]["phase3"]
        self.assertEqual(job["env"]["OWNERSHIP_HIP_COMMIT"], "${{ inputs.commit }}")
        self.assertNotIn("continue-on-error", job)
        for step in job["steps"]:
            self.assertNotIn("continue-on-error", step)
        oracle = [step for step in job["steps"] if step.get("name") == "Execute the authoritative hardware oracle"]
        self.assertEqual(oracle, [{"name": "Execute the authoritative hardware oracle",
                                   "run": ".venv/bin/python scripts/ownership_hip_ci.py"}])
        artifact = job["steps"][-1]
        self.assertEqual(artifact["if"], "always()")
        self.assertEqual(artifact["uses"], "actions/upload-artifact@v7")
        self.assertEqual(artifact["with"]["path"], "target/ownership-hip-ci/")
        self.assertEqual(artifact["with"]["if-no-files-found"], "error")


if __name__ == "__main__":
    unittest.main()
