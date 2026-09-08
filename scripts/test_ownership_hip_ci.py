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
    def test_manual_dispatch_keeps_untrusted_pr_events_off_the_gpu_host(self):
        source = (ci.ROOT / ".github/workflows/ownership-hip.yml").read_text()
        self.assertIn("  workflow_dispatch:", source)
        self.assertNotIn("pull_request:", source)
        self.assertNotIn("pull_request_target:", source)
        self.assertNotIn("  push:", source)
        self.assertIn("  contents: read", source)
        self.assertIn("persist-credentials: false", source)
        self.assertIn("runs-on: [self-hosted, linux, x64, chelis-hip-gfx1151]", source)

    def test_oracle_failure_cannot_be_hidden_by_the_artifact_step(self):
        source = (ci.ROOT / ".github/workflows/ownership-hip.yml").read_text()
        self.assertIn("ref: ${{ inputs.commit }}", source)
        self.assertIn("OWNERSHIP_HIP_COMMIT: ${{ inputs.commit }}", source)
        self.assertIn("run: .venv/bin/python scripts/ownership_hip_ci.py\n", source)
        self.assertNotIn("continue-on-error", source)
        self.assertNotIn("|| true", source)
        self.assertIn("if: always()", source)
        self.assertIn("path: target/ownership-hip-ci/", source)
        self.assertIn("if-no-files-found: error", source)


if __name__ == "__main__":
    unittest.main()
