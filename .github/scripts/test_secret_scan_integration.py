"""End-to-end suppression and ref-isolation contract for the scanner driver."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import secret_scan


DRIVER = Path(__file__).with_name("secret_scan.py")
FAKE_TOKEN = "AKIA" + hashlib.sha256(b"private secret-scan regression fixture").hexdigest()[:16].upper()


class SecretScanIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("SECRET_SCAN_TEST_BINARY")
        if not configured:
            raise RuntimeError("SECRET_SCAN_TEST_BINARY must point to the checksum-verified Gitleaks binary")
        cls.binary = Path(configured).resolve()
        if not cls.binary.is_file() or not os.access(cls.binary, os.X_OK):
            raise RuntimeError("SECRET_SCAN_TEST_BINARY must point to an executable Gitleaks binary")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="secret-scan-integration-")
        self.root = Path(self.temp.name)
        self.repo = self.root / "candidate"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "user.name", "Private scanner fixture")
        self.trusted_config = self.root / "trusted-gitleaks.toml"
        self.trusted_config.write_text('title = "Trusted integration config"\n[extend]\nuseDefault = true\n')
        self.empty_ignore = self.root / "trusted-empty.gitleaksignore"
        self.empty_ignore.write_text("")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    def commit_all(self, message):
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-m", message],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.git("rev-parse", "HEAD")

    def gitleaks_report(self, report):
        completed = subprocess.run(
            [
                str(self.binary), "git", str(self.repo),
                "--config", str(self.trusted_config),
                "--gitleaks-ignore-path", str(self.empty_ignore),
                "--ignore-gitleaks-allow", "--redact=100", "--no-banner", "--no-color",
                "--report-format", "json", "--report-path", str(report),
                "--log-opts=--all --full-history -m",
            ],
            capture_output=True,
            text=True,
        )
        combined = completed.stdout + completed.stderr
        self.assertFalse(FAKE_TOKEN in combined, "synthetic fixture unexpectedly appeared in captured Gitleaks output")
        self.assertEqual(completed.returncode, 1)
        report_text = report.read_text()
        self.assertFalse(FAKE_TOKEN in report_text, "synthetic fixture unexpectedly appeared in Gitleaks report")
        rows = json.loads(report_text)
        self.assertTrue(rows)
        return rows[0]["Fingerprint"]

    def run_driver(self, event, env_extra=None):
        event_file = self.root / "event.json"
        event_file.write_text(json.dumps(event))
        env = os.environ.copy()
        env.update(env_extra or {})
        env["GITLEAKS_CONFIG"] = str(self.repo / ".gitleaks.toml")
        env["GITLEAKS_CONFIG_TOML"] = (self.repo / ".gitleaks.toml").read_text()
        completed = subprocess.run(
            [
                sys.executable, str(DRIVER),
                "--event-name", "push", "--event-file", str(event_file),
                "--repo", str(self.repo), "--gitleaks", str(self.binary),
                "--config", str(self.trusted_config),
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertFalse(
            FAKE_TOKEN in completed.stdout + completed.stderr,
            "synthetic fixture unexpectedly appeared in scanner-driver output",
        )
        return completed

    def test_historical_detection_survives_source_suppressions_and_ambient_config(self):
        (self.repo / "initial").write_text("initial clean state\n")
        initial = self.commit_all("initial clean state")
        (self.repo / "credential.txt").write_text(f"fixture_key = {FAKE_TOKEN} # gitleaks:allow\n")
        introduced = self.commit_all("synthetic credential introduced")

        # Derive a real Gitleaks fingerprint in private temporary output, with
        # inline suppression disabled, then install that exact source ignore.
        fingerprint = self.gitleaks_report(self.root / "fingerprints.json")
        (self.repo / "credential.txt").unlink()
        self.commit_all("synthetic credential removed")
        (self.repo / ".gitleaksignore").write_text(fingerprint + "\n")
        (self.repo / ".gitleaks.toml").write_text(
            'title = "Untrusted source config"\n'
            "[extend]\nuseDefault = true\n"
            '[[allowlists]]\nregexTarget = "match"\nregexes = ["(?s).*"]\n'
        )
        suppressions = self.commit_all("add candidate-controlled suppressions")
        (self.repo / "maintenance.txt").write_text("harmless cleanup follow-up\n")
        cleanup = self.commit_all("cleanup-only follow-up")
        self.git("update-ref", "refs/remotes/origin/main", initial)

        cleanup_only = self.run_driver({"before": suppressions, "after": cleanup})
        self.assertEqual(cleanup_only.returncode, 0, msg="cleanup-only push range should contain no credential introduction")

        # A zero before SHA models a newly created ref. Detached HEAD ensures
        # the scanner cannot depend on a checked-out branch name.
        self.git("checkout", "--detach", cleanup)
        incremental = self.run_driver({"before": initial, "after": cleanup})
        self.assertEqual(incremental.returncode, 1, msg="incremental history must retain detection despite checkout suppressions")
        new_ref = self.run_driver({"ref": "refs/heads/candidate", "before": "0" * 40, "after": cleanup})
        self.assertEqual(new_ref.returncode, 1, msg="new-ref fallback must find a credential introduced after the main fork")

        prepared = secret_scan.prepare_source(self.repo, self.root / "prepared-bare")
        prepared_repo = prepared
        prepared_head = subprocess.check_output(
            ["git", "--git-dir", str(prepared_repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        self.assertEqual(prepared_head, cleanup)
        original_refs = self.git("show-ref")
        copied_refs = subprocess.check_output(
            ["git", "--git-dir", str(prepared_repo), "show-ref"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        self.assertTrue(set(original_refs.splitlines()).issubset(set(copied_refs.splitlines())))

    def test_new_branch_excludes_main_history_but_detects_candidate_secret(self):
        (self.repo / "initial").write_text("initial clean state\n")
        self.commit_all("initial")
        (self.repo / "old-credential.txt").write_text(f"fixture_key = {FAKE_TOKEN}\n")
        self.commit_all("historical main credential")
        (self.repo / "old-credential.txt").unlink()
        main = self.commit_all("remove historical main credential")
        self.git("update-ref", "refs/remotes/origin/main", main)
        self.git("checkout", "-b", "candidate")
        (self.repo / ".gitleaks.toml").write_text('title = "Untrusted source config"\n')
        (self.repo / "clean.txt").write_text("clean candidate change\n")
        clean = self.commit_all("clean candidate")
        self.gitleaks_report(self.root / "historical-findings.json")

        zero_before = {"ref": "refs/heads/candidate", "before": "0" * 40, "after": clean}
        missing_before = {"ref": "refs/heads/candidate", "before": "a" * 40, "after": clean}
        for event in (zero_before, missing_before):
            with self.subTest(event=event["before"]):
                self.assertEqual(self.run_driver(event).returncode, 0)

        (self.repo / "new-credential.txt").write_text(f"fixture_key = {FAKE_TOKEN}\n")
        leaked = self.commit_all("candidate credential")
        for event in (
            {"ref": "refs/heads/candidate", "before": "0" * 40, "after": leaked},
            {"ref": "refs/heads/candidate", "before": "a" * 40, "after": leaked},
        ):
            with self.subTest(event=event["before"]):
                self.assertEqual(
                    self.run_driver(event).returncode,
                    1,
                    msg="new candidate credentials must still fail the fallback scan",
                )

    def test_new_tag_on_main_scans_its_history(self):
        (self.repo / "initial").write_text("initial clean state\n")
        self.commit_all("initial")
        (self.repo / "credential.txt").write_text(f"fixture_key = {FAKE_TOKEN}\n")
        tagged = self.commit_all("credential in tagged history")
        self.git("update-ref", "refs/remotes/origin/main", tagged)
        (self.repo / ".gitleaks.toml").write_text('title = "Untrusted source config"\n')
        result = self.run_driver({"ref": "refs/tags/release", "before": "0" * 40, "after": tagged})
        self.assertEqual(result.returncode, 1, msg="a first tag push on main must scan the tagged history")



if __name__ == "__main__":
    unittest.main()
