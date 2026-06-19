"""Unit tests for `drift_report_issue.py`.

Run via: `python3 -m unittest scripts.test_drift_report_issue` from repo
root, or `python3 scripts/test_drift_report_issue.py`.

The script files/updates a chelis-HEAD drift tracking issue in a downstream
shell repo via `gh`. The tests mock `subprocess.run` and assert the BEHAVIOR
(which `gh` subcommands run, with which repo/title) for each
(JOB_STATUS, existing-issue) combination, without touching the network.
"""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "drift_report_issue", here / "drift_report_issue.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dri = _load_module()


def _ok(stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


BASE_ENV = {
    "SHELL_REPO": "nautilus",
    "HEAD_VERSION": "0.7.27",
    "RUN_URL": "https://example/run/1",
    "CHELIS_SHA": "deadbeef",
}


def _gh_subcommands(run_mock) -> list[list[str]]:
    """Extract the gh subcommand (args after the `gh` binary) per call."""
    subs = []
    for call in run_mock.call_args_list:
        argv = call.args[0]
        assert argv[0] == "gh"
        subs.append(argv[1:])
    return subs


class FailureNoExistingIssueTests(unittest.TestCase):
    def test_opens_issue_in_shell_repo(self):
        env = {**BASE_ENV, "JOB_STATUS": "failure"}
        # First call: issue list -> [] (no existing). Then: issue create.
        with mock.patch.object(
            dri.subprocess, "run", side_effect=[_ok("[]"), _ok("")]
        ) as run:
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 0)
        subs = _gh_subcommands(run)
        self.assertEqual(subs[0][:2], ["issue", "list"])
        self.assertIn("Chelis-Lang/nautilus", subs[0])
        self.assertEqual(subs[1][:2], ["issue", "create"])
        # Title must name the shell and target the shell repo.
        self.assertIn("chelis HEAD drift: nautilus", subs[1])
        self.assertIn("Chelis-Lang/nautilus", subs[1])


class FailureCreateLabelFallbackTests(unittest.TestCase):
    def test_falls_back_to_labelless_create_when_label_missing(self):
        env = {**BASE_ENV, "JOB_STATUS": "failure"}
        # list -> []; create --label -> fails; create (no label) -> ok.
        with mock.patch.object(
            dri.subprocess,
            "run",
            side_effect=[
                _ok("[]"),
                subprocess.CompletedProcess([], 1, stdout="", stderr="no label"),
                _ok(""),
            ],
        ) as run:
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 0)
        subs = _gh_subcommands(run)
        # Third call is the label-less create.
        self.assertEqual(subs[2][:2], ["issue", "create"])
        self.assertNotIn("--label", subs[2])


class FailureExistingIssueTests(unittest.TestCase):
    def test_comments_on_existing_instead_of_duplicating(self):
        env = {**BASE_ENV, "JOB_STATUS": "failure"}
        existing = json.dumps([{"number": 7, "title": "chelis HEAD drift: nautilus"}])
        with mock.patch.object(
            dri.subprocess, "run", side_effect=[_ok(existing), _ok("")]
        ) as run:
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 0)
        subs = _gh_subcommands(run)
        self.assertEqual(subs[1][:2], ["issue", "comment"])
        self.assertIn("7", subs[1])
        # Must NOT create a duplicate.
        self.assertNotIn(["issue", "create"], [s[:2] for s in subs])


class SuccessClosesOpenIssueTests(unittest.TestCase):
    def test_comments_then_closes(self):
        env = {**BASE_ENV, "JOB_STATUS": "success"}
        existing = json.dumps([{"number": 7, "title": "chelis HEAD drift: nautilus"}])
        with mock.patch.object(
            dri.subprocess, "run", side_effect=[_ok(existing), _ok(""), _ok("")]
        ) as run:
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 0)
        subs = _gh_subcommands(run)
        self.assertEqual([s[:2] for s in subs], [
            ["issue", "list"],
            ["issue", "comment"],
            ["issue", "close"],
        ])


class SuccessNoIssueIsNoOpTests(unittest.TestCase):
    def test_only_searches_then_stops(self):
        env = {**BASE_ENV, "JOB_STATUS": "success"}
        with mock.patch.object(
            dri.subprocess, "run", side_effect=[_ok("[]")]
        ) as run:
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 0)
        subs = _gh_subcommands(run)
        self.assertEqual([s[:2] for s in subs], [["issue", "list"]])


class CancelledIsTotalNoOpTests(unittest.TestCase):
    def test_no_gh_calls_at_all(self):
        env = {**BASE_ENV, "JOB_STATUS": "cancelled"}
        with mock.patch.object(dri.subprocess, "run") as run:
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 0)
        run.assert_not_called()


class MissingEnvTests(unittest.TestCase):
    def test_missing_env_returns_two(self):
        with mock.patch.dict(dri.os.environ, {"SHELL_REPO": "x"}, clear=True):
            rc = dri.main()
        self.assertEqual(rc, 2)


class GhFailureTests(unittest.TestCase):
    def test_gh_failure_returns_three(self):
        env = {**BASE_ENV, "JOB_STATUS": "failure"}
        with mock.patch.object(
            dri.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
        ):
            with mock.patch.dict(dri.os.environ, env, clear=True):
                rc = dri.main()
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
