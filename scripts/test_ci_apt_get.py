"""Unit tests for `ci_apt_get.py`.

Run via: `python3 -m unittest scripts.test_ci_apt_get` from repo root,
or `python3 scripts/test_ci_apt_get.py`.

The helper runs `apt-get` as a subprocess, so the tests mock
`subprocess.run` and assert the *behavior* (command shape, the retry loop,
that a transient flake self-heals, and that a genuinely-broken install still
fails after the attempts are exhausted) without touching the real package
manager. `time.sleep` is injected so the retry backoff does not actually
wait.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ci_apt_get", here / "ci_apt_get.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cag = _load_module()


def _completed(returncode=0):
    return mock.Mock(returncode=returncode, stdout="", stderr="")


class CommandShapeTests(unittest.TestCase):
    def test_first_success_runs_update_then_install_with_sudo(self):
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            rc = cag.apt_get(["gcc", "libopenblas-dev"], sleep=lambda _s: None)
        self.assertEqual(rc, 0)
        cmds = [call.args[0] for call in run.call_args_list]
        self.assertEqual(cmds[0], ["sudo", "apt-get", "update"])
        self.assertEqual(
            cmds[1], ["sudo", "apt-get", "install", "-y", "gcc", "libopenblas-dev"]
        )
        # Exactly the update+install pair on a clean first attempt.
        self.assertEqual(len(cmds), 2)

    def test_no_sudo_drops_the_sudo_prefix(self):
        # The debian:11 container job runs as root and has no `sudo`.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(["m4"], sudo=False, sleep=lambda _s: None)
        cmds = [call.args[0] for call in run.call_args_list]
        self.assertEqual(cmds[0], ["apt-get", "update"])
        self.assertEqual(cmds[1], ["apt-get", "install", "-y", "m4"])

    def test_no_install_recommends_is_forwarded_to_install_only(self):
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(
                ["cmake", "git"],
                sudo=False,
                no_install_recommends=True,
                sleep=lambda _s: None,
            )
        cmds = [call.args[0] for call in run.call_args_list]
        # update never carries --no-install-recommends; install does, before
        # the package names.
        self.assertEqual(cmds[0], ["apt-get", "update"])
        self.assertEqual(
            cmds[1],
            ["apt-get", "install", "-y", "--no-install-recommends", "cmake", "git"],
        )

    def test_empty_package_list_is_a_noop(self):
        with mock.patch.object(cag.subprocess, "run") as run:
            self.assertEqual(cag.apt_get([], sleep=lambda _s: None), 0)
        run.assert_not_called()


class RetryTests(unittest.TestCase):
    def test_transient_update_failure_then_success_retries_and_passes(self):
        # First attempt: `apt-get update` resets (rc 100), so install is not
        # even reached; second attempt: both succeed. End result is success.
        results = [
            _completed(returncode=100),  # attempt 1 update: reset
            _completed(returncode=0),  # attempt 2 update
            _completed(returncode=0),  # attempt 2 install
        ]
        sleeps: list[float] = []
        with mock.patch.object(cag.subprocess, "run", side_effect=results) as run:
            rc = cag.apt_get(["gcc"], sleep=sleeps.append)
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 3)
        # Exactly one backoff between the two attempts.
        self.assertEqual(sleeps, [cag.BACKOFF_SECONDS])

    def test_transient_install_failure_then_success(self):
        # update ok but install resets on attempt 1; both ok on attempt 2.
        results = [
            _completed(returncode=0),  # attempt 1 update
            _completed(returncode=100),  # attempt 1 install: reset
            _completed(returncode=0),  # attempt 2 update
            _completed(returncode=0),  # attempt 2 install
        ]
        with mock.patch.object(cag.subprocess, "run", side_effect=results) as run:
            rc = cag.apt_get(["gcc"], sleep=lambda _s: None)
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 4)

    def test_persistent_failure_exhausts_attempts_and_returns_nonzero(self):
        # A genuinely-broken install (e.g. unknown package) fails every
        # attempt. The retries must NOT mask it: the helper returns the last
        # non-zero exit so the job still goes red. This is the "cannot turn
        # a real packaging error green" invariant.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=100)
        ) as run:
            rc = cag.apt_get(["nope-not-a-package"], attempts=3, sleep=lambda _s: None)
        self.assertEqual(rc, 100)
        # 3 attempts; each stops at the failing `update` (rc!=0 short-circuits
        # install), so 3 subprocess calls, not 6.
        self.assertEqual(run.call_count, 3)

    def test_no_sleep_after_the_final_attempt(self):
        # Backoff happens BETWEEN attempts only, never after the last one
        # (no point sleeping when we are about to give up).
        sleeps: list[float] = []
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=1)
        ):
            cag.apt_get(["gcc"], attempts=3, sleep=sleeps.append)
        self.assertEqual(len(sleeps), 2)  # between 1->2 and 2->3, not after 3

    def test_single_attempt_does_not_sleep(self):
        sleeps: list[float] = []
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=1)
        ):
            rc = cag.apt_get(["gcc"], attempts=1, sleep=sleeps.append)
        self.assertEqual(rc, 1)
        self.assertEqual(sleeps, [])

    def test_spawn_failure_is_treated_as_nonzero_and_retried(self):
        # If the OS cannot spawn apt-get at all (OSError), _run returns 1 and
        # the loop retries it like any other transient failure.
        with mock.patch.object(
            cag.subprocess, "run", side_effect=OSError("boom")
        ) as run:
            rc = cag.apt_get(["gcc"], attempts=2, sleep=lambda _s: None)
        self.assertEqual(rc, 1)
        # 2 attempts, each failing to spawn `update` (install never reached).
        self.assertEqual(run.call_count, 2)


class CliTests(unittest.TestCase):
    def test_main_parses_packages_and_flags(self):
        captured = {}

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts):
            captured.update(
                packages=packages,
                sudo=sudo,
                no_install_recommends=no_install_recommends,
                attempts=attempts,
            )
            return 0

        with mock.patch.object(cag, "apt_get", fake_apt_get):
            rc = cag.main(
                ["--no-sudo", "--no-install-recommends", "cmake", "libclang-dev"]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["packages"], ["cmake", "libclang-dev"])
        self.assertFalse(captured["sudo"])
        self.assertTrue(captured["no_install_recommends"])

    def test_main_defaults_to_sudo_and_recommends(self):
        captured = {}

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts):
            captured.update(sudo=sudo, no_install_recommends=no_install_recommends)
            return 0

        with mock.patch.object(cag, "apt_get", fake_apt_get):
            cag.main(["gcc"])
        self.assertTrue(captured["sudo"])
        self.assertFalse(captured["no_install_recommends"])


if __name__ == "__main__":
    unittest.main()
