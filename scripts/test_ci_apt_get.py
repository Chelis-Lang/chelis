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
import tempfile
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


def _wrap(timeout=cag.PER_COMMAND_TIMEOUT_SECONDS):
    """The coreutils `timeout` prefix the helper inserts before apt-get."""
    return ["timeout", f"--kill-after={cag.KILL_AFTER_SECONDS:g}s", f"{timeout:g}s"]


class CommandShapeTests(unittest.TestCase):
    def test_first_success_runs_update_then_install_with_sudo(self):
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            rc = cag.apt_get(["gcc", "libopenblas-dev"], sleep=lambda _s: None)
        self.assertEqual(rc, 0)
        cmds = [call.args[0] for call in run.call_args_list]
        # `timeout` sits AFTER `sudo` so it is apt-get's direct parent.
        self.assertEqual(cmds[0], ["sudo", *_wrap(), "apt-get", "update"])
        self.assertEqual(
            cmds[1],
            ["sudo", *_wrap(), "apt-get", "install", "-y", "gcc", "libopenblas-dev"],
        )
        # Exactly the update+install pair on a clean first attempt.
        self.assertEqual(len(cmds), 2)

    def test_no_sudo_drops_the_sudo_prefix_but_keeps_the_timeout(self):
        # The debian:11 container job runs as root and has no `sudo`, but the
        # timeout wrapper must still bound the command.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(["m4"], sudo=False, sleep=lambda _s: None)
        cmds = [call.args[0] for call in run.call_args_list]
        self.assertEqual(cmds[0], [*_wrap(), "apt-get", "update"])
        self.assertEqual(cmds[1], [*_wrap(), "apt-get", "install", "-y", "m4"])

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
        self.assertEqual(cmds[0], [*_wrap(), "apt-get", "update"])
        self.assertEqual(
            cmds[1],
            [*_wrap(), "apt-get", "install", "-y", "--no-install-recommends",
             "cmake", "git"],
        )

    def test_disabled_timeout_omits_the_wrapper(self):
        # `timeout=None` (from `--timeout 0`) runs apt-get unbounded: no
        # `timeout` token at all.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(["gcc"], timeout=None, sleep=lambda _s: None)
        cmds = [call.args[0] for call in run.call_args_list]
        self.assertEqual(cmds[0], ["sudo", "apt-get", "update"])
        self.assertEqual(cmds[1], ["sudo", "apt-get", "install", "-y", "gcc"])
        for cmd in cmds:
            self.assertNotIn("timeout", cmd)

    def test_empty_package_list_is_a_noop(self):
        with mock.patch.object(cag.subprocess, "run") as run:
            self.assertEqual(cag.apt_get([], sleep=lambda _s: None), 0)
        run.assert_not_called()


class BullseyeSnapshotTests(unittest.TestCase):
    def test_snapshot_sources_are_immutable_and_complete(self):
        expected = (
            "deb [check-valid-until=no] "
            "https://snapshot.debian.org/archive/debian/20260901T000000Z/ "
            "bullseye main\n"
            "deb [check-valid-until=no] "
            "https://snapshot.debian.org/archive/debian/20260901T000000Z/ "
            "bullseye-updates main\n"
            "deb [check-valid-until=no] "
            "https://snapshot.debian.org/archive/debian-security/20260901T000000Z/ "
            "bullseye-security main\n"
        )
        self.assertEqual(cag.bullseye_snapshot_sources(), expected)

    def test_configure_snapshot_replaces_every_moving_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            apt_root = Path(tmp)
            fragments = apt_root / "sources.list.d"
            fragments.mkdir()
            (apt_root / "sources.list").write_text(
                "deb http://deb.debian.org/debian bullseye main\n",
                encoding="utf-8",
            )
            (fragments / "debian.list").write_text(
                "deb http://security.debian.org bullseye-security main\n",
                encoding="utf-8",
            )
            (fragments / "debian.sources").write_text(
                "URIs: http://deb.debian.org/debian\n",
                encoding="utf-8",
            )
            (fragments / "README").write_text("keep me\n", encoding="utf-8")

            cag.configure_bullseye_snapshot(apt_root)

            self.assertEqual(
                (apt_root / "sources.list").read_text(encoding="utf-8"),
                cag.bullseye_snapshot_sources(),
            )
            self.assertFalse((fragments / "debian.list").exists())
            self.assertFalse((fragments / "debian.sources").exists())
            self.assertEqual(
                (fragments / "README").read_text(encoding="utf-8"), "keep me\n"
            )

    def test_configure_snapshot_requires_an_apt_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "missing-apt"
            with self.assertRaises(FileNotFoundError):
                cag.configure_bullseye_snapshot(absent)
            self.assertFalse(absent.exists())


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


class TimeoutTests(unittest.TestCase):
    # The ceiling is enforced by the coreutils `timeout` binary baked into the
    # command (NOT subprocess.run(timeout=)), because SIGKILLing our direct
    # child `sudo` would orphan the apt-get it spawned and leave the apt lock
    # held -- exactly the failure these tests pin down. So a "timed out"
    # attempt manifests as `timeout` exiting non-zero (124/137), which the mock
    # returns as an ordinary returncode.

    def test_custom_timeout_is_baked_into_the_command(self):
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(["gcc"], timeout=42, sleep=lambda _s: None)
        self.assertEqual(run.call_count, 2)  # update + install
        for call in run.call_args_list:
            cmd = call.args[0]
            # `timeout` immediately follows `sudo`, carrying the ceiling and
            # the --kill-after escalation.
            self.assertEqual(cmd[0], "sudo")
            self.assertEqual(cmd[1], "timeout")
            self.assertEqual(cmd[2], f"--kill-after={cag.KILL_AFTER_SECONDS:g}s")
            self.assertEqual(cmd[3], "42s")
            # subprocess.run is never given a Python-level timeout kwarg.
            self.assertNotIn("timeout", call.kwargs)

    def test_default_timeout_is_the_module_constant(self):
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(["gcc"], sleep=lambda _s: None)
        for call in run.call_args_list:
            self.assertIn(
                f"{cag.PER_COMMAND_TIMEOUT_SECONDS:g}s", call.args[0]
            )

    def test_shipped_default_wrapper_tokens_are_pinned_literally(self):
        # Everything else derives the wrapper tokens from the constants and so
        # would move silently with them. Pin the SHIPPED values literally here
        # so an accidental change to PER_COMMAND_TIMEOUT_SECONDS (300) or
        # KILL_AFTER_SECONDS (30) trips this test instead of shipping quietly.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed()
        ) as run:
            cag.apt_get(["gcc"], sleep=lambda _s: None)
        update_cmd = run.call_args_list[0].args[0]
        self.assertEqual(
            update_cmd[:5],
            ["sudo", "timeout", "--kill-after=30s", "300s", "apt-get"],
        )

    def test_timed_out_command_is_retried_and_self_heals(self):
        # attempt 1 `update` is killed by the wrapper (coreutils `timeout`
        # returns 124); the loop must treat it as a failed attempt and retry,
        # NOT poison the retry the way an orphaned apt-get would. attempt 2:
        # both succeed.
        results = [
            _completed(returncode=124),  # attempt 1 update: timed out + killed
            _completed(returncode=0),  # attempt 2 update
            _completed(returncode=0),  # attempt 2 install
        ]
        sleeps: list[float] = []
        with mock.patch.object(cag.subprocess, "run", side_effect=results) as run:
            rc = cag.apt_get(["gcc"], sleep=sleeps.append)
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleeps, [cag.BACKOFF_SECONDS])

    def test_persistent_timeout_exhausts_attempts_and_returns_the_code(self):
        # A mirror that stalls on every attempt must not wedge forever: each
        # attempt is killed by the wrapper (124), and the helper gives up with
        # that code so the job goes red instead of burning the 6-hour cap.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=124)
        ) as run:
            rc = cag.apt_get(["gcc"], attempts=3, sleep=lambda _s: None)
        self.assertEqual(rc, 124)
        self.assertIn(rc, cag.TIMEOUT_EXIT_CODES)
        self.assertEqual(run.call_count, 3)

    def test_kill_after_exit_code_is_recognised_as_a_timeout(self):
        # 137 (128+9) is what `timeout` returns when apt-get ignored SIGTERM
        # and had to be SIGKILLed at --kill-after; it is still a timeout kill.
        self.assertIn(137, cag.TIMEOUT_EXIT_CODES)
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=137)
        ):
            rc = cag.apt_get(["gcc"], attempts=2, sleep=lambda _s: None)
        self.assertEqual(rc, 137)


class CliTests(unittest.TestCase):
    def test_main_parses_packages_and_flags(self):
        captured = {}

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts, timeout):
            captured.update(
                packages=packages,
                sudo=sudo,
                no_install_recommends=no_install_recommends,
                attempts=attempts,
                timeout=timeout,
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
        # Default timeout is the module constant when --timeout is not passed.
        self.assertEqual(captured["timeout"], cag.PER_COMMAND_TIMEOUT_SECONDS)

    def test_main_defaults_to_sudo_and_recommends(self):
        captured = {}

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts, timeout):
            captured.update(sudo=sudo, no_install_recommends=no_install_recommends)
            return 0

        with mock.patch.object(cag, "apt_get", fake_apt_get):
            cag.main(["gcc"])
        self.assertTrue(captured["sudo"])
        self.assertFalse(captured["no_install_recommends"])

    def test_main_forwards_custom_timeout(self):
        captured = {}

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts, timeout):
            captured.update(timeout=timeout)
            return 0

        with mock.patch.object(cag, "apt_get", fake_apt_get):
            cag.main(["--timeout", "90", "gcc"])
        self.assertEqual(captured["timeout"], 90)

    def test_main_zero_timeout_disables_the_ceiling(self):
        # `--timeout 0` maps to None so subprocess.run applies no ceiling.
        captured = {}

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts, timeout):
            captured.update(timeout=timeout)
            return 0

        with mock.patch.object(cag, "apt_get", fake_apt_get):
            cag.main(["--timeout", "0", "gcc"])
        self.assertIsNone(captured["timeout"])

    def test_main_configures_snapshot_before_apt(self):
        events = []

        def fake_configure():
            events.append("snapshot")

        def fake_apt_get(packages, *, sudo, no_install_recommends, attempts, timeout):
            events.append("apt")
            return 0

        with (
            mock.patch.object(cag, "configure_bullseye_snapshot", fake_configure),
            mock.patch.object(cag, "apt_get", fake_apt_get),
        ):
            rc = cag.main(["--debian-bullseye-snapshot", "--no-sudo", "cmake"])

        self.assertEqual(rc, 0)
        self.assertEqual(events, ["snapshot", "apt"])


if __name__ == "__main__":
    unittest.main()
