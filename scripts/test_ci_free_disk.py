"""Unit tests for `ci_free_disk.py`.

Run via: `python3 -m unittest scripts.test_ci_free_disk` from repo root,
or `python3 scripts/test_ci_free_disk.py`.

The script removes root-owned system directories on the CI runner, so the
tests mock `subprocess.run`/`shutil.which`/`sys.platform` and assert the
*behavior* (which paths it tries to remove, that it is non-fatal, and that
it no-ops off Linux) without touching the real filesystem.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ci_free_disk", here / "ci_free_disk.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cfd = _load_module()


class NoOpOffLinuxTests(unittest.TestCase):
    def test_non_linux_does_nothing_and_returns_zero(self):
        # On macOS/Windows the script must be a pure no-op: no subprocess
        # at all (so running it locally cannot delete anything).
        with (
            mock.patch.object(cfd.sys, "platform", "darwin"),
            mock.patch.object(cfd.subprocess, "run") as run,
            mock.patch.object(cfd.shutil, "which") as which,
        ):
            self.assertEqual(cfd.free_disk_space(), 0)
            run.assert_not_called()
            which.assert_not_called()


class LinuxBehaviorTests(unittest.TestCase):
    def _completed(self, returncode=0):
        return mock.Mock(returncode=returncode, stdout="", stderr="")

    def test_removes_every_purge_path_with_sudo_rm_rf(self):
        with (
            mock.patch.object(cfd.sys, "platform", "linux"),
            mock.patch.object(cfd.shutil, "which", return_value=None),
            mock.patch.object(cfd.subprocess, "run", return_value=self._completed()) as run,
        ):
            self.assertEqual(cfd.free_disk_space(), 0)

        rm_targets = [
            call.args[0][-1]
            for call in run.call_args_list
            if call.args and call.args[0][:3] == ["sudo", "rm", "-rf"]
        ]
        # Every advertised path is attempted, exactly once each.
        self.assertEqual(rm_targets, list(cfd.PURGE_PATHS))

    def test_purge_paths_are_targeted_not_the_whole_tool_cache(self):
        # Guard the safety invariant: only the CodeQL subdir of the tool
        # cache is removed, never all of /opt/hostedtoolcache (which would
        # break the Rust/nextest/uv installs that run after this step).
        self.assertIn("/opt/hostedtoolcache/CodeQL", cfd.PURGE_PATHS)
        self.assertNotIn("/opt/hostedtoolcache", cfd.PURGE_PATHS)

    def test_prunes_docker_images_only_when_docker_present(self):
        with (
            mock.patch.object(cfd.sys, "platform", "linux"),
            mock.patch.object(cfd.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(cfd.subprocess, "run", return_value=self._completed()) as run,
        ):
            cfd.free_disk_space()
        cmds = [call.args[0] for call in run.call_args_list if call.args]
        self.assertIn(["sudo", "docker", "image", "prune", "--all", "--force"], cmds)

    def test_skips_docker_prune_when_docker_absent(self):
        with (
            mock.patch.object(cfd.sys, "platform", "linux"),
            mock.patch.object(cfd.shutil, "which", return_value=None),
            mock.patch.object(cfd.subprocess, "run", return_value=self._completed()) as run,
        ):
            cfd.free_disk_space()
        cmds = [call.args[0] for call in run.call_args_list if call.args]
        self.assertNotIn(["sudo", "docker", "image", "prune", "--all", "--force"], cmds)

    def test_non_fatal_when_a_remove_fails(self):
        # A non-zero `rm` exit (e.g. path absent) must not raise or change
        # the success return — the step is best-effort.
        with (
            mock.patch.object(cfd.sys, "platform", "linux"),
            mock.patch.object(cfd.shutil, "which", return_value=None),
            mock.patch.object(cfd.subprocess, "run", return_value=self._completed(returncode=1)),
        ):
            self.assertEqual(cfd.free_disk_space(), 0)

    def test_non_fatal_when_subprocess_cannot_spawn(self):
        # If the OS refuses to spawn (OSError), the helper swallows it and
        # still returns success.
        with (
            mock.patch.object(cfd.sys, "platform", "linux"),
            mock.patch.object(cfd.shutil, "which", return_value=None),
            mock.patch.object(cfd.subprocess, "run", side_effect=OSError("boom")),
        ):
            self.assertEqual(cfd.free_disk_space(), 0)


if __name__ == "__main__":
    unittest.main()
