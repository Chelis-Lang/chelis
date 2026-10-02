"""Exercise the hosted project command shims without uv or a Rust toolchain."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
import subprocess
import sys
import tempfile
import unittest
import venv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_hosted_commands


class HostedCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "checkout"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts" / "gate.py").write_text(
            "import sys; print(sys.executable); print(sys.argv[1:])\n"
        )
        self.commands = Path(self.temporary.name) / "commands"
        self.github_env = Path(self.temporary.name) / "github-env"
        self.github_path = Path(self.temporary.name) / "github-path"
        self.github_path.write_text("/existing/bin\n")

    def runner_path(self):
        published = self.github_path.read_text().splitlines()
        return os.pathsep.join([*reversed(published), os.environ["PATH"]])

    def test_shims_run_commands_and_the_gate_with_the_venv_interpreter(self):
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(self.root / ".venv")
        interpreter = self.root / ".venv" / "bin" / "python"
        ci_hosted_commands.publish(self.root, self.commands, self.github_env, self.github_path)
        published_environment = dict(line.split("=", 1) for line in self.github_env.read_text().splitlines())
        self.assertEqual(published_environment["PYO3_PYTHON"], str(interpreter))
        self.assertTrue(published_environment["PYO3_ENVIRONMENT_SIGNATURE"].startswith("chelis-pyo3-v1-"))
        environment = {**os.environ, "PATH": self.runner_path(), "PYO3_PYTHON": str(interpreter)}
        command_file = Path(self.temporary.name) / "step"
        command_file.write_text("python -c 'import sys; print(sys.executable)'\nchelis-gate lint-and-unit --list\n")
        result = subprocess.run(
            ["chelis-ci-shell", "run", str(command_file)],
            env=environment, text=True, capture_output=True, cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [str(interpreter), str(interpreter), "['lint-and-unit', '--list']"],
        )
        # A failing command fails the step, like the Devenv shell.
        command_file.write_text("exit 17\n")
        failed = subprocess.run(
            ["chelis-ci-shell", "run", str(command_file)], env=environment, capture_output=True,
        )
        self.assertEqual(failed.returncode, 17)
        self.assertEqual(
            subprocess.run(["chelis-ci-shell", "not-run"], env=environment, capture_output=True).returncode,
            64,
        )

    def test_discovery_override_is_rejected_before_anything_is_published(self):
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(self.root / ".venv")
        with mock.patch.dict(os.environ, {"PYO3_CONFIG_FILE": "/foreign/config"}):
            with self.assertRaisesRegex(RuntimeError, "PYO3_CONFIG_FILE"):
                ci_hosted_commands.publish(self.root, self.commands, self.github_env, self.github_path)
        self.assertFalse(self.commands.exists())
        self.assertFalse(self.github_env.exists())
        self.assertEqual(self.github_path.read_text(), "/existing/bin\n")

    def test_republication_replaces_stale_signature_with_deterministic_scalar(self):
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(self.root / ".venv")
        with mock.patch.dict(os.environ, {"PYO3_ENVIRONMENT_SIGNATURE": "stale"}):
            ci_hosted_commands.publish(self.root, self.commands, self.github_env, self.github_path)
            first = self.github_env.read_text()
            ci_hosted_commands.publish(self.root, self.commands, self.github_env, self.github_path)
        self.assertEqual(self.github_env.read_text(), first + first)
        self.assertNotIn("stale", first)

    def test_missing_venv_is_refused_before_anything_is_published(self):
        with self.assertRaises(RuntimeError):
            ci_hosted_commands.publish(self.root, self.commands, self.github_env, self.github_path)
        self.assertFalse(self.commands.exists())
        self.assertFalse(self.github_env.exists())
        self.assertEqual(self.github_path.read_text(), "/existing/bin\n")


if __name__ == "__main__":
    unittest.main()
