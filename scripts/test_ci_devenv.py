"""Exercise the CI environment handoff without evaluating Nix or building Rust."""
from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_devenv


class ProjectActivationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "github-env"
        self.output.write_text("EXISTING=kept\n")
        self.path_output = self.root / "github-path"
        self.path_output.write_text("/existing/bin\n")
        self.devenv = self.root / "devenv"
        self.state = self.root / "state"
        self.environment = self.state / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(self.environment)
        self.command_bin = self.root / "commands/bin"
        self.command_bin.mkdir(parents=True)
        self.wrapper = self.command_bin / "chelis-ci-shell"
        self.wrapper.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(str(self.environment / 'bin/python'))} "
            f"{shlex.quote(str(Path(ci_devenv.__file__)))} \"$@\"\n"
        )
        self.wrapper.chmod(0o700)

    def activate(self, body):
        shell_environment = (
            f"export DEVENV_DOTFILE={shlex.quote(str(self.state))}\n"
            f"export DEVENV_STATE={shlex.quote(str(self.state))}\n"
            f"export PYO3_PYTHON={shlex.quote(str(self.environment / 'bin/python'))}\n"
            f"export PATH={shlex.quote(str(self.environment / 'bin'))}:"
            f"{shlex.quote(str(self.command_bin))}:\"$PATH\"\n"
            "export LD_LIBRARY_PATH=/project/lib\n"
            "export GITHUB_TOKEN=must-not-publish\n"
        )
        self.devenv.write_text(
            f"#!{sys.executable}\nimport pathlib, sys\n"
            "if 'print-dev-env' in sys.argv:\n"
            f"    print({shell_environment!r})\n"
            "    raise SystemExit(0)\n" + body
        )
        self.devenv.chmod(0o700)
        return ci_devenv.activate(self.root, str(self.devenv), self.output, self.path_output)

    def initialize_body(self):
        receipt = self.state / "load-exports"
        return f"pathlib.Path({str(receipt)!r}).write_text(':\\n')\n"

    def test_success_publishes_complete_environment_without_github_credentials(self):
        self.assertEqual(self.activate(self.initialize_body()), 0)
        values = {"EXISTING": "kept"}
        lines = iter(self.output.read_text().splitlines()[1:])
        for line in lines:
            key, delimiter = line.split("<<", 1)
            value = []
            for item in lines:
                if item == delimiter:
                    break
                value.append(item)
            values[key] = "\n".join(value)
        self.assertNotIn("GITHUB_TOKEN", values)
        self.assertNotIn("LD_LIBRARY_PATH", values)
        self.assertIn("/project/lib", values["CHELIS_CI_LIBRARY_PATH"].split(os.pathsep))
        # Custom-shell lookup sees the runner's path commands before step env.
        runner_path = os.pathsep.join(reversed(self.path_output.read_text().splitlines()))
        self.assertEqual(shutil.which("chelis-ci-shell", path=runner_path), str(self.wrapper))
        command_file = self.root / "managed-python-command"
        command_file.write_text("python -c 'import sys; print(sys.prefix)'\n")
        result = subprocess.run(
            ["chelis-ci-shell", "run", str(command_file)],
            env={**os.environ, **values, "PATH": runner_path},
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.environment))

    def test_project_library_path_is_scoped_to_command_and_failure_propagates(self):
        command_file = self.root / "actions-command"
        command_file.write_text('printf "%s\\n" "$LD_LIBRARY_PATH"\nexit 17\n')
        environment = dict(os.environ, CHELIS_CI_LIBRARY_PATH="/project/lib")
        environment.pop("LD_LIBRARY_PATH", None)
        result = subprocess.run(
            [sys.executable, str(Path(ci_devenv.__file__)), "run", str(command_file)],
            env=environment, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 17)
        self.assertEqual(result.stdout, "/project/lib\n")
        self.assertNotIn("LD_LIBRARY_PATH", environment)

    def test_command_cannot_run_without_successful_project_activation(self):
        command_file = self.root / "actions-command"
        command_file.write_text('printf "payload ran\\n"\n')
        environment = dict(os.environ)
        environment.pop("CHELIS_CI_LIBRARY_PATH", None)
        result = subprocess.run(
            [sys.executable, str(Path(ci_devenv.__file__)), "run", str(command_file)],
            env=environment, text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("payload ran", result.stdout)

    def test_initialization_failure_never_publishes(self):
        result = self.activate(self.initialize_body() + "raise SystemExit(7)\n")
        self.assertEqual(result, 7)
        self.assertEqual(self.output.read_text(), "EXISTING=kept\n")
        self.assertEqual(self.path_output.read_text(), "/existing/bin\n")

    def test_zero_exit_without_payload_never_publishes(self):
        (self.state / "load-exports").write_text("stale completion\n")
        self.assertNotEqual(self.activate("pass\n"), 0)
        self.assertEqual(self.output.read_text(), "EXISTING=kept\n")
        self.assertEqual(self.path_output.read_text(), "/existing/bin\n")


    def test_cancelled_child_never_publishes(self):
        result = self.activate("import os, signal; os.kill(os.getpid(), signal.SIGTERM)\n")
        self.assertEqual(result, 143)
        self.assertEqual(self.output.read_text(), "EXISTING=kept\n")
        self.assertEqual(self.path_output.read_text(), "/existing/bin\n")

    def test_capture_preserves_activated_python_identity_and_rejects_mismatch(self):
        state = self.root / "state"
        environment = state / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
        interpreter = environment / "bin/python"
        capture = self.root / "capture.json"
        command = [str(interpreter), str(Path(ci_devenv.__file__)), "capture", str(capture)]
        env = dict(os.environ, DEVENV_STATE=str(state), PYO3_PYTHON=str(interpreter))
        subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        recorded = json.loads(capture.read_text())
        self.assertEqual(recorded["PYO3_PYTHON"], str(interpreter))
        self.assertEqual(recorded["VIRTUAL_ENV"], str(environment))
        capture.unlink()
        env["PYO3_PYTHON"] = sys.executable
        failed = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(capture.exists())


if __name__ == "__main__":
    unittest.main()
