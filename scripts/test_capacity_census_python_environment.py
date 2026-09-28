"""C6 native execution must use the selected interpreter's dependencies.

An ambient site directory cannot substitute for a missing owned dependency.
These probes launch real Python processes from a separate fixture environment.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv


class NativePythonEnvironment(unittest.TestCase):
    def test_owned_dependency_loads_and_ambient_dependency_is_unavailable(self):
        with tempfile.TemporaryDirectory(prefix="native-python-env-") as directory:
            root = Path(directory)
            owned = root / "owned"
            venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(owned)
            interpreter = owned / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            paths = json.loads(subprocess.check_output([
                str(interpreter), "-I", "-c",
                "import json,sysconfig; print(json.dumps(sysconfig.get_paths()))",
            ], text=True))
            package = Path(paths["purelib"]) / "census_owned_dependency.py"
            package.write_text("VALUE = 'owned'\n")
            foreign = root / "foreign"
            foreign.mkdir()
            (foreign / package.name).write_text("VALUE = 'foreign'\n")
            (foreign / "census_foreign_only.py").write_text("VALUE = 'foreign'\n")
            scripts = str(Path(__file__).resolve().parent)
            child = (
                "import importlib.util, json, site, sys; "
                "import census_owned_dependency as dependency; "
                "assert dependency.VALUE == 'owned'; "
                "assert importlib.util.find_spec('census_foreign_only') is None; "
                "assert not site.ENABLE_USER_SITE; "
                "assert sys.flags.safe_path; "
                "print(json.dumps({'dependency': dependency.__file__, 'prefix': sys.prefix}))"
            )
            driver = (
                "import os, subprocess, sys; "
                f"sys.path.insert(0, {scripts!r}); "
                "from capacity_census_wire_runner import _managed_python_environment; "
                f"subprocess.run([sys.executable, '-c', {child!r}], "
                "env=_managed_python_environment(), check=True)"
            )
            environment = {
                **os.environ,
                "PYO3_PYTHON": "/unrelated/python",
                "VIRTUAL_ENV": "/unrelated/venv",
                "PYTHONHOME": str(root / "missing-home"),
                "PYTHONPATH": str(foreign),
                "PYTHONUSERBASE": str(foreign),
            }
            result = subprocess.run(
                [str(interpreter), "-I", "-c", driver], cwd=foreign,
                env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            packet = json.loads(result.stdout)
            self.assertEqual(Path(packet["dependency"]).resolve(), package.resolve())
            self.assertEqual(Path(packet["prefix"]).resolve(), owned.resolve())


class InterpreterOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="native-interpreter-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name).resolve()
        self.root = self.base / "workspace"
        self.root.mkdir()

    def interpreter(self, prefix):
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(prefix)
        return prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def owned(self, interpreter, state=None):
        scripts = str(Path(__file__).resolve().parent)
        probe = (
            "import json, sys; from pathlib import Path; "
            f"sys.path.insert(0, {scripts!r}); "
            "from capacity_census_native_execution import _owned_interpreter; "
            f"print(json.dumps(_owned_interpreter(Path({str(self.root)!r}))))"
        )
        environment = dict(os.environ)
        environment.pop("DEVENV_STATE", None)
        if state is not None:
            environment["DEVENV_STATE"] = str(state)
        return json.loads(subprocess.check_output(
            [str(interpreter), "-I", "-c", probe],
            cwd=self.root, env=environment, text=True,
        ))

    def test_checkout_and_activated_devenv_interpreters_are_owned(self):
        self.assertTrue(self.owned(self.interpreter(self.root / ".venv")))
        state = self.root / ".devenv/profiles/ci/state"
        self.assertTrue(self.owned(self.interpreter(state / "venv"), state))

    def test_missing_mismatched_and_foreign_state_cannot_admit_an_interpreter(self):
        state = self.root / ".devenv/profiles/ci/state"
        interpreter = self.interpreter(state / "venv")
        self.assertFalse(self.owned(interpreter))
        self.assertFalse(self.owned(interpreter, state.parent / "other-state"))
        foreign = self.base / "other-worktree/.devenv/state"
        self.assertFalse(self.owned(self.interpreter(foreign / "venv"), foreign))

    @unittest.skipIf(os.name == "nt", "directory symlinks require Windows privileges")
    def test_symlinks_cannot_admit_foreign_devenv_state_or_venvs(self):
        foreign = self.base / "other-worktree/.devenv/state"
        self.interpreter(foreign / "venv")
        state = self.root / ".devenv/profiles/ci/state"
        state.mkdir(parents=True)
        (state / "venv").symlink_to(foreign / "venv", target_is_directory=True)
        self.assertFalse(self.owned(state / "venv/bin/python", state))
        alias = self.root / ".devenv/foreign-state"
        alias.symlink_to(foreign, target_is_directory=True)
        self.assertFalse(self.owned(alias / "venv/bin/python", alias))


if __name__ == "__main__":
    unittest.main()
