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


if __name__ == "__main__":
    unittest.main()
