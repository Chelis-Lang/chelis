"""Tests for `scripts/ci_setup_uv_python.py`.

Run via `python3 -m unittest scripts/test_ci_setup_uv_python.py` from the
repo root.

The script's two side effects are (1) `uv venv` and (2) writing to
`$GITHUB_ENV`. We test the pure helpers in isolation. Integration is
covered by CI itself (the LD_LIBRARY_PATH wiring proves out via the
chelis-python test binaries successfully loading libpython).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ci_setup_uv_python  # noqa: E402


class RunnerLibpathVarTest(unittest.TestCase):
    def test_linux_returns_ld_library_path(self) -> None:
        with mock.patch("platform.system", return_value="Linux"):
            self.assertEqual(ci_setup_uv_python.runner_libpath_var(), "LD_LIBRARY_PATH")

    def test_darwin_returns_dyld_library_path(self) -> None:
        with mock.patch("platform.system", return_value="Darwin"):
            self.assertEqual(ci_setup_uv_python.runner_libpath_var(), "DYLD_LIBRARY_PATH")

    def test_unsupported_platform_raises(self) -> None:
        with mock.patch("platform.system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "Unsupported platform"):
                ci_setup_uv_python.runner_libpath_var()


class AppendToGithubEnvTest(unittest.TestCase):
    def test_writes_new_var_when_no_existing(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as fh:
            github_env_path = fh.name
        try:
            with mock.patch.dict(
                os.environ,
                {"GITHUB_ENV": github_env_path},
                clear=False,
            ):
                # Ensure the var doesn't already exist in the env we mock.
                if "FOO_VAR" in os.environ:
                    del os.environ["FOO_VAR"]
                ci_setup_uv_python.append_to_github_env("FOO_VAR", "/foo/lib")
            with open(github_env_path, encoding="utf-8") as fh:
                contents = fh.read()
            self.assertEqual(contents, "FOO_VAR=/foo/lib\n")
        finally:
            os.unlink(github_env_path)

    def test_prepends_to_existing_var(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as fh:
            github_env_path = fh.name
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_ENV": github_env_path,
                    "BAR_VAR": "/already/here",
                },
                clear=False,
            ):
                ci_setup_uv_python.append_to_github_env("BAR_VAR", "/foo/lib")
            with open(github_env_path, encoding="utf-8") as fh:
                contents = fh.read()
            self.assertEqual(contents, "BAR_VAR=/foo/lib:/already/here\n")
        finally:
            os.unlink(github_env_path)

    def test_no_github_env_is_noop(self) -> None:
        # If GITHUB_ENV isn't set, the function logs to stderr and returns
        # without writing anywhere — used as a local dry-run guard.
        with mock.patch.dict(os.environ, {}, clear=True):
            # Should not raise.
            ci_setup_uv_python.append_to_github_env("BAZ", "/baz")


if __name__ == "__main__":
    unittest.main()
