"""Tests for `scripts/ci_setup_uv_python.py`.

Run via `python3 -m unittest scripts/test_ci_setup_uv_python.py` from the
repo root.

The script's two side effects are (1) `uv venv` and (2) writing to
`$GITHUB_ENV`. We test the pure helpers in isolation and use mocks to
exercise the subprocess wrappers without actually invoking uv. Real
end-to-end proof lives in CI itself (the LD_LIBRARY_PATH wiring proves
out via the chelis-python test binaries successfully loading libpython).
"""

from __future__ import annotations

import os
import subprocess
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

    def test_dedupes_repeated_path_entries(self) -> None:
        # When the same lib dir is already present in the existing value
        # (e.g., a workflow step ran this helper twice within one job),
        # the dedupe step collapses repeats while preserving first-
        # occurrence order. Without this, the value grows linearly on
        # re-runs.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as fh:
            github_env_path = fh.name
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_ENV": github_env_path,
                    "BAZ_VAR": "/foo/lib:/other/lib",
                },
                clear=False,
            ):
                ci_setup_uv_python.append_to_github_env("BAZ_VAR", "/foo/lib")
            with open(github_env_path, encoding="utf-8") as fh:
                contents = fh.read()
            self.assertEqual(contents, "BAZ_VAR=/foo/lib:/other/lib\n")
        finally:
            os.unlink(github_env_path)

    def test_no_github_env_is_noop(self) -> None:
        # If GITHUB_ENV isn't set, the function logs to stderr and returns
        # without writing anywhere — used as a local dry-run guard.
        with mock.patch.dict(os.environ, {}, clear=True):
            ci_setup_uv_python.append_to_github_env("BAZ", "/baz")  # no raise


class LibdirForTest(unittest.TestCase):
    def test_returns_stripped_stdout(self) -> None:
        fake_python = Path("/fake/.venv/bin/python")
        fake_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="/some/lib/dir\n",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=fake_result) as run:
            self.assertEqual(
                ci_setup_uv_python.libdir_for(fake_python),
                "/some/lib/dir",
            )
            run.assert_called_once()
            args = run.call_args.args[0]
            self.assertEqual(args[0], str(fake_python))
            self.assertIn("sysconfig.get_config_var", args[2])

    def test_empty_stdout_raises(self) -> None:
        # Bare-empty stdout (rare) must surface as a runtime error rather
        # than silently writing "" to GITHUB_ENV.
        fake_python = Path("/fake/.venv/bin/python")
        fake_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="   \n",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=fake_result):
            with self.assertRaisesRegex(RuntimeError, "returned empty"):
                ci_setup_uv_python.libdir_for(fake_python)

    def test_subprocess_failure_propagates(self) -> None:
        fake_python = Path("/fake/.venv/bin/python")
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["fake"], stderr="boom"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                ci_setup_uv_python.libdir_for(fake_python)


class CreateVenvTest(unittest.TestCase):
    def test_uv_not_installed_raises_systemexit_with_hint(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(SystemExit) as cm:
                ci_setup_uv_python.create_venv("3.11")
        self.assertIn("`uv` not found on PATH", str(cm.exception))

    def test_uv_venv_failure_raises_systemexit_with_hint(self) -> None:
        err = subprocess.CalledProcessError(1, ["uv", "venv", "--python", "3.11"])
        with mock.patch("subprocess.run", side_effect=err):
            with self.assertRaises(SystemExit) as cm:
                ci_setup_uv_python.create_venv("3.11")
        msg = str(cm.exception)
        self.assertIn("uv venv", msg)
        self.assertIn("3.11", msg)
        self.assertIn("exited with code 1", msg)

    def test_venv_python_missing_after_uv_success_raises(self) -> None:
        # Defensive: if `uv venv` returns 0 but the expected interpreter
        # path doesn't materialize, surface that as a FileNotFoundError
        # rather than silently returning a non-existent path.
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            with mock.patch.object(Path, "exists", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    ci_setup_uv_python.create_venv("3.11")


class PinnedPythonVersionTest(unittest.TestCase):
    def test_reads_version_from_actual_pyproject(self) -> None:
        # Lock that the script can parse the repo's actual pyproject.
        # Asserts the shape, not a specific value, so a future bump
        # doesn't require updating this test.
        version = ci_setup_uv_python.pinned_python_version()
        self.assertRegex(version, r"^\d+\.\d+(?:\.\d+)?$")

    def test_missing_pyproject_raises(self) -> None:
        fake_path = Path("/nonexistent/pyproject.toml")
        with self.assertRaisesRegex(RuntimeError, "not found"):
            ci_setup_uv_python.pinned_python_version(fake_path)

    def test_unparseable_pyproject_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as fh:
            fh.write("[project]\nname = 'foo'\n")  # no requires-python
            bad_path = Path(fh.name)
        try:
            with self.assertRaisesRegex(RuntimeError, "could not parse"):
                ci_setup_uv_python.pinned_python_version(bad_path)
        finally:
            bad_path.unlink()

    def test_parses_various_requires_python_styles(self) -> None:
        for spec, expected in [
            ('requires-python = ">=3.11"', "3.11"),
            ("requires-python = '~=3.12'", "3.12"),
            ('requires-python = ">= 3.10"', "3.10"),
            ('requires-python = "==3.11.4"', "3.11.4"),
        ]:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as fh:
                fh.write(f"[project]\n{spec}\n")
                path = Path(fh.name)
            try:
                self.assertEqual(
                    ci_setup_uv_python.pinned_python_version(path),
                    expected,
                )
            finally:
                path.unlink()


class PythonAbiVersionTest(unittest.TestCase):
    def test_returns_major_minor(self) -> None:
        fake_python = Path("/fake/.venv/bin/python")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="3.11\n", stderr=""
        )
        with mock.patch("subprocess.run", return_value=fake_result):
            self.assertEqual(
                ci_setup_uv_python.python_abi_version(fake_python),
                "3.11",
            )

    def test_empty_stdout_raises(self) -> None:
        fake_python = Path("/fake/.venv/bin/python")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="   \n", stderr=""
        )
        with mock.patch("subprocess.run", return_value=fake_result):
            with self.assertRaisesRegex(RuntimeError, "empty"):
                ci_setup_uv_python.python_abi_version(fake_python)


def _signature_for(version_stdout: str, libdir: str) -> str:
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=version_stdout, stderr=""
    )
    with mock.patch("subprocess.run", return_value=fake_result):
        return ci_setup_uv_python.interpreter_signature(
            Path("/fake/.venv/bin/python"), libdir
        )


class InterpreterSignatureTest(unittest.TestCase):
    """`PYO3_ENVIRONMENT_SIGNATURE` is the only input that makes PyO3 rerun
    its build configuration when the interpreter behind an unchanged
    `PYO3_PYTHON` path changes (chelis#2895)."""

    V16 = "3.11.16 (main, Sep  2 2026, 18:00:00) [Clang 22.1.0 ]\n"
    V17 = "3.11.17 (main, Oct  1 2026, 20:58:47) [Clang 22.1.3 ]\n"
    LIB16 = "/tmp/uv-python-dir/cpython-3.11.16-linux-x86_64-gnu/lib"
    LIB17 = "/tmp/uv-python-dir/cpython-3.11.17-linux-x86_64-gnu/lib"

    def test_patch_release_changes_signature(self) -> None:
        self.assertNotEqual(
            _signature_for(self.V16, self.LIB16),
            _signature_for(self.V17, self.LIB17),
        )

    def test_same_interpreter_keeps_signature(self) -> None:
        # A stable signature for an unchanged interpreter keeps cached
        # PyO3 builds valid; only a real change may trigger a rebuild.
        self.assertEqual(
            _signature_for(self.V17, self.LIB17),
            _signature_for(self.V17, self.LIB17),
        )

    def test_rebuilt_release_at_same_version_changes_signature(self) -> None:
        rebuilt = "3.11.17 (main, Oct  9 2026, 08:00:00) [Clang 22.1.4 ]\n"
        self.assertNotEqual(
            _signature_for(self.V17, self.LIB17),
            _signature_for(rebuilt, self.LIB17),
        )

    def test_moved_libdir_changes_signature(self) -> None:
        self.assertNotEqual(
            _signature_for(self.V17, self.LIB17),
            _signature_for(self.V17, "/elsewhere/cpython-3.11.17/lib"),
        )

    def test_signature_is_one_line_with_version_and_libdir(self) -> None:
        multiline = "3.11.17 (main, Oct  1 2026, 20:58:47)\n[Clang 22.1.3 ]\n"
        signature = _signature_for(multiline, self.LIB17)
        self.assertEqual(
            signature,
            f"3.11.17 (main, Oct 1 2026, 20:58:47) [Clang 22.1.3 ] {self.LIB17}",
        )

    def test_queries_the_given_interpreter(self) -> None:
        fake_python = Path("/fake/.venv/bin/python")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=self.V17, stderr=""
        )
        with mock.patch("subprocess.run", return_value=fake_result) as run:
            ci_setup_uv_python.interpreter_signature(fake_python, self.LIB17)
        args = run.call_args.args[0]
        self.assertEqual(args[0], str(fake_python))
        self.assertIn("sys.version", args[2])

    def test_real_interpreter_signature_names_its_version(self) -> None:
        python = Path(sys.executable)
        signature = ci_setup_uv_python.interpreter_signature(python, "/some/lib")
        self.assertTrue(signature.startswith(sys.version.split()[0] + " "))
        self.assertTrue(signature.endswith(" /some/lib"))
        self.assertNotIn("\n", signature)

    def test_empty_stdout_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "empty sys.version"):
            _signature_for("  \n", self.LIB17)


class SetGithubEnvTest(unittest.TestCase):
    def _written(self, var: str, value: str, inherited: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as fh:
            github_env_path = fh.name
        try:
            with mock.patch.dict(
                os.environ,
                {"GITHUB_ENV": github_env_path, **inherited},
                clear=True,
            ):
                ci_setup_uv_python.set_github_env(var, value)
            with open(github_env_path, encoding="utf-8") as fh:
                return fh.read()
        finally:
            os.unlink(github_env_path)

    def test_writes_value_verbatim_with_colons(self) -> None:
        # A path-list merge would split the build time on ':'.
        value = "3.11.17 (main, Oct 1 2026, 20:58:47) [Clang 22.1.3 ] /a/lib"
        self.assertEqual(
            self._written("SIG", value, {}),
            f"SIG={value}\n",
        )

    def test_replaces_inherited_value(self) -> None:
        self.assertEqual(
            self._written("SIG", "new", {"SIG": "old"}),
            "SIG=new\n",
        )

    def test_multiline_value_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "one line"):
            self._written("SIG", "a\nb", {})

    def test_no_github_env_is_noop(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            ci_setup_uv_python.set_github_env("SIG", "value")  # no raise


class MainExportsTest(unittest.TestCase):
    def _run_main(self) -> str:
        with tempfile.TemporaryDirectory() as td:
            github_env = Path(td) / "github_env"
            github_env.touch()
            libdir = "/uv/cpython-3.11.17-linux-x86_64-gnu/lib"
            module = ci_setup_uv_python
            with mock.patch.dict(os.environ, {"GITHUB_ENV": str(github_env)}, clear=True), \
                    mock.patch.object(sys, "argv", ["ci_setup_uv_python.py"]), \
                    mock.patch("platform.system", return_value="Linux"), \
                    mock.patch.object(module, "create_venv", return_value=Path("/v/bin/python")), \
                    mock.patch.object(module, "libdir_for", return_value=libdir), \
                    mock.patch.object(module, "python_abi_version", return_value="3.11"), \
                    mock.patch.object(module, "discover_python_libdirs", return_value=[Path(libdir)]), \
                    mock.patch.object(module, "ensure_default_uv_root_mirror"), \
                    mock.patch.object(module, "ensure_link_symlink"), \
                    mock.patch.object(module, "interpreter_signature", return_value="3.11.17 sig"):
                self.assertEqual(module.main(), 0)
            return github_env.read_text(encoding="utf-8")

    def test_exports_library_path_and_pyo3_signature(self) -> None:
        self.assertEqual(
            self._run_main().splitlines(),
            [
                "LD_LIBRARY_PATH=/uv/cpython-3.11.17-linux-x86_64-gnu/lib",
                "PYO3_ENVIRONMENT_SIGNATURE=3.11.17 sig",
            ],
        )


class EnsureLinkSymlinkTest(unittest.TestCase):
    """The Linux symlink path is the one that mattered for PR #184 CI;
    macOS is no-op and tested for symmetry. Use real temp dirs so the
    symlink syscall actually executes."""

    def test_linux_creates_symlink_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            target = libdir / "libpython3.11.so.1.0"
            target.touch()
            link_name = libdir / "libpython3.11.so"
            self.assertFalse(link_name.exists())
            with mock.patch("platform.system", return_value="Linux"):
                ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")
            self.assertTrue(link_name.is_symlink())
            self.assertEqual(
                Path(os.readlink(link_name)),
                Path("libpython3.11.so.1.0"),
            )

    def test_linux_noop_when_symlink_already_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            target = libdir / "libpython3.11.so.1.0"
            target.touch()
            link_name = libdir / "libpython3.11.so"
            link_name.symlink_to(target.name)
            mtime_before = link_name.lstat().st_mtime
            with mock.patch("platform.system", return_value="Linux"):
                ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")
            self.assertEqual(link_name.lstat().st_mtime, mtime_before)

    def test_linux_raises_when_target_missing(self) -> None:
        # If no libpython files at all are present, surface the broken
        # uv install with a directory listing rather than silently
        # leaving a dangling link.
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            with mock.patch("platform.system", return_value="Linux"):
                with self.assertRaisesRegex(RuntimeError, "no libpython3.11 files found"):
                    ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")

    def test_linux_glob_picks_shortest_versioned_name(self) -> None:
        # uv may ship multiple versioned files (e.g., `.so.1.0` AND
        # `.so.1.0.X` for a debug build). The helper picks the shortest
        # name so the linker resolves to the canonical SONAME target.
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            (libdir / "libpython3.11.so.1.0").touch()
            (libdir / "libpython3.11.so.1.0.debug").touch()
            link_name = libdir / "libpython3.11.so"
            with mock.patch("platform.system", return_value="Linux"):
                ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")
            self.assertEqual(
                Path(os.readlink(link_name)),
                Path("libpython3.11.so.1.0"),
            )

    def test_linux_replaces_broken_symlink(self) -> None:
        # If a previous run left a broken symlink (target deleted), the
        # helper replaces it rather than skipping or crashing.
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            broken_target = libdir / "missing.so"
            link_name = libdir / "libpython3.11.so"
            link_name.symlink_to(broken_target.name)
            self.assertTrue(link_name.is_symlink())
            self.assertFalse(link_name.exists())  # broken
            (libdir / "libpython3.11.so.1.0").touch()
            with mock.patch("platform.system", return_value="Linux"):
                ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")
            self.assertEqual(
                Path(os.readlink(link_name)),
                Path("libpython3.11.so.1.0"),
            )

    def test_darwin_noop_when_dylib_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            dylib = libdir / "libpython3.11.dylib"
            dylib.touch()
            with mock.patch("platform.system", return_value="Darwin"):
                ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")
            self.assertTrue(dylib.is_file())  # Untouched.

    def test_unsupported_platform_is_silent_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            with mock.patch("platform.system", return_value="Windows"):
                ci_setup_uv_python.ensure_link_symlink(libdir, "3.11")
            self.assertEqual(list(libdir.iterdir()), [])


class DiscoverPythonLibdirsTest(unittest.TestCase):
    """Discovery walks UV_PYTHON_INSTALL_DIR + ~/.local/share/uv/python/
    to find every libpython install. Needed because pyo3-build-config
    can query a different uv install copy than `.venv/bin/python`."""

    def test_primary_libdir_included_first(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            libdir = Path(td)
            (libdir / "libpython3.11.so.1.0").touch()
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch.dict(os.environ, {}, clear=True):
                    result = ci_setup_uv_python.discover_python_libdirs(str(libdir), "3.11")
            self.assertGreaterEqual(len(result), 1)
            self.assertEqual(result[0].resolve(), libdir.resolve())

    def test_primary_libdir_skipped_when_no_libpython(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty"
            empty.mkdir()
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch.object(Path, "home", return_value=Path("/nonexistent")):
                        result = ci_setup_uv_python.discover_python_libdirs(str(empty), "3.11")
            self.assertEqual(result, [])

    def test_finds_uv_python_install_dir_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            install = root / "cpython-3.11.15-linux-x86_64-gnu"
            (install / "lib").mkdir(parents=True)
            (install / "lib" / "libpython3.11.so.1.0").touch()
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch.dict(
                    os.environ,
                    {"UV_PYTHON_INSTALL_DIR": str(root)},
                    clear=True,
                ):
                    with mock.patch.object(Path, "home", return_value=Path("/nonexistent")):
                        result = ci_setup_uv_python.discover_python_libdirs(
                            "/some/primary/that/does/not/exist", "3.11"
                        )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "lib")

    def test_dedupes_roots_that_resolve_to_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            install = root / "cpython-3.11.15-linux-x86_64-gnu"
            (install / "lib").mkdir(parents=True)
            (install / "lib" / "libpython3.11.so.1.0").touch()
            libdir = install / "lib"
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch.dict(
                    os.environ,
                    {"UV_PYTHON_INSTALL_DIR": str(root)},
                    clear=True,
                ):
                    with mock.patch.object(Path, "home", return_value=Path("/nonexistent")):
                        result = ci_setup_uv_python.discover_python_libdirs(str(libdir), "3.11")
            self.assertEqual(len(result), 1)


class EnsureDefaultUvRootMirrorTest(unittest.TestCase):
    """The mirror exists to defeat pyo3-build-config caching that may
    persist a -L flag pointing at `~/.local/share/uv/python/<install>/lib`
    even when uv staged Python elsewhere."""

    def test_mirrors_libpython_to_default_root(self) -> None:
        with tempfile.TemporaryDirectory() as actual_root_str:
            with tempfile.TemporaryDirectory() as fake_home_str:
                actual_root = Path(actual_root_str)
                install = actual_root / "cpython-3.11.15-linux-x86_64-gnu"
                libdir = install / "lib"
                libdir.mkdir(parents=True)
                lib_file = libdir / "libpython3.11.so.1.0"
                lib_file.touch()
                fake_home = Path(fake_home_str)
                with mock.patch("platform.system", return_value="Linux"):
                    with mock.patch.object(Path, "home", return_value=fake_home):
                        ci_setup_uv_python.ensure_default_uv_root_mirror(
                            str(libdir), "3.11"
                        )
                mirror = (
                    fake_home
                    / ".local"
                    / "share"
                    / "uv"
                    / "python"
                    / "cpython-3.11.15-linux-x86_64-gnu"
                    / "lib"
                    / "libpython3.11.so"
                )
                self.assertTrue(mirror.is_symlink())
                self.assertEqual(
                    Path(os.readlink(mirror)),
                    lib_file.resolve(),
                )

    def test_noop_when_libdir_is_not_uv_style(self) -> None:
        # System Python lives at /usr/lib/python3.11; libdir is /usr/lib
        # whose parent is /usr, not a cpython-X install. Helper should
        # noop rather than create a mirror.
        with tempfile.TemporaryDirectory() as actual_root_str:
            with tempfile.TemporaryDirectory() as fake_home_str:
                libdir = Path(actual_root_str) / "lib"
                libdir.mkdir()
                (libdir / "libpython3.11.so.1.0").touch()
                fake_home = Path(fake_home_str)
                with mock.patch("platform.system", return_value="Linux"):
                    with mock.patch.object(Path, "home", return_value=fake_home):
                        ci_setup_uv_python.ensure_default_uv_root_mirror(
                            str(libdir), "3.11"
                        )
                # No mirror should have been created.
                default_root = fake_home / ".local" / "share" / "uv" / "python"
                self.assertFalse(default_root.exists())

    def test_noop_when_libdir_has_no_libpython(self) -> None:
        with tempfile.TemporaryDirectory() as actual_root_str:
            with tempfile.TemporaryDirectory() as fake_home_str:
                install = Path(actual_root_str) / "cpython-3.11.15-linux-x86_64-gnu"
                libdir = install / "lib"
                libdir.mkdir(parents=True)
                # No libpython file in this dir.
                fake_home = Path(fake_home_str)
                with mock.patch("platform.system", return_value="Linux"):
                    with mock.patch.object(Path, "home", return_value=fake_home):
                        ci_setup_uv_python.ensure_default_uv_root_mirror(
                            str(libdir), "3.11"
                        )
                default_root = fake_home / ".local" / "share" / "uv" / "python"
                # The dir might exist as we mkdir'd it, but no symlink
                # should have been created.
                if default_root.exists():
                    mirror_libdir = (
                        default_root / "cpython-3.11.15-linux-x86_64-gnu" / "lib"
                    )
                    self.assertFalse((mirror_libdir / "libpython3.11.so").exists())


if __name__ == "__main__":
    unittest.main()
