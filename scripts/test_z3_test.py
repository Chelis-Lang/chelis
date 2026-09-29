"""Tests for platform-specific Z3 discovery and environment setup."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "z3_test", Path(__file__).with_name("z3_test.py")
)
assert _SPEC and _SPEC.loader
z3_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(z3_test)


class EnvGuard:
    """Save and restore selected environment variables."""

    def __init__(self, *names: str) -> None:
        self.names = names
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> "EnvGuard":
        for name in self.names:
            self.saved[name] = os.environ.get(name)
        return self

    def set(self, name: str, value: str) -> None:
        os.environ[name] = value

    def unset(self, name: str) -> None:
        os.environ.pop(name, None)

    def __exit__(self, *exc) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class CandidateDirTests(unittest.TestCase):
    def test_explicit_override_is_first_candidate(self):
        with EnvGuard("Z3_LIBRARY_PATH_OVERRIDE") as guard:
            guard.set("Z3_LIBRARY_PATH_OVERRIDE", "/explicit/override/lib")
            dirs = z3_test.candidate_lib_dirs("Darwin")
            self.assertEqual(dirs[0], Path("/explicit/override/lib"))

    def test_no_override_does_not_crash(self):
        with EnvGuard("Z3_LIBRARY_PATH_OVERRIDE") as guard:
            guard.unset("Z3_LIBRARY_PATH_OVERRIDE")
            self.assertIsInstance(z3_test.candidate_lib_dirs("Linux"), list)


class FindLibz3Tests(unittest.TestCase):
    def test_requires_platform_link_library_name(self):
        with tempfile.TemporaryDirectory() as td:
            lib = Path(td)
            (lib / "libz3.so.4.15").write_bytes(b"")
            self.assertFalse(z3_test.dir_has_linkable_libz3(lib, "Linux"))
            self.assertFalse(z3_test.dir_has_linkable_libz3(lib, "Darwin"))
            (lib / "libz3.dll").write_bytes(b"")
            self.assertFalse(z3_test.dir_has_linkable_libz3(lib, "Windows"))
            (lib / "libz3.lib").write_bytes(b"")
            self.assertTrue(z3_test.dir_has_linkable_libz3(lib, "Windows"))
            (lib / "liblibz3.dll.a").write_bytes(b"")
            self.assertTrue(z3_test.dir_has_linkable_libz3(lib, "Windows"))
            (lib / "libz3.dylib").write_bytes(b"")
            self.assertTrue(z3_test.dir_has_linkable_libz3(lib, "Darwin"))
            (lib / "libz3.so").write_bytes(b"")
            self.assertTrue(z3_test.dir_has_linkable_libz3(lib, "Linux"))

    def test_windows_link_search_paths_use_windows_separator(self):
        with EnvGuard("LIB", "Z3_LIBRARY_PATH_OVERRIDE") as guard:
            guard.set("Z3_LIBRARY_PATH_OVERRIDE", "")
            guard.set("LIB", r"C:\z3\lib;D:\other\lib")
            dirs = z3_test.candidate_lib_dirs("Windows")
            self.assertIn(Path(r"C:\z3\lib"), dirs)
            self.assertIn(Path(r"D:\other\lib"), dirs)

    def test_find_returns_first_linkable_candidate(self):
        with tempfile.TemporaryDirectory() as td, EnvGuard(
            "Z3_LIBRARY_PATH_OVERRIDE"
        ) as guard:
            lib = Path(td)
            (lib / "libz3.dylib").write_bytes(b"")
            guard.set("Z3_LIBRARY_PATH_OVERRIDE", str(lib))
            self.assertEqual(z3_test.find_libz3_dir("Darwin"), lib)

    def test_linux_searches_existing_linker_and_loader_paths(self):
        with EnvGuard(
            "Z3_LIBRARY_PATH_OVERRIDE", "LD_LIBRARY_PATH", "LIBRARY_PATH"
        ) as guard:
            guard.unset("Z3_LIBRARY_PATH_OVERRIDE")
            guard.set("LD_LIBRARY_PATH", "/runtime/z3")
            guard.set("LIBRARY_PATH", "/linker/z3")
            dirs = z3_test.candidate_lib_dirs("Linux")
            self.assertLess(
                dirs.index(Path("/runtime/z3")), dirs.index(Path("/linker/z3"))
            )


class BuildEnvTests(unittest.TestCase):
    def test_linux_uses_ld_library_path_and_preserves_existing_entries(self):
        with EnvGuard("LD_LIBRARY_PATH") as guard:
            guard.set("LD_LIBRARY_PATH", "/pre/existing")
            env = z3_test.build_env(Path("/some/z3/lib"), "Linux")
            self.assertEqual(env["Z3_LIBRARY_PATH_OVERRIDE"], "/some/z3/lib")
            self.assertEqual(
                env["LD_LIBRARY_PATH"], "/some/z3/lib:/pre/existing"
            )

    def test_macos_uses_dyld_library_path(self):
        with EnvGuard("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH") as guard:
            guard.set("DYLD_LIBRARY_PATH", "/pre/existing")
            guard.set("LD_LIBRARY_PATH", "/linux/path")
            env = z3_test.build_env(Path("/opt/homebrew/opt/z3/lib"), "Darwin")
            self.assertEqual(
                env["DYLD_LIBRARY_PATH"],
                "/opt/homebrew/opt/z3/lib:/pre/existing",
            )
            self.assertNotIn("LD_LIBRARY_PATH", env)

    def test_windows_uses_path_separator(self):
        with EnvGuard("PATH") as guard:
            guard.set("PATH", r"C:\existing")
            env = z3_test.build_env(Path(r"C:\z3\lib"), "Windows")
            self.assertEqual(
                env["PATH"], r"C:\z3\lib;C:\existing"
            )

    def test_none_needs_no_override(self):
        self.assertEqual(z3_test.build_env(None, "Darwin"), {})


class DefaultArgsTests(unittest.TestCase):
    def test_default_args_target_chelis_prove_z3(self):
        self.assertEqual(
            z3_test.DEFAULT_ARGS, ["-p", "chelis-prove", "--features", "z3"]
        )


if __name__ == "__main__":
    unittest.main()
