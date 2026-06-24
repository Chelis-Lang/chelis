#!/usr/bin/env python3
"""Unit tests for `z3_test.py` (the prebuilt-libz3 link wrapper).

Run via: `python3 -m unittest scripts.test_z3_test` from repo root, or
`python3 scripts/test_z3_test.py`.

Covers the PURE helpers: override precedence in `candidate_lib_dirs`,
`find_libz3_dir`'s requirement for an unversioned `libz3.so` (the `-lz3` link
target, not just a versioned `.so.4.15`), and `build_env`'s override-vs-system
behaviour (an explicit/discovered dir sets both link + loader vars and PREPENDS
to any existing `LD_LIBRARY_PATH`; the apt-libz3 case is an empty override).
"""

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
    """Save/restore selected env vars around a test body."""

    def __init__(self, *names: str) -> None:
        self.names = names
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> "EnvGuard":
        for n in self.names:
            self.saved[n] = os.environ.get(n)
        return self

    def set(self, name: str, value: str) -> None:
        os.environ[name] = value

    def unset(self, name: str) -> None:
        os.environ.pop(name, None)

    def __exit__(self, *exc) -> None:
        for n, v in self.saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


class CandidateDirTests(unittest.TestCase):
    def test_override_is_first_candidate(self):
        with EnvGuard("Z3_LIBRARY_PATH_OVERRIDE") as g:
            g.set("Z3_LIBRARY_PATH_OVERRIDE", "/explicit/override/lib")
            dirs = z3_test.candidate_lib_dirs()
            self.assertEqual(
                dirs[0],
                Path("/explicit/override/lib"),
                "an explicit Z3_LIBRARY_PATH_OVERRIDE must take priority",
            )

    def test_no_override_does_not_crash(self):
        with EnvGuard("Z3_LIBRARY_PATH_OVERRIDE") as g:
            g.unset("Z3_LIBRARY_PATH_OVERRIDE")
            dirs = z3_test.candidate_lib_dirs()
            self.assertIsInstance(dirs, list)


class FindLibz3Tests(unittest.TestCase):
    def test_requires_unversioned_so(self):
        with tempfile.TemporaryDirectory() as td:
            lib = Path(td) / "lib"
            lib.mkdir()
            # A versioned-only libz3.so.4.15 is NOT a linkable -lz3 target.
            (lib / "libz3.so.4.15").write_bytes(b"")
            self.assertFalse(
                z3_test.dir_has_linkable_libz3(lib),
                "a versioned-only libz3.so.4.15 must be rejected",
            )
            # The unversioned symlink makes it linkable.
            (lib / "libz3.so").write_bytes(b"")
            self.assertTrue(z3_test.dir_has_linkable_libz3(lib))

    def test_find_returns_first_linkable_candidate(self):
        # find_libz3_dir scans candidate_lib_dirs in order; an override dir with
        # a linkable libz3.so is selected first.
        with tempfile.TemporaryDirectory() as td, EnvGuard(
            "Z3_LIBRARY_PATH_OVERRIDE"
        ) as g:
            lib = Path(td) / "lib"
            lib.mkdir()
            (lib / "libz3.so").write_bytes(b"")
            g.set("Z3_LIBRARY_PATH_OVERRIDE", str(lib))
            self.assertEqual(z3_test.find_libz3_dir(), lib)


class BuildEnvTests(unittest.TestCase):
    def test_with_lib_dir_sets_both_vars(self):
        with EnvGuard("LD_LIBRARY_PATH") as g:
            g.unset("LD_LIBRARY_PATH")
            env = z3_test.build_env(Path("/some/z3/lib"))
            self.assertEqual(env["Z3_LIBRARY_PATH_OVERRIDE"], "/some/z3/lib")
            self.assertEqual(env["LD_LIBRARY_PATH"], "/some/z3/lib")

    def test_preserves_existing_ld_library_path(self):
        with EnvGuard("LD_LIBRARY_PATH") as g:
            g.set("LD_LIBRARY_PATH", "/pre/existing")
            env = z3_test.build_env(Path("/some/z3/lib"))
            self.assertEqual(
                env["LD_LIBRARY_PATH"],
                "/some/z3/lib:/pre/existing",
                "the discovered dir must PREPEND, preserving existing entries",
            )

    def test_none_is_empty_for_system_libz3(self):
        # The apt libz3-dev (CI) case: no override needed, env is empty.
        self.assertEqual(z3_test.build_env(None), {})


class DefaultArgsTests(unittest.TestCase):
    def test_default_args_target_chelis_prove_z3(self):
        self.assertEqual(
            z3_test.DEFAULT_ARGS, ["-p", "chelis-prove", "--features", "z3"]
        )


if __name__ == "__main__":
    unittest.main()
