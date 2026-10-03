#!/usr/bin/env python3
"""The Python sdist carries the chelis-std package the bundle's build script packs.

`bindings/python/build_backend/chelis_build_backend.py` adds every git-tracked
file under `packages/chelis-std` to the sdist maturin writes. These tests run
that step on a minimal sdist, without maturin, against a scratch repository
and against this checkout.
"""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bindings/python/build_backend"))
# The backend imports maturin for its other hooks; this step needs none of it.
sys.modules.setdefault("maturin", types.ModuleType("maturin"))
import chelis_build_backend as backend  # noqa: E402

TOP = "chelis-0.1.0"


def minimal_sdist(directory: Path) -> Path:
    sdist = directory / f"{TOP}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        data = b"[workspace]\n"
        info = tarfile.TarInfo(f"{TOP}/Cargo.toml")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return sdist


def members(sdist: Path) -> dict[str, bytes]:
    with tarfile.open(sdist, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def tracked(root: Path) -> list[str]:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", "packages/chelis-std"],
        check=True, capture_output=True, text=True,
    ).stdout
    return sorted(listing.splitlines())


class RuntimePackageInSdistTests(unittest.TestCase):
    def test_the_sdist_gains_exactly_the_tracked_package_files(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "repo"
            files = {
                "packages/chelis-std/reef.toml": "[package]\n",
                "packages/chelis-std/src/core.ch": "module Std.Core\n",
                "packages/chelis-std/extra/more.ch": "module Std.More\n",
                "packages/chelis-std/SKILL.md": "notes\n",
                "README.md": "outside the package\n",
            }
            for relative, text in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            (root / "packages/chelis-std/src/.DS_Store").write_text("untracked")
            sdist = minimal_sdist(Path(scratch))
            backend.add_runtime_package(sdist, root)
            found = members(sdist)
        expected = sorted(name for name in files if name.startswith("packages/chelis-std/"))
        added = sorted(name.removeprefix(f"{TOP}/") for name in found if "/packages/" in name)
        self.assertEqual(added, expected)
        for name in expected:
            self.assertEqual(found[f"{TOP}/{name}"], files[name].encode())
        self.assertEqual(found[f"{TOP}/Cargo.toml"], b"[workspace]\n")

    def test_this_checkout_adds_its_git_ls_files_listing(self):
        expected = tracked(REPO_ROOT)
        self.assertIn("packages/chelis-std/reef.toml", expected)
        self.assertEqual(backend.runtime_package_files(REPO_ROOT), expected)
        with tempfile.TemporaryDirectory() as scratch:
            sdist = minimal_sdist(Path(scratch))
            backend.add_runtime_package(sdist, REPO_ROOT)
            added = sorted(
                name.removeprefix(f"{TOP}/") for name in members(sdist) if "/packages/" in name
            )
        self.assertEqual(added, expected)

    def test_a_package_git_does_not_track_is_an_error(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            with self.assertRaisesRegex(RuntimeError, "tracks no files"):
                backend.runtime_package_files(root)


if __name__ == "__main__":
    unittest.main()
