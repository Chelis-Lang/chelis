#!/usr/bin/env python3
"""Positive and negative fixtures for `check_std_bundle_untracked.py`.

Each test builds a scratch git repository. Clean trees, ignored build
outputs, and locks without a bundled chelis-std pass. A tracked file under
either generated directory, a tracked lock with a bundled chelis-std entry,
and an unreadable tracked lock fail; the staged warning mode reports the same
findings and exits 0.
"""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_std_bundle_untracked as guard


BUNDLED_LOCK = """schema = "1"

[package]
name = "probe"
version = "0.1.0"

[[dependencies]]
name = "chelis-std"
version = "0.4.0"
compiler = "=0.18.12"
archive_sha256 = "faec56b03d81dca15edea895d1ef79a3ae9e71a665ef673e0dde91f55ee8d4f0"
shell_sha256 = "9f35434d1e8547f5bb772e4bcc86824a1e0fb0138e348a10853e2ef197727830"

[dependencies.source]
kind = "bundled"
compiler_version = "0.18.12"
"""

PATH_ONLY_LOCK = """schema = "1"

[package]
name = "probe"
version = "0.1.0"

[[dependencies]]
name = "helper"
version = "0.1.0"
compiler = "=0.18.12"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "../helper"
"""


class Repository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q")
        self.git("config", "user.email", "probe@example.invalid")
        self.git("config", "user.name", "Probe")
        self.write("README.md", "probe\n")
        self.git("add", "README.md")
        self.git("commit", "-q", "-m", "init")

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.root), *args], check=True)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def track(self, relative: str, text: str) -> None:
        self.write(relative, text)
        self.git("add", "-f", relative)

    def run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = guard.main(["--repo", str(self.root), *args], out=out, err=err)
        return code, out.getvalue(), err.getvalue()


class StdBundleTrackingTests(unittest.TestCase):
    def repository(self) -> Repository:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Repository(Path(directory.name))

    def assert_refused(self, repo: Repository, path: str) -> str:
        code, out, err = repo.run()
        self.assertEqual(code, 1, err)
        self.assertIn("std bundle tracking: FAIL", out)
        self.assertIn(path, err)
        self.assertIn(guard.PRODUCER, err)
        return err

    def test_clean_tree_passes(self):
        repo = self.repository()
        code, out, err = repo.run()
        self.assertEqual((code, err), (0, ""))
        self.assertIn("std bundle tracking: PASS", out)

    def test_untracked_build_outputs_pass(self):
        repo = self.repository()
        repo.write("packages/chelis-std/dist/chelis-std-0.4.0.tar.zst", "archive")
        repo.write("crates/chelis-std-bundle/dist/chelis-std-0.4.0.chb", "shell")
        repo.write("packages/chelis-std/reef.lock", BUNDLED_LOCK)
        self.assertEqual(repo.run()[0], 0)

    def test_tracked_lock_without_a_bundled_runtime_passes(self):
        repo = self.repository()
        repo.track("examples/probe/reef.lock", PATH_ONLY_LOCK)
        self.assertEqual(repo.run()[0], 0)

    def test_similarly_named_files_pass(self):
        repo = self.repository()
        repo.track("packages/chelis-std/distribution.md", "notes")
        repo.track("docs/reef.lock.md", BUNDLED_LOCK)
        self.assertEqual(repo.run()[0], 0)

    def test_tracked_package_dist_is_refused(self):
        repo = self.repository()
        repo.track("packages/chelis-std/dist/chelis-std-0.4.0.tar.zst", "archive")
        self.assert_refused(repo, "packages/chelis-std/dist/chelis-std-0.4.0.tar.zst")

    def test_tracked_bundle_dist_is_refused(self):
        repo = self.repository()
        repo.track("crates/chelis-std-bundle/dist/chelis-std-0.4.0.chb", "shell")
        self.assert_refused(repo, "crates/chelis-std-bundle/dist/chelis-std-0.4.0.chb")

    def test_tracked_bundled_lock_is_refused(self):
        repo = self.repository()
        repo.track("crates/chelis-cli/tests/fixtures/stage/reef.lock", BUNDLED_LOCK)
        err = self.assert_refused(repo, "crates/chelis-cli/tests/fixtures/stage/reef.lock")
        self.assertIn("bundled chelis-std", err)

    def test_root_level_bundled_lock_is_refused(self):
        repo = self.repository()
        repo.track("reef.lock", BUNDLED_LOCK)
        self.assert_refused(repo, "reef.lock")

    def test_unreadable_tracked_lock_is_refused(self):
        repo = self.repository()
        repo.track("examples/probe/reef.lock", "[[dependencies]\nname =")
        err = self.assert_refused(repo, "examples/probe/reef.lock")
        self.assertIn("cannot be read as a reef lock", err)

    def test_index_bytes_are_checked_not_the_working_tree(self):
        repo = self.repository()
        repo.track("examples/probe/reef.lock", BUNDLED_LOCK)
        repo.write("examples/probe/reef.lock", PATH_ONLY_LOCK)
        self.assert_refused(repo, "examples/probe/reef.lock")

    def test_every_finding_is_reported(self):
        repo = self.repository()
        repo.track("packages/chelis-std/dist/a.chb", "shell")
        repo.track("crates/chelis-std-bundle/dist/b.tar.zst", "archive")
        repo.track("examples/probe/reef.lock", BUNDLED_LOCK)
        err = self.assert_refused(repo, "packages/chelis-std/dist/a.chb")
        self.assertIn("crates/chelis-std-bundle/dist/b.tar.zst", err)
        self.assertIn("examples/probe/reef.lock", err)

    def test_staged_warning_reports_and_exits_zero(self):
        repo = self.repository()
        repo.track("packages/chelis-std/dist/a.chb", "shell")
        code, out, err = repo.run("--staged", "--warn")
        self.assertEqual(code, 0)
        self.assertNotIn("FAIL", out)
        self.assertIn("std bundle tracking: warning", err)
        self.assertIn("packages/chelis-std/dist/a.chb", err)

    def test_staged_mode_ignores_committed_files(self):
        repo = self.repository()
        repo.track("packages/chelis-std/dist/a.chb", "shell")
        repo.git("commit", "-q", "-m", "commit the generated file")
        code, out, err = repo.run("--staged", "--warn")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(repo.run()[0], 1)


if __name__ == "__main__":
    unittest.main()
