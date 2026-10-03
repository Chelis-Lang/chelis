#!/usr/bin/env python3
"""Unit controls for `check_std_bundle_reproducible.py`.

The cargo builds run in CI's lint-and-unit stage; these tests pin how the
check finds the bundle's build-script output among cargo's messages, how it
asks cargo for a second run of the build script and refuses one that did not
happen, and how it compares the two runs, without running cargo.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_std_bundle_reproducible as check


def executed(package_id: str, out_dir: str) -> str:
    return json.dumps(
        {"reason": "build-script-executed", "package_id": package_id, "out_dir": out_dir}
    )


BUNDLE = "path+file:///w/crates/chelis-std-bundle#0.18.12"


class BundleOutDirTests(unittest.TestCase):
    def test_the_bundle_build_script_output_is_found(self):
        messages = "\n".join(
            [
                executed("registry+https://github.com/rust-lang/crates.io-index#zstd-sys@2.0.15", "/t/zstd"),
                json.dumps({"reason": "compiler-artifact", "package_id": BUNDLE}),
                "warning: not json",
                executed(BUNDLE, "/t/out"),
            ]
        )
        self.assertEqual(check.bundle_out_dir(messages), Path("/t/out"))

    def test_the_legacy_package_id_form_is_found(self):
        messages = executed("chelis-std-bundle 0.18.12 (path+file:///w)", "/t/out")
        self.assertEqual(check.bundle_out_dir(messages), Path("/t/out"))

    def test_a_package_id_naming_the_package_after_the_path_is_found(self):
        messages = executed("path+file:///w/crates/bundle#chelis-std-bundle@0.18.12", "/t/out")
        self.assertEqual(check.bundle_out_dir(messages), Path("/t/out"))

    def test_a_similarly_named_package_is_not_the_bundle(self):
        messages = executed("path+file:///w/crates/chelis-std-bundle-proto#0.1.0", "/t/x")
        with self.assertRaises(check.ReproducibilityError):
            check.bundle_out_dir(messages)

    def test_a_missing_output_is_an_error(self):
        with self.assertRaises(check.ReproducibilityError):
            check.bundle_out_dir("")

    def test_two_outputs_are_an_error(self):
        messages = "\n".join([executed(BUNDLE, "/t/a"), executed(BUNDLE, "/t/b")])
        with self.assertRaises(check.ReproducibilityError):
            check.bundle_out_dir(messages)


class ExportHeadTests(unittest.TestCase):
    """The check builds the committed tree, never the working tree."""

    def test_the_export_holds_committed_bytes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            src = repo / "packages/chelis-std/src"
            src.mkdir(parents=True)
            git = ["git", "-C", str(repo)]
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "config", "user.email", "probe@example.invalid"], check=True)
            subprocess.run([*git, "config", "user.name", "Probe"], check=True)
            (src / "a.ch").write_text("committed\n")
            subprocess.run([*git, "add", "."], check=True)
            subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
            (src / "a.ch").write_text("edited\n")
            (src / "b.ch").write_text("untracked\n")
            exported = check.export_head(repo, Path(tmp) / "export")
            exported_src = exported / "packages/chelis-std/src"
            self.assertEqual((exported_src / "a.ch").read_text(), "committed\n")
            self.assertFalse((exported_src / "b.ch").exists())
            self.assertFalse((exported / ".git").exists())


class BuildTwiceTests(unittest.TestCase):
    """The second run shares the target directory and must rerun the script."""

    def run_twice(self, first_dir: str, second_dir: str) -> list[tuple]:
        outputs = {name: name.encode() for name in check.OUTPUTS}
        calls = []

        def build(source, target_dir, *extra):
            calls.append((source, target_dir, extra))
            return Path(first_dir if len(calls) == 1 else second_dir), dict(outputs)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(check, "build", build):
            target = Path(tmp) / "target"
            (target / "stale").mkdir(parents=True)
            first, second = check.build_twice(Path("/w"), target)
            self.assertFalse((target / "stale").exists(), "the target starts fresh")
        self.assertEqual(first, second)
        return calls

    def test_the_second_run_reuses_the_target_with_the_bundle_option(self):
        calls = self.run_twice("/t/build/a/out", "/t/build/b/out")
        self.assertEqual([call[1] for call in calls], [calls[0][1]] * 2)
        self.assertEqual(calls[0][2], ())
        self.assertEqual(calls[1][2], ("--config", check.SECOND_RUN_CONFIG))
        self.assertTrue(
            check.SECOND_RUN_CONFIG.startswith(f"profile.dev.package.{check.PACKAGE}.")
        )

    def test_a_second_run_that_reports_the_first_out_dir_is_an_error(self):
        with self.assertRaisesRegex(check.ReproducibilityError, "did not make cargo run"):
            self.run_twice("/t/build/a/out", "/t/build/a/out")


class DifferencesTests(unittest.TestCase):
    def outputs(self, **overrides: bytes) -> dict[str, bytes]:
        values = {name: name.encode() for name in check.OUTPUTS}
        values.update({key.replace("_", "."): value for key, value in overrides.items()})
        return values

    def test_identical_builds_have_no_differences(self):
        self.assertEqual(check.differences(self.outputs(), self.outputs()), [])

    def test_each_differing_output_is_named(self):
        first = self.outputs()
        second = dict(first)
        second["chelis-std.chb"] = b"other shell"
        second["chelis-std.version"] = b"9.9.9"
        found = check.differences(first, second)
        self.assertEqual(len(found), 2)
        self.assertTrue(found[0].startswith("chelis-std.chb: "))
        self.assertTrue(found[1].startswith("chelis-std.version: "))


if __name__ == "__main__":
    unittest.main()
