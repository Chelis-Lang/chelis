"""Regression tests for disposable Python wheel smoke inputs."""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SMOKE_PATH = Path(__file__).with_name("python_wheel_smoke.py")
SMOKE_SPEC = importlib.util.spec_from_file_location("python_wheel_smoke_test", SMOKE_PATH)
if SMOKE_SPEC is None or SMOKE_SPEC.loader is None:
    raise AssertionError(f"cannot load wheel smoke at {SMOKE_PATH}")
SMOKE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE)


class SourceCopyTests(unittest.TestCase):
    def test_editable_source_is_refreshed_at_the_same_build_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-source-") as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            stamp = checkout / "stamp.txt"
            with patch.object(SMOKE, "REPO_ROOT", checkout), patch.dict(
                os.environ, {"CARGO_TARGET_DIR": ""}
            ):
                target = SMOKE.target_directory()
                for value in ("first", "second"):
                    stamp.write_text(value, encoding="utf-8")
                    with SMOKE.temporary_smoke_directory(target, checkout) as scratch:
                        source = SMOKE.editable_source_directory(target, scratch)
                        SMOKE.copy_source_tree(source)
                        self.assertEqual(
                            (source / "stamp.txt").read_text(encoding="utf-8"), value
                        )
                        self.assertFalse(source.is_relative_to(checkout))
                        if value == "first":
                            original_path = source
                        else:
                            self.assertEqual(source, original_path)

    def test_editable_source_rejects_overlapping_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-overlap-") as temporary:
            target = Path(temporary) / "cargo"
            target.mkdir()
            with self.assertRaises(SMOKE.SmokeError):
                SMOKE.editable_source_directory(target, target)

    def test_source_copy_rejects_links_that_escape_its_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-links-") as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            outside = base / "outside"
            outside.write_text("not part of source copy", encoding="utf-8")
            (source / "external-link").symlink_to(outside)

            with self.assertRaisesRegex(SMOKE.SmokeError, "escapes its disposable root"):
                SMOKE.validate_source_links(source)

    def test_source_copy_accepts_links_within_its_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-links-") as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            payload = source / "payload"
            payload.write_text("copied", encoding="utf-8")
            (source / "payload-link").symlink_to(payload)

            SMOKE.validate_source_links(source)

    def test_consumer_path_removes_global_chelis_executables(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-path-") as temporary:
            base = Path(temporary)
            checkout = base / "checkout"
            tools = base / "tools"
            compiler_dir = base / "compiler"
            for directory in (checkout, tools, compiler_dir):
                directory.mkdir()
            (tools / "chelis").write_text("not executed", encoding="utf-8")
            compiler = compiler_dir / "clang"
            compiler.write_text("not executed", encoding="utf-8")
            for executable in (tools / "chelis", compiler):
                executable.chmod(0o755)

            with patch.dict(os.environ, {"PATH": str(tools)}):
                consumer_path = SMOKE.checkout_free_path(
                    checkout_root=checkout,
                    compiler=compiler,
                    consumer_cwd=base,
                )

            self.assertIsNone(shutil.which("chelis", path=consumer_path))
            self.assertEqual(shutil.which("clang", path=consumer_path), str(compiler))

    def test_smoke_directory_is_stable_and_removed_between_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-target-") as temporary:
            target = Path(temporary) / "cargo"
            common_root = SMOKE.REPO_ROOT.parent

            with SMOKE.temporary_smoke_directory(target, common_root) as first:
                self.assertTrue(first.is_dir())
                (first / "payload").write_text("task-owned", encoding="utf-8")

            self.assertFalse(first.exists())
            with SMOKE.temporary_smoke_directory(target, common_root) as second:
                self.assertEqual(second, first)

    def test_developer_target_is_absent_during_execution_and_restored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-wheel-target-") as temporary:
            target = Path(temporary) / "cargo"
            target.mkdir()
            (target / "artifact").write_text("owned", encoding="utf-8")
            with SMOKE.without_developer_target(target):
                self.assertFalse(target.exists())
            self.assertEqual((target / "artifact").read_text(encoding="utf-8"), "owned")
            with self.assertRaises(RuntimeError):
                with SMOKE.without_developer_target(target):
                    raise RuntimeError("consumer failed")
            self.assertEqual((target / "artifact").read_text(encoding="utf-8"), "owned")


if __name__ == "__main__":
    unittest.main()
