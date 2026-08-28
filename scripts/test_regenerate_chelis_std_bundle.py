"""Unit tests for the chelis-std shipped-artifact regeneration pipeline."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "regenerate_chelis_std_bundle", here / "regenerate_chelis_std_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regen = _load_module()


class RuntimeLockRegenerationTests(unittest.TestCase):
    def make_package(self, root: Path) -> Path:
        package = root / "chelis-std"
        (package / "src").mkdir(parents=True)
        (package / "dist").mkdir()
        (package / "reef.toml").write_text("[package]\nname = \"chelis-std\"\n")
        (package / "reef.lock").write_text("stale lock\n")
        (package / "src/test.ch").write_text("module Std.Test\n")
        (package / "dist/artifact.chb").write_text("final artifact\n")
        return package

    def test_lock_staging_excludes_generated_lock_and_dist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = self.make_package(root)
            staged = root / "staged"

            regen.stage_runtime_package_for_lock(package, staged)

            self.assertEqual(
                (staged / "reef.toml").read_text(),
                (package / "reef.toml").read_text(),
            )
            self.assertEqual(
                (staged / "src/test.ch").read_text(),
                (package / "src/test.ch").read_text(),
            )
            self.assertFalse((staged / "reef.lock").exists())
            self.assertFalse((staged / "dist").exists())

    def test_runtime_lock_refresh_keeps_the_final_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = self.make_package(root)
            artifact = package / "dist/artifact.chb"
            before = artifact.read_bytes()

            def fake_run(command, *, cwd):
                self.assertEqual(command[:3], ["/tmp/chelis", "reef", "build"])
                staged = Path(command[3])
                self.assertNotEqual(staged, package)
                (staged / "reef.lock").write_text("generated final lock\n")
                (staged / "dist").mkdir()
                (staged / "dist/incidental.chb").write_text("discard me\n")
                return SimpleNamespace(returncode=0)

            with mock.patch.object(regen.subprocess, "run", side_effect=fake_run):
                regen.regenerate_runtime_lock(
                    Path("/tmp/chelis"), package, repository_root=root
                )

            self.assertEqual((package / "reef.lock").read_text(), "generated final lock\n")
            self.assertEqual(artifact.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
