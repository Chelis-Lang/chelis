"""Unit tests for the chelis-std shipped-artifact regeneration pipeline."""

import importlib.util
import hashlib
import os
import subprocess
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    def test_owned_output_inventory_is_complete_and_exact(self):
        repo = regen.repo_root()
        version = regen.chelis_std_version(repo)
        relative = {
            path.relative_to(repo).as_posix()
            for path in regen.owned_generated_outputs(repo, version)
        }
        self.assertEqual(
            relative,
            {
                f"packages/chelis-std/dist/chelis-std-{version}.tar.zst",
                f"packages/chelis-std/dist/chelis-std-{version}.chb",
                f"crates/chelis-std-bundle/dist/chelis-std-{version}.tar.zst",
                f"crates/chelis-std-bundle/dist/chelis-std-{version}.chb",
                "packages/chelis-std/reef.lock",
            },
        )


class RealGeneratorFixedPointTests(unittest.TestCase):
    """Executable regression for the canonical, repository-owning pipeline."""

    def test_two_real_debug_regenerations_reach_a_byte_fixed_point(self):
        repo = regen.repo_root()
        version = regen.chelis_std_version(repo)
        outputs = regen.owned_generated_outputs(repo, version)
        before = {path: path.read_bytes() for path in outputs}
        command = [sys.executable, str(repo / "scripts/regenerate_chelis_std_bundle.py"), "--debug"]
        env = os.environ.copy()

        try:
            snapshots = []
            for iteration in (1, 2):
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"real generator pass {iteration} failed:\n{completed.stdout}\n{completed.stderr}",
                )
                snapshots.append({path.relative_to(repo): _sha256(path) for path in outputs})

            self.assertEqual(
                snapshots[0],
                snapshots[1],
                "two unchanged invocations of the supported generator must emit identical bytes",
            )
        finally:
            for path, contents in before.items():
                path.write_bytes(contents)


if __name__ == "__main__":
    unittest.main()
