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


class CanonicalBundleEnvironmentTests(unittest.TestCase):
    def test_missing_parent_epoch_becomes_canonical_zero(self):
        with mock.patch.dict(os.environ, {"PATH": "/test/bin"}, clear=True):
            environment = regen.canonical_bundle_environment()

        self.assertEqual(environment["SOURCE_DATE_EPOCH"], "0")
        self.assertEqual(environment["PATH"], "/test/bin")

    def test_nonzero_parent_epoch_is_overridden(self):
        with mock.patch.dict(
            os.environ,
            {"PATH": "/test/bin", "SOURCE_DATE_EPOCH": "315532800"},
            clear=True,
        ):
            environment = regen.canonical_bundle_environment()

        self.assertEqual(environment["SOURCE_DATE_EPOCH"], "0")
        self.assertEqual(environment["PATH"], "/test/bin")


class CargoTargetDirectoryTests(unittest.TestCase):
    def test_default_target_is_under_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(regen.cargo_target_dir(repository), repository / "target")

    def test_absolute_target_stays_absolute(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            target = Path(temporary) / "agent-target"
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(target)}, clear=True):
                self.assertEqual(regen.cargo_target_dir(repository), target)

    def test_relative_target_resolves_from_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": "target/agent"}, clear=True):
                self.assertEqual(
                    regen.cargo_target_dir(repository), repository / "target/agent"
                )


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

            def fake_run(command, *, cwd, env):
                self.assertEqual(command[:3], ["/tmp/chelis", "reef", "build"])
                self.assertEqual(env["SOURCE_DATE_EPOCH"], "0")
                staged = Path(command[3])
                self.assertNotEqual(staged, package)
                (staged / "reef.lock").write_text("generated final lock\n")
                (staged / "dist").mkdir()
                (staged / "dist/incidental.chb").write_text("discard me\n")
                return SimpleNamespace(returncode=0)

            with (
                mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "315532800"}),
                mock.patch.object(regen.subprocess, "run", side_effect=fake_run),
            ):
                regen.regenerate_runtime_lock(
                    Path("/tmp/chelis"), package, repository_root=root
                )

            self.assertEqual((package / "reef.lock").read_text(), "generated final lock\n")
            self.assertEqual(artifact.read_bytes(), before)

    def test_regeneration_pins_epoch_for_every_build_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            package_dist = repo / "packages/chelis-std/dist"
            bundle_dist = repo / "crates/chelis-std-bundle/dist"
            target_dir = repo / "target/debug"
            package_dist.mkdir(parents=True)
            bundle_dist.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (repo / "packages/chelis-std/reef.toml").write_text(
                '[package]\nname = "chelis-std"\nversion = "0.4.0"\n'
            )
            (target_dir / "chelis").write_bytes(b"test binary")
            (package_dist / "chelis-std-0.4.0.tar.zst").write_bytes(b"archive")
            (package_dist / "chelis-std-0.4.0.chb").write_bytes(b"shell")

            calls = []

            def fake_run(command, *, cwd, env):
                calls.append((command, cwd, env.copy()))
                return SimpleNamespace(returncode=0)

            with (
                mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "315532800"}),
                mock.patch.object(regen.subprocess, "run", side_effect=fake_run),
                mock.patch.object(regen, "regenerate_runtime_lock") as regenerate_lock,
            ):
                self.assertEqual(regen.regenerate(repo, debug=True, show_diff=False), 0)

            self.assertEqual(len(calls), 3)
            for _command, cwd, environment in calls:
                self.assertEqual(cwd, repo)
                self.assertEqual(environment["SOURCE_DATE_EPOCH"], "0")
            regenerate_lock.assert_called_once_with(
                target_dir / "chelis",
                repo / "packages/chelis-std",
                repository_root=repo,
            )

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

    def test_check_fails_closed_when_any_owned_output_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = tuple(root / f"output-{index}" for index in range(5))
            for path in outputs[:-1]:
                path.write_bytes(b"current")

            with (
                mock.patch.object(regen, "chelis_std_version", return_value="0.4.0"),
                mock.patch.object(regen, "owned_generated_outputs", return_value=outputs),
                mock.patch.object(regen, "regenerate") as regenerate,
            ):
                self.assertEqual(regen.check_generated_outputs(root, debug=True), 1)
                regenerate.assert_not_called()

    def test_check_detects_stale_bytes_at_each_owned_output_and_restores_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = tuple(root / f"output-{index}" for index in range(5))
            for path in outputs:
                path.write_bytes(b"committed")

            for stale in outputs:
                def fake_regenerate(_repo, *, debug, show_diff):
                    self.assertTrue(debug)
                    self.assertFalse(show_diff)
                    stale.write_bytes(b"canonical")
                    return 0

                with (
                    mock.patch.object(regen, "chelis_std_version", return_value="0.4.0"),
                    mock.patch.object(regen, "owned_generated_outputs", return_value=outputs),
                    mock.patch.object(regen, "regenerate", side_effect=fake_regenerate),
                ):
                    self.assertEqual(regen.check_generated_outputs(root, debug=True), 1)

                self.assertEqual(stale.read_bytes(), b"committed")

    def test_check_detects_cross_process_nondeterminism_and_restores_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = tuple(root / f"output-{index}" for index in range(5))
            for path in outputs:
                path.write_bytes(b"committed")
            calls = 0

            def fake_regenerate(_repo, *, debug, show_diff):
                nonlocal calls
                calls += 1
                outputs[0].write_bytes(b"committed" if calls == 1 else b"different")
                return 0

            with (
                mock.patch.object(regen, "chelis_std_version", return_value="0.4.0"),
                mock.patch.object(regen, "owned_generated_outputs", return_value=outputs),
                mock.patch.object(regen, "regenerate", side_effect=fake_regenerate),
            ):
                self.assertEqual(regen.check_generated_outputs(root, debug=True), 1)

            for path in outputs:
                self.assertEqual(path.read_bytes(), b"committed")


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
            # Exercise the production comparison against the same freshly
            # generated pair, without another two full compiler/reef builds.
            self.assertEqual(regen.snapshot_violations(snapshots[0], *snapshots), ([], []))
            stale = dict(snapshots[0])
            name = next(iter(stale))
            stale[name] = "deliberately stale"
            self.assertEqual(regen.snapshot_violations(stale, *snapshots), ([name], []))
            self.assertEqual(regen.snapshot_violations(snapshots[0], snapshots[0], stale), ([], [name]))
        finally:
            for path, contents in before.items():
                path.write_bytes(contents)



if __name__ == "__main__":
    unittest.main()
