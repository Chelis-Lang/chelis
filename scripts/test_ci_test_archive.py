"""Archive reuse preserves test selection and rejects the wrong build."""

import tempfile
import json
import os
import subprocess
import tomllib
import builtins
import runpy
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml

from scripts import ci_test_archive as archive
from scripts import gate


class ArchiveTests(unittest.TestCase):
    def test_gate_bootstrap_import_does_not_require_python_311_modules(self):
        original = builtins.__import__
        def before_handoff(name, *args, **kwargs):
            if name == "tomllib":
                raise ModuleNotFoundError("tomllib is unavailable before the gate handoff")
            return original(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=before_handoff):
            runpy.run_path(str(Path(archive.__file__)))

    def test_current_cargo_artifacts_own_staticlib_inclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            current = deps / "libchelis_runtime-current.a"
            current.write_bytes(b"current")
            (deps / "libchelis_runtime-stale.a").write_bytes(b"stale")
            message = {"reason": "compiler-artifact", "target": {
                "name": "chelis_runtime", "crate_types": ["staticlib", "rlib"]},
                "filenames": [str(current)]}
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps(message))
            with patch.object(archive, "ROOT", root), \
                    patch.dict(os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), \
                    patch.object(archive.subprocess, "run", return_value=result):
                config = tomllib.loads(archive.archive_config("workspace"))
                self.assertEqual(config["profile"]["default"]["archive"]["include"], [{
                    "path": "debug/deps/libchelis_runtime-current.a", "relative-to": "target",
                    "on-missing": "error"}])
                # A cached file alone never supplies authority for the archive.
                result.stdout = ""
                with self.assertRaisesRegex(ValueError, "did not report"):
                    archive.archive_config("workspace")
                result.stdout = json.dumps(message)
                current.unlink()
                with self.assertRaisesRegex(ValueError, "Missing runtime"):
                    archive.archive_config("workspace")

    def test_package_and_binary_selection_survive_archive_reuse(self):
        command = ["cargo", "nextest", "run", "-p", "chelis-deep", "-p", "chelis-types",
                   "--test", "stamp_to_typed", "--test", "typed_metadata", "--no-fail-fast"]
        result = archive.reuse_command(command, Path("build.tar.zst"))
        expression = result[result.index("-E") + 1]
        self.assertIn("package(=chelis-deep) | package(=chelis-types)", expression)
        self.assertIn("binary(=stamp_to_typed) | binary(=typed_metadata)", expression)
        self.assertNotIn("-p", result)
        self.assertNotIn("--test", result)
        self.assertIn("--no-fail-fast", result)

    def test_filter_and_hash_partition_are_preserved(self):
        command = ["cargo", "nextest", "run", "--workspace", "--profile", "ci",
                   "--partition", "hash:1/2", "-E", "test(=probe)"]
        result = archive.reuse_command(command, Path("build.tar.zst"))
        for token in ("ci", "hash:1/2", "test(=probe)"):
            self.assertIn(token, result)
        self.assertNotIn("--workspace", result)

    def test_unknown_build_options_cannot_silently_change_configuration(self):
        for option in ("--features", "--all-features", "--release", "--no-default-features"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                archive.reuse_command(["cargo", "nextest", "run", option], Path("build.tar.zst"))

    def test_manifest_rejects_other_head_configuration_host_or_checkout(self):
        current = {"head": "a" * 40, "configuration": "workspace", "system": "Linux",
                   "machine": "x86_64", "workspace": "/work/chelis", "nextest": "0.9.136"}
        archive.check_identity(current, current)
        for key in current:
            with self.subTest(key=key), self.assertRaises(ValueError):
                archive.check_identity(current, {**current, key: "different"})

    def test_archive_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar.zst"
            path.write_bytes(b"original")
            digest = archive.digest(path)
            self.assertEqual(archive.digest(path), digest)
            path.write_bytes(b"corrupt")
            self.assertNotEqual(archive.digest(path), digest)

    def test_support_slices_are_exactly_the_existing_full_support(self):
        selections = [gate.selected_stage_commands("integration", tests_only=False,
                      support_only=True, partition=None, support_slice=part)
                      for part in ("frontend", "domain")]
        self.assertEqual(selections[0] + selections[1], gate.STAGES["integration"][1:])
        self.assertEqual(len(selections[0]), 2)
        self.assertEqual(len(selections[1]), 1)

    def test_support_slice_requires_support_only(self):
        with self.assertRaises(SystemExit):
            gate.parse_args(["integration", "--tests-only", "--support-slice", "frontend"])

    def test_explicit_bad_archive_cannot_fall_back_to_compilation(self):
        with self.assertRaises(ValueError):
            archive.from_environment({archive.ARCHIVE_ENV: ""})
        with patch.object(archive, "verify", side_effect=ValueError("wrong head")):
            with self.assertRaisesRegex(ValueError, "wrong head"):
                archive.from_environment({archive.ARCHIVE_ENV: "build.tar.zst"})

    def test_frontend_support_uses_archive_for_all_three_rust_selections(self):
        from scripts import compiler_front_end_performance as frontend
        calls = []
        def runner(command, **kwargs):
            from subprocess import CompletedProcess
            calls.append(command)
            return CompletedProcess(command, 0)
        with patch.object(frontend.ci_test_archive, "from_environment", return_value=Path("build.tar.zst")):
            frontend.run_oracle(runner=runner, environment={})
        self.assertEqual(len(calls), 4)
        for command in calls[1:]:
            self.assertIn("--archive-file", command)
            self.assertNotIn("-p", command)
        self.assertIn("only", calls[-1])
        self.assertIn("--test-threads", calls[-1])
        self.assertIn("1", calls[-1])

    def test_gate_archive_selector_retains_profile_and_partition(self):
        commands = gate.selected_stage_commands("integration", tests_only=True,
                    support_only=False, partition="hash:2/2", test_archive=Path("build.tar.zst"))
        self.assertEqual(len(commands), 1)
        self.assertIn("ci", commands[0])
        self.assertIn("hash:2/2", commands[0])
        self.assertIn("--archive-file", commands[0])


def check_workflow(jobs):
    for configuration, producer, consumer in (
        ("workspace", "workspace-test-build", "workspace-tests-shard"),
        ("generalization", "generalization-test-build", "generalize-sweep-oracle-shard"),
    ):
        assert producer in jobs[consumer]["needs"], "consumer must depend on its producer"
        assert jobs[producer]["needs"] == ["changes"]
        for job in (producer, consumer):
            assert jobs[job]["runs-on"] == "ubuntu-latest"
            installs = [s for s in jobs[job]["steps"] if s.get("name") == "Install cargo-nextest"]
            assert installs[0]["with"]["tool"] == "cargo-nextest@0.9.136"
        uploads = [s for s in jobs[producer]["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
        downloads = [s for s in jobs[consumer]["steps"] if s.get("uses", "").startswith("actions/download-artifact")]
        assert uploads[0]["with"]["name"] == downloads[0]["with"]["name"] == f"linux-test-build-{configuration}"
        assert "run-id" not in downloads[0]["with"], "archives must come from the current run"
        assert "github-token" not in downloads[0]["with"]
        assert uploads[0]["with"]["if-no-files-found"] == "error"
        runs = [s.get("run", "") for s in jobs[producer]["steps"]]
        assert any(f"create --configuration {configuration}" in cmd for cmd in runs)


class WorkflowTests(unittest.TestCase):
    def jobs(self):
        return yaml.safe_load((archive.ROOT / ".github/workflows/ci.yml").read_text())["jobs"]

    def test_two_configurations_build_once_for_their_existing_shards(self):
        check_workflow(self.jobs())

    def test_missing_dependency_cannot_race_the_upload(self):
        jobs = self.jobs()
        jobs["workspace-tests-shard"]["needs"] = ["changes"]
        with self.assertRaisesRegex(AssertionError, "depend on its producer"):
            check_workflow(jobs)

    def test_other_run_cannot_supply_an_archive(self):
        jobs = self.jobs()
        download = next(s for s in jobs["workspace-tests-shard"]["steps"]
                        if s.get("uses", "").startswith("actions/download-artifact"))
        download["with"]["run-id"] = "123"
        with self.assertRaisesRegex(AssertionError, "current run"):
            check_workflow(jobs)


if __name__ == "__main__":
    unittest.main()
