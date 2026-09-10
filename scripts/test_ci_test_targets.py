"""Selection must bound compilation, preserve units, and reject silent omissions."""
import copy
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

import yaml
from unittest import mock

from scripts import ci_test_targets as targets
from scripts import gate


def metadata():
    return {"workspace_members": ["p"], "packages": [{
        "id": "p", "name": "p", "features": {"default": []},
        "targets": [
            {"name": "p", "kind": ["lib"], "test": True},
            {"name": "app", "kind": ["bin"], "test": True},
            {"name": "smoke", "kind": ["test"], "test": True},
            {"name": "exhaustive", "kind": ["test"], "test": True},
        ],
    }]}


def listing():
    return {"rust-suites": {name: {"testcases": {
        "positive": {"ignored": False, "filter-match": {"status": "matches"}},
        "manual": {"ignored": True, "filter-match": {"status": "mismatch"}},
    }} for name in ("p", "p::bin/app", "p::smoke")}}


class TargetSelectionTests(unittest.TestCase):
    def test_selects_all_units_and_only_named_integration_targets(self):
        args = targets.cargo_args(metadata(), [("p", "smoke")])
        self.assertEqual(args, ["--workspace", "--lib", "--bins", "--test", "smoke"])
        self.assertNotIn("--tests", args)
        self.assertNotIn("--all-targets", args)
        targets.validate_listing(listing(), metadata(), [("p", "smoke")])

    def test_missing_duplicate_or_ambiguous_target_is_rejected(self):
        cases = [([], metadata()), ([("p", "missing")], metadata()),
                 ([("p", "smoke"), ("p", "smoke")], metadata()),
                 ([("missing", "smoke")], metadata())]
        duplicate = metadata()
        sibling = copy.deepcopy(duplicate["packages"][0])
        sibling.update(id="q", name="q")
        duplicate["packages"].append(sibling)
        duplicate["workspace_members"].append("q")
        cases.append(([("p", "smoke")], duplicate))
        for selection, data in cases:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                targets.cargo_args(data, selection)

    def test_manifest_requires_nonempty_exact_package_target_rows(self):
        for content in ("", "version = 1", "version = 2\ntarget = []", "version = 1\ntarget = []",
                        'version = 1\n[[target]]\npackage = "p"\nname = "smoke"\nextra = true',
                        'version = 1\n[[target]]\npackage = ""\nname = "smoke"',
                        "invalid [["):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "selection.toml"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    targets.read_targets(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.toml"
            path.write_text('version = 1\n[[target]]\npackage = "p"\nname = "smoke"\n')
            self.assertEqual(targets.read_targets(path), [("p", "smoke")])

    def test_receipts_reject_missing_units_extra_binaries_and_filtered_tests(self):
        for mutation in ("missing-lib", "missing-bin", "missing-test", "extra", "filtered", "empty"):
            data = listing()
            if mutation.startswith("missing"):
                del data["rust-suites"][{"missing-lib": "p", "missing-bin": "p::bin/app", "missing-test": "p::smoke"}[mutation]]
            elif mutation == "extra":
                data["rust-suites"]["p::exhaustive"] = {"testcases": {}}
            elif mutation == "filtered":
                data["rust-suites"]["p::smoke"]["testcases"]["positive"]["filter-match"]["status"] = "mismatch"
            else:
                data["rust-suites"]["p::smoke"]["testcases"] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                targets.validate_listing(data, metadata(), [("p", "smoke")])

    def test_feature_gated_units_follow_declared_default_features(self):
        data = metadata()
        data["packages"][0]["targets"][1]["required-features"] = ["probe"]
        expected = listing()
        del expected["rust-suites"]["p::bin/app"]
        targets.validate_listing(expected, data, [("p", "smoke")])
        data["packages"][0]["features"] = {"default": ["probe"], "probe": []}
        targets.validate_listing(listing(), data, [("p", "smoke")])

    def test_new_unlisted_integration_is_automatically_outside_fast_lane(self):
        data = metadata()
        data["packages"][0]["targets"].append({"name": "new_heavy", "kind": ["test"], "test": True})
        self.assertEqual(targets.cargo_args(data, [("p", "smoke")]), targets.cargo_args(metadata(), [("p", "smoke")]))
        targets.validate_listing(listing(), data, [("p", "smoke")])

    def test_ci_fast_is_separate_from_full_and_manual_gate_commands(self):
        self.assertEqual(gate.parse_args(["ci-fast"]).stage, "ci-fast")
        self.assertNotIn("ci-fast", gate.STAGE_ORDER)
        self.assertEqual(gate.selected_stage_commands("ci-fast", tests_only=False, support_only=False, partition=None),
                         [[gate.MANAGED_PYTHON, "scripts/ci_test_targets.py"]])
        self.assertIn(gate.NEXTEST_WORKSPACE_CI, gate.STAGES["integration"])

    def test_missing_target_or_bad_compile_receipt_stops_before_execution(self):
        for failure in ("missing-target", "extra-binary", "build-failure"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / ".config").mkdir()
                name = "absent" if failure == "missing-target" else "smoke"
                (root / ".config/ci-test-targets.toml").write_text(
                    f'version = 1\n[[target]]\npackage = "p"\nname = "{name}"\n')
                calls = []
                def run(command, **kwargs):
                    calls.append(command)
                    if failure == "build-failure" and command[1] == "build":
                        raise subprocess.CalledProcessError(1, command)
                    payload = metadata() if command[1] == "metadata" else listing()
                    if command[1:3] == ["nextest", "list"]:
                        payload["rust-suites"]["p::exhaustive"] = {"testcases": {}}
                    return mock.Mock(stdout=json.dumps(payload), returncode=0)
                with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), mock.patch.object(targets.subprocess, "run", side_effect=run):
                    with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                        targets.run(root)
                self.assertFalse(any(c[1:3] == ["nextest", "run"] for c in calls))
                if failure == "missing-target":
                    self.assertEqual(len(calls), 1)
                else:
                    self.assertTrue((root / "target/ci-fast/timing.json").is_file())

    def test_pr_workflow_has_one_bounded_worker_and_no_heavy_backdoor(self):
        root = Path(__file__).resolve().parents[1]
        jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
        self.assertNotIn("workspace-tests-shard", jobs)
        self.assertNotIn("workspace-tests", jobs)
        worker = jobs["ci-fast"]
        self.assertNotIn("strategy", worker)
        self.assertEqual(worker["timeout-minutes"], 20)
        commands = [s.get("run", "") for s in worker["steps"]]
        self.assertIn("python3 scripts/gate.py ci-fast", commands)
        self.assertNotIn("cargo nextest run --workspace", str(jobs))
        self.assertEqual(jobs["integration"]["needs"], ["changes", "ci-fast"])
        self.assertIn("ci-fast=${{ needs.ci-fast.result }}", str(jobs["integration"]))
        sanitizer = [s.get("run") for s in jobs["backend-sanitizers"]["steps"]]
        self.assertIn("cargo test -p chelis-backend-c --lib", sanitizer)
        self.assertIn("cargo test -p chelis-backend-c --doc", sanitizer)
        self.assertNotIn("cargo test -p chelis-backend-c", sanitizer)

    def test_runner_builds_products_once_then_lists_and_runs_same_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".config").mkdir()
            (root / ".config/ci-test-targets.toml").write_text('version = 1\n[[target]]\npackage = "p"\nname = "smoke"\n')
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                payload = metadata() if command[1] == "metadata" else listing()
                return mock.Mock(stdout=json.dumps(payload), returncode=0)
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), mock.patch.object(targets.subprocess, "run", side_effect=run):
                targets.run(root)
            self.assertEqual(calls[1], ["cargo", "build", "--workspace", "--lib", "--bins", "--locked"])
            self.assertEqual(calls[2][0:3], ["cargo", "nextest", "list"])
            self.assertEqual(calls[3][0:3], ["cargo", "nextest", "run"])
            for command in calls[2:]:
                self.assertIn("--ignore-default-filter", command)
                self.assertIn("--lib", command)
                self.assertIn("--bins", command)
                self.assertEqual(command[command.index("--test") + 1], "smoke")
            self.assertTrue((root / "target/ci-fast/selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
