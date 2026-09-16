"""Selection must bound compilation, preserve units, and reject silent omissions."""
import copy
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

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


def shared_metadata():
    data = metadata()
    sibling = copy.deepcopy(data["packages"][0])
    sibling.update(id="q", name="q")
    data["packages"].append(sibling)
    data["workspace_members"].append("q")
    return data


def write_junit(root, binaries, *, skipped=None):
    skipped = skipped or set()
    document = ET.Element("testsuites")
    for binary in binaries:
        suite = ET.SubElement(document, "testsuite", name=binary)
        case = ET.SubElement(
            suite, "testcase", name="positive", classname=binary
        )
        if binary in skipped:
            ET.SubElement(case, "skipped")
    path = root / "target/nextest/ci-fast/junit.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(document).write(path)


class TargetSelectionTests(unittest.TestCase):
    def test_selects_all_units_and_only_named_integration_targets(self):
        groups = targets.cargo_selections(metadata(), [("p", "smoke")])
        self.assertEqual(groups, [["--workspace", "--lib", "--bins", "--test", "smoke"]])
        args = groups[0]
        self.assertNotIn("--tests", args)
        self.assertNotIn("--all-targets", args)
        targets.validate_listing(listing(), metadata(), [("p", "smoke")])

    def test_missing_or_duplicate_identity_is_rejected(self):
        cases = [([], metadata()), ([("p", "missing")], metadata()),
                 ([("p", "smoke"), ("p", "smoke")], metadata()),
                 ([("missing", "smoke")], metadata())]
        duplicate = metadata()
        duplicate["packages"][0]["targets"].append(copy.deepcopy(duplicate["packages"][0]["targets"][2]))
        cases.append(([("p", "smoke")], duplicate))
        for selection, data in cases:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                targets.cargo_selections(data, selection)

    def test_shared_names_are_scoped_to_exact_packages_without_repeating_units(self):
        data = shared_metadata()
        data["packages"][0]["targets"].append({"name": "unique", "kind": ["test"]})
        self.assertEqual(targets.cargo_selections(data, [("p", "unique"), ("p", "smoke"), ("q", "exhaustive")]), [
            ["--workspace", "--lib", "--bins", "--test", "unique"],
            ["-p", "p", "--test", "smoke"],
            ["-p", "q", "--test", "exhaustive"],
        ])
        self.assertEqual(targets.cargo_selections(data, [("q", "smoke")]), [
            ["--workspace", "--lib", "--bins"], ["-p", "q", "--test", "smoke"],
        ])

    def test_listing_union_rejects_duplicate_execution_and_unselected_siblings(self):
        units = listing()
        del units["rust-suites"]["p::smoke"]
        units["rust-suites"].update({name.replace("p", "q", 1): copy.deepcopy(suite)
                                     for name, suite in list(units["rust-suites"].items())})
        selected = {"rust-suites": {"p::smoke": listing()["rust-suites"]["p::smoke"]}}
        merged = targets.merge_listings([units, selected])
        targets.validate_listing(merged, shared_metadata(), [("p", "smoke")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            targets.merge_listings([units, selected, selected])
        merged["rust-suites"]["q::smoke"] = selected["rust-suites"]["p::smoke"]
        with self.assertRaisesRegex(ValueError, "extra"):
            targets.validate_listing(merged, shared_metadata(), [("p", "smoke")])

    def test_integration_required_features_must_be_default_enabled(self):
        data = metadata()
        data["packages"][0]["targets"][2]["required-features"] = ["probe"]
        with self.assertRaisesRegex(ValueError, "default features"):
            targets.cargo_selections(data, [("p", "smoke")])
        data["packages"][0]["features"] = {"default": ["probe"], "probe": []}
        self.assertEqual(targets.cargo_selections(data, [("p", "smoke")]),
                         targets.cargo_selections(metadata(), [("p", "smoke")]))

    def test_manifest_requires_nonempty_exact_package_target_rows(self):
        for content in ("", "version = 1", "version = 2\nstanding_target = []",
                        'version = 2\n[[standing_target]]\npackage = "p"\nname = "smoke"\nextra = true',
                        'version = 2\n[[standing_target]]\npackage = ""\nname = "smoke"',
                        'version = 2\n[[standing_target]]\npackage = "p"\nname = "smoke"\n'
                        '[[target_exclusion]]\npackage = "p"\nname = "heavy"\n',
                        "invalid [["):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "selection.toml"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    targets.read_targets(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.toml"
            path.write_text('version = 2\n[[standing_target]]\npackage = "p"\nname = "smoke"\n')
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
        self.assertEqual(targets.cargo_selections(data, [("p", "smoke")]), targets.cargo_selections(metadata(), [("p", "smoke")]))
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
                    f'version = 2\n[[standing_target]]\npackage = "p"\nname = "{name}"\n')
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
                        targets.run(root, candidate_sha="b" * 40)
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
        self.assertEqual(worker["timeout-minutes"], 30)
        commands = [s.get("run", "") for s in worker["steps"]]
        self.assertIn("python3 scripts/gate.py ci-fast", commands)
        self.assertNotIn("cargo nextest run --workspace", str(jobs))
        self.assertEqual(
            jobs["integration"]["needs"],
            ["changes", "ci-fast", "change-owned-report"],
        )
        self.assertIn("ci-fast=${{ needs.ci-fast.result }}", str(jobs["integration"]))
        self.assertIn(
            "change-owned-report=${{ needs.change-owned-report.result }}",
            str(jobs["integration"]),
        )
        sanitizer = [s.get("run") for s in jobs["backend-sanitizers"]["steps"]]
        self.assertIn("cargo test -p chelis-backend-c --lib", sanitizer)
        self.assertIn("cargo test -p chelis-backend-c --doc", sanitizer)
        self.assertNotIn("cargo test -p chelis-backend-c", sanitizer)

    def test_runner_builds_products_once_then_lists_and_runs_same_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".config").mkdir()
            (root / ".config/ci-test-targets.toml").write_text(
                'version = 2\n[[standing_target]]\npackage = "p"\nname = "smoke"\n')
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                payload = metadata() if command[1] == "metadata" else listing()
                if command[1:3] == ["nextest", "run"]:
                    write_junit(root, listing()["rust-suites"])
                return mock.Mock(stdout=json.dumps(payload), returncode=0)
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), mock.patch.object(targets.subprocess, "run", side_effect=run):
                targets.run(root, candidate_sha="b" * 40)
            self.assertEqual(calls[1], ["cargo", "build", "--workspace", "--lib", "--bins", "--locked"])
            self.assertEqual(calls[2][0:3], ["cargo", "nextest", "list"])
            self.assertEqual(calls[3][0:3], ["cargo", "nextest", "run"])
            for command in calls[2:]:
                self.assertIn("--ignore-default-filter", command)
                self.assertIn("--lib", command)
                self.assertIn("--bins", command)
                self.assertEqual(command[command.index("--test") + 1], "smoke")
            self.assertTrue((root / "target/ci-fast/selection.json").is_file())
            coverage = json.loads(
                (root / "target/ci-fast/coverage.json").read_text()
            )
            self.assertEqual(coverage["candidate_sha"], "b" * 40)
            self.assertEqual(coverage["execution"], targets.STANDING_EXECUTION)
            self.assertEqual(coverage["selected_targets"], ["p::smoke"])
            self.assertEqual(coverage["executed_targets"], ["p::smoke"])
            self.assertEqual(
                coverage["selected_tests"],
                ["p::smoke::positive"],
            )
            self.assertEqual(
                coverage["executed_tests"],
                coverage["selected_tests"],
            )
            targets.ci_change_owned.verify_standing_coverage_digest(coverage)

    def test_standing_receipt_rejects_a_skipped_selected_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".config").mkdir()
            (root / ".config/ci-test-targets.toml").write_text(
                'version = 2\n[[standing_target]]\n'
                'package = "p"\nname = "smoke"\n'
            )

            def run(command, **kwargs):
                payload = (
                    metadata() if command[1] == "metadata" else listing()
                )
                if command[1:3] == ["nextest", "run"]:
                    write_junit(
                        root,
                        listing()["rust-suites"],
                        skipped={"p::smoke"},
                    )
                return mock.Mock(
                    stdout=json.dumps(payload), returncode=0
                )

            with (
                mock.patch.dict(
                    os.environ,
                    {"CARGO_TARGET_DIR": str(root / "target")},
                ),
                mock.patch.object(
                    targets.subprocess, "run", side_effect=run
                ),
                self.assertRaisesRegex(
                    ValueError, "skipped rather than executed"
                ),
            ):
                targets.run(root, candidate_sha="b" * 40)
            coverage = json.loads(
                (root / "target/ci-fast/coverage.json").read_text()
            )
            self.assertFalse(coverage["success"])
            self.assertEqual(coverage["executed_targets"], [])
            self.assertEqual(coverage["executed_tests"], [])
            self.assertIn(
                "skipped rather than executed",
                " ".join(coverage["failures"]),
            )

    def test_group_receipts_preserve_junit_commands_and_timings_and_reject_missing_junit(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / ".config").mkdir()
                (root / ".config/ci-test-targets.toml").write_text(
                    'version = 2\n[[standing_target]]\npackage = "p"\nname = "smoke"\n'
                    '[[standing_target]]\npackage = "q"\nname = "smoke"\n')
                calls = []
                def run(command, **kwargs):
                    calls.append(command)
                    if command[1] == "metadata":
                        return mock.Mock(stdout=json.dumps(shared_metadata()), returncode=0)
                    binaries = (["p", "p::bin/app", "q", "q::bin/app"] if "--workspace" in command
                                else [command[command.index("-p") + 1] + "::smoke"])
                    if command[1:3] == ["nextest", "run"] and not (missing and "q::smoke" in binaries):
                        write_junit(root, binaries)
                    suites = {binary: listing()["rust-suites"]["p::smoke"] for binary in binaries}
                    return mock.Mock(stdout=json.dumps({"rust-suites": suites}), returncode=0)
                with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), mock.patch.object(targets.subprocess, "run", side_effect=run):
                    if missing:
                        with self.assertRaisesRegex(ValueError, "JUnit"):
                            targets.run(root, candidate_sha="b" * 40)
                    else:
                        targets.run(root, candidate_sha="b" * 40)
                self.assertEqual(sum(c[1] == "build" for c in calls), 1)
                junit = ET.parse(root / "target/nextest/ci-fast/junit.xml")
                cases = [case.get("classname") for case in junit.iter("testcase")]
                self.assertEqual(len(cases), len(set(cases)))
                self.assertEqual(set(cases), {"p", "p::bin/app", "q", "q::bin/app", "p::smoke"} | (set() if missing else {"q::smoke"}))
                receipts = root / "target/ci-fast"
                self.assertEqual(len(json.loads((receipts / "commands.json").read_text())), 7)
                self.assertEqual(len(json.loads((receipts / "timing.json").read_text())), 7)
                coverage = json.loads((receipts / "coverage.json").read_text())
                self.assertEqual(coverage["success"], not missing)
                if missing:
                    self.assertTrue(coverage["failures"])
                    self.assertNotEqual(
                        coverage["selected_tests"],
                        coverage["executed_tests"],
                    )


if __name__ == "__main__":
    unittest.main()
