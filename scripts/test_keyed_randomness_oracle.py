"""#2413 acceptance receipts: no missing, ignored, skipped, or failed witness passes.

Spec-derived obligations: [05-RNG-1/2], [04-LIN-9/10], spec/06 replay
and key rows, and spec/07's typed #2503 refusal. These tests precede the
runner implementation and exercise its evidence boundary without compiling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

from scripts import keyed_randomness_oracle as oracle


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / "target/agents/2413-keyed-oracle"
        self.target.mkdir(parents=True)
        config = self.root / ".config/nextest.toml"
        config.parent.mkdir()
        config.write_text('[profile.ci-full.junit]\npath = "junit.xml"\n')
        self.binary = self.target / "probe"
        self.binary.write_text("fixture")
        self.suite = oracle.Suite("chelis-types", "probe", ("good", "negative"), ("checker",))
        self.expected = {"chelis-types::probe::good", "chelis-types::probe::negative"}
        self.xml = self.root / "junit.xml"

    def listing(self):
        return {
            "rust-build-meta": {"target-directory": str(self.target)},
            "rust-suites": {"chelis-types::probe": {
                "binary-id": "chelis-types::probe", "package-name": "chelis-types",
                "status": "listed", "binary-path": str(self.binary),
                "cwd": str(self.root / "crates/chelis-types"),
                "testcases": {name: {"ignored": False, "filter-match": {"status": "matches"}}
                              for name in self.suite.tests},
            }},
        }

    def validate(self, packet):
        return oracle.validate_listing(packet, (self.suite,), self.root, self.target)

    def junit(self, names=("good", "negative"), child=""):
        self.xml.write_text('<testsuites><testsuite>' + ''.join(
            f'<testcase classname="chelis-types::probe" name="{name}">{child}</testcase>'
            for name in names) + '</testsuite></testsuites>')
        return self.xml

    def test_exact_listing_and_terminal_passes_are_accepted(self):
        self.assertEqual(self.validate(self.listing()), self.expected)
        self.assertEqual(oracle.validate_junit(self.junit(), self.expected), self.expected)

    def test_missing_or_ignored_or_filtered_required_identity_is_refused(self):
        for mutation in ("missing", "ignored", "filtered", "malformed"):
            packet = self.listing()
            cases = packet["rust-suites"]["chelis-types::probe"]["testcases"]
            if mutation == "missing":
                del cases["negative"]
            elif mutation == "ignored":
                cases["negative"]["ignored"] = True
            elif mutation == "filtered":
                cases["negative"]["filter-match"]["status"] = "mismatch"
            else:
                cases["negative"]["ignored"] = "false"
            with self.subTest(mutation=mutation), self.assertRaises(oracle.OracleFailure):
                self.validate(packet)

    def test_empty_wrong_or_extra_selected_suite_cannot_substitute_for_required_one(self):
        for suites in ({}, {"wrong": {}}, {**self.listing()["rust-suites"], "other": {}}):
            packet = self.listing()
            packet["rust-suites"] = suites
            with self.subTest(suites=suites), self.assertRaises(oracle.OracleFailure):
                self.validate(packet)

    def test_additional_tests_are_unselected_not_silent_replacements(self):
        packet = self.listing()
        cases = packet["rust-suites"]["chelis-types::probe"]["testcases"]
        cases["extra"] = {"ignored": False, "filter-match": {"status": "mismatch"}}
        self.assertEqual(self.validate(packet), self.expected)
        cases["extra"]["filter-match"]["status"] = "matches"
        with self.assertRaises(oracle.OracleFailure):
            self.validate(packet)

    def test_foreign_target_checkout_or_binary_is_refused(self):
        for field in ("target-directory", "cwd", "binary-path", "binary-id", "status"):
            packet = self.listing()
            if field == "target-directory":
                packet["rust-build-meta"][field] = "/foreign"
            else:
                packet["rust-suites"]["chelis-types::probe"][field] = "/foreign"
            with self.subTest(field=field), self.assertRaises(oracle.OracleFailure):
                self.validate(packet)

    def test_every_nonpassing_junit_outcome_is_refused_even_with_exit_zero(self):
        for tag in ("failure", "error", "skipped", "rerunFailure", "rerunError", "flakyFailure", "flakyError"):
            with self.subTest(tag=tag), self.assertRaises(oracle.OracleFailure):
                oracle.validate_junit(self.junit(child=f"<{tag}/>"), self.expected)

    def test_missing_duplicate_extra_or_malformed_execution_is_refused(self):
        for names in ((), ("good",), ("good", "good"), ("good", "negative", "extra")):
            with self.subTest(names=names), self.assertRaises(oracle.OracleFailure):
                oracle.validate_junit(self.junit(names), self.expected)
        self.xml.write_text("not xml")
        with self.assertRaises(oracle.OracleFailure):
            oracle.validate_junit(self.xml, self.expected)

    def test_retirement_scan_accepts_current_shape_and_refuses_old_dispatch_markers(self):
        for package in ("chelis-ir", "chelis-compiler-api"):
            folder = self.root / "crates" / package / "src"
            folder.mkdir(parents=True)
            (folder / "lib.rs").write_text("pub fn run() {}\n")
        oracle.check_retirement(self.root)
        for marker in oracle.RETIRED:
            path = self.root / "crates/chelis-ir/src/lib.rs"
            path.write_text(f"fn {marker}() {{}}\n")
            with self.subTest(marker=marker), self.assertRaises(oracle.OracleFailure):
                oracle.check_retirement(self.root)

    def test_retirement_scan_refuses_missing_source_roots(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_retirement(self.root)

    def test_command_pins_exact_binary_and_test_names_and_overrides_default_filter(self):
        config = self.root / "isolated.toml"
        command = oracle.nextest_command("list", (self.suite,), config)
        self.assertIn("--ignore-default-filter", command)
        self.assertIn("--message-format", command)
        self.assertEqual(command[command.index("--config-file") + 1], str(config))
        expression = command[command.index("-E") + 1]
        self.assertIn("binary_id(=chelis-types::probe)", expression)
        self.assertIn("test(=negative)", expression)
        run = oracle.nextest_command("run", (self.suite,), config)
        self.assertIn("--no-fail-fast", run)
        self.assertEqual(run[run.index("--retries") + 1], "0")

    def test_receipt_binds_selected_executions_to_unchanged_clean_sha(self):
        self.assertEqual(self.execute()["status"], "pass")
        failed = self.execute(end_sha="b" * 40)
        self.assertEqual(failed["status"], "fail")
        self.assertIn("changed", failed["error"])

    def test_real_catalog_commands_enable_required_target_features(self):
        for package in dict.fromkeys(suite.package for suite in oracle.SUITES):
            suites = tuple(suite for suite in oracle.SUITES if suite.package == package)
            manifest = tomllib.loads((oracle.ROOT / "crates" / package / "Cargo.toml").read_text())
            targets = {target["name"]: target for target in manifest.get("test", [])}
            for action in ("list", "run"):
                with self.subTest(package=package, action=action):
                    command = oracle.nextest_command(action, suites, self.root / "isolated.toml")
                    if package == "chelis-compiler-api":
                        self.assertIn("--features", command)
                        self.assertEqual(command[command.index("--features") + 1],
                                         "chelis-compiler-api/ownership-ledger")
                        enabled = {"ownership-ledger"}
                    else:
                        self.assertNotIn("--features", command)
                        enabled = set()
                    for suite in suites:
                        required = set(targets.get(suite.target, {}).get("required-features", []))
                        self.assertLessEqual(required, enabled, suite.binary)

    def test_build_list_and_run_lock_the_dependency_graph(self):
        self.execute()
        self.assertEqual(len(self.last_commands), 3)
        for command in self.last_commands:
            with self.subTest(command=command):
                self.assertIn("--locked", command)

    def test_nextest_junit_store_is_isolated_and_fail_closed_on_config_drift(self):
        config = oracle.isolated_nextest_config(self.root, self.root, self.target)
        settings = tomllib.loads(config.read_text())
        self.assertEqual(settings["store"]["dir"], str(self.target / "nextest"))
        self.assertEqual(settings["profile"]["ci-full"]["junit"]["path"], "junit.xml")
        source = self.root / ".config/nextest.toml"
        source.write_text(source.read_text() + '\n[store]\ndir = "target/nextest"\n')
        with self.assertRaisesRegex(oracle.OracleFailure, "receipt routing"):
            oracle.isolated_nextest_config(self.root, self.root, self.target)

    def execute(self, *, end_sha="a" * 40, failure=None, missing_junit=False, failed_case=False):
        evidence = self.root / f"evidence-{len(list(self.root.glob('evidence-*')))}"
        evidence.mkdir()
        env = {"CARGO_TARGET_DIR": str(self.target)}
        self.last_commands = []
        def command(argv, root, environment, output, label, timeout):
            self.last_commands.append(argv)
            if failure == label and not failed_case:
                raise oracle.OracleFailure("command failed")
            if "list" in argv:
                return json.dumps(self.listing())
            if "run" in argv and not missing_junit:
                destination = self.target / "nextest/ci-full/junit.xml"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(self.junit(child="<failure/>" if failed_case else "").read_text())
                if failure == label:
                    raise oracle.OracleFailure("command failed")
            return ""
        with mock.patch.object(oracle, "candidate_sha", side_effect=["a" * 40, end_sha]), \
             mock.patch.object(oracle, "check_retirement"), \
             mock.patch.object(oracle, "run_command", side_effect=command):
            return oracle.execute(self.root, evidence, env, 10, (self.suite,))

    def test_nonzero_child_and_stale_junit_never_produce_pass_receipt(self):
        for label in ("build", "chelis-types-list", "chelis-types-run"):
            with self.subTest(label=label):
                self.assertEqual(self.execute(failure=label)["status"], "fail")
        stale = self.target / "nextest/ci-full/junit.xml"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(self.junit().read_text())
        self.assertEqual(self.execute(missing_junit=True)["status"], "fail")

    def test_failed_executions_are_not_reported_as_unrun(self):
        receipt = self.execute(failure="chelis-types-run", failed_case=True)
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(set(receipt["failed"]), self.expected)
        self.assertEqual(receipt["passed"], [])
        self.assertEqual(receipt["unrun"], [])

    def test_hung_child_is_terminated_and_logged(self):
        with self.assertRaisesRegex(oracle.OracleFailure, "timeout"):
            oracle.run_command([sys.executable, "-c", "import time; time.sleep(60)"],
                               self.root, os.environ.copy(), self.root, "hung", 0.05)
        self.assertTrue((self.root / "hung.stderr").is_file())

    def test_clean_sha_is_required(self):
        outputs = [mock.Mock(stdout="a" * 40), mock.Mock(stdout=" M changed.rs\n")]
        with mock.patch.object(oracle.subprocess, "run", side_effect=outputs):
            with self.assertRaisesRegex(oracle.OracleFailure, "clean committed"):
                oracle.candidate_sha(self.root)

    def test_catalog_is_unique_and_includes_future_tensor_target_and_par_refusal(self):
        identities = [identity for suite in oracle.SUITES for identity in suite.identities()]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertIn("chelis-compiler-api::key_tensor_forms::tensor_key_forms_match_the_independent_scalar_reference_in_eval_and_c", identities)
        self.assertIn("chelis-types::jit_par_passthrough::par_is_rejected_with_the_typed_issue_fence", identities)


if __name__ == "__main__":
    unittest.main()
