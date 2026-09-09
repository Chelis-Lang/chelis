"""Execution-receipt, source-freshness, discovery, and continuous-gate controls."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts import dtype_builtin_atom_closure_oracle as oracle


class ExecutionTests(unittest.TestCase):
    def test_exact_execution_passes(self):
        oracle.verify_execution(["a"], [{"id": "a", "outcome": "passed"}])

    def test_zero_skip_failure_duplicate_and_missing_execution_fail(self):
        for selected, executed in [
            ([], []), (["a"], []), (["a", "a"], [{"id": "a", "outcome": "passed"}]),
            (["a"], [{"id": "a", "outcome": "skipped"}]),
            (["a"], [{"id": "a", "outcome": "failed"}]),
            (["a"], [{"id": "b", "outcome": "passed"}]),
            (["a"], [{"id": "a", "outcome": "passed"}] * 2),
        ]:
            with self.subTest(selected=selected, executed=executed), self.assertRaises(oracle.OracleError):
                oracle.verify_execution(selected, executed)

    def test_compiled_sources_union_only_after_duplicate_checks(self):
        packet = {"builtins": ["Numeric:add:TableA"], "risc": ["Numeric:add:TableA"]}
        self.assertEqual(oracle.discover(packet), ["Numeric:add:TableA"])
        for changed in ({}, {**packet, "risc": []}, {**packet, "builtins": packet["builtins"] * 2}):
            with self.assertRaises(oracle.OracleError):
                oracle.discover(changed)

    def test_nextest_ignored_or_zero_selection_is_not_a_pass(self):
        packet = {"rust-suites": {"types::unit": {"testcases": {
            "test": {"ignored": False, "filter-match": {"status": "matches"}}}}}}
        self.assertEqual(oracle.nextest_selection(packet), ["rust:types::unit::test"])
        packet["rust-suites"]["types::unit"]["testcases"]["test"]["ignored"] = True
        with self.assertRaises(oracle.OracleError):
            oracle.nextest_selection(packet)
        with self.assertRaises(oracle.OracleError):
            oracle.nextest_selection({"rust-suites": {}})

    def test_junit_skips_failures_and_stale_names_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            for child in ("<skipped/>", "<failure/>", "<error/>", "<rerunFailure/>"):
                path.write_text(f'<testsuites><testsuite name="types"><testcase name="a">{child}</testcase></testsuite></testsuites>')
                with self.assertRaises(oracle.OracleError):
                    oracle.junit_execution(path, ["rust:types::a"])
            path.write_text('<testsuites><testsuite name="types"><testcase name="a"/></testsuite></testsuites>')
            self.assertEqual(oracle.junit_execution(path, ["rust:types::a"]),
                             [{"id": "rust:types::a", "outcome": "passed"}])
            with self.assertRaises(oracle.OracleError):
                oracle.junit_execution(path, ["rust:types::other"])

    def test_framework_failure_never_produces_success_receipt(self):
        import io
        for action in (lambda: self.fail("negative control"), lambda: self.skipTest("negative control")):
            case = oracle.Case("probe", "negative", action)
            result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=oracle.Receipts).run(case)
            self.assertEqual(result.executed, [])
            with self.assertRaises(oracle.OracleError):
                oracle.verify_execution([case.id()], result.executed)


class SourceTests(unittest.TestCase):
    def test_committed_bytes_and_modes_own_the_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
            git("init", "-q")
            git("config", "user.name", "Oracle fixture")
            git("config", "user.email", "oracle@example.invalid")
            source = root / "source"
            source.write_text("committed\n")
            git("add", "source")
            git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")
            head, digest = oracle.source_identity(root)
            self.assertEqual((len(head), len(digest)), (40, 64))
            git("update-index", "--assume-unchanged", "source")
            source.write_text("mutated\n")
            with self.assertRaisesRegex(oracle.OracleError, "bytes differ"):
                oracle.source_identity(root)
            source.write_text("committed\n")
            source.chmod(0o755)
            with self.assertRaisesRegex(oracle.OracleError, "mode differs"):
                oracle.source_identity(root)
            source.chmod(0o644)
            (root / "untracked").write_text("must not enter evidence")
            with self.assertRaisesRegex(oracle.OracleError, "clean"):
                oracle.source_identity(root)


class ContinuousGateTests(unittest.TestCase):
    def test_integration_support_stage_owns_the_oracle(self):
        from scripts import gate
        command = [gate.MANAGED_PYTHON, "scripts/dtype_builtin_atom_closure_oracle.py"]
        self.assertEqual(gate.STAGES["integration"].count(command), 1)

    def test_registry_change_cannot_skip_the_oracle(self):
        from scripts import ci_detect_docs_only as routing
        for path in ("spec/05-risc-primitives.md", "spec/registry/builtin_semantic_identities.md"):
            self.assertFalse(routing.is_docs_only([path]), path)
        self.assertTrue(routing.is_docs_only(["docs/book/src/intro.md"]))

    def test_nested_nextest_uses_a_distinct_receipt(self):
        import tomllib
        config = tomllib.loads((oracle.ROOT / ".config/nextest.toml").read_text())
        self.assertEqual(config["profile"][oracle.PROFILE]["junit"]["path"], "junit.xml")
        self.assertNotIn(oracle.PROFILE, ("ci", "ci-full"))


if __name__ == "__main__":
    unittest.main()
