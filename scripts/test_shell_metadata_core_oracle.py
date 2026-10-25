#!/usr/bin/env python3
"""Test evidence rejection and immutable baseline isolation."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import shell_metadata_core_oracle as oracle
from no_std_core_support import OracleFailure


class EvidenceTests(unittest.TestCase):
    def metadata(self) -> dict:
        return {
            "format_version": 8,
            "package": {"name": "demo", "version": "1.2.3"},
            "compiler": "=1.2.3",
            "modules": [{
                "module": "Demo",
                "exports": [{
                    "name": "value", "kind": "Value", "type_repr": "(t-var {} t0)",
                    "type_variable_restrictions": [{"variable": "t0", "domain": {"active_set": ["f32"]}}],
                    "collection_obligations": [{"float_classify": {"kind": "nan", "operand": "(t-var {} t0)", "result": "(t-var {} t0)"}}],
                    "effects": [], "has_body": False,
                }],
            }],
            "dependencies": [{"name": "dep", "version": ""}],
            "archive_sha256": "a" * 64,
        }

    def test_every_metadata_model_requires_complete_fields(self) -> None:
        complete = self.metadata()
        oracle.validate_metadata_evidence(complete)
        paths = (
            (), ("package",), ("dependencies", 0), ("modules", 0),
            ("modules", 0, "exports", 0),
            ("modules", 0, "exports", 0, "type_variable_restrictions", 0),
            ("modules", 0, "exports", 0, "type_variable_restrictions", 0, "domain"),
            ("modules", 0, "exports", 0, "collection_obligations", 0, "float_classify"),
        )
        for path in paths:
            model = complete
            for key in path:
                model = model[key]
            for field in model:
                incomplete = deepcopy(complete)
                target = incomplete
                for key in path:
                    target = target[key]
                del target[field]
                with self.subTest(path=path, field=field), self.assertRaises(OracleFailure):
                    oracle.validate_metadata_evidence(incomplete)

    def test_metadata_evidence_rejects_invalid_containers_and_scalars(self) -> None:
        export = ("modules", 0, "exports", 0)
        mutations = (
            (("modules",), None), (("dependencies",), ""), (("format_version",), True),
            (("modules", 0, "exports"), None), ((*export, "kind"), None),
            ((*export, "type_repr"), {}), ((*export, "has_body"), 1),
            ((*export, "effects"), [None]), ((*export, "type_variable_restrictions"), None),
            ((*export, "type_variable_restrictions", 0, "domain"), None),
            ((*export, "collection_obligations"), None),
            ((*export, "collection_obligations", 0), {}),
            ((*export, "collection_obligations", 0, "float_classify", "operand"), None),
            ((*export, "collection_obligations", 0, "float_classify", "kind"), "unknown"),
        )
        for path, value in mutations:
            incomplete = self.metadata()
            target = incomplete
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(OracleFailure):
                oracle.validate_metadata_evidence(incomplete)

    def test_snapshot_requires_one_complete_structured_record(self) -> None:
        for output in ("", "SHELL_SNAPSHOT {}\nSHELL_SNAPSHOT {}\n", "SHELL_SNAPSHOT {\n", "SHELL_SNAPSHOT []\n"):
            with self.subTest(output=output), self.assertRaises(OracleFailure):
                oracle.parse_record(output, "SHELL_SNAPSHOT ")

    def test_unrelated_command_output_is_not_evidence(self) -> None:
        with self.assertRaises(OracleFailure):
            oracle.parse_record('command passed\n{"cases": {}}\n', "SHELL_SNAPSHOT ")

    def test_missing_case_inventory_fails_before_comparison(self) -> None:
        for record in ({}, {"cases": {}, "wire": {}, "bundle": {}}, {"cases": [], "wire": {}, "bundle": {}}):
            with self.subTest(record=record), self.assertRaises(OracleFailure):
                oracle.compare_snapshots(record, record)

    def test_filtered_or_unexpected_case_inventory_fails(self) -> None:
        names = oracle.case_inventory()
        for selected in (names - {"native-sample"}, names | {"unexpected"}):
            record = {"cases": dict.fromkeys(selected, {}), "wire": {}, "bundle": {}}
            with self.subTest(count=len(selected)), self.assertRaises(OracleFailure):
                oracle.validate_snapshot(record)


class BaselineTests(unittest.TestCase):
    def git(self, directory: Path, *arguments: str) -> str:
        result = subprocess.run(("git", *arguments), cwd=directory, text=True, capture_output=True, check=True)
        return result.stdout.strip()

    def repository(self, directory: Path, *, complete: bool) -> tuple[Path, str]:
        root = directory / "source"
        root.mkdir()
        self.git(root, "init", "--quiet")
        files = {
            "crates/chelis-shell/src/lib.rs": b"pub fn original_validation() {}\n",
            "crates/chelis-shell/Cargo.toml": b'[package]\nname = "chelis-shell"\nversion = "0.0.0"\n',
            "Cargo.lock": b"version = 4\n",
            "packages/chelis-std/reef.toml": b'[package]\nname = "chelis-std"\n',
        }
        if not complete:
            del files["crates/chelis-shell/src/lib.rs"]
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.git(root, "add", ".")
        self.git(root, "-c", "user.name=Oracle", "-c", "user.email=oracle@example.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Baseline fixture")
        return root, self.git(root, "rev-parse", "HEAD")

    def test_baseline_uses_committed_source_not_candidate_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, revision = self.repository(directory, complete=True)
            source = root / "crates/chelis-shell/src/lib.rs"
            original = source.read_bytes()
            source.write_bytes(b"pub fn changed_candidate_validation() {}\n")
            with patch.object(oracle, "ROOT", root), patch.object(oracle, "BASELINE", revision), contextlib.redirect_stdout(io.StringIO()):
                baseline = oracle.checkout_baseline(directory / "baseline")
            self.assertEqual((baseline / "crates/chelis-shell/src/lib.rs").read_bytes(), original)
            self.assertNotEqual(source.read_bytes(), original)
            self.assertEqual(self.git(baseline, "status", "--porcelain"), "")

    def test_missing_baseline_in_complete_clone_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, _ = self.repository(directory, complete=True)
            with patch.object(oracle, "ROOT", root), patch.object(oracle, "BASELINE", "f" * 40), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(OracleFailure, "absent from a complete clone"):
                    oracle.checkout_baseline(directory / "baseline")

    def test_missing_original_source_fails_without_candidate_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, revision = self.repository(directory, complete=False)
            with patch.object(oracle, "ROOT", root), patch.object(oracle, "BASELINE", revision), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OracleFailure, "missing original baseline evidence"):
                    oracle.checkout_baseline(directory / "baseline")


class FrozenBundleTests(unittest.TestCase):
    def bundle(self, directory: Path, *, compiler: str = "=0.21.1"):
        directory.mkdir()
        shell = ("shell " + compiler).encode()
        archive = ("archive " + compiler).encode()
        (directory / "runtime.chb").write_bytes(shell)
        (directory / "runtime.tar.zst").write_bytes(archive)
        metadata = EvidenceTests().metadata()
        metadata["compiler"] = compiler
        metadata["archive_sha256"] = hashlib.sha256(archive).hexdigest()
        evidence = {
            "shell_sha256": hashlib.sha256(shell).hexdigest(),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "shell": metadata,
        }
        return oracle.FrozenBundle(directory, evidence, metadata["package"].copy(), compiler)

    def test_release_versions_retain_distinct_exact_input_bytes_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = self.bundle(root / "old")
            new = self.bundle(root / "new", compiler="=0.22.0")
            oracle.validate_frozen_bundle(old)
            oracle.validate_frozen_bundle(new)
            self.assertNotEqual(old.evidence, new.evidence)
            self.assertEqual(old.evidence["shell"]["compiler"], "=0.21.1")
            self.assertEqual(new.evidence["shell"]["compiler"], "=0.22.0")

    def test_frozen_inputs_reject_byte_digest_and_declared_identity_tampering(self) -> None:
        for changed in ("runtime.chb", "runtime.tar.zst", "shell_sha256", "archive_sha256", "compiler", "name", "version"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                bundle = self.bundle(Path(temporary) / "input")
                if changed.startswith("runtime."):
                    with (bundle.directory / changed).open("ab") as output:
                        output.write(b"tampered")
                elif changed in ("shell_sha256", "archive_sha256"):
                    bundle.evidence[changed] = "0" * 64
                elif changed == "compiler":
                    bundle.evidence["shell"][changed] = "=0.22.0"
                else:
                    bundle.evidence["shell"]["package"][changed] = "wrong"
                with self.assertRaises(OracleFailure):
                    oracle.validate_frozen_bundle(bundle)

    def test_both_implementations_receive_each_identical_frozen_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline = directory / "baseline"
            old = self.bundle(directory / "old")
            new = self.bundle(directory / "new", compiler="=0.22.0")
            records = [{"bundle": bundle.evidence, "wire": {}} for bundle in (old, old, new, new)]
            with patch.object(oracle, "freeze_bundle", side_effect=(old, new)) as freeze, \
                 patch.object(oracle, "snapshot", side_effect=records) as snapshot, \
                 patch.object(oracle, "compare_snapshots") as compare, \
                 contextlib.redirect_stdout(io.StringIO()):
                oracle.compare_bundle_inputs(baseline, directory)
            self.assertEqual(freeze.call_args_list, [
                unittest.mock.call(baseline, directory / "baseline-input", candidate=False),
                unittest.mock.call(oracle.ROOT, directory / "candidate-input", candidate=True),
            ])
            self.assertEqual(snapshot.call_args_list, [
                unittest.mock.call(baseline, candidate=False, bundle=old),
                unittest.mock.call(oracle.ROOT, candidate=True, bundle=old),
                unittest.mock.call(baseline, candidate=False, bundle=new),
                unittest.mock.call(oracle.ROOT, candidate=True, bundle=new),
            ])
            self.assertEqual(compare.call_args_list, [
                unittest.mock.call(records[0], records[1]),
                unittest.mock.call(records[2], records[3]),
            ])

    def test_snapshot_rejects_output_not_bound_to_the_frozen_input(self) -> None:
        for path, value in (
            (("shell_sha256",), "0" * 64),
            (("archive_sha256",), "0" * 64),
            (("shell", "archive_sha256"), "0" * 64),
            (("shell", "compiler"), "=0.22.0"),
            (("shell", "modules", 0, "exports", 0, "name"), "wrong"),
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as temporary:
                bundle = self.bundle(Path(temporary) / "input")
                changed = deepcopy(bundle.evidence)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                output = "SHELL_SNAPSHOT " + json.dumps({"bundle": changed})
                result = subprocess.CompletedProcess([], 0, output, "")
                with patch.object(oracle, "consumer", return_value=result), \
                     self.assertRaisesRegex(OracleFailure, "frozen input"):
                    oracle.snapshot(oracle.ROOT, candidate=True, bundle=bundle)


MANDATORY_TESTS = (
    "EvidenceTests.test_snapshot_requires_one_complete_structured_record",
    "EvidenceTests.test_unrelated_command_output_is_not_evidence",
    "EvidenceTests.test_missing_case_inventory_fails_before_comparison",
    "EvidenceTests.test_filtered_or_unexpected_case_inventory_fails",
    "EvidenceTests.test_every_metadata_model_requires_complete_fields",
    "EvidenceTests.test_metadata_evidence_rejects_invalid_containers_and_scalars",
    "BaselineTests.test_baseline_uses_committed_source_not_candidate_changes",
    "BaselineTests.test_missing_baseline_in_complete_clone_fails",
    "BaselineTests.test_missing_original_source_fails_without_candidate_fallback",
    "FrozenBundleTests.test_release_versions_retain_distinct_exact_input_bytes_and_identity",
    "FrozenBundleTests.test_frozen_inputs_reject_byte_digest_and_declared_identity_tampering",
    "FrozenBundleTests.test_both_implementations_receive_each_identical_frozen_pair",
    "FrozenBundleTests.test_snapshot_rejects_output_not_bound_to_the_frozen_input",
)


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed: set[str] = set()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.add(test.id().removeprefix("__main__."))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    expected = set(MANDATORY_TESTS)
    if not result.wasSuccessful() or result.passed != expected or result.skipped:
        return 1
    print("SHELL_ORACLE_TESTS " + json.dumps({"passed": sorted(result.passed), "skipped": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
