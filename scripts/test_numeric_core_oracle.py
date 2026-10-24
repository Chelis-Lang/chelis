#!/usr/bin/env python3
"""Failure controls for complete numeric acceptance evidence."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numeric_core_oracle as oracle
from no_std_core_support import OracleFailure


def costs():
    return [{"operation": op, "elements": n, "allocations": 1,
        "allocated_bytes": n * 8, "nanoseconds": n}
        for n in oracle.SCALES for op in oracle.COST_OPERATIONS]


class NumericCoreOracleTests(unittest.TestCase):
    def test_phase_evidence_is_an_exact_disjoint_partition(self):
        groups = list(oracle.PHASE_EVIDENCE.values())
        self.assertEqual(set(oracle.PHASE_EVIDENCE), {"extraction", "surface-suites", "surface-controls"})
        self.assertEqual(set().union(*groups), oracle.EVIDENCE)
        self.assertEqual(sum(map(len, groups)), len(oracle.EVIDENCE))
        for phase, required in oracle.PHASE_EVIDENCE.items():
            oracle.require_evidence(set(required), phase)
            for missing in required:
                with self.subTest(phase=phase, missing=missing):
                    with self.assertRaisesRegex(OracleFailure, "missing evidence"):
                        oracle.require_evidence(set(required) - {missing}, phase)
            with self.assertRaises(OracleFailure):
                oracle.require_evidence(set(oracle.EVIDENCE), phase)

    def test_default_preserves_combined_inventory_setup_and_order_while_phases_are_disjoint(self):
        surface = [(('surface',), {'surface-case'})] * len(oracle.SURFACE_SUITES)
        numeric = [(('numeric',), {'numeric-case'})]
        for selected in (None, *oracle.PHASE_EVIDENCE):
            calls = []
            def record(name, value):
                def execute(*args, **kwargs):
                    calls.append(name)
                    return value
                return execute
            partial = set(oracle.PHASE_EVIDENCE['extraction']) - {'numeric-suites', 'python-controls'}
            with self.subTest(selected=selected), \
                 patch.object(oracle, 'run', side_effect=record('command', None)), \
                 patch.object(oracle, 'workspace_environment', return_value={}), \
                 patch.object(oracle, 'baseline_clone', side_effect=record('baseline', Path('baseline'))), \
                 patch.object(oracle, 'extraction_checks', side_effect=record('comparisons', partial)), \
                 patch.object(oracle, 'preserved_suite_plan', side_effect=record('surface-inventory', surface[0])), \
                 patch.object(oracle, 'numeric_suite_plan', side_effect=record('numeric-inventory', numeric)), \
                 patch.object(oracle, 'run_suites', side_effect=lambda suites: calls.append('surface-run' if suites == surface else 'numeric-run')), \
                 patch.object(oracle, 'surface_controls', side_effect=record('surface-controls', {'surface-controls'})), \
                 patch.object(oracle, 'python_tests', side_effect=record('python-controls', None)), \
                 patch('builtins.print'):
                oracle.run_oracle(selected)
                expected = {
                    None: ['command', 'baseline', 'comparisons', 'command',
                        *(['surface-inventory'] * len(oracle.SURFACE_SUITES)),
                        'numeric-inventory', 'command', 'surface-run', 'surface-controls',
                        'numeric-run', 'python-controls', 'python-controls'],
                    'extraction': ['command', 'baseline', 'comparisons', 'command',
                        'numeric-inventory', 'command', 'numeric-run', 'python-controls', 'python-controls'],
                    'surface-suites': ['command', 'baseline',
                        *(['surface-inventory'] * len(oracle.SURFACE_SUITES)), 'command', 'surface-run'],
                    'surface-controls': ['command', 'surface-controls'],
                }
                self.assertEqual(calls, expected[selected])

    def test_failed_or_incomplete_phase_cannot_report_success(self):
        for result in (set(), OracleFailure("failed control")):
            with self.subTest(result=result), patch.object(oracle, "run"), \
                 patch.object(oracle, "surface_controls", side_effect=result if isinstance(result, Exception) else None,
                              return_value=result), patch("builtins.print") as output:
                with self.assertRaises(OracleFailure):
                    oracle.run_oracle("surface-controls")
                self.assertFalse(any("PASS" in str(call) for call in output.call_args_list))

    def test_unknown_phase_is_rejected_before_execution(self):
        with patch.object(oracle, "run") as run:
            with self.assertRaisesRegex(OracleFailure, "unknown"):
                oracle.run_oracle("missing")
            run.assert_not_called()

    def test_complete_feature_coverage(self):
        oracle.require_features(set(oracle.FEATURES))
        for feature in oracle.FEATURES:
            with self.subTest(feature=feature):
                with self.assertRaisesRegex(OracleFailure, "feature coverage"):
                    oracle.require_features(set(oracle.FEATURES) - {feature})

    def test_each_evidence_group_is_required(self):
        oracle.require_evidence(set(oracle.EVIDENCE))
        for group in oracle.EVIDENCE:
            with self.subTest(group=group):
                with self.assertRaisesRegex(OracleFailure, "missing evidence"):
                    oracle.require_evidence(set(oracle.EVIDENCE) - {group})

    def test_empty_and_duplicate_records_fail(self):
        for text in ("", "NUMERIC {}\nNUMERIC {}\n", "NUMERIC invalid\n", "NUMERIC []\n"):
            with self.subTest(text=text):
                with self.assertRaises(OracleFailure):
                    oracle.record(text, "NUMERIC", {"value"})

    def test_missing_record_field_fails(self):
        with self.assertRaisesRegex(OracleFailure, "fields"):
            oracle.record('NUMERIC {"value":1}\n', "NUMERIC", {"value", "wire"})
        self.assertEqual(oracle.record('NUMERIC {"value":1}\n', "NUMERIC", {"value"}), {"value": 1})

    def test_exact_comparison_rejects_changed_or_absent_fixtures(self):
        oracle.compare_exact({"integer": [1, 2]}, {"integer": [1, 2]}, "wire")
        for candidate in ({}, {"integer": [1, 3]}):
            with self.assertRaisesRegex(OracleFailure, "wire"):
                oracle.compare_exact({"integer": [1, 2]}, candidate, "wire")

    def test_comparison_identifies_changed_and_missing_nested_fields(self):
        expected = {"wide-headers": {"wide.h": {"symbol": "wide"}}}
        for actual, message in (
            ({"wide-headers": {"wide.h": {"symbol": "other"}}},
             "wide-headers.*wide.h.*symbol.*expected.*wide.*actual.*other"),
            ({"wide-headers": {}}, "wide-headers.*wide.h.*missing"),
            ({**expected, "extra": "value"}, "extra.*unexpected"),
        ):
            with self.subTest(actual=actual):
                with self.assertRaisesRegex(OracleFailure, message):
                    oracle.compare_exact(expected, actual, "generated C")
        with self.assertRaisesRegex(OracleFailure, "empty expected"):
            oracle.compare_exact({}, {}, "generated C")

    def test_comparison_identifies_sequence_element_and_long_header_difference(self):
        with self.assertRaisesRegex(OracleFailure, r"wire.*bytes.*\[1\].*expected 2.*actual 3"):
            oracle.compare_exact({"bytes": [1, 2]}, {"bytes": [1, 3]}, "wire")
        with self.assertRaisesRegex(OracleFailure, "header.*character 1000.*expected.*before.*actual.*after"):
            oracle.compare_exact({"header": "x" * 1000 + "before"},
                {"header": "x" * 1000 + "after"}, "generated C")

    @patch.object(oracle, "workspace_environment", return_value={})
    def test_generated_abi_requires_every_validated_fixture(self, _environment):
        artifacts = {"wide": {"source": "source bytes", "header": "header bytes"}}
        valid = {"wide": {"program_identity": "Numeric.Main", "declarations": {}}}
        def validate(crate, source, **kwargs):
            self.assertEqual(json.loads(Path(kwargs["env"]["CHELIS_NUMERIC_ARTIFACTS"]).read_text()), artifacts)
            return subprocess.CompletedProcess([], 0, "GENERATED_ABI " + json.dumps(valid), "")
        with patch.object(oracle, "consumer", side_effect=validate):
            self.assertEqual(oracle.generated_abi_snapshot(artifacts), valid)
        for result in ("", "GENERATED_ABI {}", "GENERATED_ABI {\"other\":{}}"):
            with patch.object(oracle, "consumer", return_value=subprocess.CompletedProcess([], 0, result, "")):
                with self.assertRaisesRegex(OracleFailure, "GENERATED_ABI"):
                    oracle.generated_abi_snapshot(artifacts)
        with patch.object(oracle, "consumer", side_effect=OracleFailure("wide: source digest disagrees")):
            with self.assertRaisesRegex(OracleFailure, "source digest disagrees"):
                oracle.generated_abi_snapshot(artifacts)
        with self.assertRaisesRegex(OracleFailure, "empty generated"):
            oracle.generated_abi_snapshot({})

    @patch.object(oracle, "workspace_environment", return_value={})
    def test_generated_snapshot_requires_program_and_both_runtime_headers(self, _environment):
        for missing in ("wide.h", "chelis_runtime.h", "chelis_blas.h"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                def execute(command, **kwargs):
                    if "--output" in command:
                        output = Path(command[command.index("--output") + 1])
                        (output / "wide.c").write_text("source")
                        for name in {"wide.h", "chelis_runtime.h", "chelis_blas.h"} - {missing}:
                            (output / name).write_text("header")
                    return subprocess.CompletedProcess([], 0, "9007199254740993\nout = ()\n", "")
                with patch.object(oracle, "toolchain", return_value={"compiler": "cc", "compile_flags": [], "link_flags": []}), patch.object(oracle, "run", side_effect=execute):
                    with self.assertRaisesRegex(OracleFailure, "wide.*missing required.*" + missing):
                        oracle.generated_snapshot(oracle.ROOT, Path(temporary))

    def test_cost_measurements_are_required(self):
        valid = costs()
        oracle.inspect_costs(valid)
        for field in ("operation", "allocations", "allocated_bytes", "nanoseconds"):
            invalid = [dict(row) for row in valid]
            invalid[0].pop(field)
            with self.assertRaises(OracleFailure):
                oracle.inspect_costs(invalid)
        with self.assertRaisesRegex(OracleFailure, "cost coverage"):
            oracle.inspect_costs(valid[:-1])
        for invalid in (None, [1, 2, 3], [dict(row, nanoseconds=0) for row in valid],
            [dict(row, allocations=True) for row in valid]):
            with self.assertRaises(OracleFailure):
                oracle.inspect_costs(invalid)

    def test_per_element_allocation_regression_fails(self):
        baseline = costs()
        candidate = [dict(row, allocations=row["elements"]) for row in baseline]
        with self.assertRaisesRegex(OracleFailure, "allocation"):
            oracle.compare_costs(baseline, candidate)

    def test_requested_byte_regression_fails(self):
        baseline = costs()
        candidate = [dict(row, allocated_bytes=row["allocated_bytes"] + 1) for row in baseline]
        with self.assertRaisesRegex(OracleFailure, "allocation"):
            oracle.compare_costs(baseline, candidate)

    def test_each_tensor_operation_is_required(self):
        for operation in oracle.COST_OPERATIONS:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(OracleFailure, "coverage"):
                    oracle.inspect_costs([row for row in costs() if row["operation"] != operation])

    def test_real_metadata_control_must_reject(self):
        with patch.object(oracle.support, "_normal_packages", return_value={oracle.CRATE, "chelis-vocab"}):
            with self.assertRaisesRegex(OracleFailure, "unexpectedly passed"):
                oracle.expect_graph_rejection({}, "std")

    def test_metadata_rejection_must_have_correct_reason(self):
        with patch.object(oracle.support, "_normal_packages", side_effect=OracleFailure("missing graph")):
            with self.assertRaisesRegex(OracleFailure, "another reason"):
                oracle.expect_graph_rejection({}, "std")

    def test_crmath_is_the_only_additional_numeric_target_owner(self):
        accepted = {oracle.CRATE, "chelis-vocab", "chelis-crmath", "chelis-no-std-consumer"}
        with patch.object(oracle.support, "_normal_packages", return_value=accepted):
            self.assertEqual(oracle.inspect_graph({}), accepted)
        for name in ("chelis-abi", "chelis-types", "chelis-new-host-service", "reef-runtime"):
            with self.subTest(name=name):
                with patch.object(oracle.support, "_normal_packages", return_value=accepted | {name}):
                    with self.assertRaisesRegex(OracleFailure, "forbidden"):
                        oracle.inspect_graph({})

    def test_numeric_owner_and_vocabulary_are_required(self):
        for names in ({oracle.CRATE}, {"chelis-vocab"}):
            with patch.object(oracle.support, "_normal_packages", return_value=names):
                with self.assertRaisesRegex(OracleFailure, "omits"):
                    oracle.inspect_graph({})

    def test_runtime_midpoint_mutation_fails(self):
        with self.assertRaisesRegex(OracleFailure, "runtime"):
            oracle.compare_exact({"rounding": [0x7a73, 0x3f81]},
                {"rounding": [0x7a74, 0x3f80]}, "runtime")

    def test_runtime_mutation_requires_numeric_failure(self):
        with patch.object(oracle.support, "_package"), patch.object(oracle.support, "_execute") as execute:
            execute.return_value = subprocess.CompletedProcess([], 101, "", "direct runtime rounding")
            oracle.runtime_mutation_rejected()
            execute.return_value = subprocess.CompletedProcess([], 0, "", "")
            with self.assertRaisesRegex(OracleFailure, "unexpectedly passed"):
                oracle.runtime_mutation_rejected()
            execute.return_value = subprocess.CompletedProcess([], 101, "", "cannot compile dependency")
            with self.assertRaisesRegex(OracleFailure, "another reason"):
                oracle.runtime_mutation_rejected()

    def test_baseline_has_exact_immutable_identity(self):
        with patch.object(oracle, "run") as run, patch.object(oracle.support, "_execute") as execute:
            execute.return_value = subprocess.CompletedProcess([], 0, "", "")
            run.return_value = subprocess.CompletedProcess([], 0, "wrong-revision\n", "")
            with self.assertRaisesRegex(OracleFailure, "identity"):
                oracle.baseline_clone(Path("/isolated"))
            run.return_value = subprocess.CompletedProcess([], 0, oracle.BASELINE_REVISION + "\n", "")
            self.assertEqual(oracle.baseline_clone(Path("/isolated")), Path("/isolated/baseline"))

    def test_missing_baseline_fetch_is_mandatory(self):
        with patch.object(oracle, "run") as run, patch.object(oracle.support, "_execute") as execute:
            execute.return_value = subprocess.CompletedProcess([], 1, "", "absent")
            run.side_effect = [subprocess.CompletedProcess([], 0, "", ""), OracleFailure("fetch failed")]
            with self.assertRaisesRegex(OracleFailure, "fetch failed"):
                oracle.baseline_clone(Path("/isolated"))

    def test_missing_surface_receipt_fails(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(oracle, "run", return_value=result):
            with self.assertRaisesRegex(OracleFailure, "surface"):
                oracle.surface_controls()

    def test_python_evidence_needs_every_named_case(self):
        required = ("module.Tests.mandatory",)
        for value in (
            {"selected": [], "passed": [], "skipped": []},
            {"selected": [required[0]], "passed": [], "skipped": []},
            {"selected": [required[0]], "passed": [required[0]], "skipped": [required[0]]},
            {"selected": ["other"], "passed": ["other"], "skipped": []},
            {"selected": [required[0]], "passed": [required[0], required[0]], "skipped": []},
        ):
            with self.subTest(value=value):
                result = subprocess.CompletedProcess([], 0, "PYTHON " + json.dumps(value), "")
                with patch.object(oracle, "run", return_value=result):
                    with self.assertRaisesRegex(OracleFailure, "Python"):
                        oracle.python_tests(("module",), required)

    def test_failed_command_sets_nonzero_exit(self):
        with patch.object(oracle, "run_oracle", side_effect=OracleFailure("missing evidence")):
            self.assertEqual(oracle.main(), 1)


if __name__ == "__main__":
    unittest.main()
