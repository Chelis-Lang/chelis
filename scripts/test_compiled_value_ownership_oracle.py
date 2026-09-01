#!/usr/bin/env python3
"""Unit and mutation tests for chelis#1286's Phase 0 ownership oracle."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import compiled_value_ownership_oracle as oracle  # noqa: E402


class ManifestContractTests(unittest.TestCase):
    def test_manifest_is_complete_unique_and_well_typed(self) -> None:
        fixtures = oracle.fixture_manifest()
        mutations = oracle.mutation_manifest()
        oracle.validate_manifest(fixtures, mutations)
        self.assertEqual(len({fixture.id for fixture in fixtures}), len(fixtures))
        self.assertEqual(len({mutation.id for mutation in mutations}), len(mutations))

    def test_every_child_has_positive_and_negative_parity(self) -> None:
        fixtures = oracle.fixture_manifest()
        for issue in oracle.OWNERSHIP_CHILD_ISSUES:
            rows = [fixture for fixture in fixtures if fixture.issue == issue]
            with self.subTest(issue=issue):
                self.assertTrue(any(row.polarity is oracle.Polarity.POSITIVE for row in rows))
                self.assertTrue(any(row.polarity is oracle.Polarity.NEGATIVE for row in rows))

    def test_closed_children_are_green_controls(self) -> None:
        fixtures = oracle.fixture_manifest()
        for issue in (1222, 1344):
            rows = [fixture for fixture in fixtures if fixture.issue == issue]
            self.assertTrue(rows)
            self.assertTrue(all(isinstance(row.expected, oracle.MustPass) for row in rows))

    def test_open_children_use_typed_expected_failures(self) -> None:
        fixtures = oracle.fixture_manifest()
        open_children = oracle.OWNERSHIP_CHILD_ISSUES - {1222, 1344}
        for issue in open_children:
            rows = [fixture for fixture in fixtures if fixture.issue == issue]
            with self.subTest(issue=issue):
                self.assertTrue(
                    any(
                        isinstance(row.expected, oracle.ExpectedFailure)
                        and row.expected.issue == issue
                        and row.expected.detector == row.detector
                        for row in rows
                    )
                )

    def test_expected_failures_name_a_future_green_phase(self) -> None:
        for fixture in oracle.fixture_manifest():
            if isinstance(fixture.expected, oracle.ExpectedFailure):
                with self.subTest(fixture=fixture.id):
                    self.assertIn(fixture.green_by, range(1, 5))

    def test_host_nonzero_expected_failures_freeze_exact_receipts(self) -> None:
        for fixture in oracle.fixture_manifest():
            if not isinstance(fixture.expected, oracle.ExpectedFailure):
                continue
            if fixture.detector is not oracle.Detector.NONZERO_EXIT:
                continue
            if fixture.platform is oracle.Platform.HIP_HARDWARE:
                continue
            with self.subTest(fixture=fixture.id):
                self.assertTrue(
                    fixture.expected_exit is not None
                    or fixture.diagnostic_fragments
                )

    def test_ledger_expected_failures_freeze_exact_receipts(self) -> None:
        for fixture in oracle.fixture_manifest():
            if not isinstance(fixture.expected, oracle.ExpectedFailure):
                continue
            with self.subTest(fixture=fixture.id):
                if fixture.detector is oracle.Detector.LEDGER_LEAK:
                    receipt = fixture.ledger_receipt
                    self.assertIsNotNone(receipt)
                    assert receipt is not None
                    self.assertIsNotNone(receipt.live_owners)
                    self.assertIsNotNone(receipt.live_bytes)
                    self.assertTrue(receipt.live_kinds)
                elif fixture.detector is oracle.Detector.PEAK_BOUND:
                    receipt = fixture.ledger_receipt
                    self.assertIsNotNone(receipt)
                    assert receipt is not None
                    self.assertIsNotNone(receipt.peak_live_bytes)
                elif fixture.detector is oracle.Detector.INVALID_RELEASE:
                    receipt = fixture.ledger_receipt
                    self.assertIsNotNone(receipt)
                    assert receipt is not None
                    self.assertEqual(receipt.invalid_operations, 1)

    def test_external_prerequisite_is_not_an_ownership_child(self) -> None:
        fixtures = oracle.fixture_manifest()
        rows = [fixture for fixture in fixtures if fixture.issue == 1339]
        self.assertTrue(rows)
        self.assertTrue(all(row.external_prerequisite for row in rows))
        self.assertNotIn(1339, oracle.OWNERSHIP_CHILD_ISSUES)

    def test_forward_capture_freezes_the_complete_expected_output(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "forward-captured-list"
        )
        self.assertEqual(
            fixture.expected_output,
            "capture = [1, 2]\nlater = [1, 2]",
        )

    def test_option_scalars_freeze_every_emitted_root(self) -> None:
        fixtures = {row.id: row for row in oracle.fixture_manifest()}
        self.assertEqual(
            fixtures["option-scalar-some"].expected_output,
            "option_value = 7\nout = 7",
        )
        self.assertEqual(
            fixtures["option-scalar-none"].expected_output,
            "option_value = 0\nout = 0",
        )

    def test_option_and_recursive_function_projection_universe_is_frozen(self) -> None:
        ids = {fixture.id for fixture in oracle.fixture_manifest()}
        required = {
            "option-scalar-some",
            "option-scalar-none",
            "option-string",
            "option-nested-string",
            "option-mapped-file",
            "option-nested-mapped-file",
            "reject-option-function-c",
            "reject-list-function-c",
            "reject-tuple-function-c",
            "reject-dict-function-c",
            "reject-adt-function-c",
            "reject-option-function-hip",
            "reject-option-function-metal",
            "contextual-callback-c",
            "contextual-callback-hip",
            "contextual-callback-metal",
        }
        self.assertTrue(required <= ids, required - ids)

    def test_phase_zero_is_hardware_independent_but_hardware_row_exists(self) -> None:
        phase_zero = oracle.select_fixtures("0", require_hip=False)
        self.assertFalse(any(row.platform is oracle.Platform.HIP_HARDWARE for row in phase_zero))
        self.assertTrue(
            any(
                row.platform is oracle.Platform.HIP_HARDWARE
                for row in oracle.fixture_manifest()
            )
        )

    def test_closure_phases_require_explicit_hip_hardware(self) -> None:
        for phase in ("3", "4", "complete"):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    "requires --require-hip",
                ):
                    oracle.select_fixtures(phase, require_hip=False)

    def test_launch_selector_is_the_exact_frozen_subset(self) -> None:
        selected = oracle.select_fixtures("launch", require_hip=False)
        self.assertEqual({row.issue for row in selected}, {1339, 1344, 1346, 1356})
        self.assertTrue(all(row.backend is oracle.Backend.C for row in selected))

    def test_every_fixture_source_exists(self) -> None:
        for fixture in oracle.fixture_manifest():
            if fixture.source is not None:
                with self.subTest(fixture=fixture.id):
                    self.assertTrue((oracle.REPO_ROOT / fixture.source).is_file())

    def test_mutation_identity_set_is_frozen(self) -> None:
        self.assertEqual(
            {mutation.id for mutation in oracle.mutation_manifest()},
            {
                "heap-kind-without-finalizer",
                "option-host-carrier-omitted",
                "mapped-file-carrier-omitted",
                "recursive-function-rejection-weakened",
                "tensor-clone-omitted",
                "branch-clone-borrowed",
                "entry-borrow-reuse-admitted",
                "metal-reuse-admitted",
                "tail-drop-delayed",
                "backend-local-ownership-predicate-restored",
                "ledger-release-omitted",
                "ledger-release-duplicated",
                "ledger-event-stream-emptied",
                "manifest-receipt-omitted",
            },
        )


class LedgerContractTests(unittest.TestCase):
    def write_ledger(self, records: list[dict[str, object]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "ledger.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def test_valid_ledger_preserves_deterministic_counts_and_bytes(self) -> None:
        path = self.write_ledger(
            [
                {"event": "header", "schema": oracle.LEDGER_SCHEMA},
                {
                    "event": "allocate",
                    "id": 1,
                    "kind": "String",
                    "bytes": 6,
                    "owners_before": 0,
                    "owners_after": 1,
                    "site": "test",
                },
                {
                    "event": "release",
                    "id": 1,
                    "kind": "String",
                    "bytes": 6,
                    "owners_before": 1,
                    "owners_after": 0,
                    "site": "test",
                },
                {
                    "event": "finalize",
                    "id": 1,
                    "kind": "String",
                    "bytes": 6,
                    "owners_before": 0,
                    "owners_after": 0,
                    "site": "test",
                },
                {
                    "event": "summary",
                    "allocations": 1,
                    "finalized": 1,
                    "live_owners": 0,
                    "live_bytes": 0,
                    "peak_live_bytes": 6,
                    "invalid_operations": 0,
                },
            ]
        )
        ledger = oracle.load_ledger(path)
        self.assertEqual(ledger.summary.live_owners, 0)
        self.assertEqual(ledger.summary.peak_live_bytes, 6)
        self.assertEqual(ledger.kind_allocations["String"], 1)

    def test_empty_event_stream_fails_zero_vacuity(self) -> None:
        path = self.write_ledger(
            [
                {"event": "header", "schema": oracle.LEDGER_SCHEMA},
                {
                    "event": "summary",
                    "allocations": 0,
                    "finalized": 0,
                    "live_owners": 0,
                    "live_bytes": 0,
                    "peak_live_bytes": 0,
                    "invalid_operations": 0,
                },
            ]
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "zero-vacuity"):
            oracle.load_ledger(path, minimum_allocations=1)

    def test_missing_summary_fails_closed(self) -> None:
        path = self.write_ledger(
            [{"event": "header", "schema": oracle.LEDGER_SCHEMA}]
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "summary"):
            oracle.load_ledger(path)


class ReceiptContractTests(unittest.TestCase):
    def test_ledger_receipt_count_drift_fails_closed(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "aggregate-tensor-list"
        )
        ledger = oracle.Ledger(
            records=(),
            summary=oracle.LedgerSummary(12, 0, 11, 224, 224, 0),
            kind_allocations=oracle.Counter(),
            live_kind_counts=oracle.Counter(
                {"List": 4, "Tensor": 4, "TensorStorage": 4}
            ),
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "frozen ledger receipt drift"):
            oracle.assert_ledger_receipt(fixture, ledger)

    def test_expected_failure_requires_the_exact_detector(self) -> None:
        expected = oracle.ExpectedFailure(543, oracle.Detector.LEDGER_LEAK)
        oracle.classify_result(
            expected,
            oracle.Detection.failure(oracle.Detector.LEDGER_LEAK, "two tensors live"),
            "fixture",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "wrong detector"):
            oracle.classify_result(
                expected,
                oracle.Detection.failure(oracle.Detector.INVALID_RELEASE, "bad release"),
                "fixture",
            )

    def test_unexpected_success_fails_closed(self) -> None:
        expected = oracle.ExpectedFailure(543, oracle.Detector.LEDGER_LEAK)
        with self.assertRaisesRegex(oracle.OracleFailure, "unexpectedly passed"):
            oracle.classify_result(expected, oracle.Detection.success("clean"), "fixture")

    def test_nonzero_receipt_rejects_exit_and_diagnostic_drift(self) -> None:
        tensor = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "forward-captured-tensor"
        )
        self.assertIs(
            oracle.detect_nonzero(tensor, -11, ""),
            oracle.Detector.NONZERO_EXIT,
        )
        self.assertIs(
            oracle.detect_nonzero(tensor, -6, ""),
            oracle.Detector.MANIFEST,
        )

        option = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "option-nested-string"
        )
        expected_diagnostic = (
            "unsupported: unresolved host type `Option(String)` on boxing a "
            "resolved host value; no fallback representation is permitted"
        )
        self.assertIs(
            oracle.detect_nonzero(option, 1, expected_diagnostic),
            oracle.Detector.NONZERO_EXIT,
        )
        self.assertIs(
            oracle.detect_nonzero(option, 1, "unrelated parser failure"),
            oracle.Detector.MANIFEST,
        )

    def test_must_pass_rejects_a_detected_failure(self) -> None:
        with self.assertRaisesRegex(oracle.OracleFailure, "must pass"):
            oracle.classify_result(
                oracle.MustPass(),
                oracle.Detection.failure(oracle.Detector.NONZERO_EXIT, "exit 1"),
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
