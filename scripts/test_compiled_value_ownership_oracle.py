#!/usr/bin/env python3
"""Unit and mutation tests for chelis#1286's Phase 0 ownership oracle."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import compiled_value_ownership_oracle as oracle  # noqa: E402


EXPECTED_CHILD_ISSUES = frozenset(
    {543, 544, 1206, 1214, 1222, 1344, 1346, 1352, 1356}
)

EXPECTED_FIXTURE_IDS = frozenset(
    """
    oracle-self-tests runtime-ledger-process-tests
    aggregate-tensor-list aggregate-tensor-tuple aggregate-tensor-dict
    aggregate-tensor-adt aggregate-tensor-nested-repeated aggregate-scalar-control
    list-string-4-threshold-control list-string-5-threshold
    tuple-string-1-threshold-control tuple-string-2-threshold
    nested-string-1-threshold-control nested-string-2-threshold
    dict-string-1-threshold-control dict-string-2-threshold
    aggregate-refcount-scalar-control recursive-depth-1-control
    recursive-scalar-control recursive-depth-32 recursive-depth-128
    recursive-depth-288 c-caller-owned-reuse c-caller-owned-view-reuse
    hip-caller-owned-reuse hip-caller-owned-view-reuse hip-no-reuse-control
    hip-caller-bytes-unchanged-hardware root-alias-clean distinct-roots-clean
    captured-copy-clean fresh-function-binding-clean fold-alias-single-owner
    fold-fresh-control if-mixed-fresh-arm match-adt-mixed-fresh-arm
    match-option-mixed-fresh-control if-alias-control fresh-call-argument
    nested-fresh-call-argument variable-call-argument-control
    forward-captured-list forward-captured-tensor
    unannotated-forward-capture-control option-scalar-some option-scalar-none
    option-string option-nested-string mapped-file-direct option-mapped-file
    option-nested-mapped-file reject-option-function-c reject-list-function-c
    reject-tuple-function-c reject-dict-function-c reject-adt-function-c
    contextual-callback-c reject-option-function-hip reject-list-function-hip
    reject-tuple-function-hip reject-dict-function-hip reject-adt-function-hip
    contextual-callback-hip reject-option-function-metal
    reject-list-function-metal reject-tuple-function-metal
    reject-dict-function-metal reject-adt-function-metal
    contextual-callback-metal
    """.split()
)

EXPECTED_MUTATIONS = frozenset(
    {
        ("heap-kind-without-finalizer", 1, oracle.Detector.SOURCE_CONTRACT, "future structural registry"),
        ("option-host-carrier-omitted", 1, oracle.Detector.SOURCE_CONTRACT, "frozen option fixture census"),
        ("mapped-file-carrier-omitted", 1, oracle.Detector.MANIFEST, "mapped-file fixture census"),
        ("recursive-function-rejection-weakened", 0, oracle.Detector.EXACT_REJECTION, "exact #879 matrix"),
        ("tensor-clone-omitted", 1, oracle.Detector.LEDGER_LEAK, "aggregate tensor balance"),
        ("branch-clone-borrowed", 2, oracle.Detector.LEDGER_LEAK, "branch fixture parity"),
        ("entry-borrow-reuse-admitted", 3, oracle.Detector.SOURCE_CONTRACT, "C and HIP caller-storage negatives"),
        ("metal-reuse-admitted", 3, oracle.Detector.SOURCE_CONTRACT, "typed Metal no-reuse contract"),
        ("tail-drop-delayed", 3, oracle.Detector.PEAK_BOUND, "depth 32/128/288 peak bound"),
        ("backend-local-ownership-predicate-restored", 2, oracle.Detector.SOURCE_CONTRACT, "verified backend boundary"),
        ("ledger-release-omitted", 0, oracle.Detector.LEDGER_LEAK, "balanced process ledger test"),
        ("ledger-release-duplicated", 0, oracle.Detector.INVALID_RELEASE, "duplicate-release process test"),
        ("ledger-event-stream-emptied", 0, oracle.Detector.ZERO_VACUITY, "empty-ledger parser test"),
        ("manifest-receipt-omitted", 0, oracle.Detector.MANIFEST, "receipt bijection"),
    }
)

EXPECTED_COUNTERPARTS = {
    543: {
        "positive": frozenset(
            {
                "aggregate-tensor-list",
                "aggregate-tensor-tuple",
                "aggregate-tensor-dict",
                "aggregate-tensor-adt",
                "aggregate-tensor-nested-repeated",
            }
        ),
        "negative": frozenset({"aggregate-scalar-control"}),
    },
    544: {
        "positive": frozenset(
            {
                "list-string-4-threshold-control",
                "list-string-5-threshold",
                "tuple-string-1-threshold-control",
                "tuple-string-2-threshold",
                "nested-string-1-threshold-control",
                "nested-string-2-threshold",
                "dict-string-1-threshold-control",
                "dict-string-2-threshold",
            }
        ),
        "negative": frozenset({"aggregate-refcount-scalar-control"}),
    },
    1206: {
        "positive": frozenset(
            {
                "recursive-depth-1-control",
                "recursive-depth-32",
                "recursive-depth-128",
                "recursive-depth-288",
            }
        ),
        "negative": frozenset({"recursive-scalar-control"}),
    },
    1214: {
        "positive": frozenset(
            {
                "c-caller-owned-reuse",
                "c-caller-owned-view-reuse",
                "hip-caller-owned-reuse",
                "hip-caller-owned-view-reuse",
                "hip-caller-bytes-unchanged-hardware",
            }
        ),
        "negative": frozenset({"hip-no-reuse-control"}),
    },
    1222: {
        "positive": frozenset({"root-alias-clean"}),
        "negative": frozenset({"distinct-roots-clean"}),
    },
    1344: {
        "positive": frozenset({"captured-copy-clean"}),
        "negative": frozenset({"fresh-function-binding-clean"}),
    },
    1346: {
        "positive": frozenset({"fold-alias-single-owner"}),
        "negative": frozenset({"fold-fresh-control"}),
    },
    1352: {
        "positive": frozenset(
            {"if-mixed-fresh-arm", "match-adt-mixed-fresh-arm"}
        ),
        "negative": frozenset(
            {"if-alias-control", "match-option-mixed-fresh-control"}
        ),
    },
    1356: {
        "positive": frozenset(
            {"fresh-call-argument", "nested-fresh-call-argument"}
        ),
        "negative": frozenset({"variable-call-argument-control"}),
    },
}


class ManifestContractTests(unittest.TestCase):
    def test_manifest_is_complete_unique_and_well_typed(self) -> None:
        fixtures = oracle.fixture_manifest()
        mutations = oracle.mutation_manifest()
        oracle.validate_manifest(fixtures, mutations)
        self.assertEqual(len({fixture.id for fixture in fixtures}), len(fixtures))
        self.assertEqual(len({mutation.id for mutation in mutations}), len(mutations))

    def test_every_test_command_has_a_frozen_nonempty_census(self) -> None:
        fixtures = oracle.fixture_manifest()
        test_rows = tuple(
            row
            for row in fixtures
            if row.action in {oracle.Action.COMMAND, oracle.Action.HIP_HARDWARE}
        )
        self.assertTrue(test_rows)
        for row in test_rows:
            with self.subTest(fixture=row.id):
                self.assertTrue(row.test_census)
                malformed = tuple(
                    replace(candidate, test_receipt=())
                    if candidate.id == row.id
                    else candidate
                    for candidate in fixtures
                )
                with self.assertRaisesRegex(
                    oracle.OracleFailure, "frozen nonempty test census"
                ):
                    oracle.validate_manifest(malformed, oracle.mutation_manifest())

    def test_test_command_receipts_freeze_exact_execution_outcomes(self) -> None:
        rows = tuple(
            row
            for row in oracle.fixture_manifest()
            if row.action in {oracle.Action.COMMAND, oracle.Action.HIP_HARDWARE}
        )
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(fixture=row.id):
                self.assertEqual(
                    row.test_census,
                    tuple(receipt.name for receipt in row.test_receipt),
                )
                self.assertTrue(
                    all(
                        receipt.outcome
                        in {oracle.TestOutcome.PASSED, oracle.TestOutcome.FAILED}
                        for receipt in row.test_receipt
                    )
                )
        failed = {
            row.id
            for row in rows
            if any(
                receipt.outcome is oracle.TestOutcome.FAILED
                for receipt in row.test_receipt
            )
        }
        self.assertEqual(
            failed,
            {"hip-caller-owned-reuse", "hip-caller-owned-view-reuse"},
        )

    def test_self_test_census_matches_the_loaded_suite(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "oracle-self-tests"
        )
        self.assertEqual(fixture.test_census, _self_test_names())

    def test_frozen_fixture_child_and_mutation_universes_are_literal(self) -> None:
        fixtures = oracle.fixture_manifest()
        mutations = oracle.mutation_manifest()
        self.assertEqual({fixture.id for fixture in fixtures}, EXPECTED_FIXTURE_IDS)
        self.assertEqual(oracle.OWNERSHIP_CHILD_ISSUES, EXPECTED_CHILD_ISSUES)
        self.assertEqual(
            {
                (row.id, row.activation_phase, row.detector, row.canary)
                for row in mutations
            },
            EXPECTED_MUTATIONS,
        )

    def test_child_counterpart_identities_are_frozen(self) -> None:
        fixtures = oracle.fixture_manifest()
        observed = {
            issue: {
                polarity.value: frozenset(
                    row.id
                    for row in fixtures
                    if row.issue == issue and row.polarity is polarity
                )
                for polarity in oracle.Polarity
            }
            for issue in EXPECTED_CHILD_ISSUES
        }
        self.assertEqual(observed, EXPECTED_COUNTERPARTS)

    def test_removing_a_child_and_all_its_rows_fails_closed(self) -> None:
        fixtures = tuple(row for row in oracle.fixture_manifest() if row.issue != 543)
        with mock.patch.object(
            oracle,
            "OWNERSHIP_CHILD_ISSUES",
            oracle.OWNERSHIP_CHILD_ISSUES - {543},
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "frozen fixture universe"):
                oracle.validate_manifest(fixtures, oracle.mutation_manifest())

    def test_undeclared_fixture_file_fails_closed(self) -> None:
        fixtures = oracle.fixture_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in {row.source for row in fixtures if row.source is not None}:
                assert source is not None
                path = root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            extra = root / oracle.FIXTURE_ROOT / "undeclared.ch"
            extra.write_text("out = 0i64\n")
            with self.assertRaisesRegex(oracle.OracleFailure, "fixture directory drift"):
                oracle.validate_manifest(
                    fixtures,
                    oracle.mutation_manifest(),
                    repository_root=root,
                )

    def test_every_child_has_positive_and_negative_parity(self) -> None:
        fixtures = oracle.fixture_manifest()
        for issue in EXPECTED_CHILD_ISSUES:
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
        open_children = EXPECTED_CHILD_ISSUES - {1222, 1344}
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

    def test_every_ledger_executable_freezes_exact_stdout(self) -> None:
        for fixture in oracle.fixture_manifest():
            if fixture.action is oracle.Action.LEDGER_BUILD_RUN:
                with self.subTest(fixture=fixture.id):
                    self.assertIsNotNone(fixture.expected_output)

    def test_recursive_function_fixtures_reach_named_value_projection(self) -> None:
        for source in {
            row.source
            for row in oracle.fixture_manifest()
            if row.detector is oracle.Detector.EXACT_REJECTION
            and row.action is oracle.Action.BUILD_REJECT
        }:
            assert source is not None
            text = (oracle.REPO_ROOT / source).read_text()
            with self.subTest(source=source):
                self.assertIn("def identity", text)
                self.assertNotIn("fn (", text)

    def test_recursive_function_metal_rows_are_typed_expected_failures(self) -> None:
        rows = [
            row
            for row in oracle.fixture_manifest()
            if row.action is oracle.Action.BUILD_REJECT
        ]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(fixture=row.id):
                if row.backend is oracle.Backend.METAL:
                    self.assertEqual(
                        row.expected,
                        oracle.ExpectedFailure(879, oracle.Detector.EXACT_REJECTION),
                    )
                else:
                    self.assertIsInstance(row.expected, oracle.MustPass)

    def test_c_and_hip_reuse_rows_name_exact_behavioral_tests(self) -> None:
        rows = {row.id: row for row in oracle.fixture_manifest()}
        for fixture_id in (
            "c-caller-owned-reuse",
            "c-caller-owned-view-reuse",
            "hip-caller-owned-reuse",
            "hip-caller-owned-view-reuse",
            "hip-no-reuse-control",
            "hip-caller-bytes-unchanged-hardware",
        ):
            with self.subTest(fixture=fixture_id):
                self.assertIn(fixture_id, rows)
                self.assertEqual(len(rows[fixture_id].test_census), 1)

        control = rows["hip-no-reuse-control"]
        test_name = (
            "emit::tests::"
            "fused_without_reusable_input_keeps_non_in_place_kernel_shape"
        )
        self.assertEqual(control.test_census, (test_name,))
        self.assertEqual(
            control.command,
            (
                "cargo",
                "test",
                "-p",
                "chelis-backend-hip",
                "--lib",
                test_name,
                "--",
                "--exact",
            ),
        )

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

    def invalid_event_ledger(
        self, event: str, state: str
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = [
            {"event": "header", "schema": oracle.LEDGER_SCHEMA}
        ]
        if state != "unknown":
            records.append(
                {
                    "event": "allocate",
                    "id": 1,
                    "kind": "String",
                    "bytes": 6,
                    "owners_before": 0,
                    "owners_after": 1,
                    "site": "test",
                }
            )
            if state in {"zero", "finalized"}:
                records.append(
                    {
                        "event": "release",
                        "id": 1,
                        "kind": "String",
                        "bytes": 6,
                        "owners_before": 1,
                        "owners_after": 0,
                        "site": "test",
                    }
                )
            if state == "finalized":
                records.append(
                    {
                        "event": "finalize",
                        "id": 1,
                        "kind": "String",
                        "bytes": 6,
                        "owners_before": 0,
                        "owners_after": 0,
                        "site": "test",
                    }
                )

        records.append(
            {
                "event": event,
                "id": None if state == "unknown" else 1,
                "site": "test",
            }
        )

        if state == "live":
            records.append(
                {
                    "event": "release",
                    "id": 1,
                    "kind": "String",
                    "bytes": 6,
                    "owners_before": 1,
                    "owners_after": 0,
                    "site": "test",
                }
            )
        if state in {"live", "zero"}:
            records.append(
                {
                    "event": "finalize",
                    "id": 1,
                    "kind": "String",
                    "bytes": 6,
                    "owners_before": 0,
                    "owners_after": 0,
                    "site": "test",
                }
            )

        allocations = 0 if state == "unknown" else 1
        records.append(
            {
                "event": "summary",
                "allocations": allocations,
                "finalized": allocations,
                "live_owners": 0,
                "live_bytes": 0,
                "peak_live_bytes": 0 if state == "unknown" else 6,
                "invalid_operations": 1,
            }
        )
        return records

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

    def test_transition_missing_schema_fields_fails_closed(self) -> None:
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
                    "owners_before": 1,
                    "owners_after": 0,
                },
                {
                    "event": "finalize",
                    "id": 1,
                    "owners_before": 0,
                    "owners_after": 0,
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
        with self.assertRaisesRegex(oracle.OracleFailure, "exact keys"):
            oracle.load_ledger(path)

    def test_transition_kind_and_bytes_must_match_allocation(self) -> None:
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
                    "kind": "List",
                    "bytes": 99,
                    "owners_before": 1,
                    "owners_after": 0,
                    "site": "test",
                },
                {
                    "event": "summary",
                    "allocations": 1,
                    "finalized": 0,
                    "live_owners": 0,
                    "live_bytes": 0,
                    "peak_live_bytes": 6,
                    "invalid_operations": 0,
                },
            ]
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "kind/bytes"):
            oracle.load_ledger(path)

    def test_zero_owner_allocation_requires_finalization(self) -> None:
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
                    "event": "summary",
                    "allocations": 1,
                    "finalized": 0,
                    "live_owners": 0,
                    "live_bytes": 0,
                    "peak_live_bytes": 6,
                    "invalid_operations": 0,
                },
            ]
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "missing finalization"):
            oracle.load_ledger(path)

    def test_live_owner_allocation_remains_valid_without_finalization(self) -> None:
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
                    "event": "summary",
                    "allocations": 1,
                    "finalized": 0,
                    "live_owners": 1,
                    "live_bytes": 6,
                    "peak_live_bytes": 6,
                    "invalid_operations": 0,
                },
            ]
        )
        self.assertEqual(oracle.load_ledger(path).summary.live_owners, 1)

    def test_invalid_event_truth_table_accepts_only_invalid_operations(self) -> None:
        accepted = [
            ("invalid_allocate", "live"),
            ("invalid_allocate", "zero"),
            ("invalid_retain", "unknown"),
            ("invalid_retain", "zero"),
            ("invalid_retain", "finalized"),
            ("invalid_release", "unknown"),
            ("invalid_release", "zero"),
            ("invalid_release", "finalized"),
            ("invalid_resize", "unknown"),
            ("invalid_resize", "zero"),
            ("invalid_resize", "finalized"),
            ("invalid_borrow", "unknown"),
            ("invalid_borrow", "zero"),
            ("invalid_borrow", "finalized"),
            ("invalid_finalize", "unknown"),
            ("invalid_finalize", "live"),
            ("invalid_finalize", "finalized"),
        ]
        for event, state in accepted:
            with self.subTest(event=event, state=state):
                ledger = oracle.load_ledger(
                    self.write_ledger(self.invalid_event_ledger(event, state))
                )
                self.assertEqual(ledger.invalid_events[event], 1)

        rejected = [
            ("invalid_allocate", "unknown"),
            ("invalid_allocate", "finalized"),
            ("invalid_retain", "live"),
            ("invalid_release", "live"),
            ("invalid_resize", "live"),
            ("invalid_borrow", "live"),
            ("invalid_finalize", "zero"),
        ]
        for event, state in rejected:
            with self.subTest(event=event, state=state):
                with self.assertRaisesRegex(
                    oracle.OracleFailure, "does not match reconstructed state"
                ):
                    oracle.load_ledger(
                        self.write_ledger(self.invalid_event_ledger(event, state))
                    )

    def test_invalid_event_requires_exact_identity_and_site_schema(self) -> None:
        path = self.write_ledger(
            [
                {"event": "header", "schema": oracle.LEDGER_SCHEMA},
                {"event": "invalid_release"},
                {
                    "event": "summary",
                    "allocations": 0,
                    "finalized": 0,
                    "live_owners": 0,
                    "live_bytes": 0,
                    "peak_live_bytes": 0,
                    "invalid_operations": 1,
                },
            ]
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "exact keys"):
            oracle.load_ledger(path)


class ReceiptContractTests(unittest.TestCase):
    def test_empty_python_and_cargo_test_suites_fail_zero_vacuity(self) -> None:
        rows = {
            row.id: row
            for row in oracle.fixture_manifest()
            if row.id in {"oracle-self-tests", "runtime-ledger-process-tests"}
        }
        context = mock.Mock(environment={})
        empty_list = oracle.subprocess.CompletedProcess(
            args=("test-runner",),
            returncode=0,
            stdout="",
            stderr="Ran 0 tests\n\nOK\n",
        )
        for fixture_id in ("oracle-self-tests", "runtime-ledger-process-tests"):
            with self.subTest(fixture=fixture_id):
                with mock.patch.object(oracle, "_run", return_value=empty_list) as run:
                    detection = oracle._execute_command(context, rows[fixture_id])
                self.assertEqual(run.call_count, 1)
                self.assertIs(detection.detector, oracle.Detector.ZERO_VACUITY)

    def test_listed_but_skipped_or_ignored_tests_fail_zero_vacuity(self) -> None:
        rows = {row.id: row for row in oracle.fixture_manifest()}
        cases = (
            ("oracle-self-tests", "skipped 'mutant'"),
            ("runtime-ledger-process-tests", "ignored"),
            ("c-caller-owned-reuse", "ignored"),
            ("hip-caller-bytes-unchanged-hardware", "ignored"),
        )
        context = mock.Mock(environment={})
        for fixture_id, skipped_status in cases:
            fixture = rows[fixture_id]
            listed = oracle.subprocess.CompletedProcess(
                args=("test-list",),
                returncode=0,
                stdout="".join(f"{name}: test\n" for name in fixture.test_census),
                stderr="",
            )
            if fixture_id == "oracle-self-tests":
                module = oracle._python_unittest_target(fixture.command)
                self.assertIsNotNone(module)
                execution_lines = [
                    f"{name.rsplit('.', 1)[-1]} ({module}.{name}) ... "
                    f"{skipped_status if index == 0 else 'ok'}"
                    for index, name in enumerate(fixture.test_census)
                ]
                execution_lines.append(
                    f"Ran {len(fixture.test_census)} tests in 0.001s"
                )
            else:
                execution_lines = [
                    f"test {name} ... {skipped_status}"
                    for name in fixture.test_census
                ]
                execution_lines.append(
                    "test result: ok. 0 passed; 0 failed; "
                    f"{len(fixture.test_census)} ignored;"
                )
            executed = oracle.subprocess.CompletedProcess(
                args=("test-runner",),
                returncode=0,
                stdout="\n".join(execution_lines),
                stderr="",
            )
            if fixture.action is oracle.Action.HIP_HARDWARE:
                fixture = replace(
                    fixture,
                    command=tuple(
                        item for item in fixture.command if item != "--ignored"
                    ),
                )
            with self.subTest(fixture=fixture_id):
                with mock.patch.object(
                    oracle, "_run", side_effect=(listed, executed)
                ) as run:
                    detection = oracle._execute_command(context, fixture)
                self.assertEqual(run.call_count, 2)
                self.assertIs(detection.detector, oracle.Detector.ZERO_VACUITY)

    def test_expected_failing_test_must_execute_and_fail(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "hip-caller-owned-reuse"
        )
        name = fixture.test_census[0]
        listed = oracle.subprocess.CompletedProcess(
            args=("test-list",),
            returncode=0,
            stdout=f"{name}: test\n",
            stderr="",
        )
        executed = oracle.subprocess.CompletedProcess(
            args=("test-runner",),
            returncode=101,
            stdout=(
                f"test {name} ... FAILED\n"
                "test result: FAILED. 0 passed; 1 failed; 0 ignored;\n"
            ),
            stderr="",
        )
        context = mock.Mock(environment={})
        with mock.patch.object(oracle, "_run", side_effect=(listed, executed)):
            detection = oracle._execute_command(context, fixture)
        self.assertIs(detection.detector, oracle.Detector.SOURCE_CONTRACT)

    def test_forged_module_main_cannot_prove_python_test_execution(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "oracle-self-tests"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "test_forged_receipt.py").write_text(
                """\
import sys
import unittest


class ForgedReceiptTests(unittest.TestCase):
    @unittest.skip("mutant disabled the body")
    def test_body(self):
        raise AssertionError("the body must not be reported as passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--list-tests"]:
        print("ForgedReceiptTests.test_body: test")
    else:
        print("test_body (__main__.ForgedReceiptTests.test_body) ... ok")
        print("Ran 1 test in 0.001s")
        print("OK")
"""
            )
            fixture = replace(
                fixture,
                command=(
                    "{python}",
                    "-m",
                    "unittest",
                    "-v",
                    "scripts.test_forged_receipt",
                ),
                test_receipt=(
                    oracle.TestCaseReceipt(
                        "ForgedReceiptTests.test_body", oracle.TestOutcome.PASSED
                    ),
                ),
            )
            real_run = oracle._run

            def run_in_temporary_root(*args, **kwargs):
                return real_run(*args, **kwargs, cwd=root)

            context = mock.Mock(environment=os.environ.copy())
            with mock.patch.object(
                oracle, "_run", side_effect=run_in_temporary_root
            ) as run:
                detection = oracle._execute_command(context, fixture)

        self.assertEqual(run.call_count, 2)
        self.assertIs(detection.detector, oracle.Detector.ZERO_VACUITY)
        self.assertIn("SKIPPED", detection.detail)

    def test_execution_receipt_rejects_missing_extra_and_wrong_outcomes(self) -> None:
        rows = {row.id: row for row in oracle.fixture_manifest()}
        runtime_names = rows["runtime-ledger-process-tests"].test_census
        cases = (
            (
                "runtime-ledger-process-tests",
                ("test result: ok. 0 passed; 0 failed; 0 ignored;",),
                0,
            ),
            (
                "runtime-ledger-process-tests",
                tuple(
                    f"test {name} ... ok"
                    for name in runtime_names
                )
                + (
                    "test undeclared_extra ... ok",
                    "test result: ok. "
                    f"{len(runtime_names) + 1} passed; 0 failed; 0 ignored;",
                ),
                0,
            ),
            (
                "c-caller-owned-reuse",
                (
                    "test "
                    f"{rows['c-caller-owned-reuse'].test_census[0]} ... FAILED",
                    "test result: FAILED. 0 passed; 1 failed; 0 ignored;",
                ),
                101,
            ),
            (
                "hip-caller-owned-reuse",
                (
                    "test "
                    f"{rows['hip-caller-owned-reuse'].test_census[0]} ... ok",
                    "test result: ok. 1 passed; 0 failed; 0 ignored;",
                ),
                0,
            ),
        )
        context = mock.Mock(environment={})
        for fixture_id, execution_lines, returncode in cases:
            fixture = rows[fixture_id]
            listed = oracle.subprocess.CompletedProcess(
                args=("test-list",),
                returncode=0,
                stdout="".join(f"{name}: test\n" for name in fixture.test_census),
                stderr="",
            )
            executed = oracle.subprocess.CompletedProcess(
                args=("test-runner",),
                returncode=returncode,
                stdout="\n".join(execution_lines),
                stderr="test result: FAILED" if returncode else "",
            )
            with self.subTest(fixture=fixture_id, lines=execution_lines):
                with mock.patch.object(
                    oracle, "_run", side_effect=(listed, executed)
                ):
                    detection = oracle._execute_command(context, fixture)
                self.assertIs(detection.detector, oracle.Detector.ZERO_VACUITY)

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
            invalid_events=oracle.Counter(),
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

    def test_invalid_retain_cannot_satisfy_invalid_release_receipt(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "fresh-call-argument"
        )
        ledger = oracle.Ledger(
            records=(),
            summary=oracle.LedgerSummary(1, 0, 0, 0, 1, 1),
            kind_allocations=oracle.Counter({"List": 1}),
            live_kind_counts=oracle.Counter(),
            invalid_events=oracle.Counter({"invalid_retain": 1}),
        )
        with mock.patch.object(oracle, "load_ledger", return_value=ledger):
            detection = oracle._ledger_detection(
                fixture,
                oracle.subprocess.CompletedProcess(
                    args=("fixture",),
                    returncode=1,
                    stdout=fixture.expected_output or "",
                    stderr="compiled ownership ledger detected invalid list release",
                ),
                Path("unused.jsonl"),
            )
        self.assertIs(detection.detector, oracle.Detector.MANIFEST)

    def test_balanced_ledger_cannot_hide_wrong_stdout(self) -> None:
        fixture = next(
            row
            for row in oracle.fixture_manifest()
            if row.id == "aggregate-scalar-control"
        )
        ledger = oracle.Ledger(
            records=(),
            summary=oracle.LedgerSummary(1, 1, 0, 0, 1, 0),
            kind_allocations=oracle.Counter({"List": 1}),
            live_kind_counts=oracle.Counter(),
            invalid_events=oracle.Counter(),
        )
        with mock.patch.object(oracle, "load_ledger", return_value=ledger):
            detection = oracle._ledger_detection(
                fixture,
                oracle.subprocess.CompletedProcess(
                    args=("fixture",),
                    returncode=0,
                    stdout="wrong output\n",
                    stderr="",
                ),
                Path("unused.jsonl"),
            )
        self.assertIs(detection.detector, oracle.Detector.OUTPUT_MISMATCH)


def _flatten_tests(suite: unittest.TestSuite) -> tuple[unittest.TestCase, ...]:
    tests: list[unittest.TestCase] = []
    for candidate in suite:
        if isinstance(candidate, unittest.TestSuite):
            tests.extend(_flatten_tests(candidate))
        elif isinstance(candidate, unittest.TestCase):
            tests.append(candidate)
    return tuple(tests)


def _self_test_names() -> tuple[str, ...]:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    prefix = f"{__name__}."
    return tuple(
        sorted(test.id().removeprefix(prefix) for test in _flatten_tests(suite))
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--list-tests"]:
        for name in _self_test_names():
            print(f"{name}: test")
    else:
        unittest.main()
