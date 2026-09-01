#!/usr/bin/env python3
"""Executable Phase 0 ownership oracle for chelis#1286.

The driver freezes the initial ownership fixture universe, builds generated
artifacts against the runtime's private allocation ledger, and distinguishes
known failures from regressions through typed detector receipts. Expected
failures are exact: a different failure or an unexpected success is red.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path("crates/chelis-cli/tests/fixtures/compiled_value_ownership")
LEDGER_SCHEMA = "compiled-value-ownership-ledger-v1"
OWNERSHIP_CHILD_ISSUES = frozenset(
    {543, 544, 1206, 1214, 1222, 1344, 1346, 1352, 1356}
)
FROZEN_OWNERSHIP_CHILD_ISSUES = frozenset(
    {543, 544, 1206, 1214, 1222, 1344, 1346, 1352, 1356}
)
FROZEN_FIXTURE_IDS = frozenset(
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
FROZEN_SELF_TEST_CENSUS = tuple(
    """
    LedgerContractTests.test_empty_event_stream_fails_zero_vacuity
    LedgerContractTests.test_invalid_event_requires_exact_identity_and_site_schema
    LedgerContractTests.test_invalid_event_truth_table_accepts_only_invalid_operations
    LedgerContractTests.test_live_owner_allocation_remains_valid_without_finalization
    LedgerContractTests.test_missing_summary_fails_closed
    LedgerContractTests.test_transition_kind_and_bytes_must_match_allocation
    LedgerContractTests.test_transition_missing_schema_fields_fails_closed
    LedgerContractTests.test_valid_ledger_preserves_deterministic_counts_and_bytes
    LedgerContractTests.test_zero_owner_allocation_requires_finalization
    ManifestContractTests.test_c_and_hip_reuse_rows_name_exact_behavioral_tests
    ManifestContractTests.test_child_counterpart_identities_are_frozen
    ManifestContractTests.test_closed_children_are_green_controls
    ManifestContractTests.test_closure_phases_require_explicit_hip_hardware
    ManifestContractTests.test_every_child_has_positive_and_negative_parity
    ManifestContractTests.test_every_fixture_source_exists
    ManifestContractTests.test_every_ledger_executable_freezes_exact_stdout
    ManifestContractTests.test_every_test_command_has_a_frozen_nonempty_census
    ManifestContractTests.test_expected_failures_name_a_future_green_phase
    ManifestContractTests.test_external_prerequisite_is_not_an_ownership_child
    ManifestContractTests.test_forward_capture_freezes_the_complete_expected_output
    ManifestContractTests.test_frozen_fixture_child_and_mutation_universes_are_literal
    ManifestContractTests.test_host_nonzero_expected_failures_freeze_exact_receipts
    ManifestContractTests.test_launch_selector_is_the_exact_frozen_subset
    ManifestContractTests.test_ledger_expected_failures_freeze_exact_receipts
    ManifestContractTests.test_manifest_is_complete_unique_and_well_typed
    ManifestContractTests.test_mutation_identity_set_is_frozen
    ManifestContractTests.test_open_children_use_typed_expected_failures
    ManifestContractTests.test_option_and_recursive_function_projection_universe_is_frozen
    ManifestContractTests.test_option_scalars_freeze_every_emitted_root
    ManifestContractTests.test_phase_zero_is_hardware_independent_but_hardware_row_exists
    ManifestContractTests.test_recursive_function_fixtures_reach_named_value_projection
    ManifestContractTests.test_recursive_function_metal_rows_are_typed_expected_failures
    ManifestContractTests.test_removing_a_child_and_all_its_rows_fails_closed
    ManifestContractTests.test_self_test_census_matches_the_loaded_suite
    ManifestContractTests.test_test_command_receipts_freeze_exact_execution_outcomes
    ManifestContractTests.test_undeclared_fixture_file_fails_closed
    ReceiptContractTests.test_balanced_ledger_cannot_hide_wrong_stdout
    ReceiptContractTests.test_empty_python_and_cargo_test_suites_fail_zero_vacuity
    ReceiptContractTests.test_execution_receipt_rejects_missing_extra_and_wrong_outcomes
    ReceiptContractTests.test_expected_failing_test_must_execute_and_fail
    ReceiptContractTests.test_expected_failure_requires_the_exact_detector
    ReceiptContractTests.test_forged_module_main_cannot_prove_python_test_execution
    ReceiptContractTests.test_invalid_retain_cannot_satisfy_invalid_release_receipt
    ReceiptContractTests.test_ledger_receipt_count_drift_fails_closed
    ReceiptContractTests.test_listed_but_skipped_or_ignored_tests_fail_zero_vacuity
    ReceiptContractTests.test_must_pass_rejects_a_detected_failure
    ReceiptContractTests.test_nonzero_receipt_rejects_exit_and_diagnostic_drift
    ReceiptContractTests.test_unexpected_success_fails_closed
    """.split()
)
FROZEN_RUNTIME_LEDGER_TEST_CENSUS = tuple(
    """
    balanced_string_records_owner_transitions_and_summary
    duplicate_release_records_invalid_operation_before_exit
    ledger_process_probe
    tensor_storage_uses_portable_payload_bytes
    """.split()
)
FROZEN_COUNTERPARTS = {
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


class OracleFailure(RuntimeError):
    """A fail-closed ownership-oracle error."""


class Backend(str, Enum):
    C = "c"
    HIP = "hip"
    METAL = "metal"
    RUNTIME = "runtime"


class Platform(str, Enum):
    ANY = "any"
    HOST_C = "host-c"
    HIP_HARDWARE = "hip-hardware"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Detector(str, Enum):
    MANIFEST = "manifest"
    LEDGER_LEAK = "ledger-leak"
    INVALID_RELEASE = "invalid-release"
    PEAK_BOUND = "peak-live-bytes"
    OUTPUT_MISMATCH = "output-mismatch"
    NONZERO_EXIT = "nonzero-exit"
    EXACT_REJECTION = "exact-879-rejection"
    SOURCE_CONTRACT = "source-contract"
    ZERO_VACUITY = "zero-vacuity"


class Action(str, Enum):
    COMMAND = "command"
    LEDGER_BUILD_RUN = "ledger-build-run"
    BUILD_RUN = "build-run"
    BUILD_ONLY = "build-only"
    BUILD_REJECT = "build-reject"
    CHECK_REJECT = "check-reject"
    HIP_HARDWARE = "hip-hardware"


class TestOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    IGNORED = "ignored"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class MustPass:
    pass


@dataclass(frozen=True)
class ExpectedFailure:
    issue: int
    detector: Detector


Expected = MustPass | ExpectedFailure


@dataclass(frozen=True)
class LedgerReceipt:
    live_owners: int | None = None
    live_bytes: int | None = None
    live_kinds: tuple[tuple[str, int], ...] = ()
    peak_live_bytes: int | None = None
    invalid_operations: int | None = None
    invalid_event: str | None = None


@dataclass(frozen=True)
class Detection:
    passed: bool
    detector: Detector | None
    detail: str

    @classmethod
    def success(cls, detail: str) -> Detection:
        return cls(True, None, detail)

    @classmethod
    def failure(cls, detector: Detector, detail: str) -> Detection:
        return cls(False, detector, detail)


@dataclass(frozen=True)
class TestCaseReceipt:
    name: str
    outcome: TestOutcome


@dataclass(frozen=True)
class Fixture:
    id: str
    issue: int
    polarity: Polarity
    backend: Backend
    detector: Detector
    expected: Expected
    action: Action
    source: str | None = None
    platform: Platform = Platform.ANY
    green_by: int = 0
    launch: bool = False
    external_prerequisite: bool = False
    expected_output: str | None = None
    expected_exit: int | None = None
    diagnostic_fragments: tuple[str, ...] = ()
    peak_bound: int | None = None
    ledger_receipt: LedgerReceipt | None = None
    command: tuple[str, ...] = ()
    test_receipt: tuple[TestCaseReceipt, ...] = ()
    containing_type: str | None = None

    @property
    def test_census(self) -> tuple[str, ...]:
        return tuple(receipt.name for receipt in self.test_receipt)


@dataclass(frozen=True)
class Mutation:
    id: str
    activation_phase: int
    detector: Detector
    canary: str


FROZEN_MUTATION_SPECS = frozenset(
    {
        (
            "heap-kind-without-finalizer",
            1,
            Detector.SOURCE_CONTRACT,
            "future structural registry",
        ),
        (
            "option-host-carrier-omitted",
            1,
            Detector.SOURCE_CONTRACT,
            "frozen option fixture census",
        ),
        (
            "mapped-file-carrier-omitted",
            1,
            Detector.MANIFEST,
            "mapped-file fixture census",
        ),
        (
            "recursive-function-rejection-weakened",
            0,
            Detector.EXACT_REJECTION,
            "exact #879 matrix",
        ),
        (
            "tensor-clone-omitted",
            1,
            Detector.LEDGER_LEAK,
            "aggregate tensor balance",
        ),
        (
            "branch-clone-borrowed",
            2,
            Detector.LEDGER_LEAK,
            "branch fixture parity",
        ),
        (
            "entry-borrow-reuse-admitted",
            3,
            Detector.SOURCE_CONTRACT,
            "C and HIP caller-storage negatives",
        ),
        (
            "metal-reuse-admitted",
            3,
            Detector.SOURCE_CONTRACT,
            "typed Metal no-reuse contract",
        ),
        (
            "tail-drop-delayed",
            3,
            Detector.PEAK_BOUND,
            "depth 32/128/288 peak bound",
        ),
        (
            "backend-local-ownership-predicate-restored",
            2,
            Detector.SOURCE_CONTRACT,
            "verified backend boundary",
        ),
        (
            "ledger-release-omitted",
            0,
            Detector.LEDGER_LEAK,
            "balanced process ledger test",
        ),
        (
            "ledger-release-duplicated",
            0,
            Detector.INVALID_RELEASE,
            "duplicate-release process test",
        ),
        (
            "ledger-event-stream-emptied",
            0,
            Detector.ZERO_VACUITY,
            "empty-ledger parser test",
        ),
        (
            "manifest-receipt-omitted",
            0,
            Detector.MANIFEST,
            "receipt bijection",
        ),
    }
)


@dataclass(frozen=True)
class LedgerSummary:
    allocations: int
    finalized: int
    live_owners: int
    live_bytes: int
    peak_live_bytes: int
    invalid_operations: int


@dataclass(frozen=True)
class Ledger:
    records: tuple[dict[str, object], ...]
    summary: LedgerSummary
    kind_allocations: Counter[str]
    live_kind_counts: Counter[str]
    invalid_events: Counter[str]


@dataclass(frozen=True)
class BuildArtifact:
    output_dir: Path
    build: subprocess.CompletedProcess[str]
    compile_command: tuple[str, ...] | None


def fixture_path(name: str) -> str:
    return str(FIXTURE_ROOT / name)


def xfail(issue: int, detector: Detector) -> ExpectedFailure:
    return ExpectedFailure(issue, detector)


def ledger_receipt(
    *,
    live_owners: int | None = None,
    live_bytes: int | None = None,
    live_kinds: dict[str, int] | None = None,
    peak_live_bytes: int | None = None,
    invalid_operations: int | None = None,
    invalid_event: str | None = None,
) -> LedgerReceipt:
    return LedgerReceipt(
        live_owners=live_owners,
        live_bytes=live_bytes,
        live_kinds=tuple(sorted((live_kinds or {}).items())),
        peak_live_bytes=peak_live_bytes,
        invalid_operations=invalid_operations,
        invalid_event=invalid_event,
    )


def test_receipt(
    names: Sequence[str], outcome: TestOutcome = TestOutcome.PASSED
) -> tuple[TestCaseReceipt, ...]:
    return tuple(TestCaseReceipt(name, outcome) for name in names)


def _fixture(
    id: str,
    issue: int,
    polarity: Polarity,
    detector: Detector,
    expected: Expected,
    action: Action,
    source: str | None = None,
    *,
    backend: Backend = Backend.C,
    platform: Platform = Platform.HOST_C,
    green_by: int = 0,
    launch: bool = False,
    external_prerequisite: bool = False,
    expected_output: str | None = None,
    expected_exit: int | None = None,
    diagnostic_fragments: Sequence[str] = (),
    peak_bound: int | None = None,
    command: Sequence[str] = (),
    test_receipt: Sequence[TestCaseReceipt] = (),
    containing_type: str | None = None,
) -> Fixture:
    return Fixture(
        id=id,
        issue=issue,
        polarity=polarity,
        backend=backend,
        detector=detector,
        expected=expected,
        action=action,
        source=fixture_path(source) if source is not None else None,
        platform=platform,
        green_by=green_by,
        launch=launch,
        external_prerequisite=external_prerequisite,
        expected_output=expected_output,
        expected_exit=expected_exit,
        diagnostic_fragments=tuple(diagnostic_fragments),
        peak_bound=peak_bound,
        command=tuple(command),
        test_receipt=tuple(test_receipt),
        containing_type=containing_type,
    )


def fixture_manifest() -> tuple[Fixture, ...]:
    """Return the frozen Phase 0 fixture and command universe."""

    rows: list[Fixture] = [
        _fixture(
            "oracle-self-tests",
            1286,
            Polarity.NEGATIVE,
            Detector.MANIFEST,
            MustPass(),
            Action.COMMAND,
            backend=Backend.RUNTIME,
            platform=Platform.ANY,
            command=(
                "{python}",
                "-m",
                "unittest",
                "-v",
                "scripts.test_compiled_value_ownership_oracle",
            ),
            test_receipt=test_receipt(FROZEN_SELF_TEST_CENSUS),
        ),
        _fixture(
            "runtime-ledger-process-tests",
            1286,
            Polarity.NEGATIVE,
            Detector.MANIFEST,
            MustPass(),
            Action.COMMAND,
            backend=Backend.RUNTIME,
            platform=Platform.ANY,
            command=(
                "cargo",
                "test",
                "-p",
                "chelis-runtime",
                "--features",
                "ownership-ledger",
                "--test",
                "ownership_ledger",
            ),
            test_receipt=test_receipt(FROZEN_RUNTIME_LEDGER_TEST_CENSUS),
        ),
    ]

    for id, source in (
        ("aggregate-tensor-list", "issue_543_list_tensor.ch"),
        ("aggregate-tensor-tuple", "issue_543_tuple_tensor.ch"),
        ("aggregate-tensor-dict", "issue_543_dict_tensor.ch"),
        ("aggregate-tensor-adt", "issue_543_adt_tensor.ch"),
        ("aggregate-tensor-nested-repeated", "issue_543_nested_repeated_tensor.ch"),
    ):
        rows.append(
            _fixture(
                id,
                543,
                Polarity.POSITIVE,
                Detector.LEDGER_LEAK,
                xfail(543, Detector.LEDGER_LEAK),
                Action.LEDGER_BUILD_RUN,
                source,
                green_by=1,
            )
        )
    rows.append(
        _fixture(
            "aggregate-scalar-control",
            543,
            Polarity.NEGATIVE,
            Detector.LEDGER_LEAK,
            MustPass(),
            Action.LEDGER_BUILD_RUN,
            "control_scalar_aggregate.ch",
        )
    )

    for id, source, expected in (
        (
            "list-string-4-threshold-control",
            "issue_544_list_string_4.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "list-string-5-threshold",
            "issue_544_list_string_5.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "tuple-string-1-threshold-control",
            "issue_544_tuple_string_1.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "tuple-string-2-threshold",
            "issue_544_tuple_string_2.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "nested-string-1-threshold-control",
            "issue_544_nested_string_1.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "nested-string-2-threshold",
            "issue_544_nested_string_2.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "dict-string-1-threshold-control",
            "issue_544_dict_string_1.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
        (
            "dict-string-2-threshold",
            "issue_544_dict_string_2.ch",
            xfail(544, Detector.LEDGER_LEAK),
        ),
    ):
        rows.append(
            _fixture(
                id,
                544,
                Polarity.NEGATIVE if isinstance(expected, MustPass) else Polarity.POSITIVE,
                Detector.LEDGER_LEAK,
                expected,
                Action.LEDGER_BUILD_RUN,
                source,
                green_by=1,
            )
        )
    rows.append(
        _fixture(
            "aggregate-refcount-scalar-control",
            544,
            Polarity.NEGATIVE,
            Detector.LEDGER_LEAK,
            MustPass(),
            Action.LEDGER_BUILD_RUN,
            "control_scalar_aggregate.ch",
        )
    )

    rows.append(
        _fixture(
            "recursive-depth-1-control",
            1206,
            Polarity.POSITIVE,
            Detector.LEDGER_LEAK,
            xfail(1206, Detector.LEDGER_LEAK),
            Action.LEDGER_BUILD_RUN,
            "issue_1206_depth_1.ch",
            peak_bound=512,
            green_by=3,
        )
    )
    rows.append(
        _fixture(
            "recursive-scalar-control",
            1206,
            Polarity.NEGATIVE,
            Detector.PEAK_BOUND,
            MustPass(),
            Action.LEDGER_BUILD_RUN,
            "control_recursive_scalar.ch",
            peak_bound=512,
        )
    )
    for depth in (32, 128, 288):
        rows.append(
            _fixture(
                f"recursive-depth-{depth}",
                1206,
                Polarity.POSITIVE,
                Detector.PEAK_BOUND,
                xfail(1206, Detector.PEAK_BOUND),
                Action.LEDGER_BUILD_RUN,
                f"issue_1206_depth_{depth}.ch",
                peak_bound=512,
                green_by=3,
            )
        )

    rows.extend(
        [
            _fixture(
                "c-caller-owned-reuse",
                1214,
                Polarity.POSITIVE,
                Detector.SOURCE_CONTRACT,
                MustPass(),
                Action.COMMAND,
                backend=Backend.C,
                platform=Platform.ANY,
                command=(
                    "cargo",
                    "test",
                    "-p",
                    "chelis-backend-c",
                    "--lib",
                    "emit::tests::fused_in_place_does_not_alias_a_caller_owned_input",
                    "--",
                    "--exact",
                ),
                test_receipt=test_receipt(
                    ("emit::tests::fused_in_place_does_not_alias_a_caller_owned_input",)
                ),
            ),
            _fixture(
                "c-caller-owned-view-reuse",
                1214,
                Polarity.POSITIVE,
                Detector.SOURCE_CONTRACT,
                MustPass(),
                Action.COMMAND,
                backend=Backend.C,
                platform=Platform.ANY,
                command=(
                    "cargo",
                    "test",
                    "-p",
                    "chelis-backend-c",
                    "--lib",
                    "emit::tests::fused_in_place_does_not_alias_a_view_of_a_caller_owned_input",
                    "--",
                    "--exact",
                ),
                test_receipt=test_receipt(
                    (
                        "emit::tests::fused_in_place_does_not_alias_a_view_of_a_caller_owned_input",
                    )
                ),
            ),
            _fixture(
                "hip-caller-owned-reuse",
                1214,
                Polarity.POSITIVE,
                Detector.SOURCE_CONTRACT,
                xfail(1214, Detector.SOURCE_CONTRACT),
                Action.COMMAND,
                backend=Backend.HIP,
                platform=Platform.ANY,
                green_by=3,
                command=(
                    "cargo",
                    "test",
                    "-p",
                    "chelis-backend-hip",
                    "--lib",
                    "emit::tests::fused_in_place_does_not_alias_a_caller_owned_input",
                    "--",
                    "--ignored",
                    "--exact",
                ),
                test_receipt=test_receipt(
                    ("emit::tests::fused_in_place_does_not_alias_a_caller_owned_input",),
                    TestOutcome.FAILED,
                ),
            ),
            _fixture(
                "hip-caller-owned-view-reuse",
                1214,
                Polarity.POSITIVE,
                Detector.SOURCE_CONTRACT,
                xfail(1214, Detector.SOURCE_CONTRACT),
                Action.COMMAND,
                backend=Backend.HIP,
                platform=Platform.ANY,
                green_by=3,
                command=(
                    "cargo",
                    "test",
                    "-p",
                    "chelis-backend-hip",
                    "--lib",
                    "emit::tests::fused_in_place_does_not_alias_a_view_of_a_caller_owned_input",
                    "--",
                    "--ignored",
                    "--exact",
                ),
                test_receipt=test_receipt(
                    (
                        "emit::tests::fused_in_place_does_not_alias_a_view_of_a_caller_owned_input",
                    ),
                    TestOutcome.FAILED,
                ),
            ),
            _fixture(
                "hip-no-reuse-control",
                1214,
                Polarity.NEGATIVE,
                Detector.SOURCE_CONTRACT,
                MustPass(),
                Action.COMMAND,
                backend=Backend.HIP,
                platform=Platform.ANY,
                command=(
                    "cargo",
                    "test",
                    "-p",
                    "chelis-backend-hip",
                    "--lib",
                    "emit::tests::fused_without_reusable_input_keeps_non_in_place_kernel_shape",
                    "--",
                    "--exact",
                ),
                test_receipt=test_receipt(
                    (
                        "emit::tests::fused_without_reusable_input_keeps_non_in_place_kernel_shape",
                    )
                ),
            ),
            _fixture(
                "hip-caller-bytes-unchanged-hardware",
                1214,
                Polarity.POSITIVE,
                Detector.NONZERO_EXIT,
                xfail(1214, Detector.NONZERO_EXIT),
                Action.HIP_HARDWARE,
                backend=Backend.HIP,
                platform=Platform.HIP_HARDWARE,
                green_by=3,
                command=(
                    "{python}",
                    "scripts/hip_test.py",
                    "-p",
                    "chelis-backend-hip",
                    "--test",
                    "gpu_correctness",
                    "compiled_value_ownership_caller_bytes_unchanged",
                    "--",
                    "--ignored",
                    "--test-threads=1",
                ),
                test_receipt=test_receipt(
                    ("compiled_value_ownership_caller_bytes_unchanged",)
                ),
            ),
        ]
    )

    rows.extend(
        [
            _fixture(
                "root-alias-clean",
                1222,
                Polarity.POSITIVE,
                Detector.INVALID_RELEASE,
                MustPass(),
                Action.LEDGER_BUILD_RUN,
                "issue_1222_root_alias.ch",
            ),
            _fixture(
                "distinct-roots-clean",
                1222,
                Polarity.NEGATIVE,
                Detector.INVALID_RELEASE,
                MustPass(),
                Action.LEDGER_BUILD_RUN,
                "issue_1222_distinct_roots.ch",
            ),
            _fixture(
                "captured-copy-clean",
                1344,
                Polarity.POSITIVE,
                Detector.INVALID_RELEASE,
                MustPass(),
                Action.LEDGER_BUILD_RUN,
                "issue_1344_captured_copy.ch",
                launch=True,
            ),
            _fixture(
                "fresh-function-binding-clean",
                1344,
                Polarity.NEGATIVE,
                Detector.INVALID_RELEASE,
                MustPass(),
                Action.LEDGER_BUILD_RUN,
                "issue_1344_fresh_binding.ch",
            ),
            _fixture(
                "fold-alias-single-owner",
                1346,
                Polarity.POSITIVE,
                Detector.INVALID_RELEASE,
                xfail(1346, Detector.INVALID_RELEASE),
                Action.LEDGER_BUILD_RUN,
                "issue_1346_fold_alias.ch",
                green_by=2,
                launch=True,
            ),
            _fixture(
                "fold-fresh-control",
                1346,
                Polarity.NEGATIVE,
                Detector.NONZERO_EXIT,
                MustPass(),
                Action.BUILD_RUN,
                "issue_1346_fold_fresh.ch",
            ),
        ]
    )

    for id, source in (
        ("if-mixed-fresh-arm", "issue_1352_if_fresh.ch"),
        ("match-adt-mixed-fresh-arm", "issue_1352_match_adt_fresh.ch"),
    ):
        rows.append(
            _fixture(
                id,
                1352,
                Polarity.POSITIVE,
                Detector.LEDGER_LEAK,
                xfail(1352, Detector.LEDGER_LEAK),
                Action.LEDGER_BUILD_RUN,
                source,
                green_by=2,
            )
        )
    rows.append(
        _fixture(
            "match-option-mixed-fresh-control",
            1352,
            Polarity.NEGATIVE,
            Detector.LEDGER_LEAK,
            MustPass(),
            Action.LEDGER_BUILD_RUN,
            "issue_1352_match_option_fresh.ch",
        )
    )
    rows.append(
        _fixture(
            "if-alias-control",
            1352,
            Polarity.NEGATIVE,
            Detector.LEDGER_LEAK,
            MustPass(),
            Action.LEDGER_BUILD_RUN,
            "issue_1352_if_alias.ch",
        )
    )

    rows.extend(
        [
            _fixture(
                "fresh-call-argument",
                1356,
                Polarity.POSITIVE,
                Detector.INVALID_RELEASE,
                xfail(1356, Detector.INVALID_RELEASE),
                Action.LEDGER_BUILD_RUN,
                "issue_1356_fresh_argument.ch",
                green_by=2,
                launch=True,
            ),
            _fixture(
                "nested-fresh-call-argument",
                1356,
                Polarity.POSITIVE,
                Detector.INVALID_RELEASE,
                xfail(1356, Detector.INVALID_RELEASE),
                Action.LEDGER_BUILD_RUN,
                "issue_1356_nested_fresh_argument.ch",
                green_by=2,
            ),
            _fixture(
                "variable-call-argument-control",
                1356,
                Polarity.NEGATIVE,
                Detector.INVALID_RELEASE,
                MustPass(),
                Action.LEDGER_BUILD_RUN,
                "issue_1356_variable_argument.ch",
            ),
        ]
    )

    rows.extend(
        [
            _fixture(
                "forward-captured-list",
                1339,
                Polarity.POSITIVE,
                Detector.OUTPUT_MISMATCH,
                xfail(1339, Detector.OUTPUT_MISMATCH),
                Action.BUILD_RUN,
                "issue_1339_list_forward_capture.ch",
                green_by=4,
                launch=True,
                external_prerequisite=True,
                expected_output="capture = [1, 2]\nlater = [1, 2]",
            ),
            _fixture(
                "forward-captured-tensor",
                1339,
                Polarity.POSITIVE,
                Detector.NONZERO_EXIT,
                xfail(1339, Detector.NONZERO_EXIT),
                Action.BUILD_RUN,
                "issue_1339_tensor_forward_capture.ch",
                green_by=4,
                external_prerequisite=True,
                expected_exit=-11,
            ),
            _fixture(
                "unannotated-forward-capture-control",
                1339,
                Polarity.NEGATIVE,
                Detector.EXACT_REJECTION,
                MustPass(),
                Action.CHECK_REJECT,
                "issue_1339_unannotated_control.ch",
                external_prerequisite=True,
            ),
        ]
    )

    rows.extend(
        [
            _fixture(
                "option-scalar-some",
                1286,
                Polarity.POSITIVE,
                Detector.OUTPUT_MISMATCH,
                MustPass(),
                Action.BUILD_RUN,
                "option_scalar_some.ch",
                expected_output="option_value = 7\nout = 7",
            ),
            _fixture(
                "option-scalar-none",
                1286,
                Polarity.NEGATIVE,
                Detector.OUTPUT_MISMATCH,
                MustPass(),
                Action.BUILD_RUN,
                "option_scalar_none.ch",
                expected_output="option_value = 0\nout = 0",
            ),
        ]
    )
    exact_build_failures = {
        "option-nested-string": (
            "unsupported:",
            "unresolved host type `Option(String)`",
            "on boxing a resolved host value",
            "no fallback representation is permitted",
        ),
        "option-mapped-file": (
            "unsupported:",
            "unresolved host type `MappedFile`",
            "on boxing a resolved host value",
            "no fallback representation is permitted",
        ),
        "option-nested-mapped-file": (
            "unsupported:",
            "unresolved host type `MappedFile`",
            "on boxing a resolved host value",
            "no fallback representation is permitted",
        ),
    }
    for id, source, detector in (
        ("option-string", "option_string.ch", Detector.LEDGER_LEAK),
        ("option-nested-string", "option_nested_string.ch", Detector.NONZERO_EXIT),
        ("mapped-file-direct", "mapped_file_direct.ch", Detector.LEDGER_LEAK),
        ("option-mapped-file", "option_mapped_file.ch", Detector.NONZERO_EXIT),
        ("option-nested-mapped-file", "option_nested_mapped_file.ch", Detector.NONZERO_EXIT),
    ):
        rows.append(
            _fixture(
                id,
                1286,
                Polarity.POSITIVE,
                detector,
                xfail(1286, detector),
                Action.LEDGER_BUILD_RUN,
                source,
                green_by=1,
                diagnostic_fragments=exact_build_failures.get(id, ()),
            )
        )

    recursive_function_sources = {
        "option": ("reject_option_function.ch", "Option[int8 -> int8]"),
        "list": ("reject_list_function.ch", "List[int8 -> int8]"),
        "tuple": ("reject_tuple_function.ch", "(int8 -> int8, int64)"),
        "dict": ("reject_dict_function.ch", "Dict[string, int8 -> int8]"),
        "adt": ("reject_adt_function.ch", "CallbackBox"),
    }
    for backend in (Backend.C, Backend.HIP, Backend.METAL):
        for container, (source, containing_type) in recursive_function_sources.items():
            expected: Expected = MustPass()
            green_by = 0
            if backend is Backend.METAL:
                expected = xfail(879, Detector.EXACT_REJECTION)
                green_by = 1
            rows.append(
                _fixture(
                    f"reject-{container}-function-{backend.value}",
                    879,
                    Polarity.NEGATIVE,
                    Detector.EXACT_REJECTION,
                    expected,
                    Action.BUILD_REJECT,
                    source,
                    backend=backend,
                    platform=Platform.ANY,
                    green_by=green_by,
                    diagnostic_fragments=(
                        "unsupported:",
                        "Function([Scalar(Int8)], Scalar(Int8))",
                        "C host ABI value selection (codegen:c)",
                        "unimplemented chelis#879:",
                    ),
                    containing_type=containing_type,
                )
            )
        rows.append(
            _fixture(
                f"contextual-callback-{backend.value}",
                879,
                Polarity.POSITIVE,
                Detector.EXACT_REJECTION,
                MustPass(),
                Action.BUILD_ONLY,
                "contextual_callback.ch",
                backend=backend,
                platform=Platform.ANY,
            )
        )

    receipts = {
        "aggregate-tensor-list": ledger_receipt(
            live_owners=12,
            live_bytes=224,
            live_kinds={"List": 4, "Tensor": 4, "TensorStorage": 4},
        ),
        "aggregate-tensor-tuple": ledger_receipt(
            live_owners=16,
            live_bytes=416,
            live_kinds={"List": 4, "Tensor": 4, "TensorStorage": 4, "Tuple": 4},
        ),
        "aggregate-tensor-dict": ledger_receipt(
            live_owners=6,
            live_bytes=29,
            live_kinds={"String": 2, "Tensor": 2, "TensorStorage": 2},
        ),
        "aggregate-tensor-adt": ledger_receipt(
            live_owners=3,
            live_bytes=18,
            live_kinds={"String": 1, "Tensor": 1, "TensorStorage": 1},
        ),
        "aggregate-tensor-nested-repeated": ledger_receipt(
            live_owners=6,
            live_bytes=112,
            live_kinds={"List": 2, "Tensor": 2, "TensorStorage": 2},
        ),
        "list-string-4-threshold-control": ledger_receipt(
            live_owners=4, live_bytes=8, live_kinds={"String": 4}
        ),
        "list-string-5-threshold": ledger_receipt(
            live_owners=5, live_bytes=10, live_kinds={"String": 5}
        ),
        "tuple-string-1-threshold-control": ledger_receipt(
            live_owners=1, live_bytes=2, live_kinds={"String": 1}
        ),
        "tuple-string-2-threshold": ledger_receipt(
            live_owners=2, live_bytes=4, live_kinds={"String": 2}
        ),
        "nested-string-1-threshold-control": ledger_receipt(
            live_owners=2, live_bytes=4, live_kinds={"String": 2}
        ),
        "nested-string-2-threshold": ledger_receipt(
            live_owners=4, live_bytes=8, live_kinds={"String": 4}
        ),
        "dict-string-1-threshold-control": ledger_receipt(
            live_owners=1, live_bytes=2, live_kinds={"String": 1}
        ),
        "dict-string-2-threshold": ledger_receipt(
            live_owners=2, live_bytes=4, live_kinds={"String": 2}
        ),
        "recursive-depth-1-control": ledger_receipt(
            live_owners=7,
            live_bytes=144,
            live_kinds={"List": 1, "Tensor": 3, "TensorStorage": 3},
        ),
        "recursive-depth-32": ledger_receipt(peak_live_bytes=4736),
        "recursive-depth-128": ledger_receipt(peak_live_bytes=18560),
        "recursive-depth-288": ledger_receipt(peak_live_bytes=41600),
        "fold-alias-single-owner": ledger_receipt(
            invalid_operations=1, invalid_event="invalid_release"
        ),
        "if-mixed-fresh-arm": ledger_receipt(
            live_owners=1, live_bytes=48, live_kinds={"List": 1}
        ),
        "match-adt-mixed-fresh-arm": ledger_receipt(
            live_owners=4, live_bytes=21, live_kinds={"String": 3}
        ),
        "fresh-call-argument": ledger_receipt(
            invalid_operations=1, invalid_event="invalid_release"
        ),
        "nested-fresh-call-argument": ledger_receipt(
            invalid_operations=1, invalid_event="invalid_release"
        ),
        "option-string": ledger_receipt(
            live_owners=6, live_bytes=30, live_kinds={"String": 6}
        ),
        # The run helper gives mapped-file fixtures a stable relative filename,
        # so both the String path payload and mapped byte length are portable.
        "mapped-file-direct": ledger_receipt(
            live_owners=2, live_bytes=29, live_kinds={"MappedFile": 1, "String": 1}
        ),
    }
    outputs = {
        "aggregate-tensor-list": "make = [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[3.0, 4.0])]\nout = [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[3.0, 4.0])]",
        "aggregate-tensor-tuple": "make = [(tensor(shape=[2], data=[1.0, 2.0]), 1), (tensor(shape=[2], data=[3.0, 4.0]), 2)]\nout = [(tensor(shape=[2], data=[1.0, 2.0]), 1), (tensor(shape=[2], data=[3.0, 4.0]), 2)]",
        "aggregate-tensor-dict": "out = dict(first: tensor(shape=[2], data=[1.0, 2.0]), second: tensor(shape=[2], data=[3.0, 4.0]))",
        "aggregate-tensor-adt": "out.value = tensor(shape=[2], data=[1.0, 2.0])",
        "aggregate-tensor-nested-repeated": "make = [[tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[1.0, 2.0])], [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[1.0, 2.0])]]\nout = [[tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[1.0, 2.0])], [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[1.0, 2.0])]]",
        "aggregate-scalar-control": "out = [(1, 2), (3, 4)]",
        "list-string-4-threshold-control": "values = [a, b, c, d]\nout = 4",
        "list-string-5-threshold": "values = [a, b, c, d, e]\nout = 5",
        "tuple-string-1-threshold-control": "values = [(a, 1)]\nout = 1",
        "tuple-string-2-threshold": "values = [(a, 1), (b, 2)]\nout = 2",
        "nested-string-1-threshold-control": "values = [[a, b]]\nout = 1",
        "nested-string-2-threshold": "values = [[a, b], [c, d]]\nout = 2",
        "dict-string-1-threshold-control": "values = dict(a: 1)\nout = 1",
        "dict-string-2-threshold": "values = dict(a: 1, b: 2)\nout = 2",
        "aggregate-refcount-scalar-control": "out = [(1, 2), (3, 4)]",
        "recursive-depth-1-control": "out = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])",
        "recursive-scalar-control": "value = 288\nout = [288, 288]",
        "recursive-depth-32": "out = tensor(shape=[4], data=[32.0, 32.0, 32.0, 32.0])",
        "recursive-depth-128": "out = tensor(shape=[4], data=[128.0, 128.0, 128.0, 128.0])",
        "recursive-depth-288": "out = tensor(shape=[4], data=[288.0, 288.0, 288.0, 288.0])",
        "root-alias-clean": "original = [1, 2]\nalias = [1, 2]",
        "distinct-roots-clean": "first = [1]\nsecond = [2]",
        "captured-copy-clean": "global_values = [1, 2]\ntake_length = 2\nout = 2",
        "fresh-function-binding-clean": "make_length = 2\nout = 2",
        "fold-alias-single-owner": "values = [1, 2]\nbase.0 = 0.0\nbase.1 = 0.0\npicked.0 = 0.0\npicked.1 = 0.0",
        "if-mixed-fresh-arm": "original = [1]\nselected = [2, 3]",
        "match-adt-mixed-fresh-arm": "selected = [2, 3]",
        "match-option-mixed-fresh-control": "selected = [2, 3]",
        "if-alias-control": "original = [1]\nselected = [1]",
        "fresh-call-argument": "out = [1, 2]",
        "nested-fresh-call-argument": "out = [[1], [2]]",
        "variable-call-argument-control": "source = [1, 2]\nout = [1, 2]",
        "option-string": "option_length = 6\nout = 6",
        "option-nested-string": "option_length = 6\nout = 6",
        "mapped-file-direct": "out = 18",
        "option-mapped-file": "out = 18",
        "option-nested-mapped-file": "out = 18",
    }
    invalid_process_receipts = {
        "fold-alias-single-owner": (
            1,
            ("compiled ownership ledger detected invalid tuple release",),
        ),
        "fresh-call-argument": (
            1,
            ("compiled ownership ledger detected invalid list release",),
        ),
        "nested-fresh-call-argument": (
            1,
            ("compiled ownership ledger detected invalid list release",),
        ),
    }
    frozen_rows = []
    for row in rows:
        expected_exit, diagnostics = invalid_process_receipts.get(row.id, (row.expected_exit, row.diagnostic_fragments))
        frozen_rows.append(
            replace(
                row,
                ledger_receipt=receipts.get(row.id),
                expected_output=outputs.get(row.id, row.expected_output),
                expected_exit=expected_exit,
                diagnostic_fragments=diagnostics,
            )
        )
    return tuple(frozen_rows)


def mutation_manifest() -> tuple[Mutation, ...]:
    return (
        Mutation("heap-kind-without-finalizer", 1, Detector.SOURCE_CONTRACT, "future structural registry"),
        Mutation("option-host-carrier-omitted", 1, Detector.SOURCE_CONTRACT, "frozen option fixture census"),
        Mutation("mapped-file-carrier-omitted", 1, Detector.MANIFEST, "mapped-file fixture census"),
        Mutation("recursive-function-rejection-weakened", 0, Detector.EXACT_REJECTION, "exact #879 matrix"),
        Mutation("tensor-clone-omitted", 1, Detector.LEDGER_LEAK, "aggregate tensor balance"),
        Mutation("branch-clone-borrowed", 2, Detector.LEDGER_LEAK, "branch fixture parity"),
        Mutation("entry-borrow-reuse-admitted", 3, Detector.SOURCE_CONTRACT, "C and HIP caller-storage negatives"),
        Mutation("metal-reuse-admitted", 3, Detector.SOURCE_CONTRACT, "typed Metal no-reuse contract"),
        Mutation("tail-drop-delayed", 3, Detector.PEAK_BOUND, "depth 32/128/288 peak bound"),
        Mutation("backend-local-ownership-predicate-restored", 2, Detector.SOURCE_CONTRACT, "verified backend boundary"),
        Mutation("ledger-release-omitted", 0, Detector.LEDGER_LEAK, "balanced process ledger test"),
        Mutation("ledger-release-duplicated", 0, Detector.INVALID_RELEASE, "duplicate-release process test"),
        Mutation("ledger-event-stream-emptied", 0, Detector.ZERO_VACUITY, "empty-ledger parser test"),
        Mutation("manifest-receipt-omitted", 0, Detector.MANIFEST, "receipt bijection"),
    )


def validate_manifest(
    fixtures: Sequence[Fixture],
    mutations: Sequence[Mutation],
    *,
    repository_root: Path = REPO_ROOT,
) -> None:
    ids = [fixture.id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise OracleFailure("fixture manifest contains duplicate identities")
    mutation_ids = [mutation.id for mutation in mutations]
    if len(mutation_ids) != len(set(mutation_ids)):
        raise OracleFailure("mutation manifest contains duplicate identities")
    if not fixtures or not mutations:
        raise OracleFailure("fixture and mutation manifests must be non-empty")
    if set(ids) != FROZEN_FIXTURE_IDS:
        raise OracleFailure(
            "frozen fixture universe drifted: "
            f"missing={sorted(FROZEN_FIXTURE_IDS - set(ids))}, "
            f"extra={sorted(set(ids) - FROZEN_FIXTURE_IDS)}"
        )
    if OWNERSHIP_CHILD_ISSUES != FROZEN_OWNERSHIP_CHILD_ISSUES:
        raise OracleFailure("frozen ownership child universe drifted")
    mutation_specs = {
        (mutation.id, mutation.activation_phase, mutation.detector, mutation.canary)
        for mutation in mutations
    }
    if mutation_specs != FROZEN_MUTATION_SPECS:
        raise OracleFailure("frozen mutation activation mapping drifted")
    declared_sources = {
        Path(fixture.source)
        for fixture in fixtures
        if fixture.source is not None
    }
    checked_in_sources = {
        path.relative_to(repository_root)
        for path in (repository_root / FIXTURE_ROOT).glob("*.ch")
        if path.is_file()
    }
    if checked_in_sources != declared_sources:
        raise OracleFailure(
            "fixture directory drift: "
            f"undeclared={sorted(str(path) for path in checked_in_sources - declared_sources)}, "
            f"missing={sorted(str(path) for path in declared_sources - checked_in_sources)}"
        )
    for fixture in fixtures:
        if not fixture.id or fixture.issue <= 0:
            raise OracleFailure(f"malformed fixture row: {fixture!r}")
        if isinstance(fixture.expected, ExpectedFailure):
            if fixture.expected.issue != fixture.issue:
                raise OracleFailure(f"{fixture.id}: expected-failure issue does not own the row")
            if fixture.expected.detector is not fixture.detector:
                raise OracleFailure(f"{fixture.id}: expected-failure detector does not match the row")
            if fixture.green_by not in range(1, 5):
                raise OracleFailure(f"{fixture.id}: expected failure has no future green phase")
            if fixture.detector in {
                Detector.LEDGER_LEAK,
                Detector.PEAK_BOUND,
                Detector.INVALID_RELEASE,
            } and fixture.ledger_receipt is None:
                raise OracleFailure(f"{fixture.id}: expected failure has no frozen ledger receipt")
            if (
                fixture.detector is Detector.NONZERO_EXIT
                and fixture.platform is not Platform.HIP_HARDWARE
                and fixture.expected_exit is None
                and not fixture.diagnostic_fragments
            ):
                raise OracleFailure(
                    f"{fixture.id}: host nonzero expected failure has no exact exit/diagnostic receipt"
                )
        if fixture.source is not None and not (repository_root / fixture.source).is_file():
            raise OracleFailure(f"{fixture.id}: missing fixture source {fixture.source}")
        if fixture.action in {Action.COMMAND, Action.HIP_HARDWARE}:
            if not fixture.command:
                raise OracleFailure(f"{fixture.id}: test command action has no argv")
            if not fixture.test_census:
                raise OracleFailure(
                    f"{fixture.id}: test command has no frozen nonempty test census"
                )
            if (
                any(not name for name in fixture.test_census)
                or len(fixture.test_census) != len(set(fixture.test_census))
                or fixture.test_census != tuple(sorted(fixture.test_census))
            ):
                raise OracleFailure(
                    f"{fixture.id}: frozen test census must be sorted, unique, and nonempty"
                )
            expected_outcomes = {TestOutcome.PASSED, TestOutcome.FAILED}
            if any(
                receipt.outcome not in expected_outcomes
                for receipt in fixture.test_receipt
            ):
                raise OracleFailure(
                    f"{fixture.id}: frozen execution receipt may only expect passed or failed"
                )
            if (
                any(
                    receipt.outcome is TestOutcome.FAILED
                    for receipt in fixture.test_receipt
                )
                and not isinstance(fixture.expected, ExpectedFailure)
            ):
                raise OracleFailure(
                    f"{fixture.id}: must-pass row expects a failing test execution"
                )
            cargo_test = fixture.command[:2] == ("cargo", "test")
            python_unittest = _python_unittest_target(fixture.command) is not None
            hip_test = (
                fixture.action is Action.HIP_HARDWARE
                and "scripts/hip_test.py" in fixture.command
            )
            if not (cargo_test or python_unittest or hip_test):
                raise OracleFailure(
                    f"{fixture.id}: command action is not a supported test runner"
                )
            if cargo_test or hip_test:
                cargo_argv = (
                    fixture.command
                    if cargo_test
                    else ("cargo", "test", *fixture.command[2:])
                )
                selectors = int("--lib" in cargo_argv) + int("--test" in cargo_argv)
                if selectors != 1:
                    raise OracleFailure(
                        f"{fixture.id}: cargo receipt requires exactly one test binary selector"
                    )
        elif fixture.command or fixture.test_receipt:
            raise OracleFailure(
                f"{fixture.id}: non-test action carries a test command or census"
            )
        if fixture.action is Action.LEDGER_BUILD_RUN and fixture.expected_output is None:
            raise OracleFailure(f"{fixture.id}: ledger executable has no exact stdout receipt")
        if fixture.action is Action.BUILD_REJECT:
            if fixture.containing_type is None or not fixture.diagnostic_fragments:
                raise OracleFailure(
                    f"{fixture.id}: recursive rejection lacks exact type/diagnostic receipt"
                )
        if fixture.detector is Detector.PEAK_BOUND and fixture.peak_bound is None:
            raise OracleFailure(f"{fixture.id}: peak detector has no bound")
        if fixture.launch and (fixture.backend is not Backend.C or fixture.issue not in {1339, 1344, 1346, 1356}):
            raise OracleFailure(f"{fixture.id}: launch membership exceeds the frozen subset")
        if fixture.external_prerequisite != (fixture.issue == 1339):
            raise OracleFailure(f"{fixture.id}: external prerequisite classification is inconsistent")
    for issue in FROZEN_OWNERSHIP_CHILD_ISSUES:
        issue_rows = [fixture for fixture in fixtures if fixture.issue == issue]
        polarities = {fixture.polarity for fixture in issue_rows}
        if polarities != {Polarity.POSITIVE, Polarity.NEGATIVE}:
            raise OracleFailure(f"issue #{issue} lacks exact positive/negative parity")
        counterparts = {
            polarity.value: frozenset(
                fixture.id
                for fixture in issue_rows
                if fixture.polarity is polarity
            )
            for polarity in Polarity
        }
        if counterparts != FROZEN_COUNTERPARTS[issue]:
            raise OracleFailure(f"issue #{issue} counterpart identities drifted")
    launch_issues = {fixture.issue for fixture in fixtures if fixture.launch}
    if launch_issues != {1339, 1344, 1346, 1356}:
        raise OracleFailure(f"launch subset drifted: {sorted(launch_issues)}")
    if not any(fixture.platform is Platform.HIP_HARDWARE for fixture in fixtures):
        raise OracleFailure("manifest has no explicit HIP hardware row")


def select_fixtures(phase: str, require_hip: bool) -> tuple[Fixture, ...]:
    fixtures = fixture_manifest()
    if phase in {"3", "4", "complete"} and not require_hip:
        raise OracleFailure(
            f"ownership phase {phase} requires --require-hip; "
            "hardware execution cannot be omitted from closure"
        )
    if phase == "launch":
        return tuple(fixture for fixture in fixtures if fixture.launch)
    if phase == "complete":
        selected = fixtures
    else:
        try:
            number = int(phase)
        except ValueError as error:
            raise OracleFailure(f"unknown ownership phase {phase!r}") from error
        if number not in range(0, 5):
            raise OracleFailure(f"ownership phase must be 0 through 4, got {number}")
        selected = fixtures
    if require_hip:
        return tuple(selected)
    return tuple(
        fixture for fixture in selected if fixture.platform is not Platform.HIP_HARDWARE
    )


def effective_expected(fixture: Fixture, phase: str) -> Expected:
    if phase in {"launch", "complete"}:
        return MustPass()
    number = int(phase)
    if number >= fixture.green_by:
        return MustPass()
    return fixture.expected


def classify_result(expected: Expected, detection: Detection, fixture_id: str) -> str:
    if isinstance(expected, MustPass):
        if not detection.passed:
            raise OracleFailure(
                f"{fixture_id}: must pass, but {detection.detector.value if detection.detector else 'unknown'} detected {detection.detail}"
            )
        return "PASS"
    if detection.passed:
        raise OracleFailure(
            f"{fixture_id}: unexpectedly passed; remove ExpectedFailure(#{expected.issue}, {expected.detector.value}) in the fixing change"
        )
    if detection.detector is not expected.detector:
        actual = detection.detector.value if detection.detector else "unknown"
        raise OracleFailure(
            f"{fixture_id}: wrong detector {actual}; expected {expected.detector.value} for issue #{expected.issue}"
        )
    return f"EXPECTED FAILURE #{expected.issue} ({expected.detector.value})"


def detect_nonzero(fixture: Fixture, returncode: int, diagnostic: str) -> Detector:
    """Return NONZERO_EXIT only when the fixture's frozen failure receipt matches."""

    if fixture.expected_exit is not None and returncode != fixture.expected_exit:
        return Detector.MANIFEST
    if fixture.diagnostic_fragments and not all(
        fragment in diagnostic for fragment in fixture.diagnostic_fragments
    ):
        return Detector.MANIFEST
    return Detector.NONZERO_EXIT


def _required_int(record: dict[str, object], key: str, context: str) -> int:
    value = record.get(key)
    if type(value) is not int or value < 0:
        raise OracleFailure(f"{context}: {key} must be a non-negative integer")
    return value


def _require_exact_keys(
    record: dict[str, object], required: frozenset[str], context: str
) -> None:
    actual = frozenset(record)
    if actual != required:
        raise OracleFailure(
            f"{context}: exact keys required; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )


def _required_site(record: dict[str, object], context: str) -> str:
    site = record.get("site")
    if not isinstance(site, str) or not site:
        raise OracleFailure(f"{context}: site must be a non-empty string")
    return site


def load_ledger(path: Path, *, minimum_allocations: int = 0) -> Ledger:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise OracleFailure(f"read ownership ledger {path}: {error}") from error
    if not lines:
        raise OracleFailure("ownership ledger is empty")
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise OracleFailure(f"ownership ledger line {index} is malformed JSON: {error}") from error
        if not isinstance(record, dict):
            raise OracleFailure(f"ownership ledger line {index} is not an object")
        records.append(record)
    if records[0] != {"event": "header", "schema": LEDGER_SCHEMA}:
        raise OracleFailure("ownership ledger header/schema mismatch")
    summaries = [record for record in records if record.get("event") == "summary"]
    if len(summaries) != 1 or records[-1] is not summaries[0]:
        raise OracleFailure("ownership ledger requires exactly one final summary")

    owners: dict[int, int] = {}
    bytes_by_id: dict[int, int] = {}
    kind_by_id: dict[int, str] = {}
    finalized: set[int] = set()
    kind_allocations: Counter[str] = Counter()
    computed_live_owners = 0
    computed_live_bytes = 0
    computed_peak = 0
    invalid_operations = 0
    invalid_events: Counter[str] = Counter()
    allocation_count = 0

    allowed_kinds = {"Tensor", "TensorStorage", "String", "List", "Tuple", "Dict", "Adt", "MappedFile"}
    for index, record in enumerate(records[1:-1], start=2):
        event = record.get("event")
        context = f"ownership ledger line {index}"
        if event == "allocate":
            _require_exact_keys(
                record,
                frozenset(
                    {
                        "event",
                        "id",
                        "kind",
                        "bytes",
                        "owners_before",
                        "owners_after",
                        "site",
                    }
                ),
                context,
            )
            _required_site(record, context)
            identity = _required_int(record, "id", context)
            if identity != allocation_count + 1 or identity in owners:
                raise OracleFailure(f"{context}: allocation identities are not deterministic and contiguous")
            kind = record.get("kind")
            if kind not in allowed_kinds:
                raise OracleFailure(f"{context}: unknown allocation kind {kind!r}")
            size = _required_int(record, "bytes", context)
            if record.get("owners_before") != 0 or record.get("owners_after") != 1:
                raise OracleFailure(f"{context}: allocation owner transition must be 0 -> 1")
            owners[identity] = 1
            bytes_by_id[identity] = size
            kind_by_id[identity] = str(kind)
            kind_allocations[str(kind)] += 1
            allocation_count += 1
            computed_live_owners += 1
            computed_live_bytes += size
            computed_peak = max(computed_peak, computed_live_bytes)
        elif event in {"retain", "release"}:
            _require_exact_keys(
                record,
                frozenset(
                    {
                        "event",
                        "id",
                        "kind",
                        "bytes",
                        "owners_before",
                        "owners_after",
                        "site",
                    }
                ),
                context,
            )
            _required_site(record, context)
            identity = _required_int(record, "id", context)
            if (
                identity not in owners
                or owners[identity] == 0
                or identity in finalized
            ):
                raise OracleFailure(f"{context}: transition references an unknown/finalized allocation")
            before = _required_int(record, "owners_before", context)
            after = _required_int(record, "owners_after", context)
            if (
                record.get("kind") != kind_by_id[identity]
                or record.get("bytes") != bytes_by_id[identity]
            ):
                raise OracleFailure(f"{context}: transition kind/bytes do not match allocation")
            delta = 1 if event == "retain" else -1
            if before != owners[identity] or after != before + delta or after < 0:
                raise OracleFailure(f"{context}: invalid {event} owner transition")
            owners[identity] = after
            computed_live_owners += delta
            if before == 0 and after == 1:
                computed_live_bytes += bytes_by_id[identity]
                computed_peak = max(computed_peak, computed_live_bytes)
            elif before == 1 and after == 0:
                computed_live_bytes -= bytes_by_id[identity]
        elif event == "resize":
            _require_exact_keys(
                record,
                frozenset(
                    {"event", "id", "kind", "bytes_before", "bytes_after", "site"}
                ),
                context,
            )
            _required_site(record, context)
            identity = _required_int(record, "id", context)
            if identity not in owners or owners[identity] == 0 or identity in finalized:
                raise OracleFailure(f"{context}: resize references a dead allocation")
            before = _required_int(record, "bytes_before", context)
            after = _required_int(record, "bytes_after", context)
            if record.get("kind") != kind_by_id[identity] or before != bytes_by_id[identity]:
                raise OracleFailure(f"{context}: resize byte transition does not match prior state")
            bytes_by_id[identity] = after
            computed_live_bytes = computed_live_bytes - before + after
            computed_peak = max(computed_peak, computed_live_bytes)
        elif event == "finalize":
            _require_exact_keys(
                record,
                frozenset(
                    {
                        "event",
                        "id",
                        "kind",
                        "bytes",
                        "owners_before",
                        "owners_after",
                        "site",
                    }
                ),
                context,
            )
            _required_site(record, context)
            identity = _required_int(record, "id", context)
            if identity not in owners or owners[identity] != 0 or identity in finalized:
                raise OracleFailure(f"{context}: finalization requires one live zero-owner allocation")
            if (
                record.get("kind") != kind_by_id[identity]
                or record.get("bytes") != bytes_by_id[identity]
                or record.get("owners_before") != 0
                or record.get("owners_after") != 0
            ):
                raise OracleFailure(f"{context}: finalization kind/bytes/owners do not match allocation")
            finalized.add(identity)
        elif event == "borrow":
            _require_exact_keys(
                record,
                frozenset({"event", "id", "kind", "site"}),
                context,
            )
            _required_site(record, context)
            identity = _required_int(record, "id", context)
            if identity not in owners or owners[identity] == 0 or identity in finalized:
                raise OracleFailure(f"{context}: borrow references a dead allocation")
            if record.get("kind") != kind_by_id[identity]:
                raise OracleFailure(f"{context}: borrow kind does not match allocation")
        elif event in {"invalid_allocate", "invalid_retain", "invalid_release", "invalid_finalize", "invalid_resize", "invalid_borrow"}:
            _require_exact_keys(
                record,
                frozenset({"event", "id", "site"}),
                context,
            )
            _required_site(record, context)
            identity = record.get("id")
            known_identity: int | None = None
            if identity is not None:
                known_identity = _required_int(record, "id", context)
                if known_identity == 0 or known_identity not in owners:
                    raise OracleFailure(
                        f"{context}: invalid-event identity must be null or a known positive integer"
                    )
            if event == "invalid_allocate":
                matches_state = (
                    known_identity is not None and known_identity not in finalized
                )
            elif known_identity is None:
                matches_state = True
            elif event == "invalid_finalize":
                matches_state = (
                    known_identity in finalized
                    or owners[known_identity] != 0
                )
            else:
                matches_state = (
                    known_identity in finalized
                    or owners[known_identity] == 0
                )
            if not matches_state:
                raise OracleFailure(
                    f"{context}: {event} does not match reconstructed state"
                )
            invalid_operations += 1
            invalid_events[str(event)] += 1
        else:
            raise OracleFailure(f"{context}: unknown event {event!r}")

    summary_record = summaries[0]
    _require_exact_keys(
        summary_record,
        frozenset(
            {
                "event",
                "allocations",
                "finalized",
                "live_owners",
                "live_bytes",
                "peak_live_bytes",
                "invalid_operations",
            }
        ),
        "ownership summary",
    )
    summary = LedgerSummary(
        allocations=_required_int(summary_record, "allocations", "ownership summary"),
        finalized=_required_int(summary_record, "finalized", "ownership summary"),
        live_owners=_required_int(summary_record, "live_owners", "ownership summary"),
        live_bytes=_required_int(summary_record, "live_bytes", "ownership summary"),
        peak_live_bytes=_required_int(summary_record, "peak_live_bytes", "ownership summary"),
        invalid_operations=_required_int(summary_record, "invalid_operations", "ownership summary"),
    )
    computed = (
        allocation_count,
        len(finalized),
        computed_live_owners,
        computed_live_bytes,
        computed_peak,
        invalid_operations,
    )
    reported = (
        summary.allocations,
        summary.finalized,
        summary.live_owners,
        summary.live_bytes,
        summary.peak_live_bytes,
        summary.invalid_operations,
    )
    missing_finalization = sorted(
        identity
        for identity, owner_count in owners.items()
        if owner_count == 0 and identity not in finalized
    )
    finalized_while_live = sorted(
        identity
        for identity, owner_count in owners.items()
        if owner_count > 0 and identity in finalized
    )
    if missing_finalization:
        raise OracleFailure(
            "ownership ledger zero-owner allocations are missing finalization: "
            f"{missing_finalization}"
        )
    if finalized_while_live:
        raise OracleFailure(
            "ownership ledger live-owner allocations were finalized: "
            f"{finalized_while_live}"
        )
    if computed != reported:
        raise OracleFailure(f"ownership ledger summary mismatch: computed={computed}, reported={reported}")
    if allocation_count < minimum_allocations:
        raise OracleFailure(
            f"ownership ledger zero-vacuity: expected at least {minimum_allocations} allocations, saw {allocation_count}"
        )
    live_kind_counts = Counter(
        kind_by_id[identity]
        for identity, count in owners.items()
        if count > 0 and identity not in finalized
    )
    return Ledger(
        tuple(records),
        summary,
        kind_allocations,
        live_kind_counts,
        invalid_events,
    )


def _command_text(argv: Sequence[str]) -> str:
    return shlex.join(str(argument) for argument in argv)


def _run(
    argv: Sequence[str],
    *,
    environment: dict[str, str],
    timeout: int = 300,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {_command_text(argv)}", flush=True)
    try:
        return subprocess.run(
            tuple(str(argument) for argument in argv),
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OracleFailure(f"command could not complete: {_command_text(argv)}: {error}") from error


class PhaseContext:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="chelis-ownership-phase0-")
        self.root = Path(self.temporary.name)
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PYO3_PYTHON": sys.executable,
                "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
                "CHELIS_STYLE_GATE_DISABLE": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        target_setting = self.environment.get("CARGO_TARGET_DIR", "target")
        target = Path(target_setting)
        self.target = target if target.is_absolute() else REPO_ROOT / target
        self.chelis = self.target / "debug" / "chelis"
        self.runtime_dir = self.root / "instrumented-runtime"
        self.runtime_dir.mkdir()
        self._prepared = False

    def close(self) -> None:
        self.temporary.cleanup()

    def prepare(self) -> None:
        if self._prepared:
            return
        for argv in (
            ("cargo", "build", "-p", "chelis-cli", "--bin", "chelis"),
            ("cargo", "build", "-p", "chelis-runtime", "--features", "ownership-ledger"),
        ):
            result = _run(argv, environment=self.environment, timeout=900)
            if result.returncode != 0:
                raise OracleFailure(
                    f"Phase 0 preparation failed: {_command_text(argv)}\n{result.stdout}\n{result.stderr}"
                )
        archive = self.target / "debug" / "libchelis_runtime.a"
        if not self.chelis.is_file() or not archive.is_file():
            raise OracleFailure("Phase 0 preparation did not produce chelis and the runtime archive")
        shutil.copy2(archive, self.runtime_dir / archive.name)
        self._prepared = True

    def source_copy(self, fixture: Fixture) -> Path:
        if fixture.source is None:
            raise OracleFailure(f"{fixture.id}: action requires a source fixture")
        source = (REPO_ROOT / fixture.source).read_text()
        directory = self.root / fixture.id
        directory.mkdir(exist_ok=True)
        if "__MAPPED_FILE_PATH__" in source:
            mapped = directory / "mapped.bin"
            mapped.write_bytes(b"compiled-ownership")
            source = source.replace("__MAPPED_FILE_PATH__", mapped.name)
        path = directory / f"{fixture.id.replace('-', '_')}.ch"
        path.write_text(source)
        return path

    def build(self, fixture: Fixture) -> BuildArtifact:
        self.prepare()
        source = self.source_copy(fixture)
        output = source.parent / "out"
        environment = self.environment.copy()
        environment["CHELIS_RUNTIME_DIR"] = str(self.runtime_dir)
        result = _run(
            (
                str(self.chelis),
                "build",
                str(source),
                "--target",
                fixture.backend.value,
                "--output",
                str(output),
            ),
            environment=environment,
        )
        compile_command: tuple[str, ...] | None = None
        for line in result.stdout.splitlines():
            if line.startswith("Compile: "):
                compile_command = tuple(shlex.split(line.removeprefix("Compile: ")))
        return BuildArtifact(output, result, compile_command)

    def compile_and_run(
        self, fixture: Fixture, artifact: BuildArtifact, *, ledger: bool
    ) -> tuple[subprocess.CompletedProcess[str], Path | None]:
        if artifact.compile_command is None:
            raise OracleFailure(f"{fixture.id}: successful build emitted no exact Compile command")
        compile_result = _run(
            artifact.compile_command,
            environment=self.environment,
        )
        if compile_result.returncode != 0:
            raise OracleFailure(
                f"{fixture.id}: native compile failed\n{compile_result.stdout}\n{compile_result.stderr}"
            )
        try:
            output_index = artifact.compile_command.index("-o") + 1
            executable = Path(artifact.compile_command[output_index])
        except (ValueError, IndexError) as error:
            raise OracleFailure(f"{fixture.id}: Compile command has no -o target") from error
        environment = self.environment.copy()
        ledger_path: Path | None = None
        if ledger:
            ledger_path = artifact.output_dir.parent / "ownership-ledger.jsonl"
            ledger_path.unlink(missing_ok=True)
            environment["CHELIS_OWNERSHIP_LEDGER_PATH"] = str(ledger_path)
        result = _run(
            (str(executable),),
            environment=environment,
            timeout=120,
            cwd=artifact.output_dir.parent,
        )
        return result, ledger_path


def assert_ledger_receipt(fixture: Fixture, ledger: Ledger) -> None:
    receipt = fixture.ledger_receipt
    if receipt is None:
        return
    observed = {
        "live_owners": ledger.summary.live_owners,
        "live_bytes": ledger.summary.live_bytes,
        "live_kinds": tuple(sorted(ledger.live_kind_counts.items())),
        "peak_live_bytes": ledger.summary.peak_live_bytes,
        "invalid_operations": ledger.summary.invalid_operations,
        "invalid_event": (
            next(iter(ledger.invalid_events))
            if len(ledger.invalid_events) == 1
            else None
        ),
    }
    expected = {
        "live_owners": receipt.live_owners,
        "live_bytes": receipt.live_bytes,
        "live_kinds": receipt.live_kinds or None,
        "peak_live_bytes": receipt.peak_live_bytes,
        "invalid_operations": receipt.invalid_operations,
        "invalid_event": receipt.invalid_event,
    }
    drift = {
        field: (value, observed[field])
        for field, value in expected.items()
        if value is not None and value != observed[field]
    }
    if drift:
        raise OracleFailure(f"{fixture.id}: frozen ledger receipt drift: {drift}")


def _process_receipt_detection(
    fixture: Fixture, run: subprocess.CompletedProcess[str]
) -> Detection | None:
    expected_exit = fixture.expected_exit if fixture.expected_exit is not None else 0
    diagnostic = f"{run.stdout}\n{run.stderr}"
    if run.returncode != expected_exit:
        return Detection.failure(
            Detector.NONZERO_EXIT,
            f"expected exit {expected_exit}, got {run.returncode}: {run.stderr.strip()}",
        )
    if fixture.diagnostic_fragments and not all(
        fragment in diagnostic for fragment in fixture.diagnostic_fragments
    ):
        return Detection.failure(
            Detector.MANIFEST,
            "compiled process diagnostic drifted from its frozen receipt",
        )
    if fixture.expected_output is None:
        return Detection.failure(
            Detector.MANIFEST,
            "compiled process has no frozen stdout receipt",
        )
    if run.stdout.strip() != fixture.expected_output:
        return Detection.failure(
            Detector.OUTPUT_MISMATCH,
            f"expected stdout={fixture.expected_output!r}; got {run.stdout.strip()!r}",
        )
    return None


def _ledger_detection(fixture: Fixture, run: subprocess.CompletedProcess[str], path: Path) -> Detection:
    process_failure = _process_receipt_detection(fixture, run)
    if process_failure is not None:
        return process_failure
    ledger = load_ledger(path, minimum_allocations=1)
    if ledger.summary.invalid_operations:
        if set(ledger.invalid_events) == {"invalid_release"}:
            detection = Detection.failure(
                Detector.INVALID_RELEASE,
                f"{ledger.summary.invalid_operations} invalid release operation(s)",
            )
        else:
            detection = Detection.failure(
                Detector.MANIFEST,
                f"wrong invalid ownership event class: {dict(ledger.invalid_events)}",
            )
    elif fixture.peak_bound is not None and ledger.summary.peak_live_bytes > fixture.peak_bound:
        detection = Detection.failure(
            Detector.PEAK_BOUND,
            f"peak live bytes {ledger.summary.peak_live_bytes} exceed bound {fixture.peak_bound}",
        )
    elif ledger.summary.live_owners or ledger.summary.live_bytes:
        detection = Detection.failure(
            Detector.LEDGER_LEAK,
            f"live owners={ledger.summary.live_owners}, live bytes={ledger.summary.live_bytes}, kinds={dict(ledger.live_kind_counts)}",
        )
    else:
        detection = Detection.success(
            f"balanced {ledger.summary.allocations} allocations; peak={ledger.summary.peak_live_bytes}"
        )
    if detection.detector is fixture.detector:
        assert_ledger_receipt(fixture, ledger)
    return detection


def _python_unittest_target(argv: Sequence[str]) -> str | None:
    if (
        len(argv) == 5
        and argv[0] in {"{python}", sys.executable}
        and tuple(argv[1:4]) == ("-m", "unittest", "-v")
        and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", argv[4])
        and argv[4].rsplit(".", 1)[-1].startswith("test_")
    ):
        return argv[4]
    return None


def _test_list_command(fixture: Fixture, argv: Sequence[str]) -> tuple[str, ...]:
    if fixture.action is Action.HIP_HARDWARE:
        if len(argv) < 3:
            raise OracleFailure(f"{fixture.id}: malformed HIP hardware command")
        command = ["cargo", "test", *argv[2:]]
    elif (module := _python_unittest_target(argv)) is not None:
        module_path = Path(*module.split(".")).with_suffix(".py")
        return (argv[0], str(module_path), "--list-tests")
    else:
        command = list(argv)
    if command[:2] != ["cargo", "test"]:
        raise OracleFailure(f"{fixture.id}: unsupported test-list command")
    if "--" in command:
        command.append("--list")
    else:
        command.extend(("--", "--list"))
    return tuple(command)


_CARGO_TEST_SUMMARY = re.compile(
    r"^test result: (?P<overall>ok|FAILED)\. "
    r"(?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored;"
)
_PYTHON_TEST_SUMMARY = re.compile(r"^Ran (?P<count>\d+) tests? in ")


def _record_test_outcome(
    receipts: dict[str, TestOutcome], name: str, outcome: TestOutcome
) -> None:
    if name in receipts:
        raise OracleFailure(f"duplicate execution receipt for test {name!r}")
    receipts[name] = outcome


def _cargo_test_execution_receipt(output: str) -> tuple[TestCaseReceipt, ...]:
    receipts: dict[str, TestOutcome] = {}
    summaries: list[tuple[str, int, int, int]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        summary = _CARGO_TEST_SUMMARY.match(line)
        if summary is not None:
            summaries.append(
                (
                    summary.group("overall"),
                    int(summary.group("passed")),
                    int(summary.group("failed")),
                    int(summary.group("ignored")),
                )
            )
            continue
        if not line.startswith("test ") or " ... " not in line:
            continue
        name, status = line.removeprefix("test ").rsplit(" ... ", 1)
        outcomes = {
            "ok": TestOutcome.PASSED,
            "FAILED": TestOutcome.FAILED,
            "ignored": TestOutcome.IGNORED,
        }
        outcome = outcomes.get(status)
        if outcome is None:
            raise OracleFailure(
                f"unrecognized Cargo test execution status {status!r} for {name!r}"
            )
        _record_test_outcome(receipts, name, outcome)
    if len(summaries) != 1:
        raise OracleFailure(
            f"Cargo test execution requires exactly one summary, saw {summaries!r}"
        )
    overall, passed, failed, ignored = summaries[0]
    observed_counts = (
        sum(outcome is TestOutcome.PASSED for outcome in receipts.values()),
        sum(outcome is TestOutcome.FAILED for outcome in receipts.values()),
        sum(outcome is TestOutcome.IGNORED for outcome in receipts.values()),
    )
    if (passed, failed, ignored) != observed_counts:
        raise OracleFailure(
            "Cargo test result counts do not match per-test receipts: "
            f"summary={(passed, failed, ignored)}, observed={observed_counts}"
        )
    expected_overall = "FAILED" if failed else "ok"
    if overall != expected_overall:
        raise OracleFailure(
            f"Cargo test result status {overall!r} disagrees with failed={failed}"
        )
    return tuple(
        TestCaseReceipt(name, outcome)
        for name, outcome in sorted(receipts.items())
    )


def _python_test_execution_receipt(
    output: str, module: str
) -> tuple[TestCaseReceipt, ...]:
    receipts: dict[str, TestOutcome] = {}
    ran_counts: list[int] = []
    marker = f" ({module}."
    for raw_line in output.splitlines():
        line = raw_line.strip()
        summary = _PYTHON_TEST_SUMMARY.match(line)
        if summary is not None:
            ran_counts.append(int(summary.group("count")))
            continue
        if marker not in line or " ... " not in line:
            continue
        left, status = line.rsplit(" ... ", 1)
        display_name, name_with_suffix = left.split(marker, 1)
        if not name_with_suffix.endswith(")"):
            raise OracleFailure(f"malformed unittest execution line {line!r}")
        name = name_with_suffix.removesuffix(")")
        if display_name != name.rsplit(".", 1)[-1]:
            raise OracleFailure(
                f"unittest display name {display_name!r} disagrees with {name!r}"
            )
        if status == "ok":
            outcome = TestOutcome.PASSED
        elif status in {"FAIL", "ERROR"}:
            outcome = TestOutcome.FAILED
        elif status.startswith("skipped "):
            outcome = TestOutcome.SKIPPED
        else:
            raise OracleFailure(
                f"unrecognized unittest execution status {status!r} for {name!r}"
            )
        _record_test_outcome(receipts, name, outcome)
    if ran_counts != [len(receipts)]:
        raise OracleFailure(
            "unittest execution count does not match per-test receipts: "
            f"summary={ran_counts!r}, observed={len(receipts)}"
        )
    return tuple(
        TestCaseReceipt(name, outcome)
        for name, outcome in sorted(receipts.items())
    )


def _test_execution_receipt(
    fixture: Fixture,
    argv: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> tuple[TestCaseReceipt, ...]:
    output = f"{result.stdout}\n{result.stderr}"
    if (
        fixture.action is Action.COMMAND
        and (module := _python_unittest_target(argv)) is not None
    ):
        return _python_test_execution_receipt(output, module)
    return _cargo_test_execution_receipt(output)


def _execute_command(context: PhaseContext, fixture: Fixture) -> Detection:
    argv = tuple(sys.executable if item == "{python}" else item for item in fixture.command)
    listed = _run(
        _test_list_command(fixture, argv),
        environment=context.environment,
        timeout=1200,
    )
    if listed.returncode != 0:
        return Detection.failure(
            Detector.NONZERO_EXIT,
            f"test-list preflight failed: {listed.stderr.strip()}",
        )
    names = tuple(
        line.removesuffix(": test")
        for line in listed.stdout.splitlines()
        if line.endswith(": test")
    )
    if names != fixture.test_census:
        return Detection.failure(
            Detector.ZERO_VACUITY,
            f"expected exact test census {fixture.test_census!r}, listed={names}",
        )
    result = _run(argv, environment=context.environment, timeout=1200)
    try:
        observed_receipt = _test_execution_receipt(fixture, argv, result)
    except OracleFailure as error:
        return Detection.failure(
            Detector.ZERO_VACUITY,
            f"test execution receipt is malformed or incomplete: {error}",
        )
    if observed_receipt != fixture.test_receipt:
        return Detection.failure(
            Detector.ZERO_VACUITY,
            "frozen test execution receipt drifted: "
            f"expected={fixture.test_receipt!r}, observed={observed_receipt!r}",
        )
    expected_failure = any(
        receipt.outcome is TestOutcome.FAILED for receipt in fixture.test_receipt
    )
    if result.returncode != 0:
        diagnostic = f"{result.stdout}\n{result.stderr}"
        if expected_failure:
            return Detection.failure(
                fixture.detector,
                "exact frozen behavioral test receipt failed for the current known issue",
            )
        return Detection.failure(
            detect_nonzero(fixture, result.returncode, diagnostic),
            f"command exited {result.returncode}: {result.stderr.strip()}",
        )
    if expected_failure:
        return Detection.failure(
            Detector.MANIFEST,
            "frozen failing test receipt exited zero",
        )
    return Detection.success(
        f"executed exact test receipt for {len(fixture.test_receipt)} test(s)"
    )


def execute_fixture(context: PhaseContext, fixture: Fixture) -> Detection:
    if fixture.action in {Action.COMMAND, Action.HIP_HARDWARE}:
        return _execute_command(context, fixture)
    if fixture.action is Action.CHECK_REJECT:
        context.prepare()
        source = context.source_copy(fixture)
        result = _run(
            (str(context.chelis), "check", str(source)),
            environment=context.environment,
        )
        diagnostic = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 and "unbound variable" in diagnostic:
            return Detection.success("unannotated forward capture rejects as unbound")
        return Detection.failure(
            Detector.EXACT_REJECTION,
            f"expected exact unbound-variable rejection, exit={result.returncode}",
        )

    artifact = context.build(fixture)
    if fixture.action is Action.BUILD_REJECT:
        diagnostic = f"{artifact.build.stdout}\n{artifact.build.stderr}"
        emitted = list(artifact.output_dir.glob("*")) if artifact.output_dir.is_dir() else []
        source = (REPO_ROOT / fixture.source).read_text() if fixture.source else ""
        if (
            artifact.build.returncode != 0
            and fixture.containing_type is not None
            and fixture.containing_type in source
            and all(fragment in diagnostic for fragment in fixture.diagnostic_fragments)
            and not emitted
        ):
            return Detection.success("exact recursive function-container #879 rejection")
        return Detection.failure(
            Detector.EXACT_REJECTION,
            f"exit={artifact.build.returncode}, emitted={[path.name for path in emitted]}, diagnostic={diagnostic.strip()}",
        )
    if artifact.build.returncode != 0:
        diagnostic = f"{artifact.build.stdout}\n{artifact.build.stderr}"
        return Detection.failure(
            detect_nonzero(fixture, artifact.build.returncode, diagnostic),
            f"chelis build exited {artifact.build.returncode}: {artifact.build.stderr.strip()}",
        )
    if fixture.action is Action.BUILD_ONLY:
        if artifact.compile_command is None:
            return Detection.failure(Detector.NONZERO_EXIT, "build emitted no compile command")
        return Detection.success("target emitted an exact contextual callback artifact")

    use_ledger = fixture.action is Action.LEDGER_BUILD_RUN
    run, ledger_path = context.compile_and_run(fixture, artifact, ledger=use_ledger)
    if use_ledger:
        assert ledger_path is not None
        return _ledger_detection(fixture, run, ledger_path)
    if run.returncode != 0:
        return Detection.failure(
            detect_nonzero(
                fixture,
                run.returncode,
                f"{run.stdout}\n{run.stderr}",
            ),
            f"compiled process exited {run.returncode}: {run.stderr.strip()}",
        )
    if (
        fixture.expected_output is not None
        and run.stdout.strip() != fixture.expected_output
    ):
        return Detection.failure(
            Detector.OUTPUT_MISMATCH,
            f"expected stdout={fixture.expected_output!r}; got {run.stdout.strip()!r}",
        )
    return Detection.success("compiled process exited zero with exact output")


def hip_available() -> tuple[bool, str]:
    hipcc = shutil.which("hipcc")
    rocminfo = shutil.which("rocminfo")
    if hipcc is None or rocminfo is None:
        return False, f"hipcc={hipcc!r}, rocminfo={rocminfo!r}"
    result = subprocess.run(
        (rocminfo,),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return False, f"rocminfo exited {result.returncode}: {result.stderr.strip()}"
    return True, f"hipcc={hipcc}, rocminfo={rocminfo}"


def run_phase(phase: str, *, require_hip: bool) -> None:
    fixtures = fixture_manifest()
    mutations = mutation_manifest()
    validate_manifest(fixtures, mutations)
    if phase in {"3", "4", "complete"} and not require_hip:
        raise OracleFailure(f"phase {phase} requires --require-hip")
    if require_hip:
        available, detail = hip_available()
        if not available:
            raise OracleFailure(f"BLOCKED: required HIP hardware/toolchain unavailable: {detail}")
    selected = select_fixtures(phase, require_hip)
    if not selected:
        raise OracleFailure(f"phase {phase} selected zero fixtures")
    context = PhaseContext()
    receipts: dict[str, str] = {}
    try:
        for fixture in selected:
            if fixture.id in receipts:
                raise OracleFailure(f"duplicate execution receipt for {fixture.id}")
            print(f"\n== {fixture.id} (issue #{fixture.issue}, {fixture.backend.value}) ==", flush=True)
            detection = execute_fixture(context, fixture)
            verdict = classify_result(effective_expected(fixture, phase), detection, fixture.id)
            receipts[fixture.id] = verdict
            print(f"{verdict}: {detection.detail}", flush=True)
    finally:
        context.close()
    expected_ids = {fixture.id for fixture in selected}
    if set(receipts) != expected_ids:
        missing = sorted(expected_ids - set(receipts))
        extra = sorted(set(receipts) - expected_ids)
        raise OracleFailure(f"receipt bijection failed: missing={missing}, extra={extra}")
    if phase == "launch":
        marker = "COMPILED VALUE OWNERSHIP LAUNCH SUBSET: PASS"
    elif phase == "complete":
        marker = "COMPILED VALUE OWNERSHIP ORACLE: PASS"
    else:
        marker = f"COMPILED VALUE OWNERSHIP PHASE {phase}: PASS"
    print(f"\n{marker}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("0", "1", "2", "3", "4", "launch", "complete"))
    parser.add_argument("--require-hip", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_phase(args.phase, require_hip=args.require_hip)
    except OracleFailure as error:
        print(f"COMPILED VALUE OWNERSHIP: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
