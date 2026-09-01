#!/usr/bin/env python3
"""Authoritative runtime-extent class oracle for chelis#1277.

The checked-in baseline records the pre-slice disposition and required exit
state of every generated corpus row. The runner validates the one-way status
lattice, proves that each named Rust receipt actually executed and passed,
and binds the result to a clean Git commit plus a canonical corpus digest.

Acceptance for every supported phase ends with::

    RUNTIME EXTENT ORACLE: PASS

Slice A also has a documented HIP hardware gate. This automatic runner checks
that the two exact ignored tests still exist; the hardware command printed in
the receipt must be run on the same commit and corpus digest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "scripts/runtime_extent_oracle_baseline.json"
PHASES = ("a", "b", "c", "final")
PASS_MARKER = "RUNTIME EXTENT ORACLE: PASS"
HIP_HARDWARE_COMMAND = (
    "scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- "
    "--ignored --test-threads=1"
)

START_STATES = frozenset(
    {"nonconforming_rejection", "silent_unguarded", "ice", "lane_divergent"}
)
TERMINAL_CONTROL = "rejects_exactly"
EXECUTES = "executes_exactly"
TYPED_UNSUPPORTED = re.compile(r"^typed_unsupported\(#(?P<issue>[1-9][0-9]*)\)$")


class OracleFailure(RuntimeError):
    """A runtime-extent acceptance obligation failed."""


@dataclass(frozen=True)
class CorpusRow:
    id: str
    baseline: str
    phase_a: str
    receipt: str


@dataclass(frozen=True)
class TestTarget:
    id: str
    argv: tuple[str, ...]
    expected_tests: tuple[str, ...] = ()
    list_only: bool = False


def _row(id: str, baseline: str, phase_a: str, receipt: str) -> CorpusRow:
    return CorpusRow(id, baseline, phase_a, receipt)


def generated_phase_a_corpus() -> tuple[CorpusRow, ...]:
    """Generate Slice A rows from its representation, lane, and wire matrix."""

    rows = (
        _row(
            "expand.binder.eval_c",
            "ice",
            EXECUTES,
            "cli.bare_dimension_binder_executes_and_builds_without_symbolic_dim_ice",
        ),
        _row(
            "expand.diagnostic.extent_guidance",
            "nonconforming_rejection",
            EXECUTES,
            "cli.stale_extent_guidance_is_removed_but_axis_guidance_stays_int32",
        ),
        _row(
            "expand.input_axis.eval",
            EXECUTES,
            EXECUTES,
            "ir.input_axis_expand_verifies_and_evaluates_from_shape_metadata",
        ),
        _row(
            "expand.input_axis.metadata_only",
            EXECUTES,
            EXECUTES,
            "ir.input_axis_eval_does_not_read_tensor_elements",
        ),
        _row(
            "expand.input_axis.hip_codegen",
            EXECUTES,
            EXECUTES,
            "hip_codegen.s5_input_axis_expand_reads_witness_metadata",
        ),
        _row(
            "expand.input_axis.hip_execute",
            EXECUTES,
            EXECUTES,
            "hip_manual_inventory.g5_input_axis_expand_executes_from_witness_metadata",
        ),
        _row(
            "expand.input_axis.metal_device",
            "lane_divergent",
            "lane_divergent",
            "metal_accept.compiler::metal_runtime_dim_reject_tests::metal_seam_accepts_input_axis_expand_extent",
        ),
        _row(
            "expand.input_axis.owner_matrix",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "ir.input_axis_owner_and_slot_validation_fail_closed",
        ),
        _row(
            "expand.literal_zero.eval_ir",
            "nonconforming_rejection",
            EXECUTES,
            "ir.zero_literal_expand_is_valid_and_empty",
        ),
        _row(
            "expand.literal_zero.eval_c",
            "nonconforming_rejection",
            EXECUTES,
            "cli.zero_extent_is_check_clean_and_evaluates_to_empty_tensor",
        ),
        _row(
            "expand.literal.hip_execute",
            EXECUTES,
            EXECUTES,
            "hip_manual_inventory.g5_expand_add_stride_zero",
        ),
        _row(
            "expand.negative.static",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "cli.negative_extent_remains_a_static_type_error",
        ),
        _row(
            "expand.node.eval",
            EXECUTES,
            EXECUTES,
            "ir.node_expand_verifies_and_evaluates_from_int64_scalar_input",
        ),
        _row(
            "expand.node.hip_device",
            "lane_divergent",
            "typed_unsupported(#1298)",
            "hip_reject.compiler::tests::hip_seam_rejects_node_valued_expand_with_issue_1298_receipt",
        ),
        _row(
            "expand.node.metal_device",
            "lane_divergent",
            "typed_unsupported(#1383)",
            "metal_reject.compiler::metal_runtime_dim_reject_tests::metal_seam_rejects_node_valued_expand_with_issue_1383_receipt",
        ),
        _row(
            "expand.rank_ascription",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli.shape_sourced_expand_rejects_wrong_rank_ascription",
        ),
        _row(
            "grad.input_axis.runtime_movement_source.eval_c",
            EXECUTES,
            EXECUTES,
            "symbolic_window.issue_368_runtime_symbolic_window_grad_is_half_everywhere",
        ),
        _row(
            "grad_vmap.backward_expand.eval_c",
            "ice",
            EXECUTES,
            "rank_poly.issue_383_vmap_two_stage_named_reduce_regression_matrix",
        ),
        _row(
            "ir.movement_input_cardinality",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "ir.movement_ops_reject_unowned_runtime_extent_inputs",
        ),
        _row(
            "vmap.element_derived_extent",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "ir.vmap_rejects_element_derived_extent",
        ),
        _row(
            "vmap.input_axis_shift",
            "silent_unguarded",
            EXECUTES,
            "ir.input_axis_vmap_shifts_literal_axis",
        ),
        _row(
            "vmap.shared_shape_bound",
            "silent_unguarded",
            EXECUTES,
            "ir.vmap_keeps_shape_bound_shared_and_shifts_its_axis",
        ),
        _row(
            "vmap.shared_shape_bound_ordinary_use",
            "silent_unguarded",
            EXECUTES,
            "ir.vmap_shares_shape_extent_across_bound_and_ordinary_uses",
        ),
        _row(
            "wire.capacity_census",
            EXECUTES,
            EXECUTES,
            "wire_capacity.wire_schema_numeric_fields_match_the_reviewed_baseline",
        ),
        _row(
            "wire.input_axis.invalid_owner_axis",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "wire.v7_input_axis_rejects_negative_or_out_of_range_axes_and_forbidden_owners",
        ),
        _row(
            "wire.input_axis.round_trip",
            "nonconforming_rejection",
            EXECUTES,
            "wire.v7_input_axis_round_trips_as_typed_structure",
        ),
        _row(
            "wire.legacy_v6.rejected",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "wire.v6_display_string_expand_payload_is_rejected_before_op_decode",
        ),
        _row(
            "wire.movement_input_cardinality",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "wire.v7_movement_ops_reject_unowned_runtime_extent_inputs",
        ),
        _row(
            "wire.node.invalid_carrier",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "wire.v7_expand_rejects_forbidden_carriers_slots_and_cardinality",
        ),
        _row(
            "wire.node.round_trip",
            "nonconforming_rejection",
            EXECUTES,
            "wire.v7_node_extent_round_trips_only_from_rank_zero_int64",
        ),
    )
    return tuple(sorted(rows, key=lambda row: row.id))


def canonical_corpus_bytes(rows: Sequence[CorpusRow]) -> bytes:
    payload = [
        {
            "baseline": row.baseline,
            "id": row.id,
            "phase_a": row.phase_a,
            "receipt": row.receipt,
        }
        for row in rows
    ]
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def corpus_digest(rows: Sequence[CorpusRow]) -> str:
    return hashlib.sha256(canonical_corpus_bytes(rows)).hexdigest()


def _capacity_tests() -> tuple[str, ...]:
    return (
        "a_typed_permanent_disposition_cannot_move_between_families",
        "adding_or_removing_a_public_serialized_f64_field_changes_the_census",
        "capacity_census_authority::tests::duplicate_numeric_registration_is_not_one_authority",
        "capacity_census_authority::tests::each_exact_final_class_is_recognized",
        "capacity_census_authority::tests::nonnumeric_registration_cannot_hide_numeric_flags",
        "capacity_census_authority::tests::numeric_registration_requires_exact_normative_atom_and_anchor",
        "capacity_census_authority::tests::tagged_transport_registration_is_exact_in_every_descriptor_field",
        "capacity_census_authority::tests::zero_or_multiple_final_classes_fail",
        "count_wire_axes_are_registered_without_inheriting_the_permanent_disposition",
        "managed_python::tests::absent_configuration_uses_checkout_venv",
        "managed_python::tests::existing_non_python_configuration_is_rejected",
        "managed_python::tests::external_configured_interpreter_wins",
        "managed_python::tests::invalid_explicit_configuration_does_not_fall_back",
        "managed_python::tests::missing_fallback_diagnostic_names_path_and_uv_setup",
        "managed_python::tests::relative_configured_interpreter_resolves_from_workspace",
        "permanent_wire_disposition_is_bound_to_the_complete_descriptor_set",
        "wire_schema_numeric_fields_match_the_reviewed_baseline",
    )


def automatic_targets(python: str = sys.executable) -> tuple[TestTarget, ...]:
    return (
        TestTarget("self_tests", (python, "scripts/test_runtime_extent_oracle.py")),
        TestTarget(
            "ir",
            ("cargo", "test", "-p", "chelis-ir", "--test", "runtime_extent_slice_a", "--", "--nocapture"),
            (
                "input_axis_eval_does_not_read_tensor_elements",
                "input_axis_expand_verifies_and_evaluates_from_shape_metadata",
                "input_axis_owner_and_slot_validation_fail_closed",
                "input_axis_vmap_shifts_literal_axis",
                "movement_ops_reject_unowned_runtime_extent_inputs",
                "node_expand_verifies_and_evaluates_from_int64_scalar_input",
                "vmap_keeps_shape_bound_shared_and_shifts_its_axis",
                "vmap_rejects_element_derived_extent",
                "vmap_shares_shape_extent_across_bound_and_ordinary_uses",
                "zero_literal_expand_is_valid_and_empty",
            ),
        ),
        TestTarget(
            "cli",
            ("cargo", "test", "-p", "chelis-cli", "--test", "runtime_extent_slice_a", "--", "--nocapture"),
            (
                "bare_dimension_binder_executes_and_builds_without_symbolic_dim_ice",
                "negative_extent_remains_a_static_type_error",
                "shape_sourced_expand_rejects_wrong_rank_ascription",
                "stale_extent_guidance_is_removed_but_axis_guidance_stays_int32",
                "zero_extent_is_check_clean_and_evaluates_to_empty_tensor",
            ),
        ),
        TestTarget(
            "rank_poly",
            (
                "cargo", "test", "-p", "chelis-cli", "--test", "rank_poly_tier3",
                "issue_383_vmap_two_stage_named_reduce_regression_matrix", "--", "--exact", "--nocapture",
            ),
            ("issue_383_vmap_two_stage_named_reduce_regression_matrix",),
        ),
        TestTarget(
            "symbolic_window",
            (
                "cargo", "test", "-p", "chelis-cli", "--test",
                "issue_368_grad_concat_windows",
                "issue_368_runtime_symbolic_window_grad_is_half_everywhere",
                "--", "--exact", "--nocapture",
            ),
            ("issue_368_runtime_symbolic_window_grad_is_half_everywhere",),
        ),
        TestTarget(
            "wire",
            ("cargo", "test", "-p", "chelis-compiler-api", "--test", "wire_dag_v7_runtime_extents", "--", "--nocapture"),
            (
                "v6_display_string_expand_payload_is_rejected_before_op_decode",
                "v7_expand_rejects_forbidden_carriers_slots_and_cardinality",
                "v7_input_axis_rejects_negative_or_out_of_range_axes_and_forbidden_owners",
                "v7_input_axis_round_trips_as_typed_structure",
                "v7_movement_ops_reject_unowned_runtime_extent_inputs",
                "v7_node_extent_round_trips_only_from_rank_zero_int64",
            ),
        ),
        TestTarget(
            "wire_capacity",
            ("cargo", "test", "-p", "chelis-compiler-api", "--test", "capacity_census_wire", "--", "--nocapture"),
            _capacity_tests(),
        ),
        TestTarget(
            "hip_codegen",
            (
                "cargo", "test", "-p", "chelis-backend-hip", "--test", "codegen_structure",
                "s5_input_axis_expand_reads_witness_metadata", "--", "--exact", "--nocapture",
            ),
            ("s5_input_axis_expand_reads_witness_metadata",),
        ),
        TestTarget(
            "hip_accept",
            (
                "cargo", "test", "-p", "chelis-compiler-api", "--lib",
                "compiler::tests::hip_seam_accepts_input_axis_expand_extent", "--", "--exact", "--nocapture",
            ),
            ("compiler::tests::hip_seam_accepts_input_axis_expand_extent",),
        ),
        TestTarget(
            "hip_reject",
            (
                "cargo", "test", "-p", "chelis-compiler-api", "--lib",
                "compiler::tests::hip_seam_rejects_node_valued_expand_with_issue_1298_receipt",
                "--", "--exact", "--nocapture",
            ),
            ("compiler::tests::hip_seam_rejects_node_valued_expand_with_issue_1298_receipt",),
        ),
        TestTarget(
            "metal_accept",
            (
                "cargo", "test", "-p", "chelis-compiler-api", "--lib",
                "compiler::metal_runtime_dim_reject_tests::metal_seam_accepts_input_axis_expand_extent",
                "--", "--exact", "--nocapture",
            ),
            ("compiler::metal_runtime_dim_reject_tests::metal_seam_accepts_input_axis_expand_extent",),
        ),
        TestTarget(
            "metal_reject",
            (
                "cargo", "test", "-p", "chelis-compiler-api", "--lib",
                "compiler::metal_runtime_dim_reject_tests::metal_seam_rejects_node_valued_expand_with_issue_1383_receipt",
                "--", "--exact", "--nocapture",
            ),
            ("compiler::metal_runtime_dim_reject_tests::metal_seam_rejects_node_valued_expand_with_issue_1383_receipt",),
        ),
        TestTarget(
            "hip_manual_inventory",
            (
                "cargo", "test", "-p", "chelis-backend-hip", "--test", "gpu_correctness",
                "g5_", "--", "--ignored", "--list",
            ),
            (
                "g5_expand_add_stride_zero",
                "g5_input_axis_expand_executes_from_witness_metadata",
            ),
            list_only=True,
        ),
    )


def _status_class(status: str) -> str:
    if status in START_STATES or status in {TERMINAL_CONTROL, EXECUTES}:
        return status
    if TYPED_UNSUPPORTED.fullmatch(status):
        return "typed_unsupported"
    raise OracleFailure(f"unknown runtime-extent status {status!r}")


def validate_transition(before: str, after: str) -> None:
    before_class = _status_class(before)
    after_class = _status_class(after)
    if before == after:
        return
    if before_class in START_STATES and after_class in {
        "typed_unsupported",
        EXECUTES,
        TERMINAL_CONTROL,
    }:
        return
    if before_class == "typed_unsupported" and after_class == EXECUTES:
        return
    raise OracleFailure(f"forbidden status transition {before!r} -> {after!r}")


def load_and_validate_baseline(
    path: Path = BASELINE_PATH, rows: Sequence[CorpusRow] | None = None
) -> Mapping[str, object]:
    generated = tuple(rows if rows is not None else generated_phase_a_corpus())
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OracleFailure(f"cannot read baseline {path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "corpus_sha256",
        "rows",
    }:
        raise OracleFailure("baseline must contain exactly schema_version, corpus_sha256, rows")
    if payload["schema_version"] != 1:
        raise OracleFailure(f"unsupported baseline schema {payload['schema_version']!r}")
    expected_digest = corpus_digest(generated)
    if payload["corpus_sha256"] != expected_digest:
        raise OracleFailure(
            "baseline corpus digest drift: "
            f"expected {expected_digest}, got {payload['corpus_sha256']!r}"
        )
    entries = payload["rows"]
    if not isinstance(entries, list):
        raise OracleFailure("baseline rows must be a list")
    observed: list[CorpusRow] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "baseline",
            "phase_a",
            "receipt",
        }:
            raise OracleFailure(f"baseline row {index} has the wrong fields")
        try:
            row = CorpusRow(**entry)
        except TypeError as error:
            raise OracleFailure(f"baseline row {index} is malformed: {error}") from error
        _status_class(row.baseline)
        _status_class(row.phase_a)
        validate_transition(row.baseline, row.phase_a)
        observed.append(row)
    if tuple(observed) != generated:
        raise OracleFailure("baseline rows differ from the generated corpus")
    return payload


_TEST_LINE = re.compile(r"^test (?P<name>.+) \.\.\. (?P<outcome>ok|FAILED|ignored)$")
_LIST_LINE = re.compile(r"^(?P<name>.+): test$")


def parse_test_receipt(output: str, *, list_only: bool = False) -> tuple[str, ...]:
    names: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = (_LIST_LINE if list_only else _TEST_LINE).fullmatch(line)
        if match is None:
            continue
        if not list_only and match.group("outcome") != "ok":
            raise OracleFailure(
                f"test {match.group('name')!r} did not pass: {match.group('outcome')}"
            )
        name = match.group("name")
        if name in names:
            raise OracleFailure(f"duplicate execution receipt for test {name!r}")
        names.append(name)
    if not names:
        raise OracleFailure("test command produced zero per-test receipts")
    return tuple(sorted(names))


def validate_target_receipt(target: TestTarget, completed: subprocess.CompletedProcess[str]) -> None:
    if completed.returncode != 0:
        raise OracleFailure(
            f"{target.id} exited {completed.returncode}: {' '.join(target.argv)}\n{completed.stderr}"
        )
    if not target.expected_tests:
        return
    observed = parse_test_receipt(
        f"{completed.stdout}\n{completed.stderr}", list_only=target.list_only
    )
    expected = tuple(sorted(target.expected_tests))
    if observed != expected:
        raise OracleFailure(
            f"{target.id} test receipt drift: expected {expected!r}, got {observed!r}"
        )


def validate_receipt_coverage(rows: Sequence[CorpusRow], targets: Sequence[TestTarget]) -> None:
    available = {
        f"{target.id}.{test}"
        for target in targets
        for test in target.expected_tests
    }
    missing = sorted({row.receipt for row in rows} - available)
    if missing:
        raise OracleFailure(f"corpus rows have no executable receipt: {missing}")


def _run_text(
    runner: Callable[..., subprocess.CompletedProcess[str]], argv: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    return runner(
        argv,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def exact_head_receipt(
    runner: Callable[..., subprocess.CompletedProcess[str]], *, require_clean: bool
) -> str:
    head = _run_text(runner, ("git", "rev-parse", "HEAD"))
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}\n?", head.stdout):
        raise OracleFailure(f"cannot resolve exact Git head: {head.stderr.strip()}")
    status = _run_text(runner, ("git", "status", "--porcelain"))
    if status.returncode != 0:
        raise OracleFailure(f"cannot inspect worktree status: {status.stderr.strip()}")
    if require_clean and status.stdout.strip():
        raise OracleFailure("authoritative oracle requires a clean exact-head worktree")
    return head.stdout.strip()


def validate(
    phase: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    baseline_path: Path = BASELINE_PATH,
    targets: Sequence[TestTarget] | None = None,
    require_clean: bool = True,
) -> tuple[str, str]:
    if phase not in PHASES:
        raise OracleFailure(f"unsupported phase {phase!r}")
    if phase != "a":
        raise OracleFailure(
            f"phase {phase!r} corpus is not implemented; only Slice A can report PASS"
        )
    rows = generated_phase_a_corpus()
    load_and_validate_baseline(baseline_path, rows)
    selected_targets = tuple(targets if targets is not None else automatic_targets())
    validate_receipt_coverage(rows, selected_targets)
    head = exact_head_receipt(runner, require_clean=require_clean)
    for target in selected_targets:
        completed = _run_text(runner, target.argv)
        validate_target_receipt(target, completed)
    digest = corpus_digest(rows)
    print(f"runtime_extent_head={head}", flush=True)
    print(f"runtime_extent_corpus_sha256={digest}", flush=True)
    print(f"runtime_extent_hip_manual_gate={HIP_HARDWARE_COMMAND}", flush=True)
    print(PASS_MARKER, flush=True)
    return head, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=PHASES)
    args = parser.parse_args(argv)
    try:
        validate(args.phase)
    except OracleFailure as error:
        print(f"RUNTIME EXTENT ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
