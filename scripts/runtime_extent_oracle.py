#!/usr/bin/env python3
"""Authoritative runtime-extent class oracle for chelis#1277.

The checked-in baselines record the pre-slice disposition and current exit
state of every generated corpus row. The runner validates the one-way status
lattice across the whole recorded chain, proves that each named Rust receipt
actually executed and passed, and binds the result to a clean Git commit plus
a canonical corpus digest.

Acceptance for every registered phase ends with::

    RUNTIME EXTENT ORACLE: PASS

Each slice registers its own corpus, baseline file, and test targets in
``PHASE_REGISTRY``. A phase that is not registered refuses to report success.
A registered phase reports PASS only when every row it owns has reached an
exit state (``executes_exactly``, ``rejects_exactly``, or a
``typed_unsupported(#N)`` receipt) or is recorded as deferred with the reason
its owning design-doc clause states; otherwise the run exits nonzero and
lists the rows still short of exit.

Slice A also has a documented HIP hardware gate. This automatic runner checks
that the two exact ignored tests still exist; the hardware command printed in
the receipt must be run on the same commit and corpus digest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "scripts/runtime_extent_oracle_baseline.json"
BASELINE_PATH_PHASE_B = REPO_ROOT / "scripts/runtime_extent_oracle_baseline_phase_b.json"
PHASES = ("a", "b", "c", "final")
SLICE_PHASES = ("a", "b", "c")
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
EXIT_CLASSES = frozenset({EXECUTES, TERMINAL_CONTROL, "typed_unsupported"})


class OracleFailure(RuntimeError):
    """A runtime-extent acceptance obligation failed."""


@dataclass(frozen=True)
class CorpusRow:
    """One corpus row: its recorded `main` state and its state in one phase.

    ``exit_state`` is serialized under the owning phase's ``phase_<p>`` key,
    so phase A's canonical bytes are unchanged by this multi-phase plumbing.
    """

    id: str
    baseline: str
    exit_state: str
    receipt: str


@dataclass(frozen=True)
class TestTarget:
    id: str
    argv: tuple[str, ...]
    expected_tests: tuple[str, ...] = ()
    list_only: bool = False


@dataclass(frozen=True)
class PhaseSpec:
    """Everything one slice registers: rows, baseline file, and targets.

    ``deferred`` maps a row id to the documented reason it is recorded short
    of an exit state. Only a row named here may sit at a start state without
    failing its phase.
    """

    phase: str
    corpus: tuple[CorpusRow, ...]
    baseline_path: Path
    targets: Callable[[], tuple[TestTarget, ...]]
    deferred: Mapping[str, str] = field(default_factory=dict)


def _row(id: str, baseline: str, exit_state: str, receipt: str) -> CorpusRow:
    return CorpusRow(id, baseline, exit_state, receipt)


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
            "reshape.negative.runtime_eval_c",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "runtime_negative.issue_616_runtime_reshape_negative_extent_errs_in_both_lanes",
        ),
        _row(
            "vmap.element_derived_extent",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli.vmap_rejects_element_derived_extent_at_public_checker",
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
            "vmap.shared_shape_bound.concrete_c_emit",
            "silent_unguarded",
            "silent_unguarded",
            "cli.vmap_shape_bound_with_concrete_batch_emits_c_without_to_end_ice",
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


def generated_phase_b_corpus() -> tuple[CorpusRow, ...]:
    """Generate Slice B rows from C4's source, class, and guard properties.

    Every row is declared at its recorded `main` baseline.

    **One row per lane, where a guard lands per lane.** A guard is placed on
    each lane separately, and Slice B ships as three pull requests whose lanes
    move at different times: B2a places the C and HIP guards, B2h routes the
    host interpreter so the eval lane reaches the DAG evaluator at all, and
    B2b widens acceptance. A single-valued row cannot express one lane at its
    exit state while another still waits on B2h, so a row whose receipt
    EXECUTES a program is split into `.c` and `.eval` (and `.hip` where a HIP
    guard is emitted). Rows whose receipt is a chelis-ir unit test, or a
    checker verdict that no lane varies, stay single.

    Rows-per-lane rather than lane-valued exit states keeps the oracle's row
    machinery untouched: one id, one state, one receipt, one lattice
    transition. The phase-b digest is not frozen - only `FROZEN_PHASE_A_DIGEST`
    exists - so the shape change costs nothing the oracle protects, and it is
    made deliberately rather than as a side effect.

    A receipt names the test that records the row. Receipts on rows still at a
    start state are NOT enforced: `validate_receipt_coverage` runs over
    `rows_at_exit` only, because those are the rows whose tests already exist.
    The names Slice B's first half wrote for the unmoved rows were therefore
    aspirational, and this split reconciles them with the tests that exist
    rather than renaming tests to match. A row's receipt must name a test
    registered in `phase_b_targets` before that row may reach an exit state.
    """

    rows = (
        _row(
            "class.load_load.c",
            "silent_unguarded",
            EXECUTES,
            "exec_c.an_all_interface_class_traps_at_entry_when_its_witnesses_disagree",
        ),
        _row(
            "class.load_load.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.load_load_named_class_guards_every_non_canonical_member_on_eval",
        ),
        _row(
            "class.load_op_output.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.load_and_op_output_members_share_one_guarded_class_on_c",
        ),
        _row(
            "class.load_op_output.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.load_and_op_output_members_share_one_guarded_class_on_eval",
        ),
        _row(
            "class.no_movement_consumer.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_class_with_no_movement_bound_consumer_still_guards_on_c",
        ),
        _row(
            "class.no_movement_consumer.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_class_with_no_movement_bound_consumer_still_guards_on_eval",
        ),
        _row(
            "class.op_output_op_output.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.two_op_output_members_guard_against_the_canonical_member_on_c",
        ),
        _row(
            "class.op_output_op_output.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.two_op_output_members_guard_against_the_canonical_member_on_eval",
        ),
        _row(
            "class.shared_member_node.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.two_classes_sharing_one_node_keep_separate_guards_on_c",
        ),
        _row(
            "class.shared_member_node.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.two_classes_sharing_one_node_keep_separate_guards_on_eval",
        ),
        _row(
            "class.splice_f_of_n_n.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.splicing_f_of_n_n_yields_one_member_per_output_axis_on_c",
        ),
        _row(
            "class.splice_f_of_n_n.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.splicing_f_of_n_n_yields_one_member_per_output_axis_on_eval",
        ),
        _row(
            "expand.arith_size.named_claim.c",
            "lane_divergent",
            "lane_divergent",
            "cli_slice_b.checked_arithmetic_expand_size_under_a_named_claim_agrees_on_every_lane_on_c",
        ),
        _row(
            "expand.arith_size.named_claim.eval",
            "lane_divergent",
            "lane_divergent",
            "cli_slice_b.checked_arithmetic_expand_size_under_a_named_claim_agrees_on_every_lane_on_eval",
        ),
        _row(
            "expand.foreign_claim.same_tensor_set_axis.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_same_tensor_read_under_a_foreign_claim_is_guarded_on_c",
        ),
        _row(
            "expand.foreign_claim.same_tensor_set_axis.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_same_tensor_read_under_a_foreign_claim_is_guarded_on_eval",
        ),
        _row(
            "expand.kept_axis.op_declared_source.c",
            "ice",
            "ice",
            "cli_slice_b.issue_665_expand_over_stride_builds_and_runs",
        ),
        _row(
            "expand.kept_axis.op_declared_source.eval",
            "ice",
            "ice",
            "cli_slice_b.an_op_declared_axis_on_an_expand_input_flows_through_the_kept_output_axis_on_eval",
        ),
        _row(
            "expand.literal_claim.cross_tensor_read.c",
            "lane_divergent",
            "lane_divergent",
            "cli_slice_b.issue_1374_cross_tensor_read_traps_on_c",
        ),
        _row(
            "expand.literal_claim.cross_tensor_read.eval",
            "lane_divergent",
            "lane_divergent",
            "cli_slice_b.a_literal_claim_over_a_cross_tensor_read_guards_on_every_lane_on_eval",
        ),
        _row(
            "expand.literal_claim.inlined_root.c",
            "lane_divergent",
            "lane_divergent",
            "cli_slice_b.issue_1377_literal_claim_traps_at_the_inlined_root_on_c",
        ),
        _row(
            "expand.literal_claim.exported_kernel.c",
            "nonconforming_rejection",
            EXECUTES,
            "exec_c.a_literal_claim_over_a_runtime_read_traps_at_entry_on_c",
        ),
        _row(
            "expand.literal_claim.inlined_root.eval",
            "lane_divergent",
            "lane_divergent",
            "cli_slice_b.a_literal_claim_survives_root_inlining_with_its_guard_on_eval",
        ),
        _row(
            "expand.named_claim.cross_tensor_read.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.issue_1376_same_tensor_read_under_a_foreign_claim_traps_on_c",
        ),
        _row(
            "expand.named_claim.cross_tensor_read.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_cross_tensor_read_under_a_named_claim_is_guarded_on_eval",
        ),
        _row(
            "expand.op_declared_source.hip_prologue",
            "ice",
            EXECUTES,
            "cli_slice_b.an_op_declared_witness_reaches_the_hip_prologue_without_panicking",
        ),
        _row(
            "expand.piped_shape_read.lint_fix",
            "nonconforming_rejection",
            "nonconforming_rejection",
            "cli_slice_b.the_canonical_piped_shape_read_checks_evaluates_and_builds",
        ),
        _row(
            "expand.positional.replacement.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.issue_597_positional_same_rank_replacement_executes_on_c",
        ),
        _row(
            "expand.positional.replacement.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_positional_expand_replaces_a_unit_axis_instead_of_inserting_on_eval",
        ),
        _row(
            "expand.positional.replacement.non_unit_source_static",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_static_non_unit_source_under_a_same_rank_claim_is_a_type_error",
        ),
        _row(
            "expand.positional.replacement.non_unit_source_traps.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_c",
        ),
        _row(
            "expand.positional.replacement.non_unit_source_traps.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_eval",
        ),
        _row(
            "expand.positional.replacement_zero.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_zero_positional_replacement_declares_an_empty_axis_on_c",
        ),
        _row(
            "expand.positional.replacement_zero.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_zero_positional_replacement_declares_an_empty_axis_on_eval",
        ),
        _row(
            "expand.record_projection.size",
            "nonconforming_rejection",
            "nonconforming_rejection",
            "cli_slice_b.a_record_projection_is_an_admissible_expand_size",
        ),
        _row(
            "expand.shape_derived.declared_result_survives.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_shape_derived_bound_keeps_its_declared_result_dimension_on_c",
        ),
        _row(
            "expand.shape_derived.declared_result_survives.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_shape_derived_bound_keeps_its_declared_result_dimension_on_eval",
        ),
        _row(
            "guard_order.effect_after.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.an_effect_after_the_guard_does_not_run_when_the_guard_traps_on_eval",
        ),
        _row(
            "guard_order.effect_before.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.an_effect_before_the_guard_runs_when_the_guard_traps_on_eval",
        ),
        _row(
            "guard_order.trap_after.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.c_independent_trap_after_a_mismatch_loses",
        ),
        _row(
            "guard_order.trap_after.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_later_trap_is_preempted_by_the_extent_guard_on_eval",
        ),
        _row(
            "guard_order.trap_before.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.c_independent_trap_before_a_mismatch_wins",
        ),
        _row(
            "guard_order.trap_before.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.an_earlier_trap_preempts_the_extent_guard_on_eval",
        ),
        _row(
            "ir.axis_source.cardinality",
            "silent_unguarded",
            "executes_exactly",
            "ir_sources.omitted_or_duplicated_output_axis_source_fails_before_emission",
        ),
        _row(
            "rebuild.classes_after_each_pass",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.every_rebuild_pass_preserves_the_derived_classes",
        ),
        _row(
            "reshape.named_claim.node_target.c",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_node_valued_reshape_target_under_a_named_claim_is_guarded_on_c",
        ),
        _row(
            "reshape.named_claim.node_target.eval",
            "silent_unguarded",
            "silent_unguarded",
            "cli_slice_b.a_node_valued_reshape_target_under_a_named_claim_is_guarded_on_eval",
        ),
        # Chelis#1313 removes the synthesized zero only from ReLU. Sigmoid
        # retains the same sourceless-Const class and therefore keeps #1482's
        # typed receipt live rather than falsely closing the broader issue.
        _row(
            "shrink.elementwise_const.build",
            "ice",
            "typed_unsupported(#1482)",
            "cli_slice_b.runtime_bound_shrink_consumed_elementwise_reports_a_typed_receipt",
        ),
        _row(
            "shrink.to_end.nonzero_start",
            "silent_unguarded",
            "rejects_exactly",
            "ir_sources.to_end_shrink_end_requires_a_literal_zero_start",
        ),
    )
    return tuple(sorted(rows, key=lambda row: row.id))


def canonical_corpus_bytes(rows: Sequence[CorpusRow], phase: str) -> bytes:
    payload = [
        {
            "baseline": row.baseline,
            "id": row.id,
            f"phase_{phase}": row.exit_state,
            "receipt": row.receipt,
        }
        for row in rows
    ]
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def corpus_digest(rows: Sequence[CorpusRow], phase: str) -> str:
    return hashlib.sha256(canonical_corpus_bytes(rows, phase)).hexdigest()


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


def self_test_target(python: str = sys.executable) -> TestTarget:
    """The oracle's own unit suite. Shared by every phase, so run once."""

    return TestTarget("self_tests", (python, "scripts/test_runtime_extent_oracle.py"))


def phase_a_targets(python: str = sys.executable) -> tuple[TestTarget, ...]:
    return (
        self_test_target(python),
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
            "host_actualization",
            (
                "cargo",
                "test",
                "-p",
                "chelis-ir",
                "--lib",
                "host::tests::tensor_helper_actualization_declines_input_axis_for_shrink_and_stride",
                "--",
                "--exact",
                "--nocapture",
            ),
            (
                "host::tests::tensor_helper_actualization_declines_input_axis_for_shrink_and_stride",
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
                "vmap_accepts_shape_and_shared_scalar_extent_sources",
                "vmap_rejects_element_derived_extent_at_public_checker",
                "vmap_rejects_helper_result_derived_from_tensor_elements",
                "vmap_shape_bound_with_concrete_batch_emits_c_without_to_end_ice",
                "zero_extent_is_check_clean_and_evaluates_to_empty_tensor",
            ),
        ),
        TestTarget(
            "runtime_negative",
            (
                "cargo", "test", "-p", "chelis-cli", "--test",
                "issue_616_runtime_reshape_c_parity",
                "issue_616_runtime_reshape_negative_extent_errs_in_both_lanes",
                "--", "--exact", "--nocapture",
            ),
            ("issue_616_runtime_reshape_negative_extent_errs_in_both_lanes",),
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
                "v7_shrink_rejects_a_to_end_end_over_a_non_zero_start",
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


def phase_b_targets(python: str = sys.executable) -> tuple[TestTarget, ...]:
    return (
        self_test_target(python),
        TestTarget(
            "ir_sources",
            (
                "cargo", "test", "-p", "chelis-ir", "--test",
                "runtime_extent_slice_b_sources", "--", "--nocapture",
            ),
            (
                "every_risc_op_yields_exactly_one_source_per_output_axis",
                "binding_a_to_end_bound_rejects_a_start_that_is_not_literal_zero",
                "expand_insert_maps_later_output_axes_to_input_minus_one",
                "external_axis_names_the_exact_load_not_a_string_match",
                "full_axis_symbolic_shrink_is_op_computed_not_pass_through",
                "identity_stride_one_and_zero_pad_pass_the_input_axis_through",
                "input_axis_and_scalar_input_sources_validate_their_slots",
                "omitted_or_duplicated_output_axis_source_fails_before_emission",
                "op_declared_axis_on_an_expand_input_flows_through_the_kept_output_axis",
                "reduction_and_count_shift_kept_output_axes_back_to_their_input_axis",
                "to_end_shrink_end_requires_a_literal_zero_start",
                "unsupported_but_well_typed_mapping_yields_the_registered_receipt_not_an_ice",
            ),
        ),
        TestTarget(
            "cli_slice_b",
            (
                "cargo", "test", "-p", "chelis-cli", "--test",
                "runtime_extent_slice_b", "--", "--nocapture",
            ),
            (
                "an_op_declared_witness_reaches_the_hip_prologue_without_panicking",
                "c_independent_trap_after_a_mismatch_loses",
                "c_independent_trap_before_a_mismatch_wins",
                "runtime_bound_shrink_consumed_elementwise_reports_a_typed_receipt",
                "runtime_bound_shrink_relu_builds_and_matches_eval_exactly",
            ),
        ),
        # A CLI-rooted program is not the exported kernel: `def main() =
        # f(...)` over literal tensors inlines the def, every extent becomes a
        # literal, and the classes disappear, so guard placement and rendering
        # on C are proved by DRIVEN rows that compile the exported kernel and
        # call it with runtime inputs. This target is where those rows live.
        # `exec_compile` is a large shared binary, so this target names its
        # four rows with `--exact` rather than running the whole file: the
        # receipt check requires the observed set to EQUAL the expected one,
        # and an unfiltered run would tie this phase's oracle to every
        # unrelated numeric row in that file.
        TestTarget(
            "exec_c",
            (
                "cargo", "test", "-p", "chelis-backend-c", "--test",
                "exec_compile", "--", "--nocapture", "--exact",
                "a_literal_claim_over_a_runtime_read_traps_at_entry_on_c",
                "an_all_interface_class_runs_when_its_witnesses_agree",
                "an_all_interface_class_traps_at_entry_when_its_witnesses_disagree",
                "entry_guards_run_in_assigned_slot_order_not_claim_name_order",
            ),
            (
                "a_literal_claim_over_a_runtime_read_traps_at_entry_on_c",
                "an_all_interface_class_runs_when_its_witnesses_agree",
                "an_all_interface_class_traps_at_entry_when_its_witnesses_disagree",
                "entry_guards_run_in_assigned_slot_order_not_claim_name_order",
            ),
        ),
    )


PHASE_A_DEFERRED: Mapping[str, str] = {
    "expand.input_axis.metal_device": (
        "runtime_extents.md C2.5: no Metal device-path expand row executes until "
        "chelis#1383 lands expand emission and symbolic-dim Load support"
    ),
    "vmap.shared_shape_bound.concrete_c_emit": (
        "runtime_extents.md C2.7: the guard this row waits on is Slice B's, so "
        "Slice A records it at its main baseline"
    ),
}


PHASE_REGISTRY: Mapping[str, PhaseSpec] = {
    "a": PhaseSpec(
        phase="a",
        corpus=generated_phase_a_corpus(),
        baseline_path=BASELINE_PATH,
        targets=phase_a_targets,
        deferred=PHASE_A_DEFERRED,
    ),
    "b": PhaseSpec(
        phase="b",
        corpus=generated_phase_b_corpus(),
        baseline_path=BASELINE_PATH_PHASE_B,
        targets=phase_b_targets,
        deferred={},
    ),
}


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


def registered_specs(
    registry: Mapping[str, PhaseSpec] | None = None,
) -> tuple[PhaseSpec, ...]:
    """Every registered slice phase, in lattice order."""

    active = PHASE_REGISTRY if registry is None else registry
    return tuple(active[phase] for phase in SLICE_PHASES if phase in active)


def validate_phase_chain(specs: Sequence[PhaseSpec]) -> None:
    """Enforce the one-way lattice across the whole recorded chain.

    A row declared by more than one phase records one `main` baseline and
    moves only rightward through the phases in order, so a leftward move
    anywhere in the chain fails every invocation, not only the owning
    phase's.
    """

    baselines: dict[str, str] = {}
    chain: dict[str, list[str]] = {}
    for spec in specs:
        for row in spec.corpus:
            recorded = baselines.setdefault(row.id, row.baseline)
            if recorded != row.baseline:
                raise OracleFailure(
                    f"row {row.id!r} records two different baselines: "
                    f"{recorded!r} and {row.baseline!r}"
                )
            chain.setdefault(row.id, []).append(row.exit_state)
    for row_id in sorted(chain):
        previous = baselines[row_id]
        for state in chain[row_id]:
            try:
                validate_transition(previous, state)
            except OracleFailure as error:
                raise OracleFailure(f"row {row_id!r}: {error}") from error
            previous = state


def rows_at_exit(spec: PhaseSpec) -> tuple[CorpusRow, ...]:
    """Rows whose recorded state is an exit state or a documented deferral.

    These rows' receipts name tests that already exist, so these are the rows
    receipt coverage is enforced over. A row still at a start state is listed
    by `exit_shortfall` instead, which is what fails the phase.
    """

    return tuple(
        row
        for row in spec.corpus
        if _status_class(row.exit_state) in EXIT_CLASSES or row.id in spec.deferred
    )


def exit_shortfall(spec: PhaseSpec) -> tuple[str, ...]:
    """Row ids the phase owns that are neither at an exit state nor deferred."""

    return tuple(
        row.id
        for row in spec.corpus
        if _status_class(row.exit_state) not in EXIT_CLASSES
        and row.id not in spec.deferred
    )


def load_and_validate_baseline(
    spec: PhaseSpec, path: Path | None = None
) -> Mapping[str, object]:
    generated = tuple(spec.corpus)
    baseline_path = path if path is not None else spec.baseline_path
    exit_key = f"phase_{spec.phase}"
    try:
        payload = json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OracleFailure(f"cannot read baseline {baseline_path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "corpus_sha256",
        "rows",
    }:
        raise OracleFailure("baseline must contain exactly schema_version, corpus_sha256, rows")
    if payload["schema_version"] != 1:
        raise OracleFailure(f"unsupported baseline schema {payload['schema_version']!r}")
    expected_digest = corpus_digest(generated, spec.phase)
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
            exit_key,
            "receipt",
        }:
            raise OracleFailure(f"baseline row {index} has the wrong fields")
        row = CorpusRow(
            id=entry["id"],
            baseline=entry["baseline"],
            exit_state=entry[exit_key],
            receipt=entry["receipt"],
        )
        _status_class(row.baseline)
        _status_class(row.exit_state)
        validate_transition(row.baseline, row.exit_state)
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


def dedupe_targets(targets: Sequence[TestTarget]) -> tuple[TestTarget, ...]:
    """One run per distinct command, so phases sharing a target run it once."""

    seen: set[tuple[str, ...]] = set()
    unique: list[TestTarget] = []
    for target in targets:
        if target.argv in seen:
            continue
        seen.add(target.argv)
        unique.append(target)
    return tuple(unique)


def selected_specs(
    phase: str, registry: Mapping[str, PhaseSpec] | None = None
) -> tuple[PhaseSpec, ...]:
    active = PHASE_REGISTRY if registry is None else registry
    if phase == "final":
        missing = [p for p in SLICE_PHASES if p not in active]
        if missing:
            raise OracleFailure(
                "phase 'final' requires every slice phase to be registered; "
                f"missing {missing}"
            )
        return registered_specs(active)
    spec = active.get(phase)
    if spec is None:
        registered = ", ".join(sorted(active)) or "none"
        raise OracleFailure(
            f"phase {phase!r} corpus is not implemented; "
            f"registered phases are {registered}"
        )
    return (spec,)


def validate(
    phase: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    registry: Mapping[str, PhaseSpec] | None = None,
    targets: Sequence[TestTarget] | None = None,
    require_clean: bool = True,
) -> tuple[str, str]:
    if phase not in PHASES:
        raise OracleFailure(f"unsupported phase {phase!r}")

    # The lattice binds the whole recorded chain on every invocation, so a
    # leftward move in a later phase fails an earlier phase's run too.
    validate_phase_chain(registered_specs(registry))

    selected = selected_specs(phase, registry)
    for spec in selected:
        load_and_validate_baseline(spec)

    if targets is None:
        selected_targets = dedupe_targets(
            tuple(target for spec in selected for target in spec.targets())
        )
        for spec in selected:
            validate_receipt_coverage(rows_at_exit(spec), spec.targets())
    else:
        selected_targets = dedupe_targets(tuple(targets))
        for spec in selected:
            validate_receipt_coverage(rows_at_exit(spec), selected_targets)

    head = exact_head_receipt(runner, require_clean=require_clean)
    for target in selected_targets:
        completed = _run_text(runner, target.argv)
        validate_target_receipt(target, completed)

    shortfall = [
        (spec.phase, row_id) for spec in selected for row_id in exit_shortfall(spec)
    ]
    if shortfall:
        listed = ", ".join(f"{p}:{row}" for p, row in shortfall)
        raise OracleFailure(
            f"phase {phase!r} has {len(shortfall)} row(s) short of an exit state: {listed}"
        )

    digests = [corpus_digest(spec.corpus, spec.phase) for spec in selected]
    digest = (
        digests[0]
        if len(digests) == 1
        else hashlib.sha256("".join(digests).encode()).hexdigest()
    )
    print(f"runtime_extent_head={head}", flush=True)
    for spec, phase_digest in zip(selected, digests):
        print(
            f"runtime_extent_corpus_sha256_phase_{spec.phase}={phase_digest}", flush=True
        )
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
