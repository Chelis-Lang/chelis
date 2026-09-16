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
TARGETS_PATH = REPO_ROOT / "scripts/runtime_extent_oracle_targets.json"
# `c` remains a nameable phase and is deliberately absent from SLICE_PHASES.
# Slice C was withdrawn rather than deferred: `expand` broadcasts a singleton
# axis and `insert` raises rank, so neither primitive produces a deferred
# shape and no phase-C corpus will ever be written. `final` selects
# SLICE_PHASES, so while `c` sat there the completion oracle refused on a
# corpus nobody owed. Naming the retired phase still reaches this oracle's
# own refusal, which says the phase has no registered corpus and names the
# ones that do; an argparse choice error would say nothing about why.
PHASES = ("a", "b", "c", "final")
SLICE_PHASES = ("a", "b")
PASS_MARKER = "RUNTIME EXTENT ORACLE: PASS"
SHORT_MARKER = "RUNTIME EXTENT ORACLE: RECEIPTS PASS, ROWS SHORT OF EXIT"
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
            "cli.shape_sourced_insert_rejects_wrong_rank_ascription",
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
            EXECUTES,
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
            EXECUTES,
            "cli_slice_b.load_load_named_class_guards_every_non_canonical_member_on_eval",
        ),
        _row(
            "class.load_op_output.c",
            "silent_unguarded",
            EXECUTES,
            "exec_c.a_local_class_guards_at_its_operation_and_renders_the_numeric_trap",
        ),
        _row(
            "class.local_members.all_guarded.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.every_local_member_of_one_class_is_guarded_at_its_operation_on_c",
        ),
        _row(
            "class.load_op_output.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.load_and_op_output_members_share_one_guarded_class_on_eval",
        ),
        _row(
            "class.no_movement_consumer.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_class_with_no_movement_bound_consumer_still_guards_on_c",
        ),
        _row(
            "class.no_movement_consumer.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_class_with_no_movement_bound_consumer_still_guards_on_eval",
        ),
        _row(
            "class.op_output_op_output.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.two_op_output_members_guard_against_the_canonical_member_on_c",
        ),
        _row(
            "class.op_output_op_output.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.two_op_output_members_guard_against_the_canonical_member_on_eval",
        ),
        _row(
            "class.shared_member_node.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.two_classes_sharing_one_node_keep_separate_guards_on_c",
        ),
        _row(
            "class.shared_member_node.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.two_classes_sharing_one_node_keep_separate_guards_on_eval",
        ),
        _row(
            "class.splice_f_of_n_n.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.splicing_f_of_n_n_yields_one_member_per_output_axis_on_c",
        ),
        _row(
            "class.splice_f_of_n_n.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.splicing_f_of_n_n_yields_one_member_per_output_axis_on_eval",
        ),
        _row(
            "expand.arith_size.named_claim.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.checked_arithmetic_expand_size_under_a_named_claim_agrees_on_every_lane_on_c",
        ),
        _row(
            "expand.arith_size.named_claim.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.checked_arithmetic_expand_size_under_a_named_claim_agrees_on_every_lane_on_eval",
        ),
        _row(
            "expand.foreign_claim.same_tensor_set_axis.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.issue_1376_same_tensor_read_under_a_foreign_claim_is_guarded_on_c",
        ),
        _row(
            "expand.foreign_claim.same_tensor_set_axis.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_same_tensor_read_under_a_foreign_claim_is_guarded_on_eval",
        ),
        _row(
            "expand.kept_axis.op_declared_source.c",
            "ice",
            EXECUTES,
            "cli_slice_b.issue_665_expand_over_stride_builds_and_runs",
        ),
        # An explicit BASELINE CORRECTION, not a relabelled improvement. This
        # row recorded `main` as `ice` and that was wrong: measured on
        # `3dc3f54f6`, `chelis eval --file` of chelis#665's program prints
        # `shape=[6, 3]` and exits zero. The only eval-lane reader of the
        # legacy walk was the binding inference, which does not abort, and no
        # command-line program reaches an eval-lane version of the failure
        # because `chelis eval --file` binds no inputs and every extent is
        # concrete by then. The receipt is a disposition lock; the `.c` row
        # above carries the byte-for-byte parity assertion that gives the pair
        # its teeth.
        _row(
            "expand.kept_axis.op_declared_source.eval",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.an_op_declared_axis_on_an_expand_input_flows_through_the_kept_output_axis_on_eval",
        ),
        _row(
            "expand.literal_claim.cross_tensor_read.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_literal_claim_over_a_cross_tensor_read_traps_on_c",
        ),
        _row(
            "expand.literal_claim.cross_tensor_read.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_literal_claim_over_a_cross_tensor_read_traps_on_eval",
        ),
        _row(
            "expand.literal_claim.inlined_root.c",
            "lane_divergent",
            EXECUTES,
            "literal_claim.literal_result_claim_contract",
        ),
        # B2h: the eval twin of the driven row. The value-binding form applies
        # `f` through the kernel the C lane emits for it, and the literal input
        # extent is checked at the kernel's entry by the DAG evaluator, the
        # eval analogue of the C ABI preamble.
        _row(
            "expand.literal_claim.exported_kernel.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_literal_claim_over_a_runtime_read_traps_at_entry_on_eval",
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
            EXECUTES,
            "literal_claim.literal_result_claim_contract",
        ),
        _row(
            "expand.named_claim.cross_tensor_read.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.issue_1374_cross_tensor_read_under_a_named_claim_is_guarded_on_c",
        ),
        _row(
            "expand.named_claim.cross_tensor_read.eval",
            "silent_unguarded",
            EXECUTES,
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
            EXECUTES,
            "cli_slice_b.the_canonical_piped_shape_read_checks_evaluates_and_builds",
        ),
        _row(
            "expand.positional.replacement.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.issue_597_positional_same_rank_replacement_executes_on_c",
        ),
        _row(
            "expand.positional.replacement.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_positional_expand_replaces_a_unit_axis_instead_of_inserting_on_eval",
        ),
        _row(
            "expand.positional.replacement.non_unit_source_static",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_static_non_unit_source_under_a_same_rank_claim_is_a_type_error",
        ),
        _row(
            "expand.positional.replacement.non_unit_source_traps.c",
            "silent_unguarded",
            EXECUTES,
            "exec_c.a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_c",
        ),
        _row(
            "expand.positional.replacement.non_unit_source_traps.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_eval",
        ),
        _row(
            "expand.positional.replacement.shape_size.eval_c",
            "lane_divergent",
            EXECUTES,
            "cli_broadcast.singleton_broadcast_contract",
        ),
        _row(
            "expand.positional.replacement_zero.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_zero_positional_replacement_declares_an_empty_axis_on_c",
        ),
        _row(
            "expand.positional.replacement_zero.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_zero_positional_replacement_declares_an_empty_axis_on_eval",
        ),
        _row(
            "expand.record_projection.size",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_record_projection_is_an_admissible_expand_size",
        ),
        _row(
            "expand.shape_derived.declared_result_survives.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_shape_derived_bound_keeps_its_declared_result_dimension_on_c",
        ),
        _row(
            "expand.shape_derived.declared_result_survives.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_shape_derived_bound_keeps_its_declared_result_dimension_on_eval",
        ),
        # B2h: the eval effect rows stay at baseline. `chelis eval` emits a
        # program's printed output only when the evaluation succeeds, so the
        # order of an effect against a trap is unobservable on that lane
        # today (chelis#1585); the receipts lock the trap alone.
        # B2h: the C effect rows return with chelis#1528. On main the effect was
        # absent from the emitted program (the kernel decision dropped the
        # def's `IO` effect), so nothing observable could be ordered against a
        # guard; the shared decision keeps such a body in host code on both
        # lanes and the print is emitted again. Their receipts assert the
        # order from the emitted C's statement order (the print against the
        # call into the kernel, the guard against the first allocation) and
        # the trap by execution: the printed bytes do not survive the trap's
        # abort on a buffered stdout (chelis#1591), so they cannot carry the
        # ordering assertion on every platform.
        _row(
            "guard_order.effect_after.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_effect_after_the_guard_does_not_run_when_the_guard_traps_on_c",
        ),
        _row(
            "guard_order.effect_before.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_effect_before_the_guard_runs_when_the_guard_traps_on_c",
        ),
        _row(
            "guard_order.effect_after.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_effect_after_the_guard_does_not_run_when_the_guard_traps_on_eval",
        ),
        _row(
            "guard_order.effect_before.eval",
            "silent_unguarded",
            EXECUTES,
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
            EXECUTES,
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
            EXECUTES,
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
            EXECUTES,
            "ir_classes.every_rebuild_pass_preserves_the_derived_classes",
        ),
        _row(
            "reshape.named_claim.node_target.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_node_valued_reshape_target_under_a_named_claim_is_guarded_on_c",
        ),
        _row(
            "reshape.named_claim.node_target.eval",
            "silent_unguarded",
            EXECUTES,
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
            "guard.local.numeric_carriers.eval_c",
            "silent_unguarded",
            EXECUTES,
            "exec_c.numeric_local_extent_claims_execute_exactly",
        ),
        _row(
            "claim.literal.same_shape_producer.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_pass_through_literal_claim_is_guarded_by_its_returned_same_shape_producer",
        ),
        _row(
            "claim.literal.same_shape_producer.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_pass_through_literal_claim_is_guarded_by_its_returned_same_shape_producer",
        ),
        _row(
            "claim.named.nested_fresh_result.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_nested_named_result_claim_is_enforced_through_its_resolved_binder",
        ),
        _row(
            "claim.named.nested_fresh_result.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_nested_named_result_claim_is_enforced_through_its_resolved_binder",
        ),
        _row(
            "claim.named.same_shape_producer.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_pass_through_named_claim_is_guarded_by_its_returned_same_shape_producer",
        ),
        _row(
            "claim.named.same_shape_producer.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_pass_through_named_claim_is_guarded_by_its_returned_same_shape_producer",
        ),
        _row(
            "claim.named.same_shape_producer.inlined_root.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_inlined_root_same_shape_claim_is_guarded_by_its_returned_producer",
        ),
        _row(
            "claim.named.same_shape_producer.inlined_root.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_inlined_root_same_shape_claim_is_guarded_by_its_returned_producer",
        ),
        # chelis#1948: the returned same-shape primitive owns its declared
        # result guard. Every runtime row is split by lane even though one
        # dedicated test executes both; the relation/verification controls
        # remain independent IR receipts.
        _row(
            "claim.same_shape.agreeing.c",
            EXECUTES,
            EXECUTES,
            "cli_issue_1948_same_shape.an_agreeing_same_shape_result_claim_executes_exactly",
        ),
        _row(
            "claim.same_shape.agreeing.eval",
            EXECUTES,
            EXECUTES,
            "cli_issue_1948_same_shape.an_agreeing_same_shape_result_claim_executes_exactly",
        ),
        _row(
            "claim.same_shape.fusion.named.agreeing.eval",
            EXECUTES,
            EXECUTES,
            "ir_issue_1948_same_shape.agreeing_named_result_claim_survives_fusion_and_dce",
        ),
        _row(
            "claim.same_shape.fusion.named.interior.eval",
            "silent_unguarded",
            EXECUTES,
            "ir_issue_1948_same_shape.an_interior_named_result_claim_splits_the_fusion_chain",
        ),
        _row(
            "claim.same_shape.fusion.named.trap.eval",
            "silent_unguarded",
            EXECUTES,
            "ir_issue_1948_same_shape.named_result_claim_traps_after_fusion_and_dce",
        ),
        _row(
            "claim.same_shape.distinct_paths.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_issue_1948_same_shape.distinct_candidate_paths_do_not_create_source_order_attribution",
        ),
        _row(
            "claim.same_shape.distinct_paths.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_issue_1948_same_shape.distinct_candidate_paths_do_not_create_source_order_attribution",
        ),
        _row(
            "claim.same_shape.no_origin.c",
            "silent_unguarded",
            EXECUTES,
            "cli_issue_1948_same_shape.no_claim_capable_operand_origin_does_not_drop_the_result_claim",
        ),
        _row(
            "claim.same_shape.no_origin.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_issue_1948_same_shape.no_claim_capable_operand_origin_does_not_drop_the_result_claim",
        ),
        _row(
            "claim.same_shape.operand_disagreement.c",
            EXECUTES,
            EXECUTES,
            "cli_issue_1948_same_shape.runtime_operand_disagreement_precedes_the_result_claim",
        ),
        _row(
            "claim.same_shape.operand_disagreement.eval",
            EXECUTES,
            EXECUTES,
            "cli_issue_1948_same_shape.runtime_operand_disagreement_precedes_the_result_claim",
        ),
        _row(
            "claim.same_shape.operand_one.c",
            "silent_unguarded",
            EXECUTES,
            "cli_issue_1948_same_shape.the_operand_one_witness_is_guarded_by_add",
        ),
        _row(
            "claim.same_shape.operand_one.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_issue_1948_same_shape.the_operand_one_witness_is_guarded_by_add",
        ),
        _row(
            "claim.same_shape.rank_zero.eval",
            EXECUTES,
            EXECUTES,
            "ir_issue_1948_same_shape.rank_zero_inputs_are_excluded_from_same_shape_result_claim_observation",
        ),
        _row(
            "claim.same_shape.repeated_path.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_issue_1948_same_shape.repeated_paths_to_one_candidate_produce_one_add_claim",
        ),
        _row(
            "claim.same_shape.repeated_path.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_issue_1948_same_shape.repeated_paths_to_one_candidate_produce_one_add_claim",
        ),
        _row(
            "claim.same_shape.reversed.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_issue_1948_same_shape.reversing_operands_does_not_rename_adds_result_claim",
        ),
        _row(
            "claim.same_shape.reversed.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_issue_1948_same_shape.reversing_operands_does_not_rename_adds_result_claim",
        ),
        _row(
            "claim.same_shape.specialization.blas.literal.trap.eval",
            "nonconforming_rejection",
            EXECUTES,
            "ir_issue_1948_same_shape.blas_interior_literal_claim_blocks_replacement_and_traps_after_dce",
        ),
        _row(
            "claim.same_shape.specialization.blas.named.trap.eval",
            "nonconforming_rejection",
            EXECUTES,
            "ir_issue_1948_same_shape.blas_interior_named_claim_blocks_replacement_and_traps_after_dce",
        ),
        _row(
            "claim.same_shape.specialization.blas.unclaimed.eval",
            EXECUTES,
            EXECUTES,
            "ir_issue_1948_same_shape.unclaimed_blas_region_still_specializes_and_executes",
        ),
        _row(
            "claim.same_shape.specialization.dense_gather.literal.trap.eval",
            "nonconforming_rejection",
            EXECUTES,
            "ir_issue_1948_same_shape.dense_gather_interior_literal_claim_blocks_replacement_and_traps_after_dce",
        ),
        _row(
            "claim.same_shape.specialization.dense_gather.named.trap.eval",
            "nonconforming_rejection",
            EXECUTES,
            "ir_issue_1948_same_shape.dense_gather_interior_named_claim_blocks_replacement_and_traps_after_dce",
        ),
        _row(
            "claim.same_shape.specialization.dense_gather.unclaimed.eval",
            EXECUTES,
            EXECUTES,
            "ir_issue_1948_same_shape.unclaimed_dense_gather_region_still_specializes_and_executes",
        ),
        _row(
            "claim.same_shape.specialization.named.agreeing.eval",
            EXECUTES,
            EXECUTES,
            "ir_issue_1948_same_shape.agreeing_named_result_claim_survives_specialization_and_dce",
        ),
        _row(
            "claim.same_shape.specialization.named.trap.eval",
            "silent_unguarded",
            EXECUTES,
            "ir_issue_1948_same_shape.named_result_claim_traps_after_specialization_and_dce",
        ),
        _row(
            "claim.same_shape.static_refutation",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "cli_issue_1948_same_shape.a_statically_refuted_same_shape_result_is_a_dimension_mismatch",
        ),
        _row(
            "claim.same_shape.structure.dedup",
            "silent_unguarded",
            EXECUTES,
            "ir_issue_1948_same_shape.identical_members_deduplicate_but_distinct_paths_remain",
        ),
        _row(
            "claim.same_shape.structure.malformed",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "ir_issue_1948_same_shape.malformed_same_shape_relations_are_verifier_errors",
        ),
        # Round 1's P1. A RUNTIME padding bound is a different witness from
        # a literal one, and the rows are separate because the claim's
        # precondition is the bound rather than the operand: lowering stamped
        # the claim and `host::rank_preserving_movement_type`'s Pad arm minted
        # over it, so the exported and value-binding forms were silent while
        # the inlined root trapped. Both lanes are `silent_unguarded` here,
        # unlike the literal-bound rows below: with the declared dim replaced
        # by a minted `_rt_pad_dim_N_A`, the C movement plan's target check
        # compares against that minted dim and passes, so C returned the
        # undeclared shape at exit zero too.
        _row(
            "pad.runtime_bound_claim.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_runtime_bound_pad_claim_is_guarded_in_every_activation_form",
        ),
        _row(
            "pad.runtime_bound_claim.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_runtime_bound_pad_claim_is_guarded_in_every_activation_form",
        ),
        _row(
            "pad.runtime_bound_claim.after_and_named.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_runtime_after_bound_and_a_named_pad_claim_reach_the_same_guard",
        ),
        _row(
            "pad.runtime_bound_claim.after_and_named.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_runtime_after_bound_and_a_named_pad_claim_reach_the_same_guard",
        ),
        _row(
            "pad.runtime_bound_claim.rank_two_axis.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_rank_two_pad_guards_and_reports_the_runtime_axis_it_widens",
        ),
        _row(
            "pad.runtime_bound_claim.rank_two_axis.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_rank_two_pad_guards_and_reports_the_runtime_axis_it_widens",
        ),
        # The C lane's baseline is `lane_divergent` rather than
        # `silent_unguarded`: it did not return a wrong shape, it aborted at
        # the movement plan's generic target check, reporting the allocation
        # instead of the claim and with no [04-NUM-9] context line, while eval
        # printed the padded shape at exit zero.
        _row(
            "pad.literal_claim.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_non_zero_pad_extent_is_guarded_on_both_lanes",
        ),
        _row(
            "pad.literal_claim.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_non_zero_pad_extent_is_guarded_on_both_lanes",
        ),
        # chelis#1907. The runtime-zero reproducer silently returned the
        # unstrided input on Eval while compiled C already raised the stride
        # operation's exact domain failure. The shared receipt also exercises
        # a runtime-negative step and proves both invalid values preempt a
        # disagreeing result claim before allocation or access.
        _row(
            "stride.runtime_zero.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.issue_1907_runtime_non_positive_stride_steps_trap_before_claims_on_both_lanes",
        ),
        _row(
            "stride.runtime_zero.c",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.issue_1907_runtime_non_positive_stride_steps_trap_before_claims_on_both_lanes",
        ),
        # Red-team round 1 on PR #2111 found the cross-axis half of #1907:
        # Eval compared axis 0's result claim before validating axis 1's
        # runtime step, while C validated the complete vector first. The
        # negative receipt crosses zero and negative axis-1 values with an
        # earlier axis-0 mismatch; the positive receipt keeps runtime unit
        # and non-unit multi-axis execution as non-vacuity controls.
        _row(
            "stride.multi_axis_nonpositive.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.issue_1907_multi_axis_non_positive_steps_preempt_earlier_claims_on_both_lanes",
        ),
        _row(
            "stride.multi_axis_nonpositive.c",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.issue_1907_multi_axis_non_positive_steps_preempt_earlier_claims_on_both_lanes",
        ),
        _row(
            "stride.multi_axis_positive.eval",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.issue_1907_multi_axis_positive_stride_vectors_execute_on_both_lanes",
        ),
        _row(
            "stride.multi_axis_positive.c",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.issue_1907_multi_axis_positive_stride_vectors_execute_on_both_lanes",
        ),
        # chelis#1931. A positive non-unit stride under a disagreeing declared
        # result returned the undeclared extent on Eval. C rejected only at
        # the movement target backstop, without [04-NUM-9]'s claim context.
        # The receipt covers literal and runtime carriers plus a nonzero axis.
        _row(
            "stride.nonunit_claim.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.issue_1931_positive_stride_extents_guard_declared_results_on_both_lanes",
        ),
        _row(
            "stride.nonunit_claim.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.issue_1931_positive_stride_extents_guard_declared_results_on_both_lanes",
        ),
        # chelis#1837 and chelis#1930: a declared result claim the LOWERED
        # graph fixes to another extent is rejected when the activation is
        # lowered, before any execution, with one fatal diagnostic both host
        # lanes render byte-identically. `spec/04-type-system.md` section 4.7
        # makes a violation proven from literals a type error, and section
        # 4.7.2 conditions its guard on a claim "that is not statically proven
        # equal to `size`", so these rows were never the guard's. The checker
        # reaches that verdict wherever it can SEE the extent; an extent that
        # becomes literal only because a call supplied concrete arguments is
        # one it cannot see, which is why the rejection sits at lowering.
        # `claim.literal.kernel_entry.checker` below is the partition's other
        # half.
        _row(
            "concat.literal_claim.inlined_root.c",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_declared_concat_axis_extent_an_inlined_root_refutes_is_rejected_on_both_lanes",
        ),
        _row(
            "concat.literal_claim.inlined_root.eval",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_declared_concat_axis_extent_an_inlined_root_refutes_is_rejected_on_both_lanes",
        ),
        # chelis#1930's two lanes had two different baselines for one defect:
        # eval returned the undeclared shape at exit zero, the C build failed
        # inside the IR verifier's per-owner size check.
        _row(
            "pad.identity_axis.literal_claim.inlined_root.c",
            "ice",
            TERMINAL_CONTROL,
            "cli_slice_b.a_zero_padded_identity_axis_a_declaration_refutes_is_rejected_on_both_lanes",
        ),
        _row(
            "pad.identity_axis.literal_claim.inlined_root.eval",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_zero_padded_identity_axis_a_declaration_refutes_is_rejected_on_both_lanes",
        ),
        # The smallest member of the class: the body is the parameter, so no
        # operation exists for a guard to attach to and the refuting extent is
        # `ExtentOrigin::Literal` rather than `OpComputed`.
        _row(
            "claim.literal.identity_root.c",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.an_identity_body_that_refutes_its_declared_extent_is_rejected_on_both_lanes",
        ),
        _row(
            "claim.literal.identity_root.eval",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.an_identity_body_that_refutes_its_declared_extent_is_rejected_on_both_lanes",
        ),
        # `pad.literal_claim` above in the VALUE-BINDING form, which stages its
        # argument across the host boundary and so keeps section 4.7.2's guard.
        # The inlined root hands the operation a literal instead, and chelis
        # #1911 recorded the resulting lane divergence as residual: eval
        # trapped for a comparison that could never hold while the C build
        # failed in the verifier.
        _row(
            "pad.literal_claim.inlined_root.c",
            "lane_divergent",
            TERMINAL_CONTROL,
            "cli_slice_b.a_literal_pad_claim_an_inlined_root_refutes_is_rejected_on_both_lanes",
        ),
        _row(
            "pad.literal_claim.inlined_root.eval",
            "lane_divergent",
            TERMINAL_CONTROL,
            "cli_slice_b.a_literal_pad_claim_an_inlined_root_refutes_is_rejected_on_both_lanes",
        ),
        # A NAMED claim is refutable once chelis#1800 has resolved its
        # declaring witness to a number. The unresolved form stays a guard and
        # is the control inside the same receipt.
        _row(
            "claim.named.resolved.inlined_root.c",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_resolved_named_claim_the_inlined_body_refutes_is_rejected_on_both_lanes",
        ),
        _row(
            "claim.named.resolved.inlined_root.eval",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_resolved_named_claim_the_inlined_body_refutes_is_rejected_on_both_lanes",
        ),
        # Only a named call hands the lowerer a callee name. A pipe stage, an
        # AD or vectorization boundary and a host-applied root do not, so the
        # diagnostic's owner slot reads `the signature` there; the receipt
        # asserts both spellings.
        _row(
            "claim.literal.nameless_activation.c",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_nameless_activation_that_refutes_its_own_claim_is_rejected_on_both_lanes",
        ),
        _row(
            "claim.literal.nameless_activation.eval",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_slice_b.a_nameless_activation_that_refutes_its_own_claim_is_rejected_on_both_lanes",
        ),
        # One row, not a pair, and no lane suffix: the verdict is the CHECKER's,
        # no lane varies it, and the program never reaches a lane. That is the
        # shape chelis#1836's `route.untied` rows use for the same reason.
        #
        # Its baseline EQUALS its exit state, which is the one honest reading
        # here and has six precedents in phase A (`expand.negative.static`,
        # `reshape.negative.runtime_eval_c`, `wire.legacy_v6.rejected` among
        # them): this program was already refused, correctly and with the right
        # diagnostic, on `0820ee28e`. It is C5's "Invalid-program controls
        # remain `rejects_exactly`" rather than a defect that moved, and
        # recording a start state it never occupied would claim a transition
        # that did not happen.
        #
        # This row is a CONTROL, not a repair. B2c changed nothing about the
        # program it names; the row exists because it is the PARTITION's other
        # half, and it fails only if a later change lets lowering reach a case
        # the checker owns. A literal parameter extent makes the body's own
        # result type computable, so the checker refuses the signature before
        # lowering runs at all, which is why the kernel-entry call sites cannot
        # reach the rejection above.
        _row(
            "claim.literal.kernel_entry",
            TERMINAL_CONTROL,
            TERMINAL_CONTROL,
            "cli_slice_b.a_literal_parameter_extent_keeps_the_checkers_verdict_on_both_lanes",
        ),
        _row(
            "claim.literal.nested_and_unused.eval_c",
            "silent_unguarded",
            EXECUTES,
            "literal_claim.literal_claim_transport_survives_nested_and_unused_calls",
        ),
        _row(
            "guard.local.declaration_order.eval_c",
            "lane_divergent",
            EXECUTES,
            "exec_c.local_reshape_guards_follow_declaration_order",
        ),
        _row(
            "shrink.to_end.nonzero_start",
            "silent_unguarded",
            "rejects_exactly",
            "ir_sources.to_end_shrink_end_requires_a_literal_zero_start",
        ),
        # Chelis#1797. The compiled lane already reported section 2.4.1's
        # overshoot; the eval lane answered it with a PANIC under a free or an
        # agreeing result claim, and with a claim mismatch under a disagreeing
        # one. The row therefore starts `lane_divergent`, and both halves now
        # print the same two lines for all three claim spellings.
        _row(
            "shrink.runtime_bound.overshoot.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.an_overshooting_shrink_span_reports_the_domain_error_on_eval",
        ),
        _row(
            "shrink.runtime_bound.overshoot.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.an_overshooting_shrink_span_reports_the_domain_error_on_c",
        ),
        # chelis#1836. A shape-computed route whose operand is still an
        # unresolved type variable when the route runs published the call's own
        # result variable, so a false declared shape checked at score 1 with no
        # errors on `6abca2406` and the runtime extent guard was the only thing
        # left to catch it. Four provenances, one per row: the two `match`
        # destructurings, the record field, and the chelis#1577 `copy` gate.
        # Single rows rather than `.c`/`.eval` pairs: the verdict is a CHECKER
        # rejection that no lane varies, and the programs never reach a lane.
        _row(
            "route.untied.sum.match",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_match_destructured_sum_operand_is_tied_to_the_bound_scrutinee",
        ),
        _row(
            "route.untied.matmul.match",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_match_destructured_matmul_operand_is_tied_to_the_bound_scrutinee",
        ),
        _row(
            "route.untied.sum.record",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_record_field_sum_operand_is_tied_to_the_bound_target",
        ),
        _row(
            "route.untied.sum.copy",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_gated_copy_result_ties_its_sum_consumer_to_the_bound_operand",
        ),
        # chelis#1836 through chelis#1690's operand gates. A suspended
        # `DeferredOperandGate::ShapeRoute` publishes a fresh result variable,
        # which is this issue's third provenance, so each of these routes
        # became a new instance when chelis#1690 landed. All three were
        # measured at score 1 with no errors on `e0a482248`, this branch's
        # base. `scatter` is the fourth gated route and is not registered: it
        # was not probed.
        _row(
            "route.untied.gather.gate",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_gather_gate_result_ties_its_sum_consumer_to_the_bound_operand",
        ),
        _row(
            "route.untied.trace.gate",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_trace_gate_result_ties_its_sum_consumer_to_the_bound_operand",
        ),
        _row(
            "route.untied.scatter_replace.gate",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_scatter_replace_gate_result_ties_its_sum_consumer_to_the_bound_operand",
        ),
        # chelis#1805: a tensor operand whose PRECISION is a type variable. The
        # dtype validators reject only a concrete inadmissible dtype and the
        # readiness predicate answers ready for a tensor whatever its precision
        # holds, so a float-only route reached through an unbounded `[p]` binder
        # checked clean and the compiled C lane printed `f = 2` for a true
        # 2.3333333. All four rows were measured accepted at score 1 on
        # `0820ee28e`. Each is a CHECKER verdict that no lane varies, so each is
        # one row rather than a pair.
        #
        # chelis#1942 replaces body inspection with a declared family contract
        # checked during ordinary inference. The original unbounded definitions
        # now fail at the declaration; explicitly bounded helpers retain the
        # restriction through indirect and transitive calls. The dedicated CLI
        # target also executes the five valid counterparts through eval and C.
        _row(
            "dtype.late_precision.instantiation",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_late_bound_tensor_precision_is_rejected_at_the_instantiation",
        ),
        _row(
            "dtype.late_precision.binds_one_application_later",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_precision_that_binds_one_application_later_is_decided_on_binding",
        ),
        _row(
            "dtype.late_precision.declared_bound",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.a_bounded_precision_binder_is_decided_by_its_family_at_once",
        ),
        _row(
            "dtype.late_precision.family_routes",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "types_unresolved_operand.every_family_policy_route_rejects_an_inadmissible_instantiation",
        ),
        _row(
            "dtype.late_precision.authored_contract",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_declared_operation_bounds.insufficient_authored_contracts_are_rejected_without_a_call_site",
        ),
        _row(
            "dtype.late_precision.indirect_and_transitive",
            "silent_unguarded",
            TERMINAL_CONTROL,
            "cli_declared_operation_bounds.invalid_function_value_chain_lambda_and_field_calls_stop_before_execution",
        ),
        # chelis#1822: the C preparation's `Expand` arm was gated on NO axis
        # carrying a real name, so a signature binder on a kept axis sent the
        # node to the pass-through arm and the operand's pre-expand extent was
        # stamped over the expanded axis. Only the C lane moved, so only the C
        # lane has a row for the two consumer forms; each receipt asserts the
        # eval lane as its byte-identity twin. The no-consumer form had nothing
        # to verify, so its prepared type reached codegen and the pair diverged
        # rather than failing the build, which is why both its lanes are rows.
        _row(
            "expand.named_bystander.consumer.c",
            "ice",
            EXECUTES,
            "cli_slice_b.a_named_bystander_axis_keeps_its_size_extent_on_c",
        ),
        _row(
            "expand.named_bystander.axis_zero.c",
            "ice",
            EXECUTES,
            "cli_slice_b.a_named_bystander_axis_keeps_its_size_extent_on_axis_zero_too",
        ),
        _row(
            "expand.named_bystander.no_consumer.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_named_bystander_expand_with_no_consumer_agrees_across_lanes",
        ),
        _row(
            "expand.named_bystander.no_consumer.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_named_bystander_expand_with_no_consumer_agrees_across_lanes",
        ),
        # chelis#1801. A nullary root whose extent arrives through a nested
        # helper's claim: the callee's instantiation variable met the inner
        # helper's runtime extent, nothing bound it, and def-level
        # generalization quantified it, so the root had no ABI and both lanes
        # dropped it in silence. `spec/04-type-system.md` section 3.2 now
        # makes that variable denote the extent it met.
        _row(
            "root.dim_variable.nested_helper.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_nested_helper_claim_sizes_a_root_on_c",
        ),
        _row(
            "root.dim_variable.nested_helper.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_nested_helper_claim_sizes_a_root_on_eval",
        ),
        # The same root reached through a polymorphic named def passed as an
        # ARGUMENT, so the callee variable this application minted aliases to
        # one the argument's own instantiation minted and that root is what
        # denotes the extent. chelis#1925's round 1 found this spelling still
        # dropped after the first repair.
        _row(
            "root.dim_variable.polymorphic_argument.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_polymorphic_argument_claim_sizes_a_root_on_c",
        ),
        _row(
            "root.dim_variable.polymorphic_argument.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_polymorphic_argument_claim_sizes_a_root_on_eval",
        ),
        # The same root whose extent a RESULT-ONLY binder names, which
        # chelis#1925's round-1 verification found LANE DIVERGENT on
        # `0820ee28e`: the C lane emitted an entry point and printed the value
        # while eval refused the same program with `missing symbolic dimension
        # binding \`seq\``. Both halves therefore start `lane_divergent` and
        # both now print the same line. Section 4.7 requires that agreement and
        # section 4.7.3 forbids a verdict that turns on a function boundary;
        # `main` already absorbed the one-call-shallower spelling.
        _row(
            "root.dim_variable.result_only_binder.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_result_only_binder_claim_sizes_a_root_on_c",
        ),
        _row(
            "root.dim_variable.result_only_binder.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_result_only_binder_claim_sizes_a_root_on_eval",
        ),
        # The same root through a THREE-member alias class, which is
        # chelis#1925's round-2 P1. `apply3` carries two polymorphic function
        # arguments beside the data one, so where the runtime-extent argument
        # sits decides which member roots the class when the meeting is
        # recorded. Every ordering was dropped in silence on `main`; the
        # receipt asserts the three render IDENTICALLY, because an
        # order-dependent absorption is the defect section 4.7.3 forbids and
        # three separate rows could all stay green through it.
        _row(
            "root.dim_variable.argument_order.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_three_member_alias_class_sizes_a_root_in_every_argument_order_on_c",
        ),
        _row(
            "root.dim_variable.argument_order.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_three_member_alias_class_sizes_a_root_in_every_argument_order_on_eval",
        ),
        # chelis#1771. A declared literal result extent reached through a block
        # tail, a let binding or a callee body. The `<op>` slot resolves to the
        # lowered primitive and the guard takes that primitive's source
        # position, so an effect bound after it is observed only when the guard
        # passes. All six were `silent_unguarded`: both lanes printed the
        # produced extent under a denying declaration and exited zero.
        _row(
            "return.block_bodied.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.the_c_lane_traps_on_a_block_bodied_return",
        ),
        _row(
            "return.block_bodied.literal.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.eval_traps_on_a_block_bodied_return",
        ),
        _row(
            "return.block_bodied.effect_order.c",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.the_c_lane_traps_before_an_effect_that_follows_the_producing_operation",
        ),
        _row(
            "return.block_bodied.effect_order.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.eval_traps_before_an_effect_that_follows_the_producing_operation",
        ),
        _row(
            "return.call_bodied.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.the_c_lane_traps_on_a_call_bodied_return",
        ),
        _row(
            "return.call_bodied.literal.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.eval_traps_on_a_call_bodied_return",
        ),
        # #1771/#1945: select the actual branch producer and carry each
        # caller's literal obligations to that producer inside a shared
        # callee. The call-order rows already rejected on the baseline, but
        # observed following effects before rejecting: that rejection did
        # not conform to the required observable ordering.
        _row(
            "return.if.literal.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.eval_selected_if_result_claims",
        ),
        _row(
            "return.if.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.c_selected_if_result_claims",
        ),
        _row(
            "return.match.literal.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.eval_selected_match_result_claims",
        ),
        _row(
            "return.match.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_return_boundary.c_selected_match_result_claims",
        ),
        _row(
            "return.call_bodied.effect_order.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_return_boundary.an_effect_inside_the_callee_after_the_producing_operation_is_suppressed",
        ),
        _row(
            "return.call_bodied.effect_order.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_return_boundary.an_effect_inside_the_callee_after_the_producing_operation_is_suppressed",
        ),
        _row(
            "return.shared_callee.literal.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_return_boundary.eval_shared_callee_result_claims_are_invocation_scoped",
        ),
        _row(
            "return.shared_callee.literal.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_return_boundary.c_shared_callee_result_claims_are_invocation_scoped",
        ),
        # #1771/#1945 continuation: measured eval and linked C baselines
        # silently accept inherited claims across the pure-helper route.
        _row(
            "return.pure_helper.literal.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_result_pure_helper.eval_inherited_result_claim_enters_pure_helpers",
        ),
        _row(
            "return.pure_helper.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_result_pure_helper.c_inherited_result_claim_enters_pure_helpers",
        ),
        # chelis#1923 and chelis#1791: pipe application semantics.
        # `spec/02-surf-syntax.md` section 0.1 says a pipe IS first-argument
        # insertion, and every consumer that met a `pipe` node reconstructed
        # that application for itself, not all the same way. The checker typed
        # a bare-name stage from the callee's FUNCTION type instead, so every
        # rule keyed on an application's arguments was lost downstream of it;
        # the lowerer bound the accumulator to a synthesized variable, so a
        # callee's own shape source stopped resolving. The two `to_tensor`
        # rows are the checker face, `expand_source` and `lint_fix` the
        # lowerer face. Both lanes carry a row wherever the pair diverged.
        _row(
            "pipe.bare_name_stage.to_tensor.expand.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_bare_name_pipe_stage_upstream_of_expand_checks_and_runs",
        ),
        _row(
            "pipe.bare_name_stage.to_tensor.expand.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_bare_name_pipe_stage_upstream_of_expand_checks_and_runs",
        ),
        _row(
            "pipe.bare_name_stage.to_tensor.sum.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_bare_name_pipe_stage_upstream_of_sum_checks_and_runs",
        ),
        _row(
            "pipe.bare_name_stage.to_tensor.sum.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_bare_name_pipe_stage_upstream_of_sum_checks_and_runs",
        ),
        _row(
            "pipe.bare_name_stage.expand_source.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_bare_name_stage_at_a_call_site_keeps_the_callees_expand_source",
        ),
        _row(
            "pipe.bare_name_stage.expand_source.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_bare_name_stage_at_a_call_site_keeps_the_callees_expand_source",
        ),
        _row(
            "pipe.bare_name_stage.lint_fix.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_lint_fix_of_a_direct_call_still_checks_evaluates_and_builds",
        ),
        # chelis#1791 half B: `check_expand_signature` matches the operand's
        # type before applying the size rule and its unresolved-operand arm
        # returns early, so in pipe position the rule was dropped and a
        # sourceless size reached the lowerer. The fold above repairs it
        # without touching that rule, because after the fold the operand is
        # resolved. One row, not a lane pair: this is a checker verdict, and
        # no lane varies once check rejects.
        _row(
            "expand.sourceless_size.pipe_position",
            "nonconforming_rejection",
            "rejects_exactly",
            "types_expand_size.issue1791_a_sourceless_size_rejects_in_pipe_position_too",
        ),
        # chelis#1788. One signature spelling `seq` on two parameter axes. In
        # the split-kernel tuple form each tensor leaf is lowered from its own
        # subexpression, so no DAG on this lane ever saw the binder twice: C
        # exited zero printing both outputs while eval refused, and eval refused
        # with a private sentence rather than the [04-NUM-9] pair. The
        # one-kernel twin already carried the correct rendering and is the lock
        # that the fix reproduces it without double-guarding.
        _row(
            "entry.host_tuple.repeated_binder.c",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_split_kernel_tuple_root_guards_its_repeated_binder_on_c",
        ),
        _row(
            "entry.host_tuple.repeated_binder.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_split_kernel_tuple_root_guards_its_repeated_binder_on_eval",
        ),
        _row(
            "entry.kernel.repeated_binder.c",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.a_one_kernel_root_keeps_its_repeated_binder_guard_on_both_lanes",
        ),
        _row(
            "entry.kernel.repeated_binder.eval",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.a_one_kernel_root_keeps_its_repeated_binder_guard_on_both_lanes",
        ),
        # #1788 residual: host entry owns every signature obligation before
        # body/helper execution. Eval already enforced these measured claims;
        # C omitted mixed witnesses or reported a later unrelated failure.
        # The enrolled target also checks literal/binder precedence, valid
        # independent signatures, unused witnesses and argument/body effects.
        _row(
            "entry.host_helper.unused_witness.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.mixed_helpers_keep_unused_signature_witnesses",
        ),
        _row(
            "entry.host_helper.unused_witness.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.mixed_helpers_keep_unused_signature_witnesses",
        ),
        _row(
            "entry.host_helper.signature_order.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.mixed_helpers_fail_in_signature_order",
        ),
        _row(
            "entry.host_helper.signature_order.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.mixed_helpers_fail_in_signature_order",
        ),
        _row(
            "entry.higher_order.invocation.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.higher_order_invocation_keeps_authored_entry_claim",
        ),
        _row(
            "entry.higher_order.invocation.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_signature_entry.higher_order_invocation_keeps_authored_entry_claim",
        ),
        _row(
            "entry.inline_callback.literal.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.inline_callback_entry_precedes_body_effects_on_every_invocation",
        ),
        _row(
            "entry.inline_callback.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.inline_callback_entry_precedes_body_effects_on_every_invocation",
        ),
        # chelis#1821: the gradient splice imported the forward activation and
        # rooted it nowhere, so the entry-point DCE removed it together with the
        # carrier holding its witness claims. The `live_forward` pair separates
        # the case where the whole forward chain is dead from the case where the
        # forward VALUES live and only the claim's carrier is unrooted; a repair
        # that kept reachability alone would close the first and leave the
        # second.
        #
        # The `wrt` ARGUMENT KIND is an axis of this corpus, enumerated from
        # the SPEC's category rather than from what a round happened to find.
        # Three rounds each found an unrecorded execution mode by varying that
        # kind, so a list grown witness by witness is the wrong
        # representation: `spec/04-type-system.md` lines 991-995 defines the
        # category as a "differentiable target", and the kinds it admits are a
        # float tensor of any rank, a float PRIM scalar, and an aggregate of
        # those (tuple, record, nested record, and a record with a
        # non-differentiable leaf). Each is crossed with single and multi
        # target.
        #
        # The three kinds are measured to behave differently and none folds
        # into another. A rank-0 `tensor[f32]` cotangent retains tensor
        # identity while a float prim cotangent crosses the typed
        # tensor-to-scalar boundary. The mixed-target receipt checks those
        # distinct public types and their written order. Every aggregate
        # spelling behaves identically, so one value covers all five of them.
        #
        # Every evaluator and generated-C cell retains the authored activation
        # contract. The two primitive-scalar C rows use the existing DAG
        # cotangent reconstruction route and are executable exit receipts.
        #
        # #1788 round-1 baseline e5cf8a51: eval enforces the lambda entry,
        # while beta-reduced C skips its signature and unused actual effects.
        _row(
            "entry.beta.literal.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_keep_literal_entry_and_eager_actuals",
        ),
        _row(
            "entry.beta.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_keep_literal_entry_and_eager_actuals",
        ),
        _row(
            "entry.beta.order.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_preserve_signature_order",
        ),
        _row(
            "entry.beta.order.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_preserve_signature_order",
        ),
        _row(
            "entry.beta.scopes.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_keep_outer_and_inner_claims_independent",
        ),
        _row(
            "entry.beta.scopes.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_keep_outer_and_inner_claims_independent",
        ),
        _row(
            "entry.beta.actual.eval",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_run_failing_actual_before_entry",
        ),
        _row(
            "entry.beta.actual.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.beta_reduced_callbacks_run_failing_actual_before_entry",
        ),
        # #1788 round-2 baseline 46bdc8a5: substituting a broader callable
        # actual erased the narrower formal's invocation-entry contract on
        # both host lanes. The three receipts separate one literal witness
        # from transport through aliases/local callables/wrappers and from a
        # multi-argument repeated binder. Each also checks eager argument
        # evaluation and excludes body/later effects on failure.
        _row(
            "entry.indirect.literal.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_callable_keeps_formal_literal_entry_boundary",
        ),
        _row(
            "entry.indirect.literal.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_callable_keeps_formal_literal_entry_boundary",
        ),
        _row(
            "entry.indirect.transport.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_callable_contract_survives_alias_local_and_wrapper_specialization",
        ),
        _row(
            "entry.indirect.transport.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_callable_contract_survives_alias_local_and_wrapper_specialization",
        ),
        _row(
            "entry.indirect.repeated_binder.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_callable_keeps_formal_repeated_binder_and_eager_actuals",
        ),
        _row(
            "entry.indirect.repeated_binder.c",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_callable_keeps_formal_repeated_binder_and_eager_actuals",
        ),
        # #1788 round-3 baseline f77ef306: C's checked adapter retained the
        # narrower formal boundary, but Eval attached that boundary only to a
        # Closure. Direct and locally bound grad values are both Transform
        # values; each receipt checks the failing length-three input and the
        # agreeing length-two control with exact argument/later-effect counts.
        _row(
            "entry.indirect.transform.direct.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.indirect_grad_transform_keeps_formal_entry_boundary",
        ),
        _row(
            "entry.indirect.transform.direct.c",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.indirect_grad_transform_keeps_formal_entry_boundary",
        ),
        _row(
            "entry.indirect.transform.local.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_signature_entry.local_indirect_grad_transform_keeps_formal_entry_boundary",
        ),
        _row(
            "entry.indirect.transform.local.c",
            EXECUTES,
            EXECUTES,
            "cli_signature_entry.local_indirect_grad_transform_keeps_formal_entry_boundary",
        ),
        _row(
            "grad.wrt_tensor.single.dead_forward.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.grad_over_a_disagreeing_named_claim_traps_on_eval",
        ),
        _row(
            "grad.wrt_tensor.single.dead_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.grad_over_a_disagreeing_named_claim_traps_on_c",
        ),
        _row(
            "grad.wrt_tensor.single.live_forward.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.grad_keeps_the_entry_carrier_when_the_backward_reads_the_forward_on_eval",
        ),
        _row(
            "grad.wrt_tensor.single.live_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.grad_keeps_the_entry_carrier_when_the_backward_reads_the_forward_on_c",
        ),
        # #1920/#1924: runtime transform lowering formerly used an inferred
        # result binder in place of the authored result signature. Both
        # interface witnesses existed, but no declared equality formed before
        # AD/DCE. Passing authored signatures separately restores the same
        # activation obligation for tensor and aggregate target selections.
        # Historical baseline states and row identities remain unchanged.
        _row(
            "grad.wrt_tensor.multi.dead_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_multi_target_grad_over_the_same_claim_is_still_lane_divergent",
        ),
        _row(
            "grad.wrt_tensor.multi.dead_forward.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.a_multi_target_grad_over_the_same_claim_is_still_lane_divergent",
        ),
        # #1932, recovered exactly from PR #1912 round 3. Vmap cloned each
        # authored ConstTensor payload while prepending the batch dimension,
        # so Eval panicked on cardinality and C reached a nonconforming
        # constant trap before the entry witness. The one receipt hashes the
        # 290-byte source, executes the refuted activation, and proves
        # agreeing zero/nonzero cotangents plus direct-grad and no-witness
        # controls through Eval and compiled C.
        _row(
            "grad.mapped.entry_witness.refuted.eval",
            "ice",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.refuted.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.agree_zero.eval",
            "ice",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.agree_zero.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.agree_nonzero.eval",
            "ice",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.agree_nonzero.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.refuted_reordered.eval",
            "ice",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.mapped.entry_witness.refuted_reordered.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes",
        ),
        _row(
            "grad.wrt_aggregate.single.dead_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_aggregate_typed_wrt_is_still_lane_divergent",
        ),
        _row(
            "grad.wrt_aggregate.single.dead_forward.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.an_aggregate_typed_wrt_is_still_lane_divergent",
        ),
        _row(
            "grad.wrt_aggregate.multi.dead_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.an_aggregate_typed_wrt_is_still_lane_divergent",
        ),
        _row(
            "grad.wrt_aggregate.multi.dead_forward.eval",
            "lane_divergent",
            EXECUTES,
            "cli_slice_b.an_aggregate_typed_wrt_is_still_lane_divergent",
        ),
        # #1934: primitive-scalar cotangents use the existing typed DAG pack
        # route, retaining the same forward claim and their scalar identity.
        _row(
            "grad.wrt_prim_scalar.single.dead_forward.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_prim_scalar_wrt_is_still_silent_on_eval_and_refused_on_c",
        ),
        _row(
            "grad.wrt_prim_scalar.single.dead_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_prim_scalar_wrt_is_still_silent_on_eval_and_refused_on_c",
        ),
        # Multiple selections preserve complete cotangent groups in written
        # order, including repeated targets and mixed scalar/tensor structures.
        _row(
            "grad.wrt_prim_scalar.multi.dead_forward.eval",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_prim_scalar_wrt_is_still_silent_on_eval_and_refused_on_c",
        ),
        _row(
            "grad.wrt_prim_scalar.multi.dead_forward.c",
            "silent_unguarded",
            EXECUTES,
            "cli_slice_b.a_prim_scalar_wrt_is_still_silent_on_eval_and_refused_on_c",
        ),
        # Main ecf213c18's exact tensor-target witnesses: agreeing independent
        # calls passed eval but falsely joined `n` on C; ordered entry failure
        # already named load on C but reached elementwise work on eval; the
        # computed-only claim reached later elementwise failure on both lanes.
        # The pre-v12 authored-signature draft also regressed the agreeing
        # evaluator case. These rows retain main's states, not that draft's.
        _row(
            "grad.independent_calls.agree.eval",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.independent_grad_entry_claims_agree_on_eval_and_c",
        ),
        _row(
            "grad.independent_calls.agree.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.independent_grad_entry_claims_agree_on_eval_and_c",
        ),
        _row(
            "grad.independent_calls.entry_order.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.independent_grad_entry_failures_are_ordered_on_eval_and_c",
        ),
        _row(
            "grad.independent_calls.entry_order.c",
            EXECUTES,
            EXECUTES,
            "cli_slice_b.independent_grad_entry_failures_are_ordered_on_eval_and_c",
        ),
        _row(
            "grad.independent_calls.computed_claim.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.independent_grad_computed_claims_keep_producer_on_eval_and_c",
        ),
        _row(
            "grad.independent_calls.computed_claim.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.independent_grad_computed_claims_keep_producer_on_eval_and_c",
        ),
        # chelis#1779: a runtime-shaped `to_tensor` lowers to a deliberate
        # rank-0 placeholder whose contract is to be refused so the definition
        # routes to the host lane. chelis#1693's staged host-source partition
        # runs ahead of the decision that reads that signal and cannot carry
        # the marker, so both lanes rejected a program that checks at 1.0.
        _row(
            "staged.dynamic_to_tensor.vmap_column.eval",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_runtime_shaped_to_tensor_column_routes_to_the_host_lane_on_both_lanes",
        ),
        _row(
            "staged.dynamic_to_tensor.vmap_column.c",
            "nonconforming_rejection",
            EXECUTES,
            "cli_slice_b.a_runtime_shaped_to_tensor_column_routes_to_the_host_lane_on_both_lanes",
        ),
        # Independent named binders retain separate C declarations per scope.
        _row(
            "entry.merged_scopes.declaration.c",
            "silent_unguarded",
            EXECUTES,
            "exec_c.issue_1788_two_scopes_in_one_function_share_one_declaration",
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


def render_baseline(spec: PhaseSpec) -> str:
    """The exact bytes one phase's checked baseline file must hold.

    The baseline is derived data: every field comes from the reviewed
    ``generated_phase_<p>_corpus()``. Writing it by hand is how chelis#1588
    and chelis#1742 both started, so ``--write-baseline`` renders it and a
    self-test asserts the checked files are byte-identical to this rendering.
    The digest is never typed; it is always ``corpus_digest``'s answer for
    the corpus the writer is serializing.
    """

    payload = {
        "schema_version": 1,
        "corpus_sha256": corpus_digest(spec.corpus, spec.phase),
        "rows": [
            {
                "id": row.id,
                "baseline": row.baseline,
                f"phase_{spec.phase}": row.exit_state,
                "receipt": row.receipt,
            }
            for row in spec.corpus
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def write_baseline(phase: str, registry: Mapping[str, PhaseSpec] | None = None) -> Path:
    """Rewrite one phase's checked baseline from its generated corpus.

    This regenerates derived data after a reviewed corpus edit. It is not a
    way to silence a digest-drift failure: the reviewed source is the corpus
    in this file, and phase A's digest stays pinned by
    ``FROZEN_PHASE_A_DIGEST`` in the self-test suite, which this writer
    cannot move.
    """

    active = PHASE_REGISTRY if registry is None else registry
    spec = active.get(phase)
    if spec is None:
        raise OracleFailure(f"phase {phase!r} has no registered corpus to write")
    spec.baseline_path.write_text(render_baseline(spec))
    return spec.baseline_path


_SELECTOR_MODES = ("all", "substring", "exact")
_TARGET_KINDS = ("test", "lib")
_ROW_REQUIRED = {"phase", "id", "package", "kind", "file", "selector", "expected"}
_ROW_OPTIONAL = {"list_only", "note"}


def load_target_manifest(path: Path | None = None) -> tuple[Mapping[str, object], ...]:
    """Read and structurally validate the reviewed test-target manifest.

    The manifest holds one row per cargo target the oracle runs: which test
    binary it selects, how it selects, and the exact set of per-test receipts
    that selection must produce. Two independent readers enforce it. This one
    turns each row into the command the oracle executes, so the command and
    the expectation cannot disagree. The other is
    ``crates/chelis-types/tests/runtime_extent_target_manifest.rs``, which
    parses the named sources and fails ``--fast`` when a rename, an addition
    under a substring selector, or a new ``#[ignore]`` moves a row's real
    inventory away from ``expected``. Before chelis#1742 the expectations were
    Python tuples that only this oracle read, and two merges drifted them.
    """

    manifest_path = TARGETS_PATH if path is None else path
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OracleFailure(f"cannot read target manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "targets"}:
        raise OracleFailure("target manifest must contain exactly schema_version and targets")
    if payload["schema_version"] != 1:
        raise OracleFailure(f"unsupported target manifest schema {payload['schema_version']!r}")
    rows = payload["targets"]
    if not isinstance(rows, list) or not rows:
        raise OracleFailure("target manifest must hold a non-empty targets list")
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OracleFailure(f"target manifest row {index} is not an object")
        keys = set(row)
        if not _ROW_REQUIRED <= keys or not keys <= _ROW_REQUIRED | _ROW_OPTIONAL:
            raise OracleFailure(f"target manifest row {index} has the wrong fields: {sorted(keys)}")
        if row["phase"] not in PHASES:
            raise OracleFailure(f"target manifest row {index} names unknown phase {row['phase']!r}")
        if row["kind"] not in _TARGET_KINDS:
            raise OracleFailure(f"target manifest row {index} has unknown kind {row['kind']!r}")
        identity = (row["phase"], row["id"])
        if identity in seen:
            raise OracleFailure(f"target manifest repeats target {identity[1]!r} in phase {identity[0]!r}")
        seen.add(identity)
        selector = row["selector"]
        if not isinstance(selector, dict) or selector.get("mode") not in _SELECTOR_MODES:
            raise OracleFailure(f"target {row['id']!r} has an unknown selector {selector!r}")
        if selector["mode"] == "substring":
            if set(selector) != {"mode", "value"} or not isinstance(selector["value"], str):
                raise OracleFailure(f"target {row['id']!r} needs one substring selector value")
        elif set(selector) != {"mode"}:
            raise OracleFailure(f"target {row['id']!r} carries an unused selector value")
        expected = row["expected"]
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(name, str) for name in expected)
        ):
            raise OracleFailure(f"target {row['id']!r} must name at least one expected test")
        if sorted(expected) != list(expected) or len(set(expected)) != len(expected):
            raise OracleFailure(f"target {row['id']!r} expected tests must be sorted and unique")
        source = row["file"]
        if not isinstance(source, str) or not source.startswith(f"crates/{row['package']}/"):
            raise OracleFailure(f"target {row['id']!r} file must live under its own package")
        if not (REPO_ROOT / source).is_file():
            raise OracleFailure(f"target {row['id']!r} names a missing source file {source!r}")
        if row.get("list_only") and selector["mode"] != "substring":
            # `--ignored --list` names its tests with one filter, so
            # `target_argv` reads `selector["value"]` for a listed row.
            # Without this the loader admits the row and the command builder
            # dies on a bare KeyError instead of a named failure.
            raise OracleFailure(
                f"target {row['id']!r} is listed, which needs a substring selector"
            )
        if row["kind"] == "test" and f"crates/{row['package']}/tests/" not in source:
            raise OracleFailure(f"target {row['id']!r} is an integration target outside tests/")
    return tuple(rows)


def target_argv(row: Mapping[str, object]) -> tuple[str, ...]:
    """The exact cargo command one manifest row selects.

    An ``exact`` row's names go AFTER ``--``, where the harness takes any
    number of filters. Cargo itself accepts a single ``[TESTNAME]``
    positional, so a multi-name row placed before ``--`` is rejected outright
    with "unexpected argument" and the target never runs. A ``substring``
    row's one value stays before ``--``, which is the form its command has
    always had.
    """

    selector = row["selector"]
    mode = selector["mode"]
    target_selection = (
        ["--test", Path(row["file"]).stem] if row["kind"] == "test" else ["--lib"]
    )
    head = ["cargo", "test", "-p", row["package"], *target_selection]
    if row.get("list_only"):
        tail = [selector["value"], "--", "--ignored", "--list"]
    elif mode == "exact":
        tail = ["--", "--exact", "--nocapture", *row["expected"]]
    elif mode == "substring":
        tail = [selector["value"], "--", "--nocapture"]
    else:
        tail = ["--", "--nocapture"]
    return tuple([*head, *tail])


def manifest_targets(
    phase: str, rows: Sequence[Mapping[str, object]] | None = None
) -> tuple[TestTarget, ...]:
    manifest = load_target_manifest() if rows is None else tuple(rows)
    selected = tuple(row for row in manifest if row["phase"] == phase)
    if not selected:
        raise OracleFailure(f"target manifest registers no target for phase {phase!r}")
    return tuple(
        TestTarget(
            id=str(row["id"]),
            argv=target_argv(row),
            expected_tests=tuple(row["expected"]),
            list_only=bool(row.get("list_only", False)),
        )
        for row in selected
    )


def self_test_target(python: str = sys.executable) -> TestTarget:
    """The oracle's own unit suite. Shared by every phase, so run once.

    It declares no expected tests, so it is the one target the manifest does
    not own: ``validate_target_receipt`` checks only its exit status, and
    there is no per-test expectation that could drift.
    """

    return TestTarget("self_tests", (python, "scripts/test_runtime_extent_oracle.py"))


def phase_a_targets(python: str = sys.executable) -> tuple[TestTarget, ...]:
    return (self_test_target(python), *manifest_targets("a"))


def phase_b_targets(python: str = sys.executable) -> tuple[TestTarget, ...]:
    return (self_test_target(python), *manifest_targets("b"))


PHASE_B_DEFERRED: Mapping[str, str] = {}


PHASE_A_DEFERRED: Mapping[str, str] = {
    "expand.input_axis.metal_device": (
        "runtime_extents.md C2.5: no Metal device-path expand row executes until "
        "chelis#1383 lands expand emission and symbolic-dim Load support"
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
        deferred=PHASE_B_DEFERRED,
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


def validate_deferral_keys(spec: PhaseSpec) -> None:
    """Every `deferred` key names a row of the phase it defers.

    A key with no row defers nothing and reads as though it did, so a typo
    would hide a real shortfall rather than record it. Round 2 of chelis#1912
    found this unchecked for both phases; it costs one set difference.
    """

    ids = {row.id for row in spec.corpus}
    orphans = sorted(set(spec.deferred) - ids)
    if orphans:
        raise SystemExit(
            f"phase {spec.phase!r} defers row ids that its corpus does not "
            f"contain, so they defer nothing: {', '.join(orphans)}"
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


_CLAIM_BARRIER_SOURCE = REPO_ROOT / "crates/chelis-ir/src/axis_sources.rs"
_CLAIM_BARRIER_NAMED_AND_LITERAL = """\
                        site: crate::dag::ExtentWitnessSite::ResultClaim { .. }
                            | crate::dag::ExtentWitnessSite::LiteralResultClaim,
"""
_CLAIM_BARRIER_LITERAL_ONLY = """\
                        site: crate::dag::ExtentWitnessSite::LiteralResultClaim,
"""
_SPECIALIZER_SOURCE = REPO_ROOT / "crates/chelis-ir/src/specialize.rs"
_BLAS_REGION_BARRIER = """\
        if let Some(info) = detect_matmul_pattern(dag, node.id)
            && matched_region_is_claim_free(
                &claimed_result_producers,
                info.replaced_region(node.id),
            )
"""
_BLAS_REGION_BARRIER_REMOVED = """\
        if let Some(info) = detect_matmul_pattern(dag, node.id)
"""
_DENSE_GATHER_REGION_BARRIER = """\
        if let Some(info) = detect_dense_gather_pattern(dag, node.id)
            && matched_region_is_claim_free(
                &claimed_result_producers,
                info.replaced_region(node.id),
            )
        {
"""
_DENSE_GATHER_REGION_BARRIER_REMOVED = """\
        if let Some(info) = detect_dense_gather_pattern(dag, node.id) {
"""


def remove_named_claim_from_producer_barrier(source: str) -> str:
    """Controlled mutation: regress the central barrier to literal-only."""

    if source.count(_CLAIM_BARRIER_NAMED_AND_LITERAL) != 1:
        raise OracleFailure(
            "claimed-producer barrier mutation anchor is missing or ambiguous"
        )
    return source.replace(
        _CLAIM_BARRIER_NAMED_AND_LITERAL,
        _CLAIM_BARRIER_LITERAL_ONLY,
        1,
    )


def validate_claimed_producer_barrier_mutation(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Prove Phase B detects a named-claim omission in the central barrier."""

    original = _CLAIM_BARRIER_SOURCE.read_text()
    mutated = remove_named_claim_from_producer_barrier(original)
    command = (
        "cargo",
        "test",
        "-p",
        "chelis-ir",
        "--test",
        "issue_1948_same_shape_result_claim_sources",
        "named_result_claim_traps_after_specialization_and_dce",
        "--",
        "--exact",
        "--nocapture",
    )
    try:
        _CLAIM_BARRIER_SOURCE.write_text(mutated)
        completed = _run_text(runner, command)
        output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode == 0:
            raise OracleFailure(
                "claimed-producer barrier mutation escaped the specialization receipt"
            )
        if "specialization must not eliminate a named claimed producer" not in output:
            raise OracleFailure(
                "claimed-producer barrier mutation failed for an unrelated reason"
            )
    finally:
        _CLAIM_BARRIER_SOURCE.write_text(original)
    if _CLAIM_BARRIER_SOURCE.read_text() != original:
        raise OracleFailure("claimed-producer barrier mutation did not restore its source")


def remove_specializer_region_barrier(source: str, recognizer: str) -> str:
    """Controlled mutation: let one recognizer bypass claimed producers."""

    anchors = {
        "blas": (_BLAS_REGION_BARRIER, _BLAS_REGION_BARRIER_REMOVED),
        "dense_gather": (
            _DENSE_GATHER_REGION_BARRIER,
            _DENSE_GATHER_REGION_BARRIER_REMOVED,
        ),
    }
    try:
        anchor, replacement = anchors[recognizer]
    except KeyError as error:
        raise OracleFailure(
            f"unknown specialization barrier mutation {recognizer!r}"
        ) from error
    if source.count(anchor) != 1:
        raise OracleFailure(
            f"{recognizer} specialization barrier mutation anchor is missing or ambiguous"
        )
    return source.replace(anchor, replacement, 1)


def validate_specializer_region_barrier_mutations(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Prove Phase B covers both multi-node specialization recognizers."""

    cases = (
        (
            "blas",
            "blas_interior_named_claim_blocks_replacement_and_traps_after_dce",
        ),
        (
            "dense_gather",
            "dense_gather_interior_named_claim_blocks_replacement_and_traps_after_dce",
        ),
    )
    original = _SPECIALIZER_SOURCE.read_text()
    for recognizer, test in cases:
        command = (
            "cargo",
            "test",
            "-p",
            "chelis-ir",
            "--test",
            "issue_1948_same_shape_result_claim_sources",
            test,
            "--",
            "--exact",
            "--nocapture",
        )
        try:
            _SPECIALIZER_SOURCE.write_text(
                remove_specializer_region_barrier(original, recognizer)
            )
            completed = _run_text(runner, command)
            output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode == 0:
                raise OracleFailure(
                    f"{recognizer} specialization barrier mutation escaped its receipt"
                )
            if "a declined replacement must leave a valid executable graph" not in output:
                raise OracleFailure(
                    f"{recognizer} specialization barrier mutation failed for an unrelated reason"
                )
        finally:
            _SPECIALIZER_SOURCE.write_text(original)
        if _SPECIALIZER_SOURCE.read_text() != original:
            raise OracleFailure(
                f"{recognizer} specialization barrier mutation did not restore its source"
            )


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
    allow_shortfall: bool = False,
) -> tuple[str, str]:
    """Run one phase's obligations and report.

    ``allow_shortfall`` separates the two ways a phase can be red. Every
    receipt, digest and lattice obligation still fails the run. Only the
    recorded row shortfall, which is the tracker's published state rather
    than a regression, is downgraded to a report: the run prints the head,
    the corpus digests and the rows still short, ends with ``SHORT_MARKER``
    instead of ``PASS_MARKER``, and returns zero. It is what lets a nightly
    job enforce the receipts of a phase whose rows have not all landed;
    without it the phase's real drift and its expected shortfall would be
    the same red, which is the confusion chelis#1742 is about. Phase
    ``final`` refuses the flag: the completion oracle may not excuse its
    own short rows.

    No registered phase records a shortfall today, so the nightly job passes
    the flag nowhere and demands ``PASS`` from both phases. The flag stays
    because the next phase to register a corpus starts with rows short of
    exit, and because retiring it would retire the only difference between
    a landing row and a drifted receipt; its behaviour is held by the
    self-tests' synthetic short rows rather than by a live phase.
    """

    if phase not in PHASES:
        raise OracleFailure(f"unsupported phase {phase!r}")
    if phase == "final" and allow_shortfall:
        # `final` is the class completion oracle. A completion claim that
        # excuses its own short rows is not a completion claim, so the flag
        # that lets a nightly hold a still-landing phase is refused here
        # outright rather than quietly ignored.
        raise OracleFailure(
            "phase 'final' is the completion oracle and cannot allow a row shortfall"
        )

    # The lattice binds the whole recorded chain on every invocation, so a
    # leftward move in a later phase fails an earlier phase's run too.
    validate_phase_chain(registered_specs(registry))

    selected = selected_specs(phase, registry)
    for spec in selected:
        validate_deferral_keys(spec)
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
    if registry is None and targets is None and phase in ("b", "final"):
        validate_claimed_producer_barrier_mutation(runner)
        validate_specializer_region_barrier_mutations(runner)

    shortfall = [
        (spec.phase, row_id) for spec in selected for row_id in exit_shortfall(spec)
    ]
    listed = ", ".join(f"{p}:{row}" for p, row in shortfall)
    if shortfall and not allow_shortfall:
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
    if shortfall:
        print(f"runtime_extent_rows_short={len(shortfall)}", flush=True)
        print(f"runtime_extent_rows_short_list={listed}", flush=True)
        print(SHORT_MARKER, flush=True)
    else:
        print(PASS_MARKER, flush=True)
    return head, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Runtime-extent class oracle for chelis#1277. Phase A's wire "
            "capacity leg executes the Python binding facade through the "
            "interpreter PYO3_PYTHON names, falling back to the checkout's "
            ".venv, NOT through the interpreter running this script. That "
            "interpreter needs the facade's dependencies: uv pip install "
            "--python <that interpreter> -r bindings/python/pyproject.toml. "
            "Without them the leg fails with a ModuleNotFoundError that "
            "reads like a census defect."
        )
    )
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help=(
            "report a recorded row shortfall instead of failing on it, so a "
            "job can enforce a phase's receipts while its rows are still "
            "landing; every other obligation still fails the run"
        ),
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=(
            "rewrite the selected phase's checked baseline from its generated "
            "corpus instead of validating, for use after a reviewed corpus edit"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.write_baseline:
            print(f"wrote {write_baseline(args.phase)}", flush=True)
            return 0
        validate(args.phase, allow_shortfall=args.allow_shortfall)
    except OracleFailure as error:
        print(f"RUNTIME EXTENT ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
