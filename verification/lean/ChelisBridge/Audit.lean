import ChelisBridge.Float
import ChelisBridge.Tensor
import ChelisBridge.Model
import ChelisBridge.Decode
import ChelisBridge.Ownership
import ChelisBridge.Validation
import ChelisBridge.Examples
import ChelisBridge.Runtime.Cancellation
import ChelisBridge.Runtime.Stack
import ChelisBridge.Runtime.Ids
import ChelisBridge.Runtime.Unwind
import ChelisBridge.Runtime.Scopes
import ChelisBridge.Numeric
import ChelisBridge.Numeric.Flat
import ChelisBridge.Numeric.Decode
import ChelisBridge.Numeric.Primal
import ChelisBridge.Numeric.Examples
import Lean.Util.CollectAxioms

/-!
# Axiom audit

Elaborating this module fails unless

* every theorem in `accepted` (the accepted bridge results) exists and depends only on
  `propext`, `Classical.choice` and `Quot.sound` (so no `sorryAx`, no `Lean.ofReduceBool`
  or `native_decide` auxiliary axiom, no new axiom);
* every other theorem declared in an imported `ChelisBridge` module satisfies the same
  axiom bound (nothing unaccepted may hide an escape hatch);
* no `ChelisBridge` module (imported or this one) declares an axiom or an `opaque` constant.
-/

namespace ChelisBridge.Audit

open Lean Elab Command

/-- Theorems accepted as bridge results (the final accepted set). -/
def accepted : List Name := [
  -- Float
  ``ChelisBridge.representable_abs_iff,
  ``ChelisBridge.representable_neg_iff,
  ``ChelisBridge.width_sub_one,
  ``ChelisBridge.exponentBits_pos,
  ``ChelisBridge.maxExponent_eq,
  ``ChelisBridge.ratio_snd_pos,
  ``ChelisBridge.ratio_value,
  ``ChelisBridge.value_eq,
  ``ChelisBridge.realize_eq_some,
  ``ChelisBridge.isNaN_eq_false_of_isFinite,
  ``ChelisBridge.isInfinite_eq_false_of_isFinite,
  ``ChelisBridge.maxExponent_pos,
  ``ChelisBridge.two_pow_width,
  ``ChelisBridge.pack_low,
  ``ChelisBridge.pack_toNat,
  ``ChelisBridge.pack_fields,
  ``ChelisBridge.realize_pack,
  ``ChelisBridge.realize_zero,
  ``ChelisBridge.zero_ne_zero_neg,
  ``ChelisBridge.two_zpow_pos,
  ``ChelisBridge.two_pow_nat_eq_zpow,
  ``ChelisBridge.ratioExponent_spec,
  ``ChelisBridge.roundQuotient_exact,
  ``ChelisBridge.round_exact,
  ``ChelisBridge.round_exact_int,
  ``ChelisBridge.value_ratio,
  ``ChelisBridge.realize_add,
  ``ChelisBridge.realize_mul,
  ``ChelisBridge.realize_add_of_exact,
  ``ChelisBridge.realize_mul_of_exact,
  ``ChelisBridge.realize_c16,
  ``ChelisBridge.realize_cneg16,
  ``ChelisBridge.realize_cone,
  ``ChelisBridge.add_not_associative,
  ``ChelisBridge.finite_add_not_exact,
  ``ChelisBridge.not_finite_add_exact,
  -- Tensor
  ``ChelisBridge.mapM_eq_some_iff,
  ``ChelisBridge.mapM_eq_none_of_mem,
  ``ChelisBridge.realizeList_eq_some,
  ``ChelisBridge.realizeList_eq_none_of_nonfinite,
  ``ChelisBridge.ofList_apply,
  ``ChelisBridge.ofList_length,
  ``ChelisBridge.decodeTensor_get,
  ``ChelisBridge.decodeTensor_length,
  ``ChelisBridge.decodeTensor_eq_none_of_length,
  ``ChelisBridge.decodeTensor_eq_none_of_nonfinite,
  ``ChelisBridge.decodeTensor_isSome_iff,
  ``ChelisBridge.Axis.prod_eq,
  ``ChelisBridge.flat_split_of_one,
  ``ChelisBridge.flat_split_singleton,
  ``ChelisBridge.ofList_reshape_singleton,
  ``ChelisBridge.decodeTensor_reshape_singleton,
  ``ChelisBridge.ofList_remove_singleton,
  ``ChelisBridge.decodeTensor_remove_singleton,
  ``ChelisBridge.decodeValues_get,
  -- Decode
  ``ChelisBridge.Var.offset_of_decode,
  ``ChelisBridge.Axis.position_of_decode,
  ``ChelisBridge.Operation.encode_of_decode,
  ``ChelisBridge.decodePrefix_spec,
  ``ChelisBridge.Var.owner_eq_of_offset,
  ``ChelisBridge.portVars_offsets,
  ``ChelisBridge.State.offset_spec,
  ``ChelisBridge.Kind.ports_build,
  ``ChelisBridge.decodeOperation_ports,
  ``ChelisBridge.decodeOperation_typed_ports,
  ``ChelisBridge.classifyBinary_spec,
  ``ChelisBridge.classify_add,
  ``ChelisBridge.classify_mul,
  ``ChelisBridge.classify_expand,
  ``ChelisBridge.classify_reshape,
  ``ChelisBridge.reshape_value,
  ``ChelisBridge.reshape_remove_singleton_value,
  ``ChelisBridge.classify_constTensor,
  ``ChelisBridge.classify_const,
  ``ChelisBridge.decodeBuf_tensor,
  ``ChelisBridge.step_len,
  ``ChelisBridge.run_len,
  ``ChelisBridge.decodeGraph_len,
  ``ChelisBridge.decodeRootOffsets_spec,
  ``ChelisBridge.entry_owner,
  -- Ownership
  ``ChelisBridge.Ownership.guard_some,
  ``ChelisBridge.Ownership.Owned.encode_of_decode,
  ``ChelisBridge.Ownership.rootStep_rootClone,
  ``ChelisBridge.Ownership.rootStep_transferred,
  ``ChelisBridge.Ownership.runRoots_transferred,
  ``ChelisBridge.Ownership.nodeStep_transfer,
  ``ChelisBridge.Ownership.runNodes_transfer,
  ``ChelisBridge.Ownership.runScopeDrops_transfer,
  ``ChelisBridge.Ownership.planInstrs_transfers_nodup,
  ``ChelisBridge.Ownership.decodeOwned_sound,
  -- Validation
  ``ChelisBridge.opReads_iff,
  ``ChelisBridge.Var.get_zero,
  ``ChelisBridge.Var.get_add,
  ``ChelisBridge.Var.get_scatter_of_ne,
  ``ChelisBridge.Op.vjp_get_of_not_reads,
  ``ChelisBridge.vjp_get_of_not_needs,
  ``ChelisBridge.grad_get_of_disconnected,
  ``ChelisBridge.rootSource_present,
  ``ChelisBridge.rootSource_missing,
  ``ChelisBridge.classifyRoots_eq_none,
  ``ChelisBridge.classifyRoots_spec,
  ``ChelisBridge.totalizeAux_spec,
  ``ChelisBridge.totalizeParts_eq_none_of_missing,
  ``ChelisBridge.totalize_eq_none_of_missing,
  ``ChelisBridge.totalizeParts_present,
  ``ChelisBridge.totalize_spec,
  ``ChelisBridge.artifactParts_spec,
  ``ChelisBridge.artifactParts_eq_none_of_missing,
  ``ChelisBridge.decodeArtifact_eq_none_of_missing,
  ``ChelisBridge.decodeArtifact_eq_none_of_lookup_failure,
  ``ChelisBridge.decodeArtifact_missing_disconnected,
  ``ChelisBridge.checkedGrad_bridge,
  ``ChelisBridge.acceptGradient_sound,
  ``ChelisBridge.acceptExtracted_sound,
  ``ChelisBridge.Var.getElem_offset,
  ``ChelisBridge.forall₂_getElem?,
  ``ChelisBridge.decodeRoots_get,
  ``ChelisBridge.Target.decode_spec,
  ``ChelisBridge.rootOffsets_get,
  ``ChelisBridge.decodeArtifact_spec,
  ``ChelisBridge.decodeArtifactRoots_spec,
  ``ChelisBridge.artifact_root_link,
  ``ChelisBridge.Validation.checkGradient_bridge,
  -- Runtime (RuntimeProofs)
  ``ChelisBridge.Runtime.Cancellation.permitted_iff_monotone,
  ``ChelisBridge.Runtime.Cancellation.no_visibility_bound,
  ``ChelisBridge.Runtime.Cancellation.polling_outcome,
  ``ChelisBridge.Runtime.Cancellation.frozen_entry_not_equivalent,
  ``ChelisBridge.Runtime.Cancellation.scoped_token_restores,
  ``ChelisBridge.Runtime.Cancellation.thread_isolation,
  ``ChelisBridge.Runtime.Stack.first_bail_equivalence,
  ``ChelisBridge.Runtime.Stack.old_balanced,
  ``ChelisBridge.Runtime.Stack.new_balanced,
  ``ChelisBridge.Runtime.Stack.invocation_boundary,
  ``ChelisBridge.Runtime.Stack.nonzero_entry_depth_counterexample,
  ``ChelisBridge.Runtime.Stack.reuse_after_unwind_counterexample,
  ``ChelisBridge.Runtime.Ids.distinct_below_bound,
  ``ChelisBridge.Runtime.Ids.u32_wrap_counterexample,
  ``ChelisBridge.Runtime.Ids.persistence,
  ``ChelisBridge.Runtime.Ids.distinct_across_invocations,
  ``ChelisBridge.Runtime.Ids.reset_counterexample,
  ``ChelisBridge.Runtime.Ids.capacity_distinct,
  ``ChelisBridge.Runtime.Ids.capacity_succeeds_below_bound,
  ``ChelisBridge.Runtime.Unwind.unwind_equivalence,
  ``ChelisBridge.Runtime.Unwind.entry_equivalence,
  ``ChelisBridge.Runtime.Unwind.outcomes_distinct,
  ``ChelisBridge.Runtime.Unwind.panicking_finalizer_counterexample,
  ``ChelisBridge.Runtime.Unwind.nested_catch_clears,
  ``ChelisBridge.Runtime.Unwind.clear_not_restore,
  ``ChelisBridge.Runtime.Unwind.fatal_reraise_reenables,
  ``ChelisBridge.Runtime.Unwind.other_threads_unchanged,
  ``ChelisBridge.Runtime.Unwind.hook_reads_current_thread,
  ``ChelisBridge.Runtime.Scopes.restores_of_spec,
  ``ChelisBridge.Runtime.Scopes.clears_of_spec,
  ``ChelisBridge.Runtime.Scopes.scope_obligations_are_satisfiable,
  ``ChelisBridge.Runtime.Scopes.restore_ne_clear,
  ``ChelisBridge.Runtime.Scopes.model_speculate_clears,
  ``ChelisBridge.Runtime.Scopes.model_restore_restores,
  ``ChelisBridge.Runtime.Scopes.model_rollback_restores,
  -- Numeric (BridgeNumeric)
  ``ChelisBridge.Numeric.pairLevelQ_sum,
  ``ChelisBridge.Numeric.pairLevel_length_le,
  ``ChelisBridge.Numeric.pairLevel_realize,
  ``ChelisBridge.Numeric.pairFoldN_realize,
  ``ChelisBridge.Numeric.pairFold_realize,
  ``ChelisBridge.Numeric.Realizes.get,
  ``ChelisBridge.Numeric.Realizes.cons,
  ``ChelisBridge.Numeric.Realizes.read,
  ``ChelisBridge.Numeric.sumLeaves_realize,
  ``ChelisBridge.Numeric.opEval_realize,
  ``ChelisBridge.Numeric.exact_run_realize,
  ``ChelisBridge.Numeric.ieeeEval_realize,
  ``ChelisBridge.Numeric.programEval_realize,
  ``ChelisBridge.Numeric.toReal_get,
  ``ChelisBridge.Numeric.toReal_cons,
  ``ChelisBridge.Numeric.op_eval_toReal,
  ``ChelisBridge.Numeric.prefix_eval_toReal,
  ``ChelisBridge.Numeric.read_toReal,
  ``ChelisBridge.Numeric.circuit_eval_toReal,
  ``ChelisBridge.Numeric.program_eval_toReal,
  ``ChelisBridge.Numeric.ieeeEval_realize_real,
  ``ChelisBridge.Numeric.scalars_realizes,
  ``ChelisBridge.Numeric.assoc_same_reference,
  ``ChelisBridge.Numeric.reference_value,
  ``ChelisBridge.Numeric.reassociation_changes_result,
  ``ChelisBridge.Numeric.rightAssoc_all_finite,
  ``ChelisBridge.Numeric.finite_run_not_exact,
  ``ChelisBridge.Numeric.rust_sum_order_matters,
  ``ChelisBridge.Numeric.mul_overflow_not_finite,
  ``ChelisBridge.Numeric.flat_lt,
  ``ChelisBridge.Numeric.linearToIndex_flat,
  ``ChelisBridge.Numeric.indexToLinear_coords,
  ``ChelisBridge.Numeric.coords_join,
  ``ChelisBridge.Numeric.coords_split,
  ``ChelisBridge.Numeric.length_coords,
  ``ChelisBridge.Numeric.axis_eraseIdx,
  ``ChelisBridge.Numeric.insertAt_eraseIdx,
  ``ChelisBridge.Numeric.atPosition_position,
  ``ChelisBridge.Numeric.ofList_eq_some_of,
  ``ChelisBridge.Numeric.at'_flat,
  ``ChelisBridge.Numeric.gather_flat,
  ``ChelisBridge.Numeric.zipWith_flat,
  ``ChelisBridge.Numeric.reductionGroups_flat,
  ``ChelisBridge.Numeric.flatSum_eq,
  ``ChelisBridge.Numeric.expandPicks_length,
  ``ChelisBridge.Numeric.expand_insert_flat,
  ``ChelisBridge.Numeric.expand_same_rank_flat,
  ``ChelisBridge.Numeric.reshape_insert_flat,
  ``ChelisBridge.Numeric.reshape_remove_flat,
  ``ChelisBridge.Numeric.ofList_map,
  ``ChelisBridge.Numeric.decode_mapData,
  ``ChelisBridge.Numeric.decodePrefix_mapData,
  ``ChelisBridge.Numeric.targetDecode_mapData,
  ``ChelisBridge.Numeric.ratOp_eq_mapOp,
  ``ChelisBridge.Numeric.ratPrefix_eq_mapPrefix,
  ``ChelisBridge.Numeric.liftOp_sound,
  ``ChelisBridge.Numeric.step_ops,
  ``ChelisBridge.Numeric.runF_sound,
  ``ChelisBridge.Numeric.decodeGraphF_sound,
  ``ChelisBridge.Numeric.decodeCircuitF_sound,
  ``ChelisBridge.Numeric.step_ops_nonload,
  ``ChelisBridge.Numeric.Kind.build_nonconst,
  ``ChelisBridge.Numeric.classify_nonconst,
  ``ChelisBridge.Numeric.scalarBits_of_decodeScalar,
  ``ChelisBridge.Numeric.map_value_of_forall₂,
  ``ChelisBridge.Numeric.map_value_of_realizeList,
  ``ChelisBridge.Numeric.bufBits_of_decodeBuf,
  ``ChelisBridge.Numeric.payload_none,
  ``ChelisBridge.Numeric.liftOp_nonconst,
  ``ChelisBridge.Numeric.liftOp_decodeOperation,
  ``ChelisBridge.Numeric.runF_complete,
  ``ChelisBridge.Numeric.decodeCircuitF_complete,
  ``ChelisBridge.Numeric.decoded_graph_realize,
  ``ChelisBridge.Numeric.progCircuit_eval,
  ``ChelisBridge.Numeric.selectRoot_eval,
  ``ChelisBridge.Numeric.primalCheck_sound,
  ``ChelisBridge.Numeric.acceptPrimal_sound
]

def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def checkTheorem (n : Name) : CommandElabM Unit := do
  let axioms ← liftCoreM (collectAxioms n)
  let bad := axioms.filter (fun a => !allowedAxioms.contains a)
  unless bad.isEmpty do
    throwError "{n} depends on disallowed axioms {bad.toList}"

/-- Fail unless every name is a theorem within the axiom bound, every theorem of an
imported `ChelisBridge` module is within the bound, and no constant of a `ChelisBridge`
module is an axiom or `opaque`. -/
def audit (names : List Name) : CommandElabM Unit := do
  let env ← getEnv
  for n in names do
    let some info := env.find? n | throwError "accepted theorem {n} does not exist"
    unless info matches .thmInfo _ do throwError "accepted result {n} is not a theorem"
    checkTheorem n
  let check (n : Name) : CommandElabM Unit := do
    match env.find? n with
    | some (.axiomInfo _) => throwError "bridge module declares axiom {n}"
    | some (.opaqueInfo _) => throwError "bridge module declares opaque constant {n}"
    | some (.thmInfo _) => checkTheorem n
    | _ => pure ()
  let mut scanned := 0
  for (mod, data) in env.header.moduleNames.zip env.header.moduleData do
    if (`ChelisBridge).isPrefixOf mod then
      for n in data.constNames do
        check n
        scanned := scanned + 1
  for (n, _) in env.constants.map₂.toList do
    check n
  if scanned == 0 then throwError "no ChelisBridge module constants were scanned"
  logInfo m!"ChelisBridge audit: {names.length} accepted theorems and {scanned} package constants use only {allowedAxioms}"

elab "#chelis_bridge_audit" : command => audit accepted

#chelis_bridge_audit

end ChelisBridge.Audit
