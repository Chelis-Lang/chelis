-- LaCaDiLE/AdjointTyping.lean — adjoint typing lemma (Phase 2 proof).
--
-- WS2.2 target: the adjoint transformation preserves typing. Per the
-- E-Grad reduction (figures/opsem.tex), `grad(λx:τ.e)` reduces to
--    λx:τ. λgs:τ_out. handle[{Accum}] (adjoint e x gs) with h_accum
-- so the *body of the handle* is `adjoint e x gs`. The handle's clause
-- assembles the parameter gradient and returns it as tensor[ds]; the
-- adjoint body itself is a sequence of `perform accum (...)` calls
-- whose head term has type `unit` and effect row `{accum}` (plus the
-- forwarded effects of the original body, which must lie in
-- DiffCompat).
--
-- Wave 3 calculus refinement: T-Perform now uses an `OpSigMatch`
-- relation that lets `perform accum` take a tensor argument. This
-- unblocks the var/const/unit/loc base cases, which all emit
-- `Term.perform EffectLabel.accum gSeed` with `gSeed : tensor[dsOut]`.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.AdjointTransform

namespace LaCaDiLE

/-- The adjoint transformation preserves typing.

    Statement (Wave 3 corrected): given a body `e` that produces a
    `tensor[dsOut]` under the differentiated parameter `x`, and a seed
    `gs : tensor[dsOut]` already bound in the context, the term
    `adjoint e x (var gs)` type-checks at type `unit` (the
    handle-body type — the parameter gradient is delivered through
    `accum` effects, not as a return value) with the original effect
    row extended by `{accum}`.

    The previous statement claimed result type `tensor[ds]`; that was
    wrong because `adjoint` emits `perform accum (...)` whose result
    type is `unit`. The grad-handle wraps the adjoint body and turns
    the accumulated `accum` effects into a tensor[ds] return value via
    its handler clause. -/
theorem adjoint_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (_h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, Typ.tensor ds)])
                   e (Typ.tensor dsOut) eps Gamma)
    (_h_compat : subsetEffRow eps DiffCompat = true) :
    HasType Delta Sigma
            (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
            (adjoint e x (Term.var gs))
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, Typ.tensor ds)]) := by
  -- Wave 3 calculus refactor unblocks the proof; structural induction
  -- on `e` discharges each former. Each case must construct a
  -- HasType derivation for the corresponding `adjoint`-emitted term:
  --
  --   * base cases (var, const, unit, loc): emit `perform accum gSeed`
  --     where gSeed : tensor[dsOut]. Witnessed by HasType.var on `gs`,
  --     then HasType.perform with `OpSigMatch.accumTensor dsOut`.
  --   * add: letpair on copy(gSeed) routing two tensor halves to
  --     adjoint sub-terms. Recursive use of the IH on each operand.
  --   * mul: tape-and-copy structure; uses linearity carefully.
  --   * sum/expand: structural recursion on shape-shifted gSeed.
  --   * letBind/letpair/pair/fst/snd/copy/abs/app/grad/vmap/perform:
  --     Phase 1 vestigial recursions; the IH suffices for each.
  --   * handle: recurse on body and on each clause body via
  --     adjointClauses; needs an auxiliary inductive over clause lists.
  --
  -- Witness that the var/const/unit/loc base cases discharge: the
  -- emitted term is `Term.perform EffectLabel.accum (Term.var gs)`,
  -- which type-checks via T-Var on `gs` plus T-Perform with the new
  -- `OpSigMatch.accumTensor dsOut` witness. Build the witness once,
  -- locally, so that the structural induction below can reuse it.
  have base_perform :
      HasType Delta Sigma
        (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
        (Term.perform EffectLabel.accum (Term.var gs))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] [])
        (Gamma ++ [(x, Typ.tensor ds)]) := by
    -- T-Var consumes `gs` from the tail of the context.
    have hvar :
        HasType Delta Sigma
          (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
          (Term.var gs) (Typ.tensor dsOut) []
          (Gamma ++ [(x, Typ.tensor ds)]) := by
      -- Pre-context = Gamma ++ [(x, ds)], post = []
      have h := HasType.var Delta Sigma
        (Gamma ++ [(x, Typ.tensor ds)]) [] gs (Typ.tensor dsOut)
      -- Normalize the appended-empty postfix
      simpa using h
    -- T-Perform with OpSigMatch.accumTensor.
    exact HasType.perform Delta Sigma
      (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
      (Gamma ++ [(x, Typ.tensor ds)])
      EffectLabel.accum (Term.var gs)
      (Typ.tensor dsOut) Typ.unit []
      hvar (OpSigMatch.accumTensor dsOut)
  -- Full structural induction on `e` is Wave 3 follow-up. The base
  -- witness above proves the calculus refactor is internally
  -- consistent and that the var/const/unit/loc cases will discharge
  -- via `base_perform` once the induction is set up.
  sorry

end LaCaDiLE
