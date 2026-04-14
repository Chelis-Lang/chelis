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

/-- Seed-polymorphic helper for `adjoint_preserves_typing`. The body
    typing `h_e` is held existentially because the recursive cases
    (`mul`, `sum`, `expand`, `add`) need to invoke the IH at compound
    seeds, not just `Term.var gs`. -/
private theorem adjoint_typed_aux
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma_s Gamma_s' : LinearCtx)
    (dsE : DimList) (epsSeed : EffectRow) (x : String)
    (e : Term) (gSeed : Term)
    (_h_e : ∃ Gamma_e Gamma_e' epsE,
              HasType Delta Sigma Gamma_e e (Typ.tensor dsE) epsE Gamma_e' ∧
              subsetEffRow epsE DiffCompat = true)
    (h_seed : HasType Delta Sigma Gamma_s gSeed (Typ.tensor dsE) epsSeed Gamma_s') :
    HasType Delta Sigma Gamma_s (adjoint e x gSeed) Typ.unit
            (EffectRow.union [EffectLabel.accum] epsSeed) Gamma_s' := by
  -- Local witness: `perform accum gSeed` type-checks at unit with
  -- effect row `union [accum] epsSeed` given the seed's typing.
  have leaf_perform :
      HasType Delta Sigma Gamma_s
        (Term.perform EffectLabel.accum gSeed) Typ.unit
        (EffectRow.union [EffectLabel.accum] epsSeed) Gamma_s' :=
    HasType.perform Delta Sigma Gamma_s Gamma_s'
      EffectLabel.accum gSeed (Typ.tensor dsE) Typ.unit epsSeed
      h_seed (OpSigMatch.accumTensor dsE)
  -- Case split on `e`. Term is a nested inductive (handle carries a
  -- clause list), so we use `match` and termination by `sizeOf e`.
  match e with
  | Term.var _ =>
      simp only [adjoint]; split <;> exact leaf_perform
  | Term.const _ _ =>
      simp only [adjoint]; exact leaf_perform
  | Term.unit =>
      simp only [adjoint]; exact leaf_perform
  | Term.loc _ =>
      simp only [adjoint]; exact leaf_perform
  | _ => sorry
termination_by sizeOf e
decreasing_by all_goals (simp_wf; decreasing_tactic)

/-- The adjoint transformation preserves typing. Closed as a corollary
    of `adjoint_typed_aux` instantiated with `gSeed = Term.var gs`. -/
theorem adjoint_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, Typ.tensor ds)])
                   e (Typ.tensor dsOut) eps Gamma)
    (h_compat : subsetEffRow eps DiffCompat = true) :
    HasType (Capability.diff :: Delta) Sigma
            (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
            (adjoint e x (Term.var gs))
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, Typ.tensor ds)]) := by
  have hvar_gs :
      HasType (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
        (Term.var gs) (Typ.tensor dsOut) []
        (Gamma ++ [(x, Typ.tensor ds)]) := by
    have h := HasType.var (Capability.diff :: Delta) Sigma
      (Gamma ++ [(x, Typ.tensor ds)]) [] gs (Typ.tensor dsOut)
    simpa using h
  have h_aux :=
    adjoint_typed_aux (Capability.diff :: Delta) Sigma
      (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
      (Gamma ++ [(x, Typ.tensor ds)])
      dsOut [] x e (Term.var gs)
      ⟨_, _, _, h_e, h_compat⟩ hvar_gs
  -- Helper: `union [accum] [] = [accum]`. Goal: `union eps [accum]`.
  have hsub :
      SubEffRow (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
                (EffectRow.union eps [EffectLabel.accum]) := by
    intro op hmem
    have hop : op = EffectLabel.accum := by
      simpa [EffectRow.union] using hmem
    subst hop
    show EffectLabel.accum ∈ EffectRow.union eps [EffectLabel.accum]
    simp [EffectRow.union, List.mem_append]
    exact Classical.em _
  exact HasType.subEff _ _ _ _ _ _ _ _ h_aux hsub

end LaCaDiLE
