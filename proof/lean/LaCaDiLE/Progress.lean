-- LaCaDiLE/Progress.lean — progress theorem (Phase 2 proof).
--
-- WS2.4 target: every well-typed closed term with an empty effect row
-- is either a value or can take a step.
--
-- Phase 1 caveat: `Operational.Step` encodes only head reductions. The
-- full small-step relation is the congruence closure of these head
-- rules via an evaluation context `E[·]` (opsem.tex E-Ctx), which is a
-- Phase 2 task. As a result, sub-cases of `progress` that rely on
-- stepping a strict sub-term (e.g. T-App when the function is not yet
-- a value) are left as `sorry` with a
--   -- TODO Phase 2: needs E-Ctx congruence closure
-- marker. The redex-at-top cases and the value cases close cleanly.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-! ## Progress -/

/-- Progress: a closed, effect-free, well-typed term under an arbitrary
    store typing is either a value or reducible.

    Phase 1 head-reduction limitation: cases whose reduction requires
    stepping a strict sub-term via a congruence/evaluation-context rule
    are left as `sorry` with a TODO marker. Those close automatically
    once `Operational.Step` is extended with an E-Ctx constructor
    (Phase 2). -/
theorem progress
    (sigma : Store) (Sigma : StoreTyp) (e : Term) (t : Typ)
    (h : HasType [] Sigma [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ := by
  -- Case analysis on the term syntax (not on the HasType derivation,
  -- which Lean's `cases` cannot directly eliminate because it is in a
  -- mutual block with `ClausesTyped` and carries dependent indices).
  -- For each term shape we either produce a value witness, fire a
  -- head-reduction, or leave a `sorry` with a TODO marker for the
  -- Phase 2 E-Ctx congruence closure.
  cases e with
  | var x =>
      -- T-Var requires a non-empty input context, contradicting `[]`.
      sorry  -- TODO Phase 2: var inversion under empty linear context
  | abs x tv body => exact Or.inl (IsValue.abs x tv body)
  | unit => exact Or.inl IsValue.unit
  | loc ell => exact Or.inl (IsValue.loc ell)
  | const v ds =>
      -- E-Const always fires with a fresh location.
      exact Or.inr ⟨storeExtend sigma (storeFreshLoc sigma) ⟨ds, v⟩,
                    Term.loc (storeFreshLoc sigma),
                    Step.tconst sigma v ds (storeFreshLoc sigma) rfl⟩
  | grad x tv tOut body =>
      -- E-Grad fires unconditionally on any literal `grad` term.
      exact Or.inr ⟨sigma, _, Step.tgrad sigma x tv tOut body⟩
  | vmap x tv body =>
      -- E-Vmap fires unconditionally on any literal `vmap` term.
      -- We need a dimension to lift along; any dimension works for a
      -- head reduction, so pick a literal 0.
      exact Or.inr ⟨sigma, _, Step.tvmap sigma x tv body (Dim.lit 0)⟩
  | pair e1 e2 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | app e1 e2 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | letBind x e1 e2 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | copy e1 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | letpair x y e1 e2 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | fst e1 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | snd e1 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | add e1 e2 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | mul e1 e2 =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | sum e1 i =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | expand e1 i k =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | uniformLike e1 lo hi =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | handle epsH body clauses =>
      sorry  -- TODO Phase 2: needs E-Ctx congruence closure
  | perform op e1 =>
      sorry  -- TODO Phase 2: effect-row inversion on empty output row

end LaCaDiLE
