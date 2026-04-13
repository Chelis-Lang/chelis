-- LaCaDiLE/Substitution.lean — substitution lemma (Phase 2 WS2.1).
--
-- Statement: if `e` is well-typed in `Γ₁ ++ [(x, t₁)]` and `v` is a
-- closed value of type `t₁`, then `subst e v x` is well-typed in `Γ₁`
-- with `x` removed from the output context.
--
-- Wave 1 design decision: naive named substitution + `Closed v` side
-- condition. Preservation only ever substitutes values, and values
-- reachable under reduction are closed (no free variables dangling
-- into other bindings under the linear discipline), so `Closed v` is
-- trivially dischargeable at every call site.
--
-- Helper lemmas below are scaffolded as `sorry`-stubs and filled
-- bottom-up via prove.py / manual tactic work.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-! ## Helper lemmas for substitution -/

/-- Substitution on a closed term is a no-op: a closed term has no
    free variables, so `subst e v x` equals `e` regardless of `v`/`x`. -/
theorem subst_closed
    (e v : Term) (x : String) (h : Closed e) :
    subst e v x = e := by
  sorry

/-- Values are closed under substitution: substituting into a value
    produces a value of the same shape. Used in the `E-Beta` case of
    Preservation, where the reduced term is a substitution instance
    that must still be a value if the original body was. -/
theorem subst_value
    (w v : Term) (x : String) (h : IsValue w) :
    IsValue (subst w v x) := by
  induction h with
  | loc ell => unfold subst; exact IsValue.loc ell
  | abs y t e =>
      unfold subst
      by_cases hxy : y = x
      · simp [hxy]; exact IsValue.abs x t e
      · simp [hxy]; exact IsValue.abs y t _
  | pair v1 v2 h1 h2 ih1 ih2 =>
      unfold subst
      exact IsValue.pair _ _ ih1 ih2
  | unit => unfold subst; exact IsValue.unit

/-- Weakening: adding an unused binding at the tail of the linear
    context preserves typing. "Unused" means `y` does not appear in
    the output context either. Phase 2 Wave 1 helper for substitution. -/
theorem weakening_tail
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (y : String) (t t' : Typ) (eps : EffectRow) (e : Term)
    (_h : HasType Delta Sigma Gamma e t eps Gamma')
    (_h_fresh : y ∉ linearCtxDom Gamma) :
    HasType Delta Sigma (Gamma ++ [(y, t')]) e t eps
            (Gamma' ++ [(y, t')]) := by
  sorry

/-- Exchange: swapping two adjacent unrelated bindings in the linear
    context preserves typing. Used when a substitution introduces a
    fresh binding mid-context and the surrounding derivation needs to
    thread around it. -/
theorem exchange_tail
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (x y : String) (t1 t2 : Typ) (eps : EffectRow) (e : Term)
    (_h : HasType Delta Sigma (Gamma ++ [(x, t1), (y, t2)]) e
                  (Typ.tensor []) eps Gamma')
    (_h_ne : x ≠ y) :
    HasType Delta Sigma (Gamma ++ [(y, t2), (x, t1)]) e
            (Typ.tensor []) eps Gamma' := by
  sorry

/-! ## Main theorem -/

/-- Substitution preserves typing. The `Closed v` premise makes the
    naive capture-unaware `subst` sound: since `v` has no free
    variables, no binder inside `e` can capture anything from `v`.
    Preservation discharges `Closed v` trivially because every term
    substituted under reduction is a closed value. -/
theorem subst_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
    (x : String) (t1 t2 : Typ) (eps : EffectRow)
    (e v : Term)
    (_h_e : HasType Delta Sigma (Gamma1 ++ [(x, t1)]) e t2 eps Gamma2)
    (_h_v : HasType Delta Sigma Gamma1 v t1 [] Gamma1)
    (_h_closed : Closed v) :
    HasType Delta Sigma Gamma1 (subst e v x) t2 eps
            (Gamma2.filter (fun p => p.1 ≠ x)) := by
  sorry

end LaCaDiLE
