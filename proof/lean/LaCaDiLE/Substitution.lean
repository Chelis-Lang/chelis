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

mutual

/-- Generalised no-op: if `x` is not free in `e`, substituting `v`
    for `x` is the identity. Proved by structural induction on `e`
    in a mutual block with `substClauses_notFree`. -/
theorem subst_notFree
    (e : Term) (v : Term) (x : String) (h : x ∉ freeVars e) :
    subst e v x = e := by
  match e with
  | Term.var y =>
      simp only [freeVars, List.mem_singleton] at h
      have hyx : y ≠ x := fun he => h he.symm
      simp [subst, hyx]
  | Term.abs y t body =>
      simp only [freeVars] at h
      by_cases hy : y = x
      · simp [subst, hy]
      · have hx_body : x ∉ freeVars body := by
          intro hx
          apply h
          rw [List.mem_filter]
          refine ⟨hx, ?_⟩
          simp only [bne_iff_ne, ne_eq]
          exact fun heq => hy heq.symm
        simp [subst, hy, subst_notFree body v x hx_body]
  | Term.app e1 e2 =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      simp [subst, subst_notFree e1 v x h1, subst_notFree e2 v x h2]
  | Term.letBind y e1 e2 =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      have ih1 := subst_notFree e1 v x h1
      by_cases hy : y = x
      · simp [subst, hy, ih1]
      · have hx_e2 : x ∉ freeVars e2 := by
          intro hx
          apply h2
          rw [List.mem_filter]
          refine ⟨hx, ?_⟩
          simp only [bne_iff_ne, ne_eq]
          exact fun heq => hy heq.symm
        have ih2 := subst_notFree e2 v x hx_e2
        simp [subst, hy, ih1, ih2]
  | Term.copy e =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.letpair a b e1 e2 =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      have ih1 := subst_notFree e1 v x h1
      by_cases ha : a = x
      · simp [subst, ha, ih1]
      · by_cases hb : b = x
        · simp [subst, hb, ih1]
        · have hx_e2 : x ∉ freeVars e2 := by
            intro hx
            apply h2
            rw [List.mem_filter]
            refine ⟨hx, ?_⟩
            simp only [Bool.and_eq_true, bne_iff_ne, ne_eq]
            exact ⟨fun heq => ha heq.symm, fun heq => hb heq.symm⟩
          have ih2 := subst_notFree e2 v x hx_e2
          have hcond : ¬ (a = x ∨ b = x) := fun hd => hd.elim ha hb
          simp [subst, hcond, ih1, ih2]
  | Term.pair e1 e2 =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      simp [subst, subst_notFree e1 v x h1, subst_notFree e2 v x h2]
  | Term.fst e =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.snd e =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.unit => simp [subst]
  | Term.const _ _ => simp [subst]
  | Term.add e1 e2 =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      simp [subst, subst_notFree e1 v x h1, subst_notFree e2 v x h2]
  | Term.mul e1 e2 =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      simp [subst, subst_notFree e1 v x h1, subst_notFree e2 v x h2]
  | Term.sum e _ =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.expand e _ _ =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.uniformLike e _ _ =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.grad y t tOut body =>
      simp only [freeVars] at h
      by_cases hy : y = x
      · simp [subst, hy]
      · have hx_body : x ∉ freeVars body := by
          intro hx
          apply h
          rw [List.mem_filter]
          refine ⟨hx, ?_⟩
          simp only [bne_iff_ne, ne_eq]
          exact fun heq => hy heq.symm
        simp [subst, hy, subst_notFree body v x hx_body]
  | Term.vmap y t body =>
      simp only [freeVars] at h
      by_cases hy : y = x
      · simp [subst, hy]
      · have hx_body : x ∉ freeVars body := by
          intro hx
          apply h
          rw [List.mem_filter]
          refine ⟨hx, ?_⟩
          simp only [bne_iff_ne, ne_eq]
          exact fun heq => hy heq.symm
        simp [subst, hy, subst_notFree body v x hx_body]
  | Term.handle epsH body clauses =>
      simp only [freeVars, List.mem_append, not_or] at h
      obtain ⟨h1, h2⟩ := h
      simp [subst, subst_notFree body v x h1,
            substClauses_notFree clauses v x h2]
  | Term.perform _ e =>
      simp only [freeVars] at h
      simp [subst, subst_notFree e v x h]
  | Term.loc _ => simp [subst]

theorem substClauses_notFree
    (cls : List (EffectLabel × String × String × Term))
    (v : Term) (x : String) (h : x ∉ freeVarsClauses cls) :
    substClauses cls v x = cls := by
  match cls with
  | [] => simp [substClauses]
  | (op, y, k, hb) :: rest =>
      simp only [substClauses, freeVarsClauses, List.mem_append, not_or]
        at h ⊢
      obtain ⟨h_hb, h_rest⟩ := h
      have ih_rest := substClauses_notFree rest v x h_rest
      by_cases hy : y = x
      · rw [ih_rest]; simp [hy]
      · by_cases hk : k = x
        · rw [ih_rest]; simp [hk]
        · have hxne_y : ¬ (x = y) := fun he => hy he.symm
          have hxne_k : ¬ (x = k) := fun he => hk he.symm
          have h_notfree_hb : x ∉ freeVars hb := by
            intro hx
            apply h_hb
            rw [List.mem_filter]
            refine ⟨hx, ?_⟩
            simp [hxne_y, hxne_k]
          have hb_eq := subst_notFree hb v x h_notfree_hb
          have hcond : ¬ (y = x ∨ k = x) := fun hd => hd.elim hy hk
          simp [hcond, hb_eq, ih_rest]

end

/-- Substitution on a closed term is a no-op: a closed term has no
    free variables, so `subst e v x` equals `e` regardless of `v`/`x`. -/
theorem subst_closed
    (e v : Term) (x : String) (h : Closed e) :
    subst e v x = e := by
  apply subst_notFree
  unfold Closed at h
  rw [h]
  simp

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
  sorry -- TODO Wave 2: induction on HasType derivation.
        -- Requires mutual ClausesTyped weakening helper for the `handle` case.
        -- Every constructor that binds a variable at the tail (var, abs,
        -- letBind, letpair, tgrad, tvmap) needs an associativity rewrite
        -- to re-associate the new tail binding with the existing context
        -- shape, and the `var` case specifically needs exchange_tail to
        -- push `(y, t')` past the consumed `(x, t)` binding. This three-way
        -- mutual dependence (weakening ↔ exchange ↔ ClausesTyped weakening)
        -- is the hard core of Wave 1; scheduling it as its own Wave 2 task.

/-- Exchange: swapping two adjacent unrelated bindings in the linear
    context preserves typing. Used when a substitution introduces a
    fresh binding mid-context and the surrounding derivation needs to
    thread around it. -/
theorem exchange_tail
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (x y : String) (t t1 t2 : Typ) (eps : EffectRow) (e : Term)
    (_h : HasType Delta Sigma (Gamma ++ [(x, t1), (y, t2)]) e
                  t eps Gamma')
    (_h_ne : x ≠ y) :
    HasType Delta Sigma (Gamma ++ [(y, t2), (x, t1)]) e
            t eps Gamma' := by
  sorry -- TODO Wave 2: induction on HasType + mutual ClausesTyped exchange.
        -- Symmetric to weakening_tail: each binder case needs to show the
        -- extended context (Gamma ++ [(y,t2),(x,t1)] ++ [(bound, τ)]) is
        -- still well-formed, and the var case needs to distinguish which
        -- of x, y is consumed. The permutation invariant on LinearCtx is
        -- what actually gets proved here; weakening_tail is the degenerate
        -- case where the second binding is fresh.

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
  sorry -- TODO Wave 2: main substitution lemma.
        -- Induction on _h_e (the HasType derivation of `e`).
        -- Key cases:
        --   * var: split y = x (return _h_v, using subst_closed on the result
        --     context manipulation) vs y ≠ x (apply weakening_tail to peel
        --     the unused (x,t1) binding and rebuild via .var).
        --   * abs/letBind/letpair/tgrad/tvmap: IH under extended context,
        --     then push the (x,t1) binding past the new binder via
        --     exchange_tail, then apply the IH, then rebuild the binder.
        --   * app/tadd/tmul/tpair: the left-to-right threaded contexts mean
        --     (x,t1) lives at the tail of Γ₁ going into e₁; the sub-derivation
        --     for e₁ may or may not consume it. Need a case split on whether
        --     x survives into Γ₂; each branch applies IH + subst_notFree /
        --     subst_closed on the other side.
        --   * handle: companion substClauses_preserves_typing in a mutual
        --     block; each clause extends Γ₂ with its own (arg, cont) tail so
        --     the (x,t1) binding is buried under two fresh bindings and
        --     needs exchange_tail applied twice.
        -- Blocked on weakening_tail + exchange_tail; scheduled as a single
        -- Wave 2 mutual-induction push.

end LaCaDiLE
