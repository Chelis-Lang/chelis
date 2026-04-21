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
  | Term.expand e _ =>
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

/-! ### Context filter helpers for weakening/exchange -/

/-- Appending a binding at the tail commutes with re-association. -/
@[simp] theorem append_singleton_append (G : LinearCtx) (y : String) (t : Option Typ)
    (ys : LinearCtx) :
    (G ++ [(y, t)]) ++ ys = G ++ ((y, t) :: ys) := by
  simp [List.append_assoc]

/-- Snoc equality: if `G1 ++ [a] = G2 ++ [b]` then `G1 = G2` and `a = b`. -/
theorem append_singleton_inj {α} (G1 G2 : List α) (a b : α)
    (h : G1 ++ [a] = G2 ++ [b]) : G1 = G2 ∧ a = b := by
  have := List.append_inj_right' h (by simp)
  have h1 := List.append_inj_left' h (by simp)
  refine ⟨h1, ?_⟩
  have : [a] = [b] := this
  exact List.head_eq_of_cons_eq this

/-! ### Freshness extraction helpers

Position-indexed weakening (`weakening_insert` below) is the single
outstanding proof obligation for `weakening_tail`, `exchange_tail`,
and `subst_preserves_typing`. We first extract the freshness
sub-conditions for each binder-containing term shape, then carry out
the full `HasType.rec` induction. -/

/-- `y` fresh in `Term.abs x t body` gives freshness in `body` and `y ≠ x`. -/
theorem freshInTerm_abs {y x : String} {t : Typ} {body : Term}
    (h : freshInTerm y (Term.abs x t body)) : y ≠ x ∧ freshInTerm y body := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, List.mem_filter, bne_iff_ne, ne_eq] at hf
  simp only [boundVars, List.mem_cons] at hb
  refine ⟨?_, ?_, ?_⟩
  · intro he; exact hb (Or.inl he)
  · intro hfb
    by_cases hxy : y = x
    · exact hb (Or.inl hxy)
    · exact hf ⟨hfb, fun he => hxy he⟩
  · intro hbb; exact hb (Or.inr hbb)

theorem freshInTerm_app {y : String} {e1 e2 : Term}
    (h : freshInTerm y (Term.app e1 e2)) : freshInTerm y e1 ∧ freshInTerm y e2 := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars, List.mem_append, not_or] at hf hb
  exact ⟨⟨hf.1, hb.1⟩, ⟨hf.2, hb.2⟩⟩

theorem freshInTerm_letBind {y x : String} {e1 e2 : Term}
    (h : freshInTerm y (Term.letBind x e1 e2)) :
    y ≠ x ∧ freshInTerm y e1 ∧ freshInTerm y e2 := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, List.mem_append, List.mem_filter, bne_iff_ne, ne_eq,
    not_or] at hf
  simp only [boundVars, List.mem_cons, List.mem_append, not_or] at hb
  obtain ⟨hf1, hf2⟩ := hf
  obtain ⟨hbne, hb1, hb2⟩ := hb
  refine ⟨hbne, ⟨hf1, hb1⟩, ?_, hb2⟩
  intro hfb
  by_cases hxy : y = x
  · exact hbne hxy
  · exact hf2 ⟨hfb, fun he => hxy he⟩

theorem freshInTerm_letpair {y a b : String} {e1 e2 : Term}
    (h : freshInTerm y (Term.letpair a b e1 e2)) :
    y ≠ a ∧ y ≠ b ∧ freshInTerm y e1 ∧ freshInTerm y e2 := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, List.mem_append, List.mem_filter, Bool.and_eq_true,
    bne_iff_ne, ne_eq, not_or] at hf
  simp only [boundVars, List.mem_cons, List.mem_append, not_or] at hb
  obtain ⟨hf1, hf2⟩ := hf
  obtain ⟨hane, hbne, hb1, hb2⟩ := hb
  refine ⟨hane, hbne, ⟨hf1, hb1⟩, ?_, hb2⟩
  intro hfb
  by_cases hya : y = a
  · exact hane hya
  · by_cases hyb : y = b
    · exact hbne hyb
    · exact hf2 ⟨hfb, ⟨fun he => hya he, fun he => hyb he⟩⟩

theorem freshInTerm_pair {y : String} {e1 e2 : Term}
    (h : freshInTerm y (Term.pair e1 e2)) : freshInTerm y e1 ∧ freshInTerm y e2 := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars, List.mem_append, not_or] at hf hb
  exact ⟨⟨hf.1, hb.1⟩, ⟨hf.2, hb.2⟩⟩

theorem freshInTerm_add {y : String} {e1 e2 : Term}
    (h : freshInTerm y (Term.add e1 e2)) : freshInTerm y e1 ∧ freshInTerm y e2 := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars, List.mem_append, not_or] at hf hb
  exact ⟨⟨hf.1, hb.1⟩, ⟨hf.2, hb.2⟩⟩

theorem freshInTerm_mul {y : String} {e1 e2 : Term}
    (h : freshInTerm y (Term.mul e1 e2)) : freshInTerm y e1 ∧ freshInTerm y e2 := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars, List.mem_append, not_or] at hf hb
  exact ⟨⟨hf.1, hb.1⟩, ⟨hf.2, hb.2⟩⟩

theorem freshInTerm_copy {y : String} {e : Term}
    (h : freshInTerm y (Term.copy e)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_fst {y : String} {e : Term}
    (h : freshInTerm y (Term.fst e)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_snd {y : String} {e : Term}
    (h : freshInTerm y (Term.snd e)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_sum {y : String} {e : Term} {d : Dim}
    (h : freshInTerm y (Term.sum e d)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_expand {y : String} {e : Term} {d : Dim}
    (h : freshInTerm y (Term.expand e d)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_uniformLike {y : String} {e : Term} {lo hi : Float}
    (h : freshInTerm y (Term.uniformLike e lo hi)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_perform {y : String} {op : EffectLabel} {e : Term}
    (h : freshInTerm y (Term.perform op e)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_var {y x : String}
    (h : freshInTerm y (Term.var x)) : y ≠ x := by
  rcases h with ⟨hf, _⟩
  simp only [freeVars, List.mem_singleton] at hf
  intro he; exact hf he

theorem freshInTerm_handle_body {y : String} {epsH : EffectRow} {body : Term}
    {cls : List (EffectLabel × String × String × Term)}
    (h : freshInTerm y (Term.handle epsH body cls)) : freshInTerm y body := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars, List.mem_append, not_or] at hf hb
  exact ⟨hf.1, hb.1⟩

/-- Per-clause freshness extraction from a `freshInTerm` on a handle term
    with a non-empty clause list. Gives freshness in the head clause body
    (plus inequality with the head's bound names) and freshness in the
    residual handle term. -/
theorem freshInTerm_clauses_cons {y : String} {epsH : EffectRow} {body : Term}
    {op : EffectLabel} {x k : String} {hb : Term}
    {rest : List (EffectLabel × String × String × Term)}
    (h : freshInTerm y (Term.handle epsH body ((op, x, k, hb) :: rest))) :
    y ≠ x ∧ y ≠ k ∧ freshInTerm y hb ∧
      freshInTerm y (Term.handle epsH body rest) := by
  unfold freshInTerm at h
  obtain ⟨hf, hbnd⟩ := h
  -- freeVars: body ++ ((freeVars hb).filter (!=x && !=k) ++ freeVarsClauses rest)
  have hne_x : y ≠ x := by
    intro he
    apply hbnd
    show y ∈ boundVars body ++ (x :: k :: boundVars hb ++ boundVarsClauses rest)
    apply List.mem_append_right
    rw [he]; exact List.mem_cons_self
  have hne_k : y ≠ k := by
    intro he
    apply hbnd
    show y ∈ boundVars body ++ (x :: k :: boundVars hb ++ boundVarsClauses rest)
    apply List.mem_append_right
    rw [he]
    exact List.mem_cons.mpr (Or.inr List.mem_cons_self)
  have hbnd_body : y ∉ boundVars body := by
    intro hin
    apply hbnd
    show y ∈ boundVars body ++ (x :: k :: boundVars hb ++ boundVarsClauses rest)
    exact List.mem_append_left _ hin
  have hbnd_hb : y ∉ boundVars hb := by
    intro hin
    apply hbnd
    show y ∈ boundVars body ++ (x :: k :: boundVars hb ++ boundVarsClauses rest)
    apply List.mem_append_right
    refine List.mem_cons.mpr (Or.inr (List.mem_cons.mpr (Or.inr ?_)))
    exact List.mem_append_left _ hin
  have hbnd_rest : y ∉ boundVarsClauses rest := by
    intro hin
    apply hbnd
    show y ∈ boundVars body ++ (x :: k :: boundVars hb ++ boundVarsClauses rest)
    apply List.mem_append_right
    refine List.mem_cons.mpr (Or.inr (List.mem_cons.mpr (Or.inr ?_)))
    exact List.mem_append_right _ hin
  have hf_body : y ∉ freeVars body := by
    intro hin
    apply hf
    show y ∈ freeVars body ++ ((freeVars hb).filter (fun z => z != x && z != k)
                                ++ freeVarsClauses rest)
    exact List.mem_append_left _ hin
  have hf_hb : y ∉ freeVars hb := by
    intro hin
    apply hf
    show y ∈ freeVars body ++ ((freeVars hb).filter (fun z => z != x && z != k)
                                ++ freeVarsClauses rest)
    apply List.mem_append_right
    apply List.mem_append_left
    rw [List.mem_filter]
    refine ⟨hin, ?_⟩
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq]
    exact ⟨fun he => hne_x he, fun he => hne_k he⟩
  have hf_rest : y ∉ freeVarsClauses rest := by
    intro hin
    apply hf
    show y ∈ freeVars body ++ ((freeVars hb).filter (fun z => z != x && z != k)
                                ++ freeVarsClauses rest)
    apply List.mem_append_right
    exact List.mem_append_right _ hin
  refine ⟨hne_x, hne_k, ⟨hf_hb, hbnd_hb⟩, ⟨?_, ?_⟩⟩
  · -- y ∉ freeVars (handle epsH body rest)
    intro hin
    show False
    have : y ∈ freeVars body ++ freeVarsClauses rest := hin
    rcases List.mem_append.mp this with h1 | h1
    · exact hf_body h1
    · exact hf_rest h1
  · intro hin
    show False
    have : y ∈ boundVars body ++ boundVarsClauses rest := hin
    rcases List.mem_append.mp this with h1 | h1
    · exact hbnd_body h1
    · exact hbnd_rest h1

theorem freshInTerm_handle_clause
    {y : String} {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (h : freshInTerm y (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    y ≠ x ∧ y ≠ k ∧ freshInTerm y hb := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases cl with ⟨op0, x0, k0, hb0⟩
      rcases freshInTerm_clauses_cons
          (op := op0) (x := x0) (k := k0) (hb := hb0) (rest := rest) h with
        ⟨hneqX0, hneqK0, hFreshHb0, hFreshRest⟩
      rcases List.mem_cons.mp hmem with hhead | htail
      · cases hhead
        exact ⟨hneqX0, hneqK0, hFreshHb0⟩
      · exact ih hFreshRest htail


/-! ### Append split analysis for the var case -/

/-- Shrinkage helper: if `h : HasType ... (Gpre ++ Gpost) e t eps (G2p ++ G2q)`
    and `y` is fresh in both `Gpre` and `Gpost`, then it's fresh in both
    `G2p` and `G2q`. -/
theorem weakening_insert_shrink_split
    {Delta : CapCtx} {Sigma : StoreTyp} {y : String}
    {Gpre Gpost G2p G2q : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma (Gpre ++ Gpost) e t eps (G2p ++ G2q))
    (hfp : y ∉ linearCtxDom Gpre) (hfq : y ∉ linearCtxDom Gpost) :
    y ∉ linearCtxDom G2p ∧ y ∉ linearCtxDom G2q := by
  have hshr := has_type_linear_shrinks h y
  have lem : ∀ {A B : LinearCtx} {z}, z ∈ linearCtxDom A → z ∈ linearCtxDom (A ++ B) := by
    intro A B z hz
    simp only [linearCtxDom, List.map_append, List.mem_append]
    exact Or.inl hz
  have lem2 : ∀ {A B : LinearCtx} {z}, z ∈ linearCtxDom B → z ∈ linearCtxDom (A ++ B) := by
    intro A B z hz
    simp only [linearCtxDom, List.map_append, List.mem_append]
    exact Or.inr hz
  have split : ∀ {A B : LinearCtx} {z}, z ∈ linearCtxDom (A ++ B) →
      z ∈ linearCtxDom A ∨ z ∈ linearCtxDom B := by
    intro A B z hz
    simp only [linearCtxDom, List.map_append, List.mem_append] at hz
    exact hz
  refine ⟨?_, ?_⟩
  · intro hy
    have hin := hshr (lem hy)
    rcases split hin with h1 | h1
    · exact hfp h1
    · exact hfq h1
  · intro hy
    have hin := hshr (lem2 hy)
    rcases split hin with h1 | h1
    · exact hfp h1
    · exact hfq h1

/-- If `A ++ B = C ++ D`, then either `A` is a prefix of `C` or vice versa. -/
theorem list_append_eq_split {α} :
    ∀ (A B C D : List α), A ++ B = C ++ D →
      (∃ m, C = A ++ m ∧ B = m ++ D) ∨ (∃ m, A = C ++ m ∧ D = m ++ B) := by
  intro A
  induction A with
  | nil =>
    intro B C D h
    simp at h
    exact Or.inl ⟨C, rfl, h⟩
  | cons a A ih =>
    intro B C D h
    cases C with
    | nil =>
      simp at h
      exact Or.inr ⟨a :: A, by simp, by simp [h]⟩
    | cons c C =>
      simp at h
      obtain ⟨hac, hrest⟩ := h
      rcases ih B C D hrest with ⟨m, hCm, hBm⟩ | ⟨m, hAm, hDm⟩
      · exact Or.inl ⟨m, by simp [hac, hCm], hBm⟩
      · exact Or.inr ⟨m, by simp [hac, hAm], hDm⟩

/-- Position-indexed weakening. For any split of the input context
    `Γ = Γ_pre ++ Γ_post`, inserting a fresh binding `(y, some t_y)`
    between the two halves yields a new derivation whose output context
    also has `(y, some t_y)` inserted at a matching position.

    Tombstone refactor (LinearCtx = List (String × Option Typ)):
    the motive and all case arms must thread `(y, some t_y)` through
    context positions. The abs/letBind/letpair cases no longer produce
    filter-based outputs — they strip a trailing `(x, slot)` instead.
    Sorry'd cases are marked for rework in the de Bruijn substitution
    wave. -/
theorem weakening_insert
    (Delta : CapCtx) (Sigma : StoreTyp) (y : String) (t_y : Typ)
    {Gamma Gamma' : LinearCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ Gamma_pre Gamma_post,
      Gamma = Gamma_pre ++ Gamma_post →
      y ∉ linearCtxDom Gamma_pre → y ∉ linearCtxDom Gamma_post →
      freshInTerm y e →
      ∃ Gamma'_pre Gamma'_post,
        Gamma' = Gamma'_pre ++ Gamma'_post ∧
        HasType Delta Sigma (Gamma_pre ++ [(y, some t_y)] ++ Gamma_post) e t eps
                (Gamma'_pre ++ [(y, some t_y)] ++ Gamma'_post) := by
  sorry
  -- BLOCKED: tombstone refactor changes the output shape of every
  -- binder case (abs/letBind/letpair strip a trailing slot instead of
  -- filtering). The full induction proof needs rework against the new
  -- HasType constructors. Deferred to de Bruijn substitution wave.


/-- Weakening: adding an unused binding at the tail of the linear
    context preserves typing. Derived as a corollary of
    `weakening_insert` with `Gamma_post = []`. -/
theorem weakening_tail
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (y : String) (t t' : Typ) (eps : EffectRow) (e : Term)
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (h_fresh : y ∉ linearCtxDom Gamma)
    (h_fresh_term : freshInTerm y e) :
    ∃ Gamma'_pre Gamma'_post : LinearCtx,
      Gamma' = Gamma'_pre ++ Gamma'_post ∧
      HasType Delta Sigma (Gamma ++ [(y, some t')]) e t eps
              (Gamma'_pre ++ [(y, some t')] ++ Gamma'_post) := by
  have hfp : y ∉ linearCtxDom ([] : LinearCtx) := by simp [linearCtxDom]
  have hsplit : Gamma = Gamma ++ ([] : LinearCtx) := by simp
  have hres := weakening_insert Delta Sigma y t' h Gamma [] hsplit h_fresh hfp h_fresh_term
  simpa using hres

-- NOTE: `exchange_tail` removed entirely. Its original statement
-- (rigid output context across an adjacent swap) is provably false
-- in this type system: `var` consumes the tail binding and the two
-- swap orderings produce different output contexts. No caller
-- relies on `exchange_tail`; substitution is closed via
-- `weakening_insert` directly.

/-! ## Main theorem -/

/-- Substitution preserves typing (tombstone semantics).

    Under the tombstone refactor, the input context carries
    `(x, some t1)` and the output context is whatever HasType produces
    (with `x`'s slot potentially tombstoned to `none`). The `Closed v`
    premise makes the naive capture-unaware `subst` sound. -/
theorem subst_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
    (x : String) (t1 t2 : Typ) (eps : EffectRow)
    (e v : Term)
    (_h_e : HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) e t2 eps Gamma2)
    (_h_v : HasType Delta Sigma Gamma1 v t1 [] Gamma1)
    (_h_closed : Closed v) :
    ∃ Gamma2' : LinearCtx,
      HasType Delta Sigma Gamma1 (subst e v x) t2 eps Gamma2' := by
  sorry -- BLOCKED on weakening_insert rework under tombstone semantics.
        -- The output shape is no longer a simple filter; it depends on
        -- how x's slot was consumed inside the derivation. Will be
        -- reworked in the de Bruijn substitution wave.

end LaCaDiLE
