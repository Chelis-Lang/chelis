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

/-! ### Context filter helpers for weakening/exchange -/

/-- Appending a binding at the tail commutes with re-association. -/
@[simp] theorem append_singleton_append (G : LinearCtx) (y : String) (t : Typ)
    (ys : LinearCtx) :
    (G ++ [(y, t)]) ++ ys = G ++ ((y, t) :: ys) := by
  simp [List.append_assoc]

/-- Filter commutes with append on linear contexts. -/
theorem linearCtx_filter_append (G1 G2 : LinearCtx) (p : (String × Typ) → Bool) :
    (G1 ++ G2).filter p = G1.filter p ++ G2.filter p := by
  exact List.filter_append G1 G2

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

theorem freshInTerm_sum {y : String} {e : Term} {i : Nat}
    (h : freshInTerm y (Term.sum e i)) : freshInTerm y e := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, boundVars] at hf hb
  exact ⟨hf, hb⟩

theorem freshInTerm_expand {y : String} {e : Term} {i k : Nat}
    (h : freshInTerm y (Term.expand e i k)) : freshInTerm y e := by
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
    `Γ = Γ_pre ++ Γ_post`, inserting a fresh binding `(y, t_y)` between
    the two halves yields a new derivation whose output context also
    has `(y, t_y)` inserted at a matching position. -/
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
        HasType Delta Sigma (Gamma_pre ++ [(y, t_y)] ++ Gamma_post) e t eps
                (Gamma'_pre ++ [(y, t_y)] ++ Gamma'_post) := by
  refine HasType.rec
    (motive_1 := fun Delta' Sigma' Gamma e t eps Gamma' _ =>
      ∀ Gamma_pre Gamma_post,
        Gamma = Gamma_pre ++ Gamma_post →
        y ∉ linearCtxDom Gamma_pre → y ∉ linearCtxDom Gamma_post →
        freshInTerm y e →
        ∃ Gamma'_pre Gamma'_post,
          Gamma' = Gamma'_pre ++ Gamma'_post ∧
          HasType Delta' Sigma' (Gamma_pre ++ [(y, t_y)] ++ Gamma_post) e t eps
                  (Gamma'_pre ++ [(y, t_y)] ++ Gamma'_post))
    (motive_2 := fun Delta' Sigma' Gamma2 Gamma3 t epsR cls _ =>
      ∀ Gamma2_pre Gamma2_post,
        Gamma2 = Gamma2_pre ++ Gamma2_post →
        y ∉ linearCtxDom Gamma2_pre → y ∉ linearCtxDom Gamma2_post →
        (∀ cl ∈ cls, freshInTerm y cl.2.2.2) →
        ∃ Gamma3_pre Gamma3_post,
          Gamma3 = Gamma3_pre ++ Gamma3_post ∧
          ClausesTyped Delta' Sigma'
            (Gamma2_pre ++ [(y, t_y)] ++ Gamma2_post)
            (Gamma3_pre ++ [(y, t_y)] ++ Gamma3_post) t epsR cls)
    ?var ?unit ?abs ?app ?letBind ?copy ?letpair ?tpair ?fst ?snd ?const
    ?tadd ?tmul ?tsum ?texpand ?uniformLike ?perform ?handle ?tgrad ?tvmap
    ?loc ?subEff ?nil ?cons h
  case var =>
    intro Delta' Sigma' Gpre Gpost x t
    intro Gm_pre Gm_post hsplit _hfp _hfq hft
    have hyx : y ≠ x := freshInTerm_var hft
    -- Gamma = Gpre ++ [(x,t)] ++ Gpost = Gm_pre ++ Gm_post.
    have h1 : (Gpre ++ [(x, t)]) ++ Gpost = Gm_pre ++ Gm_post := by
      -- hsplit : Gpre ++ [(x, t)] ++ Gpost = Gm_pre ++ Gm_post (up to assoc)
      simpa [List.append_assoc] using hsplit
    rcases list_append_eq_split (Gpre ++ [(x, t)]) Gpost Gm_pre Gm_post h1 with
      ⟨m, hCm, hBm⟩ | ⟨m, hAm, hDm⟩
    · -- Case A: (x,t) lives in Gm_pre (to the left of the split).
      -- Gm_pre = Gpre ++ [(x,t)] ++ m, Gpost = m ++ Gm_post.
      -- Output pre/post = (Gpre ++ m, Gm_post).
      refine ⟨Gpre ++ m, Gm_post, ?_, ?_⟩
      · -- Γ' = Gpre ++ Gpost = Gpre ++ m ++ Gm_post
        rw [hBm]; simp [List.append_assoc]
      · -- Build new var derivation with Γpre' = Gpre, Γpost' = m ++ [(y,t_y)] ++ Gm_post
        have hvar := HasType.var Delta' Sigma' Gpre (m ++ [(y, t_y)] ++ Gm_post) x t
        -- hvar input: Gpre ++ [(x,t)] ++ (m ++ [(y,t_y)] ++ Gm_post)
        -- Our goal input: Gm_pre ++ [(y,t_y)] ++ Gm_post
        -- These are equal because Gm_pre = Gpre ++ [(x,t)] ++ m.
        have hin : Gm_pre ++ [(y, t_y)] ++ Gm_post =
            Gpre ++ [(x, t)] ++ (m ++ [(y, t_y)] ++ Gm_post) := by
          rw [hCm]; simp [List.append_assoc]
        have hout : Gpre ++ m ++ [(y, t_y)] ++ Gm_post =
            Gpre ++ (m ++ [(y, t_y)] ++ Gm_post) := by
          simp [List.append_assoc]
        rw [hin, hout]
        exact hvar
    · -- Case B: Gpre ++ [(x,t)] = Gm_pre ++ m, Gm_post = m ++ Gpost.
      -- Split (Gpre, [(x,t)], Gm_pre, m) further.
      rcases list_append_eq_split Gpre [(x, t)] Gm_pre m hAm with
        ⟨m2, hGm_pre_eq, h_xt_eq⟩ | ⟨m2, hGpre_eq, hm_eq⟩
      · -- m2 is a prefix of [(x,t)]. Since [(x,t)] has length 1,
        -- either m2 = [] (then m = [(x,t)]) or m2 = [(x,t)] (then m = []).
        cases m2 with
        | nil =>
          -- Gm_pre = Gpre, m = [(x,t)]. So Gm_post = [(x,t)] ++ Gpost.
          simp only [List.append_nil] at hGm_pre_eq
          simp only [List.nil_append] at h_xt_eq
          -- hGm_pre_eq : Gm_pre = Gpre (Gm_pre was the LHS)
          -- Use rw to avoid direction issues in subst.
          rw [hGm_pre_eq] at *
          -- h_xt_eq : [(x,t)] = m, i.e. m = [(x,t)] (flipped).
          -- Goal pre/post: (Gpre, Gpost). Wait — the original Γ' = Gpre ++ Gpost.
          refine ⟨Gpre, Gpost, rfl, ?_⟩
          have hvar := HasType.var Delta' Sigma' (Gpre ++ [(y, t_y)]) Gpost x t
          -- hvar input: (Gpre ++ [(y,t_y)]) ++ [(x,t)] ++ Gpost
          --          = Gpre ++ [(y,t_y)] ++ [(x,t)] ++ Gpost
          -- hvar output: (Gpre ++ [(y,t_y)]) ++ Gpost = Gpre ++ [(y,t_y)] ++ Gpost
          -- Goal input: Gm_pre ++ [(y,t_y)] ++ Gm_post = Gpre ++ [(y,t_y)] ++ Gm_post
          -- Goal output: Gpre ++ [(y,t_y)] ++ Gpost
          have hm_val : m = [(x, t)] := h_xt_eq.symm
          have hGm_post : Gm_post = [(x, t)] ++ Gpost := by rw [hDm, hm_val]
          have hin : Gpre ++ [(y, t_y)] ++ Gm_post =
              Gpre ++ [(y, t_y)] ++ [(x, t)] ++ Gpost := by
            rw [hGm_post]; simp [List.append_assoc]
          have hin2 : Gpre ++ [(y, t_y)] ++ [(x, t)] ++ Gpost =
              Gpre ++ [(y, t_y)] ++ ([(x, t)] ++ Gpost) := by
            simp [List.append_assoc]
          rw [hin]
          -- We want goal = HasType ... (Gpre ++ [(y,t_y)] ++ [(x,t)] ++ Gpost) (var x) t []
          --                          (Gpre ++ [(y,t_y)] ++ Gpost)
          -- hvar has input: (Gpre ++ [(y,t_y)]) ++ [(x,t)] ++ Gpost
          --                = Gpre ++ [(y,t_y)] ++ [(x,t)] ++ Gpost (by assoc) ✓
          -- output: (Gpre ++ [(y,t_y)]) ++ Gpost = Gpre ++ [(y,t_y)] ++ Gpost ✓
          have hvar_rewrite : HasType Delta' Sigma'
              (Gpre ++ [(y, t_y)] ++ [(x, t)] ++ Gpost) (Term.var x) t []
              (Gpre ++ [(y, t_y)] ++ Gpost) := by
            have := hvar
            simp only [List.append_assoc] at this ⊢
            exact this
          exact hvar_rewrite
        | cons hd rest =>
          -- h_xt_eq : (hd :: rest) ++ m = [(x, t)].
          -- hd :: (rest ++ m) = [(x, t)] forces hd = (x, t),
          -- rest = [], m = []. So m2 = [(x, t)] which matches the
          -- "x is the tail" shape. Close by destructuring.
          simp only [List.cons_append] at h_xt_eq
          -- h_xt_eq : hd :: (rest ++ m) = [(x, t)]
          rcases List.cons_eq_cons.mp h_xt_eq with ⟨h_hd, h_rest_m⟩
          -- h_hd : hd = (x, t), h_rest_m : rest ++ m = []
          rcases List.append_eq_nil_iff.mp h_rest_m.symm with ⟨h_rest, h_m⟩
          subst h_rest
          subst h_m
          subst h_hd
          -- Now: hGm_pre_eq : Gm_pre = Gpre ++ [(x, t)], hDm : Gm_post = [] ++ Gpost = Gpost
          simp only [List.append_nil, List.nil_append] at hGm_pre_eq hDm
          rw [hGm_pre_eq, hDm] at *
          -- Γ' = Gpre ++ Gpost. Choose Γ'_pre = Gpre, Γ'_post = Gpost.
          refine ⟨Gpre, Gpost, rfl, ?_⟩
          -- Build var at (Gpre ++ [(x, t)]) ++ [(y, t_y)] ++ Gpost.
          have hvar := HasType.var Delta' Sigma' Gpre ([(y, t_y)] ++ Gpost) x t
          -- hvar input: Gpre ++ [(x, t)] ++ ([(y, t_y)] ++ Gpost)
          --           = Gpre ++ [(x, t)] ++ [(y, t_y)] ++ Gpost  (assoc)
          -- hvar output: Gpre ++ ([(y, t_y)] ++ Gpost)
          --            = Gpre ++ [(y, t_y)] ++ Gpost  (assoc)
          have hin : (Gpre ++ [(x, t)]) ++ [(y, t_y)] ++ Gpost =
              Gpre ++ [(x, t)] ++ ([(y, t_y)] ++ Gpost) := by
            simp [List.append_assoc]
          have hout : (Gpre ++ [(y, t_y)]) ++ Gpost =
              Gpre ++ ([(y, t_y)] ++ Gpost) := by
            simp [List.append_assoc]
          rw [hin]
          -- wait — the goal input may be in a different shape. Let me
          -- just use `simpa` to clean up.
          simpa [List.append_assoc] using hvar
      · -- m = m2 ++ [(x,t)], Gpre = Gm_pre ++ m2.
        -- Gm_post = m ++ Gpost = m2 ++ [(x,t)] ++ Gpost.
        -- Γ' = Gpre ++ Gpost = Gm_pre ++ m2 ++ Gpost.
        -- Choose Γ'_pre = Gm_pre, Γ'_post = m2 ++ Gpost.
        refine ⟨Gm_pre, m2 ++ Gpost, ?_, ?_⟩
        · rw [hGpre_eq]; simp [List.append_assoc]
        · have hvar := HasType.var Delta' Sigma' (Gm_pre ++ [(y, t_y)] ++ m2) Gpost x t
          have hin : Gm_pre ++ [(y, t_y)] ++ Gm_post =
              Gm_pre ++ [(y, t_y)] ++ m2 ++ [(x, t)] ++ Gpost := by
            rw [hDm, hm_eq]; simp [List.append_assoc]
          have hout : Gm_pre ++ [(y, t_y)] ++ (m2 ++ Gpost) =
              Gm_pre ++ [(y, t_y)] ++ m2 ++ Gpost := by simp [List.append_assoc]
          rw [hin, hout]
          have hvar_rewrite : HasType Delta' Sigma'
              (Gm_pre ++ [(y, t_y)] ++ m2 ++ [(x, t)] ++ Gpost) (Term.var x) t []
              (Gm_pre ++ [(y, t_y)] ++ m2 ++ Gpost) := by
            have := hvar
            simp only [List.append_assoc] at this ⊢
            exact this
          exact hvar_rewrite
  case unit =>
    intro Delta' Sigma' Gamma
    intro Gm_pre Gm_post hsplit _ _ _
    refine ⟨Gm_pre, Gm_post, hsplit, ?_⟩
    exact HasType.unit Delta' Sigma' _
  case abs =>
    intro Delta' Sigma' G1 G2 x t1 t2 eps' e' _hb ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hyx, hfe⟩ := freshInTerm_abs hft
    -- Sub-derivation at G1 ++ [(x,t1)] = Gm_pre ++ (Gm_post ++ [(x,t1)])
    have hsplit2 : G1 ++ [(x, t1)] = Gm_pre ++ (Gm_post ++ [(x, t1)]) := by
      rw [hsplit]; simp [List.append_assoc]
    have hfq2 : y ∉ linearCtxDom (Gm_post ++ [(x, t1)]) := by
      simp [linearCtxDom, List.map_append] at *
      exact ⟨hfq, fun he => hyx he⟩
    obtain ⟨G2_pre, G2_post, hG2eq, hinner⟩ :=
      ih Gm_pre (Gm_post ++ [(x, t1)]) hsplit2 hfp hfq2 hfe
    -- hinner : HasType ... (Gm_pre ++ [(y,t_y)] ++ (Gm_post ++ [(x,t1)])) e' t2 eps'
    --                     (G2_pre ++ [(y,t_y)] ++ G2_post)
    -- Rewrite input to ((Gm_pre ++ [(y,t_y)] ++ Gm_post) ++ [(x,t1)])
    have e1 : Gm_pre ++ [(y, t_y)] ++ (Gm_post ++ [(x, t1)]) =
        (Gm_pre ++ [(y, t_y)] ++ Gm_post) ++ [(x, t1)] := by simp [List.append_assoc]
    rw [e1] at hinner
    have habs := HasType.abs Delta' Sigma' (Gm_pre ++ [(y, t_y)] ++ Gm_post) (G2_pre ++ [(y, t_y)] ++ G2_post)
      x t1 t2 eps' e' hinner
    -- habs output is ((G2_pre ++ [(y,t_y)] ++ G2_post).filter (fun p => p.1 ≠ x))
    -- The original's output is (G2.filter (fun p => p.1 ≠ x))
    -- with G2 = G2_pre ++ [(y,t_y)] ++ G2_post... wait, no.
    -- hG2eq : G2 = G2_pre ++ G2_post. So original output = (G2_pre ++ G2_post).filter (...)
    -- We want: Γ'_pre ++ [(y,t_y)] ++ Γ'_post = original output
    --        = (G2_pre ++ G2_post).filter (p.1 ≠ x) = G2_pre.filter ++ G2_post.filter
    -- The new habs output: ((G2_pre ++ [(y,t_y)] ++ G2_post).filter (p.1 ≠ x))
    --                   = G2_pre.filter ++ [(y,t_y)].filter ++ G2_post.filter
    -- Since y ≠ x, [(y,t_y)].filter (p.1 ≠ x) = [(y,t_y)]
    refine ⟨G2_pre.filter (fun p => p.1 ≠ x), G2_post.filter (fun p => p.1 ≠ x), ?_, ?_⟩
    · rw [hG2eq, List.filter_append]
    · have hfilter_ins :
        ((G2_pre ++ [(y, t_y)] ++ G2_post).filter (fun p => p.1 ≠ x)) =
          G2_pre.filter (fun p => p.1 ≠ x) ++ [(y, t_y)] ++
            G2_post.filter (fun p => p.1 ≠ x) := by
        rw [List.filter_append, List.filter_append]
        have : ([(y, t_y)] : LinearCtx).filter (fun p => p.1 ≠ x) = [(y, t_y)] := by
          simp [List.filter, hyx]
        rw [this]
      rw [hfilter_ins] at habs
      exact habs
  case app =>
    intro Delta' Sigma' G1 G2 G3 e1 e2 t1 t2 eps' eps1 eps2 h1 _h2 ih1 ih2
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hf1, hf2⟩ := freshInTerm_app hft
    obtain ⟨G2p, G2q, hG2, h1'⟩ := ih1 Gm_pre Gm_post hsplit hfp hfq hf1
    have h1s := by
      have := h1
      rw [hsplit, hG2] at this
      exact this
    obtain ⟨hfp2, hfq2⟩ := weakening_insert_shrink_split h1s hfp hfq
    subst hG2
    obtain ⟨G3p, G3q, hG3, h2'⟩ := ih2 G2p G2q rfl hfp2 hfq2 hf2
    refine ⟨G3p, G3q, hG3, ?_⟩
    exact HasType.app Delta' Sigma' _ _ _ e1 e2 t1 t2 eps' eps1 eps2 h1' h2'
  case letBind =>
    intro Delta' Sigma' G1 G2 G3 x e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hyx, hf1, hf2⟩ := freshInTerm_letBind hft
    obtain ⟨G2p, G2q, hG2, h1'⟩ := ih1 Gm_pre Gm_post hsplit hfp hfq hf1
    have h1s := by
      have := _h1
      rw [hsplit, hG2] at this
      exact this
    obtain ⟨hfp2, hfq2a⟩ := weakening_insert_shrink_split h1s hfp hfq
    subst hG2
    have hfq2 : y ∉ linearCtxDom (G2q ++ [(x, t1)]) := by
      intro hh
      simp only [linearCtxDom, List.map_append, List.map_cons, List.map_nil,
        List.mem_append, List.mem_singleton, List.mem_cons, List.not_mem_nil,
        or_false] at hh
      rcases hh with h1 | h1
      · exact hfq2a h1
      · exact hyx h1
    have hsplit2 : G2p ++ G2q ++ [(x, t1)] = G2p ++ (G2q ++ [(x, t1)]) := by
      simp [List.append_assoc]
    obtain ⟨G3p, G3q, hG3, h2'⟩ := ih2 G2p (G2q ++ [(x, t1)]) hsplit2 hfp2 hfq2 hf2
    have e1eq : G2p ++ [(y, t_y)] ++ (G2q ++ [(x, t1)]) =
        (G2p ++ [(y, t_y)] ++ G2q) ++ [(x, t1)] := by simp [List.append_assoc]
    rw [e1eq] at h2'
    have hlet := HasType.letBind Delta' Sigma' (Gm_pre ++ [(y, t_y)] ++ Gm_post)
      (G2p ++ [(y, t_y)] ++ G2q) (G3p ++ [(y, t_y)] ++ G3q)
      x e1 e2 t1 t2 eps1 eps2 h1' h2'
    refine ⟨G3p.filter (fun p => p.1 ≠ x), G3q.filter (fun p => p.1 ≠ x), ?_, ?_⟩
    · rw [hG3, List.filter_append]
    · have hfilter_ins :
        ((G3p ++ [(y, t_y)] ++ G3q).filter (fun p => p.1 ≠ x)) =
          G3p.filter (fun p => p.1 ≠ x) ++ [(y, t_y)] ++
            G3q.filter (fun p => p.1 ≠ x) := by
        rw [List.filter_append, List.filter_append]
        have : ([(y, t_y)] : LinearCtx).filter (fun p => p.1 ≠ x) = [(y, t_y)] := by
          simp [List.filter, hyx]
        rw [this]
      rw [hfilter_ins] at hlet
      exact hlet
  case copy =>
    intro Delta' Sigma' G1 G2 e' t' eps' _h ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_copy hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.copy Delta' Sigma' _ _ e' t' eps' h'
  case letpair =>
    intro Delta' Sigma' G1 G2 G3 x yy e1 e2 t1 t2 t' eps1 eps2 _h1 _h2 ih1 ih2
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hya, hyb, hf1, hf2⟩ := freshInTerm_letpair hft
    obtain ⟨G2p, G2q, hG2, h1'⟩ := ih1 Gm_pre Gm_post hsplit hfp hfq hf1
    have h1s := by
      have := _h1
      rw [hsplit, hG2] at this
      exact this
    obtain ⟨hfp2, hfq2a⟩ := weakening_insert_shrink_split h1s hfp hfq
    subst hG2
    have hfq2 : y ∉ linearCtxDom (G2q ++ [(x, t1), (yy, t2)]) := by
      intro hh
      simp only [linearCtxDom, List.map_append, List.map_cons, List.map_nil,
        List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hh
      rcases hh with h1 | h1 | h1
      · exact hfq2a h1
      · exact hya h1
      · exact hyb h1
    have hsplit2 : G2p ++ G2q ++ [(x, t1), (yy, t2)] =
        G2p ++ (G2q ++ [(x, t1), (yy, t2)]) := by simp [List.append_assoc]
    obtain ⟨G3p, G3q, hG3, h2'⟩ :=
      ih2 G2p (G2q ++ [(x, t1), (yy, t2)]) hsplit2 hfp2 hfq2 hf2
    have e1eq : G2p ++ [(y, t_y)] ++ (G2q ++ [(x, t1), (yy, t2)]) =
        (G2p ++ [(y, t_y)] ++ G2q) ++ [(x, t1), (yy, t2)] := by
      simp [List.append_assoc]
    rw [e1eq] at h2'
    have hlp := HasType.letpair Delta' Sigma' (Gm_pre ++ [(y, t_y)] ++ Gm_post)
      (G2p ++ [(y, t_y)] ++ G2q) (G3p ++ [(y, t_y)] ++ G3q)
      x yy e1 e2 t1 t2 t' eps1 eps2 h1' h2'
    refine ⟨G3p.filter (fun p => p.1 ≠ x ∧ p.1 ≠ yy),
            G3q.filter (fun p => p.1 ≠ x ∧ p.1 ≠ yy), ?_, ?_⟩
    · rw [hG3, List.filter_append]
    · have hyxne : (y, t_y).1 ≠ x := hya
      have hyyne : (y, t_y).1 ≠ yy := hyb
      have hfilter_ins :
        ((G3p ++ [(y, t_y)] ++ G3q).filter (fun p => p.1 ≠ x ∧ p.1 ≠ yy)) =
          G3p.filter (fun p => p.1 ≠ x ∧ p.1 ≠ yy) ++ [(y, t_y)] ++
            G3q.filter (fun p => p.1 ≠ x ∧ p.1 ≠ yy) := by
        rw [List.filter_append, List.filter_append]
        have : ([(y, t_y)] : LinearCtx).filter (fun p => p.1 ≠ x ∧ p.1 ≠ yy) = [(y, t_y)] := by
          simp [List.filter, hyxne, hyyne]
        rw [this]
      rw [hfilter_ins] at hlp
      exact hlp
  case tpair =>
    intro Delta' Sigma' G1 G2 G3 e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hf1, hf2⟩ := freshInTerm_pair hft
    obtain ⟨G2p, G2q, hG2, h1'⟩ := ih1 Gm_pre Gm_post hsplit hfp hfq hf1
    have h1s := by
      have := _h1
      rw [hsplit, hG2] at this
      exact this
    obtain ⟨hfp2, hfq2⟩ := weakening_insert_shrink_split h1s hfp hfq
    subst hG2
    obtain ⟨G3p, G3q, hG3, h2'⟩ := ih2 G2p G2q rfl hfp2 hfq2 hf2
    refine ⟨G3p, G3q, hG3, ?_⟩
    exact HasType.tpair Delta' Sigma' _ _ _ e1 e2 t1 t2 eps1 eps2 h1' h2'
  case fst =>
    intro Delta' Sigma' G1 G2 e' t1 t2 eps' _h ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_fst hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.fst Delta' Sigma' _ _ e' t1 t2 eps' h'
  case snd =>
    intro Delta' Sigma' G1 G2 e' t1 t2 eps' _h ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_snd hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.snd Delta' Sigma' _ _ e' t1 t2 eps' h'
  case const =>
    intro Delta' Sigma' Gamma v ds
    intro Gm_pre Gm_post hsplit _ _ _
    refine ⟨Gm_pre, Gm_post, hsplit, ?_⟩
    exact HasType.const Delta' Sigma' _ v ds
  case tadd =>
    intro Delta' Sigma' G1 G2 G3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hf1, hf2⟩ := freshInTerm_add hft
    obtain ⟨G2p, G2q, hG2, h1'⟩ := ih1 Gm_pre Gm_post hsplit hfp hfq hf1
    have h1s := by
      have := _h1
      rw [hsplit, hG2] at this
      exact this
    obtain ⟨hfp2, hfq2⟩ := weakening_insert_shrink_split h1s hfp hfq
    subst hG2
    obtain ⟨G3p, G3q, hG3, h2'⟩ := ih2 G2p G2q rfl hfp2 hfq2 hf2
    refine ⟨G3p, G3q, hG3, ?_⟩
    exact HasType.tadd Delta' Sigma' _ _ _ e1 e2 ds eps1 eps2 h1' h2'
  case tmul =>
    intro Delta' Sigma' G1 G2 G3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨hf1, hf2⟩ := freshInTerm_mul hft
    obtain ⟨G2p, G2q, hG2, h1'⟩ := ih1 Gm_pre Gm_post hsplit hfp hfq hf1
    have h1s := by
      have := _h1
      rw [hsplit, hG2] at this
      exact this
    obtain ⟨hfp2, hfq2⟩ := weakening_insert_shrink_split h1s hfp hfq
    subst hG2
    obtain ⟨G3p, G3q, hG3, h2'⟩ := ih2 G2p G2q rfl hfp2 hfq2 hf2
    refine ⟨G3p, G3q, hG3, ?_⟩
    exact HasType.tmul Delta' Sigma' _ _ _ e1 e2 ds eps1 eps2 h1' h2'
  case tsum =>
    intro Delta' Sigma' G1 G2 e' ds i eps' _h hi ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_sum hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.tsum Delta' Sigma' _ _ e' ds i eps' h' hi
  case texpand =>
    intro Delta' Sigma' G1 G2 e' ds i k eps' _h hi ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_expand hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.texpand Delta' Sigma' _ _ e' ds i k eps' h' hi
  case uniformLike =>
    intro Delta' Sigma' G1 G2 e' ds lo hi eps' _h ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_uniformLike hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.uniformLike Delta' Sigma' _ _ e' ds lo hi eps' h'
  case perform =>
    intro Delta' Sigma' G1 G2 op e' tArg tRet eps' _h hMatch ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    have hfe := freshInTerm_perform hft
    obtain ⟨G2p, G2q, hG2, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hfe
    refine ⟨G2p, G2q, hG2, ?_⟩
    exact HasType.perform Delta' Sigma' _ _ op e' tArg tRet eps' h' hMatch
  case handle =>
    intro _Delta' _Sigma' _G1 _G2 _G3 _body _clauses _t _epsH _epsB
           _hb _hEpsH _hCl _hCov _hCT _ihb _ihCT
    intro _Gm_pre _Gm_post _hsplit _hfp _hfq _hft
    -- TODO: handle case — requires motive_2 threading. Left as internal sorry.
    sorry
  case tgrad =>
    intro Delta' Sigma' Gamma x ds dsOut e' eps' _h hsub _ih
    intro Gm_pre Gm_post hsplit hfp hfq _hft
    -- tgrad's input = output = Gamma; sub-derivation at Gamma ++ [(x, tensor ds)] -> Gamma.
    -- Simply use Gm_pre, Gm_post as the result.
    refine ⟨Gm_pre, Gm_post, hsplit, ?_⟩
    -- We need to produce a tgrad whose sub-derivation works at
    -- (Gm_pre ++ [(y,t_y)] ++ Gm_post) ++ [(x, tensor ds)] -> (Gm_pre ++ [(y,t_y)] ++ Gm_post).
    -- This needs re-applying the IH on the sub-derivation, but the sub-derivation uses
    -- a different Delta' (Capability.diff :: Delta'). That's outside our outer Delta' param.
    -- Left as internal sorry: tgrad weakening requires a Delta'-polymorphic motive.
    sorry
  case tvmap =>
    intro Delta' Sigma' Gamma x t1 t2 e' eps' d _h _ih
    intro Gm_pre Gm_post hsplit _hfp _hfq _hft
    refine ⟨Gm_pre, Gm_post, hsplit, ?_⟩
    sorry
  case loc =>
    intro Delta' Sigma' Gamma ell t' hlook
    intro Gm_pre Gm_post hsplit _ _ _
    refine ⟨Gm_pre, Gm_post, hsplit, ?_⟩
    exact HasType.loc Delta' Sigma' _ ell t' hlook
  case subEff =>
    intro Delta' Sigma' Gamma Gamma'' e' t' eps' eps'' _h hsub ih
    intro Gm_pre Gm_post hsplit hfp hfq hft
    obtain ⟨Gp, Gq, hGeq, h'⟩ := ih Gm_pre Gm_post hsplit hfp hfq hft
    refine ⟨Gp, Gq, hGeq, ?_⟩
    exact HasType.subEff Delta' Sigma' _ _ e' t' eps' eps'' h' hsub
  case nil =>
    intro Delta' Sigma' Gamma2 t' epsR
    intro Gm_pre Gm_post hsplit _ _ _
    refine ⟨Gm_pre, Gm_post, hsplit, ?_⟩
    exact ClausesTyped.nil Delta' Sigma' _ t' epsR
  case cons =>
    intros
    sorry


/-- Weakening: adding an unused binding at the tail of the linear
    context preserves typing. Derived as a corollary of
    `weakening_insert` with `Gamma_post = []`.

    The Wave 2 obstruction (naïve induction creates an unprovable
    adjacent-swap goal inside every binder case) is dissolved by the
    position-indexed helper: with `Γ_post = []`, every binder case
    re-splits to `Γ_pre / [(x, t1)]` at the next level down, and the
    IH produces the required tail-insertion directly with no exchange.

    Requires `y` to be fresh in `e` (`freshInTerm y e`) so that no
    internal T-Var can accidentally consume the inserted binding. -/
theorem weakening_tail
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (y : String) (t t' : Typ) (eps : EffectRow) (e : Term)
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (h_fresh : y ∉ linearCtxDom Gamma)
    (h_fresh_term : freshInTerm y e) :
    ∃ Gamma'_pre Gamma'_post : LinearCtx,
      Gamma' = Gamma'_pre ++ Gamma'_post ∧
      HasType Delta Sigma (Gamma ++ [(y, t')]) e t eps
              (Gamma'_pre ++ [(y, t')] ++ Gamma'_post) := by
  -- Direct corollary of `weakening_insert` with `Γ_pre = Γ`, `Γ_post = []`.
  -- The weaker existential return type (vs. the original rigid
  -- `Γ' ++ [(y, t')]`) is necessary: binder cases of the underlying
  -- induction may consume bindings that float the inserted position,
  -- so the output context's split point is not rigid.
  have hfp : y ∉ linearCtxDom ([] : LinearCtx) := by simp [linearCtxDom]
  have hsplit : Gamma = Gamma ++ ([] : LinearCtx) := by simp
  have hres := weakening_insert Delta Sigma y t' h Gamma [] hsplit h_fresh hfp h_fresh_term
  simpa using hres

/-- Exchange: swapping two adjacent unrelated bindings in the linear
    context preserves typing. Used when a substitution introduces a
    fresh binding mid-context and the surrounding derivation needs to
    thread around it. -/
-- NOTE (Wave 1 proof engineer, 2026-04-13): the originally-advised
-- `exchange_tail` statement below — which FIXES the output context
-- `Γ'` across the swap — is not provable. Counterexample: the `var`
-- case of `HasType` consumes the tail binding of its input context.
-- If `Γin = Γpre ++ [(x,t1),(y,t2)]`, the unswapped derivation of
-- `Term.var y` has output `Γpre ++ [(x,t1)]`. The swapped context
-- `Γpre ++ [(y,t2),(x,t1)]` only has `(x,t1)` at its tail, so the
-- only available var-rule derivation has output `Γpre ++ [(y,t2)]`.
-- These outputs differ, so no derivation with `Γout` rigidly equal
-- to the unswapped output exists.
--
-- The mathematically-correct form swaps BOTH endpoints: "either the
-- swapped bindings are not consumed (Γout factors as Γ'' ++ [(x,t1),
-- (y,t2)] and we return Γ'' ++ [(y,t2),(x,t1)]), or one/both are
-- consumed and Γout is a prefix that already sits below them". This
-- is a disjunction-shaped conclusion that doesn't match the shape
-- `subst_preserves_typing` actually needs.
--
-- Keeping the original statement as `sorry` for Wave 2 restructuring;
-- documenting the obstruction so the next author doesn't re-derive it.
theorem exchange_tail
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (x y : String) (t t1 t2 : Typ) (eps : EffectRow) (e : Term)
    (_h : HasType Delta Sigma (Gamma ++ [(x, t1), (y, t2)]) e
                  t eps Gamma')
    (_h_ne : x ≠ y) :
    HasType Delta Sigma (Gamma ++ [(y, t2), (x, t1)]) e
            t eps Gamma' := by
  sorry -- BLOCKED: statement is not provable as written (see note above).
        -- Wave 2 must restructure the signature (output-swap disjunction,
        -- or strengthen to an explicit "neither swapped binding is
        -- consumed" precondition) before this can be closed.

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
  sorry -- BLOCKED on weakening_tail / exchange_tail refactor.
        -- The original plan (induction on _h_e, invoking exchange_tail in
        -- every binder case) is not realizable because both helpers have
        -- unprovable statements (see notes on `weakening_tail` and
        -- `exchange_tail`). Restructuring required: the motive must
        -- generalize over a pre/post decomposition of the context so
        -- that `(x,t1)` can sit anywhere, and binder cases thread their
        -- new binding into the post-segment without needing an exchange.
        -- This is a Wave 2 signature refactor plus full induction pass.

end LaCaDiLE
