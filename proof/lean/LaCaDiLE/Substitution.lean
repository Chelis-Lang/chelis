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
  | Term.vmap y t d body =>
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

mutual

/-- Every runtime location appearing after substitution already came
    from the original term or from the substituted value. -/
theorem mem_locRefs_subst
    (e v : Term) (x : String) (ell : Loc)
    (hmem : ell ∈ locRefs (subst e v x)) :
    ell ∈ locRefs e ∨ ell ∈ locRefs v := by
  match e with
  | Term.var y =>
      by_cases hy : y = x
      · simp [subst, hy] at hmem ⊢
        exact Or.inr hmem
      · simp [subst, hy, locRefs] at hmem
  | Term.abs y t body =>
      by_cases hy : y = x
      · simp [subst, hy] at hmem
        exact Or.inl hmem
      · simp [subst, hy] at hmem
        exact mem_locRefs_subst body v x ell hmem
  | Term.app e1 e2 =>
      simp [subst, locRefs] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefs_subst e2 v x ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.letBind y e1 e2 =>
      by_cases hy : y = x
      · simp [subst, hy, locRefs] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
            (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
        · exact Or.inl (Or.inr hmem)
      · simp [subst, hy, locRefs] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
            (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
        · exact Or.elim (mem_locRefs_subst e2 v x ell hmem)
            (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.copy e =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.letpair y z e1 e2 =>
      by_cases hy : y = x
      · simp [subst, hy, locRefs] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
            (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
        · exact Or.inl (Or.inr hmem)
      · by_cases hz : z = x
        · simp [subst, hy, hz, locRefs] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
              (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
          · exact Or.inl (Or.inr hmem)
        · simp [subst, hy, hz, locRefs] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
              (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
          · exact Or.elim (mem_locRefs_subst e2 v x ell hmem)
              (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.pair e1 e2 =>
      simp [subst, locRefs] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefs_subst e2 v x ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.fst e =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.snd e =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.unit =>
      simp [subst, locRefs] at hmem
  | Term.const _ _ =>
      simp [subst, locRefs] at hmem
  | Term.add e1 e2 =>
      simp [subst, locRefs] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefs_subst e2 v x ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.mul e1 e2 =>
      simp [subst, locRefs] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefs_subst e1 v x ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefs_subst e2 v x ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.sum e d =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.expand e d =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.uniformLike e lo hi =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.grad y t tOut body =>
      by_cases hy : y = x
      · simp [subst, hy] at hmem
        exact Or.inl hmem
      · simp [subst, hy] at hmem
        exact mem_locRefs_subst body v x ell hmem
  | Term.vmap y t d body =>
      by_cases hy : y = x
      · simp [subst, hy] at hmem
        exact Or.inl hmem
      · simp [subst, hy] at hmem
        exact mem_locRefs_subst body v x ell hmem
  | Term.handle epsH body clauses =>
      simp [subst, locRefs] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefs_subst body v x ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefs_substClauses clauses v x ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.perform op e =>
      simp [subst, locRefs] at hmem ⊢
      exact mem_locRefs_subst e v x ell hmem
  | Term.loc ell' =>
      simp [subst, locRefs] at hmem
      exact Or.inl (by simpa [locRefs, hmem])

/-- Clause-list version of `mem_locRefs_subst`. -/
theorem mem_locRefs_substClauses
    (clauses : List (EffectLabel × String × String × Term))
    (v : Term) (x : String) (ell : Loc)
    (hmem : ell ∈ locRefsClauses (substClauses clauses v x)) :
    ell ∈ locRefsClauses clauses ∨ ell ∈ locRefs v := by
  match clauses with
  | [] =>
      simp [substClauses, locRefsClauses] at hmem
  | (op, y, k, hb) :: rest =>
      by_cases hy : y = x
      · simp [substClauses, subst, hy, locRefsClauses] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · exact Or.elim (mem_locRefs_substClauses rest v x ell hmem)
            (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
      · by_cases hk : k = x
        · simp [substClauses, subst, hy, hk, locRefsClauses] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.inl (Or.inl hmem)
          · exact Or.elim (mem_locRefs_substClauses rest v x ell hmem)
              (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
        · simp [substClauses, subst, hy, hk, locRefsClauses] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.elim (mem_locRefs_subst hb v x ell hmem)
              (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
          · exact Or.elim (mem_locRefs_substClauses rest v x ell hmem)
              (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)

end

mutual

/-- Every binder appearing after substitution already came from the
    original term or from the substituted value. -/
theorem mem_boundVars_subst
    (e v : Term) (x z : String)
    (hmem : z ∈ boundVars (subst e v x)) :
    z ∈ boundVars e ∨ z ∈ boundVars v := by
  match e with
  | Term.var y =>
      by_cases hy : y = x
      · simp [subst, hy] at hmem
        exact Or.inr hmem
      · simp [subst, hy, boundVars] at hmem
  | Term.abs y t body =>
      by_cases hy : y = x
      · simp [subst, hy, boundVars] at hmem ⊢
        exact Or.inl hmem
      · simp [subst, hy, boundVars] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · exact Or.elim (mem_boundVars_subst body v x z hmem)
            (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.app e1 e2 =>
      simp [subst, boundVars] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_boundVars_subst e2 v x z hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.letBind y e1 e2 =>
      by_cases hy : y = x
      · simp [subst, hy, boundVars] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · rcases hmem with hmem | hmem
          · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
              (fun h => Or.inl (Or.inr (Or.inl h))) (fun h => Or.inr h)
          · exact Or.inl (Or.inr (Or.inr hmem))
      · simp [subst, hy, boundVars] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · rcases hmem with hmem | hmem
          · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
              (fun h => Or.inl (Or.inr (Or.inl h))) (fun h => Or.inr h)
          · exact Or.elim (mem_boundVars_subst e2 v x z hmem)
              (fun h => Or.inl (Or.inr (Or.inr h))) (fun h => Or.inr h)
  | Term.copy e =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.letpair y k e1 e2 =>
      by_cases hy : y = x
      · simp [subst, hy, boundVars] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · rcases hmem with hmem | hmem
          · exact Or.inl (Or.inr (Or.inl hmem))
          · rcases hmem with hmem | hmem
            · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
                (fun h => Or.inl (Or.inr (Or.inr (Or.inl h)))) (fun h => Or.inr h)
            · exact Or.inl (Or.inr (Or.inr (Or.inr hmem)))
      · by_cases hk : k = x
        · simp [subst, hy, hk, boundVars] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.inl (Or.inl hmem)
          · rcases hmem with hmem | hmem
            · exact Or.inl (Or.inr (Or.inl hmem))
            · rcases hmem with hmem | hmem
              · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
                  (fun h => Or.inl (Or.inr (Or.inr (Or.inl h)))) (fun h => Or.inr h)
              · exact Or.inl (Or.inr (Or.inr (Or.inr hmem)))
        · simp [subst, hy, hk, boundVars] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.inl (Or.inl hmem)
          · rcases hmem with hmem | hmem
            · exact Or.inl (Or.inr (Or.inl hmem))
            · rcases hmem with hmem | hmem
              · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
                  (fun h => Or.inl (Or.inr (Or.inr (Or.inl h)))) (fun h => Or.inr h)
              · exact Or.elim (mem_boundVars_subst e2 v x z hmem)
                  (fun h => Or.inl (Or.inr (Or.inr (Or.inr h)))) (fun h => Or.inr h)
  | Term.pair e1 e2 =>
      simp [subst, boundVars] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_boundVars_subst e2 v x z hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.fst e =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.snd e =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.unit =>
      simp [subst, boundVars] at hmem
  | Term.const _ _ =>
      simp [subst, boundVars] at hmem
  | Term.add e1 e2 =>
      simp [subst, boundVars] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_boundVars_subst e2 v x z hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.mul e1 e2 =>
      simp [subst, boundVars] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_boundVars_subst e1 v x z hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_boundVars_subst e2 v x z hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.sum e d =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.expand e d =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.uniformLike e lo hi =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.grad y t tOut body =>
      by_cases hy : y = x
      · simp [subst, hy, boundVars] at hmem ⊢
        exact Or.inl hmem
      · simp [subst, hy, boundVars] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · exact Or.elim (mem_boundVars_subst body v x z hmem)
            (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.vmap y t d body =>
      by_cases hy : y = x
      · simp [subst, hy, boundVars] at hmem ⊢
        exact Or.inl hmem
      · simp [subst, hy, boundVars] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · exact Or.elim (mem_boundVars_subst body v x z hmem)
            (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.handle epsH body clauses =>
      simp [subst, boundVars] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_boundVars_subst body v x z hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_boundVars_substClauses clauses v x z hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | Term.perform op e =>
      simp [subst, boundVars] at hmem ⊢
      exact mem_boundVars_subst e v x z hmem
  | Term.loc ell =>
      simp [subst, boundVars] at hmem

/-- Clause-list version of `mem_boundVars_subst`. -/
theorem mem_boundVars_substClauses
    (clauses : List (EffectLabel × String × String × Term))
    (v : Term) (x z : String)
    (hmem : z ∈ boundVarsClauses (substClauses clauses v x)) :
    z ∈ boundVarsClauses clauses ∨ z ∈ boundVars v := by
  match clauses with
  | [] =>
      simp [substClauses, boundVarsClauses] at hmem
  | (op, y, k, hb) :: rest =>
      by_cases hy : y = x
      · simp [substClauses, subst, hy, boundVarsClauses] at hmem ⊢
        rcases hmem with hmem | hmem
        · exact Or.inl (Or.inl hmem)
        · rcases hmem with hmem | hmem
          · exact Or.inl (Or.inr (Or.inl hmem))
          · rcases hmem with hmem | hmem
            · exact Or.inl (Or.inr (Or.inr (Or.inl hmem)))
            · exact Or.elim (mem_boundVars_substClauses rest v x z hmem)
                (fun h => Or.inl (Or.inr (Or.inr (Or.inr h)))) (fun h => Or.inr h)
      · by_cases hk : k = x
        · simp [substClauses, subst, hy, hk, boundVarsClauses] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.inl (Or.inl hmem)
          · rcases hmem with hmem | hmem
            · exact Or.inl (Or.inr (Or.inl hmem))
            · rcases hmem with hmem | hmem
              · exact Or.inl (Or.inr (Or.inr (Or.inl hmem)))
              · exact Or.elim (mem_boundVars_substClauses rest v x z hmem)
                  (fun h => Or.inl (Or.inr (Or.inr (Or.inr h)))) (fun h => Or.inr h)
        · simp [substClauses, subst, hy, hk, boundVarsClauses] at hmem ⊢
          rcases hmem with hmem | hmem
          · exact Or.inl (Or.inl hmem)
          · rcases hmem with hmem | hmem
            · exact Or.inl (Or.inr (Or.inl hmem))
            · rcases hmem with hmem | hmem
              · exact Or.elim (mem_boundVars_subst hb v x z hmem)
                  (fun h => Or.inl (Or.inr (Or.inr (Or.inl h)))) (fun h => Or.inr h)
              · exact Or.elim (mem_boundVars_substClauses rest v x z hmem)
                  (fun h => Or.inl (Or.inr (Or.inr (Or.inr h)))) (fun h => Or.inr h)

end

/-- If `z` is absent from both `e` and `v`, substitution cannot
    introduce `z` as a binder in the result. -/
theorem subst_notBound
    (e v : Term) (x z : String)
    (he : z ∉ boundVars e) (hv : z ∉ boundVars v) :
    z ∉ boundVars (subst e v x) := by
  intro h
  exact (mem_boundVars_subst e v x z h).elim he hv

/-- Clause-list version of `subst_notBound`. -/
theorem substClauses_notBound
    (clauses : List (EffectLabel × String × String × Term))
    (v : Term) (x z : String)
    (hcls : z ∉ boundVarsClauses clauses) (hv : z ∉ boundVars v) :
    z ∉ boundVarsClauses (substClauses clauses v x) := by
  intro h
  exact (mem_boundVars_substClauses clauses v x z h).elim hcls hv

/-- Binder-freshness transport through substitution. If every binder
    in `e` and every binder in the substituted payload `v` is fresh for
    `rhs`, then every binder in `subst e v x` is fresh for `rhs`. -/
theorem subst_bound_fresh
    (e v rhs : Term) (x : String)
    (he : ∀ z, z ∈ boundVars e → z ∉ boundVars rhs)
    (hv : ∀ z, z ∈ boundVars v → z ∉ boundVars rhs) :
    ∀ z, z ∈ boundVars (subst e v x) → z ∉ boundVars rhs := by
  intro z hz
  rcases mem_boundVars_subst e v x z hz with hzE | hzV
  · exact he z hzE
  · exact hv z hzV

/-- Context-name freshness transport through substitution. If every
    context name in `Γ` is absent from the binders of both `e` and `v`,
    then it is also absent from the binders of `subst e v x`. -/
theorem subst_ctx_bound_fresh
    {Gamma : LinearCtx}
    (e v : Term) (x : String)
    (he : ∀ z, z ∈ linearCtxDom Gamma → z ∉ boundVars e)
    (hv : ∀ z, z ∈ linearCtxDom Gamma → z ∉ boundVars v) :
    ∀ z, z ∈ linearCtxDom Gamma → z ∉ boundVars (subst e v x) := by
  intro z hz
  exact subst_notBound e v x z (he z hz) (hv z hz)

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

/-! ### Named context insertion before a suffix

The DB weakening proof uses a positional `insertAt` counted from the
innermost end of the context. For named contexts, the corresponding
operation is “insert immediately before the suffix of length `j`”.
This keeps binder cases aligned with the existing tombstone-style body
judgments: descending under one binder increments `j` by 1, under two
binders by 2. -/

private def insertAtHead {α : Type} : Nat → α → List α → List α
  | 0, a, xs => a :: xs
  | _ + 1, a, [] => [a]
  | n + 1, a, x :: xs => x :: insertAtHead n a xs

@[simp] theorem insertAtHead_zero {α : Type} (a : α) (xs : List α) :
    insertAtHead 0 a xs = a :: xs := rfl

@[simp] theorem insertAtHead_nil_succ {α : Type} (n : Nat) (a : α) :
    insertAtHead (n + 1) a [] = [a] := rfl

@[simp] theorem insertAtHead_cons_succ {α : Type} (n : Nat) (a x : α) (xs : List α) :
    insertAtHead (n + 1) a (x :: xs) = x :: insertAtHead n a xs := rfl

theorem insertAtHead_append_len {α : Type} (xs ys : List α) (a : α) :
    insertAtHead xs.length a (xs ++ ys) = xs ++ a :: ys := by
  induction xs with
  | nil =>
      simp [insertAtHead]
  | cons x xs ih =>
      simp [insertAtHead, ih]

/-- Insert `entry` immediately before the suffix of length `j`. -/
private def insertBeforeSuffix
    (Γ : LinearCtx) (j : Nat) (entry : String × Option Typ) : LinearCtx :=
  (insertAtHead j entry Γ.reverse).reverse

theorem insertBeforeSuffix_eq_split
    (GammaPre GammaPost : LinearCtx) (entry : String × Option Typ) :
    insertBeforeSuffix (GammaPre ++ GammaPost) GammaPost.length entry =
      GammaPre ++ [entry] ++ GammaPost := by
  unfold insertBeforeSuffix
  have h :=
    insertAtHead_append_len GammaPost.reverse GammaPre.reverse entry
  simpa [List.reverse_append, List.append_assoc] using congrArg List.reverse h

@[simp] theorem insertBeforeSuffix_zero
    (Γ : LinearCtx) (entry : String × Option Typ) :
    insertBeforeSuffix Γ 0 entry = Γ ++ [entry] := by
  simpa using insertBeforeSuffix_eq_split Γ [] entry

@[simp] theorem insertBeforeSuffix_append_singleton
    (Γ : LinearCtx) (j : Nat) (entry : String × Option Typ)
    (x : String) (slot : Option Typ) :
    insertBeforeSuffix (Γ ++ [(x, slot)]) (j + 1) entry =
      insertBeforeSuffix Γ j entry ++ [(x, slot)] := by
  unfold insertBeforeSuffix
  simp [List.reverse_append, List.append_assoc]

@[simp] theorem insertBeforeSuffix_append_pair
    (Γ : LinearCtx) (j : Nat) (entry : String × Option Typ)
    (x y : String) (slotX slotY : Option Typ) :
    insertBeforeSuffix (Γ ++ [(x, slotX), (y, slotY)]) (j + 2) entry =
      insertBeforeSuffix Γ j entry ++ [(x, slotX), (y, slotY)] := by
  unfold insertBeforeSuffix
  simp [List.reverse_append, List.append_assoc]

theorem insertBeforeSuffix_var_right
    (GammaPre GammaPost : LinearCtx) (j : Nat)
    (entry : String × Option Typ)
    (x : String) (slot : Option Typ)
    (hj : j ≤ GammaPost.length) :
    insertBeforeSuffix (GammaPre ++ [(x, slot)] ++ GammaPost) j entry =
      GammaPre ++ [(x, slot)] ++ insertBeforeSuffix GammaPost j entry := by
  let postPre := GammaPost.take (GammaPost.length - j)
  let postSuf := GammaPost.drop (GammaPost.length - j)
  have hsplit : GammaPost = postPre ++ postSuf := by
    simp [postPre, postSuf, List.take_append_drop]
  have hlen : postSuf.length = j := by
    simp [postSuf]
    omega
  calc
    insertBeforeSuffix (GammaPre ++ [(x, slot)] ++ GammaPost) j entry
        = insertBeforeSuffix ((GammaPre ++ [(x, slot)] ++ postPre) ++ postSuf) postSuf.length entry := by
            simp [hsplit, hlen, List.append_assoc]
    _ = (GammaPre ++ [(x, slot)] ++ postPre) ++ [entry] ++ postSuf := by
          simpa using insertBeforeSuffix_eq_split
            (GammaPre ++ [(x, slot)] ++ postPre) postSuf entry
    _ = GammaPre ++ [(x, slot)] ++ insertBeforeSuffix GammaPost j entry := by
          have htail : insertBeforeSuffix GammaPost j entry =
              postPre ++ [entry] ++ postSuf := by
            simpa [hsplit, hlen] using insertBeforeSuffix_eq_split postPre postSuf entry
          simp [htail, List.append_assoc]

theorem insertBeforeSuffix_var_left
    (GammaPre GammaPost : LinearCtx) (j : Nat)
    (entry : String × Option Typ)
    (x : String) (slot : Option Typ)
    (hj : GammaPost.length < j)
    (hjt : j ≤ (GammaPre ++ [(x, slot)] ++ GammaPost).length) :
    let k := j - (GammaPost.length + 1)
    insertBeforeSuffix (GammaPre ++ [(x, slot)] ++ GammaPost) j entry =
      insertBeforeSuffix GammaPre k entry ++ [(x, slot)] ++ GammaPost := by
  let k := j - (GammaPost.length + 1)
  let prePre := GammaPre.take (GammaPre.length - k)
  let preSuf := GammaPre.drop (GammaPre.length - k)
  have hsplit : GammaPre = prePre ++ preSuf := by
    simp [prePre, preSuf, List.take_append_drop]
  have hj' : GammaPost.length + 1 ≤ j := by
    omega
  have hk : k ≤ GammaPre.length := by
    simp [k] at hjt
    omega
  have hprelen : preSuf.length = k := by
    simp [preSuf, Nat.sub_sub_self hk]
  have hidx : preSuf.length + (GammaPost.length + 1) = j := by
    calc
      preSuf.length + (GammaPost.length + 1)
          = k + (GammaPost.length + 1) := by simp [hprelen]
      _ = j := by
          dsimp [k]
          simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
            (Nat.add_sub_cancel' hj' : (GammaPost.length + 1) + (j - (GammaPost.length + 1)) = j)
  have hlen : (preSuf ++ [(x, slot)] ++ GammaPost).length = j := by
    calc
      (preSuf ++ [(x, slot)] ++ GammaPost).length
          = preSuf.length + (GammaPost.length + 1) := by
              simp [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      _ = j := hidx
  calc
    insertBeforeSuffix (GammaPre ++ [(x, slot)] ++ GammaPost) j entry
        = insertBeforeSuffix (prePre ++ (preSuf ++ [(x, slot)] ++ GammaPost))
            (preSuf ++ [(x, slot)] ++ GammaPost).length entry := by
            rw [hsplit, ← hlen]
            simp [List.append_assoc]
    _ = prePre ++ [entry] ++ (preSuf ++ [(x, slot)] ++ GammaPost) := by
          simpa [List.append_assoc] using
            insertBeforeSuffix_eq_split prePre (preSuf ++ [(x, slot)] ++ GammaPost) entry
    _ = insertBeforeSuffix GammaPre k entry ++ [(x, slot)] ++ GammaPost := by
          have htail : insertBeforeSuffix GammaPre k entry =
              prePre ++ [entry] ++ preSuf := by
            simpa [hsplit, hprelen] using insertBeforeSuffix_eq_split prePre preSuf entry
          simp [htail, List.append_assoc]

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

theorem freshInTerm_grad {y x : String} {t tOut : Typ} {body : Term}
    (h : freshInTerm y (Term.grad x t tOut body)) : y ≠ x ∧ freshInTerm y body := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, List.mem_filter, bne_iff_ne, ne_eq] at hf
  simp only [boundVars, List.mem_cons] at hb
  refine ⟨?_, ?_, ?_⟩
  · intro he
    exact hb (Or.inl he)
  · intro hfb
    by_cases hxy : y = x
    · exact hb (Or.inl hxy)
    · exact hf ⟨hfb, fun he => hxy he⟩
  · intro hbb
    exact hb (Or.inr hbb)

theorem freshInTerm_vmap {y x : String} {t : Typ} {d : Dim} {body : Term}
    (h : freshInTerm y (Term.vmap x t d body)) : y ≠ x ∧ freshInTerm y body := by
  rcases h with ⟨hf, hb⟩
  simp only [freeVars, List.mem_filter, bne_iff_ne, ne_eq] at hf
  simp only [boundVars, List.mem_cons] at hb
  refine ⟨?_, ?_, ?_⟩
  · intro he
    exact hb (Or.inl he)
  · intro hfb
    by_cases hxy : y = x
    · exact hb (Or.inl hxy)
    · exact hf ⟨hfb, fun he => hxy he⟩
  · intro hbb
    exact hb (Or.inr hbb)

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

theorem capturedContName_fresh_handle_body
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)} :
    freshInTerm (capturedContName (Term.handle epsH body clauses))
      (Term.handle epsH body clauses) := by
  exact capturedContName_freshInTerm _

theorem capturedContName_fresh_selected_clause
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses) :
    capturedContName (Term.handle epsH body clauses) ≠ x ∧
    capturedContName (Term.handle epsH body clauses) ≠ k ∧
    freshInTerm (capturedContName (Term.handle epsH body clauses)) hb := by
  exact freshInTerm_handle_clause
    (capturedContName_freshInTerm (Term.handle epsH body clauses)) hmem


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

/-- If a singleton slot `(y, slotY)` appears somewhere in
    `GammaPre ++ [(x, slotX)] ++ suffix` with `y ≠ x`, then removing the
    distinguished middle `x` slot still leaves `(y, slotY)` somewhere in
    `GammaPre ++ suffix`. This is the non-target `var` case for the
    context-fresh substitution theorem. -/
private theorem split_remove_middle_singleton
    {GammaPre suffix Gamma1 Gamma2 : LinearCtx}
    {x y : String} {slotX slotY : Option Typ}
    (hEq : Gamma1 ++ [(y, slotY)] ++ Gamma2 = GammaPre ++ [(x, slotX)] ++ suffix)
    (hxy : y ≠ x) :
    ∃ pre' post' : LinearCtx,
      GammaPre ++ suffix = pre' ++ [(y, slotY)] ++ post' := by
  rcases list_append_eq_split Gamma1 ([(y, slotY)] ++ Gamma2)
      GammaPre ([(x, slotX)] ++ suffix)
      (by simpa [List.append_assoc] using hEq) with
    ⟨m, hPre, hRest⟩ | ⟨m, hPre, hRest⟩
  · cases m with
    | nil =>
        have hRest' : (y, slotY) :: Gamma2 = (x, slotX) :: suffix := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead _hTail
        cases hHead
        exact False.elim (hxy rfl)
    | cons hd tl =>
        rcases hd with ⟨z, slotZ⟩
        have hRest' : (y, slotY) :: Gamma2 = (z, slotZ) :: (tl ++ [(x, slotX)] ++ suffix) := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead htl
        cases hHead
        refine ⟨Gamma1, tl ++ suffix, ?_⟩
        simp [hPre, htl, List.append_assoc]
  · cases m with
    | nil =>
        have hRest' : (x, slotX) :: suffix = (y, slotY) :: Gamma2 := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead _hTail
        cases hHead
        exact False.elim (hxy rfl.symm)
    | cons hd tl =>
        rcases hd with ⟨z, slotZ⟩
        have hRest' : (x, slotX) :: suffix = (z, slotZ) :: (tl ++ [(y, slotY)] ++ Gamma2) := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead htl
        cases hHead
        refine ⟨GammaPre ++ tl, Gamma2, ?_⟩
        simp [htl, List.append_assoc]

private theorem split_target_middle_singleton
    {GammaPre suffix Gamma1 Gamma2 : LinearCtx}
    {x : String} {slotX : Option Typ} {t : Typ}
    (hEq : Gamma1 ++ [(x, some t)] ++ Gamma2 = GammaPre ++ [(x, slotX)] ++ suffix)
    (hnd : NoDupNames (GammaPre ++ [(x, slotX)] ++ suffix)) :
    Gamma1 = GammaPre ∧ Gamma2 = suffix ∧ slotX = some t := by
  have hndSome : NoDupNames (GammaPre ++ [(x, some t)] ++ suffix) := by
    simpa [NoDupNames, linearCtxDom] using hnd
  rcases list_append_eq_split Gamma1 ([(x, some t)] ++ Gamma2)
      GammaPre ([(x, slotX)] ++ suffix)
      (by simpa [List.append_assoc] using hEq) with
    ⟨m, hPre, hRest⟩ | ⟨m, hPre, hRest⟩
  · cases m with
    | nil =>
        have hRest' : (x, some t) :: Gamma2 = (x, slotX) :: suffix := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        exact ⟨by simpa using hPre.symm, hTail, rfl⟩
    | cons hd tl =>
        rcases hd with ⟨z, slotZ⟩
        have hRest' : (x, some t) :: Gamma2 = (z, slotZ) :: (tl ++ [(x, slotX)] ++ suffix) := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        have hxPre : x ∈ linearCtxDom GammaPre := by
          rw [hPre]
          simp [linearCtxDom, List.append_assoc]
        exact False.elim
          ((noDupNames_middle_fresh_prefix (GammaPre := GammaPre) (GammaPost := suffix)
              (x := x) (t := t) hndSome) hxPre)
  · cases m with
    | nil =>
        have hRest' : (x, slotX) :: suffix = (x, some t) :: Gamma2 := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        exact ⟨by simpa using hPre, hTail.symm, rfl⟩
    | cons hd tl =>
        rcases hd with ⟨z, slotZ⟩
        have hRest' : (x, slotX) :: suffix = (z, slotZ) :: (tl ++ [(x, some t)] ++ Gamma2) := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        have hxSuf : x ∈ linearCtxDom suffix := by
          rw [hTail]
          simp [linearCtxDom, List.append_assoc]
        exact False.elim
          ((noDupNames_middle_fresh_suffix (GammaPre := GammaPre) (GammaPost := suffix)
              (x := x) (t := t) hndSome) hxSuf)

private theorem split_middle_singleton
    {GammaPre suffix Gamma1 Gamma2 : LinearCtx}
    {x : String} {slot1 slot2 : Option Typ}
    (hEq : Gamma1 ++ [(x, slot1)] ++ Gamma2 = GammaPre ++ [(x, slot2)] ++ suffix)
    (hnd : NoDupNames (GammaPre ++ [(x, slot2)] ++ suffix)) :
    Gamma1 = GammaPre ∧ Gamma2 = suffix ∧ slot1 = slot2 := by
  have hndAny : NoDupNames (GammaPre ++ [(x, some Typ.unit)] ++ suffix) := by
    simpa [NoDupNames, linearCtxDom] using hnd
  rcases list_append_eq_split Gamma1 ([(x, slot1)] ++ Gamma2)
      GammaPre ([(x, slot2)] ++ suffix)
      (by simpa [List.append_assoc] using hEq) with
    ⟨m, hPre, hRest⟩ | ⟨m, hPre, hRest⟩
  · cases m with
    | nil =>
        have hRest' : (x, slot1) :: Gamma2 = (x, slot2) :: suffix := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        exact ⟨by simpa using hPre.symm, hTail, rfl⟩
    | cons hd tl =>
        rcases hd with ⟨z, slotZ⟩
        have hRest' : (x, slot1) :: Gamma2 = (z, slotZ) :: (tl ++ [(x, slot2)] ++ suffix) := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        have hxPre : x ∈ linearCtxDom GammaPre := by
          rw [hPre]
          simp [linearCtxDom, List.append_assoc]
        exact False.elim
          ((noDupNames_middle_fresh_prefix (GammaPre := GammaPre) (GammaPost := suffix)
              (x := x) (t := Typ.unit) hndAny) hxPre)
  · cases m with
    | nil =>
        have hRest' : (x, slot2) :: suffix = (x, slot1) :: Gamma2 := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        exact ⟨by simpa using hPre, hTail.symm, rfl⟩
    | cons hd tl =>
        rcases hd with ⟨z, slotZ⟩
        have hRest' : (x, slot2) :: suffix = (z, slotZ) :: (tl ++ [(x, slot1)] ++ Gamma2) := by
          simpa [List.append_assoc] using hRest
        injection hRest' with hHead hTail
        cases hHead
        have hxSuf : x ∈ linearCtxDom suffix := by
          rw [hTail]
          simp [linearCtxDom, List.append_assoc]
        exact False.elim
          ((noDupNames_middle_fresh_suffix (GammaPre := GammaPre) (GammaPost := suffix)
              (x := x) (t := Typ.unit) hndAny) hxSuf)

private theorem slotSub_names_eq
    {out inp : LinearCtx}
    (h : SlotSub out inp) :
    linearCtxDom out = linearCtxDom inp := by
  induction out generalizing inp with
  | nil =>
      cases inp <;> simp [SlotSub] at h ⊢
  | cons outHd outTl ih =>
      cases inp with
      | nil =>
          cases h
      | cons inpHd inpTl =>
          rcases outHd with ⟨xOut, slotOut⟩
          rcases inpHd with ⟨xIn, slotIn⟩
          rcases h with ⟨hname, _hslot, htail⟩
          simp [linearCtxDom, hname]
          simpa using ih htail

private theorem boundFresh_of_slotSub
    {out inp : LinearCtx} {v : Term}
    (hsub : SlotSub out inp)
    (hFresh : ∀ y, y ∈ linearCtxDom inp → y ∉ boundVars v) :
    ∀ y, y ∈ linearCtxDom out → y ∉ boundVars v := by
  intro y hy
  have hy' : y ∈ linearCtxDom inp := by
    simpa [slotSub_names_eq hsub] using hy
  exact hFresh y hy'

private theorem slotSub_live_middle
    {GammaPre suffix : LinearCtx} {x : String}
    {slotX : Option Typ} {t1 : Typ}
    (hslot : slotX = none ∨ slotX = some t1) :
    SlotSub (GammaPre ++ [(x, slotX)] ++ suffix)
      (GammaPre ++ [(x, some t1)] ++ suffix) := by
  have hmid : SlotSub ([(x, slotX)] : LinearCtx) ([(x, some t1)] : LinearCtx) := by
    simpa [SlotSub] using hslot
  simpa [List.append_assoc] using
    slotSub_append (slotSub_refl GammaPre)
      (slotSub_append hmid (slotSub_refl suffix))

private theorem slotSub_middle_cases
    {GammaPre suffix : LinearCtx} {x : String}
    {slotX : Option Typ} {t1 : Typ}
    (hslot :
      SlotSub (GammaPre ++ [(x, slotX)] ++ suffix)
        (GammaPre ++ [(x, some t1)] ++ suffix))
    (hnd : NoDupNames (GammaPre ++ [(x, slotX)] ++ suffix)) :
    slotX = none ∨ slotX = some t1 := by
  rcases slotSub_split_target
      (GammaPre := GammaPre) (GammaPost := suffix) (x := x) (t := t1) hslot with
    ⟨outPre, slotOut, outPost, hout, _hPreSub, hslotOut, _hPostSub⟩
  have ⟨hPreEq, hPostEq, hslotEq⟩ :=
    split_middle_singleton
      (GammaPre := GammaPre) (suffix := suffix)
      (Gamma1 := outPre) (Gamma2 := outPost)
      (x := x) (slot1 := slotOut) (slot2 := slotX)
      hout.symm hnd
  subst outPre
  subst outPost
  subst slotOut
  exact hslotOut

private theorem noDupNames_remove_middle
    {GammaPre suffix : LinearCtx} {x : String} {slot : Option Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, slot)] ++ suffix)) :
    NoDupNames (GammaPre ++ suffix) := by
  have hsub : List.Sublist (GammaPre ++ suffix) (GammaPre ++ [(x, slot)] ++ suffix) := by
    induction GammaPre with
    | nil =>
        simp
    | cons hd rest ih =>
        simpa [List.append_assoc] using List.Sublist.cons₂ hd ih
  exact noDupNames_of_sublist hsub hnd

private theorem not_mem_linearCtxDom_append_singleton
    {y x : String} {Gamma : LinearCtx} {slot : Option Typ}
    (h : y ∉ linearCtxDom Gamma)
    (hyx : y ≠ x) :
    y ∉ linearCtxDom (Gamma ++ [(x, slot)]) := by
  intro hy
  simp [linearCtxDom, List.map_append, hyx] at hy
  exact h (by simpa [linearCtxDom] using hy)

private theorem not_mem_linearCtxDom_append_pair
    {z x y : String} {Gamma : LinearCtx}
    {slotX slotY : Option Typ}
    (h : z ∉ linearCtxDom Gamma)
    (hzx : z ≠ x)
    (hzy : z ≠ y) :
    z ∉ linearCtxDom (Gamma ++ [(x, slotX), (y, slotY)]) := by
  intro hz
  simp [linearCtxDom, List.map_append, hzx, hzy] at hz
  exact h (by simpa [linearCtxDom] using hz)

mutual

/-- Cutoff-indexed weakening for named contexts. Insert a fresh unused
    binding before the suffix of length `j` on both the input and output
    sides of a typing derivation. -/
private theorem weakening_beforeSuffix
    (Delta : CapCtx) (Sigma : StoreTyp) (y : String) (slot_y : Option Typ)
    {Gamma Gamma' : LinearCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ j,
      j ≤ Gamma.length →
      y ∉ linearCtxDom Gamma →
      freshInTerm y e →
      HasType Delta Sigma
        (insertBeforeSuffix Gamma j (y, slot_y))
        e t eps
        (insertBeforeSuffix Gamma' j (y, slot_y)) := by
  match h with
  | HasType.var Delta Sigma GammaPre GammaPost x tx =>
      intro j hj _hdom hfresh
      have hyx : y ≠ x := freshInTerm_var hfresh
      by_cases hpost : j ≤ GammaPost.length
      · have hin :
            insertBeforeSuffix (GammaPre ++ [(x, some tx)] ++ GammaPost) j (y, slot_y) =
              GammaPre ++ [(x, some tx)] ++ insertBeforeSuffix GammaPost j (y, slot_y) :=
          insertBeforeSuffix_var_right GammaPre GammaPost j (y, slot_y) x (some tx) hpost
        have hout :
            insertBeforeSuffix (GammaPre ++ [(x, none)] ++ GammaPost) j (y, slot_y) =
              GammaPre ++ [(x, none)] ++ insertBeforeSuffix GammaPost j (y, slot_y) :=
          insertBeforeSuffix_var_right GammaPre GammaPost j (y, slot_y) x none hpost
        rw [hin, hout]
        exact HasType.var Delta Sigma GammaPre (insertBeforeSuffix GammaPost j (y, slot_y)) x tx
      · have hpost_lt : GammaPost.length < j := Nat.lt_of_not_ge hpost
        have hin :
            insertBeforeSuffix (GammaPre ++ [(x, some tx)] ++ GammaPost) j (y, slot_y) =
              insertBeforeSuffix GammaPre (j - (GammaPost.length + 1)) (y, slot_y) ++
                [(x, some tx)] ++ GammaPost :=
          insertBeforeSuffix_var_left GammaPre GammaPost j (y, slot_y) x (some tx) hpost_lt hj
        have hout :
            insertBeforeSuffix (GammaPre ++ [(x, none)] ++ GammaPost) j (y, slot_y) =
              insertBeforeSuffix GammaPre (j - (GammaPost.length + 1)) (y, slot_y) ++
                [(x, none)] ++ GammaPost :=
          insertBeforeSuffix_var_left GammaPre GammaPost j (y, slot_y) x none hpost_lt (by
            simpa [List.length_append] using hj)
        rw [hin, hout]
        exact HasType.var Delta Sigma
          (insertBeforeSuffix GammaPre (j - (GammaPost.length + 1)) (y, slot_y))
          GammaPost x tx
  | HasType.unit Delta Sigma Gamma =>
      intro j _hj _hdom _hfresh
      exact HasType.unit Delta Sigma (insertBeforeSuffix Gamma j (y, slot_y))
  | HasType.abs Delta Sigma Gamma1 Gamma2 x t1 t2 epsBody body slot hBody =>
      intro j hj hdom hfresh
      rcases freshInTerm_abs hfresh with ⟨hyx, hfreshBody⟩
      have hdomBody :
          y ∉ linearCtxDom (Gamma1 ++ [(x, some t1)]) :=
        not_mem_linearCtxDom_append_singleton hdom hyx
      have hBody' := weakening_beforeSuffix Delta Sigma y slot_y hBody (j + 1) (by
        simpa [List.length_append] using hj) hdomBody hfreshBody
      have hBody'' :
          HasType Delta Sigma
            (insertBeforeSuffix Gamma1 j (y, slot_y) ++ [(x, some t1)])
            body t2 epsBody
            (insertBeforeSuffix Gamma2 j (y, slot_y) ++ [(x, slot)]) := by
        simpa using hBody'
      simpa using
        (HasType.abs Delta Sigma
          (insertBeforeSuffix Gamma1 j (y, slot_y))
          (insertBeforeSuffix Gamma2 j (y, slot_y))
          x t1 t2 epsBody body slot hBody'')
  | HasType.app Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 epsInner eps1 eps2 h1 h2 =>
      intro j hj hdom hfresh
      rcases freshInTerm_app hfresh with ⟨hfresh1, hfresh2⟩
      have h1' := weakening_beforeSuffix Delta Sigma y slot_y h1 j hj hdom hfresh1
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved h1] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks h1 y hy2)
      have h2' := weakening_beforeSuffix Delta Sigma y slot_y h2 j hj2 hdom2 hfresh2
      exact HasType.app Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        e1 e2 t1 t2 epsInner eps1 eps2 h1' h2'
  | HasType.letBind Delta Sigma Gamma1 Gamma2 Gamma3 x e1 e2 t1 t2 eps1 eps2 slot h1 h2 =>
      intro j hj hdom hfresh
      rcases freshInTerm_letBind hfresh with ⟨hyx, hfresh1, hfresh2⟩
      have h1' := weakening_beforeSuffix Delta Sigma y slot_y h1 j hj hdom hfresh1
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved h1] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks h1 y hy2)
      have hdomBody :
          y ∉ linearCtxDom (Gamma2 ++ [(x, some t1)]) :=
        not_mem_linearCtxDom_append_singleton hdom2 hyx
      have h2' := weakening_beforeSuffix Delta Sigma y slot_y h2 (j + 1) (by
        simpa [List.length_append] using hj2) hdomBody hfresh2
      have h2'' :
          HasType Delta Sigma
            (insertBeforeSuffix Gamma2 j (y, slot_y) ++ [(x, some t1)])
            e2 t2 eps2
            (insertBeforeSuffix Gamma3 j (y, slot_y) ++ [(x, slot)]) := by
        simpa using h2'
      exact HasType.letBind Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        x e1 e2 t1 t2 eps1 eps2 slot h1' h2''
  | HasType.copy Delta Sigma Gamma1 Gamma2 e ds eps hBody =>
      intro j hj hdom hfresh
      exact HasType.copy Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        e ds eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_copy hfresh))
  | HasType.letpair Delta Sigma Gamma1 Gamma2 Gamma3 x z e1 e2 t1 t2 t eps1 eps2 slotX slotY h1 h2 =>
      intro j hj hdom hfresh
      rcases freshInTerm_letpair hfresh with ⟨hyx, hyz, hfresh1, hfresh2⟩
      have h1' := weakening_beforeSuffix Delta Sigma y slot_y h1 j hj hdom hfresh1
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved h1] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks h1 y hy2)
      have hdomBody :
          y ∉ linearCtxDom (Gamma2 ++ [(x, some t1), (z, some t2)]) :=
        not_mem_linearCtxDom_append_pair hdom2 hyx hyz
      have h2' := weakening_beforeSuffix Delta Sigma y slot_y h2 (j + 2) (by
        simpa [List.length_append] using hj2) hdomBody hfresh2
      have h2'' :
          HasType Delta Sigma
            (insertBeforeSuffix Gamma2 j (y, slot_y) ++ [(x, some t1), (z, some t2)])
            e2 t eps2
            (insertBeforeSuffix Gamma3 j (y, slot_y) ++ [(x, slotX), (z, slotY)]) := by
        simpa using h2'
      exact HasType.letpair Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        x z e1 e2 t1 t2 t eps1 eps2 slotX slotY h1' h2''
  | HasType.tpair Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 eps1 eps2 h1 h2 =>
      intro j hj hdom hfresh
      rcases freshInTerm_pair hfresh with ⟨hfresh1, hfresh2⟩
      have h1' := weakening_beforeSuffix Delta Sigma y slot_y h1 j hj hdom hfresh1
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved h1] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks h1 y hy2)
      have h2' := weakening_beforeSuffix Delta Sigma y slot_y h2 j hj2 hdom2 hfresh2
      exact HasType.tpair Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        e1 e2 t1 t2 eps1 eps2 h1' h2'
  | HasType.fst Delta Sigma Gamma1 Gamma2 e t1 t2 eps hBody =>
      intro j hj hdom hfresh
      exact HasType.fst Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        e t1 t2 eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_fst hfresh))
  | HasType.snd Delta Sigma Gamma1 Gamma2 e t1 t2 eps hBody =>
      intro j hj hdom hfresh
      exact HasType.snd Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        e t1 t2 eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_snd hfresh))
  | HasType.const Delta Sigma Gamma v ds =>
      intro j _hj _hdom _hfresh
      exact HasType.const Delta Sigma (insertBeforeSuffix Gamma j (y, slot_y)) v ds
  | HasType.tadd Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 h1 h2 =>
      intro j hj hdom hfresh
      rcases freshInTerm_add hfresh with ⟨hfresh1, hfresh2⟩
      have h1' := weakening_beforeSuffix Delta Sigma y slot_y h1 j hj hdom hfresh1
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved h1] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks h1 y hy2)
      have h2' := weakening_beforeSuffix Delta Sigma y slot_y h2 j hj2 hdom2 hfresh2
      exact HasType.tadd Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        e1 e2 ds eps1 eps2 h1' h2'
  | HasType.tmul Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 h1 h2 =>
      intro j hj hdom hfresh
      rcases freshInTerm_mul hfresh with ⟨hfresh1, hfresh2⟩
      have h1' := weakening_beforeSuffix Delta Sigma y slot_y h1 j hj hdom hfresh1
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved h1] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks h1 y hy2)
      have h2' := weakening_beforeSuffix Delta Sigma y slot_y h2 j hj2 hdom2 hfresh2
      exact HasType.tmul Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        e1 e2 ds eps1 eps2 h1' h2'
  | HasType.tsum Delta Sigma Gamma1 Gamma2 e ds d eps hBody hmem =>
      intro j hj hdom hfresh
      exact HasType.tsum Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        e ds d eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_sum hfresh))
        hmem
  | HasType.texpand Delta Sigma Gamma1 Gamma2 e ds d eps hBody =>
      intro j hj hdom hfresh
      exact HasType.texpand Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        e ds d eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_expand hfresh))
  | HasType.uniformLike Delta Sigma Gamma1 Gamma2 e ds lo hi eps hBody =>
      intro j hj hdom hfresh
      exact HasType.uniformLike Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        e ds lo hi eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_uniformLike hfresh))
  | HasType.perform Delta Sigma Gamma1 Gamma2 op e tArg tRet eps hBody hsig =>
      intro j hj hdom hfresh
      exact HasType.perform Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        op e tArg tRet eps
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom (freshInTerm_perform hfresh))
        hsig
  | HasType.handle Delta Sigma Gamma1 Gamma2 Gamma3 body clauses t epsH epsB
      hBody hOpsIn hClsIn hCover hClauses =>
      intro j hj hdom hfresh
      have hfreshBody : freshInTerm y body := freshInTerm_handle_body hfresh
      have hBody' := weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom hfreshBody
      have hj2 : j ≤ Gamma2.length := by
        simpa [hasType_length_preserved hBody] using hj
      have hdom2 : y ∉ linearCtxDom Gamma2 := by
        intro hy2
        exact hdom (has_type_linear_shrinks hBody y hy2)
      have hClauses' := weakening_beforeSuffix_clauses Delta Sigma y slot_y hClauses j hj2 hdom2 (by
        intro op x k hb hmem
        exact freshInTerm_handle_clause hfresh hmem)
      exact HasType.handle Delta Sigma
        (insertBeforeSuffix Gamma1 j (y, slot_y))
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        body clauses t epsH epsB hBody' hOpsIn hClsIn hCover hClauses'
  | HasType.tgrad Delta Sigma Gamma x ds dsOut body eps slot hBody hsub =>
      intro j hj hdom hfresh
      rcases freshInTerm_grad hfresh with
        ⟨hyx, hfreshBody⟩
      have hdomBody :
          y ∉ linearCtxDom (Gamma ++ [(x, some (Typ.tensor ds))]) :=
        not_mem_linearCtxDom_append_singleton hdom hyx
      have hBody' := weakening_beforeSuffix (Capability.diff :: Delta) Sigma y slot_y hBody (j + 1) (by
        simpa [List.length_append] using hj) hdomBody hfreshBody
      have hBody'' :
          HasType (Capability.diff :: Delta) Sigma
            (insertBeforeSuffix Gamma j (y, slot_y) ++ [(x, some (Typ.tensor ds))])
            body (Typ.tensor dsOut) eps
            (insertBeforeSuffix Gamma j (y, slot_y) ++ [(x, slot)]) := by
        simpa using hBody'
      simpa using
        (HasType.tgrad Delta Sigma
          (insertBeforeSuffix Gamma j (y, slot_y))
          x ds dsOut body eps slot hBody'' hsub)
  | HasType.tvmap Delta Sigma Gamma x t1 t2 body eps d slot hBody =>
      intro j hj hdom hfresh
      rcases freshInTerm_vmap hfresh with
        ⟨hyx, hfreshBody⟩
      have hdomBody :
          y ∉ linearCtxDom (Gamma ++ [(x, some t1)]) :=
        not_mem_linearCtxDom_append_singleton hdom hyx
      have hBody' := weakening_beforeSuffix Delta Sigma y slot_y hBody (j + 1) (by
        simpa [List.length_append] using hj) hdomBody hfreshBody
      have hBody'' :
          HasType Delta Sigma
            (insertBeforeSuffix Gamma j (y, slot_y) ++ [(x, some t1)])
            body t2 eps
            (insertBeforeSuffix Gamma j (y, slot_y) ++ [(x, slot)]) := by
        simpa using hBody'
      simpa using
        (HasType.tvmap Delta Sigma
          (insertBeforeSuffix Gamma j (y, slot_y))
          x t1 t2 body eps d slot hBody'')
  | HasType.loc Delta Sigma Gamma ell t hlook =>
      intro j _hj _hdom _hfresh
      exact HasType.loc Delta Sigma (insertBeforeSuffix Gamma j (y, slot_y)) ell t hlook
  | HasType.subEff Delta Sigma Gamma Gamma' e t eps eps' hBody hsub =>
      intro j hj hdom hfresh
      exact HasType.subEff Delta Sigma
        (insertBeforeSuffix Gamma j (y, slot_y))
        (insertBeforeSuffix Gamma' j (y, slot_y))
        e t eps eps'
        (weakening_beforeSuffix Delta Sigma y slot_y hBody j hj hdom hfresh)
        hsub

private theorem weakening_beforeSuffix_clauses
    (Delta : CapCtx) (Sigma : StoreTyp) (y : String) (slot_y : Option Typ)
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls) :
    ∀ j,
      j ≤ Gamma2.length →
      y ∉ linearCtxDom Gamma2 →
      (∀ op x k hb, (op, x, k, hb) ∈ cls → y ≠ x ∧ y ≠ k ∧ freshInTerm y hb) →
      ClausesTyped Delta Sigma
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        t epsR cls := by
  match h with
  | ClausesTyped.nil Delta Sigma Gamma2 t epsR =>
      intro j _hj _hdom _hfresh
      exact ClausesTyped.nil Delta Sigma (insertBeforeSuffix Gamma2 j (y, slot_y)) t epsR
  | ClausesTyped.cons Delta Sigma Gamma2 Gamma3 t tArg tRet epsR op x k hb rest slotX slotK
      hMatch hBody hRest =>
      intro j hj hdom hfresh
      have hHeadFresh := hfresh op x k hb (by simp)
      rcases hHeadFresh with ⟨hyx, hyk, hfreshBody⟩
      have hdomBody :
          y ∉ linearCtxDom
            (Gamma2 ++ [(x, some tArg), (k, some (Typ.arrow tRet t epsR))]) :=
        not_mem_linearCtxDom_append_pair hdom hyx hyk
      have hBody' := weakening_beforeSuffix Delta Sigma y slot_y hBody (j + 2) (by
        simpa [List.length_append] using hj) hdomBody hfreshBody
      have hBody'' :
          HasType Delta Sigma
            (insertBeforeSuffix Gamma2 j (y, slot_y) ++
              [(x, some tArg), (k, some (Typ.arrow tRet t epsR))])
            hb t epsR
            (insertBeforeSuffix Gamma3 j (y, slot_y) ++ [(x, slotX), (k, slotK)]) := by
        simpa using hBody'
      have hRest' := weakening_beforeSuffix_clauses Delta Sigma y slot_y hRest j hj hdom (by
        intro op' x' k' hb' hmem
        exact hfresh op' x' k' hb' (by simp [hmem]))
      exact ClausesTyped.cons Delta Sigma
        (insertBeforeSuffix Gamma2 j (y, slot_y))
        (insertBeforeSuffix Gamma3 j (y, slot_y))
        t tArg tRet epsR op x k hb rest slotX slotK hMatch hBody'' hRest'

end

/-- Position-indexed weakening with an arbitrary inserted slot.
    This is the tombstone-general form used internally by the honest
    substitution proof. -/
theorem weakening_insert_slot
    (Delta : CapCtx) (Sigma : StoreTyp) (y : String) (slot_y : Option Typ)
    {Gamma Gamma' : LinearCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ Gamma_pre Gamma_post,
      Gamma = Gamma_pre ++ Gamma_post →
      y ∉ linearCtxDom Gamma_pre → y ∉ linearCtxDom Gamma_post →
      freshInTerm y e →
      ∃ Gamma'_pre Gamma'_post,
        Gamma' = Gamma'_pre ++ Gamma'_post ∧
        HasType Delta Sigma (Gamma_pre ++ [(y, slot_y)] ++ Gamma_post) e t eps
                (Gamma'_pre ++ [(y, slot_y)] ++ Gamma'_post) := by
  intro Gamma_pre Gamma_post hsplit hfp hfq h_fresh_term
  let entry : String × Option Typ := (y, slot_y)
  let j := Gamma_post.length
  have hj : j ≤ Gamma.length := by
    rw [hsplit]
    simp [j, List.length_append]
  have hdom : y ∉ linearCtxDom Gamma := by
    rw [hsplit]
    intro hy
    simp [linearCtxDom, List.map_append] at hy
    exact hy.elim (fun h => hfp (by simpa [linearCtxDom] using h))
      (fun h => hfq (by simpa [linearCtxDom] using h))
  have hweak := weakening_beforeSuffix Delta Sigma y slot_y h j hj hdom h_fresh_term
  have hin :
      insertBeforeSuffix Gamma j entry = Gamma_pre ++ [entry] ++ Gamma_post := by
    rw [hsplit]
    simpa [entry, j] using insertBeforeSuffix_eq_split Gamma_pre Gamma_post entry
  let Gamma'_pre := Gamma'.take (Gamma'.length - j)
  let Gamma'_post := Gamma'.drop (Gamma'.length - j)
  have hsplit' : Gamma' = Gamma'_pre ++ Gamma'_post := by
    simp [Gamma'_pre, Gamma'_post, List.take_append_drop]
  have hj' : j ≤ Gamma'.length := by
    have hlen : Gamma.length = Gamma'.length := hasType_length_preserved h
    rw [← hlen]
    exact hj
  have hpostlen : Gamma'_post.length = j := by
    simp [Gamma'_post, Nat.sub_sub_self hj']
  have hout :
      insertBeforeSuffix Gamma' j entry = Gamma'_pre ++ [entry] ++ Gamma'_post := by
    rw [hsplit']
    simpa [entry, hpostlen] using insertBeforeSuffix_eq_split Gamma'_pre Gamma'_post entry
  refine ⟨Gamma'_pre, Gamma'_post, hsplit', ?_⟩
  rw [hin, hout] at hweak
  simpa [entry] using hweak

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
  intro Gamma_pre Gamma_post hsplit hfp hfq h_fresh_term
  simpa using
    (weakening_insert_slot Delta Sigma y (some t_y) h
      Gamma_pre Gamma_post hsplit hfp hfq h_fresh_term)


/-- Tail weakening with an arbitrary inserted slot. -/
theorem weakening_tail_slot
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (y : String) (slot_y : Option Typ) (t : Typ) (eps : EffectRow) (e : Term)
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (h_fresh : y ∉ linearCtxDom Gamma)
    (h_fresh_term : freshInTerm y e) :
    ∃ Gamma'_pre Gamma'_post : LinearCtx,
      Gamma' = Gamma'_pre ++ Gamma'_post ∧
      HasType Delta Sigma (Gamma ++ [(y, slot_y)]) e t eps
              (Gamma'_pre ++ [(y, slot_y)] ++ Gamma'_post) := by
  have hfp : y ∉ linearCtxDom ([] : LinearCtx) := by simp [linearCtxDom]
  have hsplit : Gamma = Gamma ++ ([] : LinearCtx) := by simp
  have hres :=
    weakening_insert_slot Delta Sigma y slot_y h Gamma [] hsplit h_fresh hfp h_fresh_term
  simpa using hres

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
  simpa using
    (weakening_tail_slot Delta Sigma Gamma Gamma' y (some t') t eps e h h_fresh h_fresh_term)

private theorem freshInTerm_of_closed_notBound
    {v : Term} {y : String}
    (hClosed : Closed v)
    (hBound : y ∉ boundVars v) :
    freshInTerm y v := by
  refine ⟨?_, hBound⟩
  unfold Closed at hClosed
  intro hy
  rw [hClosed] at hy
  simp at hy

private theorem closed_typed_suffix_weaken
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma : LinearCtx} {v : Term} {t : Typ} {suffix : LinearCtx}
    (h : HasType Delta Sigma Gamma v t [] Gamma)
    (hClosed : Closed v)
    (hnd : NoDupNames (Gamma ++ suffix))
    (hBoundFresh : ∀ y, y ∈ linearCtxDom suffix → y ∉ boundVars v) :
    HasType Delta Sigma (Gamma ++ suffix) v t [] (Gamma ++ suffix) := by
  induction suffix generalizing Gamma with
  | nil =>
      simpa using h
  | cons hd rest ih =>
      rcases hd with ⟨y, slot_y⟩
      have hndFull : NoDupNames (Gamma ++ [(y, slot_y)] ++ rest) := by
        simpa [List.append_assoc] using hnd
      have hndFresh :
          NoDupNames (Gamma ++ [(y, some Typ.unit)] ++ rest) := by
        simpa [NoDupNames, linearCtxDom] using hndFull
      have hyGamma : y ∉ linearCtxDom Gamma :=
        noDupNames_middle_fresh_prefix (GammaPre := Gamma) (GammaPost := rest)
          (x := y) (t := Typ.unit) hndFresh
      have hyBound : y ∉ boundVars v := by
        exact hBoundFresh y (by simp [linearCtxDom])
      have hFresh : freshInTerm y v :=
        freshInTerm_of_closed_notBound hClosed hyBound
      have hWeak0 := weakening_beforeSuffix Delta Sigma y slot_y h 0 (by simp) hyGamma hFresh
      have hWeak :
          HasType Delta Sigma (Gamma ++ [(y, slot_y)]) v t [] (Gamma ++ [(y, slot_y)]) := by
        simpa using hWeak0
      have hndRest : NoDupNames ((Gamma ++ [(y, slot_y)]) ++ rest) := by
        simpa [List.append_assoc] using hnd
      have hBoundRest :
          ∀ z, z ∈ linearCtxDom rest → z ∉ boundVars v := by
        intro z hz
        have hzRest : ∃ slot, (z, slot) ∈ rest := by
          simpa [linearCtxDom] using hz
        exact hBoundFresh z (by simp [linearCtxDom, hzRest])
      simpa [List.append_assoc] using
        (ih (Gamma := Gamma ++ [(y, slot_y)]) hWeak hndRest hBoundRest)

private theorem closed_typed_prefix_suffix_weaken
    {Delta : CapCtx} {Sigma : StoreTyp}
    {GammaPre suffix : LinearCtx} {v : Term} {t : Typ}
    (h : HasType Delta Sigma [] v t [] [])
    (hClosed : Closed v)
    (hnd : NoDupNames (GammaPre ++ suffix))
    (hBoundFresh : ∀ y, y ∈ linearCtxDom suffix → y ∉ boundVars v) :
    HasType Delta Sigma (GammaPre ++ suffix) v t [] (GammaPre ++ suffix) := by
  have hPrefix : HasType Delta Sigma GammaPre v t [] GammaPre := by
    simpa using hasType_prefix_weaken h GammaPre
  exact closed_typed_suffix_weaken hPrefix hClosed hnd hBoundFresh

/-- Structured output witness for the context-fresh substitution core.
    The original output still contains the distinguished middle `x`
    slot, while the substituted output is the canonical removal of that
    slot from the split output context. -/
private def CtxFreshSubstResult
    (Delta : CapCtx) (Sigma : StoreTyp)
    (x : String) (t1 : Typ)
    (GammaPre suffix GammaOut : LinearCtx)
    (eSub : Term) (t2 : Typ) (eps : EffectRow) : Prop :=
  ∃ GammaOutPre slotOut GammaOutPost,
    GammaOut = GammaOutPre ++ [(x, slotOut)] ++ GammaOutPost ∧
    SlotSub GammaOutPre GammaPre ∧
    (slotOut = none ∨ slotOut = some t1) ∧
    SlotSub GammaOutPost suffix ∧
    HasType Delta Sigma (GammaPre ++ suffix) eSub t2 eps
      (GammaOutPre ++ GammaOutPost)

private theorem ctxFreshSubstResult_map
    (Delta : CapCtx) (Sigma : StoreTyp)
    (x : String) (t1 : Typ)
    {GammaPre suffix GammaOut : LinearCtx}
    {eSub : Term} {tIn tOut : Typ} {epsIn epsOut : EffectRow}
    (hRes : CtxFreshSubstResult
      Delta Sigma x t1 GammaPre suffix GammaOut eSub tIn epsIn)
    (wrapTerm : Term → Term)
    (lift :
      ∀ {GammaIn GammaOut' : LinearCtx},
        HasType Delta Sigma GammaIn eSub tIn epsIn GammaOut' →
        HasType Delta Sigma GammaIn (wrapTerm eSub) tOut epsOut GammaOut') :
    CtxFreshSubstResult
      Delta Sigma x t1 GammaPre suffix GammaOut (wrapTerm eSub) tOut epsOut := by
  rcases hRes with
    ⟨GammaOutPre, slotOut, GammaOutPost, hOut, hPre, hSlot, hPost, hTy⟩
  refine ⟨GammaOutPre, slotOut, GammaOutPost, hOut, hPre, hSlot, hPost, ?_⟩
  exact lift hTy

private theorem subst_preserves_typing_ctx_fresh_var
    (Delta : CapCtx) (Sigma : StoreTyp) (x : String) (t1 : Typ)
    {GammaPre suffix : LinearCtx} {slotX : Option Typ}
    {Gamma1 Gamma2 : LinearCtx} {y : String} {t : Typ} {v : Term}
    (hEq : Gamma1 ++ [(y, some t)] ++ Gamma2 = GammaPre ++ [(x, slotX)] ++ suffix)
    (h_live : SlotSub
      (GammaPre ++ [(x, slotX)] ++ suffix)
      (GammaPre ++ [(x, some t1)] ++ suffix))
    (hnd : NoDupNames (GammaPre ++ [(x, slotX)] ++ suffix))
    (h_v : HasType Delta Sigma [] v t1 [] [])
    (hClosed : Closed v)
    (hSuffixFresh : ∀ z, z ∈ linearCtxDom suffix → z ∉ boundVars v) :
    CtxFreshSubstResult Delta Sigma x t1
      GammaPre suffix (Gamma1 ++ [(y, none)] ++ Gamma2)
      (subst (Term.var y) v x) t [] := by
  by_cases hxy : y = x
  · subst y
    have ⟨hGamma1, hGamma2, hslotX⟩ :=
      split_target_middle_singleton hEq hnd
    subst Gamma1
    subst Gamma2
    subst slotX
    have hslotLive : some t = none ∨ some t = some t1 :=
      slotSub_middle_cases
        (GammaPre := GammaPre) (suffix := suffix)
        (x := x) (slotX := some t) (t1 := t1)
        h_live hnd
    rcases hslotLive with hslotDead | hslotLive
    · cases hslotDead
    · injection hslotLive with ht
      subst t
      have hnd' : NoDupNames (GammaPre ++ suffix) :=
        noDupNames_remove_middle hnd
      refine ⟨GammaPre, none, suffix, ?_, slotSub_refl GammaPre, Or.inl rfl,
        slotSub_refl suffix, ?_⟩
      · simp [List.append_assoc]
      · simpa [subst] using
          (closed_typed_prefix_suffix_weaken
            (GammaPre := GammaPre) (suffix := suffix)
            h_v hClosed hnd' hSuffixFresh)
  · have hslotXLive : slotX = none ∨ slotX = some t1 :=
      slotSub_middle_cases
        (GammaPre := GammaPre) (suffix := suffix)
        (x := x) (slotX := slotX) (t1 := t1)
        h_live hnd
    rcases list_append_eq_split Gamma1 ([(y, some t)] ++ Gamma2)
        GammaPre ([(x, slotX)] ++ suffix)
        (by simpa [List.append_assoc] using hEq) with
      ⟨m, hPre, hRest⟩ | ⟨m, hPre, hRest⟩
    · cases m with
      | nil =>
          have hRest' : (y, some t) :: Gamma2 = (x, slotX) :: suffix := by
            simpa [List.append_assoc] using hRest
          cases hRest'
          exact False.elim (hxy rfl)
      | cons hd tl =>
          rcases hd with ⟨z, slotZ⟩
          have hRest' : (y, some t) :: Gamma2 = (z, slotZ) :: (tl ++ [(x, slotX)] ++ suffix) := by
            simpa [List.append_assoc] using hRest
          injection hRest' with hHead hTail
          cases hHead
          refine ⟨Gamma1 ++ [(y, none)] ++ tl, slotX, suffix, ?_, ?_,
            hslotXLive, slotSub_refl suffix, ?_⟩
          · simpa [hTail, List.append_assoc]
          · have hpreSub :
              SlotSub (Gamma1 ++ [(y, none)] ++ tl) (Gamma1 ++ [(y, some t)] ++ tl) := by
              have hmid : SlotSub ([(y, none)] : LinearCtx) ([(y, some t)] : LinearCtx) := by
                simp [SlotSub]
              simpa [List.append_assoc] using
                slotSub_append (slotSub_refl Gamma1)
                  (slotSub_append hmid (slotSub_refl tl))
            simpa [hPre, List.append_assoc] using hpreSub
          · have hVar :
              HasType Delta Sigma
                (Gamma1 ++ [(y, some t)] ++ tl ++ suffix)
                (Term.var y) t [] (Gamma1 ++ [(y, none)] ++ tl ++ suffix) := by
              simpa [List.append_assoc] using
                (HasType.var Delta Sigma Gamma1 (tl ++ suffix) y t)
            simpa [subst, hxy, hPre, hTail, List.append_assoc] using hVar
    · cases m with
      | nil =>
          have hRest' : (x, slotX) :: suffix = (y, some t) :: Gamma2 := by
            simpa [List.append_assoc] using hRest
          cases hRest'
          exact False.elim (hxy rfl)
      | cons hd tl =>
          rcases hd with ⟨z, slotZ⟩
          have hRest' : (x, slotX) :: suffix = (z, slotZ) :: (tl ++ [(y, some t)] ++ Gamma2) := by
            simpa [List.append_assoc] using hRest
          injection hRest' with hHead hTail
          cases hHead
          refine ⟨GammaPre, slotX, tl ++ [(y, none)] ++ Gamma2, ?_, slotSub_refl GammaPre,
            hslotXLive, ?_, ?_⟩
          · simpa [hPre, List.append_assoc]
          · have hpostSub :
              SlotSub (tl ++ [(y, none)] ++ Gamma2) (tl ++ [(y, some t)] ++ Gamma2) := by
              have hmid : SlotSub ([(y, none)] : LinearCtx) ([(y, some t)] : LinearCtx) := by
                simp [SlotSub]
              simpa [List.append_assoc] using
                slotSub_append (slotSub_refl tl)
                  (slotSub_append hmid (slotSub_refl Gamma2))
            simpa [hTail, List.append_assoc] using hpostSub
          · have hVar :
              HasType Delta Sigma
                (GammaPre ++ tl ++ [(y, some t)] ++ Gamma2)
                (Term.var y) t [] (GammaPre ++ tl ++ [(y, none)] ++ Gamma2) := by
              simpa [List.append_assoc] using
                (HasType.var Delta Sigma (GammaPre ++ tl) Gamma2 y t)
            simpa [subst, hxy, hTail, List.append_assoc] using hVar

-- NOTE: `exchange_tail` removed entirely. Its original statement
-- (rigid output context across an adjacent swap) is provably false
-- in this type system: `var` consumes the tail binding and the two
-- swap orderings produce different output contexts. No caller
-- relies on `exchange_tail`; substitution is closed via
-- `weakening_insert` directly.

/-- Honest named substitution core for the non-captured context-fresh
    case.

    The real recursive shape is not just "tail binding plus closed
    payload". Binder cases must recurse under an arbitrary suffix after
    the distinguished name, and the current middle slot must stay on the
    canonical `none | some t1` trajectory so recursive second premises
    can split it with `slotSub_split_target`. The public
    `subst_preserves_typing_ctx_fresh` theorem below is the `suffix = []`
    specialization of this core statement. -/
private theorem subst_preserves_typing_ctx_fresh_aux
    (Delta : CapCtx) (Sigma : StoreTyp)
    (x : String) (t1 : Typ)
    {GammaPre suffix GammaOut : LinearCtx} {slotX : Option Typ}
    {e v : Term} {t2 : Typ} {eps : EffectRow}
    (h_e : HasType Delta Sigma
      (GammaPre ++ [(x, slotX)] ++ suffix) e t2 eps GammaOut)
    (h_live : SlotSub
      (GammaPre ++ [(x, slotX)] ++ suffix)
      (GammaPre ++ [(x, some t1)] ++ suffix))
    (h_nodup : NoDupNames (GammaPre ++ [(x, slotX)] ++ suffix))
    (h_ctx_fresh : ∀ y, y ∈ linearCtxDom (GammaPre ++ [(x, slotX)] ++ suffix) → y ∉ boundVars e)
    (h_v : HasType Delta Sigma [] v t1 [] [])
    (h_closed : Closed v)
    (h_suffix_fresh : ∀ y, y ∈ linearCtxDom suffix → y ∉ boundVars v)
    (h_bound_fresh : ∀ y, y ∈ boundVars e → y ∉ boundVars v) :
    CtxFreshSubstResult Delta Sigma x t1
      GammaPre suffix GammaOut (subst e v x) t2 eps := by
  have hmain :
      ∀ {Gamma : LinearCtx} {e : Term} {t2 : Typ} {eps : EffectRow} {GammaOut : LinearCtx},
        HasType Delta Sigma Gamma e t2 eps GammaOut →
        ∀ {GammaPre suffix : LinearCtx} {slotX : Option Typ},
          Gamma = GammaPre ++ [(x, slotX)] ++ suffix →
          SlotSub Gamma (GammaPre ++ [(x, some t1)] ++ suffix) →
          NoDupNames Gamma →
          (∀ y, y ∈ linearCtxDom Gamma → y ∉ boundVars e) →
          (∀ y, y ∈ linearCtxDom suffix → y ∉ boundVars v) →
          (∀ y, y ∈ boundVars e → y ∉ boundVars v) →
          CtxFreshSubstResult Delta Sigma x t1
            GammaPre suffix GammaOut (subst e v x) t2 eps := by
    intro Gamma e t2 eps GammaOut h
    induction h using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
    | var Delta Sigma Gamma1 Gamma2 y t =>
        intro GammaPre suffix slotX hEq h_live h_nodup _h_ctx_fresh h_suffix_fresh _h_bound_fresh
        exact subst_preserves_typing_ctx_fresh_var
          Delta Sigma x t1 hEq
          (by simpa [hEq] using h_live)
          (by simpa [hEq] using h_nodup)
          h_v h_closed h_suffix_fresh
    | unit Delta Sigma Gamma =>
        intro GammaPre suffix slotX hEq h_live h_nodup _h_ctx_fresh _h_suffix_fresh _h_bound_fresh
        subst Gamma
        refine ⟨GammaPre, slotX, suffix, ?_, slotSub_refl GammaPre, ?_, slotSub_refl suffix, ?_⟩
        · simp [List.append_assoc]
        · exact slotSub_middle_cases h_live h_nodup
        · simpa [subst] using HasType.unit Delta Sigma (GammaPre ++ suffix)
    | abs Delta Sigma Gamma1 Gamma2 y tArg tRes epsBody body slot hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | app Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 tArg tRes epsInner eps1 eps2 h1 h2 ih1 ih2 =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | letBind Delta Sigma Gamma1 Gamma2 Gamma3 y e1 e2 tArg tRes eps1 eps2 slot h1 h2 ih1 ih2 =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | copy Delta Sigma Gamma1 Gamma2 body ds eps hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.copy Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            (subst body v x) ds eps hTy)
    | letpair Delta Sigma Gamma1 Gamma2 Gamma3 y z e1 e2 tArg tRes t eps1 eps2 slotY slotZ h1 h2 ih1 ih2 =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | tpair Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 tLeft tRight eps1 eps2 h1 h2 ih1 ih2 =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | fst Delta Sigma Gamma1 Gamma2 body tLeft tRight eps hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.fst Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            (subst body v x) tLeft tRight eps hTy)
    | snd Delta Sigma Gamma1 Gamma2 body tLeft tRight eps hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.snd Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            (subst body v x) tLeft tRight eps hTy)
    | const Delta Sigma Gamma c ds =>
        intro GammaPre suffix slotX hEq h_live h_nodup _h_ctx_fresh _h_suffix_fresh _h_bound_fresh
        subst Gamma
        refine ⟨GammaPre, slotX, suffix, ?_, slotSub_refl GammaPre, ?_, slotSub_refl suffix, ?_⟩
        · simp [List.append_assoc]
        · exact slotSub_middle_cases h_live h_nodup
        · simpa [subst] using HasType.const Delta Sigma (GammaPre ++ suffix) c ds
    | tadd Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | tmul Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | tsum Delta Sigma Gamma1 Gamma2 body ds d eps hBody hmem ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.tsum Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            (subst body v x) ds d eps hTy hmem)
    | texpand Delta Sigma Gamma1 Gamma2 body ds d eps hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.texpand Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            (subst body v x) ds d eps hTy)
    | uniformLike Delta Sigma Gamma1 Gamma2 body ds lo hi eps hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.uniformLike Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            (subst body v x) ds lo hi eps hTy)
    | perform Delta Sigma Gamma1 Gamma2 op body tArg tRet eps hBody hsig ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hCtxBody : ∀ y, y ∈ linearCtxDom Gamma1 → y ∉ boundVars body := by
          intro y hy
          simpa [boundVars] using h_ctx_fresh y hy
        have hBoundBody : ∀ y, y ∈ boundVars body → y ∉ boundVars v := by
          intro y hy
          simpa [boundVars] using h_bound_fresh y hy
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup hCtxBody h_suffix_fresh hBoundBody
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        simpa [subst] using
          (HasType.perform Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
            op (subst body v x) tArg tRet eps hTy hsig)
    | handle Delta Sigma Gamma1 Gamma2 Gamma3 body clauses t epsH epsB
        hBody hOpsIn hClsIn hCover hClauses ihBody ihClauses =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | tgrad Delta Sigma Gamma y ds dsOut body eps slot hBody hsub ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | tvmap Delta Sigma Gamma y tArg tRes body eps d slot hBody ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        sorry
    | loc Delta Sigma Gamma ell t hlook =>
        intro GammaPre suffix slotX hEq h_live h_nodup _h_ctx_fresh _h_suffix_fresh _h_bound_fresh
        subst Gamma
        refine ⟨GammaPre, slotX, suffix, ?_, slotSub_refl GammaPre, ?_, slotSub_refl suffix, ?_⟩
        · simp [List.append_assoc]
        · exact slotSub_middle_cases h_live h_nodup
        · simpa [subst] using HasType.loc Delta Sigma (GammaPre ++ suffix) ell t hlook
    | subEff Delta Sigma Gamma Gamma' body t eps eps' hBody hsub ih =>
        intro GammaPre suffix slotX hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        have hBodySubst :=
          ih h_e h_v hEq h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh
        rcases hBodySubst with
          ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, hTy⟩
        refine ⟨outPre, slotOut, outPost, hOut, hPre, hSlot, hPost, ?_⟩
        exact HasType.subEff Delta Sigma (GammaPre ++ suffix) (outPre ++ outPost)
          (subst body v x) t eps eps' hTy hsub
    | nil Delta Sigma Gamma2 t epsR =>
        trivial
    | cons Delta Sigma Gamma2 Gamma3 t tArg tRet epsR op x k hb rest slotX slotK
        hMatch hBody hRest ihBody ihRest =>
        trivial
  exact hmain h_e rfl h_live h_nodup h_ctx_fresh h_suffix_fresh h_bound_fresh

/-- Honest named substitution theorem for the non-captured context-fresh
    case.

    This is the stable public specialization used by the non-captured
    TranslationDB wrappers: the target binding sits at the tail of the
    input context, the input names are pairwise distinct, and no context
    name is shadowed by a binder already present in `e`. The substituted
    payload is the closed runtime term, so its typing witness is the
    closed-input form `HasType ... [] v ... []`. -/
theorem subst_preserves_typing_ctx_fresh
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
    (x : String) (t1 t2 : Typ) (eps : EffectRow)
    (e v : Term)
    (h_e : HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) e t2 eps Gamma2)
    (h_nodup : NoDupNames (Gamma1 ++ [(x, some t1)]))
    (h_ctx_fresh : ∀ y, y ∈ linearCtxDom (Gamma1 ++ [(x, some t1)]) → y ∉ boundVars e)
    (h_v : HasType Delta Sigma [] v t1 [] [])
    (h_closed : Closed v)
    (h_bound_fresh : ∀ y, y ∈ boundVars e → y ∉ boundVars v) :
    ∃ Gamma2' : LinearCtx,
      HasType Delta Sigma Gamma1 (subst e v x) t2 eps Gamma2' := by
  rcases
      (subst_preserves_typing_ctx_fresh_aux
        Delta Sigma x t1
        (GammaPre := Gamma1) (suffix := ([] : LinearCtx)) (GammaOut := Gamma2)
        (slotX := some t1) (e := e) (v := v) (t2 := t2) (eps := eps)
        (by simpa using h_e)
        (slotSub_refl (Gamma1 ++ [(x, some t1)] ++ ([] : LinearCtx)))
        (by simpa using h_nodup)
        (by simpa using h_ctx_fresh)
        h_v h_closed
        (by
          intro y hy
          simp [linearCtxDom] at hy)
        h_bound_fresh) with
    ⟨outPre, _slotOut, outPost, _hout, _hpre, _hslot, _hpost, hSubst⟩
  exact ⟨outPre ++ outPost, by simpa using hSubst⟩

/-! ## Main theorem -/

/-- Lexical wrapper around `subst_preserves_typing_ctx_fresh`.

    The first substitution step in the beta/let/direct-handler wrappers
    naturally arrives with a `LexicallyScoped` premise; this theorem
    simply projects the `NoDupNames` and context-binder freshness facts
    needed by the core context-fresh theorem. -/
theorem subst_preserves_typing_lexical
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
    (x : String) (t1 t2 : Typ) (eps : EffectRow)
    (e v : Term)
    (h_e : HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) e t2 eps Gamma2)
    (h_lex : LexicallyScoped (Gamma1 ++ [(x, some t1)]) e)
    (h_v : HasType Delta Sigma [] v t1 [] [])
    (h_closed : Closed v)
    (h_bound_fresh : ∀ y, y ∈ boundVars e → y ∉ boundVars v) :
    ∃ Gamma2' : LinearCtx,
      HasType Delta Sigma Gamma1 (subst e v x) t2 eps Gamma2' := by
  rcases h_lex with ⟨h_nodup, h_ctx_fresh, _hws⟩
  exact subst_preserves_typing_ctx_fresh
    Delta Sigma Gamma1 Gamma2 x t1 t2 eps e v
    h_e h_nodup h_ctx_fresh h_v h_closed h_bound_fresh

/-- Substitution preserves typing (tombstone semantics).

    Under the tombstone refactor, the input context carries
    `(x, some t1)` and the output context is whatever HasType produces
    (with `x`'s slot potentially tombstoned to `none`). Operationally,
    the substituted term is the closed runtime payload, so the honest
    theorem uses the closed-input typing witness `HasType ... [] v ...
    []`. The `Closed v` premise makes the naive capture-unaware `subst`
    sound.

    Remaining blocker: this more permissive statement is only still
    needed by the captured-continuation preservation cases, where the
    inserted continuation term carries dormant handler-clause binders
    from the surrounding context. The context-fresh theorem above is the
    honest route for the non-captured cases. -/
theorem subst_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
    (x : String) (t1 t2 : Typ) (eps : EffectRow)
    (e v : Term)
    (_h_e : HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) e t2 eps Gamma2)
    (_h_v : HasType Delta Sigma [] v t1 [] [])
    (_h_closed : Closed v) :
    ∃ Gamma2' : LinearCtx,
      HasType Delta Sigma Gamma1 (subst e v x) t2 eps Gamma2' := by
  sorry -- Remaining blocker is theorem shape, not weakening.
        -- The old unrestricted named statement is false under shadowing,
        -- and the honest proof needs an explicit lexical/binder-freshness
        -- invariant tying `e`'s binders to the substituted value `v`.

end LaCaDiLE
