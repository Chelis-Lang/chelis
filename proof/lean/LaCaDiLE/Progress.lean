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
import LaCaDiLE.Preservation

namespace LaCaDiLE

/-! ## Helper lemmas used by Progress -/

-- DomSub, has_type_linear_shrinks, value_preserves_closed_context
-- moved to Typing.lean so both Progress and Preservation can use them.

/-- Store typing well-formedness: every location maps to a tensor
    type. Holds for every Sigma reachable from an initially-empty
    store via the Preservation cases. -/
def StoreTypTensorOnly (Sigma : StoreTyp) : Prop :=
  ∀ ell t, storeTypLookup Sigma ell = some t → ∃ ds, t = Typ.tensor ds

/-- Canonical forms: a value of arrow type is a literal abstraction.
    Uses StoreTypTensorOnly to rule out loc-of-arrow pathology. -/
theorem canonical_forms_arrow
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t1 t2 : Typ} {eps eps' : EffectRow}
    (h_wf : StoreTypTensorOnly Sigma)
    (h : HasType Delta Sigma Gamma v (Typ.arrow t1 t2 eps) eps' Gamma')
    (hv : IsValue v) :
    ∃ x body, v = Term.abs x t1 body := by
  cases hv with
  | abs x ta e =>
      -- HasType.abs gives output type `arrow ta t2' eps'` for some
      -- ta, t2', eps' — and we have ht : this = arrow t1 t2 eps, so
      -- ta = t1. Return ⟨x, e, rfl⟩.
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx)
          (x0 : String) (ta0 : Typ) (body0 : Term),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.abs x0 ta0 body0 →
          t' = Typ.arrow t1 t2 eps → ta0 = t1 from by
        have := hf _ _ _ _ _ _ _ x ta e h rfl rfl
        exact ⟨x, e, by rw [this]⟩
      intro Δ S Γ e' t' ε Γ' x0 ta0 body0 hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | abs _ _ _ _ _ _ _ _ _ _ _ =>
          intro heq ht
          cases heq
          cases ht
          rfl
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro
  | loc ell =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx) (ell2 : Loc),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.loc ell2 →
          t' = Typ.arrow t1 t2 eps → S = Sigma → False from
        hf _ _ _ _ _ _ _ _ h rfl rfl rfl
      intro Δ S Γ e' t' ε Γ' ell2 hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | loc _ _ _ ell_m t_m hlook_m =>
          intro _ ht hS
          subst hS
          obtain ⟨ds, hds⟩ := h_wf ell_m t_m hlook_m
          subst hds
          cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht hS; exact ih he ht hS
      | _ => first | (intro he _ _; cases he) | exact True.intro
  | unit =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.unit →
          t' = Typ.arrow t1 t2 eps → False from
        hf _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | unit _ _ _ => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro
  | pair _ _ _ _ =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx)
          (a b : Term),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.pair a b →
          t' = Typ.arrow t1 t2 eps → False from
        hf _ _ _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' a b hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro

/-- Canonical forms: a value of tensor type is a runtime location.
    Wave 2: case-split on `IsValue` then derive a contradiction from
    HasType for each non-loc value shape. The type-mismatch between
    (arrow/unit/pair) and tensor is the contradiction. -/
theorem canonical_forms_tensor
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {ds : DimList} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma v (Typ.tensor ds) eps Gamma')
    (hv : IsValue v) : ∃ ell, v = Term.loc ell := by
  cases hv with
  | loc ell => exact ⟨ell, rfl⟩
  | unit =>
      -- HasType.unit produces Typ.unit; unit ≠ tensor ds.
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.unit →
          t' = Typ.tensor ds → False from
        hf _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | unit => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro
  | abs _ _ _ =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx)
          (x' : String) (t1' : Typ) (e'' : Term),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.abs x' t1' e'' →
          t' = Typ.tensor ds → False from
        hf _ _ _ _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' x' t1' e'' hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | abs _ _ _ _ _ _ _ _ _ _ _ => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro
  | pair _ _ _ _ =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx)
          (a b : Term),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.pair a b →
          t' = Typ.tensor ds → False from
        hf _ _ _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' a b hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro

/-- Canonical forms: a value of pair type is a literal pair of values.
    Uses `StoreTypTensorOnly` to rule out loc-of-pair pathology. -/
theorem canonical_forms_pair
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t1 t2 : Typ} {eps : EffectRow}
    (h_wf : StoreTypTensorOnly Sigma)
    (h : HasType Delta Sigma Gamma v (Typ.pair t1 t2) eps Gamma')
    (hv : IsValue v) :
    ∃ v1 v2, v = Term.pair v1 v2 ∧ IsValue v1 ∧ IsValue v2 := by
  cases hv with
  | pair v1 v2 hv1 hv2 => exact ⟨v1, v2, rfl, hv1, hv2⟩
  | loc ell =>
      -- StoreTypTensorOnly says Sigma maps ell to a tensor, not pair.
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx) (ell2 : Loc),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.loc ell2 →
          t' = Typ.pair t1 t2 → S = Sigma → False from
        hf _ _ _ _ _ _ _ _ h rfl rfl rfl
      intro Δ S Γ e' t' ε Γ' ell2 hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | loc _ _ _ ell_m t_m hlook_m =>
          intro _ ht hS
          subst hS
          obtain ⟨ds, hds⟩ := h_wf ell_m t_m hlook_m
          subst hds
          cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih =>
          intro he ht hS; exact ih he ht hS
      | _ => first | (intro he _ _; cases he) | exact True.intro
  | abs _ _ _ =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx)
          (x' : String) (t1' : Typ) (e'' : Term),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.abs x' t1' e'' →
          t' = Typ.pair t1 t2 → False from
        hf _ _ _ _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' x' t1' e'' hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | abs _ _ _ _ _ _ _ _ _ _ _ => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro
  | unit =>
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ : LinearCtx)
          (e' : Term) (t' : Typ) (ε : EffectRow) (Γ' : LinearCtx),
          HasType Δ S Γ e' t' ε Γ' → e' = Term.unit →
          t' = Typ.pair t1 t2 → False from
        hf _ _ _ _ _ _ _ h rfl rfl
      intro Δ S Γ e' t' ε Γ' hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | unit _ _ _ => intro _ ht; cases ht
      | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he ht; exact ih he ht
      | _ => first | (intro he _; cases he) | exact True.intro

/-! ## Local inversion lemmas for Wave 3 progress sub-cases -/

/-- App inversion: strip subEff, recover sub-derivations. -/
theorem HasType.app_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.app e1 e2) t eps Gamma3) :
    ∃ Gamma2 t1 eps1 eps2 eps_inner,
      HasType Delta Sigma Gamma1 e1 (Typ.arrow t1 t eps_inner) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t1 eps2 Gamma3 := by
  generalize heq : Term.app e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | app _ _ _ Γ2 _ _ _ t1 _ eps_inner eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, t1, eps1, eps2, eps_inner, h1, h2⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _h_sub h_sub ih =>
      obtain ⟨Γ2, t1, eps1, eps2, eps_inner, hh1, hh2⟩ := ih heq
      exact ⟨Γ2, t1, eps1, eps2, eps_inner, hh1, hh2⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetBind inversion. -/
theorem HasType.letBind_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps Gamma_out) :
    ∃ Gamma2 Gamma3 t1 eps1 eps2,
      Gamma_out = Gamma3.filter (fun p => p.1 ≠ x) ∧
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, t1)]) e2 t eps2 Gamma3 := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Γ2 Γ3 _ _ _ t1 _ eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, eps1, eps2, rfl, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetPair inversion. -/
theorem HasType.letpair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps Gamma_out) :
    ∃ Gamma2 Gamma3 t1 t2 eps1 eps2,
      Gamma_out = Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y) ∧
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, t1), (y, t2)]) e2 t eps2 Gamma3 := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1' t2' _ eps1' eps2' h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1', t2', eps1', eps2', rfl, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- A location whose store-typing lookup succeeds is live in the store
    (via `StoreWf`). Used to produce the `TensorVal` witness that head
    reductions like `tadd`, `tmul`, `tsum`, etc. require. -/
theorem storeWf_lookup_witness
    {sigma : Store} {Sigma : StoreTyp}
    (h_wf : StoreWf sigma Sigma) {ell : Loc} {t : Typ}
    (h : storeTypLookup Sigma ell = some t) :
    ∃ w, storeLookup sigma ell = some w := by
  -- storeTypLookup = some t implies ell ∈ storeTypDom Sigma.
  have hmem : ell ∈ storeTypDom Sigma := by
    unfold storeTypLookup at h
    rcases hfind : Sigma.find? (fun p => p.1 = ell) with _ | ⟨ell', t'⟩
    · rw [hfind] at h; simp at h
    · have hmem_pair : (ell', t') ∈ Sigma := List.mem_of_find?_eq_some hfind
      -- The `find?` predicate holds on the found element, so ell' = ell.
      have hpred : decide (ell' = ell) = true := by
        have hp := @List.find?_some _ (fun p : Loc × Typ => decide (p.1 = ell))
                     (ell', t') Sigma hfind
        simpa using hp
      have hell_eq : ell' = ell := of_decide_eq_true hpred
      exact List.mem_map.mpr ⟨(ell', t'), hmem_pair, hell_eq⟩
  have hsome := h_wf.1 ell hmem
  exact Option.isSome_iff_exists.mp hsome

/-! ## Progress -/

/-- General Progress with loose output context and effect row.
    Closed input only; output can be any linear context and any
    effect row can arise. This is the form needed for sub-term
    recursion to work (pair_inv / app_inv etc. give sub-derivations
    at non-trivial output contexts). -/
theorem progress_aux
    (sigma : Store) (Sigma : StoreTyp)
    (h_wf : StoreTypTensorOnly Sigma)
    (h_store_wf : StoreWf sigma Sigma)
    (e : Term) (t : Typ)
    (Gamma' : LinearCtx) (eps : EffectRow)
    (h : HasType [] Sigma [] e t eps Gamma') :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ := by
  -- Case analysis on the term syntax; recursive `progress_aux` calls
  -- on strict sub-terms are accepted via the `termination_by sizeOf e`
  -- measure declared at the bottom of the definition.
  cases e with
  | var x =>
      -- T-Var requires `Γ_pre ++ [(x,t)] ++ Γ_post = []`, impossible.
      exfalso
      suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γin : LinearCtx)
          (e' : Term) (t' : Typ) (eps' : EffectRow) (Γout : LinearCtx),
          HasType Δ S Γin e' t' eps' Γout →
          Γin = [] → e' = Term.var x → False from
        hf [] Sigma [] (Term.var x) t eps Gamma' h rfl rfl
      intro Δ S Γin e' t' eps' Γout hd
      induction hd using HasType.rec
        (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
      | var _ _ Γpre Γpost _ _ =>
          intro hΓ _heq
          -- Γpre ++ [(x,t)] ++ Γpost = [] is impossible
          rcases Γpre with _ | _ <;> simp_all
      | subEff _ _ _ _ _ _ _ _ _ _ ih =>
          intro hΓ heq
          exact ih hΓ heq
      | _ =>
          first
            | (intro _ heq; cases heq)
            | exact True.intro
  | abs x tv body => exact Or.inl (IsValue.abs x tv body)
  | unit => exact Or.inl IsValue.unit
  | loc ell => exact Or.inl (IsValue.loc ell)
  | const v ds =>
      exact Or.inr ⟨storeExtend sigma (storeFreshLoc sigma) ⟨ds, v⟩,
                    Term.loc (storeFreshLoc sigma),
                    Step.tconst sigma v ds (storeFreshLoc sigma) rfl⟩
  | grad x tv tOut body =>
      exact Or.inr ⟨sigma, _, Step.tgrad sigma x tv tOut body⟩
  | vmap x tv body =>
      exact Or.inr ⟨sigma, _, Step.tvmap sigma x tv body (Dim.lit 0)⟩
  | pair e1 e2 =>
      -- `Term.pair` can only type at `Typ.pair`, so first project `t`
      -- to its pair components via a custom inversion. We then do the
      -- usual recurse-on-sub-terms dispatch.
      have ht_pair : ∃ t1 t2, t = Typ.pair t1 t2 := by
        suffices hf : ∀ (Δ : CapCtx) (S : StoreTyp) (Γ Γo : LinearCtx)
            (e' : Term) (t' : Typ) (ε : EffectRow) (a b : Term),
            HasType Δ S Γ e' t' ε Γo → e' = Term.pair a b →
            ∃ t1 t2, t' = Typ.pair t1 t2 from
          hf _ _ _ _ _ _ _ _ _ h rfl
        intro Δ S Γ Γo e' t' ε a b hd
        induction hd using HasType.rec
          (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
        | tpair _ _ _ _ _ _ _ t1 t2 _ _ _ _ _ _ =>
            intro _; exact ⟨t1, t2, rfl⟩
        | subEff _ _ _ _ _ _ _ _ _ _ ih => intro he; exact ih he
        | _ => first | (intro he; cases he) | exact True.intro
      obtain ⟨t1, t2, ht_eq⟩ := ht_pair
      subst ht_eq
      obtain ⟨Γmid, eps1, eps2, h1, h2, _⟩ := HasType.pair_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 t1 Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        rcases progress_aux sigma Sigma h_wf h_store_wf e2 t2 Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩
        · exact Or.inl (IsValue.pair e1 e2 hv1 hv2)
        · exact Or.inr ⟨sigma', Term.pair e1 e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.pairR e1) e2 e2' hstep⟩
      · exact Or.inr ⟨sigma', Term.pair e1' e2,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.pairL e2) e1 e1' hstep⟩
  | app e1 e2 =>
      obtain ⟨Γmid, t1, eps1, eps2, eps_inner, h1, h2⟩ := HasType.app_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.arrow t1 t eps_inner) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        obtain ⟨x, body, he1_eq⟩ := canonical_forms_arrow h_wf h1 hv1
        subst he1_eq
        rcases progress_aux sigma Sigma h_wf h_store_wf e2 t1 Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩
        · exact Or.inr ⟨sigma, subst body e2 x,
            Step.beta sigma x t1 body e2 hv2⟩
        · exact Or.inr ⟨sigma', Term.app (Term.abs x t1 body) e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.appR (Term.abs x t1 body))
                e2 e2' hstep⟩
      · exact Or.inr ⟨sigma', Term.app e1' e2,
          by simpa using Step.ctx sigma sigma' (EvalCtx.appL e2) e1 e1' hstep⟩
  | letBind x e1 e2 =>
      obtain ⟨Γmid, _Γ3, t1, eps1, eps2, _hfilt, h1, _h2⟩ := HasType.letBind_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 t1 Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩
      · exact Or.inr ⟨sigma, subst e2 e1 x, Step.letBind sigma x e1 e2 hv1⟩
      · exact Or.inr ⟨sigma', Term.letBind x e1' e2,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.letBind x e2) e1 e1' hstep⟩
  | copy e1 =>
      -- Wave 3: T-Copy is now tensor-only, so copy_inv gives a ds.
      -- canonical_forms_tensor applies and gives us a loc for e1.
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.copy_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
               (Typ.tensor ds) Gamma' eps h_inner with
          hv | ⟨sigma', e1', hstep⟩
      · -- e1 is a value of tensor type, so by canonical_forms_tensor
        -- it's a loc. Apply Step.copy.
        obtain ⟨ell, hell_eq⟩ := canonical_forms_tensor h_inner hv
        subst hell_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr ⟨storeExtend sigma (storeFreshLoc sigma) w,
                      Term.pair (Term.loc ell) (Term.loc (storeFreshLoc sigma)),
                      Step.copy sigma ell (storeFreshLoc sigma) w hw rfl⟩
      · exact Or.inr ⟨sigma', Term.copy e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.copy e1 e1' hstep⟩
  | letpair x y e1 e2 =>
      obtain ⟨Γmid, _Γ3, t1, t2, eps1, eps2, _hfilt, h1, _h2⟩ :=
        HasType.letpair_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.pair t1 t2) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, he1_eq, hv1v, hv2v⟩ :=
          canonical_forms_pair h_wf h1 hv1
        subst he1_eq
        exact Or.inr ⟨sigma, subst (subst e2 v1 x) v2 y,
          Step.letpair sigma x y v1 v2 e2 hv1v hv2v⟩
      · exact Or.inr ⟨sigma', Term.letpair x y e1' e2,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.letpair x y e2) e1 e1' hstep⟩
  | fst e1 =>
      obtain ⟨t2, h_inner⟩ := HasType.fst_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.pair t t2) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_wf h_inner hv
        subst heq
        exact Or.inr ⟨sigma, v1, Step.fst sigma v1 v2 hv1 hv2⟩
      · exact Or.inr ⟨sigma', Term.fst e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.fst e1 e1' hstep⟩
  | snd e1 =>
      obtain ⟨t1, h_inner⟩ := HasType.snd_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.pair t1 t) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_wf h_inner hv
        subst heq
        exact Or.inr ⟨sigma, v2, Step.snd sigma v1 v2 hv1 hv2⟩
      · exact Or.inr ⟨sigma', Term.snd e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.snd e1 e1' hstep⟩
  | add e1 e2 =>
      obtain ⟨ds, Γmid, eps1, eps2, _hteq, h1, h2⟩ := HasType.add_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.tensor ds) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        obtain ⟨ell1, he1_eq⟩ := canonical_forms_tensor h1 hv1
        subst he1_eq
        rcases progress_aux sigma Sigma h_wf h_store_wf e2
            (Typ.tensor ds) Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩
        · obtain ⟨ell2, he2_eq⟩ := canonical_forms_tensor h2 hv2
          subst he2_eq
          obtain ⟨hlk1, _⟩ := HasType.loc_inv h1
          obtain ⟨hlk2, _⟩ := HasType.loc_inv h2
          obtain ⟨w1, hw1⟩ := storeWf_lookup_witness h_store_wf hlk1
          obtain ⟨w2, hw2⟩ := storeWf_lookup_witness h_store_wf hlk2
          exact Or.inr ⟨_, _,
            Step.tadd sigma ell1 ell2 (storeFreshLoc sigma) w1 w2 hw1 hw2 rfl⟩
        · exact Or.inr ⟨sigma', Term.add (Term.loc ell1) e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.addR (Term.loc ell1))
                e2 e2' hstep⟩
      · exact Or.inr ⟨sigma', Term.add e1' e2,
          by simpa using Step.ctx sigma sigma' (EvalCtx.addL e2) e1 e1' hstep⟩
  | mul e1 e2 =>
      obtain ⟨ds, Γmid, eps1, eps2, _hteq, h1, h2⟩ := HasType.mul_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.tensor ds) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        obtain ⟨ell1, he1_eq⟩ := canonical_forms_tensor h1 hv1
        subst he1_eq
        rcases progress_aux sigma Sigma h_wf h_store_wf e2
            (Typ.tensor ds) Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩
        · obtain ⟨ell2, he2_eq⟩ := canonical_forms_tensor h2 hv2
          subst he2_eq
          obtain ⟨hlk1, _⟩ := HasType.loc_inv h1
          obtain ⟨hlk2, _⟩ := HasType.loc_inv h2
          obtain ⟨w1, hw1⟩ := storeWf_lookup_witness h_store_wf hlk1
          obtain ⟨w2, hw2⟩ := storeWf_lookup_witness h_store_wf hlk2
          exact Or.inr ⟨_, _,
            Step.tmul sigma ell1 ell2 (storeFreshLoc sigma) w1 w2 hw1 hw2 rfl⟩
        · exact Or.inr ⟨sigma', Term.mul (Term.loc ell1) e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.mulR (Term.loc ell1))
                e2 e2' hstep⟩
      · exact Or.inr ⟨sigma', Term.mul e1' e2,
          by simpa using Step.ctx sigma sigma' (EvalCtx.mulL e2) e1 e1' hstep⟩
  | sum e1 i =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.sum_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨ell, he1_eq⟩ := canonical_forms_tensor h_inner hv
        subst he1_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr ⟨_, _,
          Step.tsum sigma ell (storeFreshLoc sigma) w i hw rfl⟩
      · exact Or.inr ⟨sigma', Term.sum e1' i,
          by simpa using Step.ctx sigma sigma' (EvalCtx.sum i) e1 e1' hstep⟩
  | expand e1 i k =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.expand_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨ell, he1_eq⟩ := canonical_forms_tensor h_inner hv
        subst he1_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr ⟨_, _,
          Step.texpand sigma ell (storeFreshLoc sigma) w i k hw rfl⟩
      · exact Or.inr ⟨sigma', Term.expand e1' i k,
          by simpa using Step.ctx sigma sigma' (EvalCtx.expand i k) e1 e1' hstep⟩
  | uniformLike e1 lo hi =>
      obtain ⟨ds, eps0, _hteq, h_inner⟩ := HasType.uniformLike_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.tensor ds) Gamma' eps0 h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨ell, he1_eq⟩ := canonical_forms_tensor h_inner hv
        subst he1_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr ⟨_, _,
          Step.tuniformLike sigma ell (storeFreshLoc sigma) w lo hi hw rfl⟩
      · exact Or.inr ⟨sigma', Term.uniformLike e1' lo hi,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.uniformLike lo hi) e1 e1' hstep⟩
  | handle epsH body clauses =>
      obtain ⟨Γ2, epsB, h_body⟩ := HasType.handle_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf body t Γ2 epsB h_body with
          hv | ⟨sigma', body', hstep⟩
      · exact Or.inr ⟨sigma, body, Step.handleRet sigma epsH body clauses hv⟩
      · exact Or.inr ⟨sigma', Term.handle epsH body' clauses,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.handle epsH clauses) body body' hstep⟩
  | perform op e1 =>
      -- TODO Wave 3+: perform under empty effect row input requires
      -- a perform_inv lemma plus effect-row inversion.
      sorry
termination_by sizeOf e
decreasing_by all_goals (simp_wf; decreasing_tactic)

/-- The classic closed-form Progress: trivially follows from the
    generalized form. -/
theorem progress
    (sigma : Store) (Sigma : StoreTyp)
    (h_wf : StoreTypTensorOnly Sigma)
    (h_store_wf : StoreWf sigma Sigma)
    (e : Term) (t : Typ)
    (h : HasType [] Sigma [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ :=
  progress_aux sigma Sigma h_wf h_store_wf e t [] [] h

end LaCaDiLE
