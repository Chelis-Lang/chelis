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

/-- Linear-context domain shrinks across every HasType derivation.
    Wave 3 TODO: mechanical HasType.rec with a domain-subset motive.
    Each of the ~25 cases is short but requires careful handling of
    filter predicates in the binder cases (abs/letBind/letpair).
    Stated here so value_preserves_closed_context can cite it. -/
theorem has_type_linear_shrinks
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ x, x ∈ linearCtxDom Gamma' → x ∈ linearCtxDom Gamma := by
  sorry

/-- A well-typed value under a closed input context produces a closed
    output context. Follows directly from `has_type_linear_shrinks`:
    the output's domain is contained in the input's empty domain, so
    the output is empty. -/
theorem value_preserves_closed_context
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma' : LinearCtx}
    {v : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma [] v t eps Gamma')
    (_h_val : IsValue v) : Gamma' = [] := by
  have hshrink := has_type_linear_shrinks h
  cases hΓ' : Gamma' with
  | nil => rfl
  | cons hd tl =>
    exfalso
    have hmem : hd.1 ∈ linearCtxDom Gamma' := by
      rw [hΓ']; simp [linearCtxDom]
    have := hshrink hd.1 hmem
    simp [linearCtxDom] at this

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

/-! ## Progress -/

/-- General Progress with loose output context and effect row.
    Closed input only; output can be any linear context and any
    effect row can arise. This is the form needed for sub-term
    recursion to work (pair_inv / app_inv etc. give sub-derivations
    at non-trivial output contexts). -/
theorem progress_aux
    (sigma : Store) (Sigma : StoreTyp)
    (h_wf : StoreTypTensorOnly Sigma)
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
      -- pair_inv gives h1 : HasType [] Σ [] e1 t1 eps1 Γmid,
      -- h2 : HasType [] Σ Γmid e2 t2 eps2 Γ', hsub. Dispatch.
      cases e_eq : t with
      | pair t1 t2 =>
          rw [e_eq] at h
          obtain ⟨Γmid, eps1, eps2, h1, h2, _⟩ := HasType.pair_inv h
          rcases progress_aux sigma Sigma h_wf e1 t1 Γmid eps1 h1 with
              hv1 | ⟨sigma', e1', hstep⟩
          · -- e1 is a value. By value_preserves, Γmid = [].
            have hΓmid : Γmid = [] := value_preserves_closed_context h1 hv1
            subst hΓmid
            rcases progress_aux sigma Sigma h_wf e2 t2 Gamma' eps2 h2 with
                hv2 | ⟨sigma', e2', hstep⟩
            · exact Or.inl (IsValue.pair e1 e2 hv1 hv2)
            · exact Or.inr ⟨sigma', Term.pair e1 e2',
                by simpa using
                  Step.ctx sigma sigma' (EvalCtx.pairR e1) e2 e2' hstep⟩
          · exact Or.inr ⟨sigma', Term.pair e1' e2,
              by simpa using
                Step.ctx sigma sigma' (EvalCtx.pairL e2) e1 e1' hstep⟩
      | _ =>
          -- t is not a pair type, but `Term.pair` requires a pair type.
          -- Derive a contradiction via pair_inv — which requires
          -- Typ.pair. Sorry for now.
          sorry
  | app e1 e2 =>
      -- TODO Wave 3+: needs an app inversion lemma in Preservation.
      sorry
  | letBind x e1 e2 =>
      -- TODO Wave 3+: needs a letBind inversion lemma.
      sorry
  | copy e1 =>
      obtain ⟨t0, _hteq, h_inner⟩ := HasType.copy_inv h
      rcases progress_aux sigma Sigma h_wf e1 t0 Gamma' eps h_inner with
          hv | ⟨sigma', e1', hstep⟩
      · -- e1 is a value; copy on a value of pair-of-t0-t0 type. The
        -- reduction rule E-Copy only fires when the value is a loc,
        -- so we additionally need t0 to be a tensor — which is not
        -- guaranteed by T-Copy at this phase (copy is polymorphic in
        -- the source grammar). Stuck as sorry until copy is restricted
        -- to tensors or canonical-forms handles arbitrary types.
        -- TODO Wave 3+.
        sorry
      · exact Or.inr ⟨sigma', Term.copy e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.copy e1 e1' hstep⟩
  | letpair x y e1 e2 =>
      -- TODO Wave 3+: needs a letpair inversion lemma.
      sorry
  | fst e1 =>
      obtain ⟨t2, h_inner⟩ := HasType.fst_inv h
      rcases progress_aux sigma Sigma h_wf e1 (Typ.pair t t2) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_wf h_inner hv
        subst heq
        exact Or.inr ⟨sigma, v1, Step.fst sigma v1 v2 hv1 hv2⟩
      · exact Or.inr ⟨sigma', Term.fst e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.fst e1 e1' hstep⟩
  | snd e1 =>
      obtain ⟨t1, h_inner⟩ := HasType.snd_inv h
      rcases progress_aux sigma Sigma h_wf e1 (Typ.pair t1 t) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_wf h_inner hv
        subst heq
        exact Or.inr ⟨sigma, v2, Step.snd sigma v1 v2 hv1 hv2⟩
      · exact Or.inr ⟨sigma', Term.snd e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.snd e1 e1' hstep⟩
  | add e1 e2 =>
      -- TODO Wave 3+: full two-sub-term dispatch requires
      -- value_preserves_closed_context plus StoreWf witness lookup.
      sorry
  | mul e1 e2 =>
      sorry
  | sum e1 i =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.sum_inv h
      rcases progress_aux sigma Sigma h_wf e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · -- TODO Wave 3+: StoreWf-dependent head reduction on loc value.
        sorry
      · exact Or.inr ⟨sigma', Term.sum e1' i,
          by simpa using Step.ctx sigma sigma' (EvalCtx.sum i) e1 e1' hstep⟩
  | expand e1 i k =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.expand_inv h
      rcases progress_aux sigma Sigma h_wf e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · sorry
      · exact Or.inr ⟨sigma', Term.expand e1' i k,
          by simpa using Step.ctx sigma sigma' (EvalCtx.expand i k) e1 e1' hstep⟩
  | uniformLike e1 lo hi =>
      obtain ⟨ds, eps0, _hteq, h_inner⟩ := HasType.uniformLike_inv h
      rcases progress_aux sigma Sigma h_wf e1 (Typ.tensor ds) Gamma' eps0 h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · sorry
      · exact Or.inr ⟨sigma', Term.uniformLike e1' lo hi,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.uniformLike lo hi) e1 e1' hstep⟩
  | handle epsH body clauses =>
      obtain ⟨Γ2, epsB, h_body⟩ := HasType.handle_inv h
      rcases progress_aux sigma Sigma h_wf body t Γ2 epsB h_body with
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
    (e : Term) (t : Typ)
    (h : HasType [] Sigma [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ :=
  progress_aux sigma Sigma h_wf e t [] [] h

end LaCaDiLE
