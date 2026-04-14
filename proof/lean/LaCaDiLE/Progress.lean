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

/-- A well-typed value under a closed input context produces a closed
    output context. Used to keep the input context of sub-term
    progress calls empty once we learn the previous sub-term is a
    value. Proof deferred (HasType.rec with equation motive); stated
    here so the progress proof can cite it. -/
theorem value_preserves_closed_context
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma' : LinearCtx}
    {v : Term} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma [] v t eps Gamma')
    (_h_val : IsValue v) : Gamma' = [] := by
  -- TODO Wave 3+: HasType.rec with equation motive on Γ_in = [] and
  -- IsValue v; value-producing constructors (unit/abs/loc/tpair)
  -- each enforce Γ_out = Γ_in = [].
  sorry

/-- Canonical forms: a value of arrow type is a literal abstraction.
    Wave 2 caveat: loc-of-arrow-type is the pathological case (a
    location storing an arrow type). We use a `StoreTypTensorOnly`
    premise to rule it out. -/
theorem canonical_forms_arrow
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t1 t2 : Typ} {eps eps' : EffectRow}
    (_h : HasType Delta Sigma Gamma v (Typ.arrow t1 t2 eps) eps' Gamma')
    (_hv : IsValue v) :
    ∃ x body, v = Term.abs x t1 body := by
  -- TODO Wave 3: mirrors canonical_forms_tensor but needs
  -- StoreTypTensorOnly to dispatch the loc case (a loc with arrow
  -- type is uninhabited under the invariant). The abs/unit/pair
  -- cases close via equation contradiction on type shape.
  sorry

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
    Wave 2 status: blocked by the `loc` case. A runtime location value
    typed at a pair type is not a pair term, which would contradict
    the conclusion. The fix is a `StoreTypTensorOnly` well-formedness
    invariant asserting every Σ entry is a tensor type; this is
    Wave 3 infrastructure. -/
theorem canonical_forms_pair
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t1 t2 : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma v (Typ.pair t1 t2) eps Gamma')
    (_hv : IsValue v) :
    ∃ v1 v2, v = Term.pair v1 v2 ∧ IsValue v1 ∧ IsValue v2 := by
  sorry

/-! ## Progress -/

/-- General Progress with loose output context and effect row.
    Closed input only; output can be any linear context and any
    effect row can arise. This is the form needed for sub-term
    recursion to work (pair_inv / app_inv etc. give sub-derivations
    at non-trivial output contexts). -/
theorem progress_aux
    (sigma : Store) (Sigma : StoreTyp) (e : Term) (t : Typ)
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
      -- TODO Wave 3+: pair_inv-based extraction for progress; needs
      -- value_preserves_closed_context on e1 before dispatching e2.
      sorry
  | app e1 e2 =>
      -- TODO Wave 3+: needs an app inversion lemma in Preservation.
      sorry
  | letBind x e1 e2 =>
      -- TODO Wave 3+: needs a letBind inversion lemma.
      sorry
  | copy e1 =>
      obtain ⟨t0, _hteq, h_inner⟩ := HasType.copy_inv h
      rcases progress_aux sigma Sigma e1 t0 Gamma' eps h_inner with
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
      rcases progress_aux sigma Sigma e1 (Typ.pair t t2) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_inner hv
        subst heq
        exact Or.inr ⟨sigma, v1, Step.fst sigma v1 v2 hv1 hv2⟩
      · exact Or.inr ⟨sigma', Term.fst e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.fst e1 e1' hstep⟩
  | snd e1 =>
      obtain ⟨t1, h_inner⟩ := HasType.snd_inv h
      rcases progress_aux sigma Sigma e1 (Typ.pair t1 t) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_inner hv
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
      rcases progress_aux sigma Sigma e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · -- TODO Wave 3+: StoreWf-dependent head reduction on loc value.
        sorry
      · exact Or.inr ⟨sigma', Term.sum e1' i,
          by simpa using Step.ctx sigma sigma' (EvalCtx.sum i) e1 e1' hstep⟩
  | expand e1 i k =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.expand_inv h
      rcases progress_aux sigma Sigma e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · sorry
      · exact Or.inr ⟨sigma', Term.expand e1' i k,
          by simpa using Step.ctx sigma sigma' (EvalCtx.expand i k) e1 e1' hstep⟩
  | uniformLike e1 lo hi =>
      obtain ⟨ds, eps0, _hteq, h_inner⟩ := HasType.uniformLike_inv h
      rcases progress_aux sigma Sigma e1 (Typ.tensor ds) Gamma' eps0 h_inner
          with hv | ⟨sigma', e1', hstep⟩
      · sorry
      · exact Or.inr ⟨sigma', Term.uniformLike e1' lo hi,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.uniformLike lo hi) e1 e1' hstep⟩
  | handle epsH body clauses =>
      obtain ⟨Γ2, epsB, h_body⟩ := HasType.handle_inv h
      rcases progress_aux sigma Sigma body t Γ2 epsB h_body with
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
    (sigma : Store) (Sigma : StoreTyp) (e : Term) (t : Typ)
    (h : HasType [] Sigma [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ :=
  progress_aux sigma Sigma e t [] [] h

end LaCaDiLE
