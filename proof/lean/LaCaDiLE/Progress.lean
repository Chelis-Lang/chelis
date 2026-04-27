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

theorem ClausesTyped.mem_sig
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hcls : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR clauses)
    (hmem : (op, x, k, hb) ∈ clauses) :
    ∃ tArg, OpSigMatch op tArg Typ.unit := by
  refine ⟨opArgType op, ?_⟩
  cases op <;> simp [OpSigMatch, opArgType, opRetType]

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
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 : Typ) (eps1 eps2 : EffectRow)
      (slot : Option Typ),
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t eps2
              (Gamma3 ++ [(x, slot)]) ∧
      Gamma_out = Gamma3 := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Γ2 Γ3 _ _ _ t1 _ eps1 eps2 slot h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, eps1, eps2, slot, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih => exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetPair inversion. -/
theorem HasType.letpair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps Gamma_out) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 t2 : Typ) (eps1 eps2 : EffectRow)
      (slotX slotY : Option Typ),
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
              (Gamma3 ++ [(x, slotX), (y, slotY)]) ∧
      Gamma_out = Gamma3 := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1 t2 _ eps1 eps2 slotX slotY h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih => exact ih heq
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

/-! ## Small effect-row helpers used by `stuck_bubbles` -/

/-- If `op ∈ eps1`, then `op ∈ EffectRow.union eps1 eps2`. -/
theorem union_mem_left (eps1 eps2 : EffectRow) (op : EffectLabel)
    (h : op ∈ eps1) : op ∈ EffectRow.union eps1 eps2 := by
  unfold EffectRow.union
  exact List.mem_append_left _ h

/-- If `op ∈ eps2`, then `op ∈ EffectRow.union eps1 eps2`. -/
theorem union_mem_right (eps1 eps2 : EffectRow) (op : EffectLabel)
    (h : op ∈ eps2) : op ∈ EffectRow.union eps1 eps2 := by
  unfold EffectRow.union
  by_cases hmem : op ∈ eps1
  · exact List.mem_append_left _ hmem
  · refine List.mem_append_right _ ?_
    refine List.mem_filter.mpr ⟨h, ?_⟩
    have hcontains : eps1.contains op = false := by
      cases hc : eps1.contains op with
      | false => rfl
      | true => exact absurd (List.mem_of_elem_eq_true hc) hmem
    rw [hcontains]; rfl

/-- `op ∈ removeOp eps op'` iff `op ∈ eps` and `op ≠ op'`. -/
theorem removeOp_mem_iff (eps : EffectRow) (op op' : EffectLabel) :
    op ∈ EffectRow.removeOp eps op' ↔ op ∈ eps ∧ op ≠ op' := by
  unfold EffectRow.removeOp
  constructor
  · intro h
    obtain ⟨hmem, hne⟩ := List.mem_filter.mp h
    refine ⟨hmem, ?_⟩
    intro heq
    subst heq
    simp at hne
  · intro ⟨hmem, hne⟩
    refine List.mem_filter.mpr ⟨hmem, ?_⟩
    simp [hne]

/-- If `op ∈ eps` and `op ∉ ops`, then `op ∈ removeOps eps ops`. -/
theorem removeOps_mem_of_not_in
    {eps ops : EffectRow} {op : EffectLabel}
    (hmem : op ∈ eps) (hnot : op ∉ ops) :
    op ∈ EffectRow.removeOps eps ops := by
  induction ops with
  | nil => simpa [EffectRow.removeOps] using hmem
  | cons o rest ih =>
      have hne : op ≠ o := by
        intro heq; subst heq
        exact hnot List.mem_cons_self
      have hnot' : op ∉ rest := fun h => hnot (List.mem_cons_of_mem _ h)
      have hrec : op ∈ EffectRow.removeOps eps rest := ih hnot'
      show op ∈ EffectRow.removeOp (EffectRow.removeOps eps rest) o
      exact (removeOp_mem_iff _ _ _).mpr ⟨hrec, hne⟩

/-! ## StuckOnPerform (Wave 1 P2)

Direct multi-frame existential: `e = multiPlug Es (perform op v)`
where `v` is a value and no frame in `Es` catches `op`. Such a
term cannot take a step in isolation — it needs an enclosing
handler of `op`. It is the third disjunct of `progress_aux`. -/

inductive StuckOnPerform (op : EffectLabel) : Term → Prop where
  | mk (Es : EvalCtxChain) (v : Term)
       (hv : IsValue v)
       (hEs : EvalCtxChain.noHandleFor op Es) :
       StuckOnPerform op (multiPlug Es (Term.perform op v))

/-- A stuck-on-perform derivation forces `op ∈ eps`. Induction on
    the chain; the `cons` case inverts the enclosing frame via the
    matching `HasType.*_inv` lemma and pushes membership through
    the associated effect-row union / removeOps step. -/
theorem stuck_bubbles
    {Delta : CapCtx} {Sigma : StoreTyp}
    {op : EffectLabel} {v : Term}
    : ∀ (Es : EvalCtxChain) {Gamma Gamma' : LinearCtx}
        {t : Typ} {eps : EffectRow},
      EvalCtxChain.noHandleFor op Es →
      HasType Delta Sigma Gamma (multiPlug Es (Term.perform op v)) t eps Gamma' →
      op ∈ eps := by
  intro Es
  induction Es with
  | nil =>
      intro Gamma Gamma' t eps _hEs h
      exact hasType_perform_eff_mem h
  | cons E Es ih =>
      intro Gamma Gamma' t eps hEs h
      obtain ⟨hE, hEs'⟩ := hEs
      cases E with
      | hole =>
          simp only [multiPlug, plug] at h
          exact ih hEs' h
      | appL e2 =>
          have h' : HasType Delta Sigma Gamma
              (Term.app (multiPlug Es (Term.perform op v)) e2) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, eps1, _, h1, _, hsub⟩ := HasType.plug_app_inv h'
          exact hsub op (union_mem_left _ _ op (union_mem_left _ _ op (ih hEs' h1)))
      | appR v1 =>
          have h' : HasType Delta Sigma Gamma
              (Term.app v1 (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, _, eps2, _, h2, hsub⟩ := HasType.plug_app_inv h'
          exact hsub op (union_mem_left _ _ op (union_mem_right _ _ op (ih hEs' h2)))
      | letBind x e2 =>
          have h' : HasType Delta Sigma Gamma
              (Term.letBind x (multiPlug Es (Term.perform op v)) e2) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, eps1, _, _, h1, _, _, hsub⟩ := HasType.plug_letBind_inv h'
          exact hsub op (union_mem_left _ _ op (ih hEs' h1))
      | copy =>
          have h' : HasType Delta Sigma Gamma
              (Term.copy (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, h_inner⟩ := HasType.copy_inv h'
          exact ih hEs' h_inner
      | letpair x y e2 =>
          have h' : HasType Delta Sigma Gamma
              (Term.letpair x y (multiPlug Es (Term.perform op v)) e2) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, _, eps1, _, _, _, h1, _, _, hsub⟩ :=
            HasType.plug_letpair_inv h'
          exact hsub op (union_mem_left _ _ op (ih hEs' h1))
      | pairL e2 =>
          have h' : HasType Delta Sigma Gamma
              (Term.pair (multiPlug Es (Term.perform op v)) e2) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, eps1, _, _, h1, _, hsub⟩ := HasType.plug_pair_inv h'
          exact hsub op (union_mem_left _ _ op (ih hEs' h1))
      | pairR v1 =>
          have h' : HasType Delta Sigma Gamma
              (Term.pair v1 (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, _, eps2, _, _, h2, hsub⟩ := HasType.plug_pair_inv h'
          exact hsub op (union_mem_right _ _ op (ih hEs' h2))
      | fst tRight =>
          have h' : HasType Delta Sigma Gamma
              (Term.fst tRight (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          let h_inner := HasType.fst_inv h'
          exact ih hEs' h_inner
      | snd tLeft =>
          have h' : HasType Delta Sigma Gamma
              (Term.snd tLeft (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          let h_inner := HasType.snd_inv h'
          exact ih hEs' h_inner
      | addL e2 =>
          have h' : HasType Delta Sigma Gamma
              (Term.add (multiPlug Es (Term.perform op v)) e2) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, eps1, _, _, h1, _, hsub⟩ := HasType.plug_add_inv h'
          exact hsub op (union_mem_left _ _ op (ih hEs' h1))
      | addR v1 =>
          have h' : HasType Delta Sigma Gamma
              (Term.add v1 (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, eps2, _, _, h2, hsub⟩ := HasType.plug_add_inv h'
          exact hsub op (union_mem_right _ _ op (ih hEs' h2))
      | mulL e2 =>
          have h' : HasType Delta Sigma Gamma
              (Term.mul (multiPlug Es (Term.perform op v)) e2) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, eps1, _, _, h1, _, hsub⟩ := HasType.plug_mul_inv h'
          exact hsub op (union_mem_left _ _ op (ih hEs' h1))
      | mulR v1 =>
          have h' : HasType Delta Sigma Gamma
              (Term.mul v1 (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, eps2, _, _, h2, hsub⟩ := HasType.plug_mul_inv h'
          exact hsub op (union_mem_right _ _ op (ih hEs' h2))
      | sum d =>
          have h' : HasType Delta Sigma Gamma
              (Term.sum (multiPlug Es (Term.perform op v)) d) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, _, h_inner⟩ := HasType.sum_inv h'
          exact ih hEs' h_inner
      | expand d =>
          have h' : HasType Delta Sigma Gamma
              (Term.expand (multiPlug Es (Term.perform op v)) d) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, _, h_inner⟩ := HasType.expand_inv h'
          exact ih hEs' h_inner
      | uniformLike lo hi =>
          have h' : HasType Delta Sigma Gamma
              (Term.uniformLike (multiPlug Es (Term.perform op v)) lo hi)
              t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, eps0, _, h_inner, hsub⟩ := HasType.uniformLike_inv h'
          exact hsub op (union_mem_left _ _ op (ih hEs' h_inner))
      | handle epsH clauses =>
          have hopH : op ∉ epsH := hE
          have h' : HasType Delta Sigma Gamma
              (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses)
              t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, epsB, h_body, _, _, _, _, hsub⟩ :=
            HasType.handle_inv_strong h'
          have hopB : op ∈ epsB := ih hEs' h_body
          exact hsub op (removeOps_mem_of_not_in hopB hopH)
      | perform op' =>
          have h' : HasType Delta Sigma Gamma
              (Term.perform op' (multiPlug Es (Term.perform op v))) t eps Gamma' := by
            simpa [multiPlug, plug] using h
          obtain ⟨_, eps0, h_inner, _, hsub⟩ := HasType.perform_inv h'
          exact hsub op (union_mem_right _ _ op (ih hEs' h_inner))

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
    IsValue e ∨ (∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩) ∨
    ∃ op, StuckOnPerform op e := by
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
      exact Or.inr (Or.inl
        ⟨storeExtend sigma (storeFreshLoc sigma) ⟨ds, v⟩,
         Term.loc (storeFreshLoc sigma),
         Step.tconst sigma v ds (storeFreshLoc sigma) rfl⟩)
  | grad x tv tOut body =>
      exact Or.inr (Or.inl ⟨sigma, _, Step.tgrad sigma x tv tOut body⟩)
  | vmap x tv d body =>
      exact Or.inr (Or.inl ⟨sigma, _, Step.tvmap sigma x tv d body⟩)
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
          hv1 | ⟨sigma', e1', hstep⟩ | ⟨op, stk1⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        rcases progress_aux sigma Sigma h_wf h_store_wf e2 t2 Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩ | ⟨op, stk2⟩
        · exact Or.inl (IsValue.pair e1 e2 hv1 hv2)
        · exact Or.inr (Or.inl ⟨sigma', Term.pair e1 e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.pairR e1) e2 e2' hstep⟩)
        · obtain ⟨Es, v, hv, hEs⟩ := stk2
          refine Or.inr (Or.inr ⟨op, ?_⟩)
          have hrw : Term.pair e1 (multiPlug Es (Term.perform op v)) =
              multiPlug (EvalCtx.pairR e1 :: Es) (Term.perform op v) := by
            simp [multiPlug, plug]
          rw [hrw]
          exact StuckOnPerform.mk (EvalCtx.pairR e1 :: Es) v hv ⟨trivial, hEs⟩
      · exact Or.inr (Or.inl ⟨sigma', Term.pair e1' e2,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.pairL e2) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk1
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.pair (multiPlug Es (Term.perform op v)) e2 =
            multiPlug (EvalCtx.pairL e2 :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.pairL e2 :: Es) v hv ⟨trivial, hEs⟩
  | app e1 e2 =>
      obtain ⟨Γmid, t1, eps1, eps2, eps_inner, h1, h2⟩ := HasType.app_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.arrow t1 t eps_inner) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩ | ⟨op, stk1⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        obtain ⟨x, body, he1_eq⟩ := canonical_forms_arrow h_wf h1 hv1
        subst he1_eq
        rcases progress_aux sigma Sigma h_wf h_store_wf e2 t1 Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩ | ⟨op, stk2⟩
        · exact Or.inr (Or.inl ⟨sigma, subst body e2 x,
            Step.beta sigma x t1 body e2 hv2⟩)
        · exact Or.inr (Or.inl ⟨sigma', Term.app (Term.abs x t1 body) e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.appR (Term.abs x t1 body))
                e2 e2' hstep⟩)
        · obtain ⟨Es, v, hv, hEs⟩ := stk2
          refine Or.inr (Or.inr ⟨op, ?_⟩)
          have hrw : Term.app (Term.abs x t1 body)
              (multiPlug Es (Term.perform op v)) =
              multiPlug (EvalCtx.appR (Term.abs x t1 body) :: Es)
                (Term.perform op v) := by
            simp [multiPlug, plug]
          rw [hrw]
          exact StuckOnPerform.mk (EvalCtx.appR (Term.abs x t1 body) :: Es)
            v hv ⟨trivial, hEs⟩
      · exact Or.inr (Or.inl ⟨sigma', Term.app e1' e2,
          by simpa using Step.ctx sigma sigma' (EvalCtx.appL e2) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk1
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.app (multiPlug Es (Term.perform op v)) e2 =
            multiPlug (EvalCtx.appL e2 :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.appL e2 :: Es) v hv ⟨trivial, hEs⟩
  | letBind x e1 e2 =>
      obtain ⟨Γmid, _Γ3, t1, eps1, eps2, _hfilt, h1, _h2⟩ := HasType.letBind_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 t1 Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · exact Or.inr (Or.inl
          ⟨sigma, subst e2 e1 x, Step.letBind sigma x e1 e2 hv1⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.letBind x e1' e2,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.letBind x e2) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.letBind x (multiPlug Es (Term.perform op v)) e2 =
            multiPlug (EvalCtx.letBind x e2 :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.letBind x e2 :: Es) v hv ⟨trivial, hEs⟩
  | copy e1 =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.copy_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
               (Typ.tensor ds) Gamma' eps h_inner with
          hv | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨ell, hell_eq⟩ := canonical_forms_tensor h_inner hv
        subst hell_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr (Or.inl
          ⟨storeExtend sigma (storeFreshLoc sigma) w,
           Term.pair (Term.loc ell) (Term.loc (storeFreshLoc sigma)),
           Step.copy sigma ell (storeFreshLoc sigma) w hw rfl⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.copy e1',
          by simpa using Step.ctx sigma sigma' EvalCtx.copy e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.copy (multiPlug Es (Term.perform op v)) =
            multiPlug (EvalCtx.copy :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.copy :: Es) v hv ⟨trivial, hEs⟩
  | letpair x y e1 e2 =>
      obtain ⟨Γmid, _Γ3, t1, t2, eps1, eps2, _slotX, _slotY, h1, _h2, _hout⟩ :=
        HasType.letpair_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.pair t1 t2) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨v1, v2, he1_eq, hv1v, hv2v⟩ :=
          canonical_forms_pair h_wf h1 hv1
        subst he1_eq
        exact Or.inr (Or.inl ⟨sigma, subst (subst e2 v1 x) v2 y,
          Step.letpair sigma x y v1 v2 e2 hv1v hv2v⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.letpair x y e1' e2,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.letpair x y e2) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.letpair x y (multiPlug Es (Term.perform op v)) e2 =
            multiPlug (EvalCtx.letpair x y e2 :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.letpair x y e2 :: Es) v hv
          ⟨trivial, hEs⟩
  | fst tRight e1 =>
      let h_inner := HasType.fst_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.pair t tRight) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_wf h_inner hv
        subst heq
        exact Or.inr (Or.inl ⟨sigma, v1, Step.fst sigma tRight v1 v2 hv1 hv2⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.fst tRight e1',
          by simpa using Step.ctx sigma sigma' (EvalCtx.fst tRight) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.fst tRight (multiPlug Es (Term.perform op v)) =
            multiPlug (EvalCtx.fst tRight :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.fst tRight :: Es) v hv ⟨trivial, hEs⟩
  | snd tLeft e1 =>
      let h_inner := HasType.snd_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.pair tLeft t) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨v1, v2, heq, hv1, hv2⟩ := canonical_forms_pair h_wf h_inner hv
        subst heq
        exact Or.inr (Or.inl ⟨sigma, v2, Step.snd sigma tLeft v1 v2 hv1 hv2⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.snd tLeft e1',
          by simpa using Step.ctx sigma sigma' (EvalCtx.snd tLeft) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.snd tLeft (multiPlug Es (Term.perform op v)) =
            multiPlug (EvalCtx.snd tLeft :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.snd tLeft :: Es) v hv ⟨trivial, hEs⟩
  | add e1 e2 =>
      obtain ⟨ds, Γmid, eps1, eps2, _hteq, h1, h2⟩ := HasType.add_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.tensor ds) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩ | ⟨op, stk1⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        obtain ⟨ell1, he1_eq⟩ := canonical_forms_tensor h1 hv1
        subst he1_eq
        rcases progress_aux sigma Sigma h_wf h_store_wf e2
            (Typ.tensor ds) Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩ | ⟨op, stk2⟩
        · obtain ⟨ell2, he2_eq⟩ := canonical_forms_tensor h2 hv2
          subst he2_eq
          obtain ⟨hlk1, _⟩ := HasType.loc_inv h1
          obtain ⟨hlk2, _⟩ := HasType.loc_inv h2
          obtain ⟨w1, hw1⟩ := storeWf_lookup_witness h_store_wf hlk1
          obtain ⟨w2, hw2⟩ := storeWf_lookup_witness h_store_wf hlk2
          exact Or.inr (Or.inl ⟨_, _,
            Step.tadd sigma ell1 ell2 (storeFreshLoc sigma) w1 w2 hw1 hw2 rfl⟩)
        · exact Or.inr (Or.inl ⟨sigma', Term.add (Term.loc ell1) e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.addR (Term.loc ell1))
                e2 e2' hstep⟩)
        · obtain ⟨Es, v, hv, hEs⟩ := stk2
          refine Or.inr (Or.inr ⟨op, ?_⟩)
          have hrw : Term.add (Term.loc ell1) (multiPlug Es (Term.perform op v)) =
              multiPlug (EvalCtx.addR (Term.loc ell1) :: Es) (Term.perform op v) := by
            simp [multiPlug, plug]
          rw [hrw]
          exact StuckOnPerform.mk (EvalCtx.addR (Term.loc ell1) :: Es) v hv
            ⟨trivial, hEs⟩
      · exact Or.inr (Or.inl ⟨sigma', Term.add e1' e2,
          by simpa using Step.ctx sigma sigma' (EvalCtx.addL e2) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk1
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.add (multiPlug Es (Term.perform op v)) e2 =
            multiPlug (EvalCtx.addL e2 :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.addL e2 :: Es) v hv ⟨trivial, hEs⟩
  | mul e1 e2 =>
      obtain ⟨ds, Γmid, eps1, eps2, _hteq, h1, h2⟩ := HasType.mul_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1
          (Typ.tensor ds) Γmid eps1 h1 with
          hv1 | ⟨sigma', e1', hstep⟩ | ⟨op, stk1⟩
      · have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h1
        subst hΓmid
        obtain ⟨ell1, he1_eq⟩ := canonical_forms_tensor h1 hv1
        subst he1_eq
        rcases progress_aux sigma Sigma h_wf h_store_wf e2
            (Typ.tensor ds) Gamma' eps2 h2 with
            hv2 | ⟨sigma', e2', hstep⟩ | ⟨op, stk2⟩
        · obtain ⟨ell2, he2_eq⟩ := canonical_forms_tensor h2 hv2
          subst he2_eq
          obtain ⟨hlk1, _⟩ := HasType.loc_inv h1
          obtain ⟨hlk2, _⟩ := HasType.loc_inv h2
          obtain ⟨w1, hw1⟩ := storeWf_lookup_witness h_store_wf hlk1
          obtain ⟨w2, hw2⟩ := storeWf_lookup_witness h_store_wf hlk2
          exact Or.inr (Or.inl ⟨_, _,
            Step.tmul sigma ell1 ell2 (storeFreshLoc sigma) w1 w2 hw1 hw2 rfl⟩)
        · exact Or.inr (Or.inl ⟨sigma', Term.mul (Term.loc ell1) e2',
            by simpa using
              Step.ctx sigma sigma' (EvalCtx.mulR (Term.loc ell1))
                e2 e2' hstep⟩)
        · obtain ⟨Es, v, hv, hEs⟩ := stk2
          refine Or.inr (Or.inr ⟨op, ?_⟩)
          have hrw : Term.mul (Term.loc ell1) (multiPlug Es (Term.perform op v)) =
              multiPlug (EvalCtx.mulR (Term.loc ell1) :: Es) (Term.perform op v) := by
            simp [multiPlug, plug]
          rw [hrw]
          exact StuckOnPerform.mk (EvalCtx.mulR (Term.loc ell1) :: Es) v hv
            ⟨trivial, hEs⟩
      · exact Or.inr (Or.inl ⟨sigma', Term.mul e1' e2,
          by simpa using Step.ctx sigma sigma' (EvalCtx.mulL e2) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk1
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.mul (multiPlug Es (Term.perform op v)) e2 =
            multiPlug (EvalCtx.mulL e2 :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.mulL e2 :: Es) v hv ⟨trivial, hEs⟩
  | sum e1 d =>
      obtain ⟨ds, _hteq, _hmem, h_inner⟩ := HasType.sum_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨ell, he1_eq⟩ := canonical_forms_tensor h_inner hv
        subst he1_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr (Or.inl ⟨_, _,
          Step.tsum sigma ell (storeFreshLoc sigma) w d hw rfl⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.sum e1' d,
          by simpa using Step.ctx sigma sigma' (EvalCtx.sum d) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.sum (multiPlug Es (Term.perform op v)) d =
            multiPlug (EvalCtx.sum d :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.sum d :: Es) v hv ⟨trivial, hEs⟩
  | expand e1 d =>
      obtain ⟨ds, _hteq, h_inner⟩ := HasType.expand_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.tensor ds) Gamma' eps h_inner
          with hv | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨ell, he1_eq⟩ := canonical_forms_tensor h_inner hv
        subst he1_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr (Or.inl ⟨_, _,
          Step.texpand sigma ell (storeFreshLoc sigma) w d hw rfl⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.expand e1' d,
          by simpa using Step.ctx sigma sigma' (EvalCtx.expand d) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.expand (multiPlug Es (Term.perform op v)) d =
            multiPlug (EvalCtx.expand d :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.expand d :: Es) v hv ⟨trivial, hEs⟩
  | uniformLike e1 lo hi =>
      obtain ⟨ds, eps0, _hteq, h_inner, _hsub⟩ := HasType.uniformLike_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 (Typ.tensor ds) Gamma' eps0 h_inner
          with hv | ⟨sigma', e1', hstep⟩ | ⟨op, stk⟩
      · obtain ⟨ell, he1_eq⟩ := canonical_forms_tensor h_inner hv
        subst he1_eq
        obtain ⟨hlk, _⟩ := HasType.loc_inv h_inner
        obtain ⟨w, hw⟩ := storeWf_lookup_witness h_store_wf hlk
        exact Or.inr (Or.inl ⟨_, _,
          Step.tuniformLike sigma ell (storeFreshLoc sigma) w lo hi hw rfl⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.uniformLike e1' lo hi,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.uniformLike lo hi) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.uniformLike (multiPlug Es (Term.perform op v)) lo hi =
            multiPlug (EvalCtx.uniformLike lo hi :: Es) (Term.perform op v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.uniformLike lo hi :: Es) v hv
          ⟨trivial, hEs⟩
  | handle epsH body clauses =>
      obtain ⟨Γ2, epsB, h_body, _hHsubB, _hClIn, hClCov, hcls, _hsub⟩ :=
        HasType.handle_inv_strong h
      rcases progress_aux sigma Sigma h_wf h_store_wf body t Γ2 epsB h_body with
          hv | ⟨sigma', body', hstep⟩ | ⟨op, stk⟩
      · exact Or.inr (Or.inl
          ⟨sigma, body, Step.handleRet sigma epsH body clauses hv⟩)
      · exact Or.inr (Or.inl ⟨sigma', Term.handle epsH body' clauses,
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.handle epsH clauses) body body' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        by_cases hopH : op ∈ epsH
        · -- op is caught by this handle: pick the matching clause via
          -- `hClCov` and fire `Step.handleOpCtxs`.
          have hcov := hClCov op hopH
          obtain ⟨cl, hcl_mem, hcl_eq⟩ := hcov
          rcases hcl : cl with ⟨op', xVar, kVar, hb⟩
          subst hcl
          simp only at hcl_eq
          subst hcl_eq
          obtain ⟨tArg, hsig⟩ := ClausesTyped.mem_sig hcls hcl_mem
          refine Or.inr (Or.inl ⟨sigma, _,
            Step.handleOpCtxs sigma op' v epsH Es clauses xVar kVar hb Typ.unit
              hv ⟨tArg, hsig⟩ hcl_mem hopH hEs⟩)
        · -- op not caught: propagate stuck outward with the handle frame.
          refine Or.inr (Or.inr ⟨op, ?_⟩)
          have hrw : Term.handle epsH (multiPlug Es (Term.perform op v)) clauses =
              multiPlug (EvalCtx.handle epsH clauses :: Es) (Term.perform op v) := by
            simp [multiPlug, plug]
          rw [hrw]
          exact StuckOnPerform.mk (EvalCtx.handle epsH clauses :: Es) v hv
            ⟨hopH, hEs⟩
  | perform op e1 =>
      obtain ⟨tArg, eps0, h_inner, _hmatch, _hsub⟩ := HasType.perform_inv h
      rcases progress_aux sigma Sigma h_wf h_store_wf e1 tArg Gamma' eps0 h_inner
          with hv1 | ⟨sigma', e1', hstep⟩ | ⟨op', stk⟩
      · -- e1 is a value: the whole term is stuck on `perform op e1`
        -- at empty chain.
        refine Or.inr (Or.inr ⟨op, ?_⟩)
        have hrw : Term.perform op e1 = multiPlug [] (Term.perform op e1) := by
          simp [multiPlug]
        rw [hrw]
        exact StuckOnPerform.mk [] e1 hv1 trivial
      · exact Or.inr (Or.inl ⟨sigma', Term.perform op e1',
          by simpa using
            Step.ctx sigma sigma' (EvalCtx.perform op) e1 e1' hstep⟩)
      · obtain ⟨Es, v, hv, hEs⟩ := stk
        refine Or.inr (Or.inr ⟨op', ?_⟩)
        have hrw : Term.perform op (multiPlug Es (Term.perform op' v)) =
            multiPlug (EvalCtx.perform op :: Es) (Term.perform op' v) := by
          simp [multiPlug, plug]
        rw [hrw]
        exact StuckOnPerform.mk (EvalCtx.perform op :: Es) v hv ⟨trivial, hEs⟩
termination_by sizeOf e
decreasing_by all_goals (simp_wf; decreasing_tactic)

/-- The classic closed-form Progress: at empty outer effect row, the
    stuck disjunct from `progress_aux` is ruled out by `stuck_bubbles`
    (a `StuckOnPerform op` witness would force `op ∈ []`). -/
theorem progress
    (sigma : Store) (Sigma : StoreTyp)
    (h_wf : StoreTypTensorOnly Sigma)
    (h_store_wf : StoreWf sigma Sigma)
    (e : Term) (t : Typ)
    (h : HasType [] Sigma [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ := by
  rcases progress_aux sigma Sigma h_wf h_store_wf e t [] [] h with
    hv | hstep | ⟨op, stk⟩
  · exact Or.inl hv
  · exact Or.inr hstep
  · exfalso
    cases stk with
    | mk Es _v _hv hEs =>
        have hop : op ∈ ([] : EffectRow) := stuck_bubbles Es hEs h
        simp at hop

end LaCaDiLE
