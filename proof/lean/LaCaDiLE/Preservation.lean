-- LaCaDiLE/Preservation.lean — preservation theorem (Phase 2 proof).
--
-- WS2.5 target: reduction preserves typing and store well-formedness.
--
-- Wave 4 status: mechanical cases that do NOT depend on a typing rule
-- for `Term.loc` are closed directly by Step-case inversion plus small
-- HasType inversion lemmas. Cases that either (a) rely on the
-- TranslationDB bridge onto `SubstitutionDB`, (b) need to type a
-- freshly-allocated `Term.loc` (no `HasType.loc` rule exists in
-- Typing.lean yet — see TODO below), or (c) depend on `adjoint` /
-- `addDim` typing preservation lemmas that are themselves in progress,
-- are left as `sorry` with precise TODO markers.
--
-- KNOWN THEOREM-STATEMENT-LEVEL GAP (reported to Wave 4 coordinator):
-- `HasType` currently has no constructor for `Term.loc`. Every
-- store-allocating Step rule (tconst, copy, tadd, tmul, tsum, texpand,
-- tuniformLike) reduces to `Term.loc ellNew`, which therefore has no
-- typing derivation reachable from the current rule set. Closing those
-- cases requires adding a `HasType.loc` rule parameterized by
-- `storeTypLookup Sigma ell = some t`, plus a monotonicity lemma for
-- `StoreTyp` extension. That is a Typing.lean change, not a
-- Preservation.lean change, and has been flagged rather than done here.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational
import LaCaDiLE.AddDim
import LaCaDiLE.AdjointTyping
import LaCaDiLE.TranslationDB

namespace LaCaDiLE

/-! ## Wave 3 local oracles

These are oracle lemmas (sorried stubs) that downstream Wave 3+ proofs
rely on. They will be discharged by the canonical `has_type_linear_shrinks`
/ value-linear-closure work that is the sibling Wave 3 target. Keeping
them local keeps Preservation.lean compilable and lets the fst/snd/
handleRet cases land as real proof structure rather than raw `sorry`. -/

/-- A value of type `unit` leaves the linear context unchanged. This is
    the only shape needed by captured-handler preservation because the
    current operation signatures all take `unit` arguments. -/
theorem hasType_unit_value_preserves_context
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma v Typ.unit eps Gamma')
    (hv : IsValue v) :
    Gamma' = Gamma := by
  have hgen :
      ∀ {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
        {v : Term} {t : Typ} {eps : EffectRow},
        HasType Delta Sigma Gamma v t eps Gamma' →
        t = Typ.unit → IsValue v → Gamma' = Gamma := by
    intro Delta Sigma Gamma Gamma' v t eps h0
    induction h0 using HasType.rec
      (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
    | unit =>
        intro _ _
        rfl
    | loc =>
        intro _ _
        rfl
    | subEff _ _ _ _ _ _ _ _ _ _ ih =>
        intro hEq hv
        exact ih hEq hv
    | nil =>
        exact True.intro
    | cons =>
        exact True.intro
    | _ =>
        intro hEq hv
        cases hv <;> cases hEq
  exact hgen h rfl hv

-- (handleRet_value_preserves_typing moved below handle_inv)

/-! ## HasType inversion lemmas

`HasType` lives in a `mutual` block with `ClausesTyped`, so `cases` /
`rcases` on a `HasType` hypothesis does not always fire inside a larger
proof. We expose the handful of small inversion lemmas that the
preservation proof below needs. Each is a one-liner by `cases` in its
own top-level `theorem`, which Lean accepts. -/

/-- fst inversion: Wave 0.5 HasType.rec pattern absorbs subEff.
    Since fst propagates eps unchanged, no restatement needed. -/
theorem HasType.fst_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t tRight : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.fst tRight e) t eps Gamma2) :
    HasType Delta Sigma Gamma1 e (Typ.pair t tRight) eps Gamma2 := by
  generalize heq : Term.fst tRight e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | fst _ _ _ _ _ _ t2 _ h_inner _ =>
      cases heq
      simpa using h_inner
  | subEff Δ S Γ Γ' _ t_m eps0 eps' _h_sub h_sub ih =>
      exact HasType.subEff Δ S Γ Γ' e (Typ.pair t_m tRight) eps0 eps' (ih heq) h_sub
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

-- Wave 0.5 ripple: every inversion lemma is now blocked on the
-- subEff constructor, which `cases h` cannot dispatch without a
-- central strip_subEff helper. All inversions sorry'd; Wave 2
-- builds the helper and closes them in one pass.

theorem HasType.snd_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t tLeft : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.snd tLeft e) t eps Gamma2) :
    HasType Delta Sigma Gamma1 e (Typ.pair tLeft t) eps Gamma2 := by
  generalize heq : Term.snd tLeft e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | snd Δ S Γ1 Γ2 e' t1 t2 eps' h_inner _ih =>
      cases heq
      simpa using h_inner
  | subEff Δ S Γ Γ' e' t' eps0 eps1 _h_sub h_sub ih =>
      exact HasType.subEff Δ S Γ Γ' e (Typ.pair tLeft t') eps0 eps1 (ih heq) h_sub
  | _ => (try cases heq) <;> first | exact True.intro | (exfalso; contradiction)

/-- Pair inversion with subEff widening. Produces sub-derivations at
    internal effect rows `eps1`, `eps2` plus a `SubEffRow` witness
    connecting `union eps1 eps2` to the outer `eps`. -/
theorem HasType.pair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t1 t2 : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) (Typ.pair t1 t2) eps Gamma3) :
    ∃ Gamma2 eps1 eps2,
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.pair e1 e2 = e_in at h
  generalize htq : Typ.pair t1 t2 = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ Γ2 _ _ _ _ _ eps1 eps2 h1 h2 _ _ =>
      cases heq; cases htq
      exact ⟨Γ2, eps1, eps2, h1, h2, SubEffRow.refl _⟩
  | subEff _ _ _ _ _ _ _ _ _ h_sub ih =>
      obtain ⟨Γ2, eps1, eps2, hh1, hh2, hsr⟩ := ih heq htq
      exact ⟨Γ2, eps1, eps2, hh1, hh2, SubEffRow.trans hsr h_sub⟩
  | _ => (try cases heq) <;> (try cases htq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-! ### Inversion for store-allocating primitive terms -/

/-- Loc inversion with subEff widening. Effect-row = [] dropped. -/
theorem HasType.loc_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {ell : Loc} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.loc ell) t eps Gamma2) :
    storeTypLookup Sigma ell = some t ∧ Gamma1 = Gamma2 := by
  generalize heq : Term.loc ell = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | loc _ _ _ _ _ hlook =>
      cases heq
      exact ⟨hlook, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Const inversion with Wave 0.5 subEff widening. The exact `eps = []`
    conclusion of the raw T-Const rule doesn't survive subEff widening,
    so we only conclude `SubEffRow [] eps` (trivially true) and drop it.
    `Gamma1 = Gamma2` and `t = tensor ds` are preserved. -/
theorem HasType.const_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {v : Float} {ds : DimList} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.const v ds) t eps Gamma2) :
    t = Typ.tensor ds ∧ Gamma1 = Gamma2 := by
  generalize heq : Term.const v ds = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | const _ _ _ _ _ =>
      cases heq
      exact ⟨rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

theorem HasType.copy_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.copy e) t eps Gamma2) :
    ∃ ds, t = Typ.pair (Typ.tensor ds) (Typ.tensor ds) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.copy e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | copy _ _ _ _ _ ds _ h' _ =>
      cases heq
      exact ⟨ds, rfl, h'⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _h_sub h_sub ih =>
      obtain ⟨ds, hteq, h_inv⟩ := ih heq
      refine ⟨ds, hteq, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' h_inv h_sub
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Add inversion with subEff widening. The exact effect-row equation
    is dropped; we only produce sub-derivations at some internal effect
    rows. Callers needing the outer eps re-apply subEff. -/
theorem HasType.add_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.add e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 := by
  generalize heq : Term.add e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tadd _ _ _ Γ2 _ _ _ ds eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨ds, Γ2, eps1, eps2, rfl, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

theorem HasType.mul_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.mul e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 := by
  generalize heq : Term.mul e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tmul _ _ _ Γ2 _ _ _ ds eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨ds, Γ2, eps1, eps2, rfl, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

theorem HasType.sum_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {d : Dim} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.sum e d) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (rem ds d) ∧ d ∈ ds ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.sum e d = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tsum _ _ _ _ _ ds _ _ h' hmem _ =>
      cases heq
      exact ⟨ds, rfl, hmem, h'⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _h_sub h_sub ih =>
      obtain ⟨ds, hteq, hmem, h_inv⟩ := ih heq
      refine ⟨ds, hteq, hmem, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' h_inv h_sub
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

theorem HasType.expand_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {d : Dim} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.expand e d) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (ins ds d) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.expand e d = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | texpand _ _ _ _ _ ds _ _ h' _ =>
      cases heq
      exact ⟨ds, rfl, h'⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _h_sub h_sub ih =>
      obtain ⟨ds, hteq, h_inv⟩ := ih heq
      refine ⟨ds, hteq, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' h_inv h_sub
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- uniformLike inversion with subEff widening. Exact effect-row
    equation dropped (since subEff can widen further); the Random
    effect is only guaranteed in the inner eps0 via the union.
    Returns a SubEffRow witness `union eps0 [random] ⊆ eps` so callers
    can re-widen after rebuilding the constructor. -/
theorem HasType.uniformLike_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {lo hi : Float} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.uniformLike e lo hi) t eps Gamma2) :
    ∃ ds eps0,
      t = Typ.tensor ds ∧
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps0 Gamma2 ∧
      SubEffRow (EffectRow.union eps0 [EffectLabel.random]) eps := by
  generalize heq : Term.uniformLike e lo hi = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | uniformLike _ _ _ _ _ ds _ _ eps0 h' _ =>
      cases heq
      exact ⟨ds, eps0, rfl, h', fun _ hm => hm⟩
  | subEff _ _ _ _ _ _ _ _ _ h_sub ih =>
      obtain ⟨ds, eps0, hteq, h_inv, hsub⟩ := ih heq
      refine ⟨ds, eps0, hteq, h_inv, ?_⟩
      intro op hop
      exact h_sub op (hsub op hop)
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

-- StoreWf.extend_fresh moved to Store.lean so LinearitySoundness can use it.

theorem storeTypLookup_extend_other
    (Sigma : StoreTyp) (ell ellNew : Loc) (t t' : Typ)
    (h : storeTypLookup Sigma ell = some t) (hne : ell ≠ ellNew) :
    storeTypLookup (storeTypExtend Sigma ellNew t') ell = some t := by
  have hne' : ¬ (ellNew = ell) := fun h' => hne h'.symm
  unfold storeTypLookup storeTypExtend
  rw [show List.find? (fun p => decide (p.1 = ell)) ((ellNew, t') :: Sigma)
        = List.find? (fun p => decide (p.1 = ell)) Sigma from by
        simp [List.find?, hne']]
  exact h

theorem storeTypLookup_extend_self
    (Sigma : StoreTyp) (ell : Loc) (t : Typ) :
    storeTypLookup (storeTypExtend Sigma ell t) ell = some t := by
  simp [storeTypLookup, storeTypExtend]

theorem storeTypLookup_remove_other
    (Sigma : StoreTyp) (ellDead ell : Loc) (t : Typ)
    (h : storeTypLookup Sigma ell = some t) (hne : ell ≠ ellDead) :
    storeTypLookup (storeTypRemove Sigma ellDead) ell = some t := by
  induction Sigma generalizing ell t with
  | nil =>
      simp [storeTypLookup, List.find?] at h
  | cons hd tl ih =>
      by_cases hhdDead : hd.1 = ellDead
      · have htl : storeTypLookup tl ell = some t := by
          have hhdNe : hd.1 ≠ ell := by
            intro hEq
            exact hne (by simpa [hhdDead] using hEq.symm)
          simpa [storeTypLookup, List.find?, hhdNe] using h
        simpa [storeTypRemove, hhdDead] using (ih ell t htl hne)
      · by_cases hhdEll : hd.1 = ell
        · have hty : hd.2 = t := by
            simpa [storeTypLookup, List.find?, hhdEll] using h
          subst t
          have hne' : ell ≠ ellDead := hne
          unfold storeTypLookup storeTypRemove
          simp [List.find?, hhdEll, hne']
        · have htl : storeTypLookup tl ell = some t := by
            simpa [storeTypLookup, List.find?, hhdEll] using h
          simpa [storeTypLookup, storeTypRemove, List.find?, hhdDead, hhdEll] using
            (ih ell t htl hne)

theorem storeTypLookup_mem_dom
    {Sigma : StoreTyp} {ell : Loc} {t : Typ}
    (h : storeTypLookup Sigma ell = some t) :
    ell ∈ storeTypDom Sigma := by
  unfold storeTypLookup at h
  rcases hfind : Sigma.find? (fun p => p.1 = ell) with _ | ⟨ell', t'⟩
  · rw [hfind] at h
    simp at h
  · have hmem_pair : (ell', t') ∈ Sigma := List.mem_of_find?_eq_some hfind
    have hpred : decide (ell' = ell) = true := by
      have hp := @List.find?_some _ (fun p : Loc × Typ => decide (p.1 = ell))
        (ell', t') Sigma hfind
      simpa using hp
    have hell_eq : ell' = ell := of_decide_eq_true hpred
    exact List.mem_map.mpr ⟨(ell', t'), hmem_pair, hell_eq⟩

theorem exists_storeTypLookup_of_mem_dom
    {Sigma : StoreTyp} {ell : Loc}
    (h : ell ∈ storeTypDom Sigma) :
    ∃ t, storeTypLookup Sigma ell = some t := by
  induction Sigma with
  | nil =>
      cases h
  | cons hd tl ih =>
      simp [storeTypDom] at h
      rcases h with hhd | htl
      · subst hhd
        refine ⟨hd.2, ?_⟩
        simp [storeTypLookup, List.find?]
      · by_cases hhd : hd.1 = ell
        · subst hhd
          refine ⟨hd.2, ?_⟩
          simp [storeTypLookup, List.find?]
        · have htl' : ell ∈ storeTypDom tl := by
            rcases htl with ⟨t, ht⟩
            exact List.mem_map.mpr ⟨(ell, t), ht, rfl⟩
          rcases ih htl' with ⟨t, ht⟩
          exact ⟨t, by simpa [storeTypLookup, List.find?, hhd] using ht⟩

theorem StoreWf.lookup_isSome_of_typing
    {sigma : Store} {Sigma : StoreTyp} {ell : Loc} {t : Typ}
    (h_wf : StoreWf sigma Sigma)
    (h : storeTypLookup Sigma ell = some t) :
    (storeLookup sigma ell).isSome := by
  exact h_wf.1 ell (storeTypLookup_mem_dom h)

-- storeLookup_isSome_remove_ne, mem_storeTypDom_remove_iff, and
-- StoreWf.remove_extend all moved to Store.lean so LinearitySoundness
-- can use them without cross-file imports.

/-- Lift `addDim d` only on the store-typing entries whose locations
    occur in `locs`, leaving every other entry unchanged. This is the
    store-side repair needed by the `tvmap` preservation case: the
    transformed body needs lifted typings for its own referenced
    locations, but untouched outer-frame locations must keep their
    original types to satisfy `StoreTypOn`. -/
def addDimStoreTypOn (d : Dim) (locs : List Loc) (Sigma : StoreTyp) : StoreTyp :=
  Sigma.map (fun p =>
    if p.1 ∈ locs then (p.1, addDim d p.2) else p)

private theorem addDimStoreTypOn_entry_fst
    (d : Dim) (locs : List Loc) (p : Loc × Typ) :
    (if p.1 ∈ locs then (p.1, addDim d p.2) else p).1 = p.1 := by
  by_cases hmem : p.1 ∈ locs <;> simp [hmem]

theorem storeTypDom_addDimStoreTypOn
    (d : Dim) (locs : List Loc) (Sigma : StoreTyp) :
    storeTypDom (addDimStoreTypOn d locs Sigma) = storeTypDom Sigma := by
  unfold addDimStoreTypOn storeTypDom
  induction Sigma with
  | nil =>
      rfl
  | cons hd tl ih =>
      simp [addDimStoreTypOn_entry_fst, ih]

theorem addDimStoreTypOn_lookup_mem
    (d : Dim) (locs : List Loc) (Sigma : StoreTyp) (ell : Loc)
    (hmem : ell ∈ locs) :
    storeTypLookup (addDimStoreTypOn d locs Sigma) ell =
      storeTypLookup (addDimStoreTyp d Sigma) ell := by
  induction Sigma with
  | nil =>
      simp [storeTypLookup, addDimStoreTypOn, addDimStoreTyp]
  | cons hd tl ih =>
      by_cases hhd : hd.1 = ell
      · subst hhd
        simp [storeTypLookup, addDimStoreTypOn, addDimStoreTyp, hmem]
      · by_cases hhdmem : hd.1 ∈ locs
        · simp [storeTypLookup, addDimStoreTypOn, addDimStoreTyp, hhd, hhdmem] at ih ⊢
          exact ih
        · simp [storeTypLookup, addDimStoreTypOn, addDimStoreTyp, hhd, hhdmem] at ih ⊢
          exact ih

theorem addDimStoreTypOn_lookup_not_mem
    (d : Dim) (locs : List Loc) (Sigma : StoreTyp) (ell : Loc)
    (hnot : ell ∉ locs) :
    storeTypLookup (addDimStoreTypOn d locs Sigma) ell =
      storeTypLookup Sigma ell := by
  induction Sigma with
  | nil =>
      simp [storeTypLookup, addDimStoreTypOn]
  | cons hd tl ih =>
      by_cases hhd : hd.1 = ell
      · subst hhd
        simp [storeTypLookup, addDimStoreTypOn, hnot]
      · by_cases hhdmem : hd.1 ∈ locs
        · simp [storeTypLookup, addDimStoreTypOn, hhd, hhdmem] at ih ⊢
          exact ih
        · simp [storeTypLookup, addDimStoreTypOn, hhd, hhdmem] at ih ⊢
          exact ih

theorem StoreWf.addDimStoreTypOn
    {sigma : Store} {Sigma : StoreTyp}
    (d : Dim) (locs : List Loc)
    (h_wf : StoreWf sigma Sigma) :
    StoreWf sigma (addDimStoreTypOn d locs Sigma) := by
  simpa [StoreWf, storeTypDom_addDimStoreTypOn d locs Sigma] using h_wf

/-- Handle inversion with subEff widening. `eps = removeOps epsB epsH`
    equation dropped; only the body sub-derivation is extracted. -/
theorem HasType.handle_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {body : Term} {clauses : List (EffectLabel × String × String × Term)}
    {epsH : EffectRow} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.handle epsH body clauses) t eps Gamma3) :
    ∃ Gamma2 epsB,
      HasType Delta Sigma Gamma1 body t epsB Gamma2 := by
  generalize heq : Term.handle epsH body clauses = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | handle _ _ _ Γ2 _ _ _ _ _ epsB hb _ _ _ _ =>
      cases heq
      exact ⟨Γ2, epsB, hb⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- handleRet preservation: if a handle whose body is a value reduces
    to that value, the value re-types at the handle's outer effect row. -/
theorem handleRet_value_preserves_typing
    {Sigma : StoreTyp}
    {epsH : EffectRow} {v : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {t : Typ} {eps : EffectRow}
    (hv : IsValue v)
    (h : HasType [] Sigma [] (Term.handle epsH v clauses) t eps []) :
    HasType [] Sigma [] v t eps [] := by
  obtain ⟨Γ2, epsB, hb⟩ := HasType.handle_inv h
  have hG2 : Γ2 = [] := by
    have hdom := has_type_linear_shrinks hb
    cases Γ2 with
    | nil => rfl
    | cons p ps =>
        exfalso
        have : p.1 ∈ linearCtxDom ([] : LinearCtx) :=
          hdom p.1 (by simp [linearCtxDom])
        simpa [linearCtxDom] using this
  subst hG2
  exact HasType.value_eff_polymorphic hb hv eps

/-! ## Track C3: plug_preserves_typing

Congruence-on-typing lemma for evaluation contexts: if `plug E e` is
well-typed and we have a local inner-step that transforms `e` into
`e'` preserving its typing (under a possibly-extended store typing),
then `plug E e'` is well-typed at the same outer type/eps/contexts
under the extended store typing.

This is the structural witness needed by the full Preservation theorem
for the E-Ctx congruence rule. Downstream proofs (Sync step) will pass
Preservation itself as the `h_inner` argument.

The proof structure is an induction on `E : EvalCtx`. Every case follows
the same template: invert HasType on the plugged term to extract a
sub-derivation for `e`, apply `h_inner` to get the replacement derivation
for `e'`, and rebuild the outer constructor. Cases whose inversion lemmas
drop `SubEffRow` witnesses require an extra `HasType.subEff` widening,
which is currently left as a local sorry pending the Wave-2 inversion
upgrade; the top-level theorem statement is stable. -/

/-- Store-typing agreement restricted to the runtime locations a term
    actually mentions. This is the right granularity for `ctx`: sibling
    subterms only need the post-step store typing to agree on the
    locations they still reference, not on every location in the
    pre-step store typing. -/
def StoreTypOn (locs : List Loc) (Sigma Sigma' : StoreTyp) : Prop :=
  ∀ ell t, ell ∈ locs →
    storeTypLookup Sigma ell = some t →
    storeTypLookup Sigma' ell = some t

theorem StoreTypOn.mono
    {locs1 locs2 : List Loc} {Sigma Sigma' : StoreTyp}
    (hsub : ∀ ell, ell ∈ locs1 → ell ∈ locs2)
    (h : StoreTypOn locs2 Sigma Sigma') :
    StoreTypOn locs1 Sigma Sigma' := by
  intro ell t hell hlook
  exact h ell t (hsub ell hell) hlook

theorem StoreTypOn.append_left
    {locs1 locs2 : List Loc} {Sigma Sigma' : StoreTyp}
    (h : StoreTypOn (locs1 ++ locs2) Sigma Sigma') :
    StoreTypOn locs1 Sigma Sigma' := by
  refine StoreTypOn.mono ?_ h
  intro ell hell
  exact List.mem_append_left _ hell

theorem StoreTypOn.append_right
    {locs1 locs2 : List Loc} {Sigma Sigma' : StoreTyp}
    (h : StoreTypOn (locs1 ++ locs2) Sigma Sigma') :
    StoreTypOn locs2 Sigma Sigma' := by
  refine StoreTypOn.mono ?_ h
  intro ell hell
  exact List.mem_append_right _ hell

theorem StoreTypOn.of_sub
    {locs : List Loc} {Sigma Sigma' : StoreTyp}
    (hsub : StoreTypSub Sigma Sigma') :
    StoreTypOn locs Sigma Sigma' := by
  intro ell t _ hell
  exact hsub ell t hell

/-- Every location in `locs` is present in the runtime store. This is
    the freshness side-condition needed by runtime-linearity
    preservation: when a step allocates a fresh location, we must know
    it is fresh not only for the redex, but also for any untouched
    outer-frame references threaded through the induction. -/
def StoreLiveOn (sigma : Store) (locs : List Loc) : Prop :=
  ∀ ell, ell ∈ locs → (storeLookup sigma ell).isSome

theorem StoreLiveOn.append
    {sigma : Store} {locs1 locs2 : List Loc}
    (h1 : StoreLiveOn sigma locs1)
    (h2 : StoreLiveOn sigma locs2) :
    StoreLiveOn sigma (locs1 ++ locs2) := by
  intro ell hell
  rcases List.mem_append.mp hell with hmem1 | hmem2
  · exact h1 ell hmem1
  · exact h2 ell hmem2

theorem StoreLiveOn.append_left
    {sigma : Store} {locs1 locs2 : List Loc}
    (h : StoreLiveOn sigma (locs1 ++ locs2)) :
    StoreLiveOn sigma locs1 := by
  intro ell hell
  exact h ell (List.mem_append_left _ hell)

theorem StoreLiveOn.append_right
    {sigma : Store} {locs1 locs2 : List Loc}
    (h : StoreLiveOn sigma (locs1 ++ locs2)) :
    StoreLiveOn sigma locs2 := by
  intro ell hell
  exact h ell (List.mem_append_right _ hell)

theorem StoreLiveOn.of_subset
    {sigma : Store} {locs1 locs2 : List Loc}
    (h : StoreLiveOn sigma locs2)
    (hsub : ∀ ell, ell ∈ locs1 → ell ∈ locs2) :
    StoreLiveOn sigma locs1 := by
  intro ell hell
  exact h ell (hsub ell hell)

@[simp] theorem storeLiveOn_nil
    {sigma : Store} :
    StoreLiveOn sigma [] := by
  intro ell hell
  cases hell

theorem storeLiveOn_of_storeTypOn
    {sigma sigma' : Store} {locs : List Loc}
    {Sigma Sigma' : StoreTyp}
    (hLive : StoreLiveOn sigma locs)
    (h_wf : StoreWf sigma Sigma)
    (h_wf' : StoreWf sigma' Sigma')
    (hOn : StoreTypOn locs Sigma Sigma') :
    StoreLiveOn sigma' locs := by
  intro ell hell
  have hLiveSigma : (storeLookup sigma ell).isSome := hLive ell hell
  have hDom : ell ∈ storeTypDom Sigma := h_wf.2 ell hLiveSigma
  rcases exists_storeTypLookup_of_mem_dom hDom with ⟨t, hLook⟩
  have hLook' : storeTypLookup Sigma' ell = some t := hOn ell t hell hLook
  exact StoreWf.lookup_isSome_of_typing h_wf' hLook'

theorem storeLiveCtxLocRefs_of_storeTypOn
    {sigma sigma' : Store} {E : EvalCtx}
    {Sigma Sigma' : StoreTyp}
    (hLive : StoreLiveCtxLocRefs sigma E)
    (h_wf : StoreWf sigma Sigma)
    (h_wf' : StoreWf sigma' Sigma')
    (hOn : StoreTypOn (ctxLocRefs E) Sigma Sigma') :
    StoreLiveCtxLocRefs sigma' E := by
  simpa [StoreLiveCtxLocRefs, StoreLiveOn] using
    (storeLiveOn_of_storeTypOn (locs := ctxLocRefs E) hLive h_wf h_wf' hOn)

private theorem hasType_store_live_on_locRefs
    {sigma : Store}
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' : LinearCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (h_wf : StoreWf sigma Sigma) :
    StoreLiveOn sigma (locRefs e) := by
  induction h using HasType.rec
    (motive_2 := fun _ S_ _ _ _ _ clauses _ =>
      StoreWf sigma S_ → StoreLiveOn sigma (locRefsClauses clauses)) with
  | var =>
      simpa [locRefs, StoreLiveOn]
  | unit =>
      simpa [locRefs, StoreLiveOn]
  | abs _ _ _ _ _ _ _ _ _ _ _ ih =>
      simpa [locRefs] using ih h_wf
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact StoreLiveOn.append (ih1 h_wf) (ih2 h_wf)
  | letBind _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact StoreLiveOn.append (ih1 h_wf) (ih2 h_wf)
  | copy _ _ _ _ _ _ _ _ ih =>
      simpa [locRefs] using ih h_wf
  | letpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact StoreLiveOn.append (ih1 h_wf) (ih2 h_wf)
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact StoreLiveOn.append (ih1 h_wf) (ih2 h_wf)
  | fst _ _ _ _ _ _ _ _ _ ih =>
      simpa [locRefs] using ih h_wf
  | snd _ _ _ _ _ _ _ _ _ ih =>
      simpa [locRefs] using ih h_wf
  | const =>
      simpa [locRefs, StoreLiveOn]
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact StoreLiveOn.append (ih1 h_wf) (ih2 h_wf)
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact StoreLiveOn.append (ih1 h_wf) (ih2 h_wf)
  | tsum _ _ _ _ _ _ _ _ _h hmem ih =>
      simpa [locRefs] using ih h_wf
  | texpand _ _ _ _ _ _ _ _ _h ih =>
      simpa [locRefs] using ih h_wf
  | uniformLike Δ_ _ Γ1 Γ2 e0 ds lo hi ep _h ih =>
      simpa [locRefs] using ih h_wf
  | perform Δ_ _ Γ1 Γ2 op e0 tArg tRet ep _h hM ih =>
      simpa [locRefs] using ih h_wf
  | tgrad Δ_ _ Γ_ x ds dsOut body ep slot _h hsubEff _hSupp _hCtxSupp ih =>
      simpa [locRefs] using ih h_wf
  | tvmap Δ_ _ Γ_ x t1 t2 body ep d slot _h ih =>
      simpa [locRefs] using ih h_wf
  | handle Δ_ _ Γ1 Γ2 Γ3 body clauses t_ epsH epsB _hb hSubsH hClsH
      hCover _hcls ihBody ihClauses =>
      exact StoreLiveOn.append (ihBody h_wf) (ihClauses h_wf)
  | loc _ _ _ ell tv hlook =>
      intro ell' hell
      simp [locRefs] at hell
      subst ell'
      exact StoreWf.lookup_isSome_of_typing h_wf hlook
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih h_wf
  | nil =>
      simpa [locRefsClauses, StoreLiveOn]
  | cons Δ_ _ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest slotX slotK
      hmatch _h_body _h_rest ihBody ihRest hs =>
      exact StoreLiveOn.append (ihBody hs) (ihRest hs)

private theorem hasType_store_live_on_ctxLocRefs
    {sigma : Store}
    {Sigma : StoreTyp} {e : Term} {t : Typ} {eps : EffectRow}
    (E : EvalCtx)
    (h : HasType [] Sigma [] (plug E e) t eps [])
    (h_wf : StoreWf sigma Sigma) :
    StoreLiveOn sigma (ctxLocRefs E) := by
  apply StoreLiveOn.of_subset (hasType_store_live_on_locRefs h h_wf)
  intro ell hell
  exact (mem_locRefs_plug E e ell).2 (Or.inl hell)

private theorem hasType_store_live_on_chainLocRefs
    {sigma : Store}
    {Sigma : StoreTyp} {e : Term} {t : Typ} {eps : EffectRow}
    (Es : EvalCtxChain)
    (h : HasType [] Sigma [] (multiPlug Es e) t eps [])
    (h_wf : StoreWf sigma Sigma) :
    StoreLiveOn sigma (chainLocRefs Es) := by
  apply StoreLiveOn.of_subset (hasType_store_live_on_locRefs h h_wf)
  intro ell hell
  exact (mem_locRefs_multiPlug Es e ell).2 (Or.inl hell)

/-- Step-local store agreement on locations not mentioned by the redex.
    This is the missing store-side fact for the `ctx` case: numeric
    reductions may consume locations mentioned inside the redex, but
    they preserve typings for all locations disjoint from that redex. -/
private theorem step_preserves_store_typing_on_aux
    (Sigma : StoreTyp) (locs : List Loc)
    (c1 c2 : Config)
    (h_wf : StoreWf c1.store Sigma)
    (h_step : Step c1 c2)
    (hsep : LocRefsSeparated locs (locRefs c1.term)) :
    ∃ Sigma', StoreWf c2.store Sigma' ∧ StoreTypOn locs Sigma Sigma' := by
  induction h_step with
  | beta =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | letBind =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | letpair =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | fst =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | snd =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | handleRet =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | handleOpDirect =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | handleOpCtx =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | handleOpCtxs =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | tgrad =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | tvmap =>
      exact ⟨Sigma, h_wf, fun _ _ _ hlook => hlook⟩
  | tconst s v ds ell hell =>
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds),
        StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf, ?_⟩
      intro ell' t hmem hlook
      have hLive : (storeLookup s ell').isSome :=
        StoreWf.lookup_isSome_of_typing h_wf hlook
      have hne : ell' ≠ ell := by
        intro hEq
        exact (storeFreshLoc_ne s ell' hLive) (by rw [hEq, hell])
      exact storeTypLookup_extend_other Sigma ell' ell t (Typ.tensor ds) hlook hne
  | copy s ell ellNew w _hlook hfresh =>
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor w.shape),
        StoreWf.extend_fresh ellNew w (Typ.tensor w.shape) h_wf, ?_⟩
      intro ell' t hmem hlookTy
      have hLive : (storeLookup s ell').isSome :=
        StoreWf.lookup_isSome_of_typing h_wf hlookTy
      have hne : ell' ≠ ellNew := by
        intro hEq
        exact (storeFreshLoc_ne s ell' hLive) (by rw [hEq, hfresh])
      exact storeTypLookup_extend_other Sigma ell' ellNew t (Typ.tensor w.shape) hlookTy hne
  | tadd s ell1 ell2 ellOut w1 w2 _h1 _h2 hfresh =>
      have h_wf1 := StoreWf.remove ell1 h_wf
      have h_wf2 := StoreWf.remove ell2 h_wf1
      refine ⟨storeTypExtend
          (storeTypRemove (storeTypRemove Sigma ell1) ell2)
          ellOut (Typ.tensor w1.shape),
        StoreWf.extend_fresh ellOut (tensorOpPlaceholder w1 w2)
          (Typ.tensor w1.shape) h_wf2, ?_⟩
      intro ell t hmem hlook
      have hne1 : ell ≠ ell1 := by
        intro hEq
        subst hEq
        exact hsep ell hmem (by simp [locRefs])
      have hne2 : ell ≠ ell2 := by
        intro hEq
        subst hEq
        exact hsep ell hmem (by simp [locRefs])
      have hlook1 := storeTypLookup_remove_other Sigma ell1 ell t hlook hne1
      have hlook2 := storeTypLookup_remove_other (storeTypRemove Sigma ell1) ell2 ell t hlook1 hne2
      have hLive : (storeLookup s ell).isSome := StoreWf.lookup_isSome_of_typing h_wf hlook
      have hneOut : ell ≠ ellOut := by
        rw [hfresh]
        exact storeFreshLoc_ne s ell hLive
      exact storeTypLookup_extend_other
        (storeTypRemove (storeTypRemove Sigma ell1) ell2)
        ell ellOut t (Typ.tensor w1.shape) hlook2 hneOut
  | tmul s ell1 ell2 ellOut w1 w2 _h1 _h2 hfresh =>
      have h_wf1 := StoreWf.remove ell1 h_wf
      have h_wf2 := StoreWf.remove ell2 h_wf1
      refine ⟨storeTypExtend
          (storeTypRemove (storeTypRemove Sigma ell1) ell2)
          ellOut (Typ.tensor w1.shape),
        StoreWf.extend_fresh ellOut (tensorOpPlaceholder w1 w2)
          (Typ.tensor w1.shape) h_wf2, ?_⟩
      intro ell t hmem hlook
      have hne1 : ell ≠ ell1 := by
        intro hEq
        subst hEq
        exact hsep ell hmem (by simp [locRefs])
      have hne2 : ell ≠ ell2 := by
        intro hEq
        subst hEq
        exact hsep ell hmem (by simp [locRefs])
      have hlook1 := storeTypLookup_remove_other Sigma ell1 ell t hlook hne1
      have hlook2 := storeTypLookup_remove_other (storeTypRemove Sigma ell1) ell2 ell t hlook1 hne2
      have hLive : (storeLookup s ell).isSome := StoreWf.lookup_isSome_of_typing h_wf hlook
      have hneOut : ell ≠ ellOut := by
        rw [hfresh]
        exact storeFreshLoc_ne s ell hLive
      exact storeTypLookup_extend_other
        (storeTypRemove (storeTypRemove Sigma ell1) ell2)
        ell ellOut t (Typ.tensor w1.shape) hlook2 hneOut
  | tsum s ell ellOut w d hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
          (Typ.tensor (rem w.shape d)), ?_, ?_⟩
      · have h_isSome : (storeLookup s ell).isSome := by
          rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := rem w.shape d, data := w.data }
          (Typ.tensor (rem w.shape d)) h_wf hfresh hne
      · intro ell' t hmem hlookTy
        have hneIn : ell' ≠ ell := by
          intro hEq
          subst hEq
          exact hsep ell' hmem (by simp [locRefs])
        have hlookRem := storeTypLookup_remove_other Sigma ell ell' t hlookTy hneIn
        have hLive : (storeLookup s ell').isSome := StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hneOut : ell' ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell' hLive
        exact storeTypLookup_extend_other
          (storeTypRemove Sigma ell) ell' ellOut t (Typ.tensor (rem w.shape d))
          hlookRem hneOut
  | texpand s ell ellOut w d hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
          (Typ.tensor (ins w.shape d)), ?_, ?_⟩
      · have h_isSome : (storeLookup s ell).isSome := by
          rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := ins w.shape d, data := w.data }
          (Typ.tensor (ins w.shape d)) h_wf hfresh hne
      · intro ell' t hmem hlookTy
        have hneIn : ell' ≠ ell := by
          intro hEq
          subst hEq
          exact hsep ell' hmem (by simp [locRefs])
        have hlookRem := storeTypLookup_remove_other Sigma ell ell' t hlookTy hneIn
        have hLive : (storeLookup s ell').isSome := StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hneOut : ell' ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell' hLive
        exact storeTypLookup_extend_other
          (storeTypRemove Sigma ell) ell' ellOut t (Typ.tensor (ins w.shape d))
          hlookRem hneOut
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
          (Typ.tensor w.shape), ?_, ?_⟩
      · have h_isSome : (storeLookup s ell).isSome := by
          rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := w.shape, data := lo }
          (Typ.tensor w.shape) h_wf hfresh hne
      · intro ell' t hmem hlookTy
        have hneIn : ell' ≠ ell := by
          intro hEq
          subst hEq
          exact hsep ell' hmem (by simp [locRefs])
        have hlookRem := storeTypLookup_remove_other Sigma ell ell' t hlookTy hneIn
        have hLive : (storeLookup s ell').isSome := StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hneOut : ell' ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell' hLive
        exact storeTypLookup_extend_other
          (storeTypRemove Sigma ell) ell' ellOut t (Typ.tensor w.shape)
          hlookRem hneOut
  | ctx sigma sigma' E e e' h_inner ih =>
      have hsepInner : LocRefsSeparated locs (locRefs e) := by
        intro ell hmem hloc
        exact hsep ell hmem ((mem_locRefs_plug E e ell).2 (Or.inr hloc))
      exact ih h_wf hsepInner

private theorem step_preserves_store_typing_on_ctxLocRefs
    (Sigma : StoreTyp)
    (sigma sigma' : Store)
    (E : EvalCtx) (e e' : Term)
    (h_wf : StoreWf sigma Sigma)
    (h_linear : RuntimeLinear (plug E e))
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma', StoreWf sigma' Sigma' ∧ StoreTypOn (ctxLocRefs E) Sigma Sigma' := by
  rcases runtimeLinear_plug (E := E) (e := e) h_linear with ⟨_hlin, hsep⟩
  exact step_preserves_store_typing_on_aux
    Sigma (ctxLocRefs E) ⟨sigma, e⟩ ⟨sigma', e'⟩ h_wf h_step hsep

private theorem step_preserves_store_typing_on_ctxLocRefs_append
    (Sigma : StoreTyp) (locs : List Loc)
    (sigma sigma' : Store)
    (E : EvalCtx) (e e' : Term)
    (h_wf : StoreWf sigma Sigma)
    (h_linear : RuntimeLinear (plug E e))
    (hsep : LocRefsSeparated locs (locRefs (plug E e)))
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma', StoreWf sigma' Sigma' ∧
      StoreTypOn (ctxLocRefs E ++ locs) Sigma Sigma' := by
  rcases runtimeLinear_plug (E := E) (e := e) h_linear with ⟨_hlin, hsepCtx⟩
  have hsepBoth : LocRefsSeparated (ctxLocRefs E ++ locs) (locRefs e) := by
    intro ell hmem hloc
    rcases List.mem_append.mp hmem with hctx | hlocs
    · exact hsepCtx ell hctx hloc
    · exact hsep ell hlocs ((mem_locRefs_plug E e ell).2 (Or.inr hloc))
  exact step_preserves_store_typing_on_aux
    Sigma (ctxLocRefs E ++ locs) ⟨sigma, e⟩ ⟨sigma', e'⟩ h_wf h_step hsepBoth

/-- Store-typing weakening: every `HasType` derivation remains valid
    under a monotone extension of the store typing. Unblocks every
    binary `EvalCtx` case of `plug_preserves_typing`, since after the
    inner step advances the first sub-term to `Σ'`, the sibling sub-term
    (originally typed at `Σ`) must be lifted to `Σ'` before rebuilding
    the outer constructor. Proved by a 23-case induction on `HasType`,
    closed uniformly by reapplying each constructor at the new `Σ'`.
    Only the `loc` case consults `hsub`; every other case just threads
    the new store typing through the premises. -/
theorem hasType_store_weaken
    {Delta : CapCtx} {Sigma Sigma' : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (hsub : StoreTypSub Sigma Sigma')
    :
    HasType Delta Sigma' Gamma e t eps Gamma' := by
  induction h using HasType.rec
    (motive_2 := fun Δ_ S_ Γ2_ Γ3_ t_ εR_ cls_ _ =>
      StoreTypSub S_ Sigma' → ClausesTyped Δ_ Sigma' Γ2_ Γ3_ t_ εR_ cls_) with
  | var Δ_ _ Γpre Γpost x tv => exact HasType.var Δ_ Sigma' Γpre Γpost x tv
  | unit Δ_ _ Γ_ => exact HasType.unit Δ_ Sigma' Γ_
  | abs Δ_ _ Γ1 Γ2 x t1 t2 eps_ body slot _h ih =>
      exact HasType.abs Δ_ Sigma' Γ1 Γ2 x t1 t2 eps_ body slot (ih hsub)
  | app Δ_ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.app Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2
        (ih1 hsub) (ih2 hsub)
  | letBind Δ_ _ Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 slot _h1 _h2 ih1 ih2 =>
      exact HasType.letBind Δ_ Sigma' Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2
        slot (ih1 hsub) (ih2 hsub)
  | copy Δ_ _ Γ1 Γ2 e0 ds ep _h ih =>
      exact HasType.copy Δ_ Sigma' Γ1 Γ2 e0 ds ep (ih hsub)
  | letpair Δ_ _ Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY
            _h1 _h2 ih1 ih2 =>
      exact HasType.letpair Δ_ Sigma' Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2
        slotX slotY (ih1 hsub) (ih2 hsub)
  | tpair Δ_ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.tpair Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2
        (ih1 hsub) (ih2 hsub)
  | fst Δ_ _ Γ1 Γ2 e0 t1 t2 ep _h ih =>
      exact HasType.fst Δ_ Sigma' Γ1 Γ2 e0 t1 t2 ep (ih hsub)
  | snd Δ_ _ Γ1 Γ2 e0 t1 t2 ep _h ih =>
      exact HasType.snd Δ_ Sigma' Γ1 Γ2 e0 t1 t2 ep (ih hsub)
  | const Δ_ _ Γ_ v ds => exact HasType.const Δ_ Sigma' Γ_ v ds
  | tadd Δ_ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.tadd Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2
        (ih1 hsub) (ih2 hsub)
  | tmul Δ_ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.tmul Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2
        (ih1 hsub) (ih2 hsub)
  | tsum Δ_ _ Γ1 Γ2 e0 ds d ep _h hmem ih =>
      exact HasType.tsum Δ_ Sigma' Γ1 Γ2 e0 ds d ep (ih hsub) hmem
  | texpand Δ_ _ Γ1 Γ2 e0 ds d ep _h ih =>
      exact HasType.texpand Δ_ Sigma' Γ1 Γ2 e0 ds d ep (ih hsub)
  | uniformLike Δ_ _ Γ1 Γ2 e0 ds lo hi ep _h ih =>
      exact HasType.uniformLike Δ_ Sigma' Γ1 Γ2 e0 ds lo hi ep (ih hsub)
  | perform Δ_ _ Γ1 Γ2 op e0 tArg tRet ep _h hM ih =>
      exact HasType.perform Δ_ Sigma' Γ1 Γ2 op e0 tArg tRet ep (ih hsub) hM
  | handle Δ_ _ Γ1 Γ2 Γ3 body clauses t_ epsH epsB _hb hSubsH hClsH
           hCover _hcls ih_body ih_clauses =>
      exact HasType.handle Δ_ Sigma' Γ1 Γ2 Γ3 body clauses t_ epsH epsB
        (ih_body hsub) hSubsH hClsH hCover (ih_clauses hsub)
  | tgrad Δ_ _ Γ_ x ds dsOut body ep slot _h hsub_eff hSupp hCtxSupp ih =>
      exact HasType.tgrad Δ_ Sigma' Γ_ x ds dsOut body ep slot (ih hsub) hsub_eff
        hSupp hCtxSupp
  | tvmap Δ_ _ Γ_ x t1 t2 body ep d slot _h ih =>
      exact HasType.tvmap Δ_ Sigma' Γ_ x t1 t2 body ep d slot (ih hsub)
  | loc Δ_ _ Γ_ ell tv hlook =>
      exact HasType.loc Δ_ Sigma' Γ_ ell tv (hsub ell tv hlook)
  | subEff Δ_ _ Γ_ Γ'' e0 tv eps0 eps1 _h hSub ih =>
      exact HasType.subEff Δ_ Sigma' Γ_ Γ'' e0 tv eps0 eps1 (ih hsub) hSub
  | nil => exact ClausesTyped.nil _ Sigma' _ _ _
  | cons Δ_ _ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest slotX slotK
      hmatch _h_body _h_rest ih_body ih_rest hs =>
      exact ClausesTyped.cons Δ_ Sigma' Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest
        slotX slotK hmatch (ih_body hs) (ih_rest hs)

/-- Store-typing weakening at the granularity actually needed by a
    runtime term: agreement only on its explicit `Term.loc`
    references. This is the theorem shape required by the remaining
    `ctx` case. -/
theorem hasType_store_weaken_on_locRefs
    {Delta : CapCtx} {Sigma Sigma' : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (hsub : StoreTypOn (locRefs e) Sigma Sigma') :
    HasType Delta Sigma' Gamma e t eps Gamma' := by
  induction h using HasType.rec
    (motive_2 := fun Δ_ S_ Γ2_ Γ3_ t_ εR_ cls_ _ =>
      StoreTypOn (locRefsClauses cls_) S_ Sigma' →
        ClausesTyped Δ_ Sigma' Γ2_ Γ3_ t_ εR_ cls_) with
  | var Δ_ _ Γpre Γpost x tv =>
      exact HasType.var Δ_ Sigma' Γpre Γpost x tv
  | unit Δ_ _ Γ_ =>
      exact HasType.unit Δ_ Sigma' Γ_
  | abs Δ_ _ Γ1 Γ2 x t1 t2 eps_ body slot _h ih =>
      exact HasType.abs Δ_ Sigma' Γ1 Γ2 x t1 t2 eps_ body slot
        (ih (by simpa [locRefs] using hsub))
  | app Δ_ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.app Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2
        (ih1 (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        (ih2 (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | letBind Δ_ _ Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 slot _h1 _h2 ih1 ih2 =>
      exact HasType.letBind Δ_ Sigma' Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 slot
        (ih1 (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        (ih2 (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | copy Δ_ _ Γ1 Γ2 e0 ds ep _h ih =>
      exact HasType.copy Δ_ Sigma' Γ1 Γ2 e0 ds ep
        (ih (by simpa [locRefs] using hsub))
  | letpair Δ_ _ Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY
      _h1 _h2 ih1 ih2 =>
      exact HasType.letpair Δ_ Sigma' Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY
        (ih1 (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        (ih2 (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | tpair Δ_ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.tpair Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2
        (ih1 (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        (ih2 (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | fst Δ_ _ Γ1 Γ2 e0 t1 t2 ep _h ih =>
      exact HasType.fst Δ_ Sigma' Γ1 Γ2 e0 t1 t2 ep
        (ih (by simpa [locRefs] using hsub))
  | snd Δ_ _ Γ1 Γ2 e0 t1 t2 ep _h ih =>
      exact HasType.snd Δ_ Sigma' Γ1 Γ2 e0 t1 t2 ep
        (ih (by simpa [locRefs] using hsub))
  | const Δ_ _ Γ_ v ds =>
      exact HasType.const Δ_ Sigma' Γ_ v ds
  | tadd Δ_ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.tadd Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2
        (ih1 (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        (ih2 (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | tmul Δ_ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
      exact HasType.tmul Δ_ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2
        (ih1 (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        (ih2 (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | tsum Δ_ _ Γ1 Γ2 e0 ds d ep _h hmem ih =>
      exact HasType.tsum Δ_ Sigma' Γ1 Γ2 e0 ds d ep
        (ih (by simpa [locRefs] using hsub)) hmem
  | texpand Δ_ _ Γ1 Γ2 e0 ds d ep _h ih =>
      exact HasType.texpand Δ_ Sigma' Γ1 Γ2 e0 ds d ep
        (ih (by simpa [locRefs] using hsub))
  | uniformLike Δ_ _ Γ1 Γ2 e0 ds lo hi ep _h ih =>
      exact HasType.uniformLike Δ_ Sigma' Γ1 Γ2 e0 ds lo hi ep
        (ih (by simpa [locRefs] using hsub))
  | perform Δ_ _ Γ1 Γ2 op e0 tArg tRet ep _h hM ih =>
      exact HasType.perform Δ_ Sigma' Γ1 Γ2 op e0 tArg tRet ep
        (ih (by simpa [locRefs] using hsub)) hM
  | handle Δ_ _ Γ1 Γ2 Γ3 body clauses t_ epsH epsB _hb hSubsH hClsH
      hCover _hcls ih_body ih_clauses =>
      exact HasType.handle Δ_ Sigma' Γ1 Γ2 Γ3 body clauses t_ epsH epsB
        (ih_body (StoreTypOn.append_left (by simpa [locRefs] using hsub)))
        hSubsH hClsH hCover
        (ih_clauses (StoreTypOn.append_right (by simpa [locRefs] using hsub)))
  | tgrad Δ_ _ Γ_ x ds dsOut body ep slot _h hsub_eff hSupp hCtxSupp ih =>
      exact HasType.tgrad Δ_ Sigma' Γ_ x ds dsOut body ep slot
        (ih (by simpa [locRefs] using hsub)) hsub_eff hSupp hCtxSupp
  | tvmap Δ_ _ Γ_ x t1 t2 body ep d slot _h ih =>
      exact HasType.tvmap Δ_ Sigma' Γ_ x t1 t2 body ep d slot
        (ih (by simpa [locRefs] using hsub))
  | loc Δ_ _ Γ_ ell tv hlook =>
      exact HasType.loc Δ_ Sigma' Γ_ ell tv (hsub ell tv (by simp [locRefs]) hlook)
  | subEff Δ_ _ Γ_ Γ'' e0 tv eps0 eps1 _h hSub ih =>
      exact HasType.subEff Δ_ Sigma' Γ_ Γ'' e0 tv eps0 eps1
        (ih (by simpa [locRefs] using hsub)) hSub
  | nil =>
      exact ClausesTyped.nil _ Sigma' _ _ _
  | cons Δ_ _ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest slotX slotK
      hmatch _h_body _h_rest ih_body ih_rest hs =>
      exact ClausesTyped.cons Δ_ Sigma' Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest
        slotX slotK hmatch
        (ih_body (StoreTypOn.append_left (by simpa [locRefsClauses] using hs)))
        (ih_rest (StoreTypOn.append_right (by simpa [locRefsClauses] using hs)))

theorem clausesTyped_store_weaken_on_locRefs
    {Delta : CapCtx} {Sigma Sigma' : StoreTyp} {Gamma2 Gamma3 : LinearCtx}
    {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls)
    (hsub : StoreTypOn (locRefsClauses cls) Sigma Sigma') :
    ClausesTyped Delta Sigma' Gamma2 Gamma3 t epsR cls := by
  match h with
  | ClausesTyped.nil _ _ Γ2 t_ epsR_ =>
      exact ClausesTyped.nil _ Sigma' Γ2 t_ epsR_
  | ClausesTyped.cons _ _ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest
      slotX slotK hmatch hBody hRest =>
      exact ClausesTyped.cons _ Sigma' Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest
        slotX slotK hmatch
        (hasType_store_weaken_on_locRefs hBody
          (StoreTypOn.append_left (by simpa [locRefsClauses] using hsub)))
        (clausesTyped_store_weaken_on_locRefs hRest
          (StoreTypOn.append_right (by simpa [locRefsClauses] using hsub)))
-- Wave 5r: original proof preserved below for reference.
/-
  -- Term-mode recursor application with both motives pinned.
  refine
    @HasType.rec
      (fun Δ_ S_ Γ_ e_ t_ ε_ Γ'_ _ =>
        StoreTypSub S_ Sigma' → HasType Δ_ Sigma' Γ_ e_ t_ ε_ Γ'_)
      (fun Δ_ S_ Γ2_ Γ3_ t_ εR_ cls_ _ =>
        StoreTypSub S_ Sigma' → ClausesTyped Δ_ Sigma' Γ2_ Γ3_ t_ εR_ cls_)
      ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
      Delta Sigma Gamma e t eps Gamma' h hsub
  case _ =>
    intro Δ _ Γpre Γpost x tv hs
    exact HasType.var Δ Sigma' Γpre Γpost x tv
  case _ =>
    intro Δ _ Γ hs
    exact HasType.unit Δ Sigma' Γ
  case _ =>
    intro Δ _ Γ1 Γ2 x t1 t2 epsB body _h_body ih hs
    exact HasType.abs Δ Sigma' Γ1 Γ2 x t1 t2 epsB body (ih hs)
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 epsF eps1 eps2 _h1 _h2 ih1 ih2 hs
    exact HasType.app Δ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 epsF eps1 eps2 (ih1 hs) (ih2 hs)
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 hs
    exact HasType.letBind Δ Sigma' Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 (ih1 hs) (ih2 hs)
  case _ =>
    intro Δ _ Γ1 Γ2 e0 ds ep _h ih hs
    exact HasType.copy Δ Sigma' Γ1 Γ2 e0 ds ep (ih hs)
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 _h1 _h2 ih1 ih2 hs
    exact HasType.letpair Δ Sigma' Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 (ih1 hs) (ih2 hs)
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 hs
    exact HasType.tpair Δ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 (ih1 hs) (ih2 hs)
  case _ =>
    intro Δ _ Γ1 Γ2 e0 t1 t2 ep _h ih hs
    exact HasType.fst Δ Sigma' Γ1 Γ2 e0 t1 t2 ep (ih hs)
  case _ =>
    intro Δ _ Γ1 Γ2 e0 t1 t2 ep _h ih hs
    exact HasType.snd Δ Sigma' Γ1 Γ2 e0 t1 t2 ep (ih hs)
  case _ =>
    intro Δ _ Γ v ds hs
    exact HasType.const Δ Sigma' Γ v ds
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 hs
    exact HasType.tadd Δ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 (ih1 hs) (ih2 hs)
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 hs
    exact HasType.tmul Δ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 (ih1 hs) (ih2 hs)
  case _ =>
    intro Δ _ Γ1 Γ2 e0 ds d ep _h hmem ih hs
    exact HasType.tsum Δ Sigma' Γ1 Γ2 e0 ds d ep (ih hs) hmem
  case _ =>
    intro Δ _ Γ1 Γ2 e0 ds d ep _h ih hs
    exact HasType.texpand Δ Sigma' Γ1 Γ2 e0 ds d ep (ih hs)
  case _ =>
    intro Δ _ Γ1 Γ2 e0 ds lo hi ep _h ih hs
    exact HasType.uniformLike Δ Sigma' Γ1 Γ2 e0 ds lo hi ep (ih hs)
  case _ =>
    intro Δ _ Γ1 Γ2 op e0 tArg tRet ep _h hmatch ih hs
    exact HasType.perform Δ Sigma' Γ1 Γ2 op e0 tArg tRet ep (ih hs) hmatch
  case _ =>
    intro Δ _ Γ1 Γ2 Γ3 body clauses tr epsH epsB _hb hHsubB hClIn hClCov _hcls
          ih_body ih_cls hs
    exact HasType.handle Δ Sigma' Γ1 Γ2 Γ3 body clauses tr epsH epsB
      (ih_body hs) hHsubB hClIn hClCov (ih_cls hs)
  case _ =>
    intro Δ _ Γ x ds dsOut e0 ep _h hsubEff ih hs
    exact HasType.tgrad Δ Sigma' Γ x ds dsOut e0 ep (ih hs) hsubEff
  case _ =>
    intro Δ _ Γ x t1 t2 e0 ep d _h ih hs
    exact HasType.tvmap Δ Sigma' Γ x t1 t2 e0 ep d (ih hs)
  case _ =>
    intro Δ _ Γ ell tv hlook hs
    exact HasType.loc Δ Sigma' Γ ell tv (hs ell tv hlook)
  case _ =>
    intro Δ _ Γ Γ'' e0 tv eps0 eps1 _h hSub ih hs
    exact HasType.subEff Δ Sigma' Γ Γ'' e0 tv eps0 eps1 (ih hs) hSub
  case _ =>
    intro Δ _ Γ2 tr epsR hs
    exact ClausesTyped.nil Δ Sigma' Γ2 tr epsR
  case _ =>
    intro Δ _ Γ2 Γ3 tr tArg tRet epsR op x k hb rest _hhb _hrest ih_hb ih_rest hs
    exact ClausesTyped.cons Δ Sigma' Γ2 Γ3 tr tArg tRet epsR op x k hb rest
      (ih_hb hs) (ih_rest hs)
-/

/-- Companion to `hasType_store_weaken`: weakens a `ClausesTyped`
    derivation under a monotone store-typing extension. Proved by a
    24-case term-mode `@HasType.rec` call that mirrors the HasType
    weakening proof (every constructor rebuilds at `Sigma'`). -/
theorem clausesTyped_store_weaken
    {Delta : CapCtx} {Sigma Sigma' : StoreTyp} {Gamma2 Gamma3 : LinearCtx}
    {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls)
    (hsub : StoreTypSub Sigma Sigma') :
    ClausesTyped Delta Sigma' Gamma2 Gamma3 t epsR cls := by
  match h with
  | ClausesTyped.nil _ _ Γ2 t_ epsR_ =>
      exact ClausesTyped.nil _ Sigma' Γ2 t_ epsR_
  | ClausesTyped.cons _ _ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest
                       slotX slotK hmatch h_body h_rest =>
      exact ClausesTyped.cons _ Sigma' Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest
        slotX slotK hmatch (hasType_store_weaken h_body hsub)
        (clausesTyped_store_weaken h_rest hsub)
/-  -- Wave 5r tombstone: store_weaken proof needs slot-param update.
  refine
    @ClausesTyped.rec
      (fun Δ_ S_ Γ_ e_ t_ ε_ Γ'_ _ =>
        StoreTypSub S_ Sigma' → HasType Δ_ Sigma' Γ_ e_ t_ ε_ Γ'_)
      (fun Δ_ S_ Γ2_ Γ3_ t_ εR_ cls_ _ =>
        StoreTypSub S_ Sigma' → ClausesTyped Δ_ Sigma' Γ2_ Γ3_ t_ εR_ cls_)
      ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
      Delta Sigma Gamma2 Gamma3 t epsR cls h hsub
  -- All 24 cases: 22 HasType cases then nil/cons. Each is the same
  -- structural re-application used in `hasType_store_weaken`.
  case _ => intro Δ _ Γpre Γpost x tv hs
            exact HasType.var Δ Sigma' Γpre Γpost x tv
  case _ => intro Δ _ Γ hs
            exact HasType.unit Δ Sigma' Γ
  case _ => intro Δ _ Γ1 Γ2 x t1 t2 epsB body _h_body ih hs
            exact HasType.abs Δ Sigma' Γ1 Γ2 x t1 t2 epsB body (ih hs)
  case _ => intro Δ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 epsF eps1 eps2 _h1 _h2 ih1 ih2 hs
            exact HasType.app Δ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 epsF eps1 eps2
              (ih1 hs) (ih2 hs)
  case _ => intro Δ _ Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 hs
            exact HasType.letBind Δ Sigma' Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2
              (ih1 hs) (ih2 hs)
  case _ => intro Δ _ Γ1 Γ2 e0 ds ep _h ih hs
            exact HasType.copy Δ Sigma' Γ1 Γ2 e0 ds ep (ih hs)
  case _ => intro Δ _ Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 _h1 _h2 ih1 ih2 hs
            exact HasType.letpair Δ Sigma' Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr
              eps1 eps2 (ih1 hs) (ih2 hs)
  case _ => intro Δ _ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 hs
            exact HasType.tpair Δ Sigma' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2
              (ih1 hs) (ih2 hs)
  case _ => intro Δ _ Γ1 Γ2 e0 t1 t2 ep _h ih hs
            exact HasType.fst Δ Sigma' Γ1 Γ2 e0 t1 t2 ep (ih hs)
  case _ => intro Δ _ Γ1 Γ2 e0 t1 t2 ep _h ih hs
            exact HasType.snd Δ Sigma' Γ1 Γ2 e0 t1 t2 ep (ih hs)
  case _ => intro Δ _ Γ v ds hs
            exact HasType.const Δ Sigma' Γ v ds
  case _ => intro Δ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 hs
            exact HasType.tadd Δ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2
              (ih1 hs) (ih2 hs)
  case _ => intro Δ _ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 hs
            exact HasType.tmul Δ Sigma' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2
              (ih1 hs) (ih2 hs)
  case _ => intro Δ _ Γ1 Γ2 e0 ds d ep _h hmem ih hs
            exact HasType.tsum Δ Sigma' Γ1 Γ2 e0 ds d ep (ih hs) hmem
  case _ => intro Δ _ Γ1 Γ2 e0 ds d ep _h ih hs
            exact HasType.texpand Δ Sigma' Γ1 Γ2 e0 ds d ep (ih hs)
  case _ => intro Δ _ Γ1 Γ2 e0 ds lo hi ep _h ih hs
            exact HasType.uniformLike Δ Sigma' Γ1 Γ2 e0 ds lo hi ep (ih hs)
  case _ => intro Δ _ Γ1 Γ2 op e0 tArg tRet ep _h hmatch ih hs
            exact HasType.perform Δ Sigma' Γ1 Γ2 op e0 tArg tRet ep
              (ih hs) hmatch
  case _ => intro Δ _ Γ1 Γ2 Γ3 body clauses tr epsH epsB _hb hHsubB hClIn
                  hClCov _hcls ih_body ih_cls hs
            exact HasType.handle Δ Sigma' Γ1 Γ2 Γ3 body clauses tr epsH epsB
              (ih_body hs) hHsubB hClIn hClCov (ih_cls hs)
  case _ => intro Δ _ Γ x ds dsOut e0 ep _h hsubEff ih hs
            exact HasType.tgrad Δ Sigma' Γ x ds dsOut e0 ep (ih hs) hsubEff
  case _ => intro Δ _ Γ x t1 t2 e0 ep d _h ih hs
            exact HasType.tvmap Δ Sigma' Γ x t1 t2 e0 ep d (ih hs)
  case _ => intro Δ _ Γ ell tv hlook hs
            exact HasType.loc Δ Sigma' Γ ell tv (hs ell tv hlook)
  case _ => intro Δ _ Γ Γ'' e0 tv eps0 eps1 _h hSub ih hs
            exact HasType.subEff Δ Sigma' Γ Γ'' e0 tv eps0 eps1 (ih hs) hSub
  case _ => intro Δ _ Γ2 tr epsR hs
            exact ClausesTyped.nil Δ Sigma' Γ2 tr epsR
  case _ => intro Δ _ Γ2 Γ3 tr tArg tRet epsR op x k hb rest _hhb _hrest
                  ih_hb ih_rest hs
            exact ClausesTyped.cons Δ Sigma' Γ2 Γ3 tr tArg tRet epsR op x k
              hb rest (ih_hb hs) (ih_rest hs)
-/

/-- Plug-local app inversion: Preservation's sibling file Progress.lean
    depends on Preservation, so app_inv / letBind_inv / letpair_inv are
    defined there. We re-derive the shapes we need here under
    `plug_`-prefixed names to sidestep the import cycle. -/
theorem HasType.plug_app_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.app e1 e2) t eps Gamma3) :
    ∃ Gamma2 t1 epsBody eps1 eps2,
      HasType Delta Sigma Gamma1 e1 (Typ.arrow t1 t epsBody) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t1 eps2 Gamma3 ∧
      SubEffRow (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps := by
  generalize heq : Term.app e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | app _ _ _ Γ2 _ _ _ t1 _ epsBody eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, t1, epsBody, eps1, eps2, h1, h2, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Γ2, t1, epsBody, eps1, eps2, h1, h2, hwit⟩ := ih heq
      exact ⟨Γ2, t1, epsBody, eps1, eps2, h1, h2,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Plug-local letBind inversion with SubEffRow witness. -/
theorem HasType.plug_letBind_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps Gamma_out) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 : Typ) (eps1 eps2 : EffectRow)
      (slot : Option Typ),
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t eps2
              (Gamma3 ++ [(x, slot)]) ∧
      Gamma_out = Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Γ2 Γ3 _ _ _ t1 _ eps1 eps2 slot h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, eps1, eps2, slot, h1, h2, rfl, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Γ2, Γ3, t1, eps1, eps2, slot, h1, h2, hout, hwit⟩ := ih heq
      exact ⟨Γ2, Γ3, t1, eps1, eps2, slot, h1, h2, hout,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Plug-local letpair inversion with SubEffRow witness. -/
theorem HasType.plug_letpair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps Gamma_out) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 t2 : Typ) (eps1 eps2 : EffectRow)
      (slotX slotY : Option Typ),
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
              (Gamma3 ++ [(x, slotX), (y, slotY)]) ∧
      Gamma_out = Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1 t2 _ eps1 eps2 slotX slotY h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, rfl, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Γ2, Γ3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, hout, hwit⟩ := ih heq
      exact ⟨Γ2, Γ3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, hout,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)
-- Wave 5r: old proof below
/-
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1' t2' _ eps1' eps2' h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1', t2', eps1', eps2', rfl, h1, h2, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Γ2, Γ3, t1', t2', eps1', eps2', hfilt, h1, h2, hwit⟩ := ih heq
      exact ⟨Γ2, Γ3, t1', t2', eps1', eps2', hfilt, h1, h2,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)
-/

/-- Plug-local pair inversion (no pair-type assumption). -/
theorem HasType.plug_pair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) t eps Gamma3) :
    ∃ Gamma2 t1 t2 eps1 eps2,
      t = Typ.pair t1 t2 ∧
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.pair e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ Γ2 _ _ _ t1 t2 eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, t1, t2, eps1, eps2, rfl, h1, h2, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Γ2, t1, t2, eps1, eps2, ht, h1, h2, hwit⟩ := ih heq
      exact ⟨Γ2, t1, t2, eps1, eps2, ht, h1, h2,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Plug-local add inversion with SubEffRow witness. -/
theorem HasType.plug_add_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.add e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.add e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tadd _ _ _ Γ2 _ _ _ ds eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨ds, Γ2, eps1, eps2, rfl, h1, h2, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨ds, Γ2, eps1, eps2, ht, h1, h2, hwit⟩ := ih heq
      exact ⟨ds, Γ2, eps1, eps2, ht, h1, h2,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Plug-local mul inversion with SubEffRow witness. -/
theorem HasType.plug_mul_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.mul e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.mul e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tmul _ _ _ Γ2 _ _ _ ds eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨ds, Γ2, eps1, eps2, rfl, h1, h2, fun _ hh => hh⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨ds, Γ2, eps1, eps2, ht, h1, h2, hwit⟩ := ih heq
      exact ⟨ds, Γ2, eps1, eps2, ht, h1, h2,
             fun op hop => hSub op (hwit op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Perform inversion: extract the argument sub-derivation at the
    operation's argument type. The result type equals the operation's
    return type via `OpSigMatch`. The exact effect-row equation is
    dropped but the operation is guaranteed to be in the outer eps via
    `hasType_perform_eff_mem`. -/
theorem HasType.perform_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {op : EffectLabel} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.perform op e) t eps Gamma2) :
    ∃ tArg eps0,
      HasType Delta Sigma Gamma1 e tArg eps0 Gamma2 ∧
      OpSigMatch op tArg t ∧
      SubEffRow (EffectRow.union [op] eps0) eps := by
  generalize heq : Term.perform op e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | perform _ _ _ _ _ _ tArg _ eps0 h' hmatch _ =>
      cases heq
      exact ⟨tArg, eps0, h', hmatch, fun _ h => h⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨tArg, eps0, h', hmatch, hwit⟩ := ih heq
      refine ⟨tArg, eps0, h', hmatch, ?_⟩
      intro op' hop'
      exact hSub op' (hwit op' hop')
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Strengthened handle inversion (Wave C3): returns a body
    sub-derivation plus a `SubEffRow` witness tying the body's
    effect row (minus `epsH`) to the outer effect row via `removeOps`.
    Necessary for the `plug_preserves_typing` handle case. -/
theorem HasType.handle_inv_strong
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {body : Term} {clauses : List (EffectLabel × String × String × Term)}
    {epsH : EffectRow} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.handle epsH body clauses) t eps Gamma3) :
    ∃ Gamma2 epsB,
      HasType Delta Sigma Gamma1 body t epsB Gamma2 ∧
      (∀ op ∈ epsH, op ∈ epsB) ∧
      (∀ cl ∈ clauses, cl.1 ∈ epsH) ∧
      (∀ op ∈ epsH, ∃ cl ∈ clauses, cl.1 = op) ∧
      ClausesTyped Delta Sigma Gamma2 Gamma3 t
                   (EffectRow.removeOps epsB epsH) clauses ∧
      SubEffRow (EffectRow.removeOps epsB epsH) eps := by
  generalize heq : Term.handle epsH body clauses = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | handle _ _ _ Γ2 _ _ _ _ _ epsB hb hHsubB hClIn hClCov hcls _ _ =>
      cases heq
      exact ⟨Γ2, epsB, hb, hHsubB, hClIn, hClCov, hcls, fun _ h => h⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Γ2, epsB, hb, hHsubB, hClIn, hClCov, hcls, hwit⟩ := ih heq
      refine ⟨Γ2, epsB, hb, hHsubB, hClIn, hClCov, hcls, ?_⟩
      intro op hop
      exact hSub op (hwit op hop)
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Vmap inversion with subEff widening. The term itself is pure, so
    we only recover the body derivation and the exact arrow result
    type; any outer effect-row widening is handled separately by
    reapplying `HasType.subEff` to the rebuilt abstraction. -/
theorem HasType.vmap_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {x : String} {t1 : Typ} {d : Dim} {body : Term}
    {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.vmap x t1 d body) t eps Gamma2) :
    Gamma1 = Gamma2 ∧
    ∃ t2 epsBody slot,
      t = Typ.arrow (addDim d t1) (addDim d t2) epsBody ∧
      HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) body t2 epsBody
        (Gamma2 ++ [(x, slot)]) := by
  generalize heq : Term.vmap x t1 d body = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tvmap Delta Sigma Gamma x' t1' t2' body' epsBody d' slot hBody =>
      cases heq
      exact ⟨rfl, t2', epsBody, slot, rfl, hBody⟩
  | subEff Delta Sigma Gamma Gamma' e t' eps0 eps1 hInner hSub ih =>
      obtain ⟨hGamma, t2, epsBody, slot, ht, hBody⟩ := ih heq
      exact ⟨hGamma, t2, epsBody, slot, ht, hBody⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Grad inversion with trailing `subEff` stripped. The term itself is
    pure, so the interesting payload is the body derivation and the
    exact arrow result type. -/
theorem HasType.grad_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {x : String} {tArg tOut : Typ} {body : Term}
    {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.grad x tArg tOut body) t eps Gamma2) :
    Gamma1 = Gamma2 ∧
    ∃ ds dsOut epsBody slot,
      tArg = Typ.tensor ds ∧
      tOut = Typ.tensor dsOut ∧
      t = Typ.arrow (Typ.tensor ds)
            (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) epsBody) [] ∧
      subsetEffRow epsBody DiffCompat = true ∧
      AdjointSupported body ∧
      AdjointFreeCtxSupported (Gamma1 ++ [(x, some (Typ.tensor ds))]) body ∧
      HasType (Capability.diff :: Delta) Sigma
        (Gamma1 ++ [(x, some (Typ.tensor ds))]) body (Typ.tensor dsOut) epsBody
        (Gamma2 ++ [(x, slot)]) := by
  generalize heq : Term.grad x tArg tOut body = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tgrad Delta Sigma Gamma x' ds dsOut body' epsBody slot hBody hsub hSupp hCtxSupp =>
      cases heq
      exact ⟨rfl, ds, dsOut, epsBody, slot, rfl, rfl, rfl, hsub, hSupp, hCtxSupp, hBody⟩
  | subEff Delta Sigma Gamma Gamma' e t' eps0 eps1 hInner hSub ih =>
      obtain ⟨hGamma, ds, dsOut, epsBody, slot, htArg, htOut, ht, hDiff, hSupp, hCtxSupp, hBody⟩ := ih heq
      exact ⟨hGamma, ds, dsOut, epsBody, slot, htArg, htOut, ht, hDiff, hSupp, hCtxSupp, hBody⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

private theorem freshName_length_gt_used
    {used : List String} {base : String} {m : Nat}
    (hgt : maxStringLength used < m) :
    maxStringLength used < (freshName base m).toList.length := by
  rw [freshName_toList, List.length_append, List.length_cons, List.length_replicate]
  omega

private theorem adjointNamesFresh_of_length_bound
    {Γ : LinearCtx} {n : Nat}
    (hn : maxStringLength (linearCtxDom Γ) < n) :
    AdjointNamesFresh n Γ := by
  intro m hm _base _hbase hmem
  have hle : (freshName _base m).toList.length ≤ maxStringLength (linearCtxDom Γ) :=
    mem_maxStringLength hmem
  have hgtm : maxStringLength (linearCtxDom Γ) < m := Nat.lt_of_lt_of_le hn hm
  have hgtlen :
      maxStringLength (linearCtxDom Γ) < (freshName _base m).toList.length :=
    freshName_length_gt_used (used := linearCtxDom Γ) hgtm
  exact Nat.not_lt_of_ge hle hgtlen

/-- Extract the typed hole term from a closed one-frame evaluation
    context. The inner term may have a different type/effect row from
    the whole plugged term, but because the outer derivation starts
    from `[]`, every intermediate context exposed by inversion
    collapses back to `[]`. -/
theorem HasType.plug_inner_closed
    {Sigma : StoreTyp} {E : EvalCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType [] Sigma [] (plug E e) t eps []) :
    ∃ t0 eps0, HasType [] Sigma [] e t0 eps0 [] := by
  cases E with
  | hole =>
      exact ⟨t, eps, by simpa [plug] using h⟩
  | fst tRight =>
      have h' : HasType [] Sigma [] (Term.fst tRight e) t eps [] := by
        simpa [plug] using h
      exact ⟨Typ.pair t tRight, eps, HasType.fst_inv h'⟩
  | snd tLeft =>
      have h' : HasType [] Sigma [] (Term.snd tLeft e) t eps [] := by
        simpa [plug] using h
      exact ⟨Typ.pair tLeft t, eps, HasType.snd_inv h'⟩
  | copy =>
      have h' : HasType [] Sigma [] (Term.copy e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.copy_inv h'
      subst hteq
      exact ⟨Typ.tensor ds, eps, h_e⟩
  | sum d =>
      have h' : HasType [] Sigma [] (Term.sum e d) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, _hmem, h_e⟩ := HasType.sum_inv h'
      subst hteq
      exact ⟨Typ.tensor ds, eps, h_e⟩
  | expand d =>
      have h' : HasType [] Sigma [] (Term.expand e d) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.expand_inv h'
      subst hteq
      exact ⟨Typ.tensor ds, eps, h_e⟩
  | uniformLike lo hi =>
      have h' : HasType [] Sigma [] (Term.uniformLike e lo hi) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, eps0, hteq, h_e, _hsub⟩ := HasType.uniformLike_inv h'
      subst hteq
      exact ⟨Typ.tensor ds, eps0, h_e⟩
  | perform op =>
      have h' : HasType [] Sigma [] (Term.perform op e) t eps [] := by
        simpa [plug] using h
      obtain ⟨tArg, eps0, h_e, _hmatch, _hsub⟩ := HasType.perform_inv h'
      exact ⟨tArg, eps0, h_e⟩
  | appL e2 =>
      have h' : HasType [] Sigma [] (Term.app e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, epsBody, eps1, eps2, h_e1, _h_e2, _hsub⟩ :=
        HasType.plug_app_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_e1
      subst hGammaMid
      exact ⟨Typ.arrow t1 t epsBody, eps1, h_e1⟩
  | appR v1 =>
      have h' : HasType [] Sigma [] (Term.app v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, _epsBody, _eps1, eps2, h_v1, h_e2, _hsub⟩ :=
        HasType.plug_app_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_v1
      subst hGammaMid
      exact ⟨t1, eps2, h_e2⟩
  | letBind x e2 =>
      have h' : HasType [] Sigma [] (Term.letBind x e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, _Gamma3, t1, eps1, _eps2, _slot, h_e1, _h_e2, _hGammaOut, _hsub⟩ :=
        HasType.plug_letBind_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_e1
      subst hGammaMid
      exact ⟨t1, eps1, h_e1⟩
  | letpair x z e2 =>
      have h' : HasType [] Sigma [] (Term.letpair x z e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, _Gamma3, t1, t2, eps1, _eps2, _slotX, _slotY,
              h_e1, _h_e2, _hGammaOut, _hsub⟩ :=
        HasType.plug_letpair_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_e1
      subst hGammaMid
      exact ⟨Typ.pair t1 t2, eps1, h_e1⟩
  | pairL e2 =>
      have h' : HasType [] Sigma [] (Term.pair e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, _t2, eps1, _eps2, _hteq, h_e1, _h_e2, _hsub⟩ :=
        HasType.plug_pair_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_e1
      subst hGammaMid
      exact ⟨t1, eps1, h_e1⟩
  | pairR v1 =>
      have h' : HasType [] Sigma [] (Term.pair v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, _t1, t2, _eps1, eps2, _hteq, h_v1, h_e2, _hsub⟩ :=
        HasType.plug_pair_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_v1
      subst hGammaMid
      exact ⟨t2, eps2, h_e2⟩
  | addL e2 =>
      have h' : HasType [] Sigma [] (Term.add e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, eps1, _eps2, _hteq, h_e1, _h_e2, _hsub⟩ :=
        HasType.plug_add_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_e1
      subst hGammaMid
      exact ⟨Typ.tensor ds, eps1, h_e1⟩
  | addR v1 =>
      have h' : HasType [] Sigma [] (Term.add v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, _eps1, eps2, _hteq, h_v1, h_e2, _hsub⟩ :=
        HasType.plug_add_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_v1
      subst hGammaMid
      exact ⟨Typ.tensor ds, eps2, h_e2⟩
  | mulL e2 =>
      have h' : HasType [] Sigma [] (Term.mul e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, eps1, _eps2, _hteq, h_e1, _h_e2, _hsub⟩ :=
        HasType.plug_mul_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_e1
      subst hGammaMid
      exact ⟨Typ.tensor ds, eps1, h_e1⟩
  | mulR v1 =>
      have h' : HasType [] Sigma [] (Term.mul v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, _eps1, eps2, _hteq, h_v1, h_e2, _hsub⟩ :=
        HasType.plug_mul_inv h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_v1
      subst hGammaMid
      exact ⟨Typ.tensor ds, eps2, h_e2⟩
  | handle epsH clauses =>
      have h' : HasType [] Sigma [] (Term.handle epsH e clauses) t eps [] := by
        simpa [plug] using h
      obtain ⟨GammaMid, epsB, h_body, _hHsubB, _hClIn, _hClCov, _hcls, _hsub⟩ :=
        HasType.handle_inv_strong h'
      have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input h_body
      subst hGammaMid
      exact ⟨t, epsB, h_body⟩

/-- Multi-frame variant of `HasType.plug_inner_closed`. -/
theorem HasType.multiPlug_inner_closed
    {Sigma : StoreTyp} {Es : EvalCtxChain} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType [] Sigma [] (multiPlug Es e) t eps []) :
    ∃ t0 eps0, HasType [] Sigma [] e t0 eps0 [] := by
  induction Es generalizing Sigma t eps with
  | nil =>
      exact ⟨t, eps, by simpa [multiPlug] using h⟩
  | cons E Es ih =>
      have h' : HasType [] Sigma [] (plug E (multiPlug Es e)) t eps [] := by
        simpa [multiPlug] using h
      rcases HasType.plug_inner_closed h' with ⟨t0, eps0, h_inner⟩
      exact ih h_inner

theorem plug_preserves_typing
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {E : EvalCtx} {e e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (plug E e) t eps Gamma')
    (h_inner : ∀ {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow},
       HasType Delta Sigma Gamma0 e t0 eps0 Gamma0' →
       ∃ Sigma2, HasType Delta Sigma2 Gamma0 e' t0 eps0 Gamma0' ∧
                 StoreTypSub Sigma Sigma2) :
    ∃ Sigma2, HasType Delta Sigma2 Gamma (plug E e') t eps Gamma' ∧
              StoreTypSub Sigma Sigma2 := by
  cases E with
  | hole => simpa [plug] using h_inner h
  | fst tRight =>
      have h' : HasType Delta Sigma Gamma (Term.fst tRight e) t eps Gamma' := by
        simpa [plug] using h
      let h_e := HasType.fst_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨S2, HasType.fst _ S2 _ _ _ _ tRight _ h_e', h_sub⟩
  | snd tLeft =>
      have h' : HasType Delta Sigma Gamma (Term.snd tLeft e) t eps Gamma' := by
        simpa [plug] using h
      let h_e := HasType.snd_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨S2, HasType.snd _ S2 _ _ _ tLeft _ _ h_e', h_sub⟩
  | copy =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.copy_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      exact ⟨S2, HasType.copy _ S2 _ _ _ ds _ h_e', h_sub⟩
  | sum d =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, hteq, _hmem, h_e⟩ := HasType.sum_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      exact ⟨S2, HasType.tsum _ S2 _ _ _ ds d _ h_e' _hmem, h_sub⟩
  | expand d =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.expand_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      exact ⟨S2, HasType.texpand _ S2 _ _ _ ds d _ h_e', h_sub⟩
  | uniformLike lo hi =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, eps0, hteq, h_e, hsub_eps⟩ := HasType.uniformLike_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      refine ⟨S2, ?_, h_sub⟩
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.uniformLike _ S2 _ _ _ ds lo hi eps0 h_e') hsub_eps
  | perform op =>
      have h' := by simpa [plug] using h
      obtain ⟨tArg, eps0, h_e, hmatch, hwit⟩ := HasType.perform_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      refine ⟨S2, ?_, h_sub⟩
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.perform _ S2 _ _ op _ tArg _ eps0 h_e' hmatch)
        (fun op' hop' => hwit op' hop')
  | appL e2 =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      obtain ⟨S2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨S2, ?_, h_sub⟩
      have h_e2' := hasType_store_weaken h_e2 h_sub
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.app _ S2 _ Γmid _ _ e2 t1 _ epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | appR v1 =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      obtain ⟨S2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨S2, ?_, h_sub⟩
      have h_e1' := hasType_store_weaken h_e1 h_sub
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.app _ S2 _ Γmid _ v1 _ t1 _ epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairL e2 =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      obtain ⟨S2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨S2, ?_, h_sub⟩
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hteq
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.tpair _ S2 _ Γmid _ _ e2 t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairR v1 =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      obtain ⟨S2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨S2, ?_, h_sub⟩
      have h_e1' := hasType_store_weaken h_e1 h_sub
      subst hteq
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.tpair _ S2 _ Γmid _ v1 _ t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addL e2 =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      obtain ⟨S2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨S2, ?_, h_sub⟩
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hteq
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.tadd _ S2 _ Γmid _ _ e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addR v1 =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      obtain ⟨S2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨S2, ?_, h_sub⟩
      have h_e1' := hasType_store_weaken h_e1 h_sub
      subst hteq
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.tadd _ S2 _ Γmid _ v1 _ ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulL e2 =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      obtain ⟨S2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨S2, ?_, h_sub⟩
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hteq
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.tmul _ S2 _ Γmid _ _ e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulR v1 =>
      have h' := by simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      obtain ⟨S2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨S2, ?_, h_sub⟩
      have h_e1' := hasType_store_weaken h_e1 h_sub
      subst hteq
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.tmul _ S2 _ Γmid _ v1 _ ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | letBind x e2 =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, eps1, eps2, slot, h_e1, h_e2, hΓout, hsub_eps⟩ :=
        HasType.plug_letBind_inv h'
      obtain ⟨S2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨S2, ?_, h_sub⟩
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hΓout
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.letBind _ S2 _ Γmid _ x e' e2 t1 _ eps1 eps2 slot h_e1' h_e2')
        hsub_eps
  | letpair x y e2 =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, t2, eps1, eps2, slotX, slotY,
              h_e1, h_e2, hΓout, hsub_eps⟩ :=
        HasType.plug_letpair_inv h'
      obtain ⟨S2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨S2, ?_, h_sub⟩
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hΓout
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.letpair _ S2 _ Γmid _ x y e' e2 t1 t2 _ eps1 eps2
          slotX slotY h_e1' h_e2')
        hsub_eps
  | handle epsH clauses =>
      have h' := by simpa [plug] using h
      obtain ⟨Γmid, epsB, hb, hHsubB, hClIn, hClCov, hcls, hsub_eps⟩ :=
        HasType.handle_inv_strong h'
      obtain ⟨S2, hb', h_sub⟩ := h_inner hb
      refine ⟨S2, ?_, h_sub⟩
      have hcls' := clausesTyped_store_weaken hcls h_sub
      exact HasType.subEff _ _ _ _ _ _ _ _
        (HasType.handle _ S2 _ Γmid _ e' clauses _ epsH epsB
          hb' hHsubB hClIn hClCov hcls')
        hsub_eps

/-- Multi-frame variant of `plug_preserves_typing`. -/
theorem multiPlug_preserves_typing
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {Es : EvalCtxChain} {e e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (multiPlug Es e) t eps Gamma')
    (h_inner : ∀ {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow},
       HasType Delta Sigma Gamma0 e t0 eps0 Gamma0' →
       ∃ Sigma2, HasType Delta Sigma2 Gamma0 e' t0 eps0 Gamma0' ∧
                 StoreTypSub Sigma Sigma2) :
    ∃ Sigma2, HasType Delta Sigma2 Gamma (multiPlug Es e') t eps Gamma' ∧
              StoreTypSub Sigma Sigma2 := by
  induction Es generalizing Sigma Gamma Gamma' t eps with
  | nil =>
      simpa [multiPlug] using h_inner h
  | cons E Es ih =>
      have h' : HasType Delta Sigma Gamma (plug E (multiPlug Es e)) t eps Gamma' := by
        simpa [multiPlug] using h
      refine plug_preserves_typing h' ?_
      intro Gamma0 Gamma0' t0 eps0 h_inner'
      exact ih h_inner' h_inner

/-- Base replacement for the captured-continuation proof: under any
    surrounding linear context, `perform op v` can be replaced by a
    fresh variable of the operation's return type, consuming exactly
    that fresh slot and preserving all pre-existing slots. -/
theorem perform_to_var_preserves_typing_prefixed
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' outer : LinearCtx}
    {op : EffectLabel} {v : Term} {t tRet : Typ}
    {eps : EffectRow} {y : String}
    (h : HasType Delta Sigma Gamma (Term.perform op v) t eps Gamma')
    (hv : IsValue v)
    (hsig : ∃ tArg, OpSigMatch op tArg tRet) :
    HasType Delta Sigma ((outer ++ [(y, some tRet)]) ++ Gamma)
      (Term.var y) t eps
      ((outer ++ [(y, none)]) ++ Gamma') := by
  obtain ⟨tArg, eps0, h_v, hmatch, hsub_eps⟩ := HasType.perform_inv h
  have hArgUnit : tArg = Typ.unit := by
    cases op <;> simpa [OpSigMatch, opArgType] using hmatch.1
  subst hArgUnit
  have hGamma : Gamma' = Gamma := hasType_unit_value_preserves_context h_v hv
  subst hGamma
  obtain ⟨tArgStep, hStepSig⟩ := hsig
  have hRet : t = tRet := OpSigMatch.ret_unique hmatch hStepSig
  subst hRet
  have hVar0 :
      HasType Delta Sigma ((outer ++ [(y, some t)]) ++ Gamma')
        (Term.var y) t []
        ((outer ++ [(y, none)]) ++ Gamma') := by
    simpa [List.append_assoc] using
      (HasType.var Delta Sigma outer Gamma' y t)
  exact HasType.subEff Delta Sigma
    (((outer ++ [(y, some t)]) ++ Gamma'))
    (((outer ++ [(y, none)]) ++ Gamma'))
    (Term.var y) t [] eps hVar0
    (by
      intro op' hop'
      cases hop')

/-- One-frame context transport for the captured-continuation proof.
    The replacement theorem `h_inner` may change the hole's input and
    output contexts by prefixing a live fresh slot on input and the
    corresponding consumed slot on output. -/
theorem plug_replace_with_prefixed_hole
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' outer : LinearCtx}
    {E : EvalCtx} {e e' : Term} {t : Typ} {eps : EffectRow}
    {y : String} {ty : Typ}
    (h : HasType Delta Sigma Gamma (plug E e) t eps Gamma')
    (h_inner :
      ∀ {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow},
        HasType Delta Sigma Gamma0 e t0 eps0 Gamma0' →
        HasType Delta Sigma ((outer ++ [(y, some ty)]) ++ Gamma0)
          e' t0 eps0
          ((outer ++ [(y, none)]) ++ Gamma0')) :
    HasType Delta Sigma ((outer ++ [(y, some ty)]) ++ Gamma)
      (plug E e') t eps
      ((outer ++ [(y, none)]) ++ Gamma') := by
  cases E with
  | hole =>
      simpa [plug] using h_inner h
  | fst tRight =>
      have h' : HasType Delta Sigma Gamma (Term.fst tRight e) t eps Gamma' := by
        simpa [plug] using h
      let h_e := HasType.fst_inv h'
      exact HasType.fst Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' t tRight eps (h_inner h_e)
  | snd tLeft =>
      have h' : HasType Delta Sigma Gamma (Term.snd tLeft e) t eps Gamma' := by
        simpa [plug] using h
      let h_e := HasType.snd_inv h'
      exact HasType.snd Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' tLeft t eps (h_inner h_e)
  | copy =>
      have h' : HasType Delta Sigma Gamma (Term.copy e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.copy_inv h'
      subst hteq
      exact HasType.copy Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' ds eps (h_inner h_e)
  | sum d =>
      have h' : HasType Delta Sigma Gamma (Term.sum e d) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, hteq, hmem, h_e⟩ := HasType.sum_inv h'
      subst hteq
      exact HasType.tsum Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' ds d eps (h_inner h_e) hmem
  | expand d =>
      have h' : HasType Delta Sigma Gamma (Term.expand e d) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.expand_inv h'
      subst hteq
      exact HasType.texpand Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' ds d eps (h_inner h_e)
  | uniformLike lo hi =>
      have h' : HasType Delta Sigma Gamma (Term.uniformLike e lo hi) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, eps0, hteq, h_e, hsub_eps⟩ := HasType.uniformLike_inv h'
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.uniformLike e' lo hi) (Typ.tensor ds)
        (EffectRow.union eps0 [EffectLabel.random]) eps
        (HasType.uniformLike Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ Gamma'))
          e' ds lo hi eps0 (h_inner h_e))
        hsub_eps
  | perform op =>
      have h' : HasType Delta Sigma Gamma (Term.perform op e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨tArg, eps0, h_e, hmatch, hsub_eps⟩ := HasType.perform_inv h'
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.perform op e') t
        (EffectRow.union [op] eps0) eps
        (HasType.perform Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ Gamma'))
          op e' tArg t eps0 (h_inner h_e) hmatch)
        hsub_eps
  | appL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.app e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      have h_e1' := h_inner h_e1
      have h_e2' := hasType_prefix_weaken h_e2 (outer ++ [(y, none)])
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.app e' e2) t
        (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps
        (HasType.app Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          e' e2 t1 t epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | appR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.app v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, epsBody, eps1, eps2, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      have h_v1' := hasType_prefix_weaken h_v1 (outer ++ [(y, some ty)])
      have h_e2' := h_inner h_e2
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.app v1 e') t
        (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps
        (HasType.app Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, some ty)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          v1 e' t1 t epsBody eps1 eps2 h_v1' h_e2')
        hsub_eps
  | letBind x e2 =>
      have h' : HasType Delta Sigma Gamma (Term.letBind x e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, Gamma3, t1, eps1, eps2, slot, h_e1, h_e2, hGammaOut, hsub_eps⟩ :=
        HasType.plug_letBind_inv h'
      have h_e1' := h_inner h_e1
      have h_e2' := hasType_prefix_weaken h_e2 (outer ++ [(y, none)])
      subst Gamma'
      have h_e2'' :
          HasType Delta Sigma
            (((outer ++ [(y, none)]) ++ GammaMid) ++ [(x, some t1)])
            e2 t eps2
            (((outer ++ [(y, none)]) ++ Gamma3) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using h_e2'
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma3))
        (Term.letBind x e' e2) t
        (EffectRow.union eps1 eps2) eps
        (HasType.letBind Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma3))
          x e' e2 t1 t eps1 eps2 slot h_e1' h_e2'')
        hsub_eps
  | letpair x z e2 =>
      have h' : HasType Delta Sigma Gamma (Term.letpair x z e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, Gamma3, t1, t2, eps1, eps2, slotX, slotY,
              h_e1, h_e2, hGammaOut, hsub_eps⟩ :=
        HasType.plug_letpair_inv h'
      have h_e1' := h_inner h_e1
      have h_e2' := hasType_prefix_weaken h_e2 (outer ++ [(y, none)])
      subst Gamma'
      have h_e2'' :
          HasType Delta Sigma
            (((outer ++ [(y, none)]) ++ GammaMid) ++ [(x, some t1), (z, some t2)])
            e2 t eps2
            (((outer ++ [(y, none)]) ++ Gamma3) ++ [(x, slotX), (z, slotY)]) := by
        simpa [List.append_assoc] using h_e2'
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma3))
        (Term.letpair x z e' e2) t
        (EffectRow.union eps1 eps2) eps
        (HasType.letpair Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma3))
          x z e' e2 t1 t2 t eps1 eps2 slotX slotY h_e1' h_e2'')
        hsub_eps
  | pairL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.pair e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      have h_e1' := h_inner h_e1
      have h_e2' := hasType_prefix_weaken h_e2 (outer ++ [(y, none)])
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.pair e' e2) (Typ.pair t1 t2)
        (EffectRow.union eps1 eps2) eps
        (HasType.tpair Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          e' e2 t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.pair v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, t1, t2, eps1, eps2, hteq, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      have h_v1' := hasType_prefix_weaken h_v1 (outer ++ [(y, some ty)])
      have h_e2' := h_inner h_e2
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.pair v1 e') (Typ.pair t1 t2)
        (EffectRow.union eps1 eps2) eps
        (HasType.tpair Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, some ty)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          v1 e' t1 t2 eps1 eps2 h_v1' h_e2')
        hsub_eps
  | addL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.add e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      have h_e1' := h_inner h_e1
      have h_e2' := hasType_prefix_weaken h_e2 (outer ++ [(y, none)])
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.add e' e2) (Typ.tensor ds)
        (EffectRow.union eps1 eps2) eps
        (HasType.tadd Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          e' e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.add v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, eps1, eps2, hteq, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      have h_v1' := hasType_prefix_weaken h_v1 (outer ++ [(y, some ty)])
      have h_e2' := h_inner h_e2
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.add v1 e') (Typ.tensor ds)
        (EffectRow.union eps1 eps2) eps
        (HasType.tadd Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, some ty)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          v1 e' ds eps1 eps2 h_v1' h_e2')
        hsub_eps
  | mulL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.mul e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      have h_e1' := h_inner h_e1
      have h_e2' := hasType_prefix_weaken h_e2 (outer ++ [(y, none)])
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.mul e' e2) (Typ.tensor ds)
        (EffectRow.union eps1 eps2) eps
        (HasType.tmul Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          e' e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.mul v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, GammaMid, eps1, eps2, hteq, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      have h_v1' := hasType_prefix_weaken h_v1 (outer ++ [(y, some ty)])
      have h_e2' := h_inner h_e2
      subst hteq
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.mul v1 e') (Typ.tensor ds)
        (EffectRow.union eps1 eps2) eps
        (HasType.tmul Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, some ty)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          v1 e' ds eps1 eps2 h_v1' h_e2')
        hsub_eps
  | handle epsH clauses =>
      have h' : HasType Delta Sigma Gamma (Term.handle epsH e clauses) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨GammaMid, epsB, h_body, hHsubB, hClIn, hClCov, hcls, hsub_eps⟩ :=
        HasType.handle_inv_strong h'
      have h_body' := h_inner h_body
      have hcls' :=
        clausesTyped_prefix_weaken hcls (outer ++ [(y, none)])
      exact HasType.subEff Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        (Term.handle epsH e' clauses) t
        (EffectRow.removeOps epsB epsH) eps
        (HasType.handle Delta Sigma
          (((outer ++ [(y, some ty)]) ++ Gamma))
          (((outer ++ [(y, none)]) ++ GammaMid))
          (((outer ++ [(y, none)]) ++ Gamma'))
          e' clauses t epsH epsB h_body' hHsubB hClIn hClCov hcls')
        hsub_eps

/-- Multi-frame version of `plug_replace_with_prefixed_hole`. -/
theorem multiPlug_replace_with_prefixed_hole
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' outer : LinearCtx}
    {Es : EvalCtxChain} {e e' : Term} {t : Typ} {eps : EffectRow}
    {y : String} {ty : Typ}
    (h : HasType Delta Sigma Gamma (multiPlug Es e) t eps Gamma')
    (h_inner :
      ∀ {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow},
        HasType Delta Sigma Gamma0 e t0 eps0 Gamma0' →
        HasType Delta Sigma ((outer ++ [(y, some ty)]) ++ Gamma0)
          e' t0 eps0
          ((outer ++ [(y, none)]) ++ Gamma0')) :
    HasType Delta Sigma ((outer ++ [(y, some ty)]) ++ Gamma)
      (multiPlug Es e') t eps
      ((outer ++ [(y, none)]) ++ Gamma') := by
  revert Gamma Gamma' t eps h
  induction Es with
  | nil =>
      intro Gamma Gamma' t eps h
      simpa [multiPlug] using h_inner h
  | cons E Es ih =>
      intro Gamma Gamma' t eps h
      have h' : HasType Delta Sigma Gamma (plug E (multiPlug Es e)) t eps Gamma' := by
        simpa [multiPlug] using h
      simpa [multiPlug] using
        plug_replace_with_prefixed_hole h'
          (fun {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow}
               (h_inner' : HasType Delta Sigma Gamma0 (multiPlug Es e) t0 eps0 Gamma0') =>
            ih (Gamma := Gamma0) (Gamma' := Gamma0') (t := t0) (eps := eps0) h_inner')

/-- Closed-program specialization of `plug_preserves_typing`.
    This is the form needed by top-level preservation: when the whole
    program is closed, every intermediate context exposed by the
    one-frame evaluation context inversions collapses back to `[]`,
    so the recursive preservation hypothesis only has to handle closed
    inner terms. -/
theorem plug_preserves_typing_closed
    {Sigma : StoreTyp}
    {E : EvalCtx} {e e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType [] Sigma [] (plug E e) t eps [])
    (h_inner : ∀ {t0 : Typ} {eps0 : EffectRow},
       HasType [] Sigma [] e t0 eps0 [] →
       ∃ Sigma2, HasType [] Sigma2 [] e' t0 eps0 [] ∧
                 StoreTypSub Sigma Sigma2) :
    ∃ Sigma2, HasType [] Sigma2 [] (plug E e') t eps [] ∧
              StoreTypSub Sigma Sigma2 := by
  cases E with
  | hole =>
      simpa [plug] using h_inner h
  | fst tRight =>
      have h' : HasType [] Sigma [] (Term.fst tRight e) t eps [] := by
        simpa [plug] using h
      let h_e := HasType.fst_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨Sigma2, HasType.fst [] Sigma2 [] [] e' t tRight eps h_e', h_sub⟩
  | snd tLeft =>
      have h' : HasType [] Sigma [] (Term.snd tLeft e) t eps [] := by
        simpa [plug] using h
      let h_e := HasType.snd_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨Sigma2, HasType.snd [] Sigma2 [] [] e' tLeft t eps h_e', h_sub⟩
  | copy =>
      have h' : HasType [] Sigma [] (Term.copy e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.copy_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      exact ⟨Sigma2, HasType.copy [] Sigma2 [] [] e' ds eps h_e', h_sub⟩
  | sum d =>
      have h' : HasType [] Sigma [] (Term.sum e d) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, hmem, h_e⟩ := HasType.sum_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      exact ⟨Sigma2, HasType.tsum [] Sigma2 [] [] e' ds d eps h_e' hmem, h_sub⟩
  | expand d =>
      have h' : HasType [] Sigma [] (Term.expand e d) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.expand_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      exact ⟨Sigma2, HasType.texpand [] Sigma2 [] [] e' ds d eps h_e', h_sub⟩
  | uniformLike lo hi =>
      have h' : HasType [] Sigma [] (Term.uniformLike e lo hi) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, eps0, hteq, h_e, hsub_eps⟩ := HasType.uniformLike_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.uniformLike e' lo hi) (Typ.tensor ds)
        (EffectRow.union eps0 [EffectLabel.random]) eps
        (HasType.uniformLike [] Sigma2 [] [] e' ds lo hi eps0 h_e')
        hsub_eps
  | perform op =>
      have h' : HasType [] Sigma [] (Term.perform op e) t eps [] := by
        simpa [plug] using h
      obtain ⟨tArg, eps0, h_e, hmatch, hsub_eps⟩ := HasType.perform_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.perform op e') t (EffectRow.union [op] eps0) eps
        (HasType.perform [] Sigma2 [] [] op e' tArg t eps0 h_e' hmatch)
        hsub_eps
  | appL e2 =>
      have h' : HasType [] Sigma [] (Term.app e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      have h_e2' := hasType_store_weaken h_e2 h_sub
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.app e' e2) t
        (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps
        (HasType.app [] Sigma2 [] [] [] e' e2 t1 t epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | appR v1 =>
      have h' : HasType [] Sigma [] (Term.app v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      have h_e1' := hasType_store_weaken h_e1 h_sub
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.app v1 e') t
        (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps
        (HasType.app [] Sigma2 [] [] [] v1 e' t1 t epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairL e2 =>
      have h' : HasType [] Sigma [] (Term.pair e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.pair e' e2) (Typ.pair t1 t2) (EffectRow.union eps1 eps2) eps
        (HasType.tpair [] Sigma2 [] [] [] e' e2 t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairR v1 =>
      have h' : HasType [] Sigma [] (Term.pair v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      have h_e1' := hasType_store_weaken h_e1 h_sub
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.pair v1 e') (Typ.pair t1 t2) (EffectRow.union eps1 eps2) eps
        (HasType.tpair [] Sigma2 [] [] [] v1 e' t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addL e2 =>
      have h' : HasType [] Sigma [] (Term.add e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.add e' e2) (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tadd [] Sigma2 [] [] [] e' e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addR v1 =>
      have h' : HasType [] Sigma [] (Term.add v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      have h_e1' := hasType_store_weaken h_e1 h_sub
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.add v1 e') (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tadd [] Sigma2 [] [] [] v1 e' ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulL e2 =>
      have h' : HasType [] Sigma [] (Term.mul e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      have h_e2' := hasType_store_weaken h_e2 h_sub
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.mul e' e2) (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tmul [] Sigma2 [] [] [] e' e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulR v1 =>
      have h' : HasType [] Sigma [] (Term.mul v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      have h_e1' := hasType_store_weaken h_e1 h_sub
      subst hteq
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.mul v1 e') (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tmul [] Sigma2 [] [] [] v1 e' ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | letBind x e2 =>
      have h' : HasType [] Sigma [] (Term.letBind x e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, eps1, eps2, slot, h_e1, h_e2, hΓout, hsub_eps⟩ :=
        HasType.plug_letBind_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      have hΓ3 : Γ3 = [] := by simpa using hΓout.symm
      subst hΓmid
      subst hΓ3
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      have h_e2' := hasType_store_weaken h_e2 h_sub
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.letBind x e' e2) t (EffectRow.union eps1 eps2) eps
        (HasType.letBind [] Sigma2 [] [] [] x e' e2 t1 t eps1 eps2 slot h_e1' h_e2')
        hsub_eps
  | letpair x y e2 =>
      have h' : HasType [] Sigma [] (Term.letpair x y e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, t2, eps1, eps2, slotX, slotY,
              h_e1, h_e2, hΓout, hsub_eps⟩ :=
        HasType.plug_letpair_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      have hΓ3 : Γ3 = [] := by simpa using hΓout.symm
      subst hΓmid
      subst hΓ3
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      have h_e2' := hasType_store_weaken h_e2 h_sub
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.letpair x y e' e2) t (EffectRow.union eps1 eps2) eps
        (HasType.letpair [] Sigma2 [] [] [] x y e' e2 t1 t2 t eps1 eps2
          slotX slotY h_e1' h_e2')
        hsub_eps
  | handle epsH clauses =>
      have h' : HasType [] Sigma [] (Term.handle epsH e clauses) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, epsB, hb, hHsubB, hClIn, hClCov, hcls, hsub_eps⟩ :=
        HasType.handle_inv_strong h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input hb
      subst hΓmid
      obtain ⟨Sigma2, hb', h_sub⟩ := h_inner hb
      have hcls' := clausesTyped_store_weaken hcls h_sub
      refine ⟨Sigma2, ?_, h_sub⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.handle epsH e' clauses) t (EffectRow.removeOps epsB epsH) eps
        (HasType.handle [] Sigma2 [] [] [] e' clauses t epsH epsB
          hb' hHsubB hClIn hClCov hcls')
        hsub_eps

/-- Closed-program context transport that carries exactly the
    frame-local store agreement needed by `ctx`, plus the post-step
    store well-formedness witness for the chosen `Sigma2`. The extra
    `locs` parameter lets recursive `ctx` steps thread outer-frame
    location agreement through nested plugs. -/
theorem plug_preserves_typing_closed_on_ctxLocRefs
    {Sigma : StoreTyp} {sigma2 : Store} {locs : List Loc}
    {E : EvalCtx} {e e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType [] Sigma [] (plug E e) t eps [])
    (h_inner : ∀ {t0 : Typ} {eps0 : EffectRow},
       HasType [] Sigma [] e t0 eps0 [] →
       ∃ Sigma2, HasType [] Sigma2 [] e' t0 eps0 [] ∧
                 StoreWf sigma2 Sigma2 ∧
                 StoreTypOn (ctxLocRefs E ++ locs) Sigma Sigma2) :
    ∃ Sigma2, HasType [] Sigma2 [] (plug E e') t eps [] ∧
              StoreWf sigma2 Sigma2 ∧
              StoreTypOn (ctxLocRefs E ++ locs) Sigma Sigma2 := by
  cases E with
  | hole =>
      rcases h_inner h with ⟨Sigma2, h_e', h_wf2, h_on⟩
      simpa [plug] using ⟨Sigma2, h_e', h_wf2, h_on⟩
  | fst tRight =>
      have h' : HasType [] Sigma [] (Term.fst tRight e) t eps [] := by
        simpa [plug] using h
      let h_e := HasType.fst_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      exact ⟨Sigma2, HasType.fst [] Sigma2 [] [] e' t tRight eps h_e', h_wf2, h_on⟩
  | snd tLeft =>
      have h' : HasType [] Sigma [] (Term.snd tLeft e) t eps [] := by
        simpa [plug] using h
      let h_e := HasType.snd_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      exact ⟨Sigma2, HasType.snd [] Sigma2 [] [] e' tLeft t eps h_e', h_wf2, h_on⟩
  | copy =>
      have h' : HasType [] Sigma [] (Term.copy e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.copy_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      subst hteq
      exact ⟨Sigma2, HasType.copy [] Sigma2 [] [] e' ds eps h_e', h_wf2, h_on⟩
  | sum d =>
      have h' : HasType [] Sigma [] (Term.sum e d) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, hmem, h_e⟩ := HasType.sum_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      subst hteq
      exact ⟨Sigma2, HasType.tsum [] Sigma2 [] [] e' ds d eps h_e' hmem, h_wf2, h_on⟩
  | expand d =>
      have h' : HasType [] Sigma [] (Term.expand e d) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_e⟩ := HasType.expand_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      subst hteq
      exact ⟨Sigma2, HasType.texpand [] Sigma2 [] [] e' ds d eps h_e', h_wf2, h_on⟩
  | uniformLike lo hi =>
      have h' : HasType [] Sigma [] (Term.uniformLike e lo hi) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, eps0, hteq, h_e, hsub_eps⟩ := HasType.uniformLike_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
          (Term.uniformLike e' lo hi) (Typ.tensor ds)
          (EffectRow.union eps0 [EffectLabel.random]) eps
          (HasType.uniformLike [] Sigma2 [] [] e' ds lo hi eps0 h_e')
          hsub_eps
  | perform op =>
      have h' : HasType [] Sigma [] (Term.perform op e) t eps [] := by
        simpa [plug] using h
      obtain ⟨tArg, eps0, h_e, hmatch, hsub_eps⟩ := HasType.perform_inv h'
      rcases h_inner h_e with ⟨Sigma2, h_e', h_wf2, h_on⟩
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.perform op e') t (EffectRow.union [op] eps0) eps
        (HasType.perform [] Sigma2 [] [] op e' tArg t eps0 h_e' hmatch)
        hsub_eps
  | appL e2 =>
      have h' : HasType [] Sigma [] (Term.app e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e1 with ⟨Sigma2, h_e1', h_wf2, h_on⟩
      have h_e2' := hasType_store_weaken_on_locRefs h_e2
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.appL e2)) (locs2 := locs) h_on)
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.app e' e2) t
        (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps
        (HasType.app [] Sigma2 [] [] [] e' e2 t1 t epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | appR v1 =>
      have h' : HasType [] Sigma [] (Term.app v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e2 with ⟨Sigma2, h_e2', h_wf2, h_on⟩
      have h_e1' := hasType_store_weaken_on_locRefs h_e1
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.appR v1)) (locs2 := locs) h_on)
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.app v1 e') t
        (EffectRow.union (EffectRow.union eps1 eps2) epsBody) eps
        (HasType.app [] Sigma2 [] [] [] v1 e' t1 t epsBody eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairL e2 =>
      have h' : HasType [] Sigma [] (Term.pair e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e1 with ⟨Sigma2, h_e1', h_wf2, h_on⟩
      have h_e2' := hasType_store_weaken_on_locRefs h_e2
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.pairL e2)) (locs2 := locs) h_on)
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.pair e' e2) (Typ.pair t1 t2) (EffectRow.union eps1 eps2) eps
        (HasType.tpair [] Sigma2 [] [] [] e' e2 t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | pairR v1 =>
      have h' : HasType [] Sigma [] (Term.pair v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e2 with ⟨Sigma2, h_e2', h_wf2, h_on⟩
      have h_e1' := hasType_store_weaken_on_locRefs h_e1
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.pairR v1)) (locs2 := locs) h_on)
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.pair v1 e') (Typ.pair t1 t2) (EffectRow.union eps1 eps2) eps
        (HasType.tpair [] Sigma2 [] [] [] v1 e' t1 t2 eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addL e2 =>
      have h' : HasType [] Sigma [] (Term.add e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e1 with ⟨Sigma2, h_e1', h_wf2, h_on⟩
      have h_e2' := hasType_store_weaken_on_locRefs h_e2
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.addL e2)) (locs2 := locs) h_on)
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.add e' e2) (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tadd [] Sigma2 [] [] [] e' e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | addR v1 =>
      have h' : HasType [] Sigma [] (Term.add v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e2 with ⟨Sigma2, h_e2', h_wf2, h_on⟩
      have h_e1' := hasType_store_weaken_on_locRefs h_e1
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.addR v1)) (locs2 := locs) h_on)
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.add v1 e') (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tadd [] Sigma2 [] [] [] v1 e' ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulL e2 =>
      have h' : HasType [] Sigma [] (Term.mul e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e1 with ⟨Sigma2, h_e1', h_wf2, h_on⟩
      have h_e2' := hasType_store_weaken_on_locRefs h_e2
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.mulL e2)) (locs2 := locs) h_on)
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.mul e' e2) (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tmul [] Sigma2 [] [] [] e' e2 ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | mulR v1 =>
      have h' : HasType [] Sigma [] (Term.mul v1 e) t eps [] := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      subst hΓmid
      rcases h_inner h_e2 with ⟨Sigma2, h_e2', h_wf2, h_on⟩
      have h_e1' := hasType_store_weaken_on_locRefs h_e1
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.mulR v1)) (locs2 := locs) h_on)
      subst hteq
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.mul v1 e') (Typ.tensor ds) (EffectRow.union eps1 eps2) eps
        (HasType.tmul [] Sigma2 [] [] [] v1 e' ds eps1 eps2 h_e1' h_e2')
        hsub_eps
  | letBind x e2 =>
      have h' : HasType [] Sigma [] (Term.letBind x e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, eps1, eps2, slot, h_e1, h_e2, hΓout, hsub_eps⟩ :=
        HasType.plug_letBind_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      have hΓ3 : Γ3 = [] := by simpa using hΓout.symm
      subst hΓmid
      subst hΓ3
      rcases h_inner h_e1 with ⟨Sigma2, h_e1', h_wf2, h_on⟩
      have h_e2' := hasType_store_weaken_on_locRefs h_e2
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.letBind x e2)) (locs2 := locs) h_on)
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.letBind x e' e2) t (EffectRow.union eps1 eps2) eps
        (HasType.letBind [] Sigma2 [] [] [] x e' e2 t1 t eps1 eps2 slot h_e1' h_e2')
        hsub_eps
  | letpair x y e2 =>
      have h' : HasType [] Sigma [] (Term.letpair x y e e2) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, t2, eps1, eps2, slotX, slotY,
              h_e1, h_e2, hΓout, hsub_eps⟩ :=
        HasType.plug_letpair_inv h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input h_e1
      have hΓ3 : Γ3 = [] := by simpa using hΓout.symm
      subst hΓmid
      subst hΓ3
      rcases h_inner h_e1 with ⟨Sigma2, h_e1', h_wf2, h_on⟩
      have h_e2' := hasType_store_weaken_on_locRefs h_e2
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.letpair x y e2)) (locs2 := locs) h_on)
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.letpair x y e' e2) t (EffectRow.union eps1 eps2) eps
        (HasType.letpair [] Sigma2 [] [] [] x y e' e2 t1 t2 t eps1 eps2
          slotX slotY h_e1' h_e2')
        hsub_eps
  | handle epsH clauses =>
      have h' : HasType [] Sigma [] (Term.handle epsH e clauses) t eps [] := by
        simpa [plug] using h
      obtain ⟨Γmid, epsB, hb, hHsubB, hClIn, hClCov, hcls, hsub_eps⟩ :=
        HasType.handle_inv_strong h'
      have hΓmid : Γmid = [] := has_type_closed_output_of_closed_input hb
      subst hΓmid
      rcases h_inner hb with ⟨Sigma2, hb', h_wf2, h_on⟩
      have hcls' := clausesTyped_store_weaken_on_locRefs hcls
        (by simpa [ctxLocRefs] using StoreTypOn.append_left (locs1 := ctxLocRefs (EvalCtx.handle epsH clauses)) (locs2 := locs) h_on)
      refine ⟨Sigma2, ?_, h_wf2, h_on⟩
      exact HasType.subEff [] Sigma2 [] []
        (Term.handle epsH e' clauses) t (EffectRow.removeOps epsB epsH) eps
        (HasType.handle [] Sigma2 [] [] [] e' clauses t epsH epsB
          hb' hHsubB hClIn hClCov hcls')
        hsub_eps

/-- Closed-program specialization of context preservation for
    multi-frame evaluation-context chains. -/
theorem multiPlug_preserves_typing_closed
    {Sigma : StoreTyp}
    {Es : EvalCtxChain} {e e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType [] Sigma [] (multiPlug Es e) t eps [])
    (h_inner : ∀ {t0 : Typ} {eps0 : EffectRow},
       HasType [] Sigma [] e t0 eps0 [] →
       ∃ Sigma2, HasType [] Sigma2 [] e' t0 eps0 [] ∧
                 StoreTypSub Sigma Sigma2) :
    ∃ Sigma2, HasType [] Sigma2 [] (multiPlug Es e') t eps [] ∧
              StoreTypSub Sigma Sigma2 := by
  induction Es generalizing Sigma t eps with
  | nil =>
      simpa [multiPlug] using h_inner h
  | cons E Es ih =>
      have h' : HasType [] Sigma [] (plug E (multiPlug Es e)) t eps [] := by
        simpa [multiPlug] using h
      refine plug_preserves_typing_closed h' ?_
      intro t0 eps0 h_inner'
      exact ih h_inner' h_inner

private def ctxCounterTensor : TensorVal :=
  { shape := DimList.empty, data := 0.0 }

private def ctxCounterSigma : Store :=
  [(1, ctxCounterTensor), (2, ctxCounterTensor)]

private def ctxCounterStoreTyp : StoreTyp :=
  [(1, Typ.tensor DimList.empty), (2, Typ.tensor DimList.empty)]

private def ctxCounterTerm : Term :=
  Term.pair (Term.add (Term.loc 1) (Term.loc 2)) (Term.loc 1)

private def ctxCounterTerm' : Term :=
  Term.pair (Term.loc 3) (Term.loc 1)

/-- Concrete witness that the current top-level `ctx` theorem shape is
    too strong: an inner numeric step can consume locations still
    mentioned by sibling subterms in the surrounding frame. -/
theorem preservation_ctx_counterexample :
    HasType [] ctxCounterStoreTyp [] ctxCounterTerm
      (Typ.pair (Typ.tensor DimList.empty) (Typ.tensor DimList.empty)) [] [] ∧
    WellScoped ctxCounterTerm ∧
    StoreWf ctxCounterSigma ctxCounterStoreTyp ∧
    Step ⟨ctxCounterSigma, ctxCounterTerm⟩
      ⟨[(3, tensorOpPlaceholder ctxCounterTensor ctxCounterTensor)], ctxCounterTerm'⟩ ∧
    (∀ Sigma', ¬
      (HasType [] Sigma' [] ctxCounterTerm'
         (Typ.pair (Typ.tensor DimList.empty) (Typ.tensor DimList.empty)) [] [] ∧
       StoreWf [(3, tensorOpPlaceholder ctxCounterTensor ctxCounterTensor)] Sigma')) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · have hLoc1 :
        HasType [] ctxCounterStoreTyp [] (Term.loc 1) (Typ.tensor DimList.empty) [] [] := by
      exact HasType.loc [] ctxCounterStoreTyp [] 1 (Typ.tensor DimList.empty) (by
        simp [ctxCounterStoreTyp, storeTypLookup])
    have hLoc2 :
        HasType [] ctxCounterStoreTyp [] (Term.loc 2) (Typ.tensor DimList.empty) [] [] := by
      exact HasType.loc [] ctxCounterStoreTyp [] 2 (Typ.tensor DimList.empty) (by
        simp [ctxCounterStoreTyp, storeTypLookup])
    have hAdd :
        HasType [] ctxCounterStoreTyp []
          (Term.add (Term.loc 1) (Term.loc 2))
          (Typ.tensor DimList.empty) [] [] := by
      simpa using
        (HasType.tadd [] ctxCounterStoreTyp [] [] []
          (Term.loc 1) (Term.loc 2) DimList.empty [] [] hLoc1 hLoc2)
    simpa [ctxCounterTerm] using
      (HasType.tpair [] ctxCounterStoreTyp [] [] []
        (Term.add (Term.loc 1) (Term.loc 2)) (Term.loc 1)
        (Typ.tensor DimList.empty) (Typ.tensor DimList.empty) [] [] hAdd hLoc1)
  · simp [ctxCounterTerm, WellScoped, boundVars, List.nodup_append]
  · refine ⟨?_, ?_⟩
    · intro ell hmem
      simp [ctxCounterSigma, ctxCounterStoreTyp, storeTypDom, storeLookup] at hmem ⊢
      rcases hmem with rfl | rfl
      · simp [ctxCounterSigma, storeLookup]
      · simp [ctxCounterSigma, storeLookup]
    · intro ell hsome
      simp [ctxCounterSigma, storeLookup] at hsome
      simpa [ctxCounterStoreTyp, storeTypDom, eq_comm] using hsome
  · refine Step.ctx ctxCounterSigma
      [(3, tensorOpPlaceholder ctxCounterTensor ctxCounterTensor)]
      (EvalCtx.pairL (Term.loc 1))
      (Term.add (Term.loc 1) (Term.loc 2))
      (Term.loc 3) ?_
    simpa [ctxCounterSigma, ctxCounterTensor, storeLookup, storeRemove, storeExtend, List.find?] using
      (Step.tadd ctxCounterSigma 1 2 3 ctxCounterTensor ctxCounterTensor
        (by simp [ctxCounterSigma, ctxCounterTensor, storeLookup, List.find?])
        (by simp [ctxCounterSigma, ctxCounterTensor, storeLookup, List.find?])
        (by simp [ctxCounterSigma, storeFreshLoc]))
  · intro Sigma' hpost
    rcases hpost with ⟨hTy, hWf⟩
    obtain ⟨GammaMid, eps1, eps2, hLeft, hRight, _hSub⟩ := HasType.pair_inv hTy
    have hGammaMid : GammaMid = [] := has_type_closed_output_of_closed_input hLeft
    subst hGammaMid
    have hLookup1 :
        storeTypLookup Sigma' 1 = some (Typ.tensor DimList.empty) :=
      (HasType.loc_inv hRight).1
    have hLive1 : (storeLookup [(3, tensorOpPlaceholder ctxCounterTensor ctxCounterTensor)] 1).isSome :=
      StoreWf.lookup_isSome_of_typing hWf hLookup1
    simp [storeLookup, List.find?] at hLive1

theorem preservation_ctx_counterexample_not_runtimeLinear :
    ¬ RuntimeLinear ctxCounterTerm := by
  simp [RuntimeLinear, locRefs, ctxCounterTerm]

private def handleCtxCounterBody : Term :=
  Term.pair (Term.loc 1) (Term.var "k")

private def handleCtxCounterClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.accum, "x", "k", handleCtxCounterBody)]

private def handleCtxCounterTerm : Term :=
  Term.handle [EffectLabel.accum]
    (plug EvalCtx.copy (Term.perform EffectLabel.accum Term.unit))
    handleCtxCounterClauses

private def handleCtxCounterTerm' : Term :=
  subst (subst handleCtxCounterBody Term.unit "x")
    (Term.abs (capturedContName handleCtxCounterTerm) Typ.unit
      (Term.handle [EffectLabel.accum]
        (plug EvalCtx.copy
          (Term.var (capturedContName handleCtxCounterTerm)))
        handleCtxCounterClauses))
    "k"

/-- Concrete witness that one-step `RuntimeLinear` preservation is false
    for captured-handler reduction under the current `locRefs`
    definition: the selected clause body's location mention is copied
    into the reified continuation because the continuation retains the
    whole clause list. -/
theorem runtimeLinear_handleOpCtx_counterexample :
    RuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    ¬ RuntimeLinear handleCtxCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [RuntimeLinear, handleCtxCounterTerm, handleCtxCounterClauses,
      handleCtxCounterBody, locRefs, locRefsClauses, plug]
  · exact Step.handleOpCtx []
      EffectLabel.accum Term.unit
      [EffectLabel.accum] EvalCtx.copy
      handleCtxCounterClauses
      "x" "k" handleCtxCounterBody Typ.unit
      IsValue.unit
      ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
      (by simp [handleCtxCounterClauses])
      (by simp)
      (by simp [EvalCtx.noHandleFor])
  · simp [RuntimeLinear, handleCtxCounterTerm', handleCtxCounterTerm,
      handleCtxCounterClauses, handleCtxCounterBody, locRefs,
      locRefsClauses, plug, subst]

theorem runtimeLinear_handleOpCtxs_counterexample :
    RuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    ¬ RuntimeLinear handleCtxCounterTerm' := by
  simpa [handleCtxCounterTerm, handleCtxCounterTerm', multiPlug] using
    (show RuntimeLinear
        (Term.handle [EffectLabel.accum]
          (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
          handleCtxCounterClauses) ∧
      Step
        ⟨[],
          Term.handle [EffectLabel.accum]
            (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
            handleCtxCounterClauses⟩
        ⟨[],
          subst (subst handleCtxCounterBody Term.unit "x")
            (Term.abs
              (capturedContName
                (Term.handle [EffectLabel.accum]
                  (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                  handleCtxCounterClauses))
              Typ.unit
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy]
                  (Term.var
                    (capturedContName
                      (Term.handle [EffectLabel.accum]
                        (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                        handleCtxCounterClauses))))
                handleCtxCounterClauses))
            "k"⟩ ∧
      ¬ RuntimeLinear
        (subst (subst handleCtxCounterBody Term.unit "x")
          (Term.abs
            (capturedContName
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                handleCtxCounterClauses))
            Typ.unit
            (Term.handle [EffectLabel.accum]
              (multiPlug [EvalCtx.copy]
                (Term.var
                  (capturedContName
                    (Term.handle [EffectLabel.accum]
                      (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                      handleCtxCounterClauses))))
              handleCtxCounterClauses))
          "k") by
      refine ⟨?_, ?_, ?_⟩
      · simp [RuntimeLinear, handleCtxCounterClauses, handleCtxCounterBody,
          locRefs, locRefsClauses, multiPlug, plug]
      · exact Step.handleOpCtxs []
          EffectLabel.accum Term.unit
          [EffectLabel.accum] [EvalCtx.copy]
          handleCtxCounterClauses
          "x" "k" handleCtxCounterBody Typ.unit
          IsValue.unit
          ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
          (by simp [handleCtxCounterClauses])
          (by simp)
          (by simp [EvalCtxChain.noHandleFor, EvalCtx.noHandleFor])
      · simp [RuntimeLinear, handleCtxCounterClauses, handleCtxCounterBody,
          locRefs, locRefsClauses, multiPlug, plug, subst])

theorem activeRuntimeLinear_handleOpCtx_repaired :
    ActiveRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    ActiveRuntimeLinear handleCtxCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handleCtxCounterTerm, handleCtxCounterClauses, handleCtxCounterBody, plug]
  · exact Step.handleOpCtx []
      EffectLabel.accum Term.unit
      [EffectLabel.accum] EvalCtx.copy
      handleCtxCounterClauses
      "x" "k" handleCtxCounterBody Typ.unit
      IsValue.unit
      ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
      (by simp [handleCtxCounterClauses])
      (by simp)
      (by simp [EvalCtx.noHandleFor])
  · simp [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handleCtxCounterTerm', handleCtxCounterTerm,
      handleCtxCounterClauses, handleCtxCounterBody, plug, subst]

theorem activeRuntimeLinear_handleOpCtxs_repaired :
    ActiveRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    ActiveRuntimeLinear handleCtxCounterTerm' := by
  simpa [handleCtxCounterTerm, handleCtxCounterTerm', multiPlug] using
    (show ActiveRuntimeLinear
        (Term.handle [EffectLabel.accum]
          (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
          handleCtxCounterClauses) ∧
      Step
        ⟨[],
          Term.handle [EffectLabel.accum]
            (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
            handleCtxCounterClauses⟩
        ⟨[],
          subst (subst handleCtxCounterBody Term.unit "x")
            (Term.abs
              (capturedContName
                (Term.handle [EffectLabel.accum]
                  (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                  handleCtxCounterClauses))
              Typ.unit
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy]
                  (Term.var
                    (capturedContName
                      (Term.handle [EffectLabel.accum]
                        (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                        handleCtxCounterClauses))))
                handleCtxCounterClauses))
            "k"⟩ ∧
      ActiveRuntimeLinear
        (subst (subst handleCtxCounterBody Term.unit "x")
          (Term.abs
            (capturedContName
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                handleCtxCounterClauses))
            Typ.unit
            (Term.handle [EffectLabel.accum]
              (multiPlug [EvalCtx.copy]
                (Term.var
                  (capturedContName
                    (Term.handle [EffectLabel.accum]
                      (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                      handleCtxCounterClauses))))
              handleCtxCounterClauses))
          "k") by
      refine ⟨?_, ?_, ?_⟩
      · simp [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
          handleCtxCounterClauses, handleCtxCounterBody, multiPlug, plug]
      · exact Step.handleOpCtxs []
          EffectLabel.accum Term.unit
          [EffectLabel.accum] [EvalCtx.copy]
          handleCtxCounterClauses
          "x" "k" handleCtxCounterBody Typ.unit
          IsValue.unit
          ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
          (by simp [handleCtxCounterClauses])
          (by simp)
          (by simp [EvalCtxChain.noHandleFor, EvalCtx.noHandleFor])
      · simp [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
          handleCtxCounterClauses, handleCtxCounterBody, multiPlug, plug, subst])

/-- The recursive active-footprint invariant keeps the captured-handler
    repairs above: the dormant clause body is checked once, but not
    spuriously duplicated by continuation capture. -/
theorem deepActiveRuntimeLinear_handleOpCtx_repaired :
    DeepActiveRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    DeepActiveRuntimeLinear handleCtxCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handleCtxCounterTerm, handleCtxCounterClauses, handleCtxCounterBody, plug]
  · exact Step.handleOpCtx []
      EffectLabel.accum Term.unit
      [EffectLabel.accum] EvalCtx.copy
      handleCtxCounterClauses
      "x" "k" handleCtxCounterBody Typ.unit
      IsValue.unit
      ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
      (by simp [handleCtxCounterClauses])
      (by simp)
      (by simp [EvalCtx.noHandleFor])
  · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handleCtxCounterTerm', handleCtxCounterTerm,
      handleCtxCounterClauses, handleCtxCounterBody, plug, subst]

theorem deepActiveRuntimeLinear_handleOpCtxs_repaired :
    DeepActiveRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    DeepActiveRuntimeLinear handleCtxCounterTerm' := by
  simpa [handleCtxCounterTerm, handleCtxCounterTerm', multiPlug] using
    (show DeepActiveRuntimeLinear
        (Term.handle [EffectLabel.accum]
          (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
          handleCtxCounterClauses) ∧
      Step
        ⟨[],
          Term.handle [EffectLabel.accum]
            (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
            handleCtxCounterClauses⟩
        ⟨[],
          subst (subst handleCtxCounterBody Term.unit "x")
            (Term.abs
              (capturedContName
                (Term.handle [EffectLabel.accum]
                  (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                  handleCtxCounterClauses))
              Typ.unit
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy]
                  (Term.var
                    (capturedContName
                      (Term.handle [EffectLabel.accum]
                        (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                        handleCtxCounterClauses))))
                handleCtxCounterClauses))
            "k"⟩ ∧
      DeepActiveRuntimeLinear
        (subst (subst handleCtxCounterBody Term.unit "x")
          (Term.abs
            (capturedContName
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                handleCtxCounterClauses))
            Typ.unit
            (Term.handle [EffectLabel.accum]
              (multiPlug [EvalCtx.copy]
                (Term.var
                  (capturedContName
                    (Term.handle [EffectLabel.accum]
                      (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                      handleCtxCounterClauses))))
              handleCtxCounterClauses))
          "k") by
      refine ⟨?_, ?_, ?_⟩
      · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
          ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
          handleCtxCounterClauses, handleCtxCounterBody, multiPlug, plug]
      · exact Step.handleOpCtxs []
          EffectLabel.accum Term.unit
          [EffectLabel.accum] [EvalCtx.copy]
          handleCtxCounterClauses
          "x" "k" handleCtxCounterBody Typ.unit
          IsValue.unit
          ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
          (by simp [handleCtxCounterClauses])
          (by simp)
          (by simp [EvalCtxChain.noHandleFor, EvalCtx.noHandleFor])
      · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
          ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
          handleCtxCounterClauses, handleCtxCounterBody, multiPlug, plug, subst])

/-- The stronger handler-aware invariant keeps the captured-handler
    witness admissible: dormant clause bodies are still checked
    recursively, but they are not re-counted across the abstraction
    barrier introduced by continuation capture. -/
theorem handlerAwareRuntimeLinear_handleOpCtx_repaired :
    HandlerAwareRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    HandlerAwareRuntimeLinear handleCtxCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handleCtxCounterTerm, handleCtxCounterClauses, handleCtxCounterBody, plug]
  · exact Step.handleOpCtx []
      EffectLabel.accum Term.unit
      [EffectLabel.accum] EvalCtx.copy
      handleCtxCounterClauses
      "x" "k" handleCtxCounterBody Typ.unit
      IsValue.unit
      ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
      (by simp [handleCtxCounterClauses])
      (by simp)
      (by simp [EvalCtx.noHandleFor])
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handleCtxCounterTerm', handleCtxCounterTerm,
      handleCtxCounterClauses, handleCtxCounterBody, plug, subst]

theorem handlerAwareRuntimeLinear_handleOpCtxs_repaired :
    HandlerAwareRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    HandlerAwareRuntimeLinear handleCtxCounterTerm' := by
  simpa [handleCtxCounterTerm, handleCtxCounterTerm', multiPlug] using
    (show HandlerAwareRuntimeLinear
        (Term.handle [EffectLabel.accum]
          (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
          handleCtxCounterClauses) ∧
      Step
        ⟨[],
          Term.handle [EffectLabel.accum]
            (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
            handleCtxCounterClauses⟩
        ⟨[],
          subst (subst handleCtxCounterBody Term.unit "x")
            (Term.abs
              (capturedContName
                (Term.handle [EffectLabel.accum]
                  (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                  handleCtxCounterClauses))
              Typ.unit
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy]
                  (Term.var
                    (capturedContName
                      (Term.handle [EffectLabel.accum]
                        (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                        handleCtxCounterClauses))))
                handleCtxCounterClauses))
            "k"⟩ ∧
      HandlerAwareRuntimeLinear
        (subst (subst handleCtxCounterBody Term.unit "x")
          (Term.abs
            (capturedContName
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                handleCtxCounterClauses))
            Typ.unit
            (Term.handle [EffectLabel.accum]
              (multiPlug [EvalCtx.copy]
                (Term.var
                  (capturedContName
                    (Term.handle [EffectLabel.accum]
                      (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                      handleCtxCounterClauses))))
              handleCtxCounterClauses))
          "k") by
      refine ⟨?_, ?_, ?_⟩
      · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
          StepLocRefs, StepLocRefsClauses,
          LocRefsDisjoint, LocRefsSeparated,
          ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
          handleCtxCounterClauses, handleCtxCounterBody, multiPlug, plug]
      · exact Step.handleOpCtxs []
          EffectLabel.accum Term.unit
          [EffectLabel.accum] [EvalCtx.copy]
          handleCtxCounterClauses
          "x" "k" handleCtxCounterBody Typ.unit
          IsValue.unit
          ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
          (by simp [handleCtxCounterClauses])
          (by simp)
          (by simp [EvalCtxChain.noHandleFor, EvalCtx.noHandleFor])
      · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
          StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
          LocRefsDisjoint, LocRefsSeparated,
          ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
          handleCtxCounterClauses, handleCtxCounterBody, multiPlug, plug, subst])

/-- Adding the substitution-aware variable sidecar keeps the captured
    handler witness admissible: the clause's continuation variable stays
    hidden behind the same abstraction/handler barriers that already
    justified the concrete-location repair. -/
theorem substAwareHandlerRuntimeLinear_handleOpCtx_repaired :
    SubstAwareHandlerRuntimeLinear handleCtxCounterTerm ∧
    Step ⟨[], handleCtxCounterTerm⟩ ⟨[], handleCtxCounterTerm'⟩ ∧
    SubstAwareHandlerRuntimeLinear handleCtxCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · refine ⟨handlerAwareRuntimeLinear_handleOpCtx_repaired.1, ?_⟩
    simp [SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
      activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
      VarRefsDisjoint, VarRefsSeparated,
      handleCtxCounterTerm, handleCtxCounterClauses, handleCtxCounterBody, plug]
  · exact Step.handleOpCtx []
      EffectLabel.accum Term.unit
      [EffectLabel.accum] EvalCtx.copy
      handleCtxCounterClauses
      "x" "k" handleCtxCounterBody Typ.unit
      IsValue.unit
      ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
      (by simp [handleCtxCounterClauses])
      (by simp)
      (by simp [EvalCtx.noHandleFor])
  · refine ⟨handlerAwareRuntimeLinear_handleOpCtx_repaired.2.2, ?_⟩
    have hFresh : freshInTerm (capturedContName handleCtxCounterTerm) handleCtxCounterTerm :=
      capturedContName_freshInTerm (e := handleCtxCounterTerm)
    have hkMem : "k" ∈ boundVars handleCtxCounterTerm := by
      simp [handleCtxCounterTerm, handleCtxCounterClauses, handleCtxCounterBody,
        boundVars, boundVarsClauses, plug]
    have hkNeRight : capturedContName handleCtxCounterTerm ≠ "k" := by
      intro hEq
      exact hFresh.2 (by simpa [hEq] using hkMem)
    have hkNeLeft : "k" ≠ capturedContName handleCtxCounterTerm := hkNeRight.symm
    have hkFreshPair :
        ¬ "k" = capturedContName handleCtxCounterTerm ∧
          ¬ capturedContName handleCtxCounterTerm = "k" :=
      ⟨hkNeLeft, hkNeRight⟩
    simpa [SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
      activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
      VarRefsDisjoint, VarRefsSeparated,
      handleCtxCounterTerm', handleCtxCounterTerm,
      handleCtxCounterClauses, handleCtxCounterBody, plug, subst] using
      hkFreshPair

/-- The store-liveness sidecar keeps the same captured-handler repair
    on a configuration whose runtime locations are actually live. -/
theorem storeAwareSubstHandlerRuntimeLinear_handleOpCtx_repaired :
    StoreAwareSubstHandlerRuntimeLinear ctxCounterSigma handleCtxCounterTerm ∧
    Step ⟨ctxCounterSigma, handleCtxCounterTerm⟩
      ⟨ctxCounterSigma, handleCtxCounterTerm'⟩ ∧
    StoreAwareSubstHandlerRuntimeLinear ctxCounterSigma handleCtxCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · refine ⟨substAwareHandlerRuntimeLinear_handleOpCtx_repaired.1, ?_⟩
    intro ell hmem
    simp [handleCtxCounterTerm, handleCtxCounterClauses, handleCtxCounterBody,
      locRefs, locRefsClauses, plug] at hmem
    rcases hmem with rfl
    simp [ctxCounterSigma, storeLookup, List.find?]
  · exact Step.handleOpCtx ctxCounterSigma
      EffectLabel.accum Term.unit
      [EffectLabel.accum] EvalCtx.copy
      handleCtxCounterClauses
      "x" "k" handleCtxCounterBody Typ.unit
      IsValue.unit
      ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
      (by simp [handleCtxCounterClauses])
      (by simp)
      (by simp [EvalCtx.noHandleFor])
  · refine ⟨substAwareHandlerRuntimeLinear_handleOpCtx_repaired.2.2, ?_⟩
    intro ell hmem
    simp [handleCtxCounterTerm', handleCtxCounterTerm,
      handleCtxCounterClauses, handleCtxCounterBody,
      locRefs, locRefsClauses, plug, subst] at hmem
    rcases hmem with rfl
    simp [ctxCounterSigma, storeLookup, List.find?]

/-- Public configuration-level wrapper for the repaired captured-handler
    witness under the current store-aware runtime-safety surface. -/
theorem runtimeSafeConfig_handleOpCtx_repaired :
    RuntimeSafeConfig ⟨ctxCounterSigma, handleCtxCounterTerm⟩ ∧
    Step ⟨ctxCounterSigma, handleCtxCounterTerm⟩
      ⟨ctxCounterSigma, handleCtxCounterTerm'⟩ ∧
    RuntimeSafeConfig ⟨ctxCounterSigma, handleCtxCounterTerm'⟩ := by
  simpa [RuntimeSafeConfig, RuntimeSafe] using
    storeAwareSubstHandlerRuntimeLinear_handleOpCtx_repaired

/-- Chain-shaped wrapper for the same repaired captured-handler witness.
    This is intentionally stated with `multiPlug` to match
    `Step.handleOpCtxs`, even though for the singleton chain here the
    underlying term coincides with `handleCtxCounterTerm`. -/
theorem runtimeSafeConfig_handleOpCtxs_repaired :
    RuntimeSafeConfig
      ⟨ctxCounterSigma,
        Term.handle [EffectLabel.accum]
          (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
          handleCtxCounterClauses⟩ ∧
    Step
      ⟨ctxCounterSigma,
        Term.handle [EffectLabel.accum]
          (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
          handleCtxCounterClauses⟩
      ⟨ctxCounterSigma,
        subst (subst handleCtxCounterBody Term.unit "x")
          (Term.abs
            (capturedContName
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                handleCtxCounterClauses))
            Typ.unit
            (Term.handle [EffectLabel.accum]
              (multiPlug [EvalCtx.copy]
                (Term.var
                  (capturedContName
                    (Term.handle [EffectLabel.accum]
                      (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                      handleCtxCounterClauses))))
              handleCtxCounterClauses))
          "k"⟩ ∧
    RuntimeSafeConfig
      ⟨ctxCounterSigma,
        subst (subst handleCtxCounterBody Term.unit "x")
          (Term.abs
            (capturedContName
              (Term.handle [EffectLabel.accum]
                (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                handleCtxCounterClauses))
            Typ.unit
            (Term.handle [EffectLabel.accum]
              (multiPlug [EvalCtx.copy]
                (Term.var
                  (capturedContName
                    (Term.handle [EffectLabel.accum]
                      (multiPlug [EvalCtx.copy] (Term.perform EffectLabel.accum Term.unit))
                      handleCtxCounterClauses))))
              handleCtxCounterClauses))
          "k"⟩ := by
  simpa [RuntimeSafeConfig, RuntimeSafe, handleCtxCounterTerm, handleCtxCounterTerm',
      multiPlug] using
    (show StoreAwareSubstHandlerRuntimeLinear ctxCounterSigma handleCtxCounterTerm ∧
        Step ⟨ctxCounterSigma, handleCtxCounterTerm⟩
          ⟨ctxCounterSigma, handleCtxCounterTerm'⟩ ∧
        StoreAwareSubstHandlerRuntimeLinear ctxCounterSigma handleCtxCounterTerm' from
      storeAwareSubstHandlerRuntimeLinear_handleOpCtx_repaired)

private def handleDirectCounterBody : Term :=
  Term.letBind "z" (Term.pair (Term.loc 1) (Term.loc 1)) Term.unit

private def handleDirectCounterClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.accum, "x", "k", handleDirectCounterBody)]

private def handleDirectCounterTerm : Term :=
  Term.handle [EffectLabel.accum]
    (Term.perform EffectLabel.accum Term.unit)
    handleDirectCounterClauses

private def handleDirectCounterTerm' : Term :=
  Term.letBind "z" (Term.pair (Term.loc 1) (Term.loc 1)) Term.unit

/-- Narrowing `RuntimeLinear` to `ActiveRuntimeLinear` fixes the
    captured-handler context cases above, but it is still too weak for
    direct handled operations: a dormant clause body can become active
    in one step and expose duplicated locations that were previously
    hidden from the active footprint. -/
theorem activeRuntimeLinear_handleOpDirect_counterexample :
    HasType [] ctxCounterStoreTyp [] handleDirectCounterTerm
      Typ.unit [] [] ∧
    WellScoped handleDirectCounterTerm ∧
    StoreWf ctxCounterSigma ctxCounterStoreTyp ∧
    ActiveRuntimeLinear handleDirectCounterTerm ∧
    Step ⟨ctxCounterSigma, handleDirectCounterTerm⟩
      ⟨ctxCounterSigma, handleDirectCounterTerm'⟩ ∧
    ¬ ActiveRuntimeLinear handleDirectCounterTerm' := by
  let tDup : Typ := Typ.pair (Typ.tensor DimList.empty) (Typ.tensor DimList.empty)
  let tK : Typ := Typ.arrow Typ.unit Typ.unit []
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · have hUnit :
        HasType [] ctxCounterStoreTyp [] Term.unit Typ.unit [] [] := by
      exact HasType.unit [] ctxCounterStoreTyp []
    have hPerform :
        HasType [] ctxCounterStoreTyp []
          (Term.perform EffectLabel.accum Term.unit)
          Typ.unit [EffectLabel.accum] [] := by
      exact HasType.perform [] ctxCounterStoreTyp [] []
        EffectLabel.accum Term.unit Typ.unit Typ.unit [] hUnit
        (by simp [OpSigMatch, opArgType, opRetType])
    have hLoc1 :
        HasType [] ctxCounterStoreTyp
          [("x", some Typ.unit), ("k", some tK)]
          (Term.loc 1) (Typ.tensor DimList.empty) []
          [("x", some Typ.unit), ("k", some tK)] := by
      exact HasType.loc [] ctxCounterStoreTyp
        [("x", some Typ.unit), ("k", some tK)] 1
        (Typ.tensor DimList.empty) (by
          simp [ctxCounterStoreTyp, storeTypLookup])
    have hDup :
        HasType [] ctxCounterStoreTyp
          [("x", some Typ.unit), ("k", some tK)]
          (Term.pair (Term.loc 1) (Term.loc 1)) tDup []
          [("x", some Typ.unit), ("k", some tK)] := by
      simpa [tDup, tK] using
        (HasType.tpair [] ctxCounterStoreTyp
          [("x", some Typ.unit), ("k", some tK)]
          [("x", some Typ.unit), ("k", some tK)]
          [("x", some Typ.unit), ("k", some tK)]
          (Term.loc 1) (Term.loc 1)
          (Typ.tensor DimList.empty) (Typ.tensor DimList.empty)
          [] [] hLoc1 hLoc1)
    have hUnitBody :
        HasType [] ctxCounterStoreTyp
          ([("x", some Typ.unit), ("k", some tK)] ++ [("z", some tDup)])
          Term.unit Typ.unit []
          ([("x", some Typ.unit), ("k", some tK)] ++ [("z", some tDup)]) := by
      exact HasType.unit [] ctxCounterStoreTyp
        ([("x", some Typ.unit), ("k", some tK)] ++ [("z", some tDup)])
    have hBody :
        HasType [] ctxCounterStoreTyp
          [("x", some Typ.unit), ("k", some tK)]
          handleDirectCounterBody Typ.unit []
          [("x", some Typ.unit), ("k", some tK)] := by
      simpa [handleDirectCounterBody, tDup, tK] using
        (HasType.letBind [] ctxCounterStoreTyp
          [("x", some Typ.unit), ("k", some tK)]
          [("x", some Typ.unit), ("k", some tK)]
          [("x", some Typ.unit), ("k", some tK)]
          "z" (Term.pair (Term.loc 1) (Term.loc 1)) Term.unit
          tDup Typ.unit [] [] (some tDup) hDup hUnitBody)
    have hClauses :
        ClausesTyped [] ctxCounterStoreTyp [] [] Typ.unit []
          handleDirectCounterClauses := by
      exact ClausesTyped.cons [] ctxCounterStoreTyp [] [] Typ.unit Typ.unit Typ.unit []
        EffectLabel.accum "x" "k" handleDirectCounterBody [] (some Typ.unit) (some tK)
        (by simp [OpSigMatch, opArgType, opRetType]) hBody
        (ClausesTyped.nil [] ctxCounterStoreTyp [] Typ.unit [])
    exact HasType.handle [] ctxCounterStoreTyp [] [] [] 
      (Term.perform EffectLabel.accum Term.unit)
      handleDirectCounterClauses Typ.unit [EffectLabel.accum] [EffectLabel.accum]
      hPerform
      (by simp)
      (by simp [handleDirectCounterClauses])
      (by intro op hop; simp at hop; rcases hop with rfl; exact ⟨(EffectLabel.accum, "x", "k", handleDirectCounterBody), by simp [handleDirectCounterClauses], rfl⟩)
      hClauses
  · simpa [handleDirectCounterTerm, WellScoped, boundVars, boundVarsClauses,
      handleDirectCounterClauses, handleDirectCounterBody] using
      (show (["x", "k", "z"] : List String).Nodup by decide)
  · refine ⟨?_, ?_⟩
    · intro ell hmem
      simp [ctxCounterSigma, ctxCounterStoreTyp, storeTypDom, storeLookup] at hmem ⊢
      rcases hmem with rfl | rfl
      · simp [ctxCounterSigma, storeLookup]
      · simp [ctxCounterSigma, storeLookup]
    · intro ell hsome
      simp [ctxCounterSigma, storeLookup] at hsome
      simpa [ctxCounterStoreTyp, storeTypDom, eq_comm] using hsome
  · simp [ActiveRuntimeLinear, handleDirectCounterTerm, activeLocRefs, activeLocRefsClauses,
      handleDirectCounterClauses]
  · simpa [handleDirectCounterTerm', handleDirectCounterBody, subst] using
      (Step.handleOpDirect ctxCounterSigma
        EffectLabel.accum Term.unit [EffectLabel.accum]
        handleDirectCounterClauses
        "x" "k" handleDirectCounterBody Typ.unit
        IsValue.unit
        ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
        (by simp [handleDirectCounterClauses]))
  · simp [ActiveRuntimeLinear, handleDirectCounterTerm', activeLocRefs]

/-- The recursive candidate invariant rejects the direct-handler witness
    before reduction: the duplicated locations were dormant under
    `ActiveRuntimeLinear`, but they are visible in the clause body's own
    active footprint. -/
theorem deepActiveRuntimeLinear_handleOpDirect_blocks_counterexample :
    ¬ DeepActiveRuntimeLinear handleDirectCounterTerm := by
  simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handleDirectCounterTerm, handleDirectCounterClauses,
    handleDirectCounterBody]

/-- The stronger invariant still rejects the direct-handler witness:
    the clause body itself already duplicates a live location, so it
    fails before any handled step can expose it. -/
theorem handlerAwareRuntimeLinear_handleOpDirect_blocks_counterexample :
    ¬ HandlerAwareRuntimeLinear handleDirectCounterTerm := by
  simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses,
    LocRefsDisjoint, LocRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handleDirectCounterTerm, handleDirectCounterClauses,
    handleDirectCounterBody]

private def deepActiveGapClauseBody : Term :=
  Term.letBind "z" (Term.loc 1) Term.unit

private def deepActiveGapClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.accum, "y", "k", deepActiveGapClauseBody)]

private def deepActiveGapHandle : Term :=
  Term.handle [EffectLabel.accum]
    (Term.perform EffectLabel.accum Term.unit)
    deepActiveGapClauses

private def deepActiveGapAbsBody : Term :=
  Term.pair (Term.var "x") deepActiveGapHandle

private def deepActiveGapTerm : Term :=
  Term.app
    (Term.abs "x" (Typ.tensor DimList.empty) deepActiveGapAbsBody)
    (Term.loc 1)

private def deepActiveGapTerm' : Term :=
  subst deepActiveGapAbsBody (Term.loc 1) "x"

private def deepActiveGapTerm'' : Term :=
  Term.pair (Term.loc 1) deepActiveGapClauseBody

/-- The current DB-side deep-active dead-substitution theorem is
    stronger than the named beta wrapper actually needs: a typed
    deep-active beta step can preserve the invariant even when the
    argument location is shared with an unchanged dormant clause body.
    That overlap blocks the theorem's full-`locRefs` separation premise
    without creating a real preservation failure. -/
theorem deepActiveRuntimeLinear_beta_bridge_gap :
    HasType [] ctxCounterStoreTyp [] deepActiveGapTerm
      (Typ.pair (Typ.tensor DimList.empty) Typ.unit) [] [] ∧
    DeepActiveRuntimeLinear deepActiveGapTerm ∧
    Step ⟨ctxCounterSigma, deepActiveGapTerm⟩
      ⟨ctxCounterSigma, deepActiveGapTerm'⟩ ∧
    DeepActiveRuntimeLinear deepActiveGapTerm' ∧
    ¬ LocRefsSeparated (locRefs (Term.loc 1)) (locRefs deepActiveGapAbsBody) := by
  let tX : Typ := Typ.tensor DimList.empty
  let tK : Typ := Typ.arrow Typ.unit Typ.unit []
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · have hLoc1Nil :
        HasType [] ctxCounterStoreTyp [] (Term.loc 1) tX [] [] := by
      exact HasType.loc [] ctxCounterStoreTyp [] 1 tX (by
        simp [tX, ctxCounterStoreTyp, storeTypLookup])
    have hUnit :
        HasType [] ctxCounterStoreTyp [("x", none)] Term.unit Typ.unit []
          [("x", none)] := by
      exact HasType.unit [] ctxCounterStoreTyp [("x", none)]
    have hPerform :
        HasType [] ctxCounterStoreTyp [("x", none)]
          (Term.perform EffectLabel.accum Term.unit)
          Typ.unit [EffectLabel.accum] [("x", none)] := by
      exact HasType.perform [] ctxCounterStoreTyp [("x", none)] [("x", none)]
        EffectLabel.accum Term.unit Typ.unit Typ.unit [] hUnit
        (by simp [OpSigMatch, opArgType, opRetType])
    have hLoc1Clause :
        HasType [] ctxCounterStoreTyp
          [("x", none), ("y", some Typ.unit), ("k", some tK)]
          (Term.loc 1) tX []
          [("x", none), ("y", some Typ.unit), ("k", some tK)] := by
      exact HasType.loc [] ctxCounterStoreTyp
        [("x", none), ("y", some Typ.unit), ("k", some tK)] 1 tX (by
          simp [tX, ctxCounterStoreTyp, storeTypLookup])
    have hUnitBody :
        HasType [] ctxCounterStoreTyp
          ([("x", none), ("y", some Typ.unit), ("k", some tK)] ++
            [("z", some tX)])
          Term.unit Typ.unit []
          ([("x", none), ("y", some Typ.unit), ("k", some tK)] ++
            [("z", some tX)]) := by
      exact HasType.unit [] ctxCounterStoreTyp
        ([("x", none), ("y", some Typ.unit), ("k", some tK)] ++
          [("z", some tX)])
    have hClauseBody :
        HasType [] ctxCounterStoreTyp
          [("x", none), ("y", some Typ.unit), ("k", some tK)]
          deepActiveGapClauseBody Typ.unit []
          [("x", none), ("y", some Typ.unit), ("k", some tK)] := by
      simpa [deepActiveGapClauseBody, tX, tK] using
        (HasType.letBind [] ctxCounterStoreTyp
          [("x", none), ("y", some Typ.unit), ("k", some tK)]
          [("x", none), ("y", some Typ.unit), ("k", some tK)]
          [("x", none), ("y", some Typ.unit), ("k", some tK)]
          "z" (Term.loc 1) Term.unit tX Typ.unit [] [] (some tX)
          hLoc1Clause hUnitBody)
    have hClauses :
        ClausesTyped [] ctxCounterStoreTyp
          [("x", none)] [("x", none)] Typ.unit []
          deepActiveGapClauses := by
      exact ClausesTyped.cons [] ctxCounterStoreTyp
        [("x", none)] [("x", none)]
        Typ.unit Typ.unit Typ.unit []
        EffectLabel.accum "y" "k" deepActiveGapClauseBody
        [] (some Typ.unit) (some tK)
        (by simp [OpSigMatch, opArgType, opRetType]) hClauseBody
        (ClausesTyped.nil [] ctxCounterStoreTyp [("x", none)] Typ.unit [])
    have hHandle :
        HasType [] ctxCounterStoreTyp
          [("x", none)] deepActiveGapHandle Typ.unit []
          [("x", none)] := by
      exact HasType.handle [] ctxCounterStoreTyp
        [("x", none)] [("x", none)] [("x", none)]
        (Term.perform EffectLabel.accum Term.unit)
        deepActiveGapClauses Typ.unit
        [EffectLabel.accum] [EffectLabel.accum]
        hPerform
        (by simp)
        (by simp [deepActiveGapClauses])
        (by
          intro op hop
          simp at hop
          rcases hop with rfl
          exact ⟨(EffectLabel.accum, "y", "k", deepActiveGapClauseBody),
            by simp [deepActiveGapClauses], rfl⟩)
        hClauses
    have hVarX :
        HasType [] ctxCounterStoreTyp [("x", some tX)]
          (Term.var "x") tX [] [("x", none)] := by
      exact HasType.var [] ctxCounterStoreTyp [] [] "x" tX
    have hPairBody :
        HasType [] ctxCounterStoreTyp
          [("x", some tX)] deepActiveGapAbsBody
          (Typ.pair tX Typ.unit) [] [("x", none)] := by
      simpa [deepActiveGapAbsBody, tX] using
        (HasType.tpair [] ctxCounterStoreTyp
          [("x", some tX)] [("x", none)] [("x", none)]
          (Term.var "x") deepActiveGapHandle
          tX Typ.unit [] [] hVarX
          (by simpa using hHandle))
    have hAbs :
        HasType [] ctxCounterStoreTyp []
          (Term.abs "x" tX deepActiveGapAbsBody)
          (Typ.arrow tX (Typ.pair tX Typ.unit) []) [] [] := by
      exact HasType.abs [] ctxCounterStoreTyp [] []
        "x" tX (Typ.pair tX Typ.unit) [] deepActiveGapAbsBody none hPairBody
    simpa [deepActiveGapTerm, tX] using
      (HasType.app [] ctxCounterStoreTyp [] [] []
        (Term.abs "x" tX deepActiveGapAbsBody) (Term.loc 1)
        tX (Typ.pair tX Typ.unit) [] [] [] hAbs hLoc1Nil)
  · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      deepActiveGapTerm, deepActiveGapAbsBody, deepActiveGapHandle,
      deepActiveGapClauses, deepActiveGapClauseBody]
  · simpa [deepActiveGapTerm, deepActiveGapTerm'] using
      (Step.beta ctxCounterSigma "x" (Typ.tensor DimList.empty)
        deepActiveGapAbsBody (Term.loc 1) (IsValue.loc 1))
  · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      deepActiveGapTerm', deepActiveGapAbsBody, deepActiveGapHandle,
      deepActiveGapClauses, deepActiveGapClauseBody, subst, substClauses]
  · simp [LocRefsSeparated, locRefs, locRefsClauses,
      deepActiveGapAbsBody, deepActiveGapHandle,
      deepActiveGapClauses, deepActiveGapClauseBody]

/-- The beta-gap witness is a real compositionality failure for the
    current deep-active invariant, not just a wrapper inconvenience.
    After beta exposes the dormant handler beside an active sibling
    with the same location, one contextual `handleOpDirect` step makes
    that overlap active and breaks `DeepActiveRuntimeLinear`. -/
theorem deepActiveRuntimeLinear_beta_gap_two_step_counterexample :
    HasType [] ctxCounterStoreTyp [] deepActiveGapTerm
      (Typ.pair (Typ.tensor DimList.empty) Typ.unit) [] [] ∧
    DeepActiveRuntimeLinear deepActiveGapTerm ∧
    Step ⟨ctxCounterSigma, deepActiveGapTerm⟩
      ⟨ctxCounterSigma, deepActiveGapTerm'⟩ ∧
    DeepActiveRuntimeLinear deepActiveGapTerm' ∧
    Step ⟨ctxCounterSigma, deepActiveGapTerm'⟩
      ⟨ctxCounterSigma, deepActiveGapTerm''⟩ ∧
    ¬ DeepActiveRuntimeLinear deepActiveGapTerm'' := by
  rcases deepActiveRuntimeLinear_beta_bridge_gap with
    ⟨hTyp, hDeep0, hStep0, hDeep1, _hGap⟩
  refine ⟨hTyp, hDeep0, hStep0, hDeep1, ?_, ?_⟩
  · have hInner :
        Step ⟨ctxCounterSigma, deepActiveGapHandle⟩
          ⟨ctxCounterSigma, deepActiveGapClauseBody⟩ := by
      simpa [deepActiveGapHandle, deepActiveGapClauses,
        deepActiveGapClauseBody, subst] using
        (Step.handleOpDirect ctxCounterSigma
          EffectLabel.accum Term.unit
          [EffectLabel.accum]
          deepActiveGapClauses
          "y" "k" deepActiveGapClauseBody Typ.unit
          IsValue.unit
          ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
          (by simp [deepActiveGapClauses]))
    simpa [deepActiveGapTerm', deepActiveGapTerm'',
      deepActiveGapAbsBody, deepActiveGapHandle,
      deepActiveGapClauseBody, deepActiveGapClauses,
      plug, subst, substClauses] using
      (Step.ctx ctxCounterSigma ctxCounterSigma
        (EvalCtx.pairR (Term.loc 1))
        deepActiveGapHandle deepActiveGapClauseBody hInner)
  · simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearClauses,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      deepActiveGapTerm'', deepActiveGapClauseBody]

/-- The stronger handler-aware invariant blocks the beta-gap witness at
    the initial application: the function-side beta footprint already
    exposes the dormant handler clause's location, so it cannot coexist
    with the active argument location. -/
theorem handlerAwareRuntimeLinear_beta_gap_blocks_counterexample :
    ¬ HandlerAwareRuntimeLinear deepActiveGapTerm := by
  simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
    LocRefsDisjoint, LocRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    deepActiveGapTerm, deepActiveGapAbsBody, deepActiveGapHandle,
    deepActiveGapClauses, deepActiveGapClauseBody]

private def handlerAwareBetaSubstCounterBody : Term :=
  Term.pair (Term.var "x") (Term.var "x")

private def handlerAwareBetaSubstCounterTerm : Term :=
  Term.app
    (Term.abs "x" (Typ.tensor DimList.empty) handlerAwareBetaSubstCounterBody)
    (Term.loc 1)

private def handlerAwareBetaSubstCounterTerm' : Term :=
  Term.pair (Term.loc 1) (Term.loc 1)

/-- The final handler-aware invariant is still not closed under raw
    beta without the typing-side linear-use discipline: variables carry
    no runtime locations before substitution, so duplicating a bound
    variable is invisible until the argument location is substituted. -/
theorem handlerAwareRuntimeLinear_beta_subst_counterexample :
    HandlerAwareRuntimeLinear handlerAwareBetaSubstCounterTerm ∧
    Step ⟨([] : Store), handlerAwareBetaSubstCounterTerm⟩
      ⟨[], handlerAwareBetaSubstCounterTerm'⟩ ∧
    ¬ HandlerAwareRuntimeLinear handlerAwareBetaSubstCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handlerAwareBetaSubstCounterTerm, handlerAwareBetaSubstCounterBody]
  · simpa [handlerAwareBetaSubstCounterTerm, handlerAwareBetaSubstCounterTerm',
      handlerAwareBetaSubstCounterBody, subst] using
      (Step.beta ([] : Store) "x" (Typ.tensor DimList.empty)
        handlerAwareBetaSubstCounterBody (Term.loc 1) (IsValue.loc 1))
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handlerAwareBetaSubstCounterTerm']

/-- The substitution-aware sidecar blocks the raw beta witness before
    the step fires: the duplicated binder is already visible at the
    abstraction body's variable surface. -/
theorem substAwareHandlerRuntimeLinear_beta_subst_blocks_counterexample :
    ¬ SubstAwareHandlerRuntimeLinear handlerAwareBetaSubstCounterTerm := by
  simp [SubstAwareHandlerRuntimeLinear,
    HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
    activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
    LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handlerAwareBetaSubstCounterTerm, handlerAwareBetaSubstCounterBody]

/-- The public config-level runtime-safety surface also rejects the raw
    beta/substitution witness when the duplicated location is live in
    the store: the blocker is the substitution-aware sidecar, not store
    liveness. -/
theorem runtimeSafeConfig_beta_subst_blocks_counterexample :
    ¬ RuntimeSafeConfig ⟨ctxCounterSigma, handlerAwareBetaSubstCounterTerm⟩ := by
  simp [RuntimeSafeConfig, RuntimeSafe, StoreAwareSubstHandlerRuntimeLinear,
    StoreLiveLocRefs, ctxCounterSigma, storeLookup, List.find?,
    SubstAwareHandlerRuntimeLinear,
    HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
    activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
    LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handlerAwareBetaSubstCounterTerm, handlerAwareBetaSubstCounterBody]

private def handlerAwareHandleSubstCounterBody : Term :=
  Term.pair (Term.var "x") (Term.var "x")

private def handlerAwareHandleSubstCounterClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.accum, "x", "k", handlerAwareHandleSubstCounterBody)]

private def handlerAwareHandleSubstCounterTerm : Term :=
  Term.handle [EffectLabel.accum]
    (Term.perform EffectLabel.accum (Term.loc 1))
    handlerAwareHandleSubstCounterClauses

private def handlerAwareHandleSubstCounterTerm' : Term :=
  Term.pair (Term.loc 1) (Term.loc 1)

/-- Raw direct-handler preservation also still needs the typing-side
    one-shot continuation / linear-argument discipline: the clause body
    can duplicate `x` without violating the runtime predicate until the
    handled value is substituted. -/
theorem handlerAwareRuntimeLinear_handleOpDirect_subst_counterexample :
    HandlerAwareRuntimeLinear handlerAwareHandleSubstCounterTerm ∧
    Step ⟨([] : Store), handlerAwareHandleSubstCounterTerm⟩
      ⟨[], handlerAwareHandleSubstCounterTerm'⟩ ∧
    ¬ HandlerAwareRuntimeLinear handlerAwareHandleSubstCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handlerAwareHandleSubstCounterTerm, handlerAwareHandleSubstCounterClauses,
      handlerAwareHandleSubstCounterBody]
  · simpa [handlerAwareHandleSubstCounterTerm, handlerAwareHandleSubstCounterTerm',
      handlerAwareHandleSubstCounterClauses, handlerAwareHandleSubstCounterBody,
      subst, substClauses, directIdCont, directIdContName] using
      (Step.handleOpDirect ([] : Store)
        EffectLabel.accum (Term.loc 1) [EffectLabel.accum]
        handlerAwareHandleSubstCounterClauses
        "x" "k" handlerAwareHandleSubstCounterBody Typ.unit
        (IsValue.loc 1)
        ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
        (by simp [handlerAwareHandleSubstCounterClauses]))
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handlerAwareHandleSubstCounterTerm']

/-- The same substitution-aware sidecar also rejects the raw direct
    handler witness: duplicating the operation argument binder is
    already syntactically visible in the selected clause body. -/
theorem substAwareHandlerRuntimeLinear_handleOpDirect_subst_blocks_counterexample :
    ¬ SubstAwareHandlerRuntimeLinear handlerAwareHandleSubstCounterTerm := by
  simp [SubstAwareHandlerRuntimeLinear,
    HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
    activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
    LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handlerAwareHandleSubstCounterTerm, handlerAwareHandleSubstCounterClauses,
    handlerAwareHandleSubstCounterBody]

/-- The config-level runtime-safety surface rejects the raw direct
    handler substitution witness even when the handled location is live
    in the store. -/
theorem runtimeSafeConfig_handleOpDirect_subst_blocks_counterexample :
    ¬ RuntimeSafeConfig ⟨ctxCounterSigma, handlerAwareHandleSubstCounterTerm⟩ := by
  simp [RuntimeSafeConfig, RuntimeSafe, StoreAwareSubstHandlerRuntimeLinear,
    StoreLiveLocRefs, ctxCounterSigma, storeLookup, List.find?,
    SubstAwareHandlerRuntimeLinear,
    HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
    activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
    LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handlerAwareHandleSubstCounterTerm, handlerAwareHandleSubstCounterClauses,
    handlerAwareHandleSubstCounterBody]

private def runtimeSafeHandleCtxCaptureCounterValue : Term :=
  Term.abs "z" Typ.unit (Term.var "y")

private def runtimeSafeHandleCtxCaptureCounterBody : Term :=
  Term.abs "y" Typ.unit (Term.pair (Term.var "x") (Term.var "y"))

private def runtimeSafeHandleCtxCaptureCounterClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.accum, "x", "k", runtimeSafeHandleCtxCaptureCounterBody)]

private def runtimeSafeHandleCtxCaptureCounterTerm : Term :=
  Term.handle [EffectLabel.accum]
    (plug EvalCtx.copy
      (Term.perform EffectLabel.accum runtimeSafeHandleCtxCaptureCounterValue))
    runtimeSafeHandleCtxCaptureCounterClauses

private def runtimeSafeHandleCtxCaptureCounterTerm' : Term :=
  Term.abs "y" Typ.unit
    (Term.pair runtimeSafeHandleCtxCaptureCounterValue (Term.var "y"))

/-- The current config-level runtime-safety surface is still not
    closed under generic captured-handler steps: even when the source
    handle is runtime-safe, naive named substitution can capture a free
    variable from the handled value under a clause-local binder and
    duplicate that variable at the active runtime surface. -/
theorem runtimeSafeConfig_handleOpCtx_capture_counterexample :
    RuntimeSafeConfig ⟨([] : Store), runtimeSafeHandleCtxCaptureCounterTerm⟩ ∧
    Step ⟨[], runtimeSafeHandleCtxCaptureCounterTerm⟩
      ⟨[], runtimeSafeHandleCtxCaptureCounterTerm'⟩ ∧
    ¬ RuntimeSafeConfig ⟨[], runtimeSafeHandleCtxCaptureCounterTerm'⟩ := by
  refine ⟨?_, ?_, ?_⟩
  · simp [RuntimeSafeConfig, RuntimeSafe, StoreAwareSubstHandlerRuntimeLinear,
      StoreLiveLocRefs, SubstAwareHandlerRuntimeLinear,
      HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
      LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      locRefs, locRefsClauses,
      runtimeSafeHandleCtxCaptureCounterTerm,
      runtimeSafeHandleCtxCaptureCounterClauses,
      runtimeSafeHandleCtxCaptureCounterBody,
      runtimeSafeHandleCtxCaptureCounterValue, plug]
  · simpa [runtimeSafeHandleCtxCaptureCounterTerm,
      runtimeSafeHandleCtxCaptureCounterTerm',
      runtimeSafeHandleCtxCaptureCounterClauses,
      runtimeSafeHandleCtxCaptureCounterBody,
      runtimeSafeHandleCtxCaptureCounterValue, plug, subst] using
      (Step.handleOpCtx ([] : Store)
        EffectLabel.accum runtimeSafeHandleCtxCaptureCounterValue
        [EffectLabel.accum] EvalCtx.copy
        runtimeSafeHandleCtxCaptureCounterClauses
        "x" "k" runtimeSafeHandleCtxCaptureCounterBody Typ.unit
        (IsValue.abs "z" Typ.unit (Term.var "y"))
        ⟨Typ.unit, by simp [OpSigMatch, opArgType, opRetType]⟩
        (by simp [runtimeSafeHandleCtxCaptureCounterClauses])
        (by simp)
        (by simp [EvalCtx.noHandleFor]))
  · simp [RuntimeSafeConfig, RuntimeSafe, StoreAwareSubstHandlerRuntimeLinear,
      StoreLiveLocRefs, SubstAwareHandlerRuntimeLinear,
      HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
      LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      locRefs, locRefsClauses,
      runtimeSafeHandleCtxCaptureCounterTerm',
      runtimeSafeHandleCtxCaptureCounterValue]

private def handlerAwareCtxFreshCounterTerm : Term :=
  Term.pair (Term.const 0.0 DimList.empty) (Term.loc 1)

private def handlerAwareCtxFreshCounterTerm' : Term :=
  Term.pair (Term.loc 1) (Term.loc 1)

/-- Complex contexts are still a real raw-runtime blocker: a store step
    can allocate a fresh live location that collides with a stale
    sibling location mention outside the redex, and the bare runtime
    invariant does not relate terms to store liveness. -/
theorem handlerAwareRuntimeLinear_ctx_fresh_counterexample :
    HandlerAwareRuntimeLinear handlerAwareCtxFreshCounterTerm ∧
    Step ⟨([] : Store), handlerAwareCtxFreshCounterTerm⟩
      ⟨storeExtend [] 1 ⟨DimList.empty, 0.0⟩,
        handlerAwareCtxFreshCounterTerm'⟩ ∧
    ¬ HandlerAwareRuntimeLinear handlerAwareCtxFreshCounterTerm' := by
  refine ⟨?_, ?_, ?_⟩
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handlerAwareCtxFreshCounterTerm]
  · have hInner :
        Step ⟨([] : Store), Term.const 0.0 DimList.empty⟩
          ⟨storeExtend [] 1 ⟨DimList.empty, 0.0⟩, Term.loc 1⟩ := by
      simpa [storeFreshLoc] using
        (Step.tconst ([] : Store) 0.0 DimList.empty 1 rfl)
    simpa [handlerAwareCtxFreshCounterTerm, handlerAwareCtxFreshCounterTerm', plug] using
      (Step.ctx ([] : Store) (storeExtend [] 1 ⟨DimList.empty, 0.0⟩)
        (EvalCtx.pairL (Term.loc 1))
        (Term.const 0.0 DimList.empty) (Term.loc 1) hInner)
  · simp [HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
      StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
      LocRefsDisjoint, LocRefsSeparated,
      ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
      handlerAwareCtxFreshCounterTerm']

/-- Fresh-allocation leakage needs the store sidecar rather than the
    substitution sidecar: the initial term already mentions a stale
    location that is absent from the store. -/
theorem storeAwareSubstHandlerRuntimeLinear_ctx_fresh_blocks_counterexample :
    ¬ StoreAwareSubstHandlerRuntimeLinear ([] : Store)
        handlerAwareCtxFreshCounterTerm := by
  simp [StoreAwareSubstHandlerRuntimeLinear, StoreLiveLocRefs,
    SubstAwareHandlerRuntimeLinear,
    HandlerAwareRuntimeLinear, HandlerAwareRuntimeLinearClauses,
    SubstAwareRuntimeLinear, SubstAwareRuntimeLinearClauses,
    StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
    activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
    LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
    ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
    handlerAwareCtxFreshCounterTerm, locRefs, storeLookup, List.find?]

/-- Public configuration-level wrapper for the stale-location context
    witness: `RuntimeSafeConfig` rejects the term before the fresh
    allocation step can collide with the sibling location mention. -/
theorem runtimeSafeConfig_ctx_fresh_blocks_counterexample :
    ¬ RuntimeSafeConfig ⟨([] : Store), handlerAwareCtxFreshCounterTerm⟩ := by
  simpa [RuntimeSafeConfig, RuntimeSafe] using
    storeAwareSubstHandlerRuntimeLinear_ctx_fresh_blocks_counterexample

theorem wellScoped_plug_inner
    {E : EvalCtx} {e : Term}
    (h : WellScoped (plug E e)) :
    WellScoped e := by
  cases E
  · simpa [plug] using h
  · simp [plug, WellScoped, boundVars, List.nodup_append] at h
    exact h.1
  · simp [plug, WellScoped, boundVars, List.nodup_append] at h
    exact h.2.1
  · exact (wellScoped_letBind_body h).1
  · simpa [plug] using h
  · exact (wellScoped_letpair_body h).1
  · exact wellScoped_pair_left h
  · exact wellScoped_pair_right h
  · simpa [plug] using h
  · simpa [plug] using h
  · simp [plug, WellScoped, boundVars, List.nodup_append] at h
    exact h.1
  · simp [plug, WellScoped, boundVars, List.nodup_append] at h
    exact h.2.1
  · simp [plug, WellScoped, boundVars, List.nodup_append] at h
    exact h.1
  · simp [plug, WellScoped, boundVars, List.nodup_append] at h
    exact h.2.1
  · simpa [plug] using h
  · simpa [plug] using h
  · simpa [plug] using h
  · exact wellScoped_handle_body h
  · simpa [plug] using h

private theorem locRefsSeparated_left_of_nodup_append
    {xs ys : List Loc}
    (h : (xs ++ ys).Nodup) :
    LocRefsSeparated xs ys := by
  rcases List.nodup_append.mp h with ⟨_hxs, _hys, hxy⟩
  intro ell hx hy
  exact hxy ell hx ell hy rfl

private theorem locRefsSeparated_right_of_nodup_append
    {xs ys : List Loc}
    (h : (xs ++ ys).Nodup) :
    LocRefsSeparated ys xs := by
  rcases List.nodup_append.mp h with ⟨_hxs, _hys, hxy⟩
  intro ell hy hx
  exact hxy ell hx ell hy rfl

private theorem locRefsSeparated_left_of_append
    {xs ys zs : List Loc}
    (h : LocRefsSeparated (xs ++ ys) zs) :
    LocRefsSeparated xs zs := by
  intro ell hx hz
  exact h ell (List.mem_append_left _ hx) hz

private theorem locRefsSeparated_right_of_append
    {xs ys zs : List Loc}
    (h : LocRefsSeparated (xs ++ ys) zs) :
    LocRefsSeparated ys zs := by
  intro ell hy hz
  exact h ell (List.mem_append_right _ hy) hz

private theorem locRefsSeparated_append
    {xs ys zs : List Loc}
    (h1 : LocRefsSeparated xs zs)
    (h2 : LocRefsSeparated ys zs) :
    LocRefsSeparated (xs ++ ys) zs := by
  intro ell hxy hz
  rcases List.mem_append.mp hxy with hx | hy
  · exact h1 ell hx hz
  · exact h2 ell hy hz

private theorem locRefsSeparated_lhs_append
    {xs ys zs : List Loc}
    (h1 : LocRefsSeparated xs ys)
    (h2 : LocRefsSeparated xs zs) :
    LocRefsSeparated xs (ys ++ zs) := by
  intro ell hx hyz
  rcases List.mem_append.mp hyz with hy | hz
  · exact h1 ell hx hy
  · exact h2 ell hx hz

private theorem locRefsSeparated_rhs_left_of_append
    {xs ys zs : List Loc}
    (h : LocRefsSeparated xs (ys ++ zs)) :
    LocRefsSeparated xs ys := by
  intro ell hx hy
  exact h ell hx (List.mem_append_left _ hy)

private theorem locRefsSeparated_rhs_right_of_append
    {xs ys zs : List Loc}
    (h : LocRefsSeparated xs (ys ++ zs)) :
    LocRefsSeparated xs zs := by
  intro ell hx hz
  exact h ell hx (List.mem_append_right _ hz)

private theorem runtimeLinear_plug_replace
    {E : EvalCtx} {e e' : Term}
    (hOld : RuntimeLinear (plug E e))
    (hNew : RuntimeLinear e')
    (hSepCtxNew : LocRefsSeparated (ctxLocRefs E) (locRefs e'))
    (hSepNewCtx : LocRefsSeparated (locRefs e') (ctxLocRefs E)) :
    RuntimeLinear (plug E e') := by
  cases E with
  | hole =>
      simpa [plug] using hNew
  | appL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | appR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨hFrame, _hOldInner, _hSep⟩
      exact List.nodup_append.mpr ⟨hFrame, hNew, by
        intro ell hLocFrame ell' hLocNew hEq
        subst ell'
        exact hSepCtxNew ell hLocFrame hLocNew⟩
  | letBind x e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | copy =>
      simpa [RuntimeLinear, plug, locRefs] using hNew
  | letpair x y e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | pairL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | pairR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨hFrame, _hOldInner, _hSep⟩
      exact List.nodup_append.mpr ⟨hFrame, hNew, by
        intro ell hLocFrame ell' hLocNew hEq
        subst ell'
        exact hSepCtxNew ell hLocFrame hLocNew⟩
  | fst =>
      simpa [RuntimeLinear, plug, locRefs] using hNew
  | snd =>
      simpa [RuntimeLinear, plug, locRefs] using hNew
  | addL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | addR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨hFrame, _hOldInner, _hSep⟩
      exact List.nodup_append.mpr ⟨hFrame, hNew, by
        intro ell hLocFrame ell' hLocNew hEq
        subst ell'
        exact hSepCtxNew ell hLocFrame hLocNew⟩
  | mulL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | mulR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨hFrame, _hOldInner, _hSep⟩
      exact List.nodup_append.mpr ⟨hFrame, hNew, by
        intro ell hLocFrame ell' hLocNew hEq
        subst ell'
        exact hSepCtxNew ell hLocFrame hLocNew⟩
  | sum d =>
      simpa [RuntimeLinear, plug, locRefs] using hNew
  | expand d =>
      simpa [RuntimeLinear, plug, locRefs] using hNew
  | uniformLike lo hi =>
      simpa [RuntimeLinear, plug, locRefs] using hNew
  | handle epsH clauses =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using hOld) with
        ⟨_hOldInner, hFrame, _hSep⟩
      exact List.nodup_append.mpr ⟨hNew, hFrame, by
        intro ell hLocNew ell' hLocFrame hEq
        subst ell'
        exact hSepNewCtx ell hLocNew hLocFrame⟩
  | perform op =>
      simpa [RuntimeLinear, plug, locRefs] using hNew

private theorem deepActiveRuntimeLinear_multiPlug_chainNodup
    {Es : EvalCtxChain} {e : Term}
    (h : DeepActiveRuntimeLinear (multiPlug Es e)) :
    (activeChainLocRefs Es).Nodup := by
  induction Es generalizing e with
  | nil =>
      simp [activeChainLocRefs]
  | cons E Es ih =>
      rcases deepActiveRuntimeLinear_plug (E := E) (e := multiPlug Es e) h with
        ⟨hInner, hCtx, hSep, _hSepSymm⟩
      refine List.nodup_append.mpr ⟨deepActiveCtx_activeNodup hCtx, ih hInner, ?_⟩
      intro ell hmemE ell' hmemEs hEq
      subst ell'
      have hLocMulti : ell ∈ activeLocRefs (multiPlug Es e) := by
        exact (mem_activeLocRefs_multiPlug Es e ell).2 (Or.inl hmemEs)
      exact hSep ell hmemE hLocMulti

private theorem deepActiveRuntimeLinear_plug_replace
    {E : EvalCtx} {e e' : Term}
    (hOld : DeepActiveRuntimeLinear (plug E e))
    (hNew : DeepActiveRuntimeLinear e')
    (hSepCtxNew : LocRefsSeparated (activeCtxLocRefs E) (activeLocRefs e'))
    (hSepNewCtx : LocRefsSeparated (activeLocRefs e') (activeCtxLocRefs E)) :
    DeepActiveRuntimeLinear (plug E e') := by
  rcases deepActiveRuntimeLinear_plug (E := E) (e := e) hOld with
    ⟨_hInner, hCtx, _hSepOld, _hSepOldSymm⟩
  exact deepActiveRuntimeLinear_plug_of hCtx hNew hSepCtxNew hSepNewCtx

private theorem deepActiveRuntimeLinear_multiPlug_replace
    {Es : EvalCtxChain} {e e' : Term}
    (hOld : DeepActiveRuntimeLinear (multiPlug Es e))
    (hNew : DeepActiveRuntimeLinear e')
    (hSepCtxNew : LocRefsSeparated (activeChainLocRefs Es) (activeLocRefs e'))
    (hSepNewCtx : LocRefsSeparated (activeLocRefs e') (activeChainLocRefs Es)) :
    DeepActiveRuntimeLinear (multiPlug Es e') := by
  rcases deepActiveRuntimeLinear_multiPlug (Es := Es) (e := e) hOld with
    ⟨_hInner, hEs, _hSepOld, _hSepOldSymm⟩
  exact deepActiveRuntimeLinear_multiPlug_of
    (deepActiveRuntimeLinear_multiPlug_chainNodup hOld)
    hEs hNew hSepCtxNew hSepNewCtx

private theorem deepActiveRuntimeLinear_clause_mem
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses)
    (hClauses : DeepActiveRuntimeLinearClauses clauses) :
    DeepActiveRuntimeLinear hb := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases cl with ⟨op0, x0, k0, hb0⟩
      rcases hClauses with ⟨hHead, hTail⟩
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        exact hHead
      · exact ih htl hTail

private theorem deepActiveRuntimeLinear_handle_body
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    (h : DeepActiveRuntimeLinear (Term.handle epsH body clauses)) :
    DeepActiveRuntimeLinear body := by
  simpa [DeepActiveRuntimeLinear] using h.2.1

private theorem deepActiveRuntimeLinear_handle_clause_mem
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (h : DeepActiveRuntimeLinear (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    DeepActiveRuntimeLinear hb := by
  exact deepActiveRuntimeLinear_clause_mem hmem h.2.2

private theorem deepActiveRuntimeLinear_identity_cont
    {y : String} {t : Typ} :
    DeepActiveRuntimeLinear (Term.abs y t (Term.var y)) := by
  simp [DeepActiveRuntimeLinear, ActiveRuntimeLinear, activeLocRefs]

private theorem deepActiveRuntimeLinear_captured_handle
    {epsH : EffectRow} {E : EvalCtx} {e : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {y : String}
    (h : DeepActiveRuntimeLinear (Term.handle epsH (plug E e) clauses)) :
    DeepActiveRuntimeLinear (Term.handle epsH (plug E (Term.var y)) clauses) := by
  have hPlugVar :
      DeepActiveRuntimeLinear (plug E (Term.var y)) := by
    apply deepActiveRuntimeLinear_plug_replace
      (hOld := deepActiveRuntimeLinear_handle_body h)
    · simpa [DeepActiveRuntimeLinear]
    · simpa [LocRefsSeparated, activeLocRefs]
    · simpa [LocRefsSeparated, activeLocRefs]
  exact ⟨by
      simpa [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses]
        using deepActiveRuntimeLinear_active hPlugVar,
    hPlugVar, h.2.2⟩

private theorem deepActiveRuntimeLinear_captured_multiHandle
    {epsH : EffectRow} {Es : EvalCtxChain} {e : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {y : String}
    (h : DeepActiveRuntimeLinear (Term.handle epsH (multiPlug Es e) clauses)) :
    DeepActiveRuntimeLinear (Term.handle epsH (multiPlug Es (Term.var y)) clauses) := by
  have hPlugVar :
      DeepActiveRuntimeLinear (multiPlug Es (Term.var y)) := by
    apply deepActiveRuntimeLinear_multiPlug_replace
      (hOld := deepActiveRuntimeLinear_handle_body h)
    · simpa [DeepActiveRuntimeLinear]
    · simpa [LocRefsSeparated, activeLocRefs, activeChainLocRefs]
    · simpa [LocRefsSeparated, activeLocRefs, activeChainLocRefs]
  exact ⟨by
      simpa [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses]
        using deepActiveRuntimeLinear_active hPlugVar,
    hPlugVar, h.2.2⟩

private theorem deepActiveRuntimeLinear_captured_cont
    {epsH : EffectRow} {E : EvalCtx} {e : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {y : String} {t : Typ}
    (h : DeepActiveRuntimeLinear (Term.handle epsH (plug E e) clauses)) :
    DeepActiveRuntimeLinear
      (Term.abs y t (Term.handle epsH (plug E (Term.var y)) clauses)) := by
  exact ⟨by
      simpa [ActiveRuntimeLinear, activeLocRefs]
        using deepActiveRuntimeLinear_active (deepActiveRuntimeLinear_captured_handle (y := y) h),
    deepActiveRuntimeLinear_captured_handle (y := y) h⟩

private theorem deepActiveRuntimeLinear_captured_multiCont
    {epsH : EffectRow} {Es : EvalCtxChain} {e : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {y : String} {t : Typ}
    (h : DeepActiveRuntimeLinear (Term.handle epsH (multiPlug Es e) clauses)) :
    DeepActiveRuntimeLinear
      (Term.abs y t (Term.handle epsH (multiPlug Es (Term.var y)) clauses)) := by
  exact ⟨by
      simpa [ActiveRuntimeLinear, activeLocRefs]
        using deepActiveRuntimeLinear_active (deepActiveRuntimeLinear_captured_multiHandle (y := y) h),
    deepActiveRuntimeLinear_captured_multiHandle (y := y) h⟩

private theorem runtimeLinear_clause_mem_separated
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term} {rhsRefs : List Loc}
    (hmem : (op, x, k, hb) ∈ clauses)
    (hlin : (locRefsClauses clauses ++ rhsRefs).Nodup) :
    RuntimeLinear hb ∧
      LocRefsSeparated (locRefs hb) rhsRefs ∧
      LocRefsSeparated rhsRefs (locRefs hb) := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases cl with ⟨op', x', k', hb'⟩
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        have hsplit : (locRefs hb ++ (locRefsClauses rest ++ rhsRefs)).Nodup := by
          simpa [locRefsClauses, List.append_assoc] using hlin
        rcases List.nodup_append.mp hsplit with ⟨hlinHb, _hrest, hsepHb⟩
        refine ⟨hlinHb, ?_, ?_⟩
        · intro ell hhb hrhs
          exact hsepHb ell hhb ell (List.mem_append_right _ hrhs) rfl
        · intro ell hrhs hhb
          exact hsepHb ell hhb ell (List.mem_append_right _ hrhs) rfl
      · have hrest : (locRefsClauses rest ++ rhsRefs).Nodup := by
          simpa [locRefsClauses, List.append_assoc] using
            (List.nodup_append.mp (by simpa [locRefsClauses, List.append_assoc] using hlin)).2.1
        exact ih htl hrest

private theorem runtimeLinear_subst_singleton_separated
    {Sigma : StoreTyp}
    {x : String} {tArg tRet : Typ} {slot : Option Typ}
    {body v : Term} {eps : EffectRow} {rhsRefs : List Loc}
    (hBody : HasType [] Sigma [(x, some tArg)] body tRet eps [(x, slot)])
    (hBodyLex : LexicallyScoped [(x, some tArg)] body)
    (hV : HasType [] Sigma [] v tArg [] [])
    (hVScope : WellScoped v)
    (hVClosed : Closed v)
    (hlinBody : RuntimeLinear body)
    (hsepBodyRhs : LocRefsSeparated (locRefs body) rhsRefs)
    (hsepRhsBody : LocRefsSeparated rhsRefs (locRefs body))
    (hlinV : RuntimeLinear v)
    (hsepVBody : LocRefsSeparated (locRefs v) (locRefs body))
    (hsepBodyV : LocRefsSeparated (locRefs body) (locRefs v))
    (hsepVRhs : LocRefsSeparated (locRefs v) rhsRefs)
    (hsepRhsV : LocRefsSeparated rhsRefs (locRefs v))
    (hxBody : x ∉ boundVars body) :
    RuntimeLinear (subst body v x) ∧
      LocRefsSeparated (locRefs (subst body v x)) rhsRefs ∧
      LocRefsSeparated rhsRefs (locRefs (subst body v x)) := by
  rcases transport_typing_lexical hBody hBodyLex with
    ⟨bodyDB, hEraseBody, hBodyDB⟩
  rcases transport_typing_lexical hV (lexical_nil hVScope) with
    ⟨vDB, hEraseV, hVDB⟩
  have hEraseSubst :
      eraseTerm [] (subst body v x) = some (substDBAux 0 vDB bodyDB) := by
    simpa using
      eraseTerm_subst_tail (ρ := []) (v := v) (x := x) (by simp) hEraseV hEraseBody hxBody
  have hlinBodyDB : RuntimeLinearDB bodyDB := by
    exact (eraseTerm_runtimeLinear_iff hEraseBody).1 hlinBody
  have hlinVDB : RuntimeLinearDB vDB := by
    exact (eraseTerm_runtimeLinear_iff hEraseV).1 hlinV
  have hsepBodyRhsDB :
      LocRefsSeparated (locRefsDB bodyDB) rhsRefs := by
    simpa [eraseTerm_locRefs hEraseBody] using hsepBodyRhs
  have hsepRhsBodyDB :
      LocRefsSeparated rhsRefs (locRefsDB bodyDB) := by
    simpa [eraseTerm_locRefs hEraseBody] using hsepRhsBody
  have hsepVBodyDB :
      LocRefsSeparated (locRefsDB vDB) (locRefsDB bodyDB) := by
    simpa [eraseTerm_locRefs hEraseV, eraseTerm_locRefs hEraseBody] using hsepVBody
  have hsepBodyVDB :
      LocRefsSeparated (locRefsDB bodyDB) (locRefsDB vDB) := by
    simpa [eraseTerm_locRefs hEraseV, eraseTerm_locRefs hEraseBody] using hsepBodyV
  have hsepVRhsDB :
      LocRefsSeparated (locRefsDB vDB) rhsRefs := by
    simpa [eraseTerm_locRefs hEraseV] using hsepVRhs
  have hsepRhsVDB :
      LocRefsSeparated rhsRefs (locRefsDB vDB) := by
    simpa [eraseTerm_locRefs hEraseV] using hsepRhsV
  have hlocsSubst :
      locRefs (subst body v x) = locRefsDB (substDBAux 0 vDB bodyDB) := by
    simpa using eraseTerm_locRefs hEraseSubst
  cases hslot : slot with
  | none =>
      have hRes :=
        runtimeLinearDB_subst_dead_separated
          (Γ := []) (rhsRefs := rhsRefs)
          0 (by simp) (by simpa [hslot] using hBodyDB) hVDB
          hlinVDB hsepVBodyDB hsepBodyVDB hsepVRhsDB hsepRhsVDB
          hlinBodyDB hsepBodyRhsDB hsepRhsBodyDB
      refine ⟨?_, ?_, ?_⟩
      · exact (eraseTerm_runtimeLinear_iff hEraseSubst).2 hRes.1
      · simpa [hlocsSubst] using hRes.2.1
      · simpa [hlocsSubst] using hRes.2.2
  | some tKeep =>
      have hLiveIn :
          (eraseCtx [(x, some tArg)])[0]? = some (some tArg) := by
        simp [eraseCtx]
      have hMono := hasTypeDB_live_slot_monotone hBodyDB 0 tArg hLiveIn
      have hOutLive :
          (eraseCtx [(x, some tKeep)])[0]? = some (some tArg) := by
        rcases hMono with hLive | hDead
        · simpa [eraseCtx, hslot] using hLive
        · simp [eraseCtx, hslot] at hDead
      have htKeep : tKeep = tArg := by
        simpa [eraseCtx] using hOutLive
      subst htKeep
      have hRes :=
        runtimeLinearDB_subst_live_gen_separated
          (Γ1 := []) (Γ2 := []) (v := vDB) (rhsRefs := rhsRefs)
          0 (by simp) (by simp) (by simpa [hslot] using hBodyDB)
          hlinBodyDB hsepBodyRhsDB hsepRhsBodyDB
      refine ⟨?_, ?_, ?_⟩
      · exact (eraseTerm_runtimeLinear_iff hEraseSubst).2 hRes.1
      · simpa [hlocsSubst] using hRes.2.1
      · simpa [hlocsSubst] using hRes.2.2

private theorem runtimeLinear_beta_via_db
    {Sigma : StoreTyp}
    {x : String} {tArg tRet : Typ}
    {body v : Term} {eps : EffectRow} {locs : List Loc}
    (hTyp : HasType [] Sigma [] (Term.app (Term.abs x tArg body) v) tRet eps [])
    (hv : IsValue v)
    (hScope : WellScoped (Term.app (Term.abs x tArg body) v))
    (hLinear : RuntimeLinear (Term.app (Term.abs x tArg body) v))
    (hsep : LocRefsSeparated locs (locRefs (Term.app (Term.abs x tArg body) v))) :
    RuntimeLinear (subst body v x) ∧
      LocRefsSeparated (locRefs (subst body v x)) locs ∧
      LocRefsSeparated locs (locRefs (subst body v x)) := by
  rcases HasType.app_inv_sub_bridge hTyp with
    ⟨GammaMid, t1, epsBody, epsFun, epsArg, hFun, hArg, hSub⟩
  have hMid : GammaMid = [] := has_type_closed_output_of_closed_input hFun
  subst hMid
  rcases HasType.abs_inv hFun with
    ⟨tRet', epsBody', GammaBody, slot, hArrow, hBody, hOut⟩
  injection hArrow with hT1 _hRet _hEff
  subst t1
  subst tRet'
  subst epsBody'
  subst GammaBody
  have hClosedV : Closed v := has_type_closed_term_of_closed_input hArg
  have hArgNil : HasType [] Sigma [] v tArg [] [] :=
    HasType.value_eff_polymorphic_bridge hArg hv []
  have hScopeAbs : WellScoped (Term.abs x tArg body) := by
    simpa [plug] using
      (wellScoped_plug_inner
        (E := EvalCtx.appL v)
        (e := Term.abs x tArg body) hScope)
  have hScopeV : WellScoped v := by
    simpa [plug] using
      (wellScoped_plug_inner
        (E := EvalCtx.appR (Term.abs x tArg body))
        (e := v) hScope)
  have hBodyScope := wellScoped_abs_body hScopeAbs
  have hLocs : (locRefs body ++ locRefs v).Nodup := by
    simpa [RuntimeLinear, locRefs] using hLinear
  rcases List.nodup_append.mp hLocs with ⟨hlinBody, hlinV, _hsepBodyV⟩
  refine runtimeLinear_subst_singleton_separated
    hBody (lexical_singleton hBodyScope.1 hBodyScope.2)
    hArgNil hScopeV hClosedV hlinBody ?_ ?_ hlinV ?_ ?_ ?_ ?_ hBodyScope.1
  · intro ell hBodyLoc hLocs
    exact hsep ell hLocs (by simp [locRefs, hBodyLoc])
  · intro ell hLocs hBodyLoc
    exact hsep ell hLocs (by simp [locRefs, hBodyLoc])
  · exact locRefsSeparated_right_of_nodup_append hLocs
  · exact locRefsSeparated_left_of_nodup_append hLocs
  · intro ell hVLoc hLocs
    exact hsep ell hLocs (by simp [locRefs, hVLoc])
  · intro ell hLocs hVLoc
    exact hsep ell hLocs (by simp [locRefs, hVLoc])

private theorem runtimeLinear_letBind_via_db
    {Sigma : StoreTyp}
    {x : String} {v body : Term} {t : Typ} {eps : EffectRow} {locs : List Loc}
    (hTyp : HasType [] Sigma [] (Term.letBind x v body) t eps [])
    (hv : IsValue v)
    (hScope : WellScoped (Term.letBind x v body))
    (hLinear : RuntimeLinear (Term.letBind x v body))
    (hsep : LocRefsSeparated locs (locRefs (Term.letBind x v body))) :
    RuntimeLinear (subst body v x) ∧
      LocRefsSeparated (locRefs (subst body v x)) locs ∧
      LocRefsSeparated locs (locRefs (subst body v x)) := by
  rcases HasType.plug_letBind_inv hTyp with
    ⟨GammaMid, Gamma3, t1, eps1, eps2, slot, hV, hBody, hGammaOut, hSub⟩
  have hMid : GammaMid = [] := has_type_closed_output_of_closed_input hV
  subst hMid
  have hGamma3 : Gamma3 = [] := by simpa using hGammaOut
  subst hGamma3
  have hClosedV : Closed v := has_type_closed_term_of_closed_input hV
  have hVNil : HasType [] Sigma [] v t1 [] [] :=
    HasType.value_eff_polymorphic_bridge hV hv []
  have hScopeParts := wellScoped_letBind_body hScope
  have hLocs : (locRefs v ++ locRefs body).Nodup := by
    simpa [RuntimeLinear, locRefs] using hLinear
  rcases List.nodup_append.mp hLocs with ⟨hlinV, hlinBody, _hsep⟩
  refine runtimeLinear_subst_singleton_separated
    hBody (lexical_singleton hScopeParts.2.1 hScopeParts.2.2)
    hVNil hScopeParts.1 hClosedV hlinBody ?_ ?_ hlinV ?_ ?_ ?_ ?_ hScopeParts.2.1
  · intro ell hBodyLoc hLocs
    exact hsep ell hLocs (by simp [locRefs, hBodyLoc])
  · intro ell hLocs hBodyLoc
    exact hsep ell hLocs (by simp [locRefs, hBodyLoc])
  · exact locRefsSeparated_left_of_nodup_append hLocs
  · exact locRefsSeparated_right_of_nodup_append hLocs
  · intro ell hVLoc hLocs
    exact hsep ell hLocs (by simp [locRefs, hVLoc])
  · intro ell hLocs hVLoc
    exact hsep ell hLocs (by simp [locRefs, hVLoc])

private theorem runtimeLinear_letpair_via_db
    {Sigma : StoreTyp}
    {x y : String} {v1 v2 body : Term} {t : Typ} {eps : EffectRow} {locs : List Loc}
    (hTyp : HasType [] Sigma [] (Term.letpair x y (Term.pair v1 v2) body) t eps [])
    (hv1 : IsValue v1) (hv2 : IsValue v2)
    (hScope : WellScoped (Term.letpair x y (Term.pair v1 v2) body))
    (hLinear : RuntimeLinear (Term.letpair x y (Term.pair v1 v2) body))
    (hsep : LocRefsSeparated locs (locRefs (Term.letpair x y (Term.pair v1 v2) body))) :
    RuntimeLinear (subst (subst body v1 x) v2 y) ∧
      LocRefsSeparated (locRefs (subst (subst body v1 x) v2 y)) locs ∧
      LocRefsSeparated locs (locRefs (subst (subst body v1 x) v2 y)) := by
  rcases HasType.letpair_inv_sub_bridge hTyp with
    ⟨GammaMid, GammaBody, t1, t2, epsPair, epsBody, slotX, slotY, hPair, hBody, hOut, hSub⟩
  have hMid : GammaMid = [] := has_type_closed_output_of_closed_input hPair
  subst hMid
  subst GammaBody
  rcases HasType.pair_inv_sub_bridge hPair with
    ⟨GammaPair, epsV1, epsV2, hV1, hV2, hPairSub⟩
  have hPairMid : GammaPair = [] := has_type_closed_output_of_closed_input hV1
  subst hPairMid
  have hV1Closed : Closed v1 := has_type_closed_term_of_closed_input hV1
  have hV2Closed : Closed v2 := has_type_closed_term_of_closed_input hV2
  have hV1Nil : HasType [] Sigma [] v1 t1 [] [] :=
    HasType.value_eff_polymorphic_bridge hV1 hv1 []
  have hV2Nil : HasType [] Sigma [] v2 t2 [] [] :=
    HasType.value_eff_polymorphic_bridge hV2 hv2 []
  have hScopeParts := wellScoped_letpair_body hScope
  have hScopePair : WellScoped (Term.pair v1 v2) := hScopeParts.1
  have hxy : x ≠ y := hScopeParts.2.1
  have hxBody : x ∉ boundVars body := hScopeParts.2.2.1
  have hyBody : y ∉ boundVars body := hScopeParts.2.2.2.1
  have hScopeBody : WellScoped body := hScopeParts.2.2.2.2
  have hScopeV1 : WellScoped v1 := wellScoped_pair_left hScopePair
  have hScopeV2 : WellScoped v2 := wellScoped_pair_right hScopePair
  have hscopeNodup :
      (x :: y :: (boundVars v1 ++ boundVars v2 ++ boundVars body)).Nodup := by
    simpa [WellScoped, boundVars, List.append_assoc] using hScope
  have hyNotRest : y ∉ boundVars v1 ++ boundVars v2 ++ boundVars body := by
    simpa using (List.nodup_cons.mp (List.nodup_cons.mp hscopeNodup).2).1
  have hyV1 : y ∉ boundVars v1 := by
    intro hy
    exact hyNotRest (by simp [hy])
  have hLocs : ((locRefs v1 ++ locRefs v2) ++ locRefs body).Nodup := by
    simpa [RuntimeLinear, locRefs, List.append_assoc] using hLinear
  rcases List.nodup_append.mp hLocs with ⟨hlinV12, hlinBody, hsepV12BodyRaw⟩
  have hsepV12Body : LocRefsSeparated (locRefs v1 ++ locRefs v2) (locRefs body) :=
    locRefsSeparated_left_of_nodup_append hLocs
  have hsepBodyV12 : LocRefsSeparated (locRefs body) (locRefs v1 ++ locRefs v2) :=
    locRefsSeparated_right_of_nodup_append hLocs
  rcases List.nodup_append.mp hlinV12 with ⟨hlinV1, hlinV2, _⟩
  have hsepV1V2 : LocRefsSeparated (locRefs v1) (locRefs v2) :=
    locRefsSeparated_left_of_nodup_append hlinV12
  have hsepV2V1 : LocRefsSeparated (locRefs v2) (locRefs v1) :=
    locRefsSeparated_right_of_nodup_append hlinV12
  have hsepV1Body : LocRefsSeparated (locRefs v1) (locRefs body) :=
    locRefsSeparated_left_of_append hsepV12Body
  have hsepV2Body : LocRefsSeparated (locRefs v2) (locRefs body) :=
    locRefsSeparated_right_of_append hsepV12Body
  have hsepBodyV1 : LocRefsSeparated (locRefs body) (locRefs v1) :=
    locRefsSeparated_rhs_left_of_append hsepBodyV12
  have hsepBodyV2 : LocRefsSeparated (locRefs body) (locRefs v2) :=
    locRefsSeparated_rhs_right_of_append hsepBodyV12
  have hsepV1Locs : LocRefsSeparated (locRefs v1) locs := by
    intro ell hv1loc hlocs
    exact hsep ell hlocs (by simp [locRefs, hv1loc])
  have hsepV2Locs : LocRefsSeparated (locRefs v2) locs := by
    intro ell hv2loc hlocs
    exact hsep ell hlocs (by simp [locRefs, hv2loc])
  have hsepBodyLocs : LocRefsSeparated (locRefs body) locs := by
    intro ell hbodyloc hlocs
    exact hsep ell hlocs (by simp [locRefs, hbodyloc])
  have hsepLocsV1 : LocRefsSeparated locs (locRefs v1) := by
    intro ell hlocs hv1loc
    exact hsep ell hlocs (by simp [locRefs, hv1loc])
  have hsepLocsV2 : LocRefsSeparated locs (locRefs v2) := by
    intro ell hlocs hv2loc
    exact hsep ell hlocs (by simp [locRefs, hv2loc])
  have hsepLocsBody : LocRefsSeparated locs (locRefs body) := by
    intro ell hlocs hbodyloc
    exact hsep ell hlocs (by simp [locRefs, hbodyloc])
  have hBodyLex : LexicallyScoped [(x, some t1), (y, some t2)] body := by
    simpa using
      (lexical_letpair_body (Gamma := []) (tx := t1) (ty := t2) (lexical_nil hScope))
  rcases transport_typing_lexical hBody hBodyLex with
    ⟨bodyDB, hEraseBody, hBodyDB0⟩
  rcases transport_typing_lexical hV1Nil (lexical_nil hScopeV1) with
    ⟨v1DB, hEraseV1, hV1DB⟩
  rcases transport_typing_lexical hV2Nil (lexical_nil hScopeV2) with
    ⟨v2DB, hEraseV2, hV2DB⟩
  have hV1UnderY : HasType [] Sigma [(y, some t2)] v1 t1 [] [(y, some t2)] := by
    simpa using hasType_prefix_weaken hV1Nil [(y, some t2)]
  rcases transport_typing_lexical hV1UnderY (lexical_singleton hyV1 hScopeV1) with
    ⟨v1DBY, hEraseV1Y, hV1DBY0⟩
  have hEraseV1Y' :
      eraseTerm (ctxEnv [(y, some t2)]) v1 = some v1DB := by
    simpa [ctxEnv, linearCtxDom] using eraseTerm_suffix hEraseV1 [y]
  have hv1Eq : v1DBY = v1DB := by
    have : some v1DBY = some v1DB := by simpa [hEraseV1Y] using hEraseV1Y'
    exact Option.some.inj this
  have hV1DBY : HasTypeDB [] Sigma [some t2] v1DB t1 [] [some t2] := by
    cases hv1Eq
    simpa [eraseCtx] using hV1DBY0
  have hBodyShape :
      HasTypeDB [] Sigma (LinearCtxDB.insertAt 1 (some t1) [some t2]) bodyDB t epsBody
        (LinearCtxDB.insertAt 1 slotX [slotY]) := by
    simpa using hBodyDB0
  have hlinBodyDB : RuntimeLinearDB bodyDB := by
    exact (eraseTerm_runtimeLinear_iff hEraseBody).1 hlinBody
  have hlinV1DB : RuntimeLinearDB v1DB := by
    exact (eraseTerm_runtimeLinear_iff hEraseV1).1 hlinV1
  have hlinV2DB : RuntimeLinearDB v2DB := by
    exact (eraseTerm_runtimeLinear_iff hEraseV2).1 hlinV2
  have hsepV1BodyDB :
      LocRefsSeparated (locRefsDB v1DB) (locRefsDB bodyDB) := by
    simpa [eraseTerm_locRefs hEraseV1, eraseTerm_locRefs hEraseBody] using hsepV1Body
  have hsepBodyV1DB :
      LocRefsSeparated (locRefsDB bodyDB) (locRefsDB v1DB) := by
    simpa [eraseTerm_locRefs hEraseV1, eraseTerm_locRefs hEraseBody] using hsepBodyV1
  have hsepV2BodyDB :
      LocRefsSeparated (locRefsDB v2DB) (locRefsDB bodyDB) := by
    simpa [eraseTerm_locRefs hEraseV2, eraseTerm_locRefs hEraseBody] using hsepV2Body
  have hsepBodyV2DB :
      LocRefsSeparated (locRefsDB bodyDB) (locRefsDB v2DB) := by
    simpa [eraseTerm_locRefs hEraseV2, eraseTerm_locRefs hEraseBody] using hsepBodyV2
  have hsepV1V2DB :
      LocRefsSeparated (locRefsDB v1DB) (locRefsDB v2DB) := by
    simpa [eraseTerm_locRefs hEraseV1, eraseTerm_locRefs hEraseV2] using hsepV1V2
  have hsepV2V1DB :
      LocRefsSeparated (locRefsDB v2DB) (locRefsDB v1DB) := by
    simpa [eraseTerm_locRefs hEraseV1, eraseTerm_locRefs hEraseV2] using hsepV2V1
  have hsepV1LocsDB :
      LocRefsSeparated (locRefsDB v1DB) locs := by
    simpa [eraseTerm_locRefs hEraseV1] using hsepV1Locs
  have hsepV2LocsDB :
      LocRefsSeparated (locRefsDB v2DB) locs := by
    simpa [eraseTerm_locRefs hEraseV2] using hsepV2Locs
  have hsepBodyLocsDB :
      LocRefsSeparated (locRefsDB bodyDB) locs := by
    simpa [eraseTerm_locRefs hEraseBody] using hsepBodyLocs
  have hsepLocsV1DB :
      LocRefsSeparated locs (locRefsDB v1DB) := by
    simpa [eraseTerm_locRefs hEraseV1] using hsepLocsV1
  have hsepLocsV2DB :
      LocRefsSeparated locs (locRefsDB v2DB) := by
    simpa [eraseTerm_locRefs hEraseV2] using hsepLocsV2
  have hsepLocsBodyDB :
      LocRefsSeparated locs (locRefsDB bodyDB) := by
    simpa [eraseTerm_locRefs hEraseBody] using hsepLocsBody
  have hsepV1RhsDB :
      LocRefsSeparated (locRefsDB v1DB) (locRefsDB v2DB ++ locs) :=
    locRefsSeparated_lhs_append hsepV1V2DB hsepV1LocsDB
  have hsepRhsV1DB :
      LocRefsSeparated (locRefsDB v2DB ++ locs) (locRefsDB v1DB) :=
    locRefsSeparated_append hsepV2V1DB hsepLocsV1DB
  have hEraseAfterX :
      eraseTerm [y] (subst body v1 x) = some (substDBAux 1 v1DB bodyDB) := by
    simpa using
      eraseTerm_subst_split (ρin := [y]) (ρout := []) (v := v1) (x := x)
        (by simp [hxy]) hEraseV1 hEraseBody hxBody
  have hFirstTyping :
      HasTypeDB [] Sigma [some t2] (substDBAux 1 v1DB bodyDB) t epsBody [slotY] := by
    cases hslotX : slotX with
    | none =>
        exact subst_preserves_typing_db_gen
          (h := by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape)
          (v := v1DB) (t_v := t1) 1 [some t2] [slotY]
          (some t1) none
          (by simp) (by simp) rfl rfl
          (Or.inr (Or.inl ⟨rfl, rfl⟩)) hV1DBY
    | some tKeep =>
        have htKeep : tKeep = t1 := by
          have hLiveIn :
              (LinearCtxDB.insertAt 1 (some t1) [some t2])[1]? = some (some t1) := by
            simp
          have hMono := hasTypeDB_live_slot_monotone
            (by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape) 1 t1 hLiveIn
          rcases hMono with hLive | hDead
          · simpa [hslotX] using hLive
          · simp [hslotX] at hDead
        cases htKeep
        exact subst_preserves_typing_db_gen
          (h := by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape)
          (v := v1DB) (t_v := t1) 1 [some t2] [slotY]
          (some t1) (some t1)
          (by simp) (by simp) rfl rfl
          (Or.inl ⟨rfl, rfl⟩) hV1DBY
  have hFirstRes :
      RuntimeLinearDB (substDBAux 1 v1DB bodyDB) ∧
        LocRefsSeparated (locRefsDB (substDBAux 1 v1DB bodyDB)) (locRefsDB v2DB ++ locs) ∧
        LocRefsSeparated (locRefsDB v2DB ++ locs) (locRefsDB (substDBAux 1 v1DB bodyDB)) := by
    cases hslotX : slotX with
    | none =>
        exact runtimeLinearDB_subst_dead_gen
          (h := by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape)
          v1DB t1 1 [some t2] [slotY] (locRefsDB v2DB ++ locs)
          (by simp) (by simp) rfl rfl hV1DBY
          hlinV1DB hsepV1BodyDB hsepBodyV1DB hsepV1RhsDB hsepRhsV1DB
          hlinBodyDB
          (locRefsSeparated_lhs_append hsepBodyV2DB hsepBodyLocsDB)
          (locRefsSeparated_append hsepV2BodyDB hsepLocsBodyDB)
    | some tKeep =>
        have htKeep : tKeep = t1 := by
          have hLiveIn :
              (LinearCtxDB.insertAt 1 (some t1) [some t2])[1]? = some (some t1) := by
            simp
          have hMono := hasTypeDB_live_slot_monotone
            (by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape) 1 t1 hLiveIn
          rcases hMono with hLive | hDead
          · simpa [hslotX] using hLive
          · simp [hslotX] at hDead
        cases htKeep
        exact runtimeLinearDB_subst_live_gen_separated
          (Γ1 := [some t2]) (Γ2 := [slotY]) (v := v1DB) (rhsRefs := locRefsDB v2DB ++ locs)
          1 (by simp) (by simp) (by simpa [hslotX] using hBodyShape)
          hlinBodyDB
          (locRefsSeparated_lhs_append hsepBodyV2DB hsepBodyLocsDB)
          (locRefsSeparated_append hsepV2BodyDB hsepLocsBodyDB)
  have hyAfterX : y ∉ boundVars (subst body v1 x) := by
    exact subst_notBound body v1 x y hyBody hyV1
  have hEraseFinal :
      eraseTerm [] (subst (subst body v1 x) v2 y) =
        some (substDBAux 0 v2DB (substDBAux 1 v1DB bodyDB)) := by
    simpa using
      eraseTerm_subst_head (ρ := []) (v := v2) (x := y)
        hEraseV2 hEraseAfterX hyAfterX
  have hV2LocsDB : locRefsDB v2DB = locRefs v2 := by
    simpa using (eraseTerm_locRefs hEraseV2).symm
  have hSecondRes :
      RuntimeLinearDB (substDBAux 0 v2DB (substDBAux 1 v1DB bodyDB)) ∧
        LocRefsSeparated (locRefsDB (substDBAux 0 v2DB (substDBAux 1 v1DB bodyDB))) locs ∧
        LocRefsSeparated locs (locRefsDB (substDBAux 0 v2DB (substDBAux 1 v1DB bodyDB))) := by
    cases hslotY : slotY with
    | none =>
        exact runtimeLinearDB_subst_dead_separated
          (Γ := []) (rhsRefs := locs)
          0 (by simp) (by simpa [hslotY] using hFirstTyping) hV2DB
          hlinV2DB
          (locRefsSeparated_left_of_append hFirstRes.2.2)
          (locRefsSeparated_rhs_left_of_append (xs := locRefsDB (substDBAux 1 v1DB bodyDB))
            (ys := locRefsDB v2DB) (zs := locs) hFirstRes.2.1)
          hsepV2LocsDB hsepLocsV2DB
          hFirstRes.1
          (locRefsSeparated_rhs_right_of_append (xs := locRefsDB (substDBAux 1 v1DB bodyDB))
            (ys := locRefsDB v2DB) (zs := locs) hFirstRes.2.1)
          (locRefsSeparated_right_of_append hFirstRes.2.2)
    | some tKeep =>
        have htKeep : tKeep = t2 := by
          have hLiveIn : ([some t2])[0]? = some (some t2) := by simp
          have hMono := hasTypeDB_live_slot_monotone
            (by simpa [hslotY] using hFirstTyping) 0 t2 hLiveIn
          rcases hMono with hLive | hDead
          · simpa [hslotY] using hLive
          · simp [hslotY] at hDead
        cases htKeep
        exact runtimeLinearDB_subst_live_gen_separated
          (Γ1 := []) (Γ2 := []) (v := v2DB) (rhsRefs := locs)
          0 (by simp) (by simp) (by simpa [hslotY] using hFirstTyping)
          hFirstRes.1
          (locRefsSeparated_rhs_right_of_append (xs := locRefsDB (substDBAux 1 v1DB bodyDB))
            (ys := locRefsDB v2DB) (zs := locs) hFirstRes.2.1)
          (locRefsSeparated_right_of_append hFirstRes.2.2)
  have hlocsFinal :
      locRefs (subst (subst body v1 x) v2 y) =
        locRefsDB (substDBAux 0 v2DB (substDBAux 1 v1DB bodyDB)) := by
    simpa using eraseTerm_locRefs hEraseFinal
  refine ⟨?_, ?_, ?_⟩
  · exact (eraseTerm_runtimeLinear_iff hEraseFinal).2 hSecondRes.1
  · simpa [hlocsFinal] using hSecondRes.2.1
  · simpa [hlocsFinal] using hSecondRes.2.2

private theorem mem_locRefsClauses_of_mem_clause
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term} {ell : Loc}
    (hmem : (op, x, k, hb) ∈ clauses)
    (hell : ell ∈ locRefs hb) :
    ell ∈ locRefsClauses clauses := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases cl with ⟨op0, x0, k0, hb0⟩
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [locRefsClauses, hell]
      · simp [locRefsClauses, ih htl]

private theorem runtimeLinear_handleOpDirect_via_db
    {Sigma : StoreTyp}
    {op : EffectLabel} {v : Term} {epsH : EffectRow}
    {clauses : List (EffectLabel × String × String × Term)}
    {x k : String} {hb : Term} {tRet t : Typ} {eps : EffectRow} {locs : List Loc}
    (hTyp : HasType [] Sigma [] (Term.handle epsH (Term.perform op v) clauses) t eps [])
    (hv : IsValue v)
    (hsig : ∃ tArg, OpSigMatch op tArg tRet)
    (hmem : (op, x, k, hb) ∈ clauses)
    (hScope : WellScoped (Term.handle epsH (Term.perform op v) clauses))
    (hLinear : RuntimeLinear (Term.handle epsH (Term.perform op v) clauses))
    (hsep : LocRefsSeparated locs (locRefs (Term.handle epsH (Term.perform op v) clauses))) :
    RuntimeLinear (subst (subst hb v x) (directIdCont epsH op v clauses tRet) k) ∧
      LocRefsSeparated
        (locRefs (subst (subst hb v x) (directIdCont epsH op v clauses tRet) k)) locs ∧
      LocRefsSeparated locs
        (locRefs (subst (subst hb v x) (directIdCont epsH op v clauses tRet) k)) := by
  rcases HasType.handle_inv_strong_bridge hTyp with
    ⟨GammaBody, epsB, hPerform, _hOpsIn, _hClsIn, _hCover, hClauses, hSub⟩
  have hGammaBody : GammaBody = [] := has_type_closed_output_of_closed_input hPerform
  subst hGammaBody
  rcases HasType.perform_inv_bridge hPerform with
    ⟨tArgV, _epsV, hV, hPerfSig, _hSubPerf⟩
  rcases ClausesTyped.mem_inv hClauses hmem with
    ⟨tArgClause, tRetClause, slotX, slotK, hClauseSig, hBody⟩
  obtain ⟨tArgStep, hStepSig⟩ := hsig
  have hArgEq : tArgClause = tArgV := OpSigMatch.arg_unique hClauseSig hPerfSig
  have hRetEq : tRetClause = tRet := OpSigMatch.ret_unique hClauseSig hStepSig
  have hRetGoal : t = tRet := OpSigMatch.ret_unique hPerfSig hStepSig
  subst tArgClause
  subst tRetClause
  subst t
  let tK : Typ := Typ.arrow tRet tRet (EffectRow.removeOps epsB epsH)
  let y : String := directIdContName epsH op v clauses
  let idCont : Term := directIdCont epsH op v clauses tRet
  have hScopePerform : WellScoped (Term.perform op v) := wellScoped_handle_body hScope
  have hScopeV : WellScoped v := by
    simpa [WellScoped, boundVars] using hScopePerform
  have hClosedV : Closed v := has_type_closed_term_of_closed_input hV
  have hVNil : HasType [] Sigma [] v tArgV [] [] :=
    HasType.value_eff_polymorphic_bridge hV hv []
  rcases wellScoped_handle_clause hScope hmem with
    ⟨hxk, hxHb, hkHb, hHbScope⟩
  have hkV : k ∉ boundVars v := by
    have hkBody := (wellScoped_handle_clause_not_in_body hScope hmem).2
    simpa [boundVars] using hkBody
  have hLocs : (locRefs v ++ locRefsClauses clauses).Nodup := by
    simpa [RuntimeLinear, locRefs] using hLinear
  rcases List.nodup_append.mp hLocs with ⟨hlinV, hlinClauses, _⟩
  have hsepVClauses : LocRefsSeparated (locRefs v) (locRefsClauses clauses) :=
    locRefsSeparated_left_of_nodup_append hLocs
  have hsepClausesV : LocRefsSeparated (locRefsClauses clauses) (locRefs v) :=
    locRefsSeparated_right_of_nodup_append hLocs
  have hClausesV : (locRefsClauses clauses ++ locRefs v).Nodup := by
    refine List.nodup_append.mpr ?_
    refine ⟨hlinClauses, hlinV, ?_⟩
    intro ell hcls ell' hv' heq
    subst ell'
    exact hsepClausesV ell hcls hv'
  rcases runtimeLinear_clause_mem_separated (rhsRefs := locRefs v) hmem hClausesV with
    ⟨hlinHb, hsepHbV, hsepVHb⟩
  have hsepHbLocs : LocRefsSeparated (locRefs hb) locs := by
    intro ell hhb hlocs
    exact hsep ell hlocs (by
      simp [locRefs]
      exact Or.inr (mem_locRefsClauses_of_mem_clause hmem hhb))
  have hsepLocsHb : LocRefsSeparated locs (locRefs hb) := by
    intro ell hlocs hhb
    exact hsep ell hlocs (by
      simp [locRefs]
      exact Or.inr (mem_locRefsClauses_of_mem_clause hmem hhb))
  have hsepVLocs : LocRefsSeparated (locRefs v) locs := by
    intro ell hvloc hlocs
    exact hsep ell hlocs (by simp [locRefs, hvloc])
  have hsepLocsV : LocRefsSeparated locs (locRefs v) := by
    intro ell hlocs hvloc
    exact hsep ell hlocs (by simp [locRefs, hvloc])
  have hBodyLex : LexicallyScoped [(x, some tArgV), (k, some tK)] hb :=
    lexical_handle_clause (lexical_nil hScope) hmem
  rcases transport_typing_lexical hBody hBodyLex with
    ⟨hbDB, hEraseHb, hBodyDB⟩
  rcases transport_typing_lexical hVNil (lexical_nil hScopeV) with
    ⟨vDB, hEraseV, hVDB⟩
  have hVUnderK : HasType [] Sigma [(k, some tK)] v tArgV [] [(k, some tK)] := by
    simpa using hasType_prefix_weaken hVNil [(k, some tK)]
  rcases transport_typing_lexical hVUnderK (lexical_singleton hkV hScopeV) with
    ⟨vDBK, hEraseVK, hVDBK0⟩
  have hEraseVK' :
      eraseTerm (ctxEnv [(k, some tK)]) v = some vDB := by
    simpa [ctxEnv, linearCtxDom] using eraseTerm_suffix hEraseV [k]
  have hvEq : vDBK = vDB := by
    have : some vDBK = some vDB := by simpa [hEraseVK] using hEraseVK'
    exact Option.some.inj this
  have hVDBK : HasTypeDB [] Sigma [some tK] vDB tArgV [] [some tK] := by
    cases hvEq
    simpa [eraseCtx] using hVDBK0
  have hBodyShape :
      HasTypeDB [] Sigma (LinearCtxDB.insertAt 1 (some tArgV) [some tK]) hbDB tRet
        (EffectRow.removeOps epsB epsH) (LinearCtxDB.insertAt 1 slotX [slotK]) := by
    simpa [tK] using hBodyDB
  have hlinHbDB : RuntimeLinearDB hbDB := by
    exact (eraseTerm_runtimeLinear_iff hEraseHb).1 hlinHb
  have hlinVDB : RuntimeLinearDB vDB := by
    exact (eraseTerm_runtimeLinear_iff hEraseV).1 hlinV
  have hsepVHbDB :
      LocRefsSeparated (locRefsDB vDB) (locRefsDB hbDB) := by
    simpa [eraseTerm_locRefs hEraseV, eraseTerm_locRefs hEraseHb] using hsepVHb
  have hsepHbVDB :
      LocRefsSeparated (locRefsDB hbDB) (locRefsDB vDB) := by
    simpa [eraseTerm_locRefs hEraseV, eraseTerm_locRefs hEraseHb] using hsepHbV
  have hsepVLocsDB :
      LocRefsSeparated (locRefsDB vDB) locs := by
    simpa [eraseTerm_locRefs hEraseV] using hsepVLocs
  have hsepLocsVDB :
      LocRefsSeparated locs (locRefsDB vDB) := by
    simpa [eraseTerm_locRefs hEraseV] using hsepLocsV
  have hsepHbLocsDB :
      LocRefsSeparated (locRefsDB hbDB) locs := by
    simpa [eraseTerm_locRefs hEraseHb] using hsepHbLocs
  have hsepLocsHbDB :
      LocRefsSeparated locs (locRefsDB hbDB) := by
    simpa [eraseTerm_locRefs hEraseHb] using hsepLocsHb
  have hEraseAfterX :
      eraseTerm [k] (subst hb v x) = some (substDBAux 1 vDB hbDB) := by
    simpa [tK] using
      eraseTerm_subst_tail (ρ := [k]) (v := v) (x := x)
        (by simp [hxk]) hEraseV hEraseHb hxHb
  have hFirstTyping :
      HasTypeDB [] Sigma [some tK] (substDBAux 1 vDB hbDB) tRet
        (EffectRow.removeOps epsB epsH) [slotK] := by
    cases hslotX : slotX with
    | none =>
        exact subst_preserves_typing_db_gen
          (h := by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape)
          (v := vDB) (t_v := tArgV) 1 [some tK] [slotK]
          (some tArgV) none
          (by simp) (by simp) rfl rfl
          (Or.inr (Or.inl ⟨rfl, rfl⟩)) hVDBK
    | some tKeep =>
        have htKeep : tKeep = tArgV := by
          have hLiveIn :
              (LinearCtxDB.insertAt 1 (some tArgV) [some tK])[1]? = some (some tArgV) := by
            simp
          have hMono := hasTypeDB_live_slot_monotone
            (by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape) 1 tArgV hLiveIn
          rcases hMono with hLive | hDead
          · simpa [hslotX] using hLive
          · simp [hslotX] at hDead
        cases htKeep
        exact subst_preserves_typing_db_gen
          (h := by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape)
          (v := vDB) (t_v := tArgV) 1 [some tK] [slotK]
          (some tArgV) (some tArgV)
          (by simp) (by simp) rfl rfl
          (Or.inl ⟨rfl, rfl⟩) hVDBK
  have hFirstRes :
      RuntimeLinearDB (substDBAux 1 vDB hbDB) ∧
        LocRefsSeparated (locRefsDB (substDBAux 1 vDB hbDB)) locs ∧
        LocRefsSeparated locs (locRefsDB (substDBAux 1 vDB hbDB)) := by
    cases hslotX : slotX with
    | none =>
        exact runtimeLinearDB_subst_dead_gen
          (h := by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape)
          vDB tArgV 1 [some tK] [slotK] locs
          (by simp) (by simp) rfl rfl hVDBK
          hlinVDB hsepVHbDB hsepHbVDB hsepVLocsDB hsepLocsVDB
          hlinHbDB hsepHbLocsDB hsepLocsHbDB
    | some tKeep =>
        have htKeep : tKeep = tArgV := by
          have hLiveIn :
              (LinearCtxDB.insertAt 1 (some tArgV) [some tK])[1]? = some (some tArgV) := by
            simp
          have hMono := hasTypeDB_live_slot_monotone
            (by simpa [LinearCtxDB.insertAt, hslotX] using hBodyShape) 1 tArgV hLiveIn
          rcases hMono with hLive | hDead
          · simpa [hslotX] using hLive
          · simp [hslotX] at hDead
        cases htKeep
        exact runtimeLinearDB_subst_live_gen_separated
          (Γ1 := [some tK]) (Γ2 := [slotK]) (v := vDB) (rhsRefs := locs)
          1 (by simp) (by simp) (by simpa [hslotX] using hBodyShape)
          hlinHbDB hsepHbLocsDB hsepLocsHbDB
  have hIdBody0 :
      HasType [] Sigma [(y, some tRet)] (Term.var y) tRet [] [(y, none)] := by
    simpa [y] using (HasType.var [] Sigma [] [] y tRet)
  have hIdBody :
      HasType [] Sigma [(y, some tRet)] (Term.var y) tRet
        (EffectRow.removeOps epsB epsH) [(y, none)] := by
    exact HasType.subEff [] Sigma
      [(y, some tRet)] [(y, none)]
      (Term.var y) tRet [] (EffectRow.removeOps epsB epsH)
      hIdBody0 (by intro op hop; cases hop)
  have hIdAbs :
      HasType [] Sigma []
        idCont
        tK [] [] := by
    exact HasType.abs [] Sigma [] []
      y tRet tRet (EffectRow.removeOps epsB epsH)
      (Term.var y) none hIdBody
  have hIdScope : WellScoped idCont := by
    simpa [idCont] using
      (directIdCont_wellScoped (epsH := epsH) (op := op) (v := v) (clauses := clauses) (tRet := tRet))
  rcases transport_typing_lexical hIdAbs (lexical_nil hIdScope) with
    ⟨idDB, hEraseId, hIdDB⟩
  have hIdLocs : locRefs idCont = [] := by
    simp [idCont, directIdCont, directIdContName, locRefs]
  have hIdLocsDB : locRefsDB idDB = [] := by
    simpa [hIdLocs] using (eraseTerm_locRefs hEraseId).symm
  have hIdLinear : RuntimeLinear idCont := by
    simp [RuntimeLinear, hIdLocs]
  have hIdLinearDB : RuntimeLinearDB idDB := by
    exact (eraseTerm_runtimeLinear_iff hEraseId).1 hIdLinear
  have hkAfterX : k ∉ boundVars (subst hb v x) := by
    exact subst_notBound hb v x k hkHb hkV
  have hEraseFinal :
      eraseTerm [] (subst (subst hb v x) idCont k) =
        some (substDBAux 0 idDB (substDBAux 1 vDB hbDB)) := by
    simpa using
      eraseTerm_subst_head (ρ := []) (v := idCont) (x := k)
        hEraseId hEraseAfterX hkAfterX
  have hFinalRes :
      RuntimeLinearDB (substDBAux 0 idDB (substDBAux 1 vDB hbDB)) ∧
        LocRefsSeparated (locRefsDB (substDBAux 0 idDB (substDBAux 1 vDB hbDB))) locs ∧
        LocRefsSeparated locs (locRefsDB (substDBAux 0 idDB (substDBAux 1 vDB hbDB))) := by
    cases hslotK : slotK with
    | none =>
        exact runtimeLinearDB_subst_dead_separated
          (Γ := []) (rhsRefs := locs)
          0 (by simp) (by simpa [hslotK] using hFirstTyping) hIdDB
          hIdLinearDB
          (by simpa [hIdLocsDB, LocRefsSeparated])
          (by simpa [hIdLocsDB, LocRefsSeparated])
          (by simpa [hIdLocsDB, LocRefsSeparated])
          (by simpa [hIdLocsDB, LocRefsSeparated])
          hFirstRes.1 hFirstRes.2.1 hFirstRes.2.2
    | some tKeep =>
        have htKeep : tKeep = tK := by
          have hLiveIn : ([some tK])[0]? = some (some tK) := by simp
          have hMono := hasTypeDB_live_slot_monotone
            (by simpa [hslotK] using hFirstTyping) 0 tK hLiveIn
          rcases hMono with hLive | hDead
          · simpa [hslotK] using hLive
          · simp [hslotK] at hDead
        subst htKeep
        exact runtimeLinearDB_subst_live_gen_separated
          (Γ1 := []) (Γ2 := []) (v := idDB) (rhsRefs := locs)
          0 (by simp) (by simp) (by simpa [hslotK] using hFirstTyping)
          hFirstRes.1 hFirstRes.2.1 hFirstRes.2.2
  have hlocsFinal :
      locRefs (subst (subst hb v x) idCont k) =
        locRefsDB (substDBAux 0 idDB (substDBAux 1 vDB hbDB)) := by
    simpa using eraseTerm_locRefs hEraseFinal
  refine ⟨?_, ?_, ?_⟩
  · exact (eraseTerm_runtimeLinear_iff hEraseFinal).2 hFinalRes.1
  · rw [hlocsFinal]
    exact hFinalRes.2.1
  · rw [hlocsFinal]
    exact hFinalRes.2.2

/-- Step-indexed preservation with the runtime-linearity and frame-local
    store-agreement invariants made explicit. `locs` tracks the
    locations mentioned by any outer frame surrounding the current redex;
    the theorem returns store typing agreement on exactly those
    untouched locations. -/
private theorem preservation_aux
    (Sigma : StoreTyp)
    (c1 c2 : Config)
    (h_wf : StoreWf c1.store Sigma)
    (h_step : Step c1 c2) :
    ∀ {locs : List Loc},
      LocRefsSeparated locs (locRefs c1.term) →
      RuntimeLinear c1.term →
      ∀ {t : Typ} {eps : EffectRow},
        HasType [] Sigma [] c1.term t eps [] →
        WellScoped c1.term →
        ∃ Sigma',
          HasType [] Sigma' [] c2.term t eps [] ∧
          StoreWf c2.store Sigma' ∧
          StoreTypOn locs Sigma Sigma' := by
  induction h_step with
  | beta s x tv body v hv =>
      intro locs _hsep _hlinear t eps h_typ h_scope
      exact ⟨Sigma, preservation_beta_via_db h_typ hv h_scope, h_wf,
        fun _ _ _ hlook => hlook⟩
  | letBind s x v body hv =>
      intro locs _hsep _hlinear t eps h_typ h_scope
      exact ⟨Sigma, preservation_letBind_via_db h_typ hv h_scope, h_wf,
        fun _ _ _ hlook => hlook⟩
  | letpair s x y v1 v2 body hv1 hv2 =>
      intro locs _hsep _hlinear t eps h_typ h_scope
      exact ⟨Sigma, preservation_letpair_via_db h_typ hv1 hv2 h_scope, h_wf,
        fun _ _ _ hlook => hlook⟩
  | fst s v1 v2 hv1 hv2 =>
      intro locs _hsep _hlinear t eps h_typ _h_scope
      let h_pair := HasType.fst_inv h_typ
      obtain ⟨Γmid, eps1, eps2, h1, h2, hsub⟩ := HasType.pair_inv h_pair
      have hmid : Γmid = [] := has_type_closed_output_of_closed_input h1
      subst hmid
      have hsub1 : SubEffRow eps1 eps :=
        SubEffRow.trans (SubEffRow.union_left eps1 eps2) hsub
      exact ⟨Sigma, HasType.subEff [] Sigma [] [] _ t eps1 eps h1 hsub1, h_wf,
        fun _ _ _ hlook => hlook⟩
  | snd s v1 v2 hv1 hv2 =>
      intro locs _hsep _hlinear t eps h_typ _h_scope
      let h_pair := HasType.snd_inv h_typ
      obtain ⟨Γmid, eps1, eps2, h1, h2, hsub⟩ := HasType.pair_inv h_pair
      have hmid : Γmid = [] := has_type_closed_output_of_closed_input h1
      subst hmid
      have hsub2 : SubEffRow eps2 eps := by
        intro op hop
        apply hsub
        by_cases hop1 : op ∈ eps1
        · simp [EffectRow.union, hop1]
        · simp [EffectRow.union, hop, hop1]
      exact ⟨Sigma, HasType.subEff [] Sigma [] [] _ t eps2 eps h2 hsub2, h_wf,
        fun _ _ _ hlook => hlook⟩
  | tconst s v ds ell hell =>
      intro locs _hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ht, _hG⟩ := HasType.const_inv h_typ
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_, ?_, ?_⟩
      · subst ht
        have hlook := storeTypLookup_extend_self Sigma ell (Typ.tensor ds)
        have h_loc : HasType [] (storeTypExtend Sigma ell (Typ.tensor ds))
                              [] (Term.loc ell) (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ell (Typ.tensor ds) hlook
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_loc hsub
      · exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf
      · intro ell' t' hmem hlook
        have hLive : (storeLookup s ell').isSome :=
          StoreWf.lookup_isSome_of_typing h_wf hlook
        have hne : ell' ≠ ell := by
          intro hEq
          exact (storeFreshLoc_ne s ell' hLive) (by rw [hEq, hell])
        exact storeTypLookup_extend_other Sigma ell' ell t' (Typ.tensor ds) hlook hne
  | copy s ell ellNew w hlook hfresh =>
      intro locs _hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ds, htEq, h_e⟩ := HasType.copy_inv h_typ
      obtain ⟨hlookT, _hGE⟩ := HasType.loc_inv h_e
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor ds), ?_, ?_, ?_⟩
      · subst htEq
        have hlookSelf := storeTypLookup_extend_self Sigma ellNew (Typ.tensor ds)
        have hlookEll :
            storeTypLookup (storeTypExtend Sigma ellNew (Typ.tensor ds)) ell
              = some (Typ.tensor ds) := by
          by_cases hne : ell = ellNew
          · rw [hne]; exact hlookSelf
          · have hne' : ¬ (ellNew = ell) := fun he => hne he.symm
            simp only [storeTypLookup, storeTypExtend, List.find?,
                       hne', decide_false, Bool.false_eq_true, ite_false]
            simpa [storeTypLookup] using hlookT
        have h_l1 : HasType [] (storeTypExtend Sigma ellNew (Typ.tensor ds)) []
                      (Term.loc ell) (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ell (Typ.tensor ds) hlookEll
        have h_l2 : HasType [] (storeTypExtend Sigma ellNew (Typ.tensor ds)) []
                      (Term.loc ellNew) (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ellNew (Typ.tensor ds) hlookSelf
        have h_pair : HasType [] (storeTypExtend Sigma ellNew (Typ.tensor ds)) []
                        (Term.pair (Term.loc ell) (Term.loc ellNew))
                        (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) [] [] :=
          HasType.tpair [] _ [] [] []
            _ _ (Typ.tensor ds) (Typ.tensor ds) [] [] h_l1 h_l2
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_pair hsub
      · exact StoreWf.extend_fresh ellNew w (Typ.tensor ds) h_wf
      · intro ell' t' hmem hlookTy
        have hLive : (storeLookup s ell').isSome :=
          StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hne : ell' ≠ ellNew := by
          intro hEq
          exact (storeFreshLoc_ne s ell' hLive) (by rw [hEq, hfresh])
        exact storeTypLookup_extend_other Sigma ell' ellNew t' (Typ.tensor ds) hlookTy hne
  | tadd s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      intro locs hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ds, _Γmid, _eps1, _eps2, htEq, h_e1, h_e2⟩ := HasType.add_inv h_typ
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor ds), ?_, ?_, ?_⟩
      · subst htEq
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove (storeTypRemove Sigma ell1) ell2)
          ellOut (Typ.tensor ds)
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor ds) [] [] :=
          HasType.loc _ _ [] ellOut (Typ.tensor ds) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_wf1 := StoreWf.remove ell1 h_wf
        have h_wf2 := StoreWf.remove ell2 h_wf1
        exact StoreWf.extend_fresh ellOut (tensorOpPlaceholder w1 w2)
          (Typ.tensor ds) h_wf2
      · intro ell t' hmem hlook
        have hne1 : ell ≠ ell1 := by
          intro hEq
          subst hEq
          exact hsep ell hmem (by simp [locRefs])
        have hne2 : ell ≠ ell2 := by
          intro hEq
          subst hEq
          exact hsep ell hmem (by simp [locRefs])
        have hlook1 := storeTypLookup_remove_other Sigma ell1 ell t' hlook hne1
        have hlook2 := storeTypLookup_remove_other (storeTypRemove Sigma ell1) ell2 ell t' hlook1 hne2
        have hLive : (storeLookup s ell).isSome := StoreWf.lookup_isSome_of_typing h_wf hlook
        have hneOut : ell ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell hLive
        exact storeTypLookup_extend_other
          (storeTypRemove (storeTypRemove Sigma ell1) ell2)
          ell ellOut t' (Typ.tensor ds) hlook2 hneOut
  | tmul s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      intro locs hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ds, _Γmid, _eps1, _eps2, htEq, h_e1, h_e2⟩ := HasType.mul_inv h_typ
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor ds), ?_, ?_, ?_⟩
      · subst htEq
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove (storeTypRemove Sigma ell1) ell2)
          ellOut (Typ.tensor ds)
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor ds) [] [] :=
          HasType.loc _ _ [] ellOut (Typ.tensor ds) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_wf1 := StoreWf.remove ell1 h_wf
        have h_wf2 := StoreWf.remove ell2 h_wf1
        exact StoreWf.extend_fresh ellOut (tensorOpPlaceholder w1 w2)
          (Typ.tensor ds) h_wf2
      · intro ell t' hmem hlook
        have hne1 : ell ≠ ell1 := by
          intro hEq
          subst hEq
          exact hsep ell hmem (by simp [locRefs])
        have hne2 : ell ≠ ell2 := by
          intro hEq
          subst hEq
          exact hsep ell hmem (by simp [locRefs])
        have hlook1 := storeTypLookup_remove_other Sigma ell1 ell t' hlook hne1
        have hlook2 := storeTypLookup_remove_other (storeTypRemove Sigma ell1) ell2 ell t' hlook1 hne2
        have hLive : (storeLookup s ell).isSome := StoreWf.lookup_isSome_of_typing h_wf hlook
        have hneOut : ell ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell hLive
        exact storeTypLookup_extend_other
          (storeTypRemove (storeTypRemove Sigma ell1) ell2)
          ell ellOut t' (Typ.tensor ds) hlook2 hneOut
  | tsum s ell ellOut w d hlook hfresh =>
      intro locs hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ds, htEq, _hmem, h_loc_e⟩ := HasType.sum_inv h_typ
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (rem ds d)), ?_, ?_, ?_⟩
      · subst htEq
        have hlookNew : storeTypLookup (storeTypExtend
                          (storeTypRemove Sigma ell) ellOut
                          (Typ.tensor (rem ds d))) ellOut
                      = some (Typ.tensor (rem ds d)) :=
          storeTypLookup_extend_self _ _ _
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor (rem ds d)) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor (rem ds d)) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup s ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := rem w.shape d, data := w.data }
          (Typ.tensor (rem ds d)) h_wf hfresh hne
      · intro ell' t' hmem hlookTy
        have hneIn : ell' ≠ ell := by
          intro hEq
          subst hEq
          exact hsep ell' hmem (by simp [locRefs])
        have hlookRem := storeTypLookup_remove_other Sigma ell ell' t' hlookTy hneIn
        have hLive : (storeLookup s ell').isSome := StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hneOut : ell' ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell' hLive
        exact storeTypLookup_extend_other
          (storeTypRemove Sigma ell) ell' ellOut t' (Typ.tensor (rem ds d))
          hlookRem hneOut
  | texpand s ell ellOut w d hlook hfresh =>
      intro locs hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ds, htEq, _h_loc_e⟩ := HasType.expand_inv h_typ
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (ins ds d)), ?_, ?_, ?_⟩
      · subst htEq
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove Sigma ell) ellOut (Typ.tensor (ins ds d))
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor (ins ds d)) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor (ins ds d)) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup s ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := ins w.shape d, data := w.data }
          (Typ.tensor (ins ds d)) h_wf hfresh hne
      · intro ell' t' hmem hlookTy
        have hneIn : ell' ≠ ell := by
          intro hEq
          subst hEq
          exact hsep ell' hmem (by simp [locRefs])
        have hlookRem := storeTypLookup_remove_other Sigma ell ell' t' hlookTy hneIn
        have hLive : (storeLookup s ell').isSome := StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hneOut : ell' ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell' hLive
        exact storeTypLookup_extend_other
          (storeTypRemove Sigma ell) ell' ellOut t' (Typ.tensor (ins ds d))
          hlookRem hneOut
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      intro locs hsep _hlinear t eps h_typ _h_scope
      obtain ⟨ds, _eps0, htEq, _h_loc_e, _hsub⟩ := HasType.uniformLike_inv h_typ
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor ds), ?_, ?_, ?_⟩
      · subst htEq
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove Sigma ell) ellOut (Typ.tensor ds)
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor ds) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup s ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := w.shape, data := lo }
          (Typ.tensor ds) h_wf hfresh hne
      · intro ell' t' hmem hlookTy
        have hneIn : ell' ≠ ell := by
          intro hEq
          subst hEq
          exact hsep ell' hmem (by simp [locRefs])
        have hlookRem := storeTypLookup_remove_other Sigma ell ell' t' hlookTy hneIn
        have hLive : (storeLookup s ell').isSome := StoreWf.lookup_isSome_of_typing h_wf hlookTy
        have hneOut : ell' ≠ ellOut := by
          rw [hfresh]
          exact storeFreshLoc_ne s ell' hLive
        exact storeTypLookup_extend_other
          (storeTypRemove Sigma ell) ell' ellOut t' (Typ.tensor ds)
          hlookRem hneOut
  | handleRet s epsH v clauses hv =>
      intro locs _hsep _hlinear t eps h_typ _h_scope
      exact ⟨Sigma, handleRet_value_preserves_typing hv h_typ, h_wf,
        fun _ _ _ hlook => hlook⟩
  | handleOpDirect s op v epsH clauses x k hb tRet hv hsig hmem =>
      intro locs _hsep _hlinear t eps h_typ h_scope
      exact ⟨Sigma, preservation_handleOpDirect_via_db h_typ hv hsig hmem h_scope, h_wf,
        fun _ _ _ hlook => hlook⟩
  | handleOpCtx s op v epsH E clauses xVar kVar hb tRet hv hsig hmem hop hE =>
      intro locs _hsep _hlinear t eps h_typ h_scope
      exact ⟨Sigma, by
        rcases HasType.handle_inv_strong_bridge h_typ with
          ⟨GammaBody, epsB, hPlugPerform, hHsubB, hClIn, hClCov, hClauses, hSub⟩
        have hGammaBody : GammaBody = [] := has_type_closed_output_of_closed_input hPlugPerform
        subst hGammaBody
        rcases HasType.plug_inner_closed hPlugPerform with ⟨_tPerf, _epsPerf, hPerform⟩
        rcases HasType.perform_inv_bridge hPerform with
          ⟨tArgV, _epsV, hV, hPerfSig, _hSubPerf⟩
        rcases ClausesTyped.mem_inv hClauses hmem with
          ⟨tArgClause, tRetClause, slotX, slotK, hClauseSig, hBody⟩
        obtain ⟨tArgStep, hStepSig⟩ := hsig
        have hArgEq : tArgClause = tArgV := OpSigMatch.arg_unique hClauseSig hPerfSig
        have hRetEq : tRetClause = tRet := OpSigMatch.ret_unique hClauseSig hStepSig
        subst tArgClause
        subst tRetClause
        have hClosedV : Closed v := has_type_closed_term_of_closed_input hV
        have hVNil : HasType [] Sigma [] v tArgV [] [] := by
          exact HasType.value_eff_polymorphic hV hv []
        rcases wellScoped_handle_clause h_scope hmem with
          ⟨hxk, _hxNotHb, hkHb, _hHbScope⟩
        let y := capturedContName (Term.handle epsH (plug E (Term.perform op v)) clauses)
        have hPlugVar :
            HasType [] Sigma [(y, some tRet)]
              (plug E (Term.var y)) t epsB
              [(y, none)] := by
          simpa [List.append_assoc] using
            (plug_replace_with_prefixed_hole
              (Gamma := []) (Gamma' := []) (outer := [])
              (E := E) (e := Term.perform op v) (e' := Term.var y)
              (t := t) (eps := epsB) (y := y) (ty := tRet)
              hPlugPerform
              (fun {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow}
                   (h_inner : HasType [] Sigma Gamma0 (Term.perform op v) t0 eps0 Gamma0') =>
                perform_to_var_preserves_typing_prefixed
                  (outer := []) (tRet := tRet) (y := y) h_inner hv
                  ⟨tArgStep, hStepSig⟩))
        have hClausesY :
            ClausesTyped [] Sigma [(y, none)] [(y, none)]
              t (EffectRow.removeOps epsB epsH) clauses := by
          simpa [List.append_assoc] using
            clausesTyped_prefix_weaken hClauses [(y, none)]
        have hHandleY :
            HasType [] Sigma [(y, some tRet)]
              (Term.handle epsH (plug E (Term.var y)) clauses)
              t (EffectRow.removeOps epsB epsH)
              [(y, none)] := by
          exact HasType.handle [] Sigma [(y, some tRet)] [(y, none)] [(y, none)]
            (plug E (Term.var y)) clauses t epsH epsB
            hPlugVar hHsubB hClIn hClCov
            hClausesY
        have hK0 :
            HasType [] Sigma []
              (Term.abs y tRet
                (Term.handle epsH (plug E (Term.var y)) clauses))
              (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) [] [] := by
          exact HasType.abs [] Sigma [] []
            y tRet t (EffectRow.removeOps epsB epsH)
            (Term.handle epsH (plug E (Term.var y)) clauses) none hHandleY
        have hKClosed :
            Closed
              (Term.abs y tRet
                (Term.handle epsH (plug E (Term.var y)) clauses)) :=
          has_type_closed_term_of_closed_input hK0
        have hLexBody :
            LexicallyScoped
              ([(xVar, some tArgV)] ++
                [(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))]) hb :=
          lexical_handle_clause
            (tx := tArgV)
            (tk := Typ.arrow tRet t (EffectRow.removeOps epsB epsH))
            (lexical_nil h_scope) hmem
        have hVSupported :
            AdjointTypeSupported tArgV → AdjointSupported v :=
          adjointSupported_of_typed_value hVNil hv
        have hSuffixFreshV :
            ∀ z,
              z ∈ linearCtxDom
                    ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx) →
                z ∉ freeVars v := by
          intro z hz
          have hfreeV : freeVars v = [] := by
            simpa [Closed] using hClosedV
          simp [linearCtxDom, hfreeV] at hz ⊢
        rcases
            subst_preserves_typing_lexical_suffix
              [] Sigma
              []
              ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx)
              [(xVar, slotX), (kVar, slotK)]
              xVar tArgV t (EffectRow.removeOps epsB epsH)
              hb v
              hBody hLexBody hVNil hVSupported hClosedV hSuffixFreshV with
          ⟨GammaAfterXPre, slotAfterX, GammaAfterXPost,
            hAfterXOut, hAfterXPreSub, _hAfterXSlot, _hAfterXPostSub, hAfterX⟩
        have hAfterXPreNil : GammaAfterXPre = [] := by
          cases GammaAfterXPre with
          | nil =>
              rfl
          | cons hd tl =>
              cases hAfterXPreSub
        subst hAfterXPreNil
        have hAfterXPostEq : GammaAfterXPost = [(kVar, slotK)] := by
          have hOutCons :
              [(xVar, slotX), (kVar, slotK)] =
                (xVar, slotAfterX) :: GammaAfterXPost := by
            simpa using hAfterXOut
          have hTail : [(kVar, slotK)] = GammaAfterXPost := by
            simpa using congrArg List.tail hOutCons
          exact hTail.symm
        subst hAfterXPostEq
        have hkNotV : kVar ∉ boundVars v := by
          intro hkV
          have hkPerform : kVar ∈ boundVars (Term.perform op v) := by
            simpa [boundVars] using hkV
          have hkBody :
              kVar ∈ boundVars (plug E (Term.perform op v)) :=
            mem_boundVars_plug hkPerform
          exact (wellScoped_handle_clause_not_in_body h_scope hmem).2 hkBody
        have hAfterKNodup :
            NoDupNames
              ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx) := by
          simp [NoDupNames, linearCtxDom]
        have hAfterKCtxFresh :
            ∀ z,
              z ∈ linearCtxDom
                    ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx) →
                z ∉ boundVars (subst hb v xVar) := by
          refine
            subst_ctx_bound_fresh
              (Gamma := [(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))])
              hb v xVar ?_ ?_
          · intro z hz
            simp [linearCtxDom] at hz
            rcases hz with rfl
            exact hkHb
          · intro z hz
            simp [linearCtxDom] at hz
            rcases hz with rfl
            exact hkNotV
        rcases
            subst_preserves_typing_ctx_fresh
              [] Sigma
              []
              [(kVar, slotK)]
              kVar (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) t
              (EffectRow.removeOps epsB epsH)
              (subst hb v xVar)
              (Term.abs y tRet
                (Term.handle epsH (plug E (Term.var y)) clauses))
              hAfterX hAfterKNodup hAfterKCtxFresh hK0
              (adjointSupported_of_typed_value hK0
                (IsValue.abs y tRet (Term.handle epsH (plug E (Term.var y)) clauses)))
              hKClosed with
          ⟨GammaFinal, hFinal⟩
        have hGammaFinal : GammaFinal = [] := has_type_closed_output_of_closed_input hFinal
        subst hGammaFinal
        simpa [y] using
          (HasType.subEff [] Sigma [] [] _ _ (EffectRow.removeOps epsB epsH) eps hFinal hSub),
        h_wf, fun _ _ _ hlook => hlook⟩
  | handleOpCtxs s op v epsH Es clauses xVar kVar hb tRet hv hsig hmem hop hEs =>
      intro locs _hsep _hlinear t eps h_typ h_scope
      exact ⟨Sigma, by
        rcases HasType.handle_inv_strong_bridge h_typ with
          ⟨GammaBody, epsB, hPlugPerform, hHsubB, hClIn, hClCov, hClauses, hSub⟩
        have hGammaBody : GammaBody = [] := has_type_closed_output_of_closed_input hPlugPerform
        subst hGammaBody
        rcases HasType.multiPlug_inner_closed hPlugPerform with ⟨_tPerf, _epsPerf, hPerform⟩
        rcases HasType.perform_inv_bridge hPerform with
          ⟨tArgV, _epsV, hV, hPerfSig, _hSubPerf⟩
        rcases ClausesTyped.mem_inv hClauses hmem with
          ⟨tArgClause, tRetClause, slotX, slotK, hClauseSig, hBody⟩
        obtain ⟨tArgStep, hStepSig⟩ := hsig
        have hArgEq : tArgClause = tArgV := OpSigMatch.arg_unique hClauseSig hPerfSig
        have hRetEq : tRetClause = tRet := OpSigMatch.ret_unique hClauseSig hStepSig
        subst tArgClause
        subst tRetClause
        have hClosedV : Closed v := has_type_closed_term_of_closed_input hV
        have hVNil : HasType [] Sigma [] v tArgV [] [] := by
          exact HasType.value_eff_polymorphic hV hv []
        rcases wellScoped_handle_clause h_scope hmem with
          ⟨hxk, _hxNotHb, hkHb, _hHbScope⟩
        let y := capturedContName (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses)
        have hPlugVar :
            HasType [] Sigma [(y, some tRet)]
              (multiPlug Es (Term.var y)) t epsB
              [(y, none)] := by
          simpa [List.append_assoc] using
            (multiPlug_replace_with_prefixed_hole
              (Gamma := []) (Gamma' := []) (outer := [])
              (Es := Es) (e := Term.perform op v) (e' := Term.var y)
              (t := t) (eps := epsB) (y := y) (ty := tRet)
              hPlugPerform
              (fun {Gamma0 Gamma0' : LinearCtx} {t0 : Typ} {eps0 : EffectRow}
                   (h_inner : HasType [] Sigma Gamma0 (Term.perform op v) t0 eps0 Gamma0') =>
                perform_to_var_preserves_typing_prefixed
                  (outer := []) (tRet := tRet) (y := y) h_inner hv
                  ⟨tArgStep, hStepSig⟩))
        have hClausesY :
            ClausesTyped [] Sigma [(y, none)] [(y, none)]
              t (EffectRow.removeOps epsB epsH) clauses := by
          simpa [List.append_assoc] using
            clausesTyped_prefix_weaken hClauses [(y, none)]
        have hHandleY :
            HasType [] Sigma [(y, some tRet)]
              (Term.handle epsH (multiPlug Es (Term.var y)) clauses)
              t (EffectRow.removeOps epsB epsH)
              [(y, none)] := by
          exact HasType.handle [] Sigma [(y, some tRet)] [(y, none)] [(y, none)]
            (multiPlug Es (Term.var y)) clauses t epsH epsB
            hPlugVar hHsubB hClIn hClCov
            hClausesY
        have hK0 :
            HasType [] Sigma []
              (Term.abs y tRet
                (Term.handle epsH (multiPlug Es (Term.var y)) clauses))
              (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) [] [] := by
          exact HasType.abs [] Sigma [] []
            y tRet t (EffectRow.removeOps epsB epsH)
            (Term.handle epsH (multiPlug Es (Term.var y)) clauses) none hHandleY
        have hKClosed :
            Closed
              (Term.abs y tRet
                (Term.handle epsH (multiPlug Es (Term.var y)) clauses)) :=
          has_type_closed_term_of_closed_input hK0
        have hLexBody :
            LexicallyScoped
              ([(xVar, some tArgV)] ++
                [(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))]) hb :=
          lexical_handle_clause
            (tx := tArgV)
            (tk := Typ.arrow tRet t (EffectRow.removeOps epsB epsH))
            (lexical_nil h_scope) hmem
        have hVSupported :
            AdjointTypeSupported tArgV → AdjointSupported v :=
          adjointSupported_of_typed_value hVNil hv
        have hSuffixFreshV :
            ∀ z,
              z ∈ linearCtxDom
                    ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx) →
                z ∉ freeVars v := by
          intro z hz
          have hfreeV : freeVars v = [] := by
            simpa [Closed] using hClosedV
          simp [linearCtxDom, hfreeV] at hz ⊢
        rcases
            subst_preserves_typing_lexical_suffix
              [] Sigma
              []
              ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx)
              [(xVar, slotX), (kVar, slotK)]
              xVar tArgV t (EffectRow.removeOps epsB epsH)
              hb v
              hBody hLexBody hVNil hVSupported hClosedV hSuffixFreshV with
          ⟨GammaAfterXPre, slotAfterX, GammaAfterXPost,
            hAfterXOut, hAfterXPreSub, _hAfterXSlot, _hAfterXPostSub, hAfterX⟩
        have hAfterXPreNil : GammaAfterXPre = [] := by
          cases GammaAfterXPre with
          | nil =>
              rfl
          | cons hd tl =>
              cases hAfterXPreSub
        subst hAfterXPreNil
        have hAfterXPostEq : GammaAfterXPost = [(kVar, slotK)] := by
          have hOutCons :
              [(xVar, slotX), (kVar, slotK)] =
                (xVar, slotAfterX) :: GammaAfterXPost := by
            simpa using hAfterXOut
          have hTail : [(kVar, slotK)] = GammaAfterXPost := by
            simpa using congrArg List.tail hOutCons
          exact hTail.symm
        subst hAfterXPostEq
        have hkNotV : kVar ∉ boundVars v := by
          intro hkV
          have hkPerform : kVar ∈ boundVars (Term.perform op v) := by
            simpa [boundVars] using hkV
          have hkBody :
              kVar ∈ boundVars (multiPlug Es (Term.perform op v)) :=
            mem_boundVars_multiPlug hkPerform
          exact (wellScoped_handle_clause_not_in_body h_scope hmem).2 hkBody
        have hAfterKNodup :
            NoDupNames
              ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx) := by
          simp [NoDupNames, linearCtxDom]
        have hAfterKCtxFresh :
            ∀ z,
              z ∈ linearCtxDom
                    ([(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))] : LinearCtx) →
                z ∉ boundVars (subst hb v xVar) := by
          refine
            subst_ctx_bound_fresh
              (Gamma := [(kVar, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))])
              hb v xVar ?_ ?_
          · intro z hz
            simp [linearCtxDom] at hz
            rcases hz with rfl
            exact hkHb
          · intro z hz
            simp [linearCtxDom] at hz
            rcases hz with rfl
            exact hkNotV
        rcases
            subst_preserves_typing_ctx_fresh
              [] Sigma
              []
              [(kVar, slotK)]
              kVar (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) t
              (EffectRow.removeOps epsB epsH)
              (subst hb v xVar)
              (Term.abs y tRet
                (Term.handle epsH (multiPlug Es (Term.var y)) clauses))
              hAfterX hAfterKNodup hAfterKCtxFresh hK0
              (adjointSupported_of_typed_value hK0
                (IsValue.abs y tRet (Term.handle epsH (multiPlug Es (Term.var y)) clauses)))
              hKClosed with
          ⟨GammaFinal, hFinal⟩
        have hGammaFinal : GammaFinal = [] := has_type_closed_output_of_closed_input hFinal
        subst hGammaFinal
        simpa [y] using
          (HasType.subEff [] Sigma [] [] _ _ (EffectRow.removeOps epsB epsH) eps hFinal hSub),
        h_wf, fun _ _ _ hlook => hlook⟩
  | tgrad s x tv tOut body =>
      intro locs _hsep _hlinear t eps h_typ _h_scope
      rcases HasType.grad_inv h_typ with
        ⟨_hGamma, ds, dsOut, epsBody, slot, hTv, hTOut, hT, hDiffCompat, hSupp,
          hCtxSupp, hBody⟩
      subst tv
      subst tOut
      subst t
      let gs := gradSeedName x body
      let tmp := gradResultName x body
      let n := gradAdjointCounter x gs body
      let epsAdj : EffectRow := EffectRow.union epsBody [EffectLabel.accum]
      let epsHandle : EffectRow := EffectRow.removeOps epsAdj [EffectLabel.accum]
      let clauseBody := Term.app (Term.var "k") (Term.var "p")
      let clauses : List (EffectLabel × String × String × Term) :=
        [(EffectLabel.accum, "p", "k", clauseBody)]
      have hFullLen : maxStringLength [x, gs] < n := by
        have hmaxle : maxStringLength [x, gs] ≤
            maxStringLength (gs :: x :: (freeVars body ++ boundVars body)) := by
          simp [maxStringLength]
          exact Nat.max_le_of_le_of_le
            (Nat.le_trans
              (Nat.le_max_left x.toList.length (maxStringLength (freeVars body ++ boundVars body)))
              (Nat.le_max_right gs.toList.length
                (max x.toList.length (maxStringLength (freeVars body ++ boundVars body)))))
            (Nat.le_max_left gs.toList.length
              (max x.toList.length (maxStringLength (freeVars body ++ boundVars body))))
        exact Nat.lt_of_le_of_lt hmaxle (Nat.lt_succ_self _)
      have hFreshFull :
          AdjointNamesFresh n
            ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx) := by
        apply adjointNamesFresh_of_length_bound
        simpa [linearCtxDom, n, gs, gradAdjointCounter] using hFullLen
      have hxle : maxStringLength [x] ≤ maxStringLength [x, gs] := by
        have : x ∈ [x, gs] := by simp
        simpa [maxStringLength] using mem_maxStringLength this
      have hFreshSmall :
          AdjointNamesFresh n
            ([(x, some (Typ.tensor ds))] : LinearCtx) := by
        apply adjointNamesFresh_of_length_bound
        have hSmallLen : maxStringLength [x] < n := Nat.lt_of_le_of_lt hxle hFullLen
        simpa [linearCtxDom, n, gs, gradAdjointCounter] using hSmallLen
      let adjBody := adjointTypedFrom body (Typ.tensor dsOut) x (Term.var gs) n
      have hAdj :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
            adjBody
            Typ.unit
            epsAdj
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx) := by
        exact adjointTypedFrom_preserves_typing [] Sigma [] x gs ds dsOut body epsBody n slot
          hBody hDiffCompat hSupp hFreshFull hFreshSmall
      have hClauseSig : OpSigMatch EffectLabel.accum Typ.unit Typ.unit := by
        simp [OpSigMatch, opArgType, opRetType]
      have hVarK :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
              ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
            (Term.var "k")
            (Typ.arrow Typ.unit Typ.unit epsHandle)
            []
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
              ("k", none)] : LinearCtx) := by
        simpa [List.append_assoc, epsHandle] using
          (HasType.var [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit)] : LinearCtx)
            ([] : LinearCtx) "k" (Typ.arrow Typ.unit Typ.unit epsHandle))
      have hVarP :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
              ("k", none)] : LinearCtx)
            (Term.var "p")
            Typ.unit
            []
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", none),
              ("k", none)] : LinearCtx) := by
        simpa [List.append_assoc] using
          (HasType.var [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
            ([("k", none)] : LinearCtx) "p" Typ.unit)
      have hClauseBody :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
              ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
            clauseBody
            Typ.unit
            epsHandle
            ([(x, some (Typ.tensor ds)), (gs, none), ("p", none),
              ("k", none)] : LinearCtx) := by
        have hClauseBodyRaw :
            HasType [] Sigma
              ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
                ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
              clauseBody
              Typ.unit
              (EffectRow.union (EffectRow.union [] []) epsHandle)
              ([(x, some (Typ.tensor ds)), (gs, none), ("p", none),
                ("k", none)] : LinearCtx) := by
          simpa [clauseBody] using
            (HasType.app [] Sigma
              ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
                ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
              ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
                ("k", none)] : LinearCtx)
              ([(x, some (Typ.tensor ds)), (gs, none), ("p", none),
                ("k", none)] : LinearCtx)
              (Term.var "k") (Term.var "p") Typ.unit Typ.unit epsHandle [] [] hVarK hVarP)
        have hClauseBodySub :
            SubEffRow (EffectRow.union (EffectRow.union [] []) epsHandle) epsHandle := by
          intro op hop
          simpa [EffectRow.union, List.mem_filter] using hop
        exact HasType.subEff [] Sigma
          ([(x, some (Typ.tensor ds)), (gs, none), ("p", some Typ.unit),
            ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
          ([(x, some (Typ.tensor ds)), (gs, none), ("p", none),
            ("k", none)] : LinearCtx)
          clauseBody
          Typ.unit
          (EffectRow.union (EffectRow.union [] []) epsHandle)
          epsHandle
          hClauseBodyRaw
          hClauseBodySub
      have hClausesNil :
          ClausesTyped [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
            Typ.unit
            epsHandle
            [] :=
        ClausesTyped.nil [] Sigma
          ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
          Typ.unit epsHandle
      have hClauses :
          ClausesTyped [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
            Typ.unit
            epsHandle
            clauses := by
        exact ClausesTyped.cons [] Sigma
          ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
          ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
          Typ.unit Typ.unit Typ.unit epsHandle
          EffectLabel.accum "p" "k" clauseBody [] none none
          hClauseSig hClauseBody hClausesNil
      have hHandleRaw :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
            (Term.handle [EffectLabel.accum]
              adjBody
              clauses)
            Typ.unit
            epsHandle
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx) := by
        refine HasType.handle [] Sigma
          ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
          ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
          ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
          adjBody
          clauses
          Typ.unit
          [EffectLabel.accum]
          epsAdj
          hAdj
          ?_
          ?_
          ?_
          hClauses
        · intro op hop
          simp at hop
          subst op
          by_cases hacc : EffectLabel.accum ∈ epsBody
          · exact List.mem_append_left _ hacc
          · apply List.mem_append_right
            simp [EffectRow.union, hacc]
        · intro cl hmem
          simp [clauses, clauseBody] at hmem
          rcases hmem with rfl
          simp
        · intro op hop
          simp at hop
          subst op
          refine ⟨(EffectLabel.accum, "p", "k", clauseBody), ?_, rfl⟩
          simp [clauses]
      have hHandleSub : SubEffRow epsHandle epsBody := by
        intro op hop
        have hopInfo : op ∈ epsAdj ∧ op ≠ EffectLabel.accum := by
          simpa [epsHandle, EffectRow.removeOps, EffectRow.removeOp, List.mem_filter] using hop
        have hopAdj : op ∈ epsAdj := hopInfo.1
        have hopNe : op ≠ EffectLabel.accum := by
          intro hEq
          exact hopInfo.2 hEq
        simp [epsAdj, EffectRow.union, List.mem_append, List.mem_filter] at hopAdj
        rcases hopAdj with hopBody | hopAccum
        · exact hopBody
        · exfalso
          exact hopNe hopAccum.1
      have hHandle :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
            (Term.handle [EffectLabel.accum]
              adjBody
              clauses)
            Typ.unit
            epsBody
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx) := by
        exact HasType.subEff [] Sigma
          ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
          ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
          _
          Typ.unit
          epsHandle
          epsBody
          hHandleRaw
          hHandleSub
      have hReturn :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, none), (tmp, some Typ.unit)] : LinearCtx)
            (Term.var x)
            (Typ.tensor ds)
            []
            ([(x, none), (gs, none), (tmp, some Typ.unit)] : LinearCtx) := by
        simpa [List.append_assoc] using
          (HasType.var [] Sigma
            ([] : LinearCtx)
            ([(gs, none), (tmp, some Typ.unit)] : LinearCtx)
            x (Typ.tensor ds))
      have hLet :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
            (Term.letBind tmp
              (Term.handle [EffectLabel.accum]
                adjBody
                clauses)
              (Term.var x))
            (Typ.tensor ds)
            epsBody
            ([(x, none), (gs, none)] : LinearCtx) := by
        simpa [List.append_assoc, EffectRow.union, tmp, clauses] using
          (HasType.letBind [] Sigma
            ([(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))] : LinearCtx)
            ([(x, some (Typ.tensor ds)), (gs, none)] : LinearCtx)
            ([(x, none), (gs, none)] : LinearCtx)
            tmp
            (Term.handle [EffectLabel.accum]
              adjBody
              clauses)
            (Term.var x)
            Typ.unit
            (Typ.tensor ds)
            epsBody
            []
            (some Typ.unit)
            hHandle
            hReturn)
      have hInner :
          HasType [] Sigma
            ([(x, some (Typ.tensor ds))] : LinearCtx)
            (Term.abs gs (Typ.tensor dsOut)
              (Term.letBind tmp
                (Term.handle [EffectLabel.accum]
                  adjBody
                  clauses)
                (Term.var x)))
            (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) epsBody)
            []
            ([(x, none)] : LinearCtx) := by
        exact HasType.abs [] Sigma
          ([(x, some (Typ.tensor ds))] : LinearCtx)
          ([(x, none)] : LinearCtx)
          gs
          (Typ.tensor dsOut)
          (Typ.tensor ds)
          epsBody
          _
          none
          hLet
      have hGrad :
          HasType [] Sigma []
            (Term.abs x (Typ.tensor ds)
              (Term.abs gs (Typ.tensor dsOut)
                (Term.letBind tmp
                  (Term.handle [EffectLabel.accum]
                    adjBody
                    clauses)
                  (Term.var x))))
            (Typ.arrow
              (Typ.tensor ds)
              (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) epsBody)
              [])
            []
            [] := by
        exact HasType.abs [] Sigma [] []
          x
          (Typ.tensor ds)
          (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) epsBody)
          []
          _
          none
          hInner
      exact ⟨Sigma,
        HasType.subEff [] Sigma [] []
          _
          (Typ.arrow
            (Typ.tensor ds)
            (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) epsBody)
            [])
          []
          eps
          hGrad
          (fun _ hop => by cases hop),
        h_wf,
        fun _ _ _ hlook => hlook⟩
  | tvmap s x tv d body =>
      intro locs hsep _hlinear t eps h_typ _hscope
      rcases HasType.vmap_inv h_typ with ⟨_hGamma, tBody, epsBody, slot, htEq, hBody⟩
      subst t
      let Sigma' := addDimStoreTypOn d (locRefs body) Sigma
      have hBodyLifted :
          HasType [] (addDimStoreTyp d Sigma)
            [(x, some (addDim d tv))]
            (addDimTerm d body)
            (addDim d tBody) epsBody
            [(x, slot.map (addDim d))] := by
        simpa using addDim_preserves_typing d hBody
      have hBodySelective :
          HasType [] Sigma'
            [(x, some (addDim d tv))]
            (addDimTerm d body)
            (addDim d tBody) epsBody
            [(x, slot.map (addDim d))] := by
        apply hasType_store_weaken_on_locRefs hBodyLifted
        intro ell tEll hmem hlook
        have hmemBody : ell ∈ locRefs body := by
          simpa [locRefs_addDimTerm (d := d) (e := body)] using hmem
        simpa [Sigma', addDimStoreTypOn_lookup_mem d (locRefs body) Sigma ell hmemBody] using hlook
      have hAbs :
          HasType [] Sigma' []
            (Term.abs x (addDim d tv) (addDimTerm d body))
            (Typ.arrow (addDim d tv) (addDim d tBody) epsBody) [] [] := by
        exact HasType.abs [] Sigma' [] []
          x (addDim d tv) (addDim d tBody) epsBody
          (addDimTerm d body) (slot.map (addDim d)) hBodySelective
      have hsepBody : LocRefsSeparated locs (locRefs body) := by
        simpa [locRefs] using hsep
      have hOn : StoreTypOn locs Sigma Sigma' := by
        intro ell tEll hmem hlook
        have hnot : ell ∉ locRefs body := hsepBody ell hmem
        simpa [Sigma', addDimStoreTypOn_lookup_not_mem d (locRefs body) Sigma ell hnot] using hlook
      exact ⟨Sigma',
        HasType.subEff [] Sigma' [] []
          (Term.abs x (addDim d tv) (addDimTerm d body))
          (Typ.arrow (addDim d tv) (addDim d tBody) epsBody) [] eps hAbs
          (fun _ hop => by cases hop),
        StoreWf.addDimStoreTypOn d (locRefs body) h_wf,
        hOn⟩
  | ctx sigma sigma' E e0 e0' h_inner ih =>
      intro locs hsep h_linear t eps h_typ h_scope
      have h_scope_inner : WellScoped e0 := wellScoped_plug_inner h_scope
      rcases runtimeLinear_plug (E := E) (e := e0) h_linear with
        ⟨h_linear_inner, hsepCtx⟩
      have hsepBoth : LocRefsSeparated (ctxLocRefs E ++ locs) (locRefs e0) := by
        intro ell hmem hloc
        rcases List.mem_append.mp hmem with hctx | houter
        · exact hsepCtx ell hctx hloc
        · exact hsep ell houter ((mem_locRefs_plug E e0 ell).2 (Or.inr hloc))
      rcases plug_preserves_typing_closed_on_ctxLocRefs
          (sigma2 := sigma') (locs := locs) h_typ
          (fun {t0 : Typ} {eps0 : EffectRow} (h_inner_typ : HasType [] Sigma [] e0 t0 eps0 []) => by
            rcases ih (locs := ctxLocRefs E ++ locs) h_wf hsepBoth h_linear_inner h_inner_typ h_scope_inner with
              ⟨Sigma2, h_e0', h_wf2, h_onBoth⟩
            exact ⟨Sigma2, h_e0', h_wf2, h_onBoth⟩) with
        ⟨Sigma2, h_plug', h_wf2, h_onBoth⟩
      exact ⟨Sigma2, h_plug', h_wf2,
        StoreTypOn.append_right (locs1 := ctxLocRefs E) (locs2 := locs) h_onBoth⟩

/-- Preservation for closed runtime-linear programs. The earlier
    runtime-linear counterexample shows this extra hypothesis is not
    optional: without it, the generic `ctx` case is false because a
    sibling subterm may retain a consumed location. The remaining open
    case is only the agreed calculus-level blocker `tgrad`. -/
theorem preservation
    (sigma sigma' : Store) (Sigma : StoreTyp)
    (e e' : Term) (t : Typ) (eps : EffectRow)
    (h_typ : HasType [] Sigma [] e t eps [])
    (h_scope : WellScoped e)
    (h_linear : RuntimeLinear e)
    (h_wf : StoreWf sigma Sigma)
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma',
      HasType [] Sigma' [] e' t eps [] ∧ StoreWf sigma' Sigma' := by
  rcases preservation_aux Sigma ⟨sigma, e⟩ ⟨sigma', e'⟩ h_wf h_step
      (locs := []) (by intro ell hell; cases hell) h_linear h_typ h_scope with
    ⟨Sigma', h_typ', h_wf', _h_on⟩
  exact ⟨Sigma', h_typ', h_wf'⟩
-- Wave 5r: plug_preserves_typing needs slot-param + filter→tombstone update.
-- Original proof preserved below.
/-  -- Induction on the evaluation context. The `hole` case is a direct
  -- application of `h_inner`; the other cases mirror the shape of the
  -- corresponding `HasType` constructor after an inversion step.
  -- TODO Wave-4 Sync: close each constructor case using the matching
  -- inversion lemma (`app_inv`, `add_inv`, ...). Cases where the
  -- inversion lemma currently drops a `SubEffRow` witness need a
  -- strengthened inversion or an explicit `HasType.subEff` wrap.
  cases E with
  | hole =>
      -- plug hole e = e; directly apply the inner-step hypothesis.
      simpa [plug] using h_inner h
  | fst tRight =>
      -- plug fst e = Term.fst e.
      have h' : HasType Delta Sigma Gamma (Term.fst tRight e) t eps Gamma' := by
        simpa [plug] using h
      let h_pair := HasType.fst_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_pair
      refine ⟨Sigma2, ?_, h_sub⟩
      show HasType Delta Sigma2 Gamma (Term.fst tRight e') t eps Gamma'
      exact HasType.fst Delta Sigma2 Gamma Gamma' e' t tRight eps h_e'
  | snd tLeft =>
      have h' : HasType Delta Sigma Gamma (Term.snd tLeft e) t eps Gamma' := by
        simpa [plug] using h
      let h_pair := HasType.snd_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_pair
      refine ⟨Sigma2, ?_, h_sub⟩
      show HasType Delta Sigma2 Gamma (Term.snd tLeft e') t eps Gamma'
      exact HasType.snd Delta Sigma2 Gamma Gamma' e' tLeft t eps h_e'
  | copy =>
      have h' : HasType Delta Sigma Gamma (Term.copy e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_inner_ds⟩ := HasType.copy_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_inner_ds
      refine ⟨Sigma2, ?_, h_sub⟩
      subst hteq
      show HasType Delta Sigma2 Gamma (Term.copy e') (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) eps Gamma'
      exact HasType.copy Delta Sigma2 Gamma Gamma' e' ds eps h_e'
  | sum d =>
      have h' : HasType Delta Sigma Gamma (Term.sum e d) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, hteq, hmem, h_inner_ds⟩ := HasType.sum_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_inner_ds
      refine ⟨Sigma2, ?_, h_sub⟩
      subst hteq
      show HasType Delta Sigma2 Gamma (Term.sum e' d) (Typ.tensor (rem ds d)) eps Gamma'
      exact HasType.tsum Delta Sigma2 Gamma Gamma' e' ds d eps h_e' hmem
  | expand d =>
      have h' : HasType Delta Sigma Gamma (Term.expand e d) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, hteq, h_inner_ds⟩ := HasType.expand_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_inner_ds
      refine ⟨Sigma2, ?_, h_sub⟩
      subst hteq
      show HasType Delta Sigma2 Gamma (Term.expand e' d) (Typ.tensor (ins ds d)) eps Gamma'
      exact HasType.texpand Delta Sigma2 Gamma Gamma' e' ds d eps h_e'
  | uniformLike lo hi =>
      have h' : HasType Delta Sigma Gamma (Term.uniformLike e lo hi) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, eps0, hteq, h_inner_ds, hsub_eps⟩ := HasType.uniformLike_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_inner_ds
      refine ⟨Sigma2, ?_, h_sub⟩
      subst hteq
      -- Rebuild at the raw T-UniformLike effect row then widen via subEff.
      have h_raw : HasType Delta Sigma2 Gamma (Term.uniformLike e' lo hi)
                     (Typ.tensor ds)
                     (EffectRow.union eps0 [EffectLabel.random]) Gamma' :=
        HasType.uniformLike Delta Sigma2 Gamma Gamma' e' ds lo hi eps0 h_e'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | appL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.app e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_e2' : HasType Delta Sigma2 Γmid e2 t1 eps2 Gamma' :=
        hasType_store_weaken h_e2 h_sub
      have h_raw : HasType Delta Sigma2 Gamma (Term.app e' e2) t
                     (EffectRow.union (EffectRow.union eps1 eps2) epsBody)
                     Gamma' :=
        HasType.app Delta Sigma2 Gamma Γmid Gamma' e' e2 t1 t
          epsBody eps1 eps2 h_e1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | appR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.app v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, epsBody, eps1, eps2, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_app_inv h'
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_v1' : HasType Delta Sigma2 Gamma v1 (Typ.arrow t1 t epsBody)
                     eps1 Γmid :=
        hasType_store_weaken h_v1 h_sub
      have h_raw : HasType Delta Sigma2 Gamma (Term.app v1 e') t
                     (EffectRow.union (EffectRow.union eps1 eps2) epsBody)
                     Gamma' :=
        HasType.app Delta Sigma2 Gamma Γmid Gamma' v1 e' t1 t
          epsBody eps1 eps2 h_v1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | letBind x e2 =>
      have h' : HasType Delta Sigma Gamma (Term.letBind x e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, eps1, eps2, hΓout, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_letBind_inv h'
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_e2' : HasType Delta Sigma2 (Γmid ++ [(x, t1)]) e2 t eps2 Γ3 :=
        hasType_store_weaken h_e2 h_sub
      subst hΓout
      have h_raw : HasType Delta Sigma2 Gamma (Term.letBind x e' e2) t
                     (EffectRow.union eps1 eps2)
                     (Γ3.filter (fun p => p.1 ≠ x)) :=
        HasType.letBind Delta Sigma2 Gamma Γmid Γ3 x e' e2 t1 t
          eps1 eps2 h_e1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | letpair x y e2 =>
      have h' : HasType Delta Sigma Gamma (Term.letpair x y e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, Γ3, t1, t2, eps1, eps2, hΓout, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_letpair_inv h'
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_e2' : HasType Delta Sigma2 (Γmid ++ [(x, t1), (y, t2)])
                     e2 t eps2 Γ3 :=
        hasType_store_weaken h_e2 h_sub
      subst hΓout
      have h_raw : HasType Delta Sigma2 Gamma (Term.letpair x y e' e2) t
                     (EffectRow.union eps1 eps2)
                     (Γ3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y)) :=
        HasType.letpair Delta Sigma2 Gamma Γmid Γ3 x y e' e2 t1 t2 t
          eps1 eps2 h_e1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | pairL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.pair e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_e2' : HasType Delta Sigma2 Γmid e2 t2 eps2 Gamma' :=
        hasType_store_weaken h_e2 h_sub
      subst hteq
      have h_raw : HasType Delta Sigma2 Gamma (Term.pair e' e2)
                     (Typ.pair t1 t2)
                     (EffectRow.union eps1 eps2) Gamma' :=
        HasType.tpair Delta Sigma2 Gamma Γmid Gamma' e' e2 t1 t2
          eps1 eps2 h_e1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | pairR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.pair v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, t1, t2, eps1, eps2, hteq, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_pair_inv h'
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_v1' : HasType Delta Sigma2 Gamma v1 t1 eps1 Γmid :=
        hasType_store_weaken h_v1 h_sub
      subst hteq
      have h_raw : HasType Delta Sigma2 Gamma (Term.pair v1 e')
                     (Typ.pair t1 t2)
                     (EffectRow.union eps1 eps2) Gamma' :=
        HasType.tpair Delta Sigma2 Gamma Γmid Gamma' v1 e' t1 t2
          eps1 eps2 h_v1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | addL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.add e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_e2' : HasType Delta Sigma2 Γmid e2 (Typ.tensor ds) eps2 Gamma' :=
        hasType_store_weaken h_e2 h_sub
      subst hteq
      have h_raw : HasType Delta Sigma2 Gamma (Term.add e' e2)
                     (Typ.tensor ds)
                     (EffectRow.union eps1 eps2) Gamma' :=
        HasType.tadd Delta Sigma2 Gamma Γmid Gamma' e' e2 ds
          eps1 eps2 h_e1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | addR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.add v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_add_inv h'
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_v1' : HasType Delta Sigma2 Gamma v1 (Typ.tensor ds) eps1 Γmid :=
        hasType_store_weaken h_v1 h_sub
      subst hteq
      have h_raw : HasType Delta Sigma2 Gamma (Term.add v1 e')
                     (Typ.tensor ds)
                     (EffectRow.union eps1 eps2) Gamma' :=
        HasType.tadd Delta Sigma2 Gamma Γmid Gamma' v1 e' ds
          eps1 eps2 h_v1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | mulL e2 =>
      have h' : HasType Delta Sigma Gamma (Term.mul e e2) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_e1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      obtain ⟨Sigma2, h_e1', h_sub⟩ := h_inner h_e1
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_e2' : HasType Delta Sigma2 Γmid e2 (Typ.tensor ds) eps2 Gamma' :=
        hasType_store_weaken h_e2 h_sub
      subst hteq
      have h_raw : HasType Delta Sigma2 Gamma (Term.mul e' e2)
                     (Typ.tensor ds)
                     (EffectRow.union eps1 eps2) Gamma' :=
        HasType.tmul Delta Sigma2 Gamma Γmid Gamma' e' e2 ds
          eps1 eps2 h_e1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | mulR v1 =>
      have h' : HasType Delta Sigma Gamma (Term.mul v1 e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨ds, Γmid, eps1, eps2, hteq, h_v1, h_e2, hsub_eps⟩ :=
        HasType.plug_mul_inv h'
      obtain ⟨Sigma2, h_e2', h_sub⟩ := h_inner h_e2
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_v1' : HasType Delta Sigma2 Gamma v1 (Typ.tensor ds) eps1 Γmid :=
        hasType_store_weaken h_v1 h_sub
      subst hteq
      have h_raw : HasType Delta Sigma2 Gamma (Term.mul v1 e')
                     (Typ.tensor ds)
                     (EffectRow.union eps1 eps2) Gamma' :=
        HasType.tmul Delta Sigma2 Gamma Γmid Gamma' v1 e' ds
          eps1 eps2 h_v1' h_e2'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | perform op =>
      have h' : HasType Delta Sigma Gamma (Term.perform op e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨tArg, eps0, h_arg, hmatch, hsub_eps⟩ := HasType.perform_inv h'
      obtain ⟨Sigma2, h_arg', h_sub⟩ := h_inner h_arg
      refine ⟨Sigma2, ?_, h_sub⟩
      have h_raw : HasType Delta Sigma2 Gamma (Term.perform op e') t
                     (EffectRow.union [op] eps0) Gamma' :=
        HasType.perform Delta Sigma2 Gamma Gamma' op e' tArg t eps0
          h_arg' hmatch
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
  | handle epsH clauses =>
      have h' : HasType Delta Sigma Gamma (Term.handle epsH e clauses)
                  t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨Γmid, epsB, hb, hHsubB, hClIn, hClCov, hcls, hsub_eps⟩ :=
        HasType.handle_inv_strong h'
      obtain ⟨Sigma2, hb', h_sub⟩ := h_inner hb
      refine ⟨Sigma2, ?_, h_sub⟩
      -- Weaken the clauses via clausesTyped_store_weaken.
      have hcls' : ClausesTyped Delta Sigma2 Γmid Gamma' t
                     (EffectRow.removeOps epsB epsH) clauses :=
        clausesTyped_store_weaken hcls h_sub
      have h_raw : HasType Delta Sigma2 Gamma
                     (Term.handle epsH e' clauses) t
                     (EffectRow.removeOps epsB epsH) Gamma' :=
        HasType.handle Delta Sigma2 Gamma Γmid Gamma' e' clauses t
          epsH epsB hb' hHsubB hClIn hClCov hcls'
      exact HasType.subEff _ _ _ _ _ _ _ _ h_raw hsub_eps
-/

end LaCaDiLE
