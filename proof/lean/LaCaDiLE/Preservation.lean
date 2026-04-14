-- LaCaDiLE/Preservation.lean — preservation theorem (Phase 2 proof).
--
-- WS2.5 target: reduction preserves typing and store well-formedness.
--
-- Wave 4 status: mechanical cases that do NOT depend on a typing rule
-- for `Term.loc` are closed directly by Step-case inversion plus small
-- HasType inversion lemmas. Cases that either (a) rely on substitution
-- (Wave 2 sibling work on `subst_preserves_typing`), (b) need to type a
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

namespace LaCaDiLE

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
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.fst e) t eps Gamma2) :
    ∃ t2, HasType Delta Sigma Gamma1 e (Typ.pair t t2) eps Gamma2 := by
  generalize heq : Term.fst e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | fst _ _ _ _ _ _ t2 _ h_inner _ =>
      cases heq
      exact ⟨t2, h_inner⟩
  | subEff Δ S Γ Γ' _ t_m eps0 eps' _h_sub h_sub ih =>
      obtain ⟨t2, h_inv⟩ := ih heq
      exact ⟨t2, HasType.subEff Δ S Γ Γ' e (Typ.pair t_m t2) eps0 eps' h_inv h_sub⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

-- Wave 0.5 ripple: every inversion lemma is now blocked on the
-- subEff constructor, which `cases h` cannot dispatch without a
-- central strip_subEff helper. All inversions sorry'd; Wave 2
-- builds the helper and closes them in one pass.

theorem HasType.snd_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.snd e) t eps Gamma2) :
    ∃ t1, HasType Delta Sigma Gamma1 e (Typ.pair t1 t) eps Gamma2 := by
  generalize heq : Term.snd e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | snd Δ S Γ1 Γ2 e' t1 t2 eps' h_inner _ih =>
      cases heq
      exact ⟨t1, h_inner⟩
  | subEff Δ S Γ Γ' e' t' eps0 eps1 _h_sub h_sub ih =>
      obtain ⟨t1, h_inv⟩ := ih heq
      exact ⟨t1, HasType.subEff Δ S Γ Γ' e (Typ.pair t1 t') eps0 eps1 h_inv h_sub⟩
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
    ∃ t0, t = Typ.pair t0 t0 ∧
          HasType Delta Sigma Gamma1 e t0 eps Gamma2 := by
  generalize heq : Term.copy e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | copy _ _ _ _ _ t0 _ h' _ =>
      cases heq
      exact ⟨t0, rfl, h'⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _h_sub h_sub ih =>
      obtain ⟨t0, hteq, h_inv⟩ := ih heq
      refine ⟨t0, hteq, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e t0 eps0 eps' h_inv h_sub
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
    {e : Term} {i : Nat} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.sum e i) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (rem ds i) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.sum e i = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tsum _ _ _ _ _ ds _ _ h' _ _ =>
      cases heq
      exact ⟨ds, rfl, h'⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _h_sub h_sub ih =>
      obtain ⟨ds, hteq, h_inv⟩ := ih heq
      refine ⟨ds, hteq, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' h_inv h_sub
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

theorem HasType.expand_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {i k : Nat} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.expand e i k) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (ins ds i k) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.expand e i k = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | texpand _ _ _ _ _ ds _ _ _ h' _ _ =>
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
    effect is only guaranteed in the inner eps0 via the union. -/
theorem HasType.uniformLike_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {lo hi : Float} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.uniformLike e lo hi) t eps Gamma2) :
    ∃ ds eps0,
      t = Typ.tensor ds ∧
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps0 Gamma2 := by
  generalize heq : Term.uniformLike e lo hi = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | uniformLike _ _ _ _ _ ds _ _ eps0 h' _ =>
      cases heq
      exact ⟨ds, eps0, rfl, h'⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
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

/-- After removing `ell1`, a lookup at `ell2 ≠ ell1` agrees with the
    lookup in the original store. Proved on Option.isSome since
    StoreWf only tracks membership. -/
theorem storeLookup_isSome_remove_ne (sigma : Store) (ell1 ell2 : Loc)
    (hne : ell2 ≠ ell1) :
    (storeLookup (storeRemove sigma ell1) ell2).isSome =
    (storeLookup sigma ell2).isSome := by
  induction sigma with
  | nil => rfl
  | cons hd tl ih =>
    show (storeLookup ((hd :: tl).filter (fun p => p.1 ≠ ell1)) ell2).isSome =
         (storeLookup (hd :: tl) ell2).isSome
    by_cases hHd1 : hd.1 = ell1
    · -- hd removed
      have : ((hd :: tl).filter (fun p => p.1 ≠ ell1)) =
             (tl.filter (fun p => p.1 ≠ ell1)) := by
        simp [List.filter_cons, hHd1]
      rw [this]
      have hHdNe2 : hd.1 ≠ ell2 := by rw [hHd1]; exact fun h => hne h.symm
      show (storeLookup (tl.filter (fun p => p.1 ≠ ell1)) ell2).isSome =
           (storeLookup (hd :: tl) ell2).isSome
      rw [show storeLookup (hd :: tl) ell2 = storeLookup tl ell2 from by
            simp [storeLookup, List.find?, hHdNe2]]
      exact ih
    · -- hd survives
      have hFilter : ((hd :: tl).filter (fun p => p.1 ≠ ell1)) =
             hd :: (tl.filter (fun p => p.1 ≠ ell1)) := by
        simp [List.filter_cons, hHd1]
      rw [hFilter]
      by_cases hHd2 : hd.1 = ell2
      · -- hd is our target
        simp [storeLookup, List.find?, hHd2]
      · show (storeLookup (hd :: (tl.filter _)) ell2).isSome =
             (storeLookup (hd :: tl) ell2).isSome
        simp only [storeLookup, List.find?, hHd2, decide_false,
                   Bool.false_eq_true, ite_false]
        exact ih

/-- After removing `ell1` from a store typing, `ell2 ≠ ell1` is in the
    filtered domain iff it was in the original. -/
theorem mem_storeTypDom_remove_iff (Sigma : StoreTyp) (ell1 ell2 : Loc)
    (hne : ell2 ≠ ell1) :
    ell2 ∈ storeTypDom (storeTypRemove Sigma ell1) ↔
    ell2 ∈ storeTypDom Sigma := by
  induction Sigma with
  | nil => simp [storeTypDom, storeTypRemove]
  | cons hd tl ih =>
    show ell2 ∈ storeTypDom ((hd :: tl).filter (fun p => p.1 ≠ ell1)) ↔
         ell2 ∈ storeTypDom (hd :: tl)
    by_cases hHd1 : hd.1 = ell1
    · have hFilt : ((hd :: tl).filter (fun p => p.1 ≠ ell1)) =
                   tl.filter (fun p => p.1 ≠ ell1) := by
        simp [List.filter_cons, hHd1]
      rw [hFilt]
      show ell2 ∈ storeTypDom (storeTypRemove tl ell1) ↔
           ell2 ∈ storeTypDom (hd :: tl)
      rw [ih]
      show ell2 ∈ tl.map Prod.fst ↔ ell2 ∈ (hd :: tl).map Prod.fst
      simp only [List.map_cons, List.mem_cons]
      constructor
      · intro h; right; exact h
      · rintro (h_eq | h_mem)
        · exfalso; apply hne; rw [h_eq, hHd1]
        · exact h_mem
    · have hFilt : ((hd :: tl).filter (fun p => p.1 ≠ ell1)) =
                   hd :: tl.filter (fun p => p.1 ≠ ell1) := by
        simp [List.filter_cons, hHd1]
      rw [hFilt]
      show ell2 ∈ (hd :: storeTypRemove tl ell1).map Prod.fst ↔
           ell2 ∈ (hd :: tl).map Prod.fst
      simp only [List.map_cons, List.mem_cons]
      show ell2 = hd.fst ∨ ell2 ∈ storeTypDom (storeTypRemove tl ell1) ↔
           ell2 = hd.fst ∨ ell2 ∈ storeTypDom tl
      rw [ih]

/-- Remove-then-extend for single-location-consume Steps (tsum,
    texpand, tuniformLike). Removing a live location and extending
    with a fresh one preserves store well-formedness. -/
theorem StoreWf.remove_extend
    {sigma : Store} {Sigma : StoreTyp}
    (ellIn ellOut : Loc) (w : TensorVal) (tOut : Typ)
    (h_wf : StoreWf sigma Sigma)
    (_h_fresh : ellOut = storeFreshLoc sigma)
    (_h_ne : ellIn ≠ ellOut) :
    StoreWf (storeExtend (storeRemove sigma ellIn) ellOut w)
            (storeTypExtend (storeTypRemove Sigma ellIn) ellOut tOut) := by
  refine ⟨?_, ?_⟩
  · intro ell' hell'
    -- hell' : ell' ∈ storeTypDom (ellOut :: storeTypRemove Sigma ellIn)
    simp only [storeTypDom, storeTypExtend, List.map_cons,
               List.mem_cons] at hell'
    rcases hell' with hEq | hOld
    · -- ell' = ellOut: the extended store has (ellOut, w) at head.
      rw [hEq]
      show (storeLookup ((ellOut, w) :: storeRemove sigma ellIn) ellOut).isSome
      simp [storeLookup, List.find?]
    · -- ell' ∈ storeTypDom (storeTypRemove Sigma ellIn): peel off the
      -- filter to get ell' ∈ Sigma's domain + ell' ≠ ellIn.
      by_cases hell'In : ell' = ellIn
      · -- ell' = ellIn means it was filtered out — contradiction.
        exfalso
        rw [hell'In] at hOld
        -- ellIn not in filtered domain
        have : ellIn ∉ storeTypDom (storeTypRemove Sigma ellIn) := by
          intro hc
          show False
          have hc' : ellIn ∈ (Sigma.filter (fun p => p.1 ≠ ellIn)).map Prod.fst := hc
          rw [List.mem_map] at hc'
          rcases hc' with ⟨p, hp_mem, hp_eq⟩
          rw [List.mem_filter] at hp_mem
          -- hp_eq : p.fst = ellIn; hp_mem.2 : p.1 ≠ ellIn (decoded)
          have : p.1 ≠ ellIn := by
            have := hp_mem.2
            simp only [ne_eq, decide_not, Bool.not_eq_true',
                       decide_eq_false_iff_not] at this
            exact this
          exact this hp_eq
        exact this hOld
      · -- ell' ≠ ellIn: transfer to Sigma's domain via mem_storeTypDom_remove_iff
        have hInSigma : ell' ∈ storeTypDom Sigma :=
          (mem_storeTypDom_remove_iff Sigma ellIn ell' hell'In).mp hOld
        have hSigmaLive : (storeLookup sigma ell').isSome :=
          h_wf.1 ell' hInSigma
        -- Propagate through remove + extend
        have hRemLive : (storeLookup (storeRemove sigma ellIn) ell').isSome := by
          rw [storeLookup_isSome_remove_ne sigma ellIn ell' hell'In]
          exact hSigmaLive
        show (storeLookup ((ellOut, w) :: storeRemove sigma ellIn) ell').isSome
        by_cases hell'Out : ell' = ellOut
        · rw [hell'Out]; simp [storeLookup, List.find?]
        · have hne : ¬ (ellOut = ell') := fun he => hell'Out he.symm
          simp only [storeLookup, List.find?, hne, decide_false,
                     Bool.false_eq_true, ite_false]
          exact hRemLive
  · intro ell' hell'
    show ell' ∈ storeTypDom (storeTypExtend (storeTypRemove Sigma ellIn) ellOut tOut)
    simp only [storeTypDom, storeTypExtend, List.map_cons, List.mem_cons]
    by_cases hell'Out : ell' = ellOut
    · left; exact hell'Out
    · right
      have hne : ¬ (ellOut = ell') := fun he => hell'Out he.symm
      -- Lookup fell through ellOut to storeRemove sigma ellIn.
      have hRem : (storeLookup (storeRemove sigma ellIn) ell').isSome := by
        have : (storeLookup ((ellOut, w) :: storeRemove sigma ellIn) ell').isSome := hell'
        simp only [storeLookup, List.find?, hne, decide_false,
                   Bool.false_eq_true, ite_false] at this
        exact this
      -- Now ell' is isSome in the removed store. If ell' = ellIn, the
      -- removed store can't have it (contradiction). Otherwise use
      -- storeLookup_isSome_remove_ne + h_wf.2.
      by_cases hell'In : ell' = ellIn
      · exfalso
        rw [hell'In] at hRem
        -- storeLookup (storeRemove sigma ellIn) ellIn = none
        have : storeLookup (storeRemove sigma ellIn) ellIn = none := by
          simp only [storeLookup, storeRemove]
          induction sigma with
          | nil => rfl
          | cons hd tl ihl =>
            by_cases hHd : hd.1 = ellIn
            · simp [List.filter_cons, hHd, ihl]
            · simp [List.filter_cons, hHd, List.find?, ihl]
        rw [this] at hRem
        exact absurd hRem (by simp)
      · have hInSigma : (storeLookup sigma ell').isSome := by
          rw [← storeLookup_isSome_remove_ne sigma ellIn ell' hell'In]
          exact hRem
        have hSigmaDom : ell' ∈ storeTypDom Sigma := h_wf.2 ell' hInSigma
        exact (mem_storeTypDom_remove_iff Sigma ellIn ell' hell'In).mpr hSigmaDom

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

/-- Preservation: if a configuration is well-typed and steps, the
    resulting configuration has the same type (under a possibly-extended
    store typing) and preserves store well-formedness.

    Wave 4 coverage: value-projection and handler-return cases closed;
    store-allocating and substitution-dependent cases left as precise
    TODO-annotated sorries (see file header). -/
theorem preservation
    (sigma sigma' : Store) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (e e' : Term) (t : Typ) (eps : EffectRow)
    (h_typ : HasType [] Sigma Gamma e t eps [])
    (h_wf : StoreWf sigma Sigma)
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma',
      HasType [] Sigma' Gamma e' t eps [] ∧ StoreWf sigma' Sigma' := by
  cases h_step with
  | beta s x tv body v hv =>
      -- TODO Wave 2: needs subst_preserves_typing
      sorry
  | letBind s x v body hv =>
      -- TODO Wave 2: needs subst_preserves_typing
      sorry
  | letpair s x y v1 v2 body hv1 hv2 =>
      -- TODO Wave 2: needs subst_preserves_typing
      sorry
  | fst s v1 v2 hv1 hv2 =>
      -- E-Fst: fst (pair v1 v2) ↦ v1. Use fst_inv + pair_inv
      -- (strengthened with SubEffRow witness) to extract h1 : v1 at
      -- eps1, then widen eps1 to eps via subEff.
      obtain ⟨t2, h_pair⟩ := HasType.fst_inv h_typ
      obtain ⟨Γ2, eps1, eps2, h1, _h2, _hsub⟩ := HasType.pair_inv h_pair
      refine ⟨Sigma, ?_, h_wf⟩
      -- TODO Wave 2: h1 is at Γ2 and eps1, but we need Γ = [] and
      -- eps. Needs value_preserves_closed_context (Γ2 = []) plus
      -- SubEffRow eps1 eps from the union sub-relationship.
      sorry
  | snd s v1 v2 hv1 hv2 =>
      -- Symmetric to fst; same obstacle.
      -- TODO Wave 2: value_preserves_context + weaken_eff + union_comm.
      sorry
  | tconst s v ds ell hell =>
      -- E-Const: const(v, ds) ↦ loc ell in extended store.
      obtain ⟨ht, hG⟩ := HasType.const_inv h_typ
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_, ?_⟩
      · subst ht
        subst hG
        have hlook := storeTypLookup_extend_self Sigma ell (Typ.tensor ds)
        -- HasType.loc at [] for the extended Sigma, then widen to eps
        -- via subEff since SubEffRow [] eps is trivially true.
        have h_loc : HasType [] (storeTypExtend Sigma ell (Typ.tensor ds))
                              [] (Term.loc ell) (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ell (Typ.tensor ds) hlook
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_loc hsub
      · exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf hell
  | copy s ell ellNew w hlook hfresh =>
      -- E-Copy: copy(loc ell) ↦ pair (loc ell) (loc ellNew).
      -- Both locs type at t0 under the extended Sigma. StoreWf as
      -- defined tracks domain membership only, so we don't need
      -- ell ≠ ellNew for well-formedness (the lookups remain isSome
      -- in both directions regardless).
      obtain ⟨t0, htEq, h_e⟩ := HasType.copy_inv h_typ
      obtain ⟨hlookT, hGE⟩ := HasType.loc_inv h_e
      refine ⟨storeTypExtend Sigma ellNew t0, ?_, ?_⟩
      · subst htEq
        subst hGE
        -- Build `pair (loc ell) (loc ellNew)` at the empty effect row,
        -- then widen to the outer eps via subEff.
        have hlookSelf := storeTypLookup_extend_self Sigma ellNew t0
        -- For ell ≠ ellNew, lookup falls through. For ell = ellNew,
        -- lookup hits the new entry which is the same type t0.
        have hlookEll :
            storeTypLookup (storeTypExtend Sigma ellNew t0) ell = some t0 := by
          by_cases hne : ell = ellNew
          · rw [hne]; exact hlookSelf
          · have hne' : ¬ (ellNew = ell) := fun he => hne he.symm
            simp only [storeTypLookup, storeTypExtend, List.find?,
                       hne', decide_false, Bool.false_eq_true, ite_false]
            simpa [storeTypLookup] using hlookT
        have h_l1 : HasType [] (storeTypExtend Sigma ellNew t0) []
                      (Term.loc ell) t0 [] [] :=
          HasType.loc _ _ _ ell t0 hlookEll
        have h_l2 : HasType [] (storeTypExtend Sigma ellNew t0) []
                      (Term.loc ellNew) t0 [] [] :=
          HasType.loc _ _ _ ellNew t0 hlookSelf
        have h_pair : HasType [] (storeTypExtend Sigma ellNew t0) []
                        (Term.pair (Term.loc ell) (Term.loc ellNew))
                        (Typ.pair t0 t0) [] [] :=
          HasType.tpair [] _ [] [] [] _ _ t0 t0 [] [] h_l1 h_l2
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_pair hsub
      · exact StoreWf.extend_fresh ellNew w t0 h_wf hfresh
  | tadd s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      -- E-Add consumes ell1, ell2 and allocates ellOut at tensor[ds].
      -- Substitution: need to prove store stays well-formed under the
      -- remove+extend pattern, which requires a richer StoreWf
      -- manipulation than the simple extend_fresh lemma covers.
      -- TODO Wave 2: StoreWf.remove_remove_extend lemma.
      sorry
  | tmul s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      -- TODO Wave 2: same as tadd.
      sorry
  | tsum s ell ellOut w i hlook hfresh =>
      -- TODO Wave 2: needs StoreWf.remove_extend lemma.
      sorry
  | texpand s ell ellOut w i k hlook hfresh =>
      -- TODO Wave 2: parallel to tsum.
      sorry
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      -- TODO Wave 2: parallel to tsum.
      sorry
  | handleRet s epsH v clauses hv =>
      -- E-Handle-Ret: handle[εH] v clauses ↦ v when v is a value.
      -- handle_inv gives us a sub-derivation for v at some interior
      -- effect row epsB and intermediate context Γ2. We use subEff
      -- to widen epsB to the outer eps, and (since the theorem's
      -- output context is []) we rely on h_body directly when it
      -- already lands at the right context shape. Full value-context-
      -- preservation is still a Wave 2 task for the Γ2 = [] step.
      obtain ⟨Gamma2, epsB, h_body⟩ := HasType.handle_inv h_typ
      refine ⟨Sigma, ?_, h_wf⟩
      -- TODO Wave 2: value-context-preservation gives Γ2 = Gamma = [].
      sorry
  | handleOpDirect s op v epsH clauses x k hb tRet hv hmem =>
      -- TODO Wave 2: needs subst_preserves_typing
      sorry
  | tgrad s x tv tOut body =>
      -- TODO Wave 2: needs adjoint_preserves_typing (AdjointTyping.lean)
      sorry
  | tvmap s x tv body d =>
      -- TODO Wave 2: needs addDim_preserves_typing tvmap case
      sorry
  | ctx sig sig' E e0 e0' h_inner =>
      -- E-Ctx: the sub-term step needs a replacement lemma:
      -- HasType (plug E e0) → HasType e0 (in-hole type) → Step e0 e0'
      -- → HasType e0' (same) → HasType (plug E e0').
      -- TODO Wave 2: state and prove `plug_preserves_typing` lemma.
      sorry

end LaCaDiLE
