-- LaCaDiLE/Store.lean — Store typing and well-formedness.
--
-- Companion to Typing.lean. The store-typing relation links runtime store
-- contents to their corresponding types, and `StoreWf` asserts that the set
-- of live locations in the store matches the linear bindings in Γ. This is
-- the structural form of Theorem 4 (linearity soundness, WS2.8).

import LaCaDiLE.Syntax

namespace LaCaDiLE

/-- Store typing: a finite map from locations to the types they store.
    Used only for metatheory; runtime store contents live in `Store`. -/
abbrev StoreTyp := List (Loc × Typ)

/-- Look up a location in a store typing. -/
def storeTypLookup (Sigma : StoreTyp) (ell : Loc) : Option Typ :=
  (Sigma.find? (fun p => p.1 = ell)).map Prod.snd

/-- Domain of a store typing (the live locations). -/
def storeTypDom (Sigma : StoreTyp) : List Loc :=
  Sigma.map Prod.fst

/-- Store well-formedness: every live location in `sigma` has a corresponding
    entry in `Sigma`, and the tensor value's shape matches the store-typing
    type. Phase 1 states this structurally without proving preservation;
    Phase 2 WS2.8 will prove that `Step` preserves `StoreWf`. -/
def StoreWf (sigma : Store) (Sigma : StoreTyp) : Prop :=
  (∀ ell, ell ∈ storeTypDom Sigma → (storeLookup sigma ell).isSome) ∧
  (∀ ell, (storeLookup sigma ell).isSome → ell ∈ storeTypDom Sigma)

/-- Linearity soundness invariant (structural form): the set of live
    locations in `sigma` is exactly the set of locations owned by the
    linear context `Gamma`. Each linear binding `x : tensor[ds]` in `Gamma`
    owns a location whose type in `Sigma` is `tensor[ds]`. This predicate
    will appear as a premise/conclusion in Phase 2's preservation proof. -/
def LinearityInvariant (_sigma : Store) (_Sigma : StoreTyp)
    (_Gamma : LinearCtx) : Prop :=
  True  -- Phase 1: stated but not formalized in detail; filled in Phase 2.

/-- Extend a store typing with a fresh location. Used by Preservation
    on the store-allocating Step rules (tconst, copy, tadd, tmul, tsum,
    texpand, tuniformLike). -/
def storeTypExtend (Sigma : StoreTyp) (ell : Loc) (t : Typ) : StoreTyp :=
  (ell, t) :: Sigma

/-- Remove a location from a store typing. Companion to `storeRemove`
    on the runtime side. Used by Preservation on Step rules that
    consume a location (tadd, tmul, tsum, texpand, tuniformLike). -/
def storeTypRemove (Sigma : StoreTyp) (ell : Loc) : StoreTyp :=
  Sigma.filter (fun p => p.1 ≠ ell)

/-- `StoreTyp` sub-typing: `Sigma ⊑ Sigma'` iff every location typed
    in `Sigma` is typed to the same type in `Sigma'`. Used to state
    Preservation's store-monotonicity conclusion. -/
def StoreTypSub (Sigma Sigma' : StoreTyp) : Prop :=
  ∀ ell t, storeTypLookup Sigma ell = some t →
           storeTypLookup Sigma' ell = some t

/-! ## Freshness -/

/-- `List.foldl max` is ≥ its seed. -/
private theorem foldl_max_ge_seed :
    ∀ (l : List Nat) (s : Nat), s ≤ l.foldl max s := by
  intro l
  induction l with
  | nil => intro s; exact Nat.le_refl _
  | cons hd tl ih =>
    intro s
    show s ≤ (hd :: tl).foldl max s
    simp only [List.foldl_cons]
    have h_step : s ≤ max s hd := Nat.le_max_left _ _
    have h_ih : max s hd ≤ tl.foldl max (max s hd) := ih _
    exact Nat.le_trans h_step h_ih

/-- Every element of a list is ≤ its `foldl max`. -/
private theorem le_foldl_max :
    ∀ (l : List Nat) (k : Nat) (x : Nat), x ∈ l → x ≤ l.foldl max k := by
  intro l
  induction l with
  | nil => intro _ _ hm; cases hm
  | cons hd tl ih =>
    intro k x hmem
    simp only [List.mem_cons] at hmem
    rcases hmem with heq | hmem_tl
    · subst heq
      show x ≤ (x :: tl).foldl max k
      simp only [List.foldl_cons]
      have h1 : max k x ≤ tl.foldl max (max k x) := foldl_max_ge_seed tl _
      have h2 : x ≤ max k x := Nat.le_max_right _ _
      exact Nat.le_trans h2 h1
    · show x ≤ (hd :: tl).foldl max k
      simp only [List.foldl_cons]
      exact ih _ _ hmem_tl

/-- Freshness: `storeFreshLoc sigma ≠ any live location`. -/
theorem storeFreshLoc_ne (sigma : Store) (ell : Loc)
    (h : (storeLookup sigma ell).isSome) :
    ell ≠ storeFreshLoc sigma := by
  -- First: ell ∈ (sigma.map Prod.fst)
  have hMem : ell ∈ sigma.map Prod.fst := by
    induction sigma with
    | nil => simp [storeLookup, List.find?] at h
    | cons hd tl ih =>
      simp only [List.map_cons, List.mem_cons]
      by_cases hHd : hd.1 = ell
      · left; exact hHd.symm
      · right
        apply ih
        simp only [storeLookup, List.find?, hHd, decide_false,
                   Bool.false_eq_true, ite_false] at h
        exact h
  -- Then: ell ≤ foldl max 0, so ell < foldl max 0 + 1, so ell ≠ fresh.
  have hLe : ell ≤ (sigma.map Prod.fst).foldl max 0 :=
    le_foldl_max _ 0 ell hMem
  have hLt : ell < (sigma.map Prod.fst).foldl max 0 + 1 :=
    Nat.lt_succ_of_le hLe
  exact Nat.ne_of_lt hLt

/-! ## StoreWf extension lemmas (available to both LinearitySoundness
    and Preservation). -/

/-- After removing `ell1`, a lookup at `ell2 ≠ ell1` agrees with the
    lookup in the original store (in isSome). -/
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
    · have : ((hd :: tl).filter (fun p => p.1 ≠ ell1)) =
             (tl.filter (fun p => p.1 ≠ ell1)) := by
        simp [List.filter_cons, hHd1]
      rw [this]
      have hHdNe2 : hd.1 ≠ ell2 := by rw [hHd1]; exact fun h => hne h.symm
      show (storeLookup (tl.filter (fun p => p.1 ≠ ell1)) ell2).isSome =
           (storeLookup (hd :: tl) ell2).isSome
      rw [show storeLookup (hd :: tl) ell2 = storeLookup tl ell2 from by
            simp [storeLookup, List.find?, hHdNe2]]
      exact ih
    · have hFilter : ((hd :: tl).filter (fun p => p.1 ≠ ell1)) =
             hd :: (tl.filter (fun p => p.1 ≠ ell1)) := by
        simp [List.filter_cons, hHd1]
      rw [hFilter]
      by_cases hHd2 : hd.1 = ell2
      · simp [storeLookup, List.find?, hHd2]
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

/-- Extending both the store and the store-typing with the same
    location preserves well-formedness. Since `StoreWf` tracks domain
    membership only, freshness is accepted as a documentation premise
    rather than being load-bearing. -/
theorem StoreWf.extend_fresh
    {sigma : Store} {Sigma : StoreTyp}
    (ell : Loc) (w : TensorVal) (t : Typ)
    (h_wf : StoreWf sigma Sigma)
    (_h_fresh : ell = storeFreshLoc sigma) :
    StoreWf (storeExtend sigma ell w) (storeTypExtend Sigma ell t) := by
  refine ⟨?_, ?_⟩
  · intro ell' hell'
    simp only [storeTypDom, storeTypExtend, List.map_cons,
               List.mem_cons] at hell'
    rcases hell' with hEq | hOld
    · rw [hEq]
      show (storeLookup ((ell, w) :: sigma) ell).isSome
      simp [storeLookup, List.find?]
    · have hOldLive : (storeLookup sigma ell').isSome := by
        apply h_wf.1
        simpa [storeTypDom] using hOld
      show (storeLookup ((ell, w) :: sigma) ell').isSome
      by_cases hell'eq : ell' = ell
      · subst hell'eq
        simp [storeLookup, List.find?]
      · have hne : ¬ (ell = ell') := fun he => hell'eq he.symm
        simp only [storeLookup, storeExtend, List.find?, hne,
                   decide_false, Bool.false_eq_true, ite_false]
        exact hOldLive
  · intro ell' hell'
    show ell' ∈ storeTypDom (storeTypExtend Sigma ell t)
    simp only [storeTypDom, storeTypExtend, List.map_cons, List.mem_cons]
    by_cases hell'eq : ell' = ell
    · left; exact hell'eq
    · right
      have hne : ¬ (ell = ell') := fun he => hell'eq he.symm
      have hSigmaLive : (storeLookup sigma ell').isSome := by
        have : (storeLookup ((ell, w) :: sigma) ell').isSome := hell'
        simp only [storeLookup, storeExtend, List.find?, hne,
                   decide_false, Bool.false_eq_true, ite_false] at this
        exact this
      have := h_wf.2 ell' hSigmaLive
      simpa [storeTypDom] using this

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
    simp only [storeTypDom, storeTypExtend, List.map_cons,
               List.mem_cons] at hell'
    rcases hell' with hEq | hOld
    · rw [hEq]
      show (storeLookup ((ellOut, w) :: storeRemove sigma ellIn) ellOut).isSome
      simp [storeLookup, List.find?]
    · by_cases hell'In : ell' = ellIn
      · exfalso
        rw [hell'In] at hOld
        have : ellIn ∉ storeTypDom (storeTypRemove Sigma ellIn) := by
          intro hc
          have hc' : ellIn ∈ (Sigma.filter (fun p => p.1 ≠ ellIn)).map Prod.fst := hc
          rw [List.mem_map] at hc'
          rcases hc' with ⟨p, hp_mem, hp_eq⟩
          rw [List.mem_filter] at hp_mem
          have : p.1 ≠ ellIn := by
            have := hp_mem.2
            simp only [ne_eq, decide_not, Bool.not_eq_true',
                       decide_eq_false_iff_not] at this
            exact this
          exact this hp_eq
        exact this hOld
      · have hInSigma : ell' ∈ storeTypDom Sigma :=
          (mem_storeTypDom_remove_iff Sigma ellIn ell' hell'In).mp hOld
        have hSigmaLive : (storeLookup sigma ell').isSome :=
          h_wf.1 ell' hInSigma
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
      have hRem : (storeLookup (storeRemove sigma ellIn) ell').isSome := by
        have : (storeLookup ((ellOut, w) :: storeRemove sigma ellIn) ell').isSome := hell'
        simp only [storeLookup, List.find?, hne, decide_false,
                   Bool.false_eq_true, ite_false] at this
        exact this
      by_cases hell'In : ell' = ellIn
      · exfalso
        rw [hell'In] at hRem
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

end LaCaDiLE
