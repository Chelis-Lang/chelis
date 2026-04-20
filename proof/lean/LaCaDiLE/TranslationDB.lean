-- LaCaDiLE/TranslationDB.lean
--
-- Thin named-facing bridge onto the finished de Bruijn substitution
-- metatheory. The end-state for this module is:
--   1. erase the named substitution-dependent preservation redexes to DB,
--   2. discharge them with `SubstitutionDB`,
--   3. reflect the result back to named `HasType`,
--   4. keep `Preservation.lean` free of direct `SubstitutionDB` imports.
--
-- Two repository-level mismatches are still the real proof blockers:
--   * named typing is removal-style while DB typing is tombstone-style,
--     so reflection needs an explicit context correspondence relation;
--   * named syntax is lexical (`subst` stops at the nearest binder
--     name), but `HasType.var` can consume any same-named binding from
--     the linear context. That means term-only named→DB erasure is not
--     sound on arbitrary derivations; the completed bridge needs either
--     a scoping invariant or a derivation-guided erasure relation.
--   * named syntax uses `Dim` in `sum` / `expand`, while `SyntaxDB`
--     still uses positional `Nat` indices, so full erasure needs the
--     planned dim-representation bridge.
--
-- This file lands the bridge surface and the concrete scaffolding the
-- follow-up proof pass will use. The four named-facing wrapper lemmas
-- below are the only active substitution blockers that Preservation
-- should depend on.

import LaCaDiLE.Syntax
import LaCaDiLE.SyntaxDB
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.TypingDB
import LaCaDiLE.Operational
import LaCaDiLE.SubstitutionDB

namespace LaCaDiLE

/-- Binder names tracked innermost-first, matching DB index order. -/
abbrev BinderEnv := List String

/-- Lookup a binder name in the innermost-first environment. -/
def lookupBinder : BinderEnv → String → Option Nat
  | [], _ => none
  | y :: ys, x =>
      if y = x then
        some 0
      else
        Nat.succ <$> lookupBinder ys x

@[simp] theorem lookupBinder_head (ρ : BinderEnv) (x : String) :
    lookupBinder (x :: ρ) x = some 0 := by
  simp [lookupBinder]

@[simp] theorem lookupBinder_head_ne {ρ : BinderEnv} {x y : String}
    (h : y ≠ x) :
    lookupBinder (y :: ρ) x = Nat.succ <$> lookupBinder ρ x := by
  simp [lookupBinder, h]

@[simp] theorem lookupBinder_eq_none_of_not_mem
    {ρ : BinderEnv} {x : String}
    (h : x ∉ ρ) :
    lookupBinder ρ x = none := by
  induction ρ with
  | nil =>
      simp [lookupBinder]
  | cons y ys ih =>
      simp at h
      have hy : y ≠ x := by simpa [eq_comm] using h.1
      have hys : x ∉ ys := h.2
      simp [lookupBinder, hy, ih hys]

theorem lookupBinder_some_of_mem
    {ρ : BinderEnv} {x : String}
    (h : x ∈ ρ) :
    ∃ i, lookupBinder ρ x = some i := by
  induction ρ with
  | nil =>
      cases h
  | cons y ys ih =>
      rcases List.mem_cons.mp h with rfl | htl
      · exact ⟨0, by simp [lookupBinder]⟩
      · by_cases hy : y = x
        · subst hy
          exact ⟨0, by simp [lookupBinder]⟩
        · rcases ih htl with ⟨i, hi⟩
          exact ⟨i + 1, by simp [lookupBinder, hy, hi]⟩

theorem lookupBinder_append_left
    {ρ₁ ρ₂ : BinderEnv} {x : String} {i : Nat}
    (h : lookupBinder ρ₁ x = some i) :
    lookupBinder (ρ₁ ++ ρ₂) x = some i := by
  induction ρ₁ generalizing i with
  | nil =>
      cases h
  | cons y ys ih =>
      by_cases hy : y = x
      · subst hy
        have hi : i = 0 := by
          simpa [lookupBinder] using h.symm
        subst hi
        simp [lookupBinder]
      · cases hys : lookupBinder ys x with
        | none =>
            simp [lookupBinder, hy, hys] at h
        | some j =>
            simp [lookupBinder, hy, hys] at h
            cases h
            simpa [lookupBinder, hy, hys] using (ih hys)

theorem lookupBinder_append_of_none
    {ρ₁ ρ₂ : BinderEnv} {x : String}
    (h : lookupBinder ρ₁ x = none) :
    lookupBinder (ρ₁ ++ ρ₂) x = ((Nat.add ρ₁.length) <$> lookupBinder ρ₂ x) := by
  induction ρ₁ with
  | nil =>
      cases hρ₂ : lookupBinder ρ₂ x with
      | none =>
          simp [hρ₂]
      | some i =>
          simp [hρ₂]
  | cons y ys ih =>
      by_cases hy : y = x
      · subst hy
        simp [lookupBinder] at h
      · have hys : lookupBinder ys x = none := by
          simpa [lookupBinder, hy] using h
        simp [hy, ih hys]
        cases hρ₂ : lookupBinder ρ₂ x with
        | none =>
            simp
        | some i =>
            simp [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm]

theorem lookupBinder_append_target
    {ρin ρout : BinderEnv} {x : String}
    (hx : x ∉ ρin) :
    lookupBinder (ρin ++ x :: ρout) x = some ρin.length := by
  have hnone : lookupBinder ρin x = none := lookupBinder_eq_none_of_not_mem hx
  rw [lookupBinder_append_of_none hnone]
  simp [lookupBinder]

theorem lookupBinder_append_after_target
    {ρin ρout : BinderEnv} {x y : String}
    (hx : lookupBinder ρin y = none)
    (hxy : y ≠ x) :
    lookupBinder (ρin ++ x :: ρout) y =
      ((fun i => ρin.length + i + 1) <$> lookupBinder ρout y) := by
  rw [lookupBinder_append_of_none hx]
  have hxy' : x ≠ y := by
    intro h
    exact hxy h.symm
  cases hρ : lookupBinder ρout y with
  | none =>
      simp [lookupBinder, hxy', hρ]
  | some i =>
      simp [lookupBinder, hxy', hρ, Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

theorem lookupBinder_append_singleton_ne
    {ρ : BinderEnv} {x y : String}
    (hxy : y ≠ x) :
    lookupBinder (ρ ++ [x]) y = lookupBinder ρ y := by
  have hxy' : x ≠ y := by
    intro h
    exact hxy h.symm
  cases hρ : lookupBinder ρ y with
  | none =>
      rw [lookupBinder_append_of_none hρ]
      simp [lookupBinder, hxy']
  | some i =>
      exact (lookupBinder_append_left hρ).trans (by simp)

theorem lookupBinder_some_lt_length
    {ρ : BinderEnv} {x : String} {i : Nat}
    (h : lookupBinder ρ x = some i) :
    i < ρ.length := by
  induction ρ generalizing i with
  | nil =>
      cases h
  | cons y ys ih =>
      by_cases hy : y = x
      · subst hy
        have hi : i = 0 := by
          simpa [lookupBinder] using h.symm
        subst hi
        simp
      · cases hys : lookupBinder ys x with
        | none =>
            simp [lookupBinder, hy, hys] at h
        | some j =>
            simp [lookupBinder, hy, hys] at h
            cases h
            exact Nat.succ_lt_succ (ih hys)

/-- Active named bindings, erased to DB slots in innermost-first order. -/
def eraseCtx (Gamma : LinearCtx) : LinearCtxDB :=
  Gamma.reverse.map (fun p => some p.2)

@[simp] theorem eraseCtx_nil : eraseCtx [] = [] := rfl

/-- Binder names in innermost-first order, matching `eraseCtx`. -/
def ctxEnv (Gamma : LinearCtx) : BinderEnv :=
  (linearCtxDom Gamma).reverse

@[simp] theorem ctxEnv_nil : ctxEnv [] = [] := rfl

@[simp] theorem eraseCtx_append
    (Gamma1 Gamma2 : LinearCtx) :
    eraseCtx (Gamma1 ++ Gamma2) = eraseCtx Gamma2 ++ eraseCtx Gamma1 := by
  simp [eraseCtx, List.reverse_append, List.map_append]

@[simp] theorem eraseCtx_append_singleton
    (Gamma : LinearCtx) (x : String) (t : Typ) :
    eraseCtx (Gamma ++ [(x, t)]) = some t :: eraseCtx Gamma := by
  simpa [eraseCtx] using (eraseCtx_append Gamma [(x, t)])

@[simp] theorem eraseCtx_append_pair
    (Gamma : LinearCtx) (x : String) (tx : Typ) (y : String) (ty : Typ) :
    eraseCtx (Gamma ++ [(x, tx), (y, ty)]) = some ty :: some tx :: eraseCtx Gamma := by
  simpa [eraseCtx] using (eraseCtx_append Gamma [(x, tx), (y, ty)])

@[simp] theorem ctxEnv_append
    (Gamma1 Gamma2 : LinearCtx) :
    ctxEnv (Gamma1 ++ Gamma2) = ctxEnv Gamma2 ++ ctxEnv Gamma1 := by
  simp [ctxEnv, linearCtxDom, List.reverse_append, List.map_append]

@[simp] theorem ctxEnv_append_singleton
    (Gamma : LinearCtx) (x : String) (t : Typ) :
    ctxEnv (Gamma ++ [(x, t)]) = x :: ctxEnv Gamma := by
  simpa [ctxEnv, linearCtxDom] using (ctxEnv_append Gamma [(x, t)])

@[simp] theorem ctxEnv_append_pair
    (Gamma : LinearCtx) (x : String) (tx : Typ) (y : String) (ty : Typ) :
    ctxEnv (Gamma ++ [(x, tx), (y, ty)]) = y :: x :: ctxEnv Gamma := by
  simpa [ctxEnv, linearCtxDom] using (ctxEnv_append Gamma [(x, tx), (y, ty)])

@[simp] theorem mem_ctxEnv
    {Gamma : LinearCtx} {x : String} :
    x ∈ ctxEnv Gamma ↔ x ∈ linearCtxDom Gamma := by
  simp [ctxEnv]

@[simp] theorem length_ctxEnv
    (Gamma : LinearCtx) :
    (ctxEnv Gamma).length = Gamma.length := by
  simp [ctxEnv, linearCtxDom]

theorem lookupBinder_ctxEnv_append_target
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hx : x ∉ linearCtxDom GammaPost) :
    lookupBinder (ctxEnv (GammaPre ++ [(x, t)] ++ GammaPost)) x =
      some GammaPost.length := by
  have hx' : x ∉ ctxEnv GammaPost := by
    simpa using hx
  have hshape :
      ctxEnv (GammaPre ++ [(x, t)] ++ GammaPost) =
        ctxEnv GammaPost ++ [x] ++ ctxEnv GammaPre := by
    simp [ctxEnv, linearCtxDom, List.reverse_append, List.reverse_cons, List.append_assoc]
  rw [hshape]
  simpa [List.append_assoc] using
    (lookupBinder_append_target (ρin := ctxEnv GammaPost) (ρout := ctxEnv GammaPre) hx')

/-- Removal-style named contexts correspond to tombstone-style DB
    contexts by dropping dead DB slots and forgetting names from the
    live ones. The DB head is the innermost live slot. -/
inductive CtxCorr : LinearCtx → LinearCtxDB → Prop where
  | nil :
      CtxCorr [] []
  | dead
      {Gamma : LinearCtx} {GammaDB : LinearCtxDB} :
      CtxCorr Gamma GammaDB →
      CtxCorr Gamma (none :: GammaDB)
  | live
      {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
      {x : String} {t : Typ} :
      CtxCorr Gamma GammaDB →
      CtxCorr (Gamma ++ [(x, t)]) (some t :: GammaDB)

theorem ctxCorr_eraseCtx
    (Gamma : LinearCtx) :
    CtxCorr Gamma (eraseCtx Gamma) := by
  have hrev : ∀ r : LinearCtx, CtxCorr r.reverse (eraseCtx r.reverse) := by
    intro r
    induction r with
    | nil =>
        simp [eraseCtx]
        exact CtxCorr.nil
    | cons p ps ih =>
        cases p with
        | mk x t =>
            simpa [List.reverse_cons, eraseCtx_append_singleton] using
              (CtxCorr.live (x := x) (t := t) ih)
  simpa using hrev Gamma.reverse

theorem ctxCorr_append_eraseCtx
    {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
    (h : CtxCorr Gamma GammaDB) :
    ∀ tail : LinearCtx, CtxCorr (Gamma ++ tail) (eraseCtx tail ++ GammaDB) := by
  intro tail
  have hrev : ∀ r : LinearCtx, CtxCorr (Gamma ++ r.reverse) (eraseCtx r.reverse ++ GammaDB) := by
    intro r
    induction r with
    | nil =>
        simp [eraseCtx]
        simpa using h
    | cons p ps ih =>
        cases p with
        | mk x t =>
            simpa [List.reverse_cons, List.append_assoc, eraseCtx_append_singleton] using
              (CtxCorr.live (x := x) (t := t) ih)
  simpa using hrev tail.reverse

theorem ctxCorr_length_le
    {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
    (h : CtxCorr Gamma GammaDB) :
    Gamma.length ≤ GammaDB.length := by
  induction h with
  | nil =>
      simp
  | dead _ ih =>
      exact Nat.le_trans ih (Nat.le_succ _)
  | live _ ih =>
      simpa [List.length_append] using Nat.succ_le_succ ih

theorem ctxCorr_length_eq_eraseCtx
    {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
    (h : CtxCorr Gamma GammaDB)
    (hlen : Gamma.length = GammaDB.length) :
    GammaDB = eraseCtx Gamma := by
  revert hlen
  induction h with
  | nil =>
      intro hlen
      simp [eraseCtx]
  | dead h ih =>
      intro hlen
      have hle := ctxCorr_length_le h
      simp at hlen
      omega
  | @live Gamma GammaDB x t h ih =>
      intro hlen
      have hlen' : Gamma.length = GammaDB.length := by
        simpa [List.length_append] using hlen
      simpa [eraseCtx] using congrArg (fun db => some t :: db) (ih hlen')

theorem ctxCorr_singleton_length_one
    {x : String} {t : Typ} {GammaDB : LinearCtxDB}
    (h : CtxCorr ([(x, t)] : LinearCtx) GammaDB)
    (hlen : GammaDB.length = 1) :
    GammaDB = [some t] := by
  have hlen' : ([(x, t)] : LinearCtx).length = GammaDB.length := by
    simpa using hlen.symm
  simpa [eraseCtx] using ctxCorr_length_eq_eraseCtx h hlen'

theorem noDupNames_middle_fresh_suffix
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, t)] ++ GammaPost)) :
    x ∉ linearCtxDom GammaPost := by
  intro hx
  unfold NoDupNames at hnd
  simp [linearCtxDom, List.nodup_append] at hnd
  have hx' : ∃ ty, (x, ty) ∈ GammaPost := by
    simpa [linearCtxDom] using hx
  rcases hx' with ⟨ty, hmem⟩
  exact hnd.2.1.1 ty hmem

private theorem filter_ctx_eq_self_of_fresh
    {Gamma : LinearCtx} {x : String}
    (hx : x ∉ linearCtxDom Gamma) :
    Gamma.filter (fun p => p.1 ≠ x) = Gamma := by
  apply List.filter_eq_self.2
  intro p hp
  have hp' : p.1 ≠ x := by
    intro hpx
    apply hx
    unfold linearCtxDom
    exact List.mem_map.mpr ⟨p, hp, hpx⟩
  simp [hp']

theorem eraseCtx_consume_target
    (GammaPre GammaPost : LinearCtx) (x : String) (t : Typ) :
    (eraseCtx (GammaPre ++ [(x, t)] ++ GammaPost)).set GammaPost.length none =
      eraseCtx GammaPost ++ none :: eraseCtx GammaPre := by
  have hshape :
      eraseCtx (GammaPre ++ [(x, t)] ++ GammaPost) =
        eraseCtx GammaPost ++ some t :: eraseCtx GammaPre := by
    simp [eraseCtx, List.reverse_append, List.reverse_cons, List.map_append, List.append_assoc]
  rw [hshape]
  rw [LinearCtxDB.set_append_ge (eraseCtx GammaPost) (some t :: eraseCtx GammaPre)
        GammaPost.length none (by simp [eraseCtx])]
  simp [eraseCtx]

theorem ctxCorr_consume_target
    (GammaPre GammaPost : LinearCtx) (x : String) (t : Typ) :
    CtxCorr (GammaPre ++ GammaPost)
      ((eraseCtx (GammaPre ++ [(x, t)] ++ GammaPost)).set GammaPost.length none) := by
  rw [eraseCtx_consume_target GammaPre GammaPost x t]
  exact ctxCorr_append_eraseCtx (CtxCorr.dead (ctxCorr_eraseCtx GammaPre)) GammaPost

theorem eraseCtx_get_target
    (GammaPre GammaPost : LinearCtx) (x : String) (t : Typ) :
    (eraseCtx (GammaPre ++ [(x, t)] ++ GammaPost))[GammaPost.length]? = some (some t) := by
  have hshape :
      eraseCtx (GammaPre ++ [(x, t)] ++ GammaPost) =
        eraseCtx GammaPost ++ some t :: eraseCtx GammaPre := by
    simp [eraseCtx, List.reverse_append, List.reverse_cons, List.map_append, List.append_assoc]
  rw [hshape]
  rw [LinearCtxDB.getElem?_append_ge (eraseCtx GammaPost) (some t :: eraseCtx GammaPre)
        GammaPost.length (by simp [eraseCtx])]
  simp [eraseCtx]

theorem ctxCorr_singleton_live
    (x : String) (t : Typ) :
    CtxCorr ([(x, t)] : LinearCtx) [some t] := by
  simpa [eraseCtx] using (ctxCorr_eraseCtx ([(x, t)] : LinearCtx))

theorem ctxCorr_singleton_dead
    (_x : String) (_t : Typ) :
    CtxCorr ([] : LinearCtx) [none] := by
  exact CtxCorr.dead CtxCorr.nil

theorem ctxCorr_pair_live
    (x : String) (tx : Typ) (y : String) (ty : Typ) :
    CtxCorr ([(x, tx), (y, ty)] : LinearCtx) [some ty, some tx] := by
  simpa [eraseCtx] using (ctxCorr_eraseCtx ([(x, tx), (y, ty)] : LinearCtx))

theorem ctxCorr_pair_left_live
    (x : String) (tx : Typ) (_y : String) (_ty : Typ) :
    CtxCorr ([(x, tx)] : LinearCtx) [none, some tx] := by
  exact CtxCorr.dead (ctxCorr_singleton_live x tx)

theorem ctxCorr_pair_right_live
    (_x : String) (_tx : Typ) (y : String) (ty : Typ) :
    CtxCorr ([(y, ty)] : LinearCtx) [some ty, none] := by
  exact CtxCorr.live (x := y) (t := ty) (CtxCorr.dead CtxCorr.nil)

theorem ctxCorr_pair_dead
    (_x : String) (_tx : Typ) (_y : String) (_ty : Typ) :
    CtxCorr ([] : LinearCtx) [none, none] := by
  exact CtxCorr.dead (CtxCorr.dead CtxCorr.nil)

theorem ctxCorr_dead_singleton_inv
    {Gamma : LinearCtx}
    (h : CtxCorr Gamma [none]) :
    Gamma = [] := by
  cases h with
  | dead h' =>
      cases h' with
      | nil => rfl

theorem ctxCorr_live_singleton_inv
    {Gamma : LinearCtx} {t : Typ}
    (h : CtxCorr Gamma [some t]) :
    ∃ x, Gamma = [(x, t)] := by
  cases h with
  | live h' =>
      cases h' with
      | nil =>
          exact ⟨_, rfl⟩

theorem ctxCorr_pair_dead_inv
    {Gamma : LinearCtx}
    (h : CtxCorr Gamma [none, none]) :
    Gamma = [] := by
  cases h with
  | dead h' =>
      exact ctxCorr_dead_singleton_inv h'

theorem ctxCorr_pair_left_live_inv
    {Gamma : LinearCtx} {tx : Typ}
    (h : CtxCorr Gamma [none, some tx]) :
    ∃ x, Gamma = [(x, tx)] := by
  cases h with
  | dead h' =>
      exact ctxCorr_live_singleton_inv h'

theorem ctxCorr_pair_right_live_inv
    {Gamma : LinearCtx} {ty : Typ}
    (h : CtxCorr Gamma [some ty, none]) :
    ∃ y, Gamma = [(y, ty)] := by
  cases h with
  | live h' =>
      cases h' with
      | dead h'' =>
          cases h'' with
          | nil =>
              exact ⟨_, rfl⟩

theorem ctxCorr_pair_live_inv
    {Gamma : LinearCtx} {tx ty : Typ}
    (h : CtxCorr Gamma [some ty, some tx]) :
    ∃ x y, Gamma = [(x, tx), (y, ty)] := by
  cases h with
  | live h' =>
      cases h' with
      | live h'' =>
          cases h'' with
          | nil =>
              exact ⟨_, _, rfl⟩

/-- Tombstone-aware binder environment: live slots carry both the
    source name and type, dead slots remain positionally present as
    `none`. This is the environment shape needed to erase named terms
    directly against DB middle contexts without compacting out consumed
    slots. -/
abbrev BinderSlots := List (Option (String × Typ))

/-- Forget binder names from a tombstone-aware environment, yielding the
    underlying DB linear context. -/
def binderSlotTypes : BinderSlots → LinearCtxDB
  | [] => []
  | none :: rest => none :: binderSlotTypes rest
  | some (_, t) :: rest => some t :: binderSlotTypes rest

/-- Forget binder types from a tombstone-aware environment, keeping only
    the tombstone-sensitive name layout used by erasure. -/
def binderSlotNames : BinderSlots → List (Option String)
  | [] => []
  | none :: rest => none :: binderSlotNames rest
  | some (x, _) :: rest => some x :: binderSlotNames rest

@[simp] theorem binderSlotTypes_set_none
    (rho : BinderSlots) (i : Nat) :
    binderSlotTypes (rho.set i none) = (binderSlotTypes rho).set i none := by
  induction rho generalizing i with
  | nil =>
      cases i <;> simp [List.set, binderSlotTypes]
  | cons hd tl ih =>
      cases i with
      | zero =>
          cases hd <;> simp [List.set, binderSlotTypes]
      | succ j =>
          cases hd <;> simp [List.set, binderSlotTypes, ih]

@[simp] theorem binderSlotNames_set_none
    (rho : BinderSlots) (i : Nat) :
    binderSlotNames (rho.set i none) = (binderSlotNames rho).set i none := by
  induction rho generalizing i with
  | nil =>
      cases i <;> simp [List.set, binderSlotNames]
  | cons hd tl ih =>
      cases i with
      | zero =>
          cases hd <;> simp [List.set, binderSlotNames]
      | succ j =>
          cases hd <;> simp [List.set, binderSlotNames, ih]

/-- Lookup a binder name in a tombstone-aware environment. Dead slots
    still count toward the returned de Bruijn index. -/
def lookupBinderSlots : BinderSlots → String → Option Nat
  | [], _ => none
  | none :: rest, x =>
      Nat.succ <$> lookupBinderSlots rest x
  | some (y, _) :: rest, x =>
      if y = x then some 0 else Nat.succ <$> lookupBinderSlots rest x

/-- Name-aware analogue of `CtxCorr`: preserves the precise live-slot
    names needed by tombstone-aware erasure. -/
inductive SlotCorr : LinearCtx → BinderSlots → Prop where
  | nil :
      SlotCorr [] []
  | dead
      {Gamma : LinearCtx} {rho : BinderSlots} :
      SlotCorr Gamma rho →
      SlotCorr Gamma (none :: rho)
  | live
      {Gamma : LinearCtx} {rho : BinderSlots}
      {x : String} {t : Typ} :
      SlotCorr Gamma rho →
      SlotCorr (Gamma ++ [(x, t)]) (some (x, t) :: rho)

/-- Output binder slots are obtained from the input slots only by
    consuming live bindings into tombstones. Names, types, order, and
    length are preserved positionally. This is the invariant the bridge
    needs to peel binder heads after recursive DB transport. -/
inductive SlotKillsOnly : BinderSlots → BinderSlots → Prop where
  | nil :
      SlotKillsOnly [] []
  | dead_dead
      {rhoOut rhoIn : BinderSlots} :
      SlotKillsOnly rhoOut rhoIn →
      SlotKillsOnly (none :: rhoOut) (none :: rhoIn)
  | dead_live
      {rhoOut rhoIn : BinderSlots} {x : String} {t : Typ} :
      SlotKillsOnly rhoOut rhoIn →
      SlotKillsOnly (none :: rhoOut) (some (x, t) :: rhoIn)
  | live
      {rhoOut rhoIn : BinderSlots} {x : String} {t : Typ} :
      SlotKillsOnly rhoOut rhoIn →
      SlotKillsOnly (some (x, t) :: rhoOut) (some (x, t) :: rhoIn)

theorem slotKillsOnly_refl
    (rho : BinderSlots) :
    SlotKillsOnly rho rho := by
  induction rho with
  | nil =>
      exact SlotKillsOnly.nil
  | cons hd tl ih =>
      cases hd with
      | none =>
          exact SlotKillsOnly.dead_dead ih
      | some p =>
          cases p with
          | mk x t =>
              exact SlotKillsOnly.live ih

theorem slotKillsOnly_trans
    {rho1 rho2 rho3 : BinderSlots}
    (h12 : SlotKillsOnly rho1 rho2)
    (h23 : SlotKillsOnly rho2 rho3) :
    SlotKillsOnly rho1 rho3 := by
  induction h12 generalizing rho3 with
  | nil =>
      cases h23
      exact SlotKillsOnly.nil
  | dead_dead h12 ih =>
      cases h23 with
      | dead_dead h23 =>
          exact SlotKillsOnly.dead_dead (ih h23)
      | dead_live h23 =>
          exact SlotKillsOnly.dead_live (ih h23)
  | dead_live h12 ih =>
      cases h23 with
      | live h23 =>
          exact SlotKillsOnly.dead_live (ih h23)
  | live h12 ih =>
      cases h23 with
      | live h23 =>
          exact SlotKillsOnly.live (ih h23)

theorem slotKillsOnly_length
    {rhoOut rhoIn : BinderSlots}
    (h : SlotKillsOnly rhoOut rhoIn) :
    rhoOut.length = rhoIn.length := by
  induction h with
  | nil =>
      rfl
  | dead_dead _ ih =>
      simpa [ih]
  | dead_live _ ih =>
      simpa [ih]
  | live _ ih =>
      simpa [ih]

theorem slotKillsOnly_cons_inv
    {rhoOut rhoIn : BinderSlots} {x : String} {t : Typ}
    (h : SlotKillsOnly rhoOut (some (x, t) :: rhoIn)) :
    ∃ slot rhoTail,
      rhoOut = slot :: rhoTail ∧
      SlotKillsOnly rhoTail rhoIn ∧
      (slot = none ∨ slot = some (x, t)) := by
  cases h with
  | dead_live htail =>
      exact ⟨none, _, rfl, htail, Or.inl rfl⟩
  | live htail =>
      exact ⟨some (x, t), _, rfl, htail, Or.inr rfl⟩

/-- Deterministic output-slot projection: keep exactly the live input
    slots whose named bindings survive in the target named context,
    tombstoning the rest in place. This gives a canonical DB-side
    middle context for a known named output context. -/
def restrictSlots (Gamma : LinearCtx) : BinderSlots → BinderSlots
  | [] => []
  | none :: rho => none :: restrictSlots Gamma rho
  | some p :: rho =>
      (if p.1 ∈ linearCtxDom Gamma then some p else none) :: restrictSlots Gamma rho

theorem restrictSlots_killsOnly
    {Gamma : LinearCtx} {rho : BinderSlots} :
    SlotKillsOnly (restrictSlots Gamma rho) rho := by
  induction rho with
  | nil =>
      simp [restrictSlots, SlotKillsOnly.nil]
  | cons hd tl ih =>
      cases hd with
      | none =>
          simpa [restrictSlots] using SlotKillsOnly.dead_dead ih
      | some p =>
          by_cases hmem : p.1 ∈ linearCtxDom Gamma
          · simpa [restrictSlots, hmem] using SlotKillsOnly.live ih
          · simpa [restrictSlots, hmem] using SlotKillsOnly.dead_live (x := p.1) (t := p.2) ih

theorem slotCorr_binderSlotTypes
    {Gamma : LinearCtx} {rho : BinderSlots}
    (h : SlotCorr Gamma rho) :
    CtxCorr Gamma (binderSlotTypes rho) := by
  induction h with
  | nil =>
      exact CtxCorr.nil
  | dead _ ih =>
      exact CtxCorr.dead ih
  | live _ ih =>
      exact CtxCorr.live ih

/-- Fully-live tombstone-aware environment corresponding to a named
    linear context. -/
def eraseBinderSlots (Gamma : LinearCtx) : BinderSlots :=
  Gamma.reverse.map some

@[simp] theorem eraseBinderSlots_nil :
    eraseBinderSlots [] = [] := rfl

@[simp] theorem binderSlotTypes_map_some
    (xs : List (String × Typ)) :
    binderSlotTypes (xs.map some) = xs.map (fun p => some p.2) := by
  induction xs with
  | nil =>
      simp [binderSlotTypes]
  | cons p ps ih =>
      cases p
      simp [binderSlotTypes, ih]

@[simp] theorem binderSlotNames_map_some
    (xs : List (String × Typ)) :
    binderSlotNames (xs.map some) = xs.map (fun p => some p.1) := by
  induction xs with
  | nil =>
      simp [binderSlotNames]
  | cons p ps ih =>
      cases p
      simp [binderSlotNames, ih]

@[simp] theorem eraseBinderSlots_append
    (Gamma1 Gamma2 : LinearCtx) :
    eraseBinderSlots (Gamma1 ++ Gamma2) =
      eraseBinderSlots Gamma2 ++ eraseBinderSlots Gamma1 := by
  simp [eraseBinderSlots, List.reverse_append, List.map_append]

@[simp] theorem binderSlotTypes_eraseBinderSlots
    (Gamma : LinearCtx) :
    binderSlotTypes (eraseBinderSlots Gamma) = eraseCtx Gamma := by
  simpa [eraseBinderSlots, eraseCtx] using
    (binderSlotTypes_map_some Gamma.reverse)

@[simp] theorem binderSlotNames_eraseBinderSlots
    (Gamma : LinearCtx) :
    binderSlotNames (eraseBinderSlots Gamma) = (ctxEnv Gamma).map some := by
  simpa [eraseBinderSlots, ctxEnv, linearCtxDom] using
    (binderSlotNames_map_some Gamma.reverse)

theorem slotCorr_eraseBinderSlots
    (Gamma : LinearCtx) :
    SlotCorr Gamma (eraseBinderSlots Gamma) := by
  have hrev : ∀ r : LinearCtx, SlotCorr r.reverse (eraseBinderSlots r.reverse) := by
    intro r
    induction r with
    | nil =>
        simp [eraseBinderSlots]
        exact SlotCorr.nil
    | cons p ps ih =>
        cases p with
        | mk x t =>
            simpa [List.reverse_cons, eraseBinderSlots_append] using
              (SlotCorr.live (x := x) (t := t) ih)
  simpa using hrev Gamma.reverse

@[simp] theorem lookupBinderSlots_dead
    {rho : BinderSlots} {x : String} :
    lookupBinderSlots (none :: rho) x = Nat.succ <$> lookupBinderSlots rho x := rfl

@[simp] theorem lookupBinderSlots_live_head
    {rho : BinderSlots} {x : String} {t : Typ} :
    lookupBinderSlots (some (x, t) :: rho) x = some 0 := by
  simp [lookupBinderSlots]

@[simp] theorem lookupBinderSlots_live_head_ne
    {rho : BinderSlots} {x y : String} {t : Typ}
    (h : y ≠ x) :
    lookupBinderSlots (some (y, t) :: rho) x = Nat.succ <$> lookupBinderSlots rho x := by
  simp [lookupBinderSlots, h]

theorem lookupBinderSlots_none_of_fresh
    {Gamma : LinearCtx} {rho : BinderSlots} {x : String}
    (hslot : SlotCorr Gamma rho)
    (hx : x ∉ linearCtxDom Gamma) :
    lookupBinderSlots rho x = none := by
  induction hslot with
  | nil =>
      simp [lookupBinderSlots]
  | dead hprev ih =>
      simpa [lookupBinderSlots] using ih hx
  | @live Gamma rho y t hprev ih =>
      have hxPrev : x ∉ linearCtxDom Gamma := by
        intro hmem
        apply hx
        unfold linearCtxDom at hmem ⊢
        exact List.mem_map.mpr <| by
          rcases List.mem_map.mp hmem with ⟨p, hp, rfl⟩
          exact ⟨p, by simpa [List.mem_append] using Or.inl hp, rfl⟩
      have hxy : y ≠ x := by
        intro hEq
        apply hx
        subst hEq
        unfold linearCtxDom
        exact List.mem_map.mpr ⟨(y, t), by simp [List.mem_append], rfl⟩
      simpa [lookupBinderSlots, hxy] using ih hxPrev

theorem fresh_of_lookupBinderSlots_none
    {Gamma : LinearCtx} {rho : BinderSlots} {x : String}
    (hslot : SlotCorr Gamma rho)
    (hlook : lookupBinderSlots rho x = none) :
    x ∉ linearCtxDom Gamma := by
  induction hslot with
  | nil =>
      simp [linearCtxDom]
  | dead hprev ih =>
      exact ih (by simpa [lookupBinderSlots] using hlook)
  | @live Gamma rho y t hprev ih =>
      have hxy : y ≠ x := by
        intro hEq
        subst hEq
        simp [lookupBinderSlots] at hlook
      have htail : lookupBinderSlots rho x = none := by
        simpa [lookupBinderSlots, hxy] using hlook
      intro hmem
      have hfreshPrev := ih htail
      simp [linearCtxDom, List.mem_append] at hmem
      rcases hmem with hmem | hmem
      · exact hfreshPrev (by simpa [linearCtxDom] using hmem)
      · exact hxy hmem.symm

theorem slotKillsOnly_lookup_none
    {rhoOut rhoIn : BinderSlots} {x : String}
    (hkill : SlotKillsOnly rhoOut rhoIn)
    (hlook : lookupBinderSlots rhoIn x = none) :
    lookupBinderSlots rhoOut x = none := by
  revert hlook
  induction hkill with
  | nil =>
      intro hlook
      simpa [lookupBinderSlots] using hlook
  | dead_dead htail ih =>
      intro hlook
      have htailLook := by
        simpa [lookupBinderSlots] using hlook
      simpa [lookupBinderSlots] using ih htailLook
  | @dead_live rhoOut rhoIn y t htail ih =>
      intro hlook
      have hxy : y ≠ x := by
        intro hEq
        subst hEq
        simp [lookupBinderSlots] at hlook
      have htailLook : lookupBinderSlots rhoIn x = none := by
        simpa [lookupBinderSlots, hxy] using hlook
      simpa [lookupBinderSlots] using ih htailLook
  | @live rhoOut rhoIn y t htail ih =>
      intro hlook
      have hxy : y ≠ x := by
        intro hEq
        subst hEq
        simp [lookupBinderSlots] at hlook
      have htailLook : lookupBinderSlots rhoIn x = none := by
        simpa [lookupBinderSlots, hxy] using hlook
      simpa [lookupBinderSlots, hxy] using ih htailLook

theorem slotKillsOnly_set_none
    (rho : BinderSlots) (i : Nat) :
    SlotKillsOnly (rho.set i none) rho := by
  induction rho generalizing i with
  | nil =>
      cases i <;> simp [List.set, SlotKillsOnly.nil]
  | cons hd tl ih =>
      cases i with
      | zero =>
          cases hd with
          | none =>
              simp [List.set, slotKillsOnly_refl, SlotKillsOnly.dead_dead]
          | some p =>
              cases p with
              | mk x t =>
                  simp [List.set, slotKillsOnly_refl, SlotKillsOnly.dead_live]
      | succ j =>
          cases hd with
          | none =>
              simpa [List.set] using (SlotKillsOnly.dead_dead (ih j))
          | some p =>
              cases p with
              | mk x t =>
                  simpa [List.set] using (SlotKillsOnly.live (ih j))

theorem slotCorr_dead_inv
    {Gamma : LinearCtx} {rho : BinderSlots}
    (hslot : SlotCorr Gamma (none :: rho)) :
    SlotCorr Gamma rho := by
  cases hslot with
  | dead hprev =>
      exact hprev

theorem slotCorr_live_inv
    {Gamma : LinearCtx} {rho : BinderSlots} {x : String} {t : Typ}
    (hslot : SlotCorr Gamma (some (x, t) :: rho)) :
    ∃ GammaPre,
      Gamma = GammaPre ++ [(x, t)] ∧
      SlotCorr GammaPre rho := by
  cases hslot with
  | live hprev =>
      exact ⟨_, rfl, hprev⟩

theorem slotCorr_peel_single
    {GammaBody Gamma : LinearCtx}
    {rhoBody rho : BinderSlots}
    {x : String} {t : Typ}
    (hslotBody : SlotCorr GammaBody rhoBody)
    (hkill : SlotKillsOnly rhoBody (some (x, t) :: rho))
    (hslot : SlotCorr Gamma rho)
    (hx : x ∉ linearCtxDom Gamma) :
    ∃ slot rhoTail,
      rhoBody = slot :: rhoTail ∧
      SlotCorr (GammaBody.filter (fun p => p.1 ≠ x)) rhoTail ∧
      SlotKillsOnly rhoTail rho := by
  rcases slotKillsOnly_cons_inv hkill with ⟨slot, rhoTail, hrho, htail, hslotHead⟩
  refine ⟨slot, rhoTail, hrho, ?_, htail⟩
  rcases hslotHead with rfl | rfl
  · rw [hrho] at hslotBody
    have hprev : SlotCorr GammaBody rhoTail := slotCorr_dead_inv hslotBody
    have hlookNone : lookupBinderSlots rho x = none :=
      lookupBinderSlots_none_of_fresh hslot hx
    have hlookTail : lookupBinderSlots rhoTail x = none :=
      slotKillsOnly_lookup_none htail hlookNone
    have hxBody : x ∉ linearCtxDom GammaBody :=
      fresh_of_lookupBinderSlots_none hprev hlookTail
    have hfilter : GammaBody.filter (fun p => p.1 ≠ x) = GammaBody :=
      filter_ctx_eq_self_of_fresh hxBody
    rw [hfilter]
    exact hprev
  · rw [hrho] at hslotBody
    rcases slotCorr_live_inv hslotBody with ⟨GammaPre, rfl, hprev⟩
    have hlookNone : lookupBinderSlots rho x = none :=
      lookupBinderSlots_none_of_fresh hslot hx
    have hlookTail : lookupBinderSlots rhoTail x = none :=
      slotKillsOnly_lookup_none htail hlookNone
    have hxPre : x ∉ linearCtxDom GammaPre :=
      fresh_of_lookupBinderSlots_none hprev hlookTail
    have hfilter : GammaPre.filter (fun p => p.1 ≠ x) = GammaPre :=
      filter_ctx_eq_self_of_fresh hxPre
    rw [show (GammaPre ++ [(x, t)]).filter (fun p => p.1 ≠ x) = GammaPre.filter (fun p => p.1 ≠ x) by
      simp [List.filter_append]]
    rw [hfilter]
    exact hprev

theorem slotCorr_peel_two
    {GammaBody Gamma : LinearCtx}
    {rhoBody rho : BinderSlots}
    {x y : String} {tx ty : Typ}
    (hslotBody : SlotCorr GammaBody rhoBody)
    (hkill : SlotKillsOnly rhoBody (some (y, ty) :: some (x, tx) :: rho))
    (hslot : SlotCorr Gamma rho)
    (hx : x ∉ linearCtxDom Gamma)
    (hy : y ∉ linearCtxDom Gamma)
    (hxy : x ≠ y) :
    ∃ slotY slotX rhoTail,
      rhoBody = slotY :: slotX :: rhoTail ∧
      SlotCorr (GammaBody.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y)) rhoTail ∧
      SlotKillsOnly rhoTail rho := by
  have hslotX : SlotCorr (Gamma ++ [(x, tx)]) (some (x, tx) :: rho) :=
    SlotCorr.live hslot
  have hyX : y ∉ linearCtxDom (Gamma ++ [(x, tx)]) := by
    intro hmem
    simp [linearCtxDom, List.mem_append] at hmem
    rcases hmem with hmem | hmem
    · exact hy (by simpa [linearCtxDom] using hmem)
    · exact hxy hmem.symm
  rcases slotCorr_peel_single hslotBody hkill hslotX hyX with
    ⟨slotY, rhoMid, hrhoY, hslotMid, hkillMid⟩
  rcases slotCorr_peel_single hslotMid hkillMid hslot hx with
    ⟨slotX, rhoTail, hrhoX, hslotTail, hkillTail⟩
  refine ⟨slotY, slotX, rhoTail, ?_, ?_, hkillTail⟩
  · rw [hrhoY, hrhoX]
  · simpa [List.filter_filter, and_left_comm, and_assoc] using hslotTail

/-- Temporary bridge from named dimensions to DB positional slots.
    This is intentionally a placeholder until the Phase 1 dim-index
    representation change is mirrored into `SyntaxDB`. The substitution
    redexes routed through this module do not inspect the payload. -/
def eraseDim (_d : Dim) : Nat := 0

mutual

/-- Syntax erasure from named terms into DB terms relative to an
    innermost-first binder environment. Returns `none` when a free
    variable is not represented in the environment. -/
def eraseTerm (rho : BinderEnv) : Term → Option TermDB
  | Term.var x =>
      TermDB.var <$> lookupBinder rho x
  | Term.abs x t body =>
      TermDB.abs t <$> eraseTerm (x :: rho) body
  | Term.app e1 e2 => do
      let e1' <- eraseTerm rho e1
      let e2' <- eraseTerm rho e2
      pure (TermDB.app e1' e2')
  | Term.letBind x e1 e2 => do
      let e1' <- eraseTerm rho e1
      let e2' <- eraseTerm (x :: rho) e2
      pure (TermDB.letBind e1' e2')
  | Term.copy e =>
      TermDB.copy <$> eraseTerm rho e
  | Term.letpair x y e1 e2 => do
      let e1' <- eraseTerm rho e1
      let e2' <- eraseTerm (y :: x :: rho) e2
      pure (TermDB.letpair e1' e2')
  | Term.pair e1 e2 => do
      let e1' <- eraseTerm rho e1
      let e2' <- eraseTerm rho e2
      pure (TermDB.pair e1' e2')
  | Term.fst e =>
      TermDB.fst <$> eraseTerm rho e
  | Term.snd e =>
      TermDB.snd <$> eraseTerm rho e
  | Term.unit =>
      pure TermDB.unit
  | Term.const v ds =>
      pure (TermDB.const v ds)
  | Term.add e1 e2 => do
      let e1' <- eraseTerm rho e1
      let e2' <- eraseTerm rho e2
      pure (TermDB.add e1' e2')
  | Term.mul e1 e2 => do
      let e1' <- eraseTerm rho e1
      let e2' <- eraseTerm rho e2
      pure (TermDB.mul e1' e2')
  | Term.sum e d =>
      TermDB.sum <$> eraseTerm rho e <*> pure (eraseDim d)
  | Term.expand e d =>
      TermDB.expand <$> eraseTerm rho e <*> pure (eraseDim d) <*> pure 0
  | Term.uniformLike e lo hi =>
      TermDB.uniformLike <$> eraseTerm rho e <*> pure lo <*> pure hi
  | Term.grad x t tOut body =>
      TermDB.grad t tOut <$> eraseTerm (x :: rho) body
  | Term.vmap x t body =>
      TermDB.vmap t <$> eraseTerm (x :: rho) body
  | Term.handle epsH body clauses => do
      let body' <- eraseTerm rho body
      let clauses' <- eraseClauses rho clauses
      pure (TermDB.handle epsH body' clauses')
  | Term.perform op e =>
      TermDB.perform op <$> eraseTerm rho e
  | Term.loc ell =>
      pure (TermDB.loc ell)

/-- Clause erasure. Handler bodies see continuation at DB index `0`
    and operation argument at DB index `1`, so the environment order is
    `k :: x :: rho`. -/
def eraseClauses
    (rho : BinderEnv) :
    List (EffectLabel × String × String × Term) →
    Option (List (EffectLabel × TermDB))
  | [] =>
      pure []
  | (op, x, k, hb) :: rest => do
      let hb' <- eraseTerm (k :: x :: rho) hb
      let rest' <- eraseClauses rho rest
      pure ((op, hb') :: rest')

end

/-- Tombstone-aware binder-name environment used for the next transport
    pass. Live slots are `some x`; consumed slots remain as `none` and
    still count toward DB indices. -/
abbrev BinderNameSlots := List (Option String)

/-- Lookup in a tombstone-aware binder-name environment. Dead slots are
    skipped for matching but still increment the returned index. -/
def lookupBinderNames : BinderNameSlots → String → Option Nat
  | [], _ => none
  | none :: ys, x =>
      Nat.succ <$> lookupBinderNames ys x
  | some y :: ys, x =>
      if y = x then some 0 else Nat.succ <$> lookupBinderNames ys x

mutual

/-- Syntax erasure against a tombstone-aware binder-name environment. -/
def eraseTermNames (rho : BinderNameSlots) : Term → Option TermDB
  | Term.var x =>
      TermDB.var <$> lookupBinderNames rho x
  | Term.abs x t body =>
      TermDB.abs t <$> eraseTermNames (some x :: rho) body
  | Term.app e1 e2 => do
      let e1' <- eraseTermNames rho e1
      let e2' <- eraseTermNames rho e2
      pure (TermDB.app e1' e2')
  | Term.letBind x e1 e2 => do
      let e1' <- eraseTermNames rho e1
      let e2' <- eraseTermNames (some x :: rho) e2
      pure (TermDB.letBind e1' e2')
  | Term.copy e =>
      TermDB.copy <$> eraseTermNames rho e
  | Term.letpair x y e1 e2 => do
      let e1' <- eraseTermNames rho e1
      let e2' <- eraseTermNames (some y :: some x :: rho) e2
      pure (TermDB.letpair e1' e2')
  | Term.pair e1 e2 => do
      let e1' <- eraseTermNames rho e1
      let e2' <- eraseTermNames rho e2
      pure (TermDB.pair e1' e2')
  | Term.fst e =>
      TermDB.fst <$> eraseTermNames rho e
  | Term.snd e =>
      TermDB.snd <$> eraseTermNames rho e
  | Term.unit =>
      pure TermDB.unit
  | Term.const v ds =>
      pure (TermDB.const v ds)
  | Term.add e1 e2 => do
      let e1' <- eraseTermNames rho e1
      let e2' <- eraseTermNames rho e2
      pure (TermDB.add e1' e2')
  | Term.mul e1 e2 => do
      let e1' <- eraseTermNames rho e1
      let e2' <- eraseTermNames rho e2
      pure (TermDB.mul e1' e2')
  | Term.sum e d =>
      TermDB.sum <$> eraseTermNames rho e <*> pure (eraseDim d)
  | Term.expand e d =>
      TermDB.expand <$> eraseTermNames rho e <*> pure (eraseDim d) <*> pure 0
  | Term.uniformLike e lo hi =>
      TermDB.uniformLike <$> eraseTermNames rho e <*> pure lo <*> pure hi
  | Term.grad x t tOut body =>
      TermDB.grad t tOut <$> eraseTermNames (some x :: rho) body
  | Term.vmap x t body =>
      TermDB.vmap t <$> eraseTermNames (some x :: rho) body
  | Term.handle epsH body clauses => do
      let body' <- eraseTermNames rho body
      let clauses' <- eraseClausesNames rho clauses
      pure (TermDB.handle epsH body' clauses')
  | Term.perform op e =>
      TermDB.perform op <$> eraseTermNames rho e
  | Term.loc ell =>
      pure (TermDB.loc ell)

/-- Clause erasure against a tombstone-aware binder-name environment. -/
def eraseClausesNames
    (rho : BinderNameSlots) :
    List (EffectLabel × String × String × Term) →
    Option (List (EffectLabel × TermDB))
  | [] =>
      pure []
  | (op, x, k, hb) :: rest => do
      let hb' <- eraseTermNames (some k :: some x :: rho) hb
      let rest' <- eraseClausesNames rho rest
      pure ((op, hb') :: rest')

end

@[simp] theorem lookupBinderNames_map_some
    {rho : BinderEnv} {x : String} :
    lookupBinderNames (rho.map some) x = lookupBinder rho x := by
  induction rho with
  | nil =>
      simp [lookupBinderNames, lookupBinder]
  | cons y ys ih =>
      by_cases hy : y = x
      · subst hy
        simp [lookupBinderNames, lookupBinder]
      · simp [lookupBinderNames, lookupBinder, hy, ih]

@[simp] theorem lookupBinderNames_binderSlotNames
    {rho : BinderSlots} {x : String} :
    lookupBinderNames (binderSlotNames rho) x = lookupBinderSlots rho x := by
  induction rho with
  | nil =>
      simp [lookupBinderNames, lookupBinderSlots, binderSlotNames]
  | cons hd tl ih =>
      cases hd with
      | none =>
          simp [lookupBinderNames, lookupBinderSlots, binderSlotNames, ih]
      | some p =>
          cases p with
          | mk y t =>
              by_cases hy : y = x
              · subst hy
                simp [lookupBinderNames, lookupBinderSlots, binderSlotNames]
              · simp [lookupBinderNames, lookupBinderSlots, binderSlotNames, hy, ih]

theorem lookupBinderNames_append_left
    {pref tail : BinderNameSlots} {x : String} {i : Nat}
    (h : lookupBinderNames pref x = some i) :
    lookupBinderNames (pref ++ tail) x = some i := by
  induction pref generalizing i with
  | nil =>
      cases h
  | cons hd tl ih =>
      cases hd with
      | none =>
          cases htl : lookupBinderNames tl x with
          | none =>
              simp [lookupBinderNames, htl] at h
          | some j =>
              simp [lookupBinderNames, htl] at h
              cases h
              simpa [lookupBinderNames, htl] using (ih htl)
      | some y =>
          by_cases hy : y = x
          · subst hy
            have hi : i = 0 := by
              simpa [lookupBinderNames] using h.symm
            subst hi
            simp [lookupBinderNames]
          · cases htl : lookupBinderNames tl x with
            | none =>
                simp [lookupBinderNames, hy, htl] at h
            | some j =>
                simp [lookupBinderNames, hy, htl] at h
                cases h
                simpa [lookupBinderNames, hy, htl] using (ih htl)

theorem lookupBinderNames_append_of_none
    {pref tail : BinderNameSlots} {x : String}
    (h : lookupBinderNames pref x = none) :
    lookupBinderNames (pref ++ tail) x = ((Nat.add pref.length) <$> lookupBinderNames tail x) := by
  induction pref with
  | nil =>
      cases htail : lookupBinderNames tail x with
      | none =>
          simp [htail]
      | some i =>
          simp [htail]
  | cons hd tl ih =>
      cases hd with
      | none =>
          have htl : lookupBinderNames tl x = none := by
            simpa [lookupBinderNames] using h
          simp [lookupBinderNames, ih htl]
          cases htail : lookupBinderNames tail x <;> simp [Nat.add_assoc, Nat.add_comm]
      | some y =>
          by_cases hy : y = x
          · simp [lookupBinderNames, hy] at h
          · have htl : lookupBinderNames tl x = none := by
              simpa [lookupBinderNames, hy] using h
            simp [lookupBinderNames, hy, ih htl]
            cases htail : lookupBinderNames tail x <;> simp [Nat.add_assoc, Nat.add_comm]

mutual

theorem eraseTermNames_map_some
    (rho : BinderEnv) :
    ∀ e, eraseTermNames (rho.map some) e = eraseTerm rho e
  | Term.var x => by
      simp [eraseTermNames, eraseTerm, lookupBinderNames_map_some]
  | Term.abs x t body => by
      simpa using
        congrArg (Option.map (TermDB.abs t))
          (eraseTermNames_map_some (x :: rho) body)
  | Term.app e1 e2 => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e1,
        eraseTermNames_map_some rho e2]
  | Term.letBind x e1 e2 => by
      change
        ((eraseTermNames (rho.map some) e1).bind fun e1' =>
            (eraseTermNames ((x :: rho).map some) e2).bind fun e2' =>
              some (TermDB.letBind e1' e2')) =
          ((eraseTerm rho e1).bind fun e1' =>
            (eraseTerm (x :: rho) e2).bind fun e2' =>
              some (TermDB.letBind e1' e2'))
      rw [eraseTermNames_map_some rho e1, eraseTermNames_map_some (x :: rho) e2]
  | Term.copy e => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.letpair x y e1 e2 => by
      change
        ((eraseTermNames (rho.map some) e1).bind fun e1' =>
            (eraseTermNames ((y :: x :: rho).map some) e2).bind fun e2' =>
              some (TermDB.letpair e1' e2')) =
          ((eraseTerm rho e1).bind fun e1' =>
            (eraseTerm (y :: x :: rho) e2).bind fun e2' =>
              some (TermDB.letpair e1' e2'))
      rw [eraseTermNames_map_some rho e1, eraseTermNames_map_some (y :: x :: rho) e2]
  | Term.pair e1 e2 => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e1,
        eraseTermNames_map_some rho e2]
  | Term.fst e => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.snd e => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.unit => by
      simp [eraseTermNames, eraseTerm]
  | Term.const v ds => by
      simp [eraseTermNames, eraseTerm]
  | Term.add e1 e2 => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e1,
        eraseTermNames_map_some rho e2]
  | Term.mul e1 e2 => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e1,
        eraseTermNames_map_some rho e2]
  | Term.sum e d => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.expand e d => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.uniformLike e lo hi => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.grad x t tOut body => by
      simpa using
        congrArg (Option.map (TermDB.grad t tOut))
          (eraseTermNames_map_some (x :: rho) body)
  | Term.vmap x t body => by
      simpa using
        congrArg (Option.map (TermDB.vmap t))
          (eraseTermNames_map_some (x :: rho) body)
  | Term.handle epsH body clauses => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho body,
        eraseClausesNames_map_some rho clauses]
  | Term.perform op e => by
      simp [eraseTermNames, eraseTerm, eraseTermNames_map_some rho e]
  | Term.loc ell => by
      simp [eraseTermNames, eraseTerm]

theorem eraseClausesNames_map_some
    (rho : BinderEnv) :
    ∀ clauses, eraseClausesNames (rho.map some) clauses = eraseClauses rho clauses
  | [] => by
      simp [eraseClausesNames, eraseClauses]
  | (op, x, k, hb) :: rest => by
      change
        ((eraseTermNames ((k :: x :: rho).map some) hb).bind fun hb' =>
            (eraseClausesNames (rho.map some) rest).bind fun rest' =>
              some ((op, hb') :: rest')) =
          ((eraseTerm (k :: x :: rho) hb).bind fun hb' =>
            (eraseClauses rho rest).bind fun rest' =>
              some ((op, hb') :: rest'))
      rw [eraseTermNames_map_some (k :: x :: rho) hb, eraseClausesNames_map_some rho rest]

end

@[simp] theorem eraseTermNames_eraseBinderSlots
    (Gamma : LinearCtx) (e : Term) :
    eraseTermNames (binderSlotNames (eraseBinderSlots Gamma)) e =
      eraseTerm (ctxEnv Gamma) e := by
  simpa [binderSlotNames_eraseBinderSlots] using
    (eraseTermNames_map_some (ctxEnv Gamma) e)

@[simp] theorem eraseClausesNames_eraseBinderSlots
    (Gamma : LinearCtx)
    (clauses : List (EffectLabel × String × String × Term)) :
    eraseClausesNames (binderSlotNames (eraseBinderSlots Gamma)) clauses =
      eraseClauses (ctxEnv Gamma) clauses := by
  simpa [binderSlotNames_eraseBinderSlots] using
    (eraseClausesNames_map_some (ctxEnv Gamma) clauses)

mutual

theorem eraseTerm_suffix :
    ∀ {ρ : BinderEnv} {e : Term} {eDB : TermDB},
      eraseTerm ρ e = some eDB -> ∀ σ, eraseTerm (ρ ++ σ) e = some eDB
  | ρ, Term.var x, eDB, h, σ => by
      cases hρx : lookupBinder ρ x with
      | none =>
          simp [eraseTerm, hρx] at h
      | some i =>
          simp [eraseTerm, hρx] at h
          cases h
          simpa [eraseTerm, lookupBinder_append_left hρx]
  | ρ, Term.abs x t body, eDB, h, σ => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      cases h
      simpa [eraseTerm, List.cons_append] using
        congrArg (Option.map (TermDB.abs t)) (eraseTerm_suffix hbody σ)
  | ρ, Term.app e1 e2, eDB, h, σ => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix h1 σ, eraseTerm_suffix h2 σ]
  | ρ, Term.letBind x e1 e2, eDB, h, σ => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm (x :: ρ) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      have h2' : eraseTerm (x :: (ρ ++ σ)) e2 = some e2DB := by
        simpa [List.cons_append] using eraseTerm_suffix h2 σ
      simp [eraseTerm, eraseTerm_suffix h1 σ, h2']
  | ρ, Term.copy e, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix he σ]
  | ρ, Term.letpair x y e1 e2, eDB, h, σ => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm (y :: x :: ρ) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      have h2' : eraseTerm (y :: x :: (ρ ++ σ)) e2 = some e2DB := by
        simpa [List.cons_append] using eraseTerm_suffix h2 σ
      simp [eraseTerm, eraseTerm_suffix h1 σ, h2']
  | ρ, Term.pair e1 e2, eDB, h, σ => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix h1 σ, eraseTerm_suffix h2 σ]
  | ρ, Term.fst e, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix he σ]
  | ρ, Term.snd e, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix he σ]
  | ρ, Term.unit, eDB, h, σ => by
      simpa [eraseTerm] using h
  | ρ, Term.const v ds, eDB, h, σ => by
      simpa [eraseTerm] using h
  | ρ, Term.add e1 e2, eDB, h, σ => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix h1 σ, eraseTerm_suffix h2 σ]
  | ρ, Term.mul e1 e2, eDB, h, σ => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix h1 σ, eraseTerm_suffix h2 σ]
  | ρ, Term.sum e d, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at h
        cases h
      · simp [eraseTerm, he] at h
        simpa [eraseTerm, eraseTerm_suffix he σ] using h
  | ρ, Term.expand e d, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at h
        cases h
      · simp [eraseTerm, he] at h
        simpa [eraseTerm, eraseTerm_suffix he σ] using h
  | ρ, Term.uniformLike e lo hi, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at h
        cases h
      · simp [eraseTerm, he] at h
        simpa [eraseTerm, eraseTerm_suffix he σ] using h
  | ρ, Term.grad x t tOut body, eDB, h, σ => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      cases h
      simpa [eraseTerm, List.cons_append] using
        congrArg (Option.map (TermDB.grad t tOut)) (eraseTerm_suffix hbody σ)
  | ρ, Term.vmap x t body, eDB, h, σ => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      cases h
      simpa [eraseTerm, List.cons_append] using
        congrArg (Option.map (TermDB.vmap t)) (eraseTerm_suffix hbody σ)
  | ρ, Term.handle epsH body clauses, eDB, h, σ => by
      rcases hbody : eraseTerm ρ body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      rcases hcls : eraseClauses ρ clauses with _ | clsDB <;> simp [eraseTerm, hbody, hcls] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix hbody σ, eraseClauses_suffix hcls σ]
  | ρ, Term.perform op e, eDB, h, σ => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [eraseTerm, eraseTerm_suffix he σ]
  | ρ, Term.loc ell, eDB, h, σ => by
      simpa [eraseTerm] using h

theorem eraseClauses_suffix :
    ∀ {ρ : BinderEnv}
      {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses ρ clauses = some clausesDB -> ∀ σ, eraseClauses (ρ ++ σ) clauses = some clausesDB
  | ρ, [], clausesDB, h, σ => by
      simpa [eraseClauses] using h
  | ρ, (op, x, k, hb) :: rest, clausesDB, h, σ => by
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;> simp [eraseClauses, hhb] at h
      rcases hrest : eraseClauses ρ rest with _ | restDB <;> simp [eraseClauses, hhb, hrest] at h
      cases h
      have hhb' : eraseTerm (k :: x :: (ρ ++ σ)) hb = some hbDB := by
        simpa [List.cons_append] using eraseTerm_suffix hhb σ
      simp [eraseClauses, hhb', eraseClauses_suffix hrest σ]

end

@[simp] theorem eraseTerm_var_at_target
    {ρ : BinderEnv} {x : String}
    (hx : x ∉ ρ) :
    eraseTerm (ρ ++ [x]) (Term.var x) = some (TermDB.var ρ.length) := by
  simp [eraseTerm, lookupBinder_append_target hx]

@[simp] theorem eraseTerm_var_before_target
    {ρ : BinderEnv} {x y : String}
    (hxy : y ≠ x) :
    eraseTerm (ρ ++ [x]) (Term.var y) = TermDB.var <$> lookupBinder ρ y := by
  simp [eraseTerm, lookupBinder_append_singleton_ne hxy]

@[simp] theorem eraseTerm_var_ctx_target
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hx : x ∉ linearCtxDom GammaPost) :
    eraseTerm (ctxEnv (GammaPre ++ [(x, t)] ++ GammaPost)) (Term.var x) =
      some (TermDB.var GammaPost.length) := by
  rw [show ctxEnv (GammaPre ++ [(x, t)] ++ GammaPost) =
      ctxEnv GammaPost ++ [x] ++ ctxEnv GammaPre by
      simp [ctxEnv, linearCtxDom, List.reverse_append, List.reverse_cons, List.append_assoc]]
  simp [eraseTerm, lookupBinder_append_target, hx, List.append_assoc]

theorem transport_var_lexical
    {Delta : CapCtx} {Sigma : StoreTyp}
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, t)] ++ GammaPost)) :
    let Gamma := GammaPre ++ [(x, t)] ++ GammaPost
    let i := GammaPost.length
    eraseTerm (ctxEnv Gamma) (Term.var x) = some (TermDB.var i) ∧
    HasTypeDB Delta Sigma (eraseCtx Gamma) (TermDB.var i) t []
      ((eraseCtx Gamma).set i none) ∧
    CtxCorr (GammaPre ++ GammaPost) ((eraseCtx Gamma).set i none) := by
  let Gamma := GammaPre ++ [(x, t)] ++ GammaPost
  let i := GammaPost.length
  have hsuf : x ∉ linearCtxDom GammaPost :=
    noDupNames_middle_fresh_suffix hnd
  have herase :
      eraseTerm (ctxEnv Gamma) (Term.var x) = some (TermDB.var i) := by
    simpa [Gamma, i] using (eraseTerm_var_ctx_target (GammaPre := GammaPre) (GammaPost := GammaPost)
      (x := x) (t := t) hsuf)
  have hdb :
      HasTypeDB Delta Sigma (eraseCtx Gamma) (TermDB.var i) t []
        ((eraseCtx Gamma).set i none) := by
    simpa [Gamma, i] using
      (HasTypeDB.var Delta Sigma (eraseCtx Gamma) i t
        (eraseCtx_get_target GammaPre GammaPost x t))
  have hcorr :
      CtxCorr (GammaPre ++ GammaPost) ((eraseCtx Gamma).set i none) := by
    simpa [Gamma, i] using (ctxCorr_consume_target GammaPre GammaPost x t)
  exact ⟨herase, hdb, hcorr⟩

/-- Local invariant for the narrow wrapper bridge: the named context is
    a lexical binder stack with pairwise-distinct names, every stack
    name is disjoint from the remaining binders in the term, and the
    term's own binders are distinct. This is strictly smaller than a
    global named↔DB correspondence relation; it is exactly the shape
    produced by the four substitution wrappers after inversion. -/
def LexicallyScoped (Gamma : LinearCtx) (e : Term) : Prop :=
  NoDupNames Gamma ∧
    (∀ x, x ∈ linearCtxDom Gamma → x ∉ boundVars e) ∧
    WellScoped e

theorem lexical_nil
    {e : Term}
    (hws : WellScoped e) :
    LexicallyScoped [] e := by
  refine ⟨noDupNames_nil, ?_, hws⟩
  intro x hx
  simpa [linearCtxDom] using hx

theorem lexical_singleton
    {x : String} {tx : Typ} {e : Term}
    (hx : x ∉ boundVars e)
    (hws : WellScoped e) :
    LexicallyScoped [(x, tx)] e := by
  refine ⟨noDupNames_singleton x tx, ?_, hws⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with rfl
  exact hx

theorem lexical_pair
    {x y : String} {tx ty : Typ} {e : Term}
    (hxy : x ≠ y)
    (hx : x ∉ boundVars e)
    (hy : y ∉ boundVars e)
    (hws : WellScoped e) :
    LexicallyScoped [(x, tx), (y, ty)] e := by
  refine ⟨(noDupNames_pair_iff x y tx ty).2 hxy, ?_, hws⟩
  intro z hz
  simp [linearCtxDom] at hz
  rcases hz with rfl | rfl
  · exact hx
  · exact hy

theorem noDupNames_of_sublist
    {Gamma' Gamma : LinearCtx}
    (hsub : List.Sublist Gamma' Gamma)
    (hnd : NoDupNames Gamma) :
    NoDupNames Gamma' := by
  unfold NoDupNames at hnd ⊢
  simpa [linearCtxDom] using (hsub.map Prod.fst).nodup hnd

theorem noDupNames_middle_fresh_prefix
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, t)] ++ GammaPost)) :
    x ∉ linearCtxDom GammaPre := by
  have hsub : List.Sublist (GammaPre ++ [(x, t)])
      (GammaPre ++ [(x, t)] ++ GammaPost) := by
    induction GammaPost with
    | nil =>
        simp
    | cons p rest ih =>
        simp [List.append_assoc]
  have hnd' : NoDupNames (GammaPre ++ [(x, t)]) :=
    noDupNames_of_sublist hsub hnd
  intro hx
  have hnd'' := hnd'
  unfold NoDupNames at hnd''
  simp [linearCtxDom, List.nodup_append] at hnd''
  have hx' : ∃ ty, (x, ty) ∈ GammaPre := by
    simpa [linearCtxDom] using hx
  rcases hx' with ⟨ty, hmem⟩
  exact (hnd''.2 x ty hmem) rfl

theorem slotCorr_consume_name
    {Gamma : LinearCtx} {rho : BinderSlots} {x : String} {t : Typ}
    (hslot : SlotCorr Gamma rho)
    (hnd : NoDupNames Gamma)
    (hmem : (x, t) ∈ Gamma) :
    ∃ i,
      lookupBinderSlots rho x = some i ∧
      (binderSlotTypes rho)[i]? = some (some t) ∧
      SlotCorr (Gamma.filter (fun p => p.1 ≠ x)) (rho.set i none) := by
  induction hslot generalizing x t with
  | nil =>
      cases hmem
  | dead hprev ih =>
      rcases ih hnd hmem with ⟨i, hlook, hget, hcorr⟩
      refine ⟨i + 1, ?_, ?_, ?_⟩
      · simp [lookupBinderSlots, hlook]
      · simpa [binderSlotTypes] using hget
      · simpa [List.set] using (SlotCorr.dead hcorr)
  | @live Gamma0 rho0 y ty hprev ih =>
      have hndPrev : NoDupNames Gamma0 := by
        have hsub : List.Sublist Gamma0 (Gamma0 ++ [(y, ty)]) := by
          induction Gamma0 with
          | nil =>
              simp
          | cons a rest ih =>
              simp [ih]
        exact noDupNames_of_sublist hsub hnd
      have hyFresh : y ∉ linearCtxDom Gamma0 := by
        have hnd' : NoDupNames (Gamma0 ++ [(y, ty)] ++ ([] : LinearCtx)) := by
          simpa using hnd
        simpa using
          (noDupNames_middle_fresh_prefix (GammaPre := Gamma0) (GammaPost := ([] : LinearCtx))
            (x := y) (t := ty) hnd')
      have hmem' : (x, t) ∈ Gamma0 ∨ (x, t) = (y, ty) := by
        simpa [List.mem_append] using hmem
      rcases hmem' with hmem0 | hlast
      · have hxy : y ≠ x := by
          intro hyx
          subst hyx
          apply hyFresh
          unfold linearCtxDom
          exact List.mem_map.mpr ⟨(y, t), hmem0, rfl⟩
        rcases ih hndPrev hmem0 with ⟨i, hlook, hget, hcorr⟩
        refine ⟨i + 1, ?_, ?_, ?_⟩
        · simp [lookupBinderSlots, hxy, hlook]
        · simpa [binderSlotTypes] using hget
        · simpa [List.filter_append, hxy, List.set] using
            (SlotCorr.live (x := y) (t := ty) hcorr)
      · rcases Prod.mk.inj hlast with ⟨rfl, rfl⟩
        refine ⟨0, ?_, ?_, ?_⟩
        · simp [lookupBinderSlots]
        · simp [binderSlotTypes]
        · have hfilter0 : Gamma0.filter (fun p => p.1 ≠ x) = Gamma0 :=
            filter_ctx_eq_self_of_fresh hyFresh
          have hfilter : (Gamma0 ++ [(x, t)]).filter (fun p => p.1 ≠ x) = Gamma0 := by
            rw [List.filter_append, hfilter0]
            simp
          rw [hfilter]
          simpa [List.set] using (SlotCorr.dead hprev)

theorem HasType.var_mem_of_eq
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {tau : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e tau eps Gamma') :
    ∀ {x : String} {t : Typ}, e = Term.var x -> tau = t -> (x, t) ∈ Gamma := by
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | var _ _ GammaPre GammaPost y ty =>
      intro x t heq hty
      cases heq
      cases hty
      simp [List.mem_append]
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro x t heq hty
      exact ih heq hty
  | nil _ _ _ _ _ =>
      exact True.intro
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      exact True.intro
  | _ =>
      intro x t heq hty
      cases heq

theorem HasType.var_mem_of_typing
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {x : String} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (Term.var x) t eps Gamma') :
    (x, t) ∈ Gamma :=
  HasType.var_mem_of_eq h rfl rfl

theorem HasType.var_output_filter_of_eq
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {tau : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e tau eps Gamma')
    (hnd : NoDupNames Gamma) :
    ∀ {x : String} {t : Typ},
      e = Term.var x -> tau = t ->
      Gamma' = Gamma.filter (fun p => p.1 ≠ x) := by
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | var _ _ GammaPre GammaPost y ty =>
      intro x t heq hty
      cases heq
      cases hty
      have hpre : y ∉ linearCtxDom GammaPre :=
        noDupNames_middle_fresh_prefix
          (GammaPre := GammaPre) (GammaPost := GammaPost) (x := y) (t := ty) hnd
      have hpost : y ∉ linearCtxDom GammaPost :=
        noDupNames_middle_fresh_suffix
          (GammaPre := GammaPre) (GammaPost := GammaPost) (x := y) (t := ty) hnd
      symm
      have hpreFilter :
          GammaPre.filter (fun p => p.1 ≠ y) = GammaPre := by
        simpa using filter_ctx_eq_self_of_fresh (Gamma := GammaPre) (x := y) hpre
      have hpostFilter :
          GammaPost.filter (fun p => p.1 ≠ y) = GammaPost := by
        simpa using filter_ctx_eq_self_of_fresh (Gamma := GammaPost) (x := y) hpost
      have hpreFilter' :
          GammaPre.filter (fun p => !decide (p.1 = y)) = GammaPre := by
        simpa using hpreFilter
      have hpostFilter' :
          GammaPost.filter (fun p => !decide (p.1 = y)) = GammaPost := by
        simpa using hpostFilter
      simp [hpreFilter', hpostFilter']
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro x t heq hty
      exact ih hnd heq hty
  | nil _ _ _ _ _ =>
      exact True.intro
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      exact True.intro
  | _ =>
      intro x t heq hty
      cases heq

theorem HasType.var_output_filter_of_noDup
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {x : String} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (Term.var x) t eps Gamma')
    (hnd : NoDupNames Gamma) :
    Gamma' = Gamma.filter (fun p => p.1 ≠ x) :=
  HasType.var_output_filter_of_eq h hnd rfl rfl

theorem transport_var_slotCorr
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {rho : BinderSlots} {x : String} {t : Typ}
    (h : HasType Delta Sigma Gamma (Term.var x) t [] Gamma')
    (hnd : NoDupNames Gamma)
    (hslot : SlotCorr Gamma rho) :
    ∃ i,
      eraseTermNames (binderSlotNames rho) (Term.var x) = some (TermDB.var i) ∧
      HasTypeDB Delta Sigma (binderSlotTypes rho) (TermDB.var i) t []
        ((binderSlotTypes rho).set i none) ∧
      SlotCorr Gamma' (rho.set i none) := by
  have hmem : (x, t) ∈ Gamma :=
    HasType.var_mem_of_typing h
  rcases slotCorr_consume_name hslot hnd hmem with ⟨i, hlook, hget, hcorr⟩
  refine ⟨i, ?_, ?_, ?_⟩
  · simp [eraseTermNames, lookupBinderNames_binderSlotNames, hlook]
  · exact HasTypeDB.var Delta Sigma (binderSlotTypes rho) i t hget
  · simpa [HasType.var_output_filter_of_noDup h hnd] using hcorr

theorem lexical_sublist
    {Gamma' Gamma : LinearCtx} {e : Term}
    (hlex : LexicallyScoped Gamma e)
    (hsub : List.Sublist Gamma' Gamma) :
    LexicallyScoped Gamma' e := by
  rcases hlex with ⟨hnd, hdom, hws⟩
  refine ⟨noDupNames_of_sublist hsub hnd, ?_, hws⟩
  intro x hx
  have hx' : x ∈ linearCtxDom Gamma := by
    unfold linearCtxDom at hx ⊢
    exact (hsub.map Prod.fst).subset hx
  exact hdom x hx'

theorem lexical_output_of_typing
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' : LinearCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (hlex : LexicallyScoped Gamma e) :
    LexicallyScoped Gamma' e :=
  lexical_sublist hlex (has_type_sublist h)

theorem lexical_app_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.app e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_app_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.app e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_copy_body
    {Gamma : LinearCtx} {e : Term}
    (h : LexicallyScoped Gamma (Term.copy e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_pair_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.pair e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_pair_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.pair e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_letBind_bound
    {Gamma : LinearCtx} {x : String} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letBind x e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hbody := wellScoped_letBind_body hws
  refine ⟨hnd, ?_, ?_⟩
  · intro y hy
    have hnot := hdom y hy
    simp [boundVars] at hnot
    exact hnot.2.1
  · exact hbody.1

theorem lexical_letpair_bound
    {Gamma : LinearCtx} {x y : String} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letpair x y e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hbody := wellScoped_letpair_body hws
  refine ⟨hnd, ?_, ?_⟩
  · intro z hz
    have hnot := hdom z hz
    simp [boundVars] at hnot
    exact hnot.2.2.1
  · exact hbody.1

theorem lexical_fst_body
    {Gamma : LinearCtx} {e : Term}
    (h : LexicallyScoped (Gamma := Gamma) (Term.fst e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_snd_body
    {Gamma : LinearCtx} {e : Term}
    (h : LexicallyScoped (Gamma := Gamma) (Term.snd e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_add_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.add e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_add_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.add e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_mul_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.mul e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_mul_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.mul e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_sum_body
    {Gamma : LinearCtx} {e : Term} {d : Dim}
    (h : LexicallyScoped (Gamma := Gamma) (Term.sum e d)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_expand_body
    {Gamma : LinearCtx} {e : Term} {d : Dim}
    (h : LexicallyScoped (Gamma := Gamma) (Term.expand e d)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_uniformLike_body
    {Gamma : LinearCtx} {e : Term} {lo hi : Float}
    (h : LexicallyScoped (Gamma := Gamma) (Term.uniformLike e lo hi)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_perform_body
    {Gamma : LinearCtx} {op : EffectLabel} {e : Term}
    (h : LexicallyScoped (Gamma := Gamma) (Term.perform op e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_handle_body
    {Gamma : LinearCtx} {epsH : EffectRow}
    {body : Term} {clauses : List (EffectLabel × String × String × Term)}
    (h : LexicallyScoped Gamma (Term.handle epsH body clauses)) :
    LexicallyScoped Gamma body := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, wellScoped_handle_body hws⟩
  intro x hx
  have hnot := hdom x hx
  simp [boundVars, List.mem_append, not_or] at hnot
  exact hnot.1

theorem noDupNames_append_singleton
    {Gamma : LinearCtx} {x : String} {t : Typ}
    (h : NoDupNames Gamma)
    (hx : x ∉ linearCtxDom Gamma) :
    NoDupNames (Gamma ++ [(x, t)]) := by
  unfold NoDupNames at h ⊢
  simp [linearCtxDom, List.nodup_append]
  refine ⟨h, ?_⟩
  intro a ta hmem heq
  subst heq
  have hdommem : a ∈ linearCtxDom Gamma := by
    unfold linearCtxDom
    exact List.mem_map.mpr ⟨(a, ta), hmem, rfl⟩
  exact hx hdommem

theorem lexical_abs_body
    {Gamma : LinearCtx} {x : String} {tx t : Typ} {body : Term}
    (h : LexicallyScoped Gamma (Term.abs x t body)) :
    LexicallyScoped (Gamma ++ [(x, tx)]) body := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_abs_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2
  · subst y
    exact hbody.1

theorem sublist_append_singleton
    {Gamma' Gamma : LinearCtx} {x : String} {t : Typ}
    (hsub : List.Sublist Gamma' Gamma) :
    List.Sublist (Gamma' ++ [(x, t)]) (Gamma ++ [(x, t)]) := by
  induction hsub with
  | slnil =>
      simp
  | @cons a l1 l2 hsub ih =>
      simp [ih]
  | @cons₂ a l1 l2 hsub ih =>
      simp [ih]

theorem sublist_append_pair
    {Gamma' Gamma : LinearCtx} {x y : String} {tx ty : Typ}
    (hsub : List.Sublist Gamma' Gamma) :
    List.Sublist (Gamma' ++ [(x, tx), (y, ty)]) (Gamma ++ [(x, tx), (y, ty)]) := by
  simpa [List.append_assoc] using
    (sublist_append_singleton (x := y) (t := ty)
      (sublist_append_singleton (x := x) (t := tx) hsub))

theorem fresh_of_sublist
    {Gamma' Gamma : LinearCtx} {x : String}
    (hsub : List.Sublist Gamma' Gamma)
    (hx : x ∉ linearCtxDom Gamma) :
    x ∉ linearCtxDom Gamma' := by
  intro hx'
  apply hx
  unfold linearCtxDom at hx' ⊢
  exact (hsub.map Prod.fst).subset hx'

theorem sublist_cons_cases
    {α : Type} {l r : List α} {a : α}
    (hsub : List.Sublist l (a :: r)) :
    List.Sublist l r ∨
      ∃ l', l = a :: l' ∧ List.Sublist l' r := by
  cases hsub with
  | cons _ htail =>
      exact Or.inl htail
  | cons₂ _ htail =>
      exact Or.inr ⟨_, rfl, htail⟩

theorem sublist_append_singleton_cases
    {Gamma' Gamma : LinearCtx} {x : String} {t : Typ}
    (hsub : List.Sublist Gamma' (Gamma ++ [(x, t)])) :
    List.Sublist Gamma' Gamma ∨
      ∃ GammaPre,
        Gamma' = GammaPre ++ [(x, t)] ∧
        List.Sublist GammaPre Gamma := by
  have hrev : List.Sublist Gamma'.reverse ((x, t) :: Gamma.reverse) := by
    simpa [List.reverse_append] using hsub.reverse
  rcases sublist_cons_cases hrev with htail | ⟨GammaRevPre, hrevEq, hrevSub⟩
  · left
    simpa [List.reverse_reverse] using htail.reverse
  · right
    refine ⟨GammaRevPre.reverse, ?_, ?_⟩
    · have := congrArg List.reverse hrevEq
      simpa [List.reverse_reverse, List.reverse_cons] using this
    · simpa [List.reverse_reverse] using hrevSub.reverse

theorem sublist_append_pair_cases
    {Gamma' Gamma : LinearCtx} {x y : String} {tx ty : Typ}
    (hsub : List.Sublist Gamma' (Gamma ++ [(x, tx), (y, ty)])) :
    List.Sublist Gamma' Gamma ∨
      (∃ GammaPre, Gamma' = GammaPre ++ [(x, tx)] ∧ List.Sublist GammaPre Gamma) ∨
      (∃ GammaPre, Gamma' = GammaPre ++ [(y, ty)] ∧ List.Sublist GammaPre Gamma) ∨
      (∃ GammaPre, Gamma' = GammaPre ++ [(x, tx), (y, ty)] ∧ List.Sublist GammaPre Gamma) := by
  have hsubY : List.Sublist Gamma' ((Gamma ++ [(x, tx)]) ++ [(y, ty)]) := by
    simpa [List.append_assoc] using hsub
  rcases
      sublist_append_singleton_cases
        (Gamma := Gamma ++ [(x, tx)]) (Gamma' := Gamma') (x := y) (t := ty) hsubY with
    hdrop | ⟨GammaMid, hmid, hmidSub⟩
  · rcases
      sublist_append_singleton_cases
        (Gamma := Gamma) (Gamma' := Gamma') (x := x) (t := tx) hdrop with
      hdrop' | ⟨GammaPre, hpre, hpreSub⟩
    · exact Or.inl hdrop'
    · exact Or.inr <| Or.inl ⟨GammaPre, hpre, hpreSub⟩
  · rcases
      sublist_append_singleton_cases
        (Gamma := Gamma) (Gamma' := GammaMid) (x := x) (t := tx) hmidSub with
      hdrop' | ⟨GammaPre, hpre, hpreSub⟩
    · right
      right
      left
      refine ⟨GammaMid, ?_, hdrop'⟩
      simpa [hmid, List.append_assoc]
    · right
      right
      right
      refine ⟨GammaPre, ?_, hpreSub⟩
      simp [hmid, hpre, List.append_assoc]

theorem lookupBinderSlots_eq_of_slotKillsOnly
    {GammaIn GammaOut : LinearCtx}
    {rhoIn rhoOut : BinderSlots}
    {x : String}
    (hslotIn : SlotCorr GammaIn rhoIn)
    (hslotOut : SlotCorr GammaOut rhoOut)
    (hkill : SlotKillsOnly rhoOut rhoIn)
    (hsub : List.Sublist GammaOut GammaIn)
    (hnd : NoDupNames GammaIn)
    (hmem : x ∈ linearCtxDom GammaOut) :
    lookupBinderSlots rhoOut x = lookupBinderSlots rhoIn x := by
  revert GammaOut rhoOut hslotOut hkill hsub hmem
  induction hslotIn with
  | nil =>
      intro GammaOut rhoOut hslotOut hkill hsub hmem
      cases hkill
      cases hslotOut
      simpa [linearCtxDom] using hmem
  | dead hslotIn' ih =>
      intro GammaOut rhoOut hslotOut hkill hsub hmem
      cases hkill with
      | dead_dead hkill' =>
          cases hslotOut with
          | dead hslotOut' =>
              exact congrArg (Option.map Nat.succ) (ih hnd hslotOut' hkill' hsub hmem)
  | @live GammaPrev rhoPrev y t hslotIn' ih =>
      intro GammaOut rhoOut hslotOut hkill hsub hmem
      cases hkill with
      | dead_live hkill' =>
          cases hslotOut with
          | dead hslotOut' =>
              have hyPrev : y ∉ linearCtxDom GammaPrev := by
                have hnd' : NoDupNames (GammaPrev ++ [(y, t)] ++ ([] : LinearCtx)) := by
                  simpa using hnd
                simpa using
                  (noDupNames_middle_fresh_prefix
                    (GammaPre := GammaPrev) (GammaPost := ([] : LinearCtx))
                    (x := y) (t := t) hnd')
              have hlookInTail : lookupBinderSlots rhoPrev y = none :=
                lookupBinderSlots_none_of_fresh hslotIn' hyPrev
              have hyOut : y ∉ linearCtxDom GammaOut := by
                apply fresh_of_lookupBinderSlots_none hslotOut'
                exact slotKillsOnly_lookup_none hkill' hlookInTail
              rcases
                  sublist_append_singleton_cases
                    (Gamma := GammaPrev) (Gamma' := GammaOut) (x := y) (t := t) hsub with
                hsubPrev | ⟨GammaOutPrev, hshape, hsubPrev⟩
              · have hndPrev : NoDupNames GammaPrev := by
                  exact noDupNames_of_sublist
                    (show List.Sublist GammaPrev (GammaPrev ++ [(y, t)]) from by simp) hnd
                have hxy : y ≠ x := by
                  intro hEq
                  subst hEq
                  exact hyOut hmem
                simpa [lookupBinderSlots, hxy] using
                  congrArg (Option.map Nat.succ)
                    (ih hndPrev hslotOut' hkill' hsubPrev hmem)
              · have : y ∈ linearCtxDom GammaOut := by
                  rw [hshape]
                  simp [linearCtxDom]
                exact (hyOut this).elim
      | live hkill' =>
          cases hslotOut with
          | @live GammaOutPrev rhoOutPrev y t hslotOut' =>
              have hndPrev : NoDupNames GammaPrev := by
                exact noDupNames_of_sublist
                  (show List.Sublist GammaPrev (GammaPrev ++ [(y, t)]) from by simp) hnd
              have hndOut : NoDupNames (GammaOutPrev ++ [(y, t)]) :=
                noDupNames_of_sublist hsub hnd
              rcases
                  sublist_append_singleton_cases
                    (Gamma := GammaPrev) (Gamma' := GammaOutPrev ++ [(y, t)]) (x := y) (t := t) hsub with
                hdrop | ⟨GammaMid, hshape, hsubPrev⟩
              · have hyPrev : y ∉ linearCtxDom GammaPrev := by
                  have hnd' : NoDupNames (GammaPrev ++ [(y, t)] ++ ([] : LinearCtx)) := by
                    simpa using hnd
                  simpa using
                    (noDupNames_middle_fresh_prefix
                      (GammaPre := GammaPrev) (GammaPost := ([] : LinearCtx))
                      (x := y) (t := t) hnd')
                have : y ∈ linearCtxDom GammaPrev := by
                  have hyPair : (y, t) ∈ GammaOutPrev ++ [(y, t)] := by
                    simp
                  unfold linearCtxDom
                  exact List.mem_map.mpr ⟨(y, t), hdrop.subset hyPair, rfl⟩
                exact (hyPrev this).elim
              · have hmid : GammaOutPrev = GammaMid := by
                  have hrev := congrArg List.reverse hshape
                  have hrev' : [(y, t)] ++ GammaOutPrev.reverse = [(y, t)] ++ GammaMid.reverse := by
                    simpa [List.reverse_append] using hrev
                  simpa using congrArg List.reverse ((List.append_right_inj [(y, t)]).1 hrev')
                subst hmid
                simp [linearCtxDom] at hmem
                rcases hmem with hmemPrev | rfl
                · have hyOutPrev : y ∉ linearCtxDom GammaOutPrev := by
                    have hnd' : NoDupNames (GammaOutPrev ++ [(y, t)] ++ ([] : LinearCtx)) := by
                      simpa using hndOut
                    simpa using
                      (noDupNames_middle_fresh_prefix
                        (GammaPre := GammaOutPrev) (GammaPost := ([] : LinearCtx))
                        (x := y) (t := t) hnd')
                  have hxy : y ≠ x := by
                    intro hEq
                    subst hEq
                    have hmemPrev' : y ∈ linearCtxDom GammaOutPrev := by
                      simpa [linearCtxDom] using hmemPrev
                    exact hyOutPrev hmemPrev'
                  have hmemPrev' : x ∈ linearCtxDom GammaOutPrev := by
                    simpa [linearCtxDom] using hmemPrev
                  simpa [lookupBinderSlots, hxy] using
                    congrArg (Option.map Nat.succ)
                      (ih hndPrev hslotOut' hkill' hsubPrev hmemPrev')
                · simp [lookupBinderSlots]

theorem restrictSlots_append_singleton_fresh
    {GammaCtx GammaKeep : LinearCtx} {rho : BinderSlots} {x : String} {tx : Typ}
    (hslot : SlotCorr GammaCtx rho)
    (hxCtx : x ∉ linearCtxDom GammaCtx) :
    restrictSlots (GammaKeep ++ [(x, tx)]) rho = restrictSlots GammaKeep rho := by
  induction hslot with
  | nil =>
      simp [restrictSlots]
  | dead hprev ih =>
      simp [restrictSlots, ih hxCtx]
  | @live GammaPrev rhoPrev y ty hprev ih =>
      have hxPrev : x ∉ linearCtxDom GammaPrev := by
        intro hmem
        apply hxCtx
        unfold linearCtxDom at hmem ⊢
        rcases List.mem_map.mp hmem with ⟨p, hp, hpEq⟩
        exact List.mem_map.mpr ⟨p, by simp [List.mem_append, hp], hpEq⟩
      have hyx : y ≠ x := by
        intro hEq
        subst hEq
        apply hxCtx
        simp [linearCtxDom]
      have hmemEq :
          (y ∈ linearCtxDom (GammaKeep ++ [(x, tx)])) ↔
          (y ∈ linearCtxDom GammaKeep) := by
        simp [linearCtxDom, hyx]
      rw [restrictSlots, restrictSlots]
      simp only [hmemEq]
      rw [ih hxPrev]

theorem lookupBinderSlots_eq_restrictSlots
    {GammaIn GammaKeep : LinearCtx}
    {rho : BinderSlots}
    {z : String}
    (hslot : SlotCorr GammaIn rho)
    (hsub : List.Sublist GammaKeep GammaIn)
    (hnd : NoDupNames GammaIn)
    (hz : z ∈ linearCtxDom GammaKeep) :
    lookupBinderSlots (restrictSlots GammaKeep rho) z = lookupBinderSlots rho z := by
  induction hslot generalizing GammaKeep with
  | nil =>
      cases hsub
      simpa [linearCtxDom] using hz
  | dead hprev ih =>
      exact congrArg (Option.map Nat.succ) (ih hsub hnd hz)
  | @live GammaPrev rhoPrev x tx hprev ih =>
      rcases
          sublist_append_singleton_cases
            (Gamma := GammaPrev) (Gamma' := GammaKeep) (x := x) (t := tx) hsub with
        hdrop | ⟨GammaKeepPrev, hshape, hsubPrev⟩
      · have hxPrev : x ∉ linearCtxDom GammaPrev := by
          have hnd' : NoDupNames (GammaPrev ++ [(x, tx)] ++ ([] : LinearCtx)) := by
            simpa using hnd
          simpa using
            (noDupNames_middle_fresh_prefix
              (GammaPre := GammaPrev) (GammaPost := ([] : LinearCtx))
              (x := x) (t := tx) hnd')
        have hxKeep : x ∉ linearCtxDom GammaKeep :=
          fresh_of_sublist hdrop hxPrev
        have hzx : x ≠ z := by
          intro hEq
          subst hEq
          exact hxKeep hz
        simpa [restrictSlots, hxKeep, lookupBinderSlots, hzx] using
          congrArg (Option.map Nat.succ) (ih hdrop (noDupNames_of_sublist
            (show List.Sublist GammaPrev (GammaPrev ++ [(x, tx)]) from by simp) hnd) hz)
      · rw [hshape] at hz
        simp [linearCtxDom] at hz
        rcases hz with hzPrev | rfl
        · have hxKeepPrev : x ∉ linearCtxDom GammaKeepPrev := by
            have hndKeep : NoDupNames (GammaKeepPrev ++ [(x, tx)]) := by
              simpa [hshape] using (noDupNames_of_sublist hsub hnd)
            have hndKeep' : NoDupNames (GammaKeepPrev ++ [(x, tx)] ++ ([] : LinearCtx)) := by
              simpa using hndKeep
            simpa using
              (noDupNames_middle_fresh_prefix
                (GammaPre := GammaKeepPrev) (GammaPost := ([] : LinearCtx))
                (x := x) (t := tx) hndKeep')
          have hzx : x ≠ z := by
            intro hEq
            subst hEq
            have hzPrev' : x ∈ linearCtxDom GammaKeepPrev := by
              simpa [linearCtxDom] using hzPrev
            exact hxKeepPrev hzPrev'
          have hndPrev : NoDupNames GammaPrev := by
            exact noDupNames_of_sublist
              (show List.Sublist GammaPrev (GammaPrev ++ [(x, tx)]) from by simp) hnd
          have hzPrev' : z ∈ linearCtxDom GammaKeepPrev := by
            simpa [linearCtxDom] using hzPrev
          have hxPrev : x ∉ linearCtxDom GammaPrev := by
            have hnd' : NoDupNames (GammaPrev ++ [(x, tx)] ++ ([] : LinearCtx)) := by
              simpa using hnd
            simpa using
              (noDupNames_middle_fresh_prefix
                (GammaPre := GammaPrev) (GammaPost := ([] : LinearCtx))
                (x := x) (t := tx) hnd')
          rw [hshape]
          have hxLive : x ∈ linearCtxDom (GammaKeepPrev ++ [(x, tx)]) := by
            simp [linearCtxDom]
          simp [restrictSlots, lookupBinderSlots, hzx, hxLive]
          rw [restrictSlots_append_singleton_fresh hprev hxPrev]
          exact congrArg (Option.map Nat.succ) (ih hsubPrev hndPrev hzPrev')
        · rw [hshape]
          simp [restrictSlots, lookupBinderSlots, linearCtxDom]

theorem slotCorr_restrictSlots
    {GammaIn GammaOut : LinearCtx} {rho : BinderSlots}
    (hslot : SlotCorr GammaIn rho)
    (hsub : List.Sublist GammaOut GammaIn)
    (hnd : NoDupNames GammaIn) :
    SlotCorr GammaOut (restrictSlots GammaOut rho) := by
  induction hslot generalizing GammaOut with
  | nil =>
      cases hsub
      exact SlotCorr.nil
  | dead hprev ih =>
      exact SlotCorr.dead (ih hsub hnd)
  | @live GammaPrev rhoPrev y t hprev ih =>
      rcases
          sublist_append_singleton_cases
            (Gamma := GammaPrev) (Gamma' := GammaOut) (x := y) (t := t) hsub with
        hdrop | ⟨GammaOutPrev, hshape, hsubPrev⟩
      · have hyPrev : y ∉ linearCtxDom GammaPrev := by
          have hnd' : NoDupNames (GammaPrev ++ [(y, t)] ++ ([] : LinearCtx)) := by
            simpa using hnd
          simpa using
            (noDupNames_middle_fresh_prefix
              (GammaPre := GammaPrev) (GammaPost := ([] : LinearCtx))
              (x := y) (t := t) hnd')
        have hyOut : y ∉ linearCtxDom GammaOut :=
          fresh_of_sublist hdrop hyPrev
        simp [restrictSlots, hyOut]
        have hndPrev : NoDupNames GammaPrev := by
          exact noDupNames_of_sublist
            (show List.Sublist GammaPrev (GammaPrev ++ [(y, t)]) from by simp) hnd
        exact SlotCorr.dead (ih hdrop hndPrev)
      · rw [hshape]
        have hndPrev : NoDupNames GammaPrev := by
          exact noDupNames_of_sublist
            (show List.Sublist GammaPrev (GammaPrev ++ [(y, t)]) from by simp) hnd
        have hyPrev : y ∉ linearCtxDom GammaPrev := by
          have hndPrev' : NoDupNames (GammaPrev ++ [(y, t)] ++ ([] : LinearCtx)) := by
            simpa using hnd
          simpa using
            (noDupNames_middle_fresh_prefix
              (GammaPre := GammaPrev) (GammaPost := ([] : LinearCtx))
              (x := y) (t := t) hndPrev')
        have hyLive : y ∈ linearCtxDom (GammaOutPrev ++ [(y, t)]) := by
          simp [linearCtxDom]
        simp [restrictSlots, hyLive]
        rw [restrictSlots_append_singleton_fresh hprev hyPrev]
        exact SlotCorr.live (x := y) (t := t) (ih hsubPrev hndPrev)

theorem lexical_letBind_body
    {Gamma : LinearCtx} {x : String} {tx : Typ} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letBind x e1 e2)) :
    LexicallyScoped (Gamma ++ [(x, tx)]) e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_letBind_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2.2
  · subst y
    exact hbody.2.1

theorem lexical_letpair_body
    {Gamma : LinearCtx} {x y : String} {tx ty : Typ} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letpair x y e1 e2)) :
    LexicallyScoped (Gamma ++ [(x, tx), (y, ty)]) e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hyGamma : y ∉ linearCtxDom Gamma := by
    intro hy
    have hnot := hdom y hy
    simp [boundVars] at hnot
  have hbody := wellScoped_letpair_body hws
  have hnd' : NoDupNames (Gamma ++ [(x, tx)]) :=
    noDupNames_append_singleton hnd hxGamma
  have hyGamma' : y ∉ linearCtxDom (Gamma ++ [(x, tx)]) := by
    intro hy
    simp [linearCtxDom] at hy
    rcases hy with hy | hy
    · exact hyGamma (by simpa [linearCtxDom] using hy)
    · exact hbody.2.1 hy.symm
  have hnd'' : NoDupNames (Gamma ++ [(x, tx)] ++ [(y, ty)]) :=
    noDupNames_append_singleton hnd' hyGamma'
  simpa [List.append_assoc] using
    (show LexicallyScoped (Gamma ++ [(x, tx)] ++ [(y, ty)]) e2 from by
      refine ⟨hnd'', ?_, hbody.2.2.2.2⟩
      intro z hz
      simp [linearCtxDom] at hz
      rcases hz with hz | hz
      · have hz' : z ∈ linearCtxDom Gamma := by
          simpa [linearCtxDom] using hz
        have hnot := hdom z hz'
        simp [boundVars] at hnot
        exact hnot.2.2.2
      · rcases hz with hz | hz
        · subst z
          exact hbody.2.2.1
        · subst z
          exact hbody.2.2.2.1)

theorem lexical_grad_body
    {Gamma : LinearCtx} {x : String} {tx tOut : Typ} {body : Term}
    (h : LexicallyScoped Gamma (Term.grad x tx tOut body)) :
    LexicallyScoped (Gamma ++ [(x, tx)]) body := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_grad_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2
  · subst y
    exact hbody.1

theorem lexical_vmap_body
    {Gamma : LinearCtx} {x : String} {tx : Typ} {body : Term}
    (h : LexicallyScoped Gamma (Term.vmap x tx body)) :
    LexicallyScoped (Gamma ++ [(x, tx)]) body := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_vmap_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2
  · subst y
    exact hbody.1

private theorem mem_boundVarsClauses_arg
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses) :
    x ∈ boundVarsClauses clauses := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [boundVarsClauses]
      · simp [boundVarsClauses, ih htl]

private theorem mem_boundVarsClauses_cont
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses) :
    k ∈ boundVarsClauses clauses := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [boundVarsClauses]
      · simp [boundVarsClauses, ih htl]

private theorem mem_boundVarsClauses_body
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k z : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses)
    (hz : z ∈ boundVars hb) :
    z ∈ boundVarsClauses clauses := by
  induction clauses generalizing op x k hb z with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [boundVarsClauses, hz]
      · simp [boundVarsClauses, ih htl hz]

theorem lexical_handle_clause
    {Gamma : LinearCtx} {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    {tx tk : Typ}
    (h : LexicallyScoped Gamma (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    LexicallyScoped (Gamma ++ [(x, tx), (k, tk)]) hb := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hclause : x ≠ k ∧ x ∉ boundVars hb ∧ k ∉ boundVars hb ∧ WellScoped hb :=
    wellScoped_handle_clause hws hmem
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars, List.mem_append, not_or] at hnot
    exact hnot.2 (mem_boundVarsClauses_arg hmem)
  have hkGamma : k ∉ linearCtxDom Gamma := by
    intro hk
    have hnot := hdom k hk
    simp [boundVars, List.mem_append, not_or] at hnot
    exact hnot.2 (mem_boundVarsClauses_cont hmem)
  have hnd' : NoDupNames (Gamma ++ [(x, tx)]) :=
    noDupNames_append_singleton hnd hxGamma
  have hkGamma' : k ∉ linearCtxDom (Gamma ++ [(x, tx)]) := by
    intro hk
    simp [linearCtxDom] at hk
    rcases hk with hk | hk
    · exact hkGamma (by simpa [linearCtxDom] using hk)
    · exact hclause.1 hk.symm
  have hnd'' : NoDupNames (Gamma ++ [(x, tx)] ++ [(k, tk)]) :=
    noDupNames_append_singleton hnd' hkGamma'
  simpa [List.append_assoc] using
    (show LexicallyScoped (Gamma ++ [(x, tx)] ++ [(k, tk)]) hb from by
      refine ⟨hnd'', ?_, hclause.2.2.2⟩
      intro z hz
      simp [linearCtxDom] at hz
      rcases hz with hz | hz
      · have hz' : z ∈ linearCtxDom Gamma := by
          simpa [linearCtxDom] using hz
        have hnot := hdom z hz'
        simp [boundVars, List.mem_append, not_or] at hnot
        intro hzb
        exact hnot.2 (mem_boundVarsClauses_body hmem hzb)
      · rcases hz with hz | hz
        · subst z
          exact hclause.2.1
        · subst z
          exact hclause.2.2.1)

/-- Abstraction inversion, stripping any outer `subEff`. The outer
    effect row may have been widened from `[]`, but the underlying
    abstraction body derivation is unchanged. -/
theorem HasType.abs_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x : String} {t1 : Typ} {body : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.abs x t1 body) t eps GammaOut) :
    ∃ t2 epsBody GammaBody,
      t = Typ.arrow t1 t2 epsBody ∧
      HasType Delta Sigma (Gamma1 ++ [(x, t1)]) body t2 epsBody GammaBody ∧
      GammaOut = GammaBody.filter (fun p => p.1 ≠ x) := by
  generalize heq : Term.abs x t1 body = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | abs _ _ _ GammaBody _ _ t2 epsBody _ hBody =>
      cases heq
      exact ⟨t2, epsBody, GammaBody, rfl, hBody, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- App inversion local to the DB bridge. `TranslationDB` cannot import
    `Progress.lean` because `Progress` already imports `Preservation`,
    which depends on this module. -/
theorem HasType.app_inv_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.app e1 e2) t eps Gamma3) :
    ∃ Gamma2 t1 eps1 eps2 epsInner,
      HasType Delta Sigma Gamma1 e1 (Typ.arrow t1 t epsInner) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t1 eps2 Gamma3 := by
  generalize heq : Term.app e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | app _ _ _ Gamma2 _ _ _ t1 _ epsInner eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, t1, eps1, eps2, epsInner, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetBind inversion local to the DB bridge. -/
theorem HasType.letBind_inv_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps GammaOut) :
    ∃ Gamma2 Gamma3 t1 eps1 eps2,
      GammaOut = Gamma3.filter (fun p => p.1 ≠ x) ∧
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, t1)]) e2 t eps2 Gamma3 := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Gamma2 Gamma3 _ _ _ t1 _ eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, Gamma3, t1, eps1, eps2, rfl, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetPair inversion local to the DB bridge. -/
theorem HasType.letpair_inv_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps GammaOut) :
    ∃ Gamma2 Gamma3 t1 t2 eps1 eps2,
      GammaOut = Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y) ∧
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, t1), (y, t2)]) e2 t eps2 Gamma3 := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Gamma2 Gamma3 _ _ _ _ t1 t2 _ eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, rfl, h1, h2⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Strong handle inversion local to the DB bridge. -/
theorem HasType.handle_inv_strong_bridge
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
  | handle _ _ _ Gamma2 _ _ _ _ _ epsB hBody hOpsIn hClsIn hCover hClauses _ _ =>
      cases heq
      exact ⟨Gamma2, epsB, hBody, hOpsIn, hClsIn, hCover, hClauses, fun _ hop => hop⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨Gamma2, epsB, hBody, hOpsIn, hClsIn, hCover, hClauses, hSub'⟩ := ih heq
      refine ⟨Gamma2, epsB, hBody, hOpsIn, hClsIn, hCover, hClauses, ?_⟩
      intro op hop
      exact hSub op (hSub' op hop)
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Membership inversion for handler clauses. Extract the typing
    derivation attached to a specific clause body from `ClausesTyped`. -/
theorem ClausesTyped.mem_inv
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hcls : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR clauses)
    (hmem : (op, x, k, hb) ∈ clauses) :
    ∃ tArg tRet,
      HasType Delta Sigma
        (Gamma2 ++ [(x, tArg), (k, Typ.arrow tRet t epsR)])
        hb t epsR Gamma3 := by
  induction clauses generalizing Gamma2 Gamma3 with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      cases hcls with
      | cons _ _ _ _ _ tArg tRet _ _ x' k' hb' rest' hHead hRest =>
          rcases List.mem_cons.mp hmem with h0 | htl
          · cases h0
            exact ⟨tArg, tRet, hHead⟩
          · exact ih hRest htl

/-- Pair-level shrinkage: every output binding is literally one of the
    input bindings, with the same name and type. This is the stronger
    form of `has_type_linear_shrinks` needed by the narrow DB bridge,
    because the singleton / two-slot wrapper bodies must recover the
    exact surviving slot types, not just the surviving names. -/
def CtxPairSub (G1 G2 : LinearCtx) : Prop :=
  ∀ p, p ∈ G1 → p ∈ G2

private theorem mem_of_mem_append_singleton_ne
    {Gamma : LinearCtx} {p : String × Typ} {x : String} {t : Typ}
    (hmem : p ∈ Gamma ++ [(x, t)])
    (hne : p.1 ≠ x) :
    p ∈ Gamma := by
  simp [List.mem_append] at hmem
  rcases hmem with hmem | hmem
  · exact hmem
  · exfalso
    have : p.1 = x := by
      simpa using congrArg Prod.fst hmem
    exact hne this

private theorem mem_of_mem_append_pair_ne
    {Gamma : LinearCtx} {p : String × Typ}
    {x y : String} {tx ty : Typ}
    (hmem : p ∈ Gamma ++ [(x, tx), (y, ty)])
    (hne_x : p.1 ≠ x) (hne_y : p.1 ≠ y) :
    p ∈ Gamma := by
  simp [List.mem_append] at hmem
  rcases hmem with hmem | hmem
  · exact hmem
  · have hmem' : p = (x, tx) ∨ p = (y, ty) := by
        simpa using hmem
    rcases hmem' with hmem | hmem
    · exfalso
      have : p.1 = x := by
        simpa using congrArg Prod.fst hmem
      exact hne_x this
    · exfalso
      have : p.1 = y := by
        simpa using congrArg Prod.fst hmem
      exact hne_y this

theorem has_type_pair_shrinks
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    CtxPairSub Gamma' Gamma := by
  induction h using HasType.rec
    (motive_2 := fun (_Δ : CapCtx) (_S : StoreTyp)
                     (Γ2 Γ3 : LinearCtx) (_ : Typ) (_ : EffectRow)
                     (_ : List (EffectLabel × String × String × Term))
                     (_ : _) => CtxPairSub Γ3 Γ2) with
  | var _ _ Γpre Γpost y t_v =>
      intro p hp
      simp [List.mem_append] at hp ⊢
      rcases hp with hp | hp
      · exact Or.inl hp
      · exact Or.inr (Or.inr hp)
  | unit _ _ _ =>
      intro p hp
      exact hp
  | abs _ _ Γ1 Γ2 y t1 _ _ _ _ ih =>
      intro p hp
      have hp2 : p ∈ Γ2 := (List.mem_filter.mp hp).1
      have hp1 : p ∈ Γ1 ++ [(y, t1)] := ih p hp2
      have hne : p.1 ≠ y := by
        simpa using (List.mem_filter.mp hp).2
      exact mem_of_mem_append_singleton_ne hp1 hne
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro p hp
      exact ih1 p (ih2 p hp)
  | letBind _ _ _ Gamma2 Γ3 y _ _ t1 _ _ _ _ _ ih1 ih2 =>
      intro p hp
      have hp3 : p ∈ Γ3 := (List.mem_filter.mp hp).1
      have hp2 : p ∈ Gamma2 ++ [(y, t1)] := ih2 p hp3
      have hne : p.1 ≠ y := by
        simpa using (List.mem_filter.mp hp).2
      have hp1 : p ∈ Gamma2 := mem_of_mem_append_singleton_ne hp2 hne
      exact ih1 p hp1
  | copy _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | letpair _ _ _ Gamma2 Γ3 x y _ _ t1 t2 _ _ _ _ _ ih1 ih2 =>
      intro p hp
      have hp3 : p ∈ Γ3 := (List.mem_filter.mp hp).1
      have hp2 : p ∈ Gamma2 ++ [(x, t1), (y, t2)] := ih2 p hp3
      have hne : p.1 ≠ x ∧ p.1 ≠ y := by
        simpa using (List.mem_filter.mp hp).2
      have hp1 : p ∈ Gamma2 :=
        mem_of_mem_append_pair_ne hp2 hne.1 hne.2
      exact ih1 p hp1
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro p hp
      exact ih1 p (ih2 p hp)
  | fst _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | snd _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | const _ _ _ _ _ =>
      intro p hp
      exact hp
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro p hp
      exact ih1 p (ih2 p hp)
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro p hp
      exact ih1 p (ih2 p hp)
  | tsum _ _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | texpand _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | perform _ _ _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihBody ihClauses =>
      intro p hp
      exact ihBody p (ihClauses p hp)
  | tgrad _ _ _ _ _ _ _ _ _ _ _ =>
      intro p hp
      exact hp
  | tvmap _ _ _ _ _ _ _ _ _ _ =>
      intro p hp
      exact hp
  | loc _ _ _ _ _ _ =>
      intro p hp
      exact hp
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro p hp
      exact ih p hp
  | nil _ _ _ _ _ =>
      intro p hp
      exact hp
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih

theorem singleton_output_pair_eq
    {Sigma : StoreTyp} {x : String} {tx : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx} {p : String × Typ}
    (h : HasType [] Sigma [(x, tx)] e t eps GammaOut)
    (hp : p ∈ GammaOut) :
    p = (x, tx) := by
  have hpair := has_type_pair_shrinks h
  have hp0 : p ∈ [(x, tx)] := hpair p hp
  simpa using hp0

theorem singleton_output_shape
    {Sigma : StoreTyp} {x : String} {tx : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, tx)] e t eps GammaOut) :
    GammaOut = [] ∨ GammaOut = [(x, tx)] := by
  have hsub : List.Sublist GammaOut ([(x, tx)] : LinearCtx) := has_type_sublist h
  cases GammaOut with
  | nil =>
      exact Or.inl rfl
  | cons p rest =>
      have hp_eq : p = (x, tx) := singleton_output_pair_eq h (by simp)
      subst hp_eq
      cases rest with
      | nil =>
          exact Or.inr rfl
      | cons q rest' =>
          exfalso
          have hle := hsub.length_le
          simp at hle

theorem singleton_output_shape_of_noDup
    {Sigma : StoreTyp} {x : String} {tx : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, tx)] e t eps GammaOut)
    (_hnd : NoDupNames GammaOut) :
    GammaOut = [] ∨ GammaOut = [(x, tx)] :=
  singleton_output_shape h

theorem pair_output_shape
    {Sigma : StoreTyp} {x y : String} {tx ty : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, tx), (y, ty)] e t eps GammaOut) :
    GammaOut = [] ∨
      GammaOut = [(x, tx)] ∨
      GammaOut = [(y, ty)] ∨
      GammaOut = [(x, tx), (y, ty)] := by
  have hsub : List.Sublist GammaOut [(x, tx), (y, ty)] := has_type_sublist h
  cases GammaOut with
  | nil =>
      exact Or.inl rfl
  | cons p rest =>
      cases rest with
      | nil =>
          have hp : p ∈ [(x, tx), (y, ty)] := hsub.subset (by simp)
          simp at hp
          rcases hp with rfl | rfl
          · exact Or.inr (Or.inl rfl)
          · exact Or.inr (Or.inr (Or.inl rfl))
      | cons q rest' =>
          cases rest' with
          | nil =>
              have hlen : (p :: q :: []).length = ([(x, tx), (y, ty)] : LinearCtx).length := by
                simp
              have heq : (p :: q :: []) = ([(x, tx), (y, ty)] : LinearCtx) :=
                hsub.eq_of_length hlen
              exact Or.inr (Or.inr (Or.inr heq))
          | cons r rest'' =>
              exfalso
              have hle := hsub.length_le
              simp at hle

theorem singleton_output_db_shape
    {Sigma : StoreTyp} {x : String} {tx : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, tx)] e t eps GammaOut) :
    ∃ GammaOutDB,
      CtxCorr GammaOut GammaOutDB ∧
      (GammaOutDB = [none] ∨ GammaOutDB = [some tx]) := by
  rcases singleton_output_shape h with rfl | rfl
  · exact ⟨[none], ctxCorr_singleton_dead x tx, Or.inl rfl⟩
  · exact ⟨[some tx], ctxCorr_singleton_live x tx, Or.inr rfl⟩

theorem pair_output_db_shape
    {Sigma : StoreTyp} {x y : String} {tx ty : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, tx), (y, ty)] e t eps GammaOut) :
    ∃ GammaOutDB,
      CtxCorr GammaOut GammaOutDB ∧
      (GammaOutDB = [none, none] ∨
        GammaOutDB = [none, some tx] ∨
        GammaOutDB = [some ty, none] ∨
        GammaOutDB = [some ty, some tx]) := by
  rcases pair_output_shape h with rfl | rfl | rfl | rfl
  · exact ⟨[none, none], ctxCorr_pair_dead x tx y ty, Or.inl rfl⟩
  · exact ⟨[none, some tx], ctxCorr_pair_left_live x tx y ty, Or.inr <| Or.inl rfl⟩
  · exact ⟨[some ty, none], ctxCorr_pair_right_live x tx y ty, Or.inr <| Or.inr <| Or.inl rfl⟩
  · exact ⟨[some ty, some tx], ctxCorr_pair_live x tx y ty, Or.inr <| Or.inr <| Or.inr rfl⟩

mutual

theorem eraseTerm_liftAux_len :
    ∀ {ρ : BinderEnv} {e : Term} {eDB : TermDB},
      eraseTerm ρ e = some eDB ->
      liftAux ρ.length 1 eDB = eDB
  | ρ, Term.var x, eDB, h => by
      cases hρx : lookupBinder ρ x with
      | none =>
          simp [eraseTerm, hρx] at h
      | some i =>
          simp [eraseTerm, hρx] at h
          cases h
          simpa using
            (liftAux_var_lt (c := ρ.length) (d := 1) (i := i)
              (lookupBinder_some_lt_length hρx))
  | ρ, Term.abs x t body, eDB, h => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      cases h
      simpa [liftAux] using congrArg (TermDB.abs t) (eraseTerm_liftAux_len hbody)
  | ρ, Term.app e1 e2, eDB, h => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len h1, eraseTerm_liftAux_len h2]
  | ρ, Term.letBind x e1 e2, eDB, h => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm (x :: ρ) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      have ih1 := eraseTerm_liftAux_len h1
      have ih2 : liftAux (ρ.length + 1) 1 e2DB = e2DB := by
        simpa using eraseTerm_liftAux_len h2
      simp [liftAux, ih1, ih2]
  | ρ, Term.copy e, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len he]
  | ρ, Term.letpair x y e1 e2, eDB, h => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm (y :: x :: ρ) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      have ih1 := eraseTerm_liftAux_len h1
      have ih2 : liftAux (ρ.length + 2) 1 e2DB = e2DB := by
        simpa using eraseTerm_liftAux_len h2
      simp [liftAux, ih1, ih2]
  | ρ, Term.pair e1 e2, eDB, h => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len h1, eraseTerm_liftAux_len h2]
  | ρ, Term.fst e, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len he]
  | ρ, Term.snd e, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len he]
  | ρ, Term.unit, eDB, h => by
      simp [eraseTerm] at h
      cases h
      simp [liftAux]
  | ρ, Term.const c ds, eDB, h => by
      simp [eraseTerm] at h
      cases h
      simp [liftAux]
  | ρ, Term.add e1 e2, eDB, h => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len h1, eraseTerm_liftAux_len h2]
  | ρ, Term.mul e1 e2, eDB, h => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at h
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len h1, eraseTerm_liftAux_len h2]
  | ρ, Term.sum e d, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at h
        cases h
      · simp [eraseTerm, he] at h
        cases h
        simpa [liftAux] using
          congrArg (fun e => TermDB.sum e (eraseDim d)) (eraseTerm_liftAux_len he)
  | ρ, Term.expand e d, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at h
        cases h
      · simp [eraseTerm, he] at h
        cases h
        simpa [liftAux] using
          congrArg (fun e => TermDB.expand e (eraseDim d) 0) (eraseTerm_liftAux_len he)
  | ρ, Term.uniformLike e lo hi, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at h
        cases h
      · simp [eraseTerm, he] at h
        cases h
        simpa [liftAux] using
          congrArg (fun e => TermDB.uniformLike e lo hi) (eraseTerm_liftAux_len he)
  | ρ, Term.grad x t tOut body, eDB, h => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      cases h
      simpa [liftAux] using
        congrArg (TermDB.grad t tOut) (eraseTerm_liftAux_len hbody)
  | ρ, Term.vmap x t body, eDB, h => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      cases h
      simpa [liftAux] using
        congrArg (TermDB.vmap t) (eraseTerm_liftAux_len hbody)
  | ρ, Term.handle epsH body clauses, eDB, h => by
      rcases hbody : eraseTerm ρ body with _ | bodyDB <;> simp [eraseTerm, hbody] at h
      rcases hcls : eraseClauses ρ clauses with _ | clsDB <;> simp [eraseTerm, hbody, hcls] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len hbody, eraseClauses_liftAux_len hcls]
  | ρ, Term.perform op e, eDB, h => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at h
      cases h
      simp [liftAux, eraseTerm_liftAux_len he]
  | ρ, Term.loc ell, eDB, h => by
      simp [eraseTerm] at h
      cases h
      simp [liftAux]

theorem eraseClauses_liftAux_len :
    ∀ {ρ : BinderEnv}
      {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses ρ clauses = some clausesDB ->
      liftClausesAux ρ.length 1 clausesDB = clausesDB
  | ρ, [], clausesDB, h => by
      simp [eraseClauses] at h
      cases h
      simp [liftClausesAux]
  | ρ, (op, x, k, hb) :: rest, clausesDB, h => by
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;> simp [eraseClauses, hhb] at h
      rcases hrest : eraseClauses ρ rest with _ | restDB <;> simp [eraseClauses, hhb, hrest] at h
      cases h
      have ihb : liftAux (ρ.length + 2) 1 hbDB = hbDB := by
        simpa using eraseTerm_liftAux_len hhb
      have ihr := eraseClauses_liftAux_len hrest
      simp [liftClausesAux, ihb, ihr]

end

mutual

theorem eraseTerm_subst_tail
    {ρ : BinderEnv} {v : Term} {x : String} {vDB : TermDB}
    (hρ : x ∉ ρ)
    (hv : eraseTerm [] v = some vDB) :
    ∀ {body : Term} {bodyDB : TermDB},
      eraseTerm (ρ ++ [x]) body = some bodyDB ->
      x ∉ boundVars body ->
      eraseTerm ρ (subst body v x) = some (substDBAux ρ.length vDB bodyDB)
  | Term.var y, bodyDB, hbody, hx => by
      by_cases hxy : y = x
      · subst hxy
        simp [eraseTerm, lookupBinder_append_target hρ] at hbody
        cases hbody
        have hvρ : eraseTerm ρ v = some vDB := by
          simpa using eraseTerm_suffix hv ρ
        simpa [subst, substDBAux] using hvρ
      · cases hρy : lookupBinder ρ y with
        | none =>
            simp [eraseTerm, hxy, lookupBinder_append_singleton_ne hxy, hρy] at hbody
        | some i =>
            have hi : i < ρ.length := lookupBinder_some_lt_length hρy
            simp [eraseTerm, subst, hxy, lookupBinder_append_singleton_ne hxy, hρy] at hbody ⊢
            cases hbody
            simpa [substDBAux, Nat.ne_of_lt hi, hi] using rfl
  | Term.abs y t body, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases hbody' : eraseTerm (y :: (ρ ++ [x])) body with _ | bodyDB' <;> simp [eraseTerm, hbody'] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hρ' : x ∉ y :: ρ := by
        simp [hx.1, hρ]
      have ih := eraseTerm_subst_tail (ρ := y :: ρ) (v := v) (x := x) hρ' hv hbody' hx.2
      have ih' :
          eraseTerm (y :: ρ) (subst body v x) =
            some (substDBAux (ρ.length + 1) vDB bodyDB') := by
        simpa using ih
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih']
  | Term.app e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρ ++ [x]) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρ ++ [x]) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_tail hρ hv h1 hx.1
      have ih2 := eraseTerm_subst_tail hρ hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.letBind y e1 e2, bodyDB, hbody, hx => by
      simp [boundVars, List.mem_append, not_or] at hx
      rcases h1 : eraseTerm (ρ ++ [x]) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (y :: (ρ ++ [x])) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have ih1 := eraseTerm_subst_tail hρ hv h1 hx.2.1
      have hρ' : x ∉ y :: ρ := by
        simp [hx.1, hρ]
      have ih2 := eraseTerm_subst_tail (ρ := y :: ρ) (v := v) (x := x) hρ' hv h2 hx.2.2
      have ih2' :
          eraseTerm (y :: ρ) (subst e2 v x) =
            some (substDBAux (ρ.length + 1) vDB e2DB) := by
        simpa using ih2
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih1, ih2']
  | Term.copy e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_tail hρ hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.letpair y z e1 e2, bodyDB, hbody, hx => by
      simp [boundVars, List.mem_append, not_or] at hx
      rcases h1 : eraseTerm (ρ ++ [x]) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (z :: y :: (ρ ++ [x])) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hzx : z ≠ x := by
        intro h
        exact hx.2.1 h.symm
      have ih1 := eraseTerm_subst_tail hρ hv h1 hx.2.2.1
      have hρ' : x ∉ z :: y :: ρ := by
        simp [hx.1, hx.2.1, hρ]
      have ih2 := eraseTerm_subst_tail (ρ := z :: y :: ρ) (v := v) (x := x) hρ' hv h2 hx.2.2.2
      have ih2' :
          eraseTerm (z :: y :: ρ) (subst e2 v x) =
            some (substDBAux (ρ.length + 2) vDB e2DB) := by
        simpa using ih2
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, hzx, substDBAux, hvlift, ih1, ih2']
  | Term.pair e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρ ++ [x]) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρ ++ [x]) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_tail hρ hv h1 hx.1
      have ih2 := eraseTerm_subst_tail hρ hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.fst e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_tail hρ hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.snd e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_tail hρ hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.unit, bodyDB, hbody, hx => by
      simp [eraseTerm] at hbody
      cases hbody
      simp [eraseTerm, subst, substDBAux]
  | Term.const c ds, bodyDB, hbody, hx => by
      simp [eraseTerm] at hbody
      cases hbody
      simp [eraseTerm, subst, substDBAux]
  | Term.add e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρ ++ [x]) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρ ++ [x]) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_tail hρ hv h1 hx.1
      have ih2 := eraseTerm_subst_tail hρ hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.mul e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρ ++ [x]) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρ ++ [x]) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_tail hρ hv h1 hx.1
      have ih2 := eraseTerm_subst_tail hρ hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.sum e d, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB
      · simp [eraseTerm, he] at hbody
        cases hbody
      · simp [eraseTerm, he] at hbody
        cases hbody
        have ih := eraseTerm_subst_tail hρ hv he hx
        simp [eraseTerm, subst, ih]
        rfl
  | Term.expand e d, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB
      · simp [eraseTerm, he] at hbody
        cases hbody
      · simp [eraseTerm, he] at hbody
        cases hbody
        have ih := eraseTerm_subst_tail hρ hv he hx
        simp [eraseTerm, subst, ih]
        rfl
  | Term.uniformLike e lo hi, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB
      · simp [eraseTerm, he] at hbody
        cases hbody
      · simp [eraseTerm, he] at hbody
        cases hbody
        have ih := eraseTerm_subst_tail hρ hv he hx
        simp [eraseTerm, subst, ih]
        rfl
  | Term.grad y t tOut body, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases hbody' : eraseTerm (y :: (ρ ++ [x])) body with _ | bodyDB' <;> simp [eraseTerm, hbody'] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hρ' : x ∉ y :: ρ := by
        simp [hx.1, hρ]
      have ih := eraseTerm_subst_tail (ρ := y :: ρ) (v := v) (x := x) hρ' hv hbody' hx.2
      have ih' :
          eraseTerm (y :: ρ) (subst body v x) =
            some (substDBAux (ρ.length + 1) vDB bodyDB') := by
        simpa using ih
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih']
  | Term.vmap y t body, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases hbody' : eraseTerm (y :: (ρ ++ [x])) body with _ | bodyDB' <;> simp [eraseTerm, hbody'] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hρ' : x ∉ y :: ρ := by
        simp [hx.1, hρ]
      have ih := eraseTerm_subst_tail (ρ := y :: ρ) (v := v) (x := x) hρ' hv hbody' hx.2
      have ih' :
          eraseTerm (y :: ρ) (subst body v x) =
            some (substDBAux (ρ.length + 1) vDB bodyDB') := by
        simpa using ih
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih']
  | Term.handle epsH body clauses, bodyDB, hbody, hx => by
      simp [boundVars, List.mem_append, not_or] at hx
      rcases hbody' : eraseTerm (ρ ++ [x]) body with _ | bodyDB' <;> simp [eraseTerm, hbody'] at hbody
      rcases hcls : eraseClauses (ρ ++ [x]) clauses with _ | clsDB <;> simp [eraseTerm, hbody', hcls] at hbody
      cases hbody
      have ihBody := eraseTerm_subst_tail hρ hv hbody' hx.1
      have ihClauses := eraseClauses_subst_tail hρ hv hcls hx.2
      simp [eraseTerm, subst, substDBAux, ihBody, ihClauses]
  | Term.perform op e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρ ++ [x]) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_tail hρ hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.loc ell, bodyDB, hbody, hx => by
      simp [eraseTerm] at hbody
      cases hbody
      simp [eraseTerm, subst, substDBAux]

theorem eraseClauses_subst_tail
    {ρ : BinderEnv} {v : Term} {x : String} {vDB : TermDB}
    (hρ : x ∉ ρ)
    (hv : eraseTerm [] v = some vDB) :
    ∀ {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses (ρ ++ [x]) clauses = some clausesDB ->
      x ∉ boundVarsClauses clauses ->
      eraseClauses ρ (substClauses clauses v x) =
        some (substClausesDBAux ρ.length vDB clausesDB)
  | [], clausesDB, hclauses, hx => by
      simp [eraseClauses] at hclauses
      cases hclauses
      simp [eraseClauses, substClauses, substClausesDBAux]
  | (op, y, k, hb) :: rest, clausesDB, hclauses, hx => by
      simp [boundVarsClauses, List.mem_append, not_or] at hx
      rcases hhb : eraseTerm (k :: y :: (ρ ++ [x])) hb with _ | hbDB <;> simp [eraseClauses, hhb] at hclauses
      rcases hrest : eraseClauses (ρ ++ [x]) rest with _ | restDB <;> simp [eraseClauses, hhb, hrest] at hclauses
      cases hclauses
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hkx : k ≠ x := by
        intro h
        exact hx.2.1 h.symm
      have hρ' : x ∉ k :: y :: ρ := by
        simp [hx.2.1, hx.1, hρ]
      have ihBody := eraseTerm_subst_tail (ρ := k :: y :: ρ) (v := v) (x := x) hρ' hv hhb hx.2.2.1
      have ihBody' :
          eraseTerm (k :: y :: ρ) (subst hb v x) =
            some (substDBAux (ρ.length + 2) vDB hbDB) := by
        simpa using ihBody
      have ihRest := eraseClauses_subst_tail hρ hv hrest hx.2.2.2
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseClauses, substClauses, hyx, hkx, substClausesDBAux, hvlift, ihBody', ihRest]

end

/-- Named-facing wrapper for the beta redex case in preservation.
    Completion path:
      1. extract the body derivation under the singleton named context,
      2. transport it to DB with the derivation-guided erasure layer,
      3. apply `subst_preserves_typing_db_gen` at cutoff `0`,
      4. reflect the resulting closed DB derivation back to named typing.

    The active blocker is step (2): with duplicate binder names,
    `eraseTerm [x] body` need not follow the same binding choices as an
    arbitrary named `HasType` derivation because `HasType.var` is
    non-lexical. -/
theorem preservation_beta_via_db
    {Sigma : StoreTyp}
    {x : String} {tArg tRet : Typ}
    {body v : Term} {eps : EffectRow}
    (h_typ : HasType [] Sigma [] (Term.app (Term.abs x tArg body) v) tRet eps [])
    (hv : IsValue v)
    (h_scope : WellScoped (Term.app (Term.abs x tArg body) v)) :
    HasType [] Sigma [] (subst body v x) tRet eps [] := by
  obtain ⟨GammaMid, tArg', epsFun, epsArg, epsBody, hAbs, hV⟩ :=
    HasType.app_inv_bridge h_typ
  have hMid : GammaMid = [] := has_type_closed_output_of_closed_input hAbs
  subst hMid
  obtain ⟨tRet', epsBody', GammaBody, hArrow, hBody, hOut⟩ :=
    HasType.abs_inv hAbs
  cases hArrow
  simp [WellScoped, boundVars, List.nodup_append] at h_scope
  have hAbsScope : WellScoped (Term.abs x tArg body) := by
    rw [wellScoped_abs_iff]
    exact ⟨h_scope.1.1, h_scope.2.1⟩
  have hBodyScope : x ∉ boundVars body ∧ WellScoped body :=
    wellScoped_abs_body hAbsScope
  have hBodyLex : LexicallyScoped [(x, tArg)] body :=
    lexical_singleton hBodyScope.1 hBodyScope.2
  -- Remaining work:
  -- 1. erase `hBody` and `hV` to DB under the scoped singleton context,
  -- 2. apply `subst_preserves_typing_db_gen`,
  -- 3. reflect the closed DB result back to named typing.
  sorry

/-- Named-facing wrapper for the let-binding redex case in
    preservation. Same blocker profile as `preservation_beta_via_db`. -/
theorem preservation_letBind_via_db
    {Sigma : StoreTyp}
    {x : String} {v body : Term} {t : Typ} {eps : EffectRow}
    (h_typ : HasType [] Sigma [] (Term.letBind x v body) t eps [])
    (hv : IsValue v)
    (h_scope : WellScoped (Term.letBind x v body)) :
    HasType [] Sigma [] (subst body v x) t eps [] := by
  obtain ⟨Gamma2, Gamma3, tArg, eps1, eps2, hOut, hV, hBody⟩ :=
    HasType.letBind_inv_bridge h_typ
  have hGamma2 : Gamma2 = [] := has_type_closed_output_of_closed_input hV
  subst hGamma2
  have hBodyScope : WellScoped v ∧ x ∉ boundVars body ∧ WellScoped body :=
    wellScoped_letBind_body h_scope
  have hValueLex : LexicallyScoped [] v :=
    lexical_nil hBodyScope.1
  have hBodyLex : LexicallyScoped [(x, tArg)] body :=
    lexical_singleton hBodyScope.2.1 hBodyScope.2.2
  -- Remaining work:
  -- 1. translate the closed value `v` to DB at cutoff `0`,
  -- 2. translate the body derivation under the singleton lexical stack `[x]`,
  -- 3. apply `subst_preserves_typing_db_gen` at cutoff `0`,
  -- 4. reflect the erased substituted term back to named typing.
  sorry

/-- Named-facing wrapper for the pair-destruct redex case in
    preservation. This needs the 2-binder version of the same
    derivation-guided transport. -/
theorem preservation_letpair_via_db
    {Sigma : StoreTyp}
    {x y : String} {v1 v2 body : Term} {t : Typ} {eps : EffectRow}
    (h_typ : HasType [] Sigma [] (Term.letpair x y (Term.pair v1 v2) body) t eps [])
    (hv1 : IsValue v1) (hv2 : IsValue v2)
    (h_scope : WellScoped (Term.letpair x y (Term.pair v1 v2) body)) :
    HasType [] Sigma [] (subst (subst body v1 x) v2 y) t eps [] := by
  obtain ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, hOut, hPair, hBody⟩ :=
    HasType.letpair_inv_bridge h_typ
  have hGamma2 : Gamma2 = [] := has_type_closed_output_of_closed_input hPair
  subst hGamma2
  have hBodyScope :
      WellScoped (Term.pair v1 v2) ∧
      x ≠ y ∧ x ∉ boundVars body ∧ y ∉ boundVars body ∧ WellScoped body :=
    wellScoped_letpair_body h_scope
  have hV1Scope : WellScoped v1 := wellScoped_pair_left hBodyScope.1
  have hV2Scope : WellScoped v2 := wellScoped_pair_right hBodyScope.1
  have hV1Lex : LexicallyScoped [] v1 := lexical_nil hV1Scope
  have hV2Lex : LexicallyScoped [] v2 := lexical_nil hV2Scope
  have hBodyLex : LexicallyScoped [(x, t1), (y, t2)] body :=
    lexical_pair hBodyScope.2.1 hBodyScope.2.2.1 hBodyScope.2.2.2.1 hBodyScope.2.2.2.2
  -- Remaining work:
  -- 1. translate `v1` / `v2` as closed DB values,
  -- 2. translate the body derivation under the two-binder lexical stack `[x, y]`,
  -- 3. apply DB substitution at cutoff `1` for `x`, then cutoff `0` for `y`,
  -- 4. reflect the resulting erased term back to the named double substitution.
  sorry

/-- Named-facing wrapper for the direct handled-operation redex case in
    preservation. This hides the clause-body DB substitution proof from
    `Preservation.lean`; the completed proof will route through
    `subst_preserves_typing_clauses_db_gen` plus the clause-specific
    reflection lemmas built over `CtxCorr`. The extra wrinkle here is
    that the clause body is under two binders (`k` at DB index `0`,
    argument `x` at DB index `1`). -/
theorem preservation_handleOpDirect_via_db
    {Sigma : StoreTyp}
    {op : EffectLabel} {v : Term} {epsH : EffectRow}
    {clauses : List (EffectLabel × String × String × Term)}
    {x k : String} {hb : Term} {tRet t : Typ} {eps : EffectRow}
    (h_typ : HasType [] Sigma [] (Term.handle epsH (Term.perform op v) clauses) t eps [])
    (hv : IsValue v)
    (hmem : (op, x, k, hb) ∈ clauses)
    (h_scope : WellScoped (Term.handle epsH (Term.perform op v) clauses)) :
    HasType [] Sigma []
      (subst (subst hb v x) (Term.abs "y" tRet (Term.var "y")) k)
      t eps [] := by
  obtain ⟨Gamma2, epsB, hBody, hOpsIn, hClsIn, hCover, hClauses, hSub⟩ :=
    HasType.handle_inv_strong_bridge h_typ
  obtain ⟨tArg, tRet', hClauseBody⟩ :=
    ClausesTyped.mem_inv hClauses hmem
  have hBodyOut : Gamma2 = [] := by
    have hPerf : HasType [] Sigma [] (Term.perform op v) t epsB Gamma2 := hBody
    exact has_type_closed_output_of_closed_input hPerf
  subst hBodyOut
  have hHandleBodyScope : WellScoped (Term.perform op v) := by
    simpa [WellScoped, boundVars, List.nodup_append] using (wellScoped_handle_body h_scope)
  have hClauseScope : x ≠ k ∧ x ∉ boundVars hb ∧ k ∉ boundVars hb ∧ WellScoped hb :=
    wellScoped_handle_clause h_scope hmem
  have hValueLex : LexicallyScoped [] v := by
    have hvScope : WellScoped v := by
      simpa [WellScoped, boundVars] using hHandleBodyScope
    exact lexical_nil hvScope
  have hClauseLex :
      LexicallyScoped [(x, tArg), (k, Typ.arrow tRet' t (EffectRow.removeOps epsB epsH))] hb :=
    lexical_pair hClauseScope.1 hClauseScope.2.1 hClauseScope.2.2.1 hClauseScope.2.2.2
  -- Remaining work:
  -- 1. transport `hClauseBody` under the lexical clause stack `[x, k]`
  --    using `hClauseScope` to rule out shadowing of either binder,
  -- 2. transport `hClauseBody` and `hV` to DB under the two-binder clause context,
  -- 3. apply `subst_preserves_typing_db_gen` twice: cutoff `1` for the
  --    operation argument `x`, then cutoff `0` for the continuation `k`,
  -- 4. reflect the erased result back to the named clause body substitution.
  sorry

end LaCaDiLE
