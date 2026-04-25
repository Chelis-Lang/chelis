-- LaCaDiLE/TranslationDB.lean
--
-- Thin named-facing bridge onto the finished de Bruijn substitution
-- metatheory. The end-state for this module is:
--   1. erase the named substitution-dependent preservation redexes to DB,
--   2. discharge them with `SubstitutionDB`,
--   3. reflect the result back to named `HasType`,
--   4. keep `Preservation.lean` free of direct `SubstitutionDB` imports.
--
-- Two repository-level blockers are still real:
--   * named→DB reflection now standardizes on tombstone-preserving
--     contexts via `CtxCorr`, but the transport proofs are still
--     incomplete;
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
import LaCaDiLE.Substitution
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

theorem lookupBinder_some_getElem
    {ρ : BinderEnv} {x : String} {i : Nat}
    (h : lookupBinder ρ x = some i) :
    ρ[i]? = some x := by
  induction ρ generalizing i with
  | nil =>
      simp [lookupBinder] at h
  | cons y ys ih =>
      by_cases hy : y = x
      · subst hy
        simp [lookupBinder] at h
        cases h
        simp
      · simp [lookupBinder, hy] at h
        rcases h with ⟨j, hj, rfl⟩
        simp [ih hj]

theorem lookupBinder_some_split
    {ρ : BinderEnv} {x : String} {i : Nat}
    (h : lookupBinder ρ x = some i) :
    ∃ ρin ρout,
      ρ = ρin ++ [x] ++ ρout ∧
      ρin.length = i ∧
      x ∉ ρin := by
  induction ρ generalizing i with
  | nil =>
      simp [lookupBinder] at h
  | cons y ys ih =>
      by_cases hy : y = x
      · subst hy
        simp [lookupBinder] at h
        cases h
        refine ⟨[], ys, by simp, rfl, by simp⟩
      · simp [lookupBinder, hy] at h
        rcases h with ⟨j, hj, rfl⟩
        rcases ih hj with ⟨ρin, ρout, hsplit, hlen, hfresh⟩
        refine ⟨y :: ρin, ρout, ?_, ?_, ?_⟩
        · simp [hsplit, List.append_assoc]
        · simp [hlen]
        · intro hxy
          simp at hxy
          rcases hxy with hxy | hmem
          · exact hy hxy.symm
          · exact hfresh hmem

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
  Gamma.reverse.map Prod.snd

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
    eraseCtx (Gamma ++ [(x, some t)]) = some t :: eraseCtx Gamma := by
  simpa [eraseCtx] using (eraseCtx_append Gamma [(x, some t)])

@[simp] theorem eraseCtx_append_dead
    (Gamma : LinearCtx) (x : String) :
    eraseCtx (Gamma ++ [(x, none)]) = none :: eraseCtx Gamma := by
  simpa [eraseCtx] using (eraseCtx_append Gamma [(x, none)])

@[simp] theorem eraseCtx_append_pair
    (Gamma : LinearCtx) (x : String) (tx : Typ) (y : String) (ty : Typ) :
    eraseCtx (Gamma ++ [(x, some tx), (y, some ty)]) = some ty :: some tx :: eraseCtx Gamma := by
  simpa [eraseCtx] using (eraseCtx_append Gamma [(x, some tx), (y, some ty)])

@[simp] theorem ctxEnv_append
    (Gamma1 Gamma2 : LinearCtx) :
    ctxEnv (Gamma1 ++ Gamma2) = ctxEnv Gamma2 ++ ctxEnv Gamma1 := by
  simp [ctxEnv, linearCtxDom, List.reverse_append, List.map_append]

@[simp] theorem ctxEnv_append_singleton
    (Gamma : LinearCtx) (x : String) (t : Typ) :
    ctxEnv (Gamma ++ [(x, some t)]) = x :: ctxEnv Gamma := by
  simpa [ctxEnv, linearCtxDom] using (ctxEnv_append Gamma [(x, some t)])

@[simp] theorem ctxEnv_append_pair
    (Gamma : LinearCtx) (x : String) (tx : Typ) (y : String) (ty : Typ) :
    ctxEnv (Gamma ++ [(x, some tx), (y, some ty)]) = y :: x :: ctxEnv Gamma := by
  simpa [ctxEnv, linearCtxDom] using (ctxEnv_append Gamma [(x, some tx), (y, some ty)])

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
    lookupBinder (ctxEnv (GammaPre ++ [(x, some t)] ++ GammaPost)) x =
      some GammaPost.length := by
  have hx' : x ∉ ctxEnv GammaPost := by
    simpa using hx
  have hshape :
      ctxEnv (GammaPre ++ [(x, some t)] ++ GammaPost) =
        ctxEnv GammaPost ++ [x] ++ ctxEnv GammaPre := by
    simp [ctxEnv, linearCtxDom, List.reverse_append, List.reverse_cons, List.append_assoc]
  rw [hshape]
  simpa [List.append_assoc] using
    (lookupBinder_append_target (ρin := ctxEnv GammaPost) (ρout := ctxEnv GammaPre) hx')

/-- Tombstone-preserving named contexts correspond to tombstone-style DB
    contexts by forgetting names. The DB head is the innermost slot. -/
inductive CtxCorr : LinearCtx → LinearCtxDB → Prop where
  | nil :
      CtxCorr [] []
  | dead
      {Gamma : LinearCtx} {GammaDB : LinearCtxDB} {x : String} :
      CtxCorr Gamma GammaDB →
      CtxCorr (Gamma ++ [(x, none)]) (none :: GammaDB)
  | live
      {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
      {x : String} {t : Typ} :
      CtxCorr Gamma GammaDB →
      CtxCorr (Gamma ++ [(x, some t)]) (some t :: GammaDB)

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
        | mk x topt =>
            cases topt with
            | none =>
                simpa [List.reverse_cons, eraseCtx_append] using
                  (CtxCorr.dead (x := x) ih)
            | some t =>
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
        | mk x topt =>
            cases topt with
            | none =>
                simpa [List.reverse_cons, List.append_assoc, eraseCtx_append] using
                  (CtxCorr.dead (x := x) ih)
            | some t =>
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
      simpa [List.length_append] using Nat.succ_le_succ ih
  | live _ ih =>
      simpa [List.length_append] using Nat.succ_le_succ ih

theorem ctxCorr_length_eq
    {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
    (h : CtxCorr Gamma GammaDB) :
    Gamma.length = GammaDB.length := by
  induction h with
  | nil =>
      simp
  | dead _ ih =>
      simpa [List.length_append] using congrArg Nat.succ ih
  | live _ ih =>
      simpa [List.length_append] using congrArg Nat.succ ih

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
  | @dead Gamma GammaDB x h ih =>
      intro hlen
      have hlen' : Gamma.length = GammaDB.length := by
        simpa [List.length_append] using hlen
      simpa [eraseCtx] using congrArg (fun db => none :: db) (ih hlen')
  | @live Gamma GammaDB x t h ih =>
      intro hlen
      have hlen' : Gamma.length = GammaDB.length := by
        simpa [List.length_append] using hlen
      simpa [eraseCtx] using congrArg (fun db => some t :: db) (ih hlen')

theorem ctxCorr_eq_eraseCtx
    {Gamma : LinearCtx} {GammaDB : LinearCtxDB}
    (h : CtxCorr Gamma GammaDB) :
    GammaDB = eraseCtx Gamma :=
  ctxCorr_length_eq_eraseCtx h (ctxCorr_length_eq h)

theorem ctxCorr_singleton_length_one
    {x : String} {t : Typ} {GammaDB : LinearCtxDB}
    (h : CtxCorr ([(x, some t)] : LinearCtx) GammaDB)
    (hlen : GammaDB.length = 1) :
    GammaDB = [some t] := by
  have hlen' : ([(x, some t)] : LinearCtx).length = GammaDB.length := by
    simpa using hlen.symm
  simpa [eraseCtx] using ctxCorr_length_eq_eraseCtx h hlen'

theorem noDupNames_middle_fresh_suffix
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, some t)] ++ GammaPost)) :
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
    (eraseCtx (GammaPre ++ [(x, some t)] ++ GammaPost)).set GammaPost.length none =
      eraseCtx GammaPost ++ none :: eraseCtx GammaPre := by
  have hshape :
      eraseCtx (GammaPre ++ [(x, some t)] ++ GammaPost) =
        eraseCtx GammaPost ++ some t :: eraseCtx GammaPre := by
    simp [eraseCtx, List.reverse_append, List.reverse_cons, List.map_append, List.append_assoc]
  rw [hshape]
  rw [LinearCtxDB.set_append_ge (eraseCtx GammaPost) (some t :: eraseCtx GammaPre)
        GammaPost.length none (by simp [eraseCtx])]
  simp [eraseCtx]

theorem ctxCorr_consume_target
    (GammaPre GammaPost : LinearCtx) (x : String) (t : Typ) :
    CtxCorr (GammaPre ++ [(x, none)] ++ GammaPost)
      ((eraseCtx (GammaPre ++ [(x, some t)] ++ GammaPost)).set GammaPost.length none) := by
  rw [eraseCtx_consume_target GammaPre GammaPost x t]
  exact ctxCorr_append_eraseCtx (CtxCorr.dead (x := x) (ctxCorr_eraseCtx GammaPre)) GammaPost

theorem eraseCtx_get_target
    (GammaPre GammaPost : LinearCtx) (x : String) (t : Typ) :
    (eraseCtx (GammaPre ++ [(x, some t)] ++ GammaPost))[GammaPost.length]? = some (some t) := by
  have hshape :
      eraseCtx (GammaPre ++ [(x, some t)] ++ GammaPost) =
        eraseCtx GammaPost ++ some t :: eraseCtx GammaPre := by
    simp [eraseCtx, List.reverse_append, List.reverse_cons, List.map_append, List.append_assoc]
  rw [hshape]
  rw [LinearCtxDB.getElem?_append_ge (eraseCtx GammaPost) (some t :: eraseCtx GammaPre)
        GammaPost.length (by simp [eraseCtx])]
  simp [eraseCtx]

theorem ctxCorr_singleton_live
    (x : String) (t : Typ) :
    CtxCorr ([(x, some t)] : LinearCtx) [some t] := by
  simpa [eraseCtx] using (ctxCorr_eraseCtx ([(x, some t)] : LinearCtx))

theorem ctxCorr_singleton_dead
    (x : String) (_t : Typ) :
    CtxCorr ([(x, none)] : LinearCtx) [none] := by
  simpa using (CtxCorr.dead (x := x) CtxCorr.nil)

theorem ctxCorr_pair_live
    (x : String) (tx : Typ) (y : String) (ty : Typ) :
    CtxCorr ([(x, some tx), (y, some ty)] : LinearCtx) [some ty, some tx] := by
  simpa [eraseCtx] using (ctxCorr_eraseCtx ([(x, some tx), (y, some ty)] : LinearCtx))

theorem ctxCorr_pair_left_live
    (x : String) (tx : Typ) (_y : String) (_ty : Typ) :
    CtxCorr ([(x, some tx), (_y, none)] : LinearCtx) [none, some tx] := by
  exact CtxCorr.dead (x := _y) (ctxCorr_singleton_live x tx)

theorem ctxCorr_pair_right_live
    (x : String) (_tx : Typ) (y : String) (ty : Typ) :
    CtxCorr ([(x, none), (y, some ty)] : LinearCtx) [some ty, none] := by
  simpa [List.append_assoc] using
    (CtxCorr.live (x := y) (t := ty) (CtxCorr.dead (x := x) CtxCorr.nil))

theorem ctxCorr_pair_dead
    (x : String) (_tx : Typ) (y : String) (_ty : Typ) :
    CtxCorr ([(x, none), (y, none)] : LinearCtx) [none, none] := by
  simpa [List.append_assoc] using
    (CtxCorr.dead (x := y) (CtxCorr.dead (x := x) CtxCorr.nil))

theorem ctxCorr_dead_singleton_inv
    {Gamma : LinearCtx}
    (h : CtxCorr Gamma [none]) :
    ∃ x, Gamma = [(x, none)] := by
  cases h with
  | dead h' =>
      cases h' with
      | nil => exact ⟨_, rfl⟩

theorem ctxCorr_live_singleton_inv
    {Gamma : LinearCtx} {t : Typ}
    (h : CtxCorr Gamma [some t]) :
    ∃ x, Gamma = [(x, some t)] := by
  cases h with
  | live h' =>
      cases h' with
          | nil =>
              exact ⟨_, rfl⟩

theorem ctxCorr_pair_dead_inv
    {Gamma : LinearCtx}
    (h : CtxCorr Gamma [none, none]) :
    ∃ x y, Gamma = [(x, none), (y, none)] := by
  cases h with
  | dead h' =>
      rcases ctxCorr_dead_singleton_inv h' with ⟨x, hx⟩
      subst hx
      exact ⟨_, _, rfl⟩

theorem ctxCorr_pair_left_live_inv
    {Gamma : LinearCtx} {tx : Typ}
    (h : CtxCorr Gamma [none, some tx]) :
    ∃ x y, Gamma = [(x, some tx), (y, none)] := by
  cases h with
  | dead h' =>
      rcases ctxCorr_live_singleton_inv h' with ⟨x, hx⟩
      subst hx
      exact ⟨_, _, rfl⟩

theorem ctxCorr_pair_right_live_inv
    {Gamma : LinearCtx} {ty : Typ}
    (h : CtxCorr Gamma [some ty, none]) :
    ∃ x y, Gamma = [(x, none), (y, some ty)] := by
  cases h with
  | live h' =>
      cases h' with
      | dead h'' =>
          cases h'' with
              | nil =>
                  exact ⟨_, _, rfl⟩

theorem ctxCorr_pair_live_inv
    {Gamma : LinearCtx} {tx ty : Typ}
    (h : CtxCorr Gamma [some ty, some tx]) :
    ∃ x y, Gamma = [(x, some tx), (y, some ty)] := by
  cases h with
  | live h' =>
      cases h' with
      | live h'' =>
          cases h'' with
              | nil =>
                  exact ⟨_, _, rfl⟩

/-- Consume the named binding for `x` in place, preserving context
    shape and all other names. -/
def consumeNameCtx (Gamma : LinearCtx) (x : String) : LinearCtx :=
  Gamma.map fun p => if p.1 = x then (p.1, none) else p

@[simp] theorem consumeNameCtx_append
    (Gamma1 Gamma2 : LinearCtx) (x : String) :
    consumeNameCtx (Gamma1 ++ Gamma2) x =
      consumeNameCtx Gamma1 x ++ consumeNameCtx Gamma2 x := by
  simp [consumeNameCtx, List.map_append]

@[simp] theorem consumeNameCtx_linearCtxDom
    (Gamma : LinearCtx) (x : String) :
    linearCtxDom (consumeNameCtx Gamma x) = linearCtxDom Gamma := by
  unfold consumeNameCtx linearCtxDom
  induction Gamma with
  | nil =>
      simp
  | cons p ps ih =>
      cases p with
      | mk y topt =>
          by_cases hy : y = x <;> simp [hy, ih]

theorem consumeNameCtx_eq_self_of_fresh
    {Gamma : LinearCtx} {x : String}
    (hx : x ∉ linearCtxDom Gamma) :
    consumeNameCtx Gamma x = Gamma := by
  unfold consumeNameCtx
  induction Gamma with
  | nil =>
      simp
  | cons p ps ih =>
      cases p with
      | mk y topt =>
          have hy : y ≠ x := by
            intro hEq
            apply hx
            simp [linearCtxDom, hEq]
          have hxTail : x ∉ linearCtxDom ps := by
            intro hmem
            apply hx
            unfold linearCtxDom at hmem ⊢
            rcases List.mem_map.mp hmem with ⟨p, hp, rfl⟩
            exact List.mem_map.mpr ⟨p, by simp [hp], rfl⟩
          simp [hy, ih hxTail]

/-- Named tombstone environment corresponding to a linear context. -/
def ctxNameSlots (Gamma : LinearCtx) : List (Option String) :=
  Gamma.reverse.map fun
    | (x, some _) => some x
    | (_, none) => none

/-- Tombstone-aware binder environment: live slots carry both the
    source name and type, dead slots remain positionally present as
    `none`. This is the environment shape needed to erase named terms
    directly against DB middle contexts without compacting out consumed
    slots. -/
abbrev BinderSlots := List (Option (String × Typ))

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

mutual

theorem eraseTerm_locRefs :
    ∀ {ρ : BinderEnv} {e : Term} {eDB : TermDB},
      eraseTerm ρ e = some eDB -> locRefs e = locRefsDB eDB
  | ρ, Term.var x, eDB, hErase => by
      cases hlook : lookupBinder ρ x with
      | none =>
          simp [eraseTerm, hlook] at hErase
      | some i =>
          simp [eraseTerm, hlook] at hErase
          cases hErase
          simp [locRefs, locRefsDB]
  | ρ, Term.abs x t body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs hbody
  | ρ, Term.app e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs h1, eraseTerm_locRefs h2]
  | ρ, Term.letBind x e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm (x :: ρ) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs h1, eraseTerm_locRefs h2]
  | ρ, Term.copy e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.letpair x y e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm (y :: x :: ρ) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs h1, eraseTerm_locRefs h2]
  | ρ, Term.pair e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs h1, eraseTerm_locRefs h2]
  | ρ, Term.fst e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.snd e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.unit, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [locRefs, locRefsDB]
  | ρ, Term.const v ds, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [locRefs, locRefsDB]
  | ρ, Term.add e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs h1, eraseTerm_locRefs h2]
  | ρ, Term.mul e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;> simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs h1, eraseTerm_locRefs h2]
  | ρ, Term.sum e d, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at hErase
        have : False := by
          simpa [Seq.seq, eraseTerm, he] using hErase
        cases this
      · simp [eraseTerm, he] at hErase
        injection hErase with hEq
        subst hEq
        simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.expand e d, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at hErase
        have : False := by
          simpa [Seq.seq, eraseTerm, he] using hErase
        cases this
      · simp [eraseTerm, he] at hErase
        injection hErase with hEq
        subst hEq
        simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.uniformLike e lo hi, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB'
      · simp [eraseTerm, he] at hErase
        have : False := by
          simpa [Seq.seq, eraseTerm, he] using hErase
        cases this
      · simp [eraseTerm, he] at hErase
        injection hErase with hEq
        subst hEq
        simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.grad x t tOut body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs hbody
  | ρ, Term.vmap x t body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs hbody
  | ρ, Term.handle epsH body clauses, eDB, hErase => by
      rcases hbody : eraseTerm ρ body with _ | bodyDB <;> simp [eraseTerm, hbody] at hErase
      rcases hclauses : eraseClauses ρ clauses with _ | clausesDB <;>
        simp [eraseTerm, hbody, hclauses] at hErase
      cases hErase
      simp [locRefs, locRefsDB, eraseTerm_locRefs hbody, eraseClauses_locRefs hclauses]
  | ρ, Term.perform op e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [locRefs, locRefsDB] using eraseTerm_locRefs he
  | ρ, Term.loc ell, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [locRefs, locRefsDB]

theorem eraseClauses_locRefs :
    ∀ {ρ : BinderEnv}
      {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses ρ clauses = some clausesDB ->
      locRefsClauses clauses = locRefsClausesDB clausesDB
  | ρ, [], clausesDB, hErase => by
      simp [eraseClauses] at hErase
      cases hErase
      simp [locRefsClauses, locRefsClausesDB]
  | ρ, (op, x, k, hb) :: rest, clausesDB, hErase => by
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;>
        simp [eraseClauses, hhb] at hErase
      rcases hrest : eraseClauses ρ rest with _ | restDB <;>
        simp [eraseClauses, hhb, hrest] at hErase
      cases hErase
      simp [locRefsClauses, locRefsClausesDB, eraseTerm_locRefs hhb, eraseClauses_locRefs hrest]

end

mutual

theorem eraseTerm_activeLocRefs :
    ∀ {ρ : BinderEnv} {e : Term} {eDB : TermDB},
      eraseTerm ρ e = some eDB ->
      activeLocRefs e = activeLocRefsDB eDB
  | ρ, Term.var x, eDB, hErase => by
      cases hρx : lookupBinder ρ x with
      | none =>
          simp [eraseTerm, hρx] at hErase
      | some i =>
          simp [eraseTerm, hρx] at hErase
          cases hErase
          simp [activeLocRefs, activeLocRefsDB]
  | ρ, Term.abs x t body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs hbody
  | ρ, Term.app e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
  | ρ, Term.letBind x e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm (x :: ρ) e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
  | ρ, Term.copy e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs he
  | ρ, Term.letpair x y e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm (y :: x :: ρ) e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
  | ρ, Term.pair e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
  | ρ, Term.fst e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs he
  | ρ, Term.snd e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs he
  | ρ, Term.unit, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB]
  | ρ, Term.const c ds, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB]
  | ρ, Term.add e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
  | ρ, Term.mul e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
  | ρ, Term.sum e d, eDB, hErase => by
      cases he : eraseTerm ρ e with
      | none =>
          simp [eraseTerm, he] at hErase
          cases hErase
      | some eDB0 =>
          simp [eraseTerm, he] at hErase
          injection hErase with hEq
          subst hEq
          simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
  | ρ, Term.expand e d, eDB, hErase => by
      cases he : eraseTerm ρ e with
      | none =>
          simp [eraseTerm, he] at hErase
          cases hErase
      | some eDB0 =>
          simp [eraseTerm, he] at hErase
          injection hErase with hEq
          subst hEq
          simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
  | ρ, Term.uniformLike e lo hi, eDB, hErase => by
      cases he : eraseTerm ρ e with
      | none =>
          simp [eraseTerm, he] at hErase
          cases hErase
      | some eDB0 =>
          simp [eraseTerm, he] at hErase
          injection hErase with hEq
          subst hEq
          simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
  | ρ, Term.grad x t tOut body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs hbody
  | ρ, Term.vmap x t body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs hbody
  | ρ, Term.handle epsH body clauses, eDB, hErase => by
      rcases hbody : eraseTerm ρ body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      rcases hclauses : eraseClauses ρ clauses with _ | clausesDB <;>
        simp [eraseTerm, hbody, hclauses] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs hbody,
        eraseClauses_activeLocRefs hclauses]
  | ρ, Term.perform op e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      simpa [activeLocRefs, activeLocRefsDB] using eraseTerm_activeLocRefs he
  | ρ, Term.loc ell, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [activeLocRefs, activeLocRefsDB]

theorem eraseClauses_activeLocRefs :
    ∀ {ρ : BinderEnv}
      {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses ρ clauses = some clausesDB ->
      activeLocRefsClauses clauses = activeLocRefsClausesDB clausesDB
  | ρ, [], clausesDB, hErase => by
      simp [eraseClauses] at hErase
      cases hErase
      simp [activeLocRefsClauses, activeLocRefsClausesDB]
  | ρ, (op, x, k, hb) :: rest, clausesDB, hErase => by
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;>
        simp [eraseClauses, hhb] at hErase
      rcases hrest : eraseClauses ρ rest with _ | restDB <;>
        simp [eraseClauses, hhb, hrest] at hErase
      cases hErase
      simp [activeLocRefsClauses, activeLocRefsClausesDB, eraseClauses_activeLocRefs hrest]

end

theorem eraseTerm_runtimeLinear_iff
    {ρ : BinderEnv} {e : Term} {eDB : TermDB}
    (hErase : eraseTerm ρ e = some eDB) :
    RuntimeLinear e ↔ RuntimeLinearDB eDB := by
  simpa [RuntimeLinear, RuntimeLinearDB, eraseTerm_locRefs hErase]

theorem eraseTerm_activeRuntimeLinear_iff
    {ρ : BinderEnv} {e : Term} {eDB : TermDB}
    (hErase : eraseTerm ρ e = some eDB) :
    ActiveRuntimeLinear e ↔ ActiveRuntimeLinearDB eDB := by
  simpa [ActiveRuntimeLinear, ActiveRuntimeLinearDB, eraseTerm_activeLocRefs hErase]

private theorem and_iff_congr2
    {A B C D : Prop}
    (h1 : A ↔ C) (h2 : B ↔ D) :
    (A ∧ B) ↔ (C ∧ D) := by
  constructor
  · intro h
    exact ⟨h1.mp h.1, h2.mp h.2⟩
  · intro h
    exact ⟨h1.mpr h.1, h2.mpr h.2⟩

private theorem and_iff_congr3
    {A B C D E F : Prop}
    (h1 : A ↔ D) (h2 : B ↔ E) (h3 : C ↔ F) :
    (A ∧ B ∧ C) ↔ (D ∧ E ∧ F) := by
  constructor
  · intro h
    exact ⟨h1.mp h.1, h2.mp h.2.1, h3.mp h.2.2⟩
  · intro h
    exact ⟨h1.mpr h.1, h2.mpr h.2.1, h3.mpr h.2.2⟩

mutual

theorem eraseTerm_deepActiveRuntimeLinear_iff :
    ∀ {ρ : BinderEnv} {e : Term} {eDB : TermDB},
      eraseTerm ρ e = some eDB ->
      (DeepActiveRuntimeLinear e ↔ DeepActiveRuntimeLinearDB eDB)
  | ρ, Term.var x, eDB, hErase => by
      cases hρx : lookupBinder ρ x with
      | none =>
          simp [eraseTerm, hρx] at hErase
      | some i =>
          simp [eraseTerm, hρx] at hErase
          cases hErase
          simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB,
            ActiveRuntimeLinear, ActiveRuntimeLinearDB]
  | ρ, Term.abs x t body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.abs x t body) ↔
            ActiveRuntimeLinearDB (TermDB.abs t bodyDB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs hbody]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff hbody)
  | ρ, Term.app e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.app e1 e2) ↔
            ActiveRuntimeLinearDB (TermDB.app e1DB e2DB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff h1)
          (eraseTerm_deepActiveRuntimeLinear_iff h2)
  | ρ, Term.letBind x e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm (x :: ρ) e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.letBind x e1 e2) ↔
            ActiveRuntimeLinearDB (TermDB.letBind e1DB e2DB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff h1)
          (eraseTerm_deepActiveRuntimeLinear_iff h2)
  | ρ, Term.copy e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.copy e) ↔
            ActiveRuntimeLinearDB (TermDB.copy eDB') := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.letpair x y e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm (y :: x :: ρ) e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.letpair x y e1 e2) ↔
            ActiveRuntimeLinearDB (TermDB.letpair e1DB e2DB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff h1)
          (eraseTerm_deepActiveRuntimeLinear_iff h2)
  | ρ, Term.pair e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.pair e1 e2) ↔
            ActiveRuntimeLinearDB (TermDB.pair e1DB e2DB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff h1)
          (eraseTerm_deepActiveRuntimeLinear_iff h2)
  | ρ, Term.fst e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.fst e) ↔
            ActiveRuntimeLinearDB (TermDB.fst eDB') := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.snd e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.snd e) ↔
            ActiveRuntimeLinearDB (TermDB.snd eDB') := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.unit, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB,
        ActiveRuntimeLinear, ActiveRuntimeLinearDB]
  | ρ, Term.const c ds, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB,
        ActiveRuntimeLinear, ActiveRuntimeLinearDB]
  | ρ, Term.add e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.add e1 e2) ↔
            ActiveRuntimeLinearDB (TermDB.add e1DB e2DB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff h1)
          (eraseTerm_deepActiveRuntimeLinear_iff h2)
  | ρ, Term.mul e1 e2, eDB, hErase => by
      rcases h1 : eraseTerm ρ e1 with _ | e1DB <;>
        simp [eraseTerm, h1] at hErase
      rcases h2 : eraseTerm ρ e2 with _ | e2DB <;>
        simp [eraseTerm, h1, h2] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.mul e1 e2) ↔
            ActiveRuntimeLinearDB (TermDB.mul e1DB e2DB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs h1, eraseTerm_activeLocRefs h2]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff h1)
          (eraseTerm_deepActiveRuntimeLinear_iff h2)
  | ρ, Term.sum e d, eDB, hErase => by
      cases he : eraseTerm ρ e with
      | none =>
          simp [eraseTerm, he] at hErase
          cases hErase
      | some eDB0 =>
          simp [eraseTerm, he] at hErase
          injection hErase with hEq
          subst hEq
          have hAct :
              ActiveRuntimeLinear (Term.sum e d) ↔
                ActiveRuntimeLinearDB (TermDB.sum eDB0 (eraseDim d)) := by
            simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
              activeLocRefs, activeLocRefsDB,
              eraseTerm_activeLocRefs he]
          simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
            and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.expand e d, eDB, hErase => by
      cases he : eraseTerm ρ e with
      | none =>
          simp [eraseTerm, he] at hErase
          cases hErase
      | some eDB0 =>
          simp [eraseTerm, he] at hErase
          injection hErase with hEq
          subst hEq
          have hAct :
              ActiveRuntimeLinear (Term.expand e d) ↔
                ActiveRuntimeLinearDB (TermDB.expand eDB0 (eraseDim d) 0) := by
            simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
              activeLocRefs, activeLocRefsDB,
              eraseTerm_activeLocRefs he]
          simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
            and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.uniformLike e lo hi, eDB, hErase => by
      cases he : eraseTerm ρ e with
      | none =>
          simp [eraseTerm, he] at hErase
          cases hErase
      | some eDB0 =>
          simp [eraseTerm, he] at hErase
          injection hErase with hEq
          subst hEq
          have hAct :
              ActiveRuntimeLinear (Term.uniformLike e lo hi) ↔
                ActiveRuntimeLinearDB (TermDB.uniformLike eDB0 lo hi) := by
            simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
              activeLocRefs, activeLocRefsDB,
              eraseTerm_activeLocRefs he]
          simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
            and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.grad x t tOut body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.grad x t tOut body) ↔
            ActiveRuntimeLinearDB (TermDB.grad t tOut bodyDB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs hbody]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff hbody)
  | ρ, Term.vmap x t body, eDB, hErase => by
      rcases hbody : eraseTerm (x :: ρ) body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.vmap x t body) ↔
            ActiveRuntimeLinearDB (TermDB.vmap t bodyDB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs hbody]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff hbody)
  | ρ, Term.handle epsH body clauses, eDB, hErase => by
      rcases hbody : eraseTerm ρ body with _ | bodyDB <;>
        simp [eraseTerm, hbody] at hErase
      rcases hclauses : eraseClauses ρ clauses with _ | clausesDB <;>
        simp [eraseTerm, hbody, hclauses] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.handle epsH body clauses) ↔
            ActiveRuntimeLinearDB (TermDB.handle epsH bodyDB clausesDB) := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB,
          eraseTerm_activeLocRefs hbody, eraseClauses_activeLocRefs hclauses]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr3 hAct
          (eraseTerm_deepActiveRuntimeLinear_iff hbody)
          (eraseClauses_deepActiveRuntimeLinear_iff hclauses)
  | ρ, Term.perform op e, eDB, hErase => by
      rcases he : eraseTerm ρ e with _ | eDB' <;> simp [eraseTerm, he] at hErase
      cases hErase
      have hAct :
          ActiveRuntimeLinear (Term.perform op e) ↔
            ActiveRuntimeLinearDB (TermDB.perform op eDB') := by
        simp [ActiveRuntimeLinear, ActiveRuntimeLinearDB,
          activeLocRefs, activeLocRefsDB, eraseTerm_activeLocRefs he]
      simpa [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB] using
        and_iff_congr2 hAct (eraseTerm_deepActiveRuntimeLinear_iff he)
  | ρ, Term.loc ell, eDB, hErase => by
      simp [eraseTerm] at hErase
      cases hErase
      simp [DeepActiveRuntimeLinear, DeepActiveRuntimeLinearDB,
        ActiveRuntimeLinear, ActiveRuntimeLinearDB]

theorem eraseClauses_deepActiveRuntimeLinear_iff :
    ∀ {ρ : BinderEnv}
      {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses ρ clauses = some clausesDB ->
      (DeepActiveRuntimeLinearClauses clauses ↔ DeepActiveRuntimeLinearClausesDB clausesDB)
  | ρ, [], clausesDB, hErase => by
      simp [eraseClauses] at hErase
      cases hErase
      simp [DeepActiveRuntimeLinearClauses, DeepActiveRuntimeLinearClausesDB]
  | ρ, (op, x, k, hb) :: rest, clausesDB, hErase => by
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;>
        simp [eraseClauses, hhb] at hErase
      rcases hrest : eraseClauses ρ rest with _ | restDB <;>
        simp [eraseClauses, hhb, hrest] at hErase
      cases hErase
      simp [DeepActiveRuntimeLinearClauses, DeepActiveRuntimeLinearClausesDB,
        eraseTerm_deepActiveRuntimeLinear_iff hhb,
        eraseClauses_deepActiveRuntimeLinear_iff hrest]

end

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
    eraseTerm (ctxEnv (GammaPre ++ [(x, some t)] ++ GammaPost)) (Term.var x) =
      some (TermDB.var GammaPost.length) := by
  rw [show ctxEnv (GammaPre ++ [(x, some t)] ++ GammaPost) =
      ctxEnv GammaPost ++ [x] ++ ctxEnv GammaPre by
      simp [ctxEnv, linearCtxDom, List.reverse_append, List.reverse_cons, List.append_assoc]]
  simp [eraseTerm, lookupBinder_append_target, hx, List.append_assoc]

theorem transport_var_lexical
    {Delta : CapCtx} {Sigma : StoreTyp}
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, some t)] ++ GammaPost)) :
    let Gamma := GammaPre ++ [(x, some t)] ++ GammaPost
    let i := GammaPost.length
    eraseTerm (ctxEnv Gamma) (Term.var x) = some (TermDB.var i) ∧
    HasTypeDB Delta Sigma (eraseCtx Gamma) (TermDB.var i) t []
      ((eraseCtx Gamma).set i none) ∧
    CtxCorr (GammaPre ++ [(x, none)] ++ GammaPost) ((eraseCtx Gamma).set i none) := by
  dsimp
  have hxPost : x ∉ linearCtxDom GammaPost :=
    noDupNames_middle_fresh_suffix hnd
  refine ⟨?_, ?_, ?_⟩
  · simpa [eraseTerm] using
      (lookupBinder_ctxEnv_append_target
        (GammaPre := GammaPre) (GammaPost := GammaPost) (x := x) (t := t) hxPost)
  · exact HasTypeDB.var Delta Sigma (eraseCtx (GammaPre ++ [(x, some t)] ++ GammaPost))
      GammaPost.length t (eraseCtx_get_target GammaPre GammaPost x t)
  · exact ctxCorr_consume_target GammaPre GammaPost x t

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
    LexicallyScoped [(x, some tx)] e := by
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
    LexicallyScoped [(x, some tx), (y, some ty)] e := by
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
    (hnd : NoDupNames (GammaPre ++ [(x, some t)] ++ GammaPost)) :
    x ∉ linearCtxDom GammaPre := by
  have hsub : List.Sublist (GammaPre ++ [(x, some t)])
      (GammaPre ++ [(x, some t)] ++ GammaPost) := by
    induction GammaPost with
    | nil =>
        simp
    | cons p rest ih =>
        simp [List.append_assoc]
  have hnd' : NoDupNames (GammaPre ++ [(x, some t)]) :=
    noDupNames_of_sublist hsub hnd
  intro hx
  have hnd'' := hnd'
  unfold NoDupNames at hnd''
  simp [linearCtxDom, List.nodup_append] at hnd''
  have hx' : ∃ ty, (x, ty) ∈ GammaPre := by
    simpa [linearCtxDom] using hx
  rcases hx' with ⟨ty, hmem⟩
  exact (hnd''.2 x ty hmem) rfl

theorem HasType.var_mem_of_eq
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {tau : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e tau eps Gamma') :
    ∀ {x : String} {t : Typ}, e = Term.var x -> tau = t -> (x, some t) ∈ Gamma := by
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
    (x, some t) ∈ Gamma :=
  HasType.var_mem_of_eq h rfl rfl

theorem HasType.var_output_consume_of_eq
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {tau : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e tau eps Gamma')
    :
    ∀ (hnd : NoDupNames Gamma) {x : String} {t : Typ},
      e = Term.var x -> tau = t ->
      Gamma' = consumeNameCtx Gamma x := by
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | var _ _ GammaPre GammaPost y ty =>
      intro hnd x t heq hty
      cases heq
      cases hty
      have hyPre : y ∉ linearCtxDom GammaPre :=
        noDupNames_middle_fresh_prefix hnd
      have hyPost : y ∉ linearCtxDom GammaPost :=
        noDupNames_middle_fresh_suffix hnd
      calc
        GammaPre ++ [(y, none)] ++ GammaPost
            = consumeNameCtx GammaPre y ++ [(y, none)] ++ consumeNameCtx GammaPost y := by
                simp [consumeNameCtx_eq_self_of_fresh hyPre,
                  consumeNameCtx_eq_self_of_fresh hyPost]
        _ = consumeNameCtx (GammaPre ++ [(y, some ty)] ++ GammaPost) y := by
              simp [consumeNameCtx_append, consumeNameCtx, List.append_assoc]
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro hnd x t heq hty
      exact ih hnd heq hty
  | nil _ _ _ _ _ =>
      exact True.intro
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      exact True.intro
  | _ =>
      intro hnd x t heq hty
      cases heq

theorem HasType.var_output_consume_of_noDup
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {x : String} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (Term.var x) t eps Gamma')
    (hnd : NoDupNames Gamma) :
    Gamma' = consumeNameCtx Gamma x :=
  HasType.var_output_consume_of_eq h hnd rfl rfl

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

theorem ctxEnv_eq_of_names_eq
    {Gamma Gamma' : LinearCtx}
    (hEq : Gamma.map Prod.fst = Gamma'.map Prod.fst) :
    ctxEnv Gamma = ctxEnv Gamma' := by
  simpa [ctxEnv, linearCtxDom] using congrArg List.reverse hEq

theorem lexical_of_names_eq
    {Gamma Gamma' : LinearCtx} {e : Term}
    (hEq : Gamma.map Prod.fst = Gamma'.map Prod.fst)
    (hlex : LexicallyScoped Gamma e) :
    LexicallyScoped Gamma' e := by
  rcases hlex with ⟨hnd, hdom, hws⟩
  refine ⟨?_, ?_, hws⟩
  · unfold NoDupNames at hnd ⊢
    simpa [linearCtxDom, hEq] using hnd
  · intro x hx
    have hx' : x ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom, hEq] using hx
    exact hdom x hx'

theorem lexical_output_of_typing
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' : LinearCtx} {e : Term} {e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (hlex : LexicallyScoped Gamma e') :
    LexicallyScoped Gamma' e' :=
  lexical_of_names_eq (hasType_names_preserved h) hlex

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
    NoDupNames (Gamma ++ [(x, some t)]) := by
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
    LexicallyScoped (Gamma ++ [(x, some tx)]) body := by
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
    List.Sublist (Gamma' ++ [(x, some t)]) (Gamma ++ [(x, some t)]) := by
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
    List.Sublist (Gamma' ++ [(x, some tx), (y, some ty)]) (Gamma ++ [(x, some tx), (y, some ty)]) := by
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
    (hsub : List.Sublist Gamma' (Gamma ++ [(x, some t)])) :
    List.Sublist Gamma' Gamma ∨
      ∃ GammaPre,
        Gamma' = GammaPre ++ [(x, some t)] ∧
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
    (hsub : List.Sublist Gamma' (Gamma ++ [(x, some tx), (y, some ty)])) :
    List.Sublist Gamma' Gamma ∨
      (∃ GammaPre, Gamma' = GammaPre ++ [(x, some tx)] ∧ List.Sublist GammaPre Gamma) ∨
      (∃ GammaPre, Gamma' = GammaPre ++ [(y, some ty)] ∧ List.Sublist GammaPre Gamma) ∨
      (∃ GammaPre, Gamma' = GammaPre ++ [(x, some tx), (y, some ty)] ∧ List.Sublist GammaPre Gamma) := by
  have hsubY : List.Sublist Gamma' ((Gamma ++ [(x, some tx)]) ++ [(y, some ty)]) := by
    simpa [List.append_assoc] using hsub
  rcases
      sublist_append_singleton_cases
        (Gamma := Gamma ++ [(x, some tx)]) (Gamma' := Gamma') (x := y) (t := ty) hsubY with
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

theorem lexical_letBind_body
    {Gamma : LinearCtx} {x : String} {tx : Typ} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letBind x e1 e2)) :
    LexicallyScoped (Gamma ++ [(x, some tx)]) e2 := by
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
    LexicallyScoped (Gamma ++ [(x, some tx), (y, some ty)]) e2 := by
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
  have hnd' : NoDupNames (Gamma ++ [(x, some tx)]) :=
    noDupNames_append_singleton hnd hxGamma
  have hyGamma' : y ∉ linearCtxDom (Gamma ++ [(x, some tx)]) := by
    intro hy
    simp [linearCtxDom] at hy
    rcases hy with hy | hy
    · exact hyGamma (by simpa [linearCtxDom] using hy)
    · exact hbody.2.1 hy.symm
  have hnd'' : NoDupNames (Gamma ++ [(x, some tx)] ++ [(y, some ty)]) :=
    noDupNames_append_singleton hnd' hyGamma'
  simpa [List.append_assoc] using
    (show LexicallyScoped (Gamma ++ [(x, some tx)] ++ [(y, some ty)]) e2 from by
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
    LexicallyScoped (Gamma ++ [(x, some tx)]) body := by
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
    LexicallyScoped (Gamma ++ [(x, some tx)]) body := by
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
    LexicallyScoped (Gamma ++ [(x, some tx), (k, some tk)]) hb := by
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
  have hnd' : NoDupNames (Gamma ++ [(x, some tx)]) :=
    noDupNames_append_singleton hnd hxGamma
  have hkGamma' : k ∉ linearCtxDom (Gamma ++ [(x, some tx)]) := by
    intro hk
    simp [linearCtxDom] at hk
    rcases hk with hk | hk
    · exact hkGamma (by simpa [linearCtxDom] using hk)
    · exact hclause.1 hk.symm
  have hnd'' : NoDupNames (Gamma ++ [(x, some tx)] ++ [(k, some tk)]) :=
    noDupNames_append_singleton hnd' hkGamma'
  simpa [List.append_assoc] using
    (show LexicallyScoped (Gamma ++ [(x, some tx)] ++ [(k, some tk)]) hb from by
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
    ∃ t2 epsBody GammaBody slot,
      t = Typ.arrow t1 t2 epsBody ∧
      HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) body t2 epsBody
        (GammaBody ++ [(x, slot)]) ∧
      GammaOut = GammaBody := by
  generalize heq : Term.abs x t1 body = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | abs _ _ _ GammaBody _ _ t2 epsBody _ slot hBody =>
      cases heq
      exact ⟨t2, epsBody, GammaBody, slot, rfl, hBody, rfl⟩
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

/-- App inversion with the outer `SubEffRow` witness retained. -/
theorem HasType.app_inv_sub_bridge
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
  | app _ _ _ Gamma2 _ _ _ t1 _ epsBody eps1 eps2 h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, t1, epsBody, eps1, eps2, h1, h2, fun _ hop => hop⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      rcases ih heq with ⟨Gamma2, t1, epsBody, eps1, eps2, h1, h2, hSub'⟩
      exact ⟨Gamma2, t1, epsBody, eps1, eps2, h1, h2,
        fun op hop => hSub op (hSub' op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- The right operand of an effect-row union embeds into the union. -/
theorem SubEffRow.union_right
    (epsLeft epsRight : EffectRow) :
    SubEffRow epsRight (EffectRow.union epsLeft epsRight) := by
  intro op hop
  show op ∈ epsLeft ++ epsRight.filter (fun o => !epsLeft.contains o)
  rw [List.mem_append]
  by_cases hLeft : op ∈ epsLeft
  · exact Or.inl hLeft
  · right
    rw [List.mem_filter]
    exact ⟨hop, by simp [hLeft]⟩

/-- LetBind inversion local to the DB bridge. -/
theorem HasType.letBind_inv_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps GammaOut) :
    ∃ Gamma2 Gamma3 t1 eps1 eps2 slot,
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t eps2
        (Gamma3 ++ [(x, slot)]) ∧
      GammaOut = Gamma3 := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Gamma2 Gamma3 _ _ _ t1 _ eps1 eps2 slot h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, Gamma3, t1, eps1, eps2, slot, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetBind inversion with the outer `SubEffRow` witness retained. -/
theorem HasType.letBind_inv_sub_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps GammaOut) :
    ∃ Gamma2 Gamma3 t1 eps1 eps2 slot,
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t eps2
        (Gamma3 ++ [(x, slot)]) ∧
      GammaOut = Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Gamma2 Gamma3 _ _ _ t1 _ eps1 eps2 slot h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, Gamma3, t1, eps1, eps2, slot, h1, h2, rfl, fun _ hop => hop⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      rcases ih heq with ⟨Gamma2, Gamma3, t1, eps1, eps2, slot, h1, h2, hOut, hSub'⟩
      exact ⟨Gamma2, Gamma3, t1, eps1, eps2, slot, h1, h2, hOut,
        fun op hop => hSub op (hSub' op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Pair inversion with the outer `SubEffRow` witness retained. -/
theorem HasType.pair_inv_sub_bridge
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
  | tpair _ _ _ Gamma2 _ _ _ _ _ eps1 eps2 h1 h2 _ _ =>
      cases heq
      cases htq
      exact ⟨Gamma2, eps1, eps2, h1, h2, fun _ hop => hop⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      rcases ih heq htq with ⟨Gamma2, eps1, eps2, h1, h2, hSub'⟩
      exact ⟨Gamma2, eps1, eps2, h1, h2, fun op hop => hSub op (hSub' op hop)⟩
  | _ => (try cases heq) <;> (try cases htq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetPair inversion local to the DB bridge. -/
theorem HasType.letpair_inv_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps GammaOut) :
    ∃ Gamma2 Gamma3 t1 t2 eps1 eps2 slotX slotY,
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
        (Gamma3 ++ [(x, slotX), (y, slotY)]) ∧
      GammaOut = Gamma3 := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Gamma2 Gamma3 _ _ _ _ t1 t2 _ eps1 eps2 slotX slotY h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- LetPair inversion with the outer `SubEffRow` witness retained. -/
theorem HasType.letpair_inv_sub_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps GammaOut) :
    ∃ Gamma2 Gamma3 t1 t2 eps1 eps2 slotX slotY,
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
        (Gamma3 ++ [(x, slotX), (y, slotY)]) ∧
      GammaOut = Gamma3 ∧
      SubEffRow (EffectRow.union eps1 eps2) eps := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Gamma2 Gamma3 _ _ _ _ t1 t2 _ eps1 eps2 slotX slotY h1 h2 _ _ =>
      cases heq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, rfl,
        fun _ hop => hop⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      rcases ih heq with
        ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, hOut, hSub'⟩
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, hOut,
        fun op hop => hSub op (hSub' op hop)⟩
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-- Perform inversion local to the DB bridge. -/
theorem HasType.perform_inv_bridge
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
      exact ⟨tArg, eps0, h', hmatch, fun _ hop => hop⟩
  | subEff _ _ _ _ _ _ _ _ _ hSub ih =>
      obtain ⟨tArg, eps0, h', hmatch, hwit⟩ := ih heq
      refine ⟨tArg, eps0, h', hmatch, ?_⟩
      intro op' hop'
      exact hSub op' (hwit op' hop')
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
    ∃ tArg tRet slotX slotK,
      OpSigMatch op tArg tRet ∧
      HasType Delta Sigma
        (Gamma2 ++ [(x, some tArg), (k, some (Typ.arrow tRet t epsR))])
        hb t epsR (Gamma3 ++ [(x, slotX), (k, slotK)]) := by
  induction clauses generalizing Gamma2 Gamma3 with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      cases hcls with
      | cons _ _ _ _ _ tArg tRet _ _ x' k' hb' rest' slotX slotK hMatch hHead hRest =>
          rcases List.mem_cons.mp hmem with h0 | htl
          · cases h0
            exact ⟨tArg, tRet, slotX, slotK, hMatch, hHead⟩
          · exact ih hRest htl

/-- Values can be re-typed at any effect row with the same type and
    contexts. Local copy of the preservation helper to avoid an import
    cycle. -/
theorem HasType.value_eff_polymorphic_bridge
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma v t eps Gamma')
    (hv : IsValue v) :
    ∀ eps', HasType Delta Sigma Gamma v t eps' Gamma' := by
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | unit Delta Sigma Gamma =>
      intro eps'
      exact HasType.subEff Delta Sigma Gamma Gamma _ _ [] eps'
        (HasType.unit Delta Sigma Gamma) (by intro op hop; cases hop)
  | abs Delta Sigma Gamma1 Gamma2 x t1 t2 epsBody body slot hBody =>
      intro eps'
      exact HasType.subEff Delta Sigma Gamma1 Gamma2 _ _ [] eps'
        (HasType.abs Delta Sigma Gamma1 Gamma2 x t1 t2 epsBody body slot hBody)
        (by intro op hop; cases hop)
  | tpair Delta Sigma Gamma1 Gamma2 Gamma3 v1 v2 t1 t2 eps1 eps2 _ _ ih1 ih2 =>
      intro eps'
      cases hv with
      | pair _ _ hv1 hv2 =>
          have h1' := ih1 hv1 []
          have h2' := ih2 hv2 []
          exact HasType.subEff Delta Sigma Gamma1 Gamma3 _ _ _ eps'
            (HasType.tpair Delta Sigma Gamma1 Gamma2 Gamma3 v1 v2 t1 t2 [] [] h1' h2')
            (by intro op hop; cases hop)
  | loc Delta Sigma Gamma ell t hLook =>
      intro eps'
      exact HasType.subEff Delta Sigma Gamma Gamma _ _ [] eps'
        (HasType.loc Delta Sigma Gamma ell t hLook)
        (by intro op hop; cases hop)
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro eps'
      exact ih hv eps'
  | nil _ _ _ _ _ =>
      exact True.intro
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      exact True.intro
  | _ =>
      exfalso
      cases hv

/-- Pointwise tombstone-preserving slot weakening on DB contexts:
    every output slot is either `none` or unchanged from the input slot
    at the same position. -/
def SlotSubDB : LinearCtxDB → LinearCtxDB → Prop
  | [], [] => True
  | out :: outs, inp :: inps => (out = none ∨ out = inp) ∧ SlotSubDB outs inps
  | _, _ => False

theorem slotSubDB_refl :
    ∀ ρ : LinearCtxDB, SlotSubDB ρ ρ
  | [] => by simp [SlotSubDB]
  | x :: xs => by
      simp [SlotSubDB, slotSubDB_refl xs]

theorem slotSubDB_tail
    {out inp : Option Typ} {outs inps : LinearCtxDB}
    (h : SlotSubDB (out :: outs) (inp :: inps)) :
    SlotSubDB outs inps :=
  h.2

theorem slotSubDB_append
    {out1 out2 inp1 inp2 : LinearCtxDB}
    (h1 : SlotSubDB out1 inp1)
    (h2 : SlotSubDB out2 inp2) :
    SlotSubDB (out1 ++ out2) (inp1 ++ inp2) := by
  revert out2 inp2 h2
  induction out1 generalizing inp1 with
  | nil =>
      intro out2 inp2 h2
      cases inp1 with
      | nil =>
          simpa [SlotSubDB] using h2
      | cons i is =>
          cases h1
  | cons o os ih =>
      intro out2 inp2 h2
      cases inp1 with
      | nil =>
          cases h1
      | cons i is =>
          rcases h1 with ⟨hhd, htl⟩
          exact ⟨hhd, ih htl h2⟩

theorem slotSubDB_trans
    {out mid inp : LinearCtxDB}
    (h1 : SlotSubDB out mid)
    (h2 : SlotSubDB mid inp) :
    SlotSubDB out inp := by
  induction out generalizing mid inp with
  | nil =>
      cases mid <;> cases inp <;> simp [SlotSubDB] at h1 h2 ⊢
  | cons o os ih =>
      cases mid with
      | nil =>
          cases h1
      | cons m ms =>
          cases inp with
          | nil =>
              cases h2
          | cons i is =>
              rcases h1 with ⟨h1hd, h1tl⟩
              rcases h2 with ⟨h2hd, h2tl⟩
              refine ⟨?_, ih h1tl h2tl⟩
              rcases h1hd with rfl | rfl
              · exact Or.inl rfl
              · exact h2hd

theorem has_type_slotSubDB
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    SlotSubDB (eraseCtx Gamma') (eraseCtx Gamma) := by
  induction h using HasType.rec
    (motive_2 := fun _ _ Γ2 Γ3 _ _ _ _ => SlotSubDB (eraseCtx Γ3) (eraseCtx Γ2)) with
  | var _ _ Γpre Γpost _ tx =>
      have hpost : SlotSubDB (eraseCtx Γpost) (eraseCtx Γpost) := slotSubDB_refl _
      have hmid : SlotSubDB [none] [some tx] := by simp [SlotSubDB]
      have hpre : SlotSubDB (eraseCtx Γpre) (eraseCtx Γpre) := slotSubDB_refl _
      simpa [eraseCtx, List.reverse_append, List.reverse_cons, List.map_append, List.append_assoc] using
        slotSubDB_append hpost (slotSubDB_append hmid hpre)
  | unit _ _ Γ =>
      simpa using slotSubDB_refl (eraseCtx Γ)
  | abs _ _ Γ1 Γ2 x t1 _ _ _ slot _ ih =>
      have hbody : SlotSubDB (slot :: eraseCtx Γ2) (some t1 :: eraseCtx Γ1) := by
        simpa [eraseCtx_append_singleton] using ih
      exact slotSubDB_tail hbody
  | app _ _ _ Γ2 _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSubDB_trans ih2 ih1
  | letBind _ _ _ Γ2 Γ3 _ _ _ t1 _ _ _ slot _ _ ih1 ih2 =>
      have hbody : SlotSubDB (slot :: eraseCtx Γ3) (some t1 :: eraseCtx Γ2) := by
        simpa [eraseCtx_append_singleton] using ih2
      exact slotSubDB_trans (slotSubDB_tail hbody) ih1
  | copy _ _ _ _ _ _ _ _ ih =>
      exact ih
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1 t2 _ _ _ slotX slotY _ _ ih1 ih2 =>
      have hbody : SlotSubDB (slotY :: slotX :: eraseCtx Γ3) (some t2 :: some t1 :: eraseCtx Γ2) := by
        simpa [eraseCtx_append_pair] using ih2
      exact slotSubDB_trans (slotSubDB_tail (slotSubDB_tail hbody)) ih1
  | tpair _ _ _ Γ2 _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSubDB_trans ih2 ih1
  | fst _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | snd _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | const _ _ Γ _ _ =>
      simpa using slotSubDB_refl (eraseCtx Γ)
  | tadd _ _ _ Γ2 _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSubDB_trans ih2 ih1
  | tmul _ _ _ Γ2 _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSubDB_trans ih2 ih1
  | tsum _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | texpand _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | perform _ _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihBody ihClauses =>
      exact slotSubDB_trans ihClauses ihBody
  | tgrad _ _ Γ _ _ _ _ _ _ _ =>
      simpa using slotSubDB_refl (eraseCtx Γ)
  | tvmap _ _ Γ _ _ _ _ _ _ _ =>
      simpa using slotSubDB_refl (eraseCtx Γ)
  | loc _ _ Γ _ _ _ =>
      simpa using slotSubDB_refl (eraseCtx Γ)
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | nil _ _ Γ _ _ =>
      simpa using slotSubDB_refl (eraseCtx Γ)
  | cons _ _ Γ2 Γ3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihRest =>
      exact ihRest

theorem slotSubDB_singleton_cases
    {out : LinearCtxDB} {t : Typ}
    (h : SlotSubDB out [some t]) :
    out = [none] ∨ out = [some t] := by
  cases out with
  | nil =>
      cases h
  | cons o rest =>
      cases rest with
      | nil =>
          rcases h.1 with ho | ho
          · exact Or.inl (by cases ho; rfl)
          · exact Or.inr (by cases ho; rfl)
      | cons o' rest' =>
          cases h.2

theorem slotSubDB_pair_cases
    {out : LinearCtxDB} {tx ty : Typ}
    (h : SlotSubDB out [some ty, some tx]) :
    out = [none, none] ∨
      out = [none, some tx] ∨
      out = [some ty, none] ∨
      out = [some ty, some tx] := by
  cases out with
  | nil =>
      cases h
  | cons o1 rest =>
      cases rest with
      | nil =>
          cases h.2
      | cons o2 rest' =>
          cases rest' with
          | nil =>
              rcases h.1 with h1 | h1 <;> rcases h.2.1 with h2 | h2
              · exact Or.inl (by cases h1; cases h2; rfl)
              · exact Or.inr (Or.inl (by cases h1; cases h2; rfl))
              · exact Or.inr (Or.inr (Or.inl (by cases h1; cases h2; rfl)))
              · exact Or.inr (Or.inr (Or.inr (by cases h1; cases h2; rfl)))
          | cons o3 rest'' =>
              cases h.2.2

theorem singleton_output_db_shape
    {Sigma : StoreTyp} {x : String} {tx : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, some tx)] e t eps GammaOut) :
    ∃ GammaOutDB,
      CtxCorr GammaOut GammaOutDB ∧
      (GammaOutDB = [none] ∨ GammaOutDB = [some tx]) := by
  refine ⟨eraseCtx GammaOut, ctxCorr_eraseCtx GammaOut, ?_⟩
  have hsub : SlotSubDB (eraseCtx GammaOut) [some tx] := by
    simpa [eraseCtx] using has_type_slotSubDB h
  exact slotSubDB_singleton_cases hsub

theorem pair_output_db_shape
    {Sigma : StoreTyp} {x y : String} {tx ty : Typ}
    {e : Term} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma [(x, some tx), (y, some ty)] e t eps GammaOut) :
    ∃ GammaOutDB,
      CtxCorr GammaOut GammaOutDB ∧
      (GammaOutDB = [none, none] ∨
        GammaOutDB = [none, some tx] ∨
        GammaOutDB = [some ty, none] ∨
        GammaOutDB = [some ty, some tx]) := by
  refine ⟨eraseCtx GammaOut, ctxCorr_eraseCtx GammaOut, ?_⟩
  have hsub : SlotSubDB (eraseCtx GammaOut) [some ty, some tx] := by
    simpa [eraseCtx] using has_type_slotSubDB h
  exact slotSubDB_pair_cases hsub

theorem eraseClauses_exists_of_mem_named
    {ρ : BinderEnv}
    {clauses : List (EffectLabel × String × String × Term)}
    {clausesDB : List (EffectLabel × TermDB)}
    {cl : EffectLabel × String × String × Term}
    (hErase : eraseClauses ρ clauses = some clausesDB)
    (hMem : cl ∈ clauses) :
    ∃ hbDB, (cl.1, hbDB) ∈ clausesDB := by
  induction clauses generalizing clausesDB with
  | nil =>
      cases hMem
  | cons hd rest ih =>
      rcases hd with ⟨op, x, k, hb⟩
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;>
        simp [eraseClauses, hhb] at hErase
      rcases hrest : eraseClauses ρ rest with _ | restDB <;>
        simp [eraseClauses, hhb, hrest] at hErase
      cases hErase
      simp only [List.mem_cons] at hMem
      rcases hMem with rfl | hMem
      · exact ⟨hbDB, by simp⟩
      · rcases ih hrest hMem with ⟨hbDB', hmemDB⟩
        exact ⟨hbDB', by simp [hmemDB]⟩

theorem eraseClauses_named_of_mem
    {ρ : BinderEnv}
    {clauses : List (EffectLabel × String × String × Term)}
    {clausesDB : List (EffectLabel × TermDB)}
    {clDB : EffectLabel × TermDB}
    (hErase : eraseClauses ρ clauses = some clausesDB)
    (hMem : clDB ∈ clausesDB) :
    ∃ cl ∈ clauses, cl.1 = clDB.1 := by
  induction clauses generalizing clausesDB with
  | nil =>
      cases hErase
      cases hMem
  | cons hd rest ih =>
      rcases hd with ⟨op, x, k, hb⟩
      rcases hhb : eraseTerm (k :: x :: ρ) hb with _ | hbDB <;>
        simp [eraseClauses, hhb] at hErase
      rcases hrest : eraseClauses ρ rest with _ | restDB <;>
        simp [eraseClauses, hhb, hrest] at hErase
      cases hErase
      simp only [List.mem_cons] at hMem
      rcases hMem with hMem | hMem
      · cases hMem
        exact ⟨(op, x, k, hb), by simp, rfl⟩
      · rcases ih hrest hMem with ⟨cl, hcl, hEq⟩
        exact ⟨cl, by simp [hcl], hEq⟩

/-- Every handler clause body is lexically scoped under the current
    base context plus its two binders. -/
def ClausesLexical
    (Gamma : LinearCtx) (t : Typ) (epsR : EffectRow)
    (clauses : List (EffectLabel × String × String × Term)) : Prop :=
  ∀ {op x k hb tArg tRet},
    (op, x, k, hb) ∈ clauses →
    LexicallyScoped
      (Gamma ++ [(x, some tArg), (k, some (Typ.arrow tRet t epsR))]) hb

theorem transport_typing_lexical
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma GammaOut : LinearCtx} {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps GammaOut)
    (hlex : LexicallyScoped Gamma e) :
    ∃ eDB,
      eraseTerm (ctxEnv Gamma) e = some eDB ∧
      HasTypeDB Delta Sigma (eraseCtx Gamma) eDB t eps (eraseCtx GammaOut) := by
  revert hlex
  induction h using HasType.rec
    (motive_2 := fun Delta Sigma GammaIn GammaOut t epsR clauses _ =>
      ClausesLexical GammaIn t epsR clauses →
      ∃ clausesDB,
        eraseClauses (ctxEnv GammaIn) clauses = some clausesDB ∧
        ClausesTypedDB Delta Sigma (eraseCtx GammaIn) (eraseCtx GammaOut) t epsR clausesDB)
    with
  | var Delta Sigma GammaPre GammaPost x tx =>
      intro hlex
      rcases hlex with ⟨hnd, _, _⟩
      rcases
        transport_var_lexical
          (Delta := Delta) (Sigma := Sigma)
          (GammaPre := GammaPre) (GammaPost := GammaPost)
          (x := x) (t := tx) hnd with
        ⟨hErase, hTy, _⟩
      have hOutEq :
          (eraseCtx (GammaPre ++ [(x, some tx)] ++ GammaPost)).set GammaPost.length none =
            eraseCtx (GammaPre ++ [(x, none)] ++ GammaPost) := by
        rw [eraseCtx_consume_target]
        simp [eraseCtx, List.reverse_append, List.reverse_cons, List.map_append, List.append_assoc]
      have hTy' :
          HasTypeDB Delta Sigma (eraseCtx (GammaPre ++ [(x, some tx)] ++ GammaPost))
            (TermDB.var GammaPost.length) tx []
            (eraseCtx (GammaPre ++ [(x, none)] ++ GammaPost)) := by
        rw [hOutEq] at hTy
        exact hTy
      exact ⟨TermDB.var GammaPost.length, hErase, hTy'⟩
  | unit Delta Sigma Gamma =>
      intro _hlex
      exact ⟨TermDB.unit, rfl, HasTypeDB.unit Delta Sigma (eraseCtx Gamma)⟩
  | abs Delta Sigma Gamma1 Gamma2 x t1 t2 epsBody body slot hBody ihBody =>
      intro hlex
      rcases ihBody (lexical_abs_body (tx := t1) hlex) with
        ⟨bodyDB, hEraseBody, hTyBody⟩
      have hEraseBody' :
          eraseTerm (x :: ctxEnv Gamma1) body = some bodyDB := by
        simpa using hEraseBody
      have hTyBody' :
          HasTypeDB Delta Sigma (some t1 :: eraseCtx Gamma1) bodyDB t2 epsBody
            (slot :: eraseCtx Gamma2) := by
        simpa [eraseCtx_append_singleton] using hTyBody
      refine ⟨TermDB.abs t1 bodyDB, ?_, ?_⟩
      · simp [eraseTerm, hEraseBody']
      · exact HasTypeDB.abs Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          slot t1 t2 epsBody bodyDB hTyBody'
  | app Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 epsInner eps1 eps2 h1 h2 ih1 ih2 =>
      intro hlex
      rcases ih1 (lexical_app_left hlex) with ⟨e1DB, hErase1, hTy1⟩
      rcases ih2 (lexical_output_of_typing h1 (lexical_app_right hlex)) with
        ⟨e2DB, hErase2, hTy2⟩
      have hErase2' : eraseTerm (ctxEnv Gamma1) e2 = some e2DB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved h1)]
        exact hErase2
      refine ⟨TermDB.app e1DB e2DB, ?_, ?_⟩
      · simp [eraseTerm, hErase1, hErase2']
      · exact HasTypeDB.app Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) e1DB e2DB t1 t2 epsInner eps1 eps2 hTy1 hTy2
  | letBind Delta Sigma Gamma1 Gamma2 Gamma3 x e1 e2 t1 t2 eps1 eps2 slot h1 h2 ih1 ih2 =>
      intro hlex
      rcases ih1 (lexical_letBind_bound hlex) with ⟨e1DB, hErase1, hTy1⟩
      have hlexBody1 : LexicallyScoped (Gamma1 ++ [(x, some t1)]) e2 :=
        lexical_letBind_body (tx := t1) hlex
      have hEqBody :
          (Gamma1 ++ [(x, some t1)]).map Prod.fst =
            (Gamma2 ++ [(x, some t1)]).map Prod.fst := by
        simp [List.map_append, hasType_names_preserved h1]
      rcases ih2 (lexical_of_names_eq hEqBody hlexBody1) with
        ⟨e2DB, hErase2, hTy2⟩
      have hErase2' :
          eraseTerm (x :: ctxEnv Gamma2) e2 = some e2DB := by
        simpa using hErase2
      have hErase2'' :
          eraseTerm (x :: ctxEnv Gamma1) e2 = some e2DB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved h1)]
        exact hErase2'
      have hTy2' :
          HasTypeDB Delta Sigma (some t1 :: eraseCtx Gamma2) e2DB t2 eps2
            (slot :: eraseCtx Gamma3) := by
        simpa [eraseCtx_append_singleton] using hTy2
      refine ⟨TermDB.letBind e1DB e2DB, ?_, ?_⟩
      · simp [eraseTerm, hErase1, hErase2'']
      · exact HasTypeDB.letBind Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) slot e1DB e2DB t1 t2 eps1 eps2 hTy1 hTy2'
  | copy Delta Sigma Gamma1 Gamma2 e ds eps hBody ih =>
      intro hlex
      rcases ih (lexical_copy_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.copy eDB, ?_, ?_⟩
      · simp [eraseTerm, hErase]
      · exact HasTypeDB.copy Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          eDB ds eps hTy
  | letpair Delta Sigma Gamma1 Gamma2 Gamma3 x y e1 e2 t1 t2 t eps1 eps2 slotX slotY h1 h2 ih1 ih2 =>
      intro hlex
      rcases ih1 (lexical_letpair_bound hlex) with ⟨e1DB, hErase1, hTy1⟩
      have hlexBody1 : LexicallyScoped (Gamma1 ++ [(x, some t1), (y, some t2)]) e2 :=
        lexical_letpair_body (tx := t1) (ty := t2) hlex
      have hEqBody :
          (Gamma1 ++ [(x, some t1), (y, some t2)]).map Prod.fst =
            (Gamma2 ++ [(x, some t1), (y, some t2)]).map Prod.fst := by
        simp [List.map_append, hasType_names_preserved h1]
      rcases ih2 (lexical_of_names_eq hEqBody hlexBody1) with
        ⟨e2DB, hErase2, hTy2⟩
      have hErase2' :
          eraseTerm (y :: x :: ctxEnv Gamma2) e2 = some e2DB := by
        simpa using hErase2
      have hErase2'' :
          eraseTerm (y :: x :: ctxEnv Gamma1) e2 = some e2DB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved h1)]
        exact hErase2'
      have hTy2' :
          HasTypeDB Delta Sigma (some t2 :: some t1 :: eraseCtx Gamma2) e2DB t eps2
            (slotY :: slotX :: eraseCtx Gamma3) := by
        simpa [eraseCtx_append_pair] using hTy2
      refine ⟨TermDB.letpair e1DB e2DB, ?_, ?_⟩
      · simp [eraseTerm, hErase1, hErase2'']
      · exact HasTypeDB.letpair Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) slotY slotX e1DB e2DB t1 t2 t eps1 eps2 hTy1 hTy2'
  | tpair Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 eps1 eps2 h1 h2 ih1 ih2 =>
      intro hlex
      rcases ih1 (lexical_pair_left hlex) with ⟨e1DB, hErase1, hTy1⟩
      rcases ih2 (lexical_output_of_typing h1 (lexical_pair_right hlex)) with
        ⟨e2DB, hErase2, hTy2⟩
      have hErase2' : eraseTerm (ctxEnv Gamma1) e2 = some e2DB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved h1)]
        exact hErase2
      refine ⟨TermDB.pair e1DB e2DB, ?_, ?_⟩
      · simp [eraseTerm, hErase1, hErase2']
      · exact HasTypeDB.tpair Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) e1DB e2DB t1 t2 eps1 eps2 hTy1 hTy2
  | fst Delta Sigma Gamma1 Gamma2 e t1 t2 eps hBody ih =>
      intro hlex
      rcases ih (lexical_fst_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.fst eDB, ?_, ?_⟩
      · simp [eraseTerm, hErase]
      · exact HasTypeDB.fst Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          eDB t1 t2 eps hTy
  | snd Delta Sigma Gamma1 Gamma2 e t1 t2 eps hBody ih =>
      intro hlex
      rcases ih (lexical_snd_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.snd eDB, ?_, ?_⟩
      · simp [eraseTerm, hErase]
      · exact HasTypeDB.snd Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          eDB t1 t2 eps hTy
  | const Delta Sigma Gamma v ds =>
      intro _hlex
      exact ⟨TermDB.const v ds, rfl,
        HasTypeDB.const Delta Sigma (eraseCtx Gamma) v ds⟩
  | tadd Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
      intro hlex
      rcases ih1 (lexical_add_left hlex) with ⟨e1DB, hErase1, hTy1⟩
      rcases ih2 (lexical_output_of_typing h1 (lexical_add_right hlex)) with
        ⟨e2DB, hErase2, hTy2⟩
      have hErase2' : eraseTerm (ctxEnv Gamma1) e2 = some e2DB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved h1)]
        exact hErase2
      refine ⟨TermDB.add e1DB e2DB, ?_, ?_⟩
      · simp [eraseTerm, hErase1, hErase2']
      · exact HasTypeDB.tadd Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) e1DB e2DB ds eps1 eps2 hTy1 hTy2
  | tmul Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
      intro hlex
      rcases ih1 (lexical_mul_left hlex) with ⟨e1DB, hErase1, hTy1⟩
      rcases ih2 (lexical_output_of_typing h1 (lexical_mul_right hlex)) with
        ⟨e2DB, hErase2, hTy2⟩
      have hErase2' : eraseTerm (ctxEnv Gamma1) e2 = some e2DB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved h1)]
        exact hErase2
      refine ⟨TermDB.mul e1DB e2DB, ?_, ?_⟩
      · simp [eraseTerm, hErase1, hErase2']
      · exact HasTypeDB.tmul Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) e1DB e2DB ds eps1 eps2 hTy1 hTy2
  | tsum Delta Sigma Gamma1 Gamma2 e ds d eps hBody hmem ih =>
      intro hlex
      rcases ih (lexical_sum_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.sum eDB (eraseDim d), ?_, ?_⟩
      · rw [eraseTerm, hErase]
        rfl
      · exact HasTypeDB.tsum Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          eDB ds (eraseDim d) eps hTy (rem ds d) trivial
  | texpand Delta Sigma Gamma1 Gamma2 e ds d eps hBody ih =>
      intro hlex
      rcases ih (lexical_expand_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.expand eDB (eraseDim d) 0, ?_, ?_⟩
      · rw [eraseTerm, hErase]
        rfl
      · exact HasTypeDB.texpand Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          eDB ds (eraseDim d) 0 eps hTy (ins ds d) trivial
  | uniformLike Delta Sigma Gamma1 Gamma2 e ds lo hi eps hBody ih =>
      intro hlex
      rcases ih (lexical_uniformLike_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.uniformLike eDB lo hi, ?_, ?_⟩
      · rw [eraseTerm, hErase]
        rfl
      · exact HasTypeDB.uniformLike Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          eDB ds lo hi eps hTy
  | perform Delta Sigma Gamma1 Gamma2 op e tArg tRet eps hBody hsig ih =>
      intro hlex
      rcases ih (lexical_perform_body hlex) with ⟨eDB, hErase, hTy⟩
      refine ⟨TermDB.perform op eDB, ?_, ?_⟩
      · simp [eraseTerm, hErase]
      · exact HasTypeDB.perform Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          op eDB tArg tRet eps hTy hsig
  | handle Delta Sigma Gamma1 Gamma2 Gamma3 body clauses t epsH epsB
      hBody hOpsIn hClsIn hCover hClauses ihBody ihClauses =>
      intro hlex
      rcases ihBody (lexical_handle_body hlex) with ⟨bodyDB, hEraseBody, hTyBody⟩
      have hClausesLex :
          ClausesLexical Gamma2 t (EffectRow.removeOps epsB epsH) clauses := by
        intro op x k hb tArg tRet hmem
        have hlex' :
            LexicallyScoped
              (Gamma1 ++ [(x, some tArg), (k, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))]) hb :=
          lexical_handle_clause
            (tx := tArg)
            (tk := Typ.arrow tRet t (EffectRow.removeOps epsB epsH))
            hlex hmem
        have hEq :
            List.map Prod.fst
              (Gamma1 ++ [(x, some tArg), (k, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))]) =
            List.map Prod.fst
              (Gamma2 ++ [(x, some tArg), (k, some (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)))]) := by
          simp [List.map_append, hasType_names_preserved hBody]
        exact lexical_of_names_eq hEq hlex'
      rcases ihClauses hClausesLex with ⟨clausesDB, hEraseClauses, hTyClauses⟩
      have hEraseClauses' : eraseClauses (ctxEnv Gamma1) clauses = some clausesDB := by
        rw [ctxEnv_eq_of_names_eq (hasType_names_preserved hBody)]
        exact hEraseClauses
      refine ⟨TermDB.handle epsH bodyDB clausesDB, ?_, ?_⟩
      · simp [eraseTerm, hEraseBody, hEraseClauses']
      · refine HasTypeDB.handle Delta Sigma (eraseCtx Gamma1) (eraseCtx Gamma2)
          (eraseCtx Gamma3) bodyDB clausesDB t epsH epsB hTyBody hOpsIn ?_ ?_ hTyClauses
        · intro clDB hmemDB
          rcases eraseClauses_named_of_mem hEraseClauses hmemDB with ⟨cl, hcl, hEqCl⟩
          simpa [hEqCl] using hClsIn cl hcl
        · intro op hop
          rcases hCover op hop with ⟨cl, hcl, hEqOp⟩
          rcases eraseClauses_exists_of_mem_named hEraseClauses hcl with ⟨hbDB, hmemDB⟩
          exact ⟨(op, hbDB), by simpa [hEqOp] using hmemDB, rfl⟩
  | tgrad Delta Sigma Gamma x ds dsOut body eps slot hBody hsub ih =>
      intro hlex
      rcases ih (lexical_grad_body hlex) with ⟨bodyDB, hEraseBody, hTyBody⟩
      have hEraseBody' :
          eraseTerm (x :: ctxEnv Gamma) body = some bodyDB := by
        simpa using hEraseBody
      have hTyBody' :
          HasTypeDB (Capability.diff :: Delta) Sigma (some (Typ.tensor ds) :: eraseCtx Gamma)
            bodyDB (Typ.tensor dsOut) eps (slot :: eraseCtx Gamma) := by
        simpa [eraseCtx_append_singleton] using hTyBody
      refine ⟨TermDB.grad (Typ.tensor ds) (Typ.tensor dsOut) bodyDB, ?_, ?_⟩
      · simp [eraseTerm, hEraseBody']
      · exact HasTypeDB.tgrad Delta Sigma (eraseCtx Gamma) slot ds dsOut bodyDB eps
          hTyBody' hsub
  | tvmap Delta Sigma Gamma x t1 t2 body eps d slot hBody ih =>
      intro hlex
      rcases ih (lexical_vmap_body hlex) with ⟨bodyDB, hEraseBody, hTyBody⟩
      have hEraseBody' :
          eraseTerm (x :: ctxEnv Gamma) body = some bodyDB := by
        simpa using hEraseBody
      have hTyBody' :
          HasTypeDB Delta Sigma (some t1 :: eraseCtx Gamma) bodyDB t2 eps
            (slot :: eraseCtx Gamma) := by
        simpa [eraseCtx_append_singleton] using hTyBody
      refine ⟨TermDB.vmap t1 bodyDB, ?_, ?_⟩
      · simp [eraseTerm, hEraseBody']
      · exact HasTypeDB.tvmap Delta Sigma (eraseCtx Gamma) slot t1 t2 bodyDB eps d
          hTyBody'
  | loc Delta Sigma Gamma ell t hlook =>
      intro _hlex
      exact ⟨TermDB.loc ell, rfl, HasTypeDB.loc Delta Sigma (eraseCtx Gamma) ell t hlook⟩
  | subEff Delta Sigma Gamma GammaOut e t eps eps' hBody hsub ih =>
      intro hlex
      rcases ih hlex with ⟨eDB, hErase, hTy⟩
      exact ⟨eDB, hErase,
        HasTypeDB.subEff Delta Sigma (eraseCtx Gamma) (eraseCtx GammaOut) eDB t eps eps' hTy hsub⟩
  | nil Delta Sigma Gamma2 t epsR =>
      exact ⟨[], rfl, ClausesTypedDB.nil Delta Sigma (eraseCtx Gamma2) t epsR⟩
  | cons Delta Sigma Gamma2 Gamma3 t tArg tRet epsR op x k hb rest slotX slotK
      hMatch hBody hRest ihBody ihRest =>
      rename_i hlex
      have hHeadLex :
          LexicallyScoped
            (Gamma2 ++ [(x, some tArg), (k, some (Typ.arrow tRet t epsR))]) hb :=
        hlex (op := op) (x := x) (k := k) (hb := hb) (tArg := tArg) (tRet := tRet) (by simp)
      rcases ihBody hHeadLex with ⟨hbDB, hEraseBody, hTyBody⟩
      have hEraseBody' :
          eraseTerm (k :: x :: ctxEnv Gamma2) hb = some hbDB := by
        simpa using hEraseBody
      have hTyBody' :
          HasTypeDB Delta Sigma
            (some (Typ.arrow tRet t epsR) :: some tArg :: eraseCtx Gamma2)
            hbDB t epsR (slotK :: slotX :: eraseCtx Gamma3) := by
        simpa [eraseCtx_append_pair] using hTyBody
      have hRestLex : ClausesLexical Gamma2 t epsR rest := by
        intro op' x' k' hb' tArg' tRet' hmem
        exact hlex
          (op := op') (x := x') (k := k') (hb := hb')
          (tArg := tArg') (tRet := tRet') (by simp [hmem])
      rcases ihRest hRestLex with ⟨restDB, hEraseRest, hTyRest⟩
      refine ⟨(op, hbDB) :: restDB, ?_, ?_⟩
      · simp [eraseClauses, hEraseBody', hEraseRest]
      · exact ClausesTypedDB.cons Delta Sigma (eraseCtx Gamma2) (eraseCtx Gamma3)
          slotK slotX t tArg tRet epsR op hbDB restDB hMatch hTyBody' hTyRest


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

mutual

theorem eraseTerm_subst_split
    {ρin ρout : BinderEnv} {v : Term} {x : String} {vDB : TermDB}
    (hρin : x ∉ ρin)
    (hv : eraseTerm [] v = some vDB) :
    ∀ {body : Term} {bodyDB : TermDB},
      eraseTerm (ρin ++ x :: ρout) body = some bodyDB ->
      x ∉ boundVars body ->
      eraseTerm (ρin ++ ρout) (subst body v x) =
        some (substDBAux ρin.length vDB bodyDB)
  | Term.var y, bodyDB, hbody, hx => by
      by_cases hxy : y = x
      · subst hxy
        simp [eraseTerm, lookupBinder_append_target hρin] at hbody
        cases hbody
        have hvρ : eraseTerm (ρin ++ ρout) v = some vDB := by
          simpa using eraseTerm_suffix hv (ρin ++ ρout)
        simpa [subst, substDBAux] using hvρ
      · cases hρin_y : lookupBinder ρin y with
        | some i =>
            have hi : i < ρin.length := lookupBinder_some_lt_length hρin_y
            simp [eraseTerm, hxy, lookupBinder_append_left hρin_y] at hbody
            cases hbody
            simp [eraseTerm, subst, hxy, lookupBinder_append_left hρin_y,
              substDBAux, Nat.ne_of_lt hi, hi]
        | none =>
            cases hρout_y : lookupBinder ρout y with
            | none =>
                simp [eraseTerm, hxy, lookupBinder_append_after_target hρin_y hxy, hρout_y] at hbody
            | some j =>
                simp [eraseTerm, hxy, lookupBinder_append_after_target hρin_y hxy, hρout_y] at hbody
                cases hbody
                have hneq : j + (ρin.length + 1) ≠ ρin.length := by omega
                have hlt : ¬ j + (ρin.length + 1) < ρin.length := by omega
                simp [eraseTerm, subst, hxy, lookupBinder_append_of_none hρin_y, hρout_y,
                  substDBAux, hneq, hlt, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
  | Term.abs y t body, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases hbody' : eraseTerm (y :: (ρin ++ x :: ρout)) body with _ | bodyDB' <;>
        simp [eraseTerm, hbody'] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hxy' : x ≠ y := by
        intro h
        exact hyx h.symm
      have hρin' : x ∉ y :: ρin := by
        simp [hxy', hρin]
      have ih := eraseTerm_subst_split (ρin := y :: ρin) (ρout := ρout)
        (v := v) (x := x) hρin' hv hbody' hx.2
      have ih' :
          eraseTerm (y :: (ρin ++ ρout)) (subst body v x) =
            some (substDBAux (ρin.length + 1) vDB bodyDB') := by
        simpa [List.cons_append] using ih
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih', List.cons_append]
  | Term.app e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρin ++ x :: ρout) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρin ++ x :: ρout) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h1 hx.1
      have ih2 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.letBind y e1 e2, bodyDB, hbody, hx => by
      simp [boundVars, List.mem_append, not_or] at hx
      rcases h1 : eraseTerm (ρin ++ x :: ρout) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (y :: (ρin ++ x :: ρout)) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hxy' : x ≠ y := by
        intro h
        exact hyx h.symm
      have ih1 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h1 hx.2.1
      have hρin' : x ∉ y :: ρin := by
        simp [hxy', hρin]
      have ih2 := eraseTerm_subst_split (ρin := y :: ρin) (ρout := ρout)
        (v := v) (x := x) hρin' hv h2 hx.2.2
      have ih2' :
          eraseTerm (y :: (ρin ++ ρout)) (subst e2 v x) =
            some (substDBAux (ρin.length + 1) vDB e2DB) := by
        simpa [List.cons_append] using ih2
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih1, ih2', List.cons_append]
  | Term.copy e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.letpair y z e1 e2, bodyDB, hbody, hx => by
      simp [boundVars, List.mem_append, not_or] at hx
      rcases h1 : eraseTerm (ρin ++ x :: ρout) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (z :: y :: (ρin ++ x :: ρout)) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hzx : z ≠ x := by
        intro h
        exact hx.2.1 h.symm
      have hxy' : x ≠ y := by
        intro h
        exact hyx h.symm
      have hxz' : x ≠ z := by
        intro h
        exact hzx h.symm
      have ih1 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h1 hx.2.2.1
      have hρin' : x ∉ z :: y :: ρin := by
        simp [hxz', hxy', hρin]
      have ih2 := eraseTerm_subst_split (ρin := z :: y :: ρin) (ρout := ρout)
        (v := v) (x := x) hρin' hv h2 hx.2.2.2
      have ih2' :
          eraseTerm (z :: y :: (ρin ++ ρout)) (subst e2 v x) =
            some (substDBAux (ρin.length + 2) vDB e2DB) := by
        simpa [List.cons_append] using ih2
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, hzx, substDBAux, hvlift, ih1, ih2', List.cons_append]
  | Term.pair e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρin ++ x :: ρout) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρin ++ x :: ρout) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h1 hx.1
      have ih2 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.fst e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.snd e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
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
      rcases h1 : eraseTerm (ρin ++ x :: ρout) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρin ++ x :: ρout) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h1 hx.1
      have ih2 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.mul e1 e2, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases h1 : eraseTerm (ρin ++ x :: ρout) e1 with _ | e1DB <;> simp [eraseTerm, h1] at hbody
      rcases h2 : eraseTerm (ρin ++ x :: ρout) e2 with _ | e2DB <;> simp [eraseTerm, h1, h2] at hbody
      cases hbody
      have ih1 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h1 hx.1
      have ih2 := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv h2 hx.2
      simp [eraseTerm, subst, substDBAux, ih1, ih2]
  | Term.sum e d, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB
      · simp [eraseTerm, he] at hbody
        cases hbody
      · simp [eraseTerm, he] at hbody
        cases hbody
        have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
        simp [eraseTerm, subst, ih]
        rfl
  | Term.expand e d, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB
      · simp [eraseTerm, he] at hbody
        cases hbody
      · simp [eraseTerm, he] at hbody
        cases hbody
        have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
        simp [eraseTerm, subst, ih]
        rfl
  | Term.uniformLike e lo hi, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB
      · simp [eraseTerm, he] at hbody
        cases hbody
      · simp [eraseTerm, he] at hbody
        cases hbody
        have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
        simp [eraseTerm, subst, ih]
        rfl
  | Term.grad y t tOut body, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases hbody' : eraseTerm (y :: (ρin ++ x :: ρout)) body with _ | bodyDB' <;>
        simp [eraseTerm, hbody'] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hxy' : x ≠ y := by
        intro h
        exact hyx h.symm
      have hρin' : x ∉ y :: ρin := by
        simp [hxy', hρin]
      have ih := eraseTerm_subst_split (ρin := y :: ρin) (ρout := ρout)
        (v := v) (x := x) hρin' hv hbody' hx.2
      have ih' :
          eraseTerm (y :: (ρin ++ ρout)) (subst body v x) =
            some (substDBAux (ρin.length + 1) vDB bodyDB') := by
        simpa [List.cons_append] using ih
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih', List.cons_append]
  | Term.vmap y t body, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases hbody' : eraseTerm (y :: (ρin ++ x :: ρout)) body with _ | bodyDB' <;>
        simp [eraseTerm, hbody'] at hbody
      cases hbody
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hxy' : x ≠ y := by
        intro h
        exact hyx h.symm
      have hρin' : x ∉ y :: ρin := by
        simp [hxy', hρin]
      have ih := eraseTerm_subst_split (ρin := y :: ρin) (ρout := ρout)
        (v := v) (x := x) hρin' hv hbody' hx.2
      have ih' :
          eraseTerm (y :: (ρin ++ ρout)) (subst body v x) =
            some (substDBAux (ρin.length + 1) vDB bodyDB') := by
        simpa [List.cons_append] using ih
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseTerm, subst, hyx, substDBAux, hvlift, ih', List.cons_append]
  | Term.handle epsH body clauses, bodyDB, hbody, hx => by
      simp [boundVars, List.mem_append, not_or] at hx
      rcases hbody' : eraseTerm (ρin ++ x :: ρout) body with _ | bodyDB' <;>
        simp [eraseTerm, hbody'] at hbody
      rcases hcls : eraseClauses (ρin ++ x :: ρout) clauses with _ | clsDB <;>
        simp [eraseTerm, hbody', hcls] at hbody
      cases hbody
      have ihBody := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv hbody' hx.1
      have ihClauses := eraseClauses_subst_split (ρin := ρin) (ρout := ρout) hρin hv hcls hx.2
      simp [eraseTerm, subst, substDBAux, ihBody, ihClauses]
  | Term.perform op e, bodyDB, hbody, hx => by
      simp [boundVars] at hx
      rcases he : eraseTerm (ρin ++ x :: ρout) e with _ | eDB <;> simp [eraseTerm, he] at hbody
      cases hbody
      have ih := eraseTerm_subst_split (ρin := ρin) (ρout := ρout) hρin hv he hx
      simp [eraseTerm, subst, substDBAux, ih]
  | Term.loc ell, bodyDB, hbody, hx => by
      simp [eraseTerm] at hbody
      cases hbody
      simp [eraseTerm, subst, substDBAux]

theorem eraseClauses_subst_split
    {ρin ρout : BinderEnv} {v : Term} {x : String} {vDB : TermDB}
    (hρin : x ∉ ρin)
    (hv : eraseTerm [] v = some vDB) :
    ∀ {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses (ρin ++ x :: ρout) clauses = some clausesDB ->
      x ∉ boundVarsClauses clauses ->
      eraseClauses (ρin ++ ρout) (substClauses clauses v x) =
        some (substClausesDBAux ρin.length vDB clausesDB)
  | [], clausesDB, hclauses, hx => by
      simp [eraseClauses] at hclauses
      cases hclauses
      simp [eraseClauses, substClauses, substClausesDBAux]
  | (op, y, k, hb) :: rest, clausesDB, hclauses, hx => by
      simp [boundVarsClauses, List.mem_append, not_or] at hx
      rcases hhb : eraseTerm (k :: y :: (ρin ++ x :: ρout)) hb with _ | hbDB <;>
        simp [eraseClauses, hhb] at hclauses
      rcases hrest : eraseClauses (ρin ++ x :: ρout) rest with _ | restDB <;>
        simp [eraseClauses, hhb, hrest] at hclauses
      cases hclauses
      have hyx : y ≠ x := by
        intro h
        exact hx.1 h.symm
      have hkx : k ≠ x := by
        intro h
        exact hx.2.1 h.symm
      have hxy' : x ≠ y := by
        intro h
        exact hyx h.symm
      have hxk' : x ≠ k := by
        intro h
        exact hkx h.symm
      have hρin' : x ∉ k :: y :: ρin := by
        simp [hxk', hxy', hρin]
      have ihBody := eraseTerm_subst_split (ρin := k :: y :: ρin) (ρout := ρout)
        (v := v) (x := x) hρin' hv hhb hx.2.2.1
      have ihBody' :
          eraseTerm (k :: y :: (ρin ++ ρout)) (subst hb v x) =
            some (substDBAux (ρin.length + 2) vDB hbDB) := by
        simpa [List.cons_append] using ihBody
      have ihRest := eraseClauses_subst_split (ρin := ρin) (ρout := ρout) hρin hv hrest hx.2.2.2
      have hvlift : lift vDB = vDB := by
        simpa [lift] using
          (eraseTerm_liftAux_len (ρ := []) (e := v) (eDB := vDB) hv)
      simp [eraseClauses, substClauses, hyx, hkx, substClausesDBAux, hvlift, ihBody', ihRest]

end

theorem eraseTerm_subst_head
    {ρ : BinderEnv} {v : Term} {x : String} {vDB : TermDB}
    (hv : eraseTerm [] v = some vDB) :
    ∀ {body : Term} {bodyDB : TermDB},
      eraseTerm (x :: ρ) body = some bodyDB ->
      x ∉ boundVars body ->
      eraseTerm ρ (subst body v x) = some (substDBAux 0 vDB bodyDB) := by
  intro body bodyDB hbody hx
  simpa using
    (eraseTerm_subst_split (ρin := []) (ρout := ρ) (v := v) (x := x) (by simp) hv hbody hx)

theorem eraseClauses_subst_head
    {ρ : BinderEnv} {v : Term} {x : String} {vDB : TermDB}
    (hv : eraseTerm [] v = some vDB) :
    ∀ {clauses : List (EffectLabel × String × String × Term)}
      {clausesDB : List (EffectLabel × TermDB)},
      eraseClauses (x :: ρ) clauses = some clausesDB ->
      x ∉ boundVarsClauses clauses ->
      eraseClauses ρ (substClauses clauses v x) =
        some (substClausesDBAux 0 vDB clausesDB) := by
  intro clauses clausesDB hclauses hx
  simpa using
    (eraseClauses_subst_split (ρin := []) (ρout := ρ) (v := v) (x := x) (by simp) hv hclauses hx)

mutual

/-- Closed substitutions commute at distinct variable names. This is
    the named-side reordering lemma needed when a two-binder reduction
    substitutes the tail binder first to match `subst_preserves_typing`,
    then swaps back to the operational order. -/
theorem subst_commute_closed
    (e v1 v2 : Term) (x y : String)
    (hxy : x ≠ y)
    (hv1 : Closed v1)
    (hv2 : Closed v2) :
    subst (subst e v1 x) v2 y =
      subst (subst e v2 y) v1 x := by
  match e with
  | Term.var z =>
      by_cases hzx : z = x
      · subst z
        simp [subst, hxy, subst_closed _ _ _ hv1]
      · by_cases hzy : z = y
        · subst z
          simp [subst, hzx, subst_closed _ _ _ hv2]
        · simp [subst, hzx, hzy]
  | Term.abs z t body =>
      by_cases hzx : z = x
      · subst z
        simp [subst, hxy]
      · by_cases hzy : z = y
        · subst z
          simp [subst, hzx]
        · simp [subst, hzx, hzy, subst_commute_closed body v1 v2 x y hxy hv1 hv2]
  | Term.app e1 e2 =>
      simp [subst, subst_commute_closed e1 v1 v2 x y hxy hv1 hv2,
        subst_commute_closed e2 v1 v2 x y hxy hv1 hv2]
  | Term.letBind z e1 e2 =>
      by_cases hzx : z = x
      · subst z
        simp [subst, hxy,
          subst_commute_closed e1 v1 v2 x y hxy hv1 hv2]
      · by_cases hzy : z = y
        · subst z
          simp [subst, hzx,
            subst_commute_closed e1 v1 v2 x y hxy hv1 hv2]
        · simp [subst, hzx, hzy,
            subst_commute_closed e1 v1 v2 x y hxy hv1 hv2,
            subst_commute_closed e2 v1 v2 x y hxy hv1 hv2]
  | Term.copy e =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.letpair z w e1 e2 =>
      by_cases hzx : z = x
      · subst z
        simp [subst, hxy,
          subst_commute_closed e1 v1 v2 x y hxy hv1 hv2]
      · by_cases hzy : z = y
        · subst z
          simp [subst, hzx,
            subst_commute_closed e1 v1 v2 x y hxy hv1 hv2]
        · by_cases hwx : w = x
          · subst w
            simp [subst, hzx, hzy, hxy,
              subst_commute_closed e1 v1 v2 x y hxy hv1 hv2]
          · by_cases hwy : w = y
            · subst w
              simp [subst, hzx, hzy, hwx,
                subst_commute_closed e1 v1 v2 x y hxy hv1 hv2]
            · simp [subst, hzx, hzy, hwx, hwy,
                subst_commute_closed e1 v1 v2 x y hxy hv1 hv2,
                subst_commute_closed e2 v1 v2 x y hxy hv1 hv2]
  | Term.pair e1 e2 =>
      simp [subst, subst_commute_closed e1 v1 v2 x y hxy hv1 hv2,
        subst_commute_closed e2 v1 v2 x y hxy hv1 hv2]
  | Term.fst e =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.snd e =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.unit =>
      simp [subst]
  | Term.const c ds =>
      simp [subst]
  | Term.add e1 e2 =>
      simp [subst, subst_commute_closed e1 v1 v2 x y hxy hv1 hv2,
        subst_commute_closed e2 v1 v2 x y hxy hv1 hv2]
  | Term.mul e1 e2 =>
      simp [subst, subst_commute_closed e1 v1 v2 x y hxy hv1 hv2,
        subst_commute_closed e2 v1 v2 x y hxy hv1 hv2]
  | Term.sum e d =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.expand e d =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.uniformLike e lo hi =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.grad z t tOut body =>
      by_cases hzx : z = x
      · subst z
        simp [subst, hxy]
      · by_cases hzy : z = y
        · subst z
          simp [subst, hzx]
        · simp [subst, hzx, hzy, subst_commute_closed body v1 v2 x y hxy hv1 hv2]
  | Term.vmap z t body =>
      by_cases hzx : z = x
      · subst z
        simp [subst, hxy]
      · by_cases hzy : z = y
        · subst z
          simp [subst, hzx]
        · simp [subst, hzx, hzy, subst_commute_closed body v1 v2 x y hxy hv1 hv2]
  | Term.handle epsH body clauses =>
      simp [subst, subst_commute_closed body v1 v2 x y hxy hv1 hv2,
        substClauses_commute_closed clauses v1 v2 x y hxy hv1 hv2]
  | Term.perform op e =>
      simp [subst, subst_commute_closed e v1 v2 x y hxy hv1 hv2]
  | Term.loc ell =>
      simp [subst]

theorem substClauses_commute_closed
    (clauses : List (EffectLabel × String × String × Term)) (v1 v2 : Term) (x y : String)
    (hxy : x ≠ y)
    (hv1 : Closed v1)
    (hv2 : Closed v2) :
    substClauses (substClauses clauses v1 x) v2 y =
      substClauses (substClauses clauses v2 y) v1 x := by
  match clauses with
  | [] =>
      simp [substClauses]
  | (op, z, k, hb) :: rest =>
      by_cases hzx : z = x
      · subst z
        simp [substClauses, hxy,
          substClauses_commute_closed rest v1 v2 x y hxy hv1 hv2]
      · by_cases hzy : z = y
        · subst z
          simp [substClauses, hzx,
            substClauses_commute_closed rest v1 v2 x y hxy hv1 hv2]
        · by_cases hkx : k = x
          · subst k
            simp [substClauses, hzx, hzy, hxy,
              substClauses_commute_closed rest v1 v2 x y hxy hv1 hv2]
          · by_cases hky : k = y
            · subst k
              simp [substClauses, hzx, hzy, hkx,
                substClauses_commute_closed rest v1 v2 x y hxy hv1 hv2]
            · simp [substClauses, hzx, hzy, hkx, hky,
                subst_commute_closed hb v1 v2 x y hxy hv1 hv2,
                substClauses_commute_closed rest v1 v2 x y hxy hv1 hv2]

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
  rcases HasType.app_inv_sub_bridge h_typ with
    ⟨GammaMid, t1, epsBody, epsFun, epsArg, hFun, hArg, hSub⟩
  have hMid : GammaMid = [] := has_type_closed_output_of_closed_input hFun
  subst hMid
  rcases HasType.abs_inv hFun with
    ⟨tRet', epsBody', GammaBody, slot, hArrow, hBody, hOut⟩
  injection hArrow with hT1 hRet hEff
  subst t1
  subst tRet'
  subst epsBody'
  subst GammaBody
  have hClosed : Closed v := has_type_closed_term_of_closed_input hArg
  have hArgNil : HasType [] Sigma [] v tArg [] [] :=
    HasType.value_eff_polymorphic_bridge hArg hv []
  rcases subst_preserves_typing [] Sigma [] [(x, slot)]
      x tArg tRet epsBody body v hBody hArgNil hClosed with
    ⟨GammaSub, hSubst⟩
  have hGammaSub : GammaSub = [] := has_type_closed_output_of_closed_input hSubst
  subst hGammaSub
  have hSubBody :
      SubEffRow epsBody eps := by
    exact SubEffRow.trans
      (SubEffRow.union_right (EffectRow.union epsFun epsArg) epsBody)
      hSub
  exact HasType.subEff [] Sigma [] [] _ _ epsBody eps hSubst hSubBody

/-- Named-facing wrapper for the let-binding redex case in
    preservation. Same blocker profile as `preservation_beta_via_db`. -/
theorem preservation_letBind_via_db
    {Sigma : StoreTyp}
    {x : String} {v body : Term} {t : Typ} {eps : EffectRow}
    (h_typ : HasType [] Sigma [] (Term.letBind x v body) t eps [])
    (hv : IsValue v)
    (h_scope : WellScoped (Term.letBind x v body)) :
    HasType [] Sigma [] (subst body v x) t eps [] := by
  rcases HasType.letBind_inv_sub_bridge h_typ with
    ⟨GammaMid, GammaBody, t1, epsVal, epsBody, slot, hVal, hBody, hOut, hSub⟩
  have hMid : GammaMid = [] := has_type_closed_output_of_closed_input hVal
  subst hMid
  subst GammaBody
  have hClosed : Closed v := has_type_closed_term_of_closed_input hVal
  have hValNil : HasType [] Sigma [] v t1 [] [] :=
    HasType.value_eff_polymorphic_bridge hVal hv []
  rcases subst_preserves_typing [] Sigma [] [(x, slot)]
      x t1 t epsBody body v hBody hValNil hClosed with
    ⟨GammaSub, hSubst⟩
  have hGammaSub : GammaSub = [] := has_type_closed_output_of_closed_input hSubst
  subst hGammaSub
  have hSubBody :
      SubEffRow epsBody eps := by
    exact SubEffRow.trans (SubEffRow.union_right epsVal epsBody) hSub
  exact HasType.subEff [] Sigma [] [] _ _ epsBody eps hSubst hSubBody

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
  rcases HasType.letpair_inv_sub_bridge h_typ with
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
  have hxy : x ≠ y := wellScoped_letpair_names_ne h_scope
  have hscopeNodup :
      (x :: y :: (boundVars v1 ++ boundVars v2 ++ boundVars body)).Nodup := by
    simpa [WellScoped, boundVars, List.append_assoc] using h_scope
  have hxNotRest : x ∉ y :: (boundVars v1 ++ boundVars v2 ++ boundVars body) := by
    simpa using (List.nodup_cons.mp hscopeNodup).1
  have hyNotRest : y ∉ boundVars v1 ++ boundVars v2 ++ boundVars body := by
    simpa using (List.nodup_cons.mp (List.nodup_cons.mp hscopeNodup).2).1
  have hxNotV2 : x ∉ boundVars v2 := by
    intro hx
    exact hxNotRest (by simp [hx])
  have hxFreshV2 : freshInTerm x v2 := by
    unfold freshInTerm
    refine ⟨?_, hxNotV2⟩
    unfold Closed at hV2Closed
    intro hx
    rw [hV2Closed] at hx
    simp at hx
  rcases weakening_tail [] Sigma [] [] x t2 t1 [] v2 hV2Nil
      (by simp [linearCtxDom]) hxFreshV2 with
    ⟨GammaPre, GammaPost, hSplit, hV2Weak⟩
  simp at hSplit
  rcases hSplit with ⟨rfl, rfl⟩
  rcases subst_preserves_typing [] Sigma [(x, some t1)] [(x, slotX), (y, slotY)]
      y t2 t epsBody body v2 hBody hV2Weak hV2Closed with
    ⟨GammaAfterY, hAfterY⟩
  rcases subst_preserves_typing [] Sigma [] GammaAfterY
      x t1 t epsBody (subst body v2 y) v1 hAfterY hV1Nil hV1Closed with
    ⟨GammaFinal, hFinal⟩
  have hGammaFinal : GammaFinal = [] := has_type_closed_output_of_closed_input hFinal
  subst hGammaFinal
  have hswap :
      subst (subst body v1 x) v2 y =
        subst (subst body v2 y) v1 x := by
    exact subst_commute_closed body v1 v2 x y hxy hV1Closed hV2Closed
  have hSubBody : SubEffRow epsBody eps := by
    exact SubEffRow.trans (SubEffRow.union_right epsPair epsBody) hSub
  simpa [hswap] using
    (HasType.subEff [] Sigma [] [] _ _ epsBody eps hFinal hSubBody)

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
    (hsig : ∃ tArg, OpSigMatch op tArg tRet)
    (hmem : (op, x, k, hb) ∈ clauses)
    (h_scope : WellScoped (Term.handle epsH (Term.perform op v) clauses)) :
    HasType [] Sigma []
      (subst (subst hb v x) (Term.abs "y" tRet (Term.var "y")) k)
      t eps [] := by
  rcases HasType.handle_inv_strong_bridge h_typ with
    ⟨GammaBody, epsB, hPerform, _hOpsIn, _hClsIn, _hCover, hClauses, hSub⟩
  have hGammaBody : GammaBody = [] := has_type_closed_output_of_closed_input hPerform
  subst hGammaBody
  rcases HasType.perform_inv_bridge hPerform with
    ⟨tArgV, epsV, hV, hPerfSig, _hSubPerf⟩
  rcases ClausesTyped.mem_inv hClauses hmem with
    ⟨tArgClause, tRetClause, slotX, slotK, hClauseSig, hBody⟩
  obtain ⟨tArgStep, hStepSig⟩ := hsig
  have hArgEq : tArgClause = tArgV := OpSigMatch.arg_unique hClauseSig hPerfSig
  have hRetClause : tRetClause = tRet := OpSigMatch.ret_unique hClauseSig hStepSig
  have hRetGoal : t = tRet := OpSigMatch.ret_unique hPerfSig hStepSig
  subst tArgClause
  subst tRetClause
  subst t
  have hClosedV : Closed v := has_type_closed_term_of_closed_input hV
  have hVNil : HasType [] Sigma [] v tArgV [] [] :=
    HasType.value_eff_polymorphic_bridge hV hv []
  rcases wellScoped_handle_clause h_scope hmem with
    ⟨hxk, _hxNotHb, _hkNotHb, _hHbScope⟩
  have hIdBody0 :
      HasType [] Sigma
        ([(x, some tArgV)] ++ [("y", some tRet)])
        (Term.var "y") tRet []
        ([(x, some tArgV)] ++ [("y", none)]) := by
    simpa using (HasType.var [] Sigma [(x, some tArgV)] [] "y" tRet)
  have hIdBody :
      HasType [] Sigma
        ([(x, some tArgV)] ++ [("y", some tRet)])
        (Term.var "y") tRet (EffectRow.removeOps epsB epsH)
        ([(x, some tArgV)] ++ [("y", none)]) := by
    exact HasType.subEff [] Sigma
      ([(x, some tArgV)] ++ [("y", some tRet)])
      ([(x, some tArgV)] ++ [("y", none)])
      (Term.var "y") tRet [] (EffectRow.removeOps epsB epsH)
      hIdBody0
      (by
        intro op hop
        cases hop)
  have hIdAbs :
      HasType [] Sigma [(x, some tArgV)]
        (Term.abs "y" tRet (Term.var "y"))
        (Typ.arrow tRet tRet (EffectRow.removeOps epsB epsH)) []
        [(x, some tArgV)] := by
    exact HasType.abs [] Sigma
      [(x, some tArgV)] [(x, some tArgV)]
      "y" tRet tRet (EffectRow.removeOps epsB epsH)
      (Term.var "y") none hIdBody
  have hIdClosed : Closed (Term.abs "y" tRet (Term.var "y")) := by
    simp [Closed, freeVars]
  rcases subst_preserves_typing [] Sigma [(x, some tArgV)] [(x, slotX), (k, slotK)]
      k (Typ.arrow tRet tRet (EffectRow.removeOps epsB epsH)) tRet
      (EffectRow.removeOps epsB epsH) hb
      (Term.abs "y" tRet (Term.var "y")) hBody hIdAbs hIdClosed with
    ⟨GammaAfterK, hAfterK⟩
  rcases subst_preserves_typing [] Sigma [] GammaAfterK
      x tArgV tRet (EffectRow.removeOps epsB epsH)
      (subst hb (Term.abs "y" tRet (Term.var "y")) k) v
      hAfterK hVNil hClosedV with
    ⟨GammaFinal, hFinal⟩
  have hGammaFinal : GammaFinal = [] := has_type_closed_output_of_closed_input hFinal
  subst hGammaFinal
  have hswap :
      subst (subst hb v x) (Term.abs "y" tRet (Term.var "y")) k =
        subst (subst hb (Term.abs "y" tRet (Term.var "y")) k) v x := by
    exact subst_commute_closed hb v (Term.abs "y" tRet (Term.var "y")) x k
      hxk hClosedV hIdClosed
  simpa [hswap] using
    (HasType.subEff [] Sigma [] [] _ _ (EffectRow.removeOps epsB epsH) eps hFinal hSub)

end LaCaDiLE
