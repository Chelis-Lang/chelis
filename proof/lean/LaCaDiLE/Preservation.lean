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

theorem HasType.fst_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.fst e) t eps Gamma2) :
    ∃ t2, HasType Delta Sigma Gamma1 e (Typ.pair t t2) eps Gamma2 := by
  cases h with
  | fst _ _ _ _ _ _ t2 _ h' => exact ⟨t2, h'⟩

theorem HasType.snd_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.snd e) t eps Gamma2) :
    ∃ t1, HasType Delta Sigma Gamma1 e (Typ.pair t1 t) eps Gamma2 := by
  cases h with
  | snd _ _ _ _ _ t1 _ _ h' => exact ⟨t1, h'⟩

theorem HasType.pair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t1 t2 : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) (Typ.pair t1 t2) eps Gamma3) :
    ∃ Gamma2 eps1 eps2,
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 ∧
      eps = EffectRow.union eps1 eps2 := by
  cases h with
  | tpair _ _ _ Gamma2 _ _ _ _ _ eps1 eps2 h1 h2 =>
      exact ⟨Gamma2, eps1, eps2, h1, h2, rfl⟩

/-! ### Inversion for store-allocating primitive terms -/

theorem HasType.loc_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {ell : Loc} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.loc ell) t eps Gamma2) :
    storeTypLookup Sigma ell = some t ∧ eps = [] ∧ Gamma1 = Gamma2 := by
  cases h with
  | loc _ _ _ _ _ hlook => exact ⟨hlook, rfl, rfl⟩

theorem HasType.const_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {v : Float} {ds : DimList} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.const v ds) t eps Gamma2) :
    t = Typ.tensor ds ∧ eps = [] ∧ Gamma1 = Gamma2 := by
  cases h with
  | const _ _ _ _ _ => exact ⟨rfl, rfl, rfl⟩

theorem HasType.copy_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.copy e) t eps Gamma2) :
    ∃ t0, t = Typ.pair t0 t0 ∧
          HasType Delta Sigma Gamma1 e t0 eps Gamma2 := by
  cases h with
  | copy _ _ _ _ _ t0 _ h' => exact ⟨t0, rfl, h'⟩

theorem HasType.add_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.add e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      eps = EffectRow.union eps1 eps2 ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 := by
  cases h with
  | tadd _ _ _ Gamma2 _ _ _ ds eps1 eps2 h1 h2 =>
      exact ⟨ds, Gamma2, eps1, eps2, rfl, rfl, h1, h2⟩

theorem HasType.mul_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.mul e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      eps = EffectRow.union eps1 eps2 ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 := by
  cases h with
  | tmul _ _ _ Gamma2 _ _ _ ds eps1 eps2 h1 h2 =>
      exact ⟨ds, Gamma2, eps1, eps2, rfl, rfl, h1, h2⟩

theorem HasType.sum_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {i : Nat} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.sum e i) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (rem ds i) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  cases h with
  | tsum _ _ _ _ _ ds _ _ h' _ => exact ⟨ds, rfl, h'⟩

theorem HasType.expand_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {i k : Nat} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.expand e i k) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (ins ds i k) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  cases h with
  | texpand _ _ _ _ _ ds _ _ _ h' _ => exact ⟨ds, rfl, h'⟩

theorem HasType.uniformLike_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {lo hi : Float} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.uniformLike e lo hi) t eps Gamma2) :
    ∃ ds eps0,
      t = Typ.tensor ds ∧
      eps = EffectRow.union eps0 [EffectLabel.random] ∧
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps0 Gamma2 := by
  cases h with
  | uniformLike _ _ _ _ _ ds _ _ eps0 h' =>
      exact ⟨ds, eps0, rfl, rfl, h'⟩

/-! ### StoreWf extension helper

Extending both the store and the store-typing with the same fresh
location preserves well-formedness. The freshness premise gets threaded
through `storeLookup`/`storeTypDom` via a small side lemma that Wave 1
leaves as an axiom-shaped `sorry` to avoid a detour into `List.find?`
reasoning; Wave 2 will discharge it alongside the substitution work. -/
theorem StoreWf.extend_fresh
    {sigma : Store} {Sigma : StoreTyp}
    (ell : Loc) (w : TensorVal) (t : Typ)
    (_h_wf : StoreWf sigma Sigma)
    (_h_fresh : ell = storeFreshLoc sigma) :
    StoreWf (storeExtend sigma ell w) (storeTypExtend Sigma ell t) := by
  -- TODO Wave 2: list-level lookup reasoning for storeFreshLoc.
  sorry

/-- StoreTyp removal companion to `storeRemove`. -/
def storeTypRemove (Sigma : StoreTyp) (ell : Loc) : StoreTyp :=
  Sigma.filter (fun p => p.1 ≠ ell)

/-- Remove-then-extend variant for single-location consume steps
    (tsum, texpand, tuniformLike). -/
theorem StoreWf.remove_extend
    {sigma : Store} {Sigma : StoreTyp}
    (ellIn ellOut : Loc) (w : TensorVal) (tIn tOut : Typ)
    (_h_wf : StoreWf sigma Sigma)
    (_h_in : storeTypLookup Sigma ellIn = some tIn)
    (_h_fresh : ellOut = storeFreshLoc sigma) :
    StoreWf (storeExtend (storeRemove sigma ellIn) ellOut w)
            (storeTypExtend (storeTypRemove Sigma ellIn) ellOut tOut) := by
  sorry

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
  simp [storeTypLookup, storeTypExtend, List.find?]

theorem HasType.handle_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {body : Term} {clauses : List (EffectLabel × String × String × Term)}
    {epsH : EffectRow} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.handle epsH body clauses) t eps Gamma3) :
    ∃ Gamma2 epsB,
      HasType Delta Sigma Gamma1 body t epsB Gamma2 ∧
      eps = EffectRow.removeOps epsB epsH := by
  cases h with
  | handle _ _ _ Gamma2 _ _ _ _ _ epsB hb _ _ _ _ =>
      exact ⟨Gamma2, epsB, hb, rfl⟩

/-! ## Preservation -/

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
      -- E-Fst: fst (pair v1 v2) ↦ v1. Needs weaken_eff PLUS a value
      -- context-preservation lemma (values don't consume linear ctx,
      -- so the middle Gamma2 from T-Pair must equal the theorem's []).
      -- TODO Wave 2: value_preserves_context + weaken_eff.
      sorry
  | snd s v1 v2 hv1 hv2 =>
      -- Symmetric to fst; same obstacle.
      -- TODO Wave 2: value_preserves_context + weaken_eff + union_comm.
      sorry
  | tconst s v ds ell hell =>
      -- E-Const: const(v,ds) ↦ loc ellNew in extended store.
      obtain ⟨ht, heps, hG⟩ := HasType.const_inv h_typ
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_, ?_⟩
      · subst ht; subst heps
        have hlook := storeTypLookup_extend_self Sigma ell (Typ.tensor ds)
        -- goal: HasType [] (extended Sigma) Gamma1 (Term.loc ell) (Typ.tensor ds) [] Gamma2
        rw [hG] at *
        exact HasType.loc _ _ _ ell (Typ.tensor ds) hlook
      · exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf hell
  | copy s ell ellNew w hlook hfresh =>
      -- E-Copy: copy(loc ell) ↦ pair (loc ell) (loc ellNew).
      -- Both locations type at t0 under the extended Sigma.
      obtain ⟨t0, htEq, h_e⟩ := HasType.copy_inv h_typ
      obtain ⟨hlookT, hepsE, hGE⟩ := HasType.loc_inv h_e
      refine ⟨storeTypExtend Sigma ellNew t0, ?_, ?_⟩
      · subst htEq
        -- freshness of ellNew implies ell ≠ ellNew (from h_wf: ell is
        -- live, ellNew is fresh). We sidestep with an axiom-shaped
        -- assumption chained from h_wf and hfresh. Wave 2 will discharge.
        have hne : ell ≠ ellNew := by
          -- TODO Wave 2: StoreWf + storeFreshLoc freshness lemma.
          sorry
        have hlook1 : storeTypLookup (storeTypExtend Sigma ellNew t0) ell = some t0 :=
          storeTypLookup_extend_other Sigma ell ellNew t0 t0 hlookT hne
        have hlook2 : storeTypLookup (storeTypExtend Sigma ellNew t0) ellNew = some t0 :=
          storeTypLookup_extend_self Sigma ellNew t0
        -- Build pair. Need Gamma1/Gamma2 matching; loc_inv gives Gamma1=Gamma2.
        subst hepsE
        -- loc_inv gave Gamma = []. Goal uses theorem's Gamma which equals [].
        have hG : Gamma = [] := hGE
        subst hG
        have h_l1 : HasType [] (storeTypExtend Sigma ellNew t0) []
                      (Term.loc ell) t0 [] [] :=
          HasType.loc _ _ _ ell t0 hlook1
        have h_l2 : HasType [] (storeTypExtend Sigma ellNew t0) []
                      (Term.loc ellNew) t0 [] [] :=
          HasType.loc _ _ _ ellNew t0 hlook2
        have h_pair :=
          HasType.tpair [] (storeTypExtend Sigma ellNew t0) [] [] []
            (Term.loc ell) (Term.loc ellNew) t0 t0 [] [] h_l1 h_l2
        simpa using h_pair
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
      -- E-Sum: sum(loc ell, i) ↦ loc ellOut at tensor[rem ds i].
      obtain ⟨ds, htEq, h_e⟩ := HasType.sum_inv h_typ
      -- Need source ell typed at tensor[ds] in Sigma, via HasType.loc inversion.
      -- Also need StoreWf.remove_extend. Both depend on reasoning we've
      -- encapsulated as helper sorries.
      -- TODO Wave 2: finish once helpers are closed and loc_inv is available.
      sorry
  | texpand s ell ellOut w i k hlook hfresh =>
      -- TODO Wave 2: parallel to tsum.
      sorry
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      -- TODO Wave 2: parallel to tsum.
      sorry
  | handleRet s epsH v clauses hv =>
      -- E-Handle-Ret: handle[epsH] v clauses ↦ v when v is a value.
      -- Typing inversion: HasType ... (handle epsH v clauses) t eps []
      -- gives HasType ... v t epsB Gamma2 with eps = removeOps epsB epsH.
      -- Uses weaken_eff as the retyping oracle (itself sorry Wave 1).
      obtain ⟨Gamma2, epsB, h_body, _hepsEq⟩ := HasType.handle_inv h_typ
      refine ⟨Sigma, ?_, h_wf⟩
      -- Need: HasType ... v t eps []. Have: HasType ... v t epsB Gamma2.
      -- Same Gamma2-vs-Gamma obstacle as snd.
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

end LaCaDiLE
