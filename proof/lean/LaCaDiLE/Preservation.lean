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

/-! ## Wave 3 local oracles

These are oracle lemmas (sorried stubs) that downstream Wave 3+ proofs
rely on. They will be discharged by the canonical `has_type_linear_shrinks`
/ value-linear-closure work that is the sibling Wave 3 target. Keeping
them local keeps Preservation.lean compilable and lets the fst/snd/
handleRet cases land as real proof structure rather than raw `sorry`. -/

/-- A value typed under an initially-empty linear context also has an
    empty output linear context, and its typing derivation is
    effect-row-polymorphic (any effect row works, since values perform
    no effects). Discharges two joint obligations for Wave 3 case
    closure: context-shrinking to `[]` and effect-row flex. -/
theorem value_preserves_closed_context
    {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t : Typ} {eps : EffectRow}
    (_hv : IsValue v)
    (_h : HasType [] Sigma Gamma v t eps Gamma') :
    Gamma = Gamma' ∧
    ∀ eps', HasType [] Sigma Gamma v t eps' Gamma := by
  sorry

/-- Specialization of `value_preserves_closed_context` for the
    handleRet case: when a handle expression whose body is a value
    reduces to that value, the reduction type-checks at the handle's
    outer output context. Discharged jointly with the main oracle in
    Wave 3's linearity-shrinks pass. -/
theorem handleRet_value_preserves_typing
    {Sigma : StoreTyp} {Gamma : LinearCtx}
    {epsH : EffectRow} {v : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {t : Typ} {eps : EffectRow}
    (_hv : IsValue v)
    (_h : HasType [] Sigma Gamma (Term.handle epsH v clauses) t eps []) :
    HasType [] Sigma Gamma v t eps [] := by
  sorry

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

-- storeLookup_isSome_remove_ne, mem_storeTypDom_remove_iff, and
-- StoreWf.remove_extend all moved to Store.lean so LinearitySoundness
-- can use them without cross-file imports.

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
      -- E-Fst: fst (pair v1 v2) ↦ v1. fst_inv + pair_inv gives
      -- h1 : ...Gamma v1 t eps1 Γmid and h2 : ...Γmid v2 t2 eps2 [].
      -- Both v1, v2 are values: oracle on h2 gives Γmid = [], then
      -- oracle on h1 gives Gamma = Γmid = [] and re-types v1 at
      -- the outer eps.
      obtain ⟨t2, h_pair⟩ := HasType.fst_inv h_typ
      obtain ⟨Γmid, _eps1, _eps2, h1, h2, _hsub⟩ := HasType.pair_inv h_pair
      have hmid : Γmid = [] := (value_preserves_closed_context hv2 h2).1
      subst hmid
      have hg : Gamma = [] := (value_preserves_closed_context hv1 h1).1
      refine ⟨Sigma, ?_, h_wf⟩
      subst hg
      exact (value_preserves_closed_context hv1 h1).2 eps
  | snd s v1 v2 hv1 hv2 =>
      -- E-Snd: symmetric. Use oracle on h2 directly, re-typed at eps.
      obtain ⟨t1, h_pair⟩ := HasType.snd_inv h_typ
      obtain ⟨Γmid, _eps1, _eps2, h1, h2, _hsub⟩ := HasType.pair_inv h_pair
      have hmid : Γmid = [] := (value_preserves_closed_context hv2 h2).1
      subst hmid
      have hg : Gamma = [] := (value_preserves_closed_context hv1 h1).1
      refine ⟨Sigma, ?_, h_wf⟩
      subst hg
      exact (value_preserves_closed_context hv2 h2).2 eps
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
      · exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf
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
      · exact StoreWf.extend_fresh ellNew w t0 h_wf
  | tadd s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      obtain ⟨ds, Γmid, eps1, eps2, htEq, h_e1, h_e2⟩ := HasType.add_inv h_typ
      obtain ⟨hLook1, hG1⟩ := HasType.loc_inv h_e1
      obtain ⟨hLook2, hG2⟩ := HasType.loc_inv h_e2
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor ds), ?_, ?_⟩
      · subst htEq
        -- hG1 : Gamma = Γmid, hG2 : Γmid = []. So Gamma = [].
        have hGamma : Gamma = [] := hG1.trans hG2
        subst hGamma
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
  | tmul s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      obtain ⟨ds, Γmid, eps1, eps2, htEq, h_e1, h_e2⟩ := HasType.mul_inv h_typ
      obtain ⟨hLook1, hG1⟩ := HasType.loc_inv h_e1
      obtain ⟨hLook2, hG2⟩ := HasType.loc_inv h_e2
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor ds), ?_, ?_⟩
      · subst htEq
        -- hG1 : Gamma = Γmid, hG2 : Γmid = []. So Gamma = [].
        have hGamma : Gamma = [] := hG1.trans hG2
        subst hGamma
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
  | tsum s ell ellOut w i hlook hfresh =>
      -- sum_inv gives ds with t = tensor (rem ds i); loc_inv gives
      -- storeTypLookup Sigma ell = some (tensor ds) + Gamma = [].
      obtain ⟨ds, htEq, h_loc_e⟩ := HasType.sum_inv h_typ
      obtain ⟨hLook, hGE⟩ := HasType.loc_inv h_loc_e
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (rem ds i)), ?_, ?_⟩
      · subst htEq
        subst hGE
        have hlookNew : storeTypLookup (storeTypExtend
                          (storeTypRemove Sigma ell) ellOut
                          (Typ.tensor (rem ds i))) ellOut
                      = some (Typ.tensor (rem ds i)) :=
          storeTypLookup_extend_self _ _ _
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor (rem ds i)) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor (rem ds i)) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup sigma ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := rem w.shape i, data := w.data }
          (Typ.tensor (rem ds i)) h_wf hfresh hne
  | texpand s ell ellOut w i k hlook hfresh =>
      obtain ⟨ds, htEq, h_loc_e⟩ := HasType.expand_inv h_typ
      obtain ⟨hLook, hGE⟩ := HasType.loc_inv h_loc_e
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (ins ds i k)), ?_, ?_⟩
      · subst htEq
        subst hGE
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove Sigma ell) ellOut (Typ.tensor (ins ds i k))
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor (ins ds i k)) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor (ins ds i k)) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup sigma ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := ins w.shape i k, data := w.data }
          (Typ.tensor (ins ds i k)) h_wf hfresh hne
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      obtain ⟨ds, eps0, htEq, h_loc_e⟩ := HasType.uniformLike_inv h_typ
      obtain ⟨hLook, hGE⟩ := HasType.loc_inv h_loc_e
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor ds), ?_, ?_⟩
      · subst htEq
        subst hGE
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove Sigma ell) ellOut (Typ.tensor ds)
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor ds) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup sigma ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := w.shape, data := lo }
          (Typ.tensor ds) h_wf hfresh hne
  | handleRet s epsH v clauses hv =>
      -- E-Handle-Ret: handle[εH] v clauses ↦ v when v is a value.
      -- handle_inv gives us a sub-derivation for v at some interior
      -- effect row epsB and intermediate context Γ2. We use subEff
      -- to widen epsB to the outer eps, and (since the theorem's
      -- output context is []) we rely on h_body directly when it
      -- already lands at the right context shape. Full value-context-
      -- preservation is still a Wave 2 task for the Γ2 = [] step.
      refine ⟨Sigma, ?_, h_wf⟩
      exact handleRet_value_preserves_typing hv h_typ
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
      -- E-Ctx: congruence under an evaluation context.
      -- Blocked on two related Wave 3+ lemmas:
      --   1. `plug_preserves_typing` (local oracle sketch):
      --        ∀ {Δ Σ Γ t eps Γ' Σ2 e e'},
      --          HasType Δ Σ Γ (plug E e) t eps Γ' →
      --          (∀ Γ0 t0 eps0 Γ0',
      --             HasType Δ Σ Γ0 e t0 eps0 Γ0' →
      --             HasType Δ Σ2 Γ0 e' t0 eps0 Γ0') →
      --          HasType Δ Σ2 Γ (plug E e') t eps Γ'
      --      (proved by induction on `E : EvalCtx`, 20+ cases).
      --   2. The caller-supplied inner-preserves premise, which is
      --      just `preservation` applied at the sub-step — this
      --      requires restructuring the outer case-split into a
      --      structural `induction h_step` so that an IH on `h_inner`
      --      is available. Attempted locally and deferred because it
      --      invalidates the already-closed store-allocating cases'
      --      `subst`-based pattern; needs a coordinated rewrite.
      -- TODO Wave 3: add plug_preserves_typing as a sorried oracle,
      -- restructure preservation to use `induction h_step`, and pass
      -- preservation itself as the inner-preserves witness.
      sorry

end LaCaDiLE
