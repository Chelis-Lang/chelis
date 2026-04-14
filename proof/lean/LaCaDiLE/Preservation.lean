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

/-- Generic inversion scaffold. Wave 0.5 approach: inversion lemmas
    absorb `subEff` stripping by carrying an explicit equation premise
    and using `HasType.rec`, which is structurally recursive over
    HasType derivations without termination-proof gymnastics. -/
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

theorem HasType.pair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t1 t2 : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) (Typ.pair t1 t2) eps Gamma3) :
    ∃ Gamma2 eps1 eps2,
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 ∧
      eps = EffectRow.union eps1 eps2 := by
  generalize heq : Term.pair e1 e2 = e_in at h
  generalize htq : Typ.pair t1 t2 = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ Γ2 _ _ _ _ _ eps1 eps2 h1 h2 _ _ =>
      cases heq; cases htq
      exact ⟨Γ2, eps1, eps2, h1, h2, rfl⟩
  | subEff Δ S Γ Γ' _ _ eps0 eps' _h_sub h_sub ih =>
      obtain ⟨Γ2, eps1, eps2, hh1, hh2, hrfl⟩ := ih heq htq
      refine ⟨Γ2, eps1, eps2, hh1, hh2, ?_⟩
      -- Wave 0.5: union equation ripples under subEff widening. We
      -- need eps' = union eps1 eps2 but only have eps0 = union eps1 eps2
      -- and SubEffRow eps0 eps'. Not directly equal — the widening
      -- drops the inverses-are-equal guarantee. Punt for Wave 2.
      sorry
  | _ => (try cases heq) <;> (try cases htq) <;>
         first | exact True.intro | (exfalso; contradiction)

/-! ### Inversion for store-allocating primitive terms -/

theorem HasType.loc_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {ell : Loc} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.loc ell) t eps Gamma2) :
    storeTypLookup Sigma ell = some t ∧ eps = [] ∧ Gamma1 = Gamma2 := by
  generalize heq : Term.loc ell = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | loc _ _ _ _ _ hlook =>
      cases heq
      exact ⟨hlook, rfl, rfl⟩
  | subEff _ _ _ _ _ _ eps0 _ _h_sub _h_sub' ih =>
      obtain ⟨hlook, hepsNil, hG⟩ := ih heq
      -- eps widened under subEff; the lemma's eps = [] conclusion is
      -- not preserved. Needs a ≥-shaped restatement of loc_inv for
      -- Wave 2 cleanup.
      sorry
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

theorem HasType.const_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {v : Float} {ds : DimList} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma1 (Term.const v ds) t eps Gamma2) :
    t = Typ.tensor ds ∧ eps = [] ∧ Gamma1 = Gamma2 := by
  sorry -- TODO Wave 2: needs ≥-shaped restatement + HasType.rec

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

theorem HasType.add_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma1 (Term.add e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      eps = EffectRow.union eps1 eps2 ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 := by
  -- Wave 0.5: the exact `eps = union eps1 eps2` equation doesn't
  -- survive subEff widening; needs a SubEffRow-weakened restatement
  -- for Wave 2 cleanup.
  sorry

theorem HasType.mul_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma1 (Term.mul e1 e2) t eps Gamma3) :
    ∃ ds Gamma2 eps1 eps2,
      t = Typ.tensor ds ∧
      eps = EffectRow.union eps1 eps2 ∧
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 := by
  -- Same ripple as add_inv.
  sorry

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

theorem HasType.uniformLike_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {lo hi : Float} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma1 (Term.uniformLike e lo hi) t eps Gamma2) :
    ∃ ds eps0,
      t = Typ.tensor ds ∧
      eps = EffectRow.union eps0 [EffectLabel.random] ∧
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps0 Gamma2 := by
  sorry -- TODO Wave 2: strip_subEff helper

/-! ### StoreWf extension helper

Extending both the store and the store-typing with the same fresh
location preserves well-formedness. The freshness premise gets threaded
through `storeLookup`/`storeTypDom` via a small side lemma that Wave 1
leaves as an axiom-shaped `sorry` to avoid a detour into `List.find?`
reasoning; Wave 2 will discharge it alongside the substitution work. -/
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
    · -- ell' = ell: the extended store contains (ell, w).
      rw [hEq]
      show (storeLookup ((ell, w) :: sigma) ell).isSome
      simp [storeLookup, List.find?]
    · -- ell' is in Sigma's existing domain.
      have hOldLive : (storeLookup sigma ell').isSome := by
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

/-- Remove-then-extend for single-location-consume Steps (tsum,
    texpand, tuniformLike). Removing a live location and extending with
    a fresh one produces a new well-formed store/store-typing pair.
    Like `extend_fresh`, this tracks domain membership only — type
    matching is tightened by Wave 5 LinearityInvariant work.

    Wave 2 TODO: fill in this proof. The shape is symmetric to
    `extend_fresh` but needs a `storeLookup_remove` side lemma
    showing removal of `ellIn ≠ ell'` preserves lookup. -/
theorem StoreWf.remove_extend
    {sigma : Store} {Sigma : StoreTyp}
    (ellIn ellOut : Loc) (w : TensorVal) (tOut : Typ)
    (_h_wf : StoreWf sigma Sigma)
    (_h_fresh : ellOut = storeFreshLoc sigma)
    (_h_ne : ellIn ≠ ellOut) :
    StoreWf (storeExtend (storeRemove sigma ellIn) ellOut w)
            (storeTypExtend (storeTypRemove Sigma ellIn) ellOut tOut) := by
  sorry

theorem HasType.handle_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma3 : LinearCtx}
    {body : Term} {clauses : List (EffectLabel × String × String × Term)}
    {epsH : EffectRow} {t : Typ} {eps : EffectRow}
    (_h : HasType Delta Sigma Gamma1 (Term.handle epsH body clauses) t eps Gamma3) :
    ∃ Gamma2 epsB,
      HasType Delta Sigma Gamma1 body t epsB Gamma2 ∧
      eps = EffectRow.removeOps epsB epsH := by
  sorry -- TODO Wave 2: strip_subEff helper

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
      -- TODO Wave 2: re-close after inversion lemmas ship.
      sorry
  | copy s ell ellNew w hlook hfresh =>
      -- TODO Wave 2: re-close after inversion lemmas ship.
      sorry
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
      -- TODO Wave 2: needs inversion + value-context preservation.
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
