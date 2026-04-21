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
import LaCaDiLE.TranslationDB

namespace LaCaDiLE

/-! ## Wave 3 local oracles

These are oracle lemmas (sorried stubs) that downstream Wave 3+ proofs
rely on. They will be discharged by the canonical `has_type_linear_shrinks`
/ value-linear-closure work that is the sibling Wave 3 target. Keeping
them local keeps Preservation.lean compilable and lets the fst/snd/
handleRet cases land as real proof structure rather than raw `sorry`. -/

/-- Values are effect-row polymorphic: a value typed at any effect row
    can be re-typed at any other effect row with the same type and
    contexts. The four value forms (unit, abs, pair, loc) are all
    natively typed at `[]` by their introduction rules, so the proof is
    a structural recursion via `HasType.rec` followed by `subEff`
    widening from `[]` to the target row. -/
theorem HasType.value_eff_polymorphic
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma v t eps Gamma') (hv : IsValue v) :
    ∀ eps', HasType Delta Sigma Gamma v t eps' Gamma' := by
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | unit Δ S G =>
      intro eps'
      exact HasType.subEff Δ S G G _ _ [] eps' (HasType.unit Δ S G)
        (fun _ hm => by cases hm)
  | abs Δ S G1 G2 y t1 t2 epsBody body slot h_body =>
      intro eps'
      exact HasType.subEff Δ S G1 G2 _ _ [] eps'
        (HasType.abs Δ S G1 G2 y t1 t2 epsBody body slot h_body)
        (fun _ hm => by cases hm)
  | tpair Δ S G1 G2 G3 v1 v2 t1 t2 eps1 eps2 _hv1 _hv2 ih1 ih2 =>
      intro eps'
      cases hv with
      | pair _ _ hp1 hp2 =>
          have h1' := ih1 hp1 ([] : EffectRow)
          have h2' := ih2 hp2 ([] : EffectRow)
          exact HasType.subEff Δ S G1 G3 _ _ _ eps'
            (HasType.tpair Δ S G1 G2 G3 v1 v2 t1 t2 [] [] h1' h2')
            (fun _ hm => by cases hm)
  | loc Δ S G ell t' hlook =>
      intro eps'
      exact HasType.subEff Δ S G G _ _ [] eps'
        (HasType.loc Δ S G ell t' hlook)
        (fun _ hm => by cases hm)
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro eps'; exact ih hv eps'
  | nil => exact True.intro
  | cons => exact True.intro
  | _ => intro _; cases hv

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

theorem StoreWf.lookup_isSome_of_typing
    {sigma : Store} {Sigma : StoreTyp} {ell : Loc} {t : Typ}
    (h_wf : StoreWf sigma Sigma)
    (h : storeTypLookup Sigma ell = some t) :
    (storeLookup sigma ell).isSome := by
  exact h_wf.1 ell (storeTypLookup_mem_dom h)

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
  | tgrad Δ_ _ Γ_ x ds dsOut body ep slot _h hsub_eff ih =>
      exact HasType.tgrad Δ_ Sigma' Γ_ x ds dsOut body ep slot (ih hsub) hsub_eff
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

/-- Prefix a common outer context onto a typing derivation. This is the
    named analogue of the DB-side tail-rebase step: the extra slots sit
    outside every local binder introduced by the derivation, so the
    constructor shape is preserved structurally. -/
theorem hasType_prefix_weaken
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (outer : LinearCtx) :
    HasType Delta Sigma (outer ++ Gamma) e t eps (outer ++ Gamma') := by
  induction h using HasType.rec
    (motive_2 := fun Δ_ S_ Γ2_ Γ3_ t_ εR_ cls_ _ =>
      ClausesTyped Δ_ S_ (outer ++ Γ2_) (outer ++ Γ3_) t_ εR_ cls_) with
  | var Δ_ S_ Γpre Γpost x tx =>
      simpa [List.append_assoc] using
        (HasType.var Δ_ S_ (outer ++ Γpre) Γpost x tx)
  | unit Δ_ S_ Γ_ =>
      simpa [List.append_assoc] using (HasType.unit Δ_ S_ (outer ++ Γ_))
  | abs Δ_ S_ Γ1 Γ2 x t1 t2 eps_ body slot hbody ih =>
      have ih' :
          HasType Δ_ S_ ((outer ++ Γ1) ++ [(x, some t1)]) body t2 eps_
            ((outer ++ Γ2) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih
      exact HasType.abs Δ_ S_ (outer ++ Γ1) (outer ++ Γ2)
        x t1 t2 eps_ body slot ih'
  | app Δ_ S_ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.app Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 t1 t2 eps_ eps1 eps2 ih1 ih2
  | letBind Δ_ S_ Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 slot h1 h2 ih1 ih2 =>
      have ih2' :
          HasType Δ_ S_ ((outer ++ Γ2) ++ [(x, some t1)]) e2 t2 eps2
            ((outer ++ Γ3) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih2
      exact HasType.letBind Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        x e1 e2 t1 t2 eps1 eps2 slot ih1 ih2'
  | copy Δ_ S_ Γ1 Γ2 e0 ds ep hbody ih =>
      exact HasType.copy Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds ep ih
  | letpair Δ_ S_ Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY
      h1 h2 ih1 ih2 =>
      have ih2' :
          HasType Δ_ S_ ((outer ++ Γ2) ++ [(x, some t1), (y, some t2)]) e2 tr eps2
            ((outer ++ Γ3) ++ [(x, slotX), (y, slotY)]) := by
        simpa [List.append_assoc] using ih2
      exact HasType.letpair Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY ih1 ih2'
  | tpair Δ_ S_ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.tpair Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 t1 t2 eps1 eps2 ih1 ih2
  | fst Δ_ S_ Γ1 Γ2 e0 t1 t2 ep hbody ih =>
      exact HasType.fst Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 t1 t2 ep ih
  | snd Δ_ S_ Γ1 Γ2 e0 t1 t2 ep hbody ih =>
      exact HasType.snd Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 t1 t2 ep ih
  | const Δ_ S_ Γ_ v ds =>
      simpa [List.append_assoc] using (HasType.const Δ_ S_ (outer ++ Γ_) v ds)
  | tadd Δ_ S_ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.tadd Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 ds eps1 eps2 ih1 ih2
  | tmul Δ_ S_ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.tmul Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 ds eps1 eps2 ih1 ih2
  | tsum Δ_ S_ Γ1 Γ2 e0 ds d ep hbody hmem ih =>
      exact HasType.tsum Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds d ep ih hmem
  | texpand Δ_ S_ Γ1 Γ2 e0 ds d ep hbody ih =>
      exact HasType.texpand Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds d ep ih
  | uniformLike Δ_ S_ Γ1 Γ2 e0 ds lo hi ep hbody ih =>
      exact HasType.uniformLike Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds lo hi ep ih
  | perform Δ_ S_ Γ1 Γ2 op e0 tArg tRet ep hbody hM ih =>
      exact HasType.perform Δ_ S_ (outer ++ Γ1) (outer ++ Γ2)
        op e0 tArg tRet ep ih hM
  | handle Δ_ S_ Γ1 Γ2 Γ3 body clauses t_ epsH epsB hb hSubsH hClsH
      hCover hcls ih_body ih_clauses =>
      exact HasType.handle Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        body clauses t_ epsH epsB ih_body hSubsH hClsH hCover ih_clauses
  | tgrad Δ_ S_ Γ_ x ds dsOut body ep slot hbody hsub_eff ih =>
      have ih' :
          HasType (Capability.diff :: Δ_) S_
            ((outer ++ Γ_) ++ [(x, some (Typ.tensor ds))])
            body (Typ.tensor dsOut) ep
            ((outer ++ Γ_) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih
      exact HasType.tgrad Δ_ S_ (outer ++ Γ_) x ds dsOut body ep slot ih' hsub_eff
  | tvmap Δ_ S_ Γ_ x t1 t2 body ep d slot hbody ih =>
      have ih' :
          HasType Δ_ S_ ((outer ++ Γ_) ++ [(x, some t1)]) body t2 ep
            ((outer ++ Γ_) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih
      exact HasType.tvmap Δ_ S_ (outer ++ Γ_) x t1 t2 body ep d slot ih'
  | loc Δ_ S_ Γ_ ell tv hlook =>
      simpa [List.append_assoc] using (HasType.loc Δ_ S_ (outer ++ Γ_) ell tv hlook)
  | subEff Δ_ S_ Γ_ Γ'' e0 tv eps0 eps1 hbody hSub ih =>
      exact HasType.subEff Δ_ S_ (outer ++ Γ_) (outer ++ Γ'')
        e0 tv eps0 eps1 ih hSub
  | nil Δ_ S_ Γ2 t_ epsR_ =>
      simpa [List.append_assoc] using (ClausesTyped.nil Δ_ S_ (outer ++ Γ2) t_ epsR_)
  | cons Δ_ S_ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest slotX slotK
      hmatch h_body h_rest ih_body ih_rest =>
      have ih_body' :
          HasType Δ_ S_
            ((outer ++ Γ2) ++ [(x, some tArg), (k, some (Typ.arrow tRet t_ epsR_))])
            hb t_ epsR_
            ((outer ++ Γ3) ++ [(x, slotX), (k, slotK)]) := by
        simpa [List.append_assoc] using ih_body
      exact ClausesTyped.cons Δ_ S_ (outer ++ Γ2) (outer ++ Γ3)
        t_ tArg tRet epsR_ op x k hb rest slotX slotK
        hmatch ih_body' ih_rest

theorem clausesTyped_prefix_weaken
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {clauses : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR clauses)
    (outer : LinearCtx) :
    ClausesTyped Delta Sigma (outer ++ Gamma2) (outer ++ Gamma3) t epsR clauses := by
  induction clauses generalizing Gamma2 Gamma3 with
  | nil =>
      cases h
      simpa [List.append_assoc] using
        (ClausesTyped.nil Delta Sigma (outer ++ Gamma2) t epsR)
  | cons cl rest ih =>
      cases h with
      | cons _ _ _ _ _ tArg tRet _ op x k hb rest slotX slotK hmatch hBody hRest =>
          exact ClausesTyped.cons Delta Sigma (outer ++ Gamma2) (outer ++ Gamma3)
            t tArg tRet epsR op x k hb rest slotX slotK hmatch
            (by
              simpa [List.append_assoc] using
                hasType_prefix_weaken hBody outer)
            (ih hRest)

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
  | fst =>
      have h' : HasType [] Sigma [] (Term.fst e) t eps [] := by
        simpa [plug] using h
      obtain ⟨t2, h_e⟩ := HasType.fst_inv h'
      exact ⟨Typ.pair t t2, eps, h_e⟩
  | snd =>
      have h' : HasType [] Sigma [] (Term.snd e) t eps [] := by
        simpa [plug] using h
      obtain ⟨t1, h_e⟩ := HasType.snd_inv h'
      exact ⟨Typ.pair t1 t, eps, h_e⟩
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
  | fst =>
      have h' : HasType Delta Sigma Gamma (Term.fst e) t eps Gamma' := by simpa [plug] using h
      obtain ⟨t2, h_e⟩ := HasType.fst_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨S2, HasType.fst _ S2 _ _ _ _ t2 _ h_e', h_sub⟩
  | snd =>
      have h' := by simpa [plug] using h
      obtain ⟨t1, h_e⟩ := HasType.snd_inv h'
      obtain ⟨S2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨S2, HasType.snd _ S2 _ _ _ t1 _ _ h_e', h_sub⟩
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
  | fst =>
      have h' : HasType Delta Sigma Gamma (Term.fst e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨t2, h_e⟩ := HasType.fst_inv h'
      exact HasType.fst Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' t t2 eps (h_inner h_e)
  | snd =>
      have h' : HasType Delta Sigma Gamma (Term.snd e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨t1, h_e⟩ := HasType.snd_inv h'
      exact HasType.snd Delta Sigma
        (((outer ++ [(y, some ty)]) ++ Gamma))
        (((outer ++ [(y, none)]) ++ Gamma'))
        e' t1 t eps (h_inner h_e)
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
  | fst =>
      have h' : HasType [] Sigma [] (Term.fst e) t eps [] := by
        simpa [plug] using h
      obtain ⟨t2, h_e⟩ := HasType.fst_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨Sigma2, HasType.fst [] Sigma2 [] [] e' t t2 eps h_e', h_sub⟩
  | snd =>
      have h' : HasType [] Sigma [] (Term.snd e) t eps [] := by
        simpa [plug] using h
      obtain ⟨t1, h_e⟩ := HasType.snd_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_e
      exact ⟨Sigma2, HasType.snd [] Sigma2 [] [] e' t1 t eps h_e', h_sub⟩
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

/-- Preservation: if a configuration is well-typed and steps, the
    resulting configuration has the same type (under a possibly-extended
    store typing) and preserves store well-formedness.

    Current in-scope open cases are the agreed calculus-level blockers
    plus the generic `ctx` congruence case. `ctx` is blocked at theorem
    shape: `plug_preserves_typing_closed` rebuilds sibling subterms via
    `StoreTypSub Sigma Sigma2`, but inner preservation currently returns
    only a post-step `Sigma2` plus `StoreWf`, and store-consuming steps
    do not satisfy global monotonic extension. Closing `ctx` needs a
    weaker frame-local store-agreement invariant, not more local proof
    search in this theorem. -/
theorem preservation
    (sigma sigma' : Store) (Sigma : StoreTyp)
    (e e' : Term) (t : Typ) (eps : EffectRow)
    (h_typ : HasType [] Sigma [] e t eps [])
    (h_scope : WellScoped e)
    (h_wf : StoreWf sigma Sigma)
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma',
      HasType [] Sigma' [] e' t eps [] ∧ StoreWf sigma' Sigma' := by
  cases h_step with
  | beta s x tv body v hv =>
      refine ⟨Sigma, ?_, h_wf⟩
      exact preservation_beta_via_db h_typ hv h_scope
  | letBind s x v body hv =>
      refine ⟨Sigma, ?_, h_wf⟩
      exact preservation_letBind_via_db h_typ hv h_scope
  | letpair s x y v1 v2 body hv1 hv2 =>
      refine ⟨Sigma, ?_, h_wf⟩
      exact preservation_letpair_via_db h_typ hv1 hv2 h_scope
  | fst s v1 v2 hv1 hv2 =>
      obtain ⟨t2, h_pair⟩ := HasType.fst_inv h_typ
      obtain ⟨Γmid, eps1, eps2, h1, h2, hsub⟩ := HasType.pair_inv h_pair
      have hmid : Γmid = [] := has_type_closed_output_of_closed_input h1
      subst hmid
      refine ⟨Sigma, ?_, h_wf⟩
      have hsub1 : SubEffRow eps1 eps :=
        SubEffRow.trans (SubEffRow.union_left eps1 eps2) hsub
      exact HasType.subEff [] Sigma [] [] _ t eps1 eps h1 hsub1
  | snd s v1 v2 hv1 hv2 =>
      obtain ⟨t1, h_pair⟩ := HasType.snd_inv h_typ
      obtain ⟨Γmid, eps1, eps2, h1, h2, hsub⟩ := HasType.pair_inv h_pair
      have hmid : Γmid = [] := has_type_closed_output_of_closed_input h1
      subst hmid
      refine ⟨Sigma, ?_, h_wf⟩
      have hsub2 : SubEffRow eps2 eps := by
        intro op hop
        apply hsub
        by_cases hop1 : op ∈ eps1
        · simp [EffectRow.union, hop1]
        · simp [EffectRow.union, hop, hop1]
      exact HasType.subEff [] Sigma [] [] _ t eps2 eps h2 hsub2
  | tconst s v ds ell hell =>
      obtain ⟨ht, _hG⟩ := HasType.const_inv h_typ
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_, ?_⟩
      · subst ht
        have hlook := storeTypLookup_extend_self Sigma ell (Typ.tensor ds)
        have h_loc : HasType [] (storeTypExtend Sigma ell (Typ.tensor ds))
                              [] (Term.loc ell) (Typ.tensor ds) [] [] :=
          HasType.loc _ _ _ ell (Typ.tensor ds) hlook
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_loc hsub
      · exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf
  | copy s ell ellNew w hlook hfresh =>
      obtain ⟨ds, htEq, h_e⟩ := HasType.copy_inv h_typ
      obtain ⟨hlookT, _hGE⟩ := HasType.loc_inv h_e
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor ds), ?_, ?_⟩
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
  | tadd s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      obtain ⟨ds, Γmid, eps1, eps2, htEq, h_e1, h_e2⟩ := HasType.add_inv h_typ
      obtain ⟨hLook1, hG1⟩ := HasType.loc_inv h_e1
      obtain ⟨hLook2, hG2⟩ := HasType.loc_inv h_e2
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor ds), ?_, ?_⟩
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
  | tmul s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      obtain ⟨ds, Γmid, eps1, eps2, htEq, h_e1, h_e2⟩ := HasType.mul_inv h_typ
      obtain ⟨hLook1, _hG1⟩ := HasType.loc_inv h_e1
      obtain ⟨hLook2, _hG2⟩ := HasType.loc_inv h_e2
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor ds), ?_, ?_⟩
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
  | tsum s ell ellOut w d hlook hfresh =>
      obtain ⟨ds, htEq, _hmem, h_loc_e⟩ := HasType.sum_inv h_typ
      obtain ⟨hLook, _hGE⟩ := HasType.loc_inv h_loc_e
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (rem ds d)), ?_, ?_⟩
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
      · have h_isSome : (storeLookup sigma ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := rem w.shape d, data := w.data }
          (Typ.tensor (rem ds d)) h_wf hfresh hne
  | texpand s ell ellOut w d hlook hfresh =>
      obtain ⟨ds, htEq, h_loc_e⟩ := HasType.expand_inv h_typ
      obtain ⟨hLook, _hGE⟩ := HasType.loc_inv h_loc_e
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (ins ds d)), ?_, ?_⟩
      · subst htEq
        have hlookNew := storeTypLookup_extend_self
          (storeTypRemove Sigma ell) ellOut (Typ.tensor (ins ds d))
        have h_locOut : HasType [] _ [] (Term.loc ellOut)
                          (Typ.tensor (ins ds d)) [] [] :=
          HasType.loc _ _ _ ellOut (Typ.tensor (ins ds d)) hlookNew
        have hsub : SubEffRow [] eps := fun _ h => by cases h
        exact HasType.subEff _ _ _ _ _ _ [] eps h_locOut hsub
      · have h_isSome : (storeLookup sigma ell).isSome := by rw [hlook]; rfl
        have hne : ell ≠ ellOut := by
          rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
        exact StoreWf.remove_extend ell ellOut
          { shape := ins w.shape d, data := w.data }
          (Typ.tensor (ins ds d)) h_wf hfresh hne
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      obtain ⟨ds, eps0, htEq, h_loc_e, _hsub⟩ := HasType.uniformLike_inv h_typ
      obtain ⟨hLook, _hGE⟩ := HasType.loc_inv h_loc_e
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor ds), ?_, ?_⟩
      · subst htEq
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
      refine ⟨Sigma, ?_, h_wf⟩
      exact handleRet_value_preserves_typing hv h_typ
  | handleOpDirect s op v epsH clauses x k hb tRet hv hsig hmem =>
      refine ⟨Sigma, ?_, h_wf⟩
      exact preservation_handleOpDirect_via_db h_typ hv hsig hmem h_scope
  | handleOpCtx s op v epsH E clauses xVar kVar hb tRet hv hsig hmem hop hE =>
      refine ⟨Sigma, ?_, h_wf⟩
      rcases HasType.handle_inv_strong_bridge h_typ with
        ⟨GammaBody, epsB, hPlugPerform, hHsubB, hClIn, hClCov, hClauses, hSub⟩
      have hGammaBody : GammaBody = [] := has_type_closed_output_of_closed_input hPlugPerform
      subst hGammaBody
      rcases HasType.plug_inner_closed hPlugPerform with ⟨_tPerf, _epsPerf, hPerform⟩
      rcases HasType.perform_inv_bridge hPerform with
        ⟨tArgV, epsV, hV, hPerfSig, _hSubPerf⟩
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
        ⟨hxk, _hxNotHb, _hkNotHb, _hHbScope⟩
      let y := capturedContName (Term.handle epsH (plug E (Term.perform op v)) clauses)
      rcases capturedContName_fresh_selected_clause
          (epsH := epsH) (body := plug E (Term.perform op v))
          (clauses := clauses) (op := op) (x := xVar) (k := kVar) (hb := hb) hmem with
        ⟨_hyx, _hyk, _hyFreshHb⟩
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
      have hK :
          HasType [] Sigma [(xVar, some tArgV)]
            (Term.abs y tRet
              (Term.handle epsH (plug E (Term.var y)) clauses))
            (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) []
            [(xVar, some tArgV)] := by
        simpa [y] using
          (hasType_prefix_weaken hK0 [(xVar, some tArgV)])
      have hKClosed :
          Closed
            (Term.abs y tRet
              (Term.handle epsH (plug E (Term.var y)) clauses)) :=
        has_type_closed_term_of_closed_input hK0
      rcases subst_preserves_typing [] Sigma [(xVar, some tArgV)] [(xVar, slotX), (kVar, slotK)]
          kVar (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) t
          (EffectRow.removeOps epsB epsH) hb
          (Term.abs y tRet
            (Term.handle epsH (plug E (Term.var y)) clauses))
          hBody hK hKClosed with
        ⟨GammaAfterK, hAfterK⟩
      rcases subst_preserves_typing [] Sigma [] GammaAfterK
          xVar tArgV t (EffectRow.removeOps epsB epsH)
          (subst hb
            (Term.abs y tRet
              (Term.handle epsH (plug E (Term.var y)) clauses)) kVar)
          v
          hAfterK hVNil hClosedV with
        ⟨GammaFinal, hFinal⟩
      have hGammaFinal : GammaFinal = [] := has_type_closed_output_of_closed_input hFinal
      subst hGammaFinal
      have hswap :
          subst (subst hb v xVar)
              (Term.abs y tRet
                (Term.handle epsH (plug E (Term.var y)) clauses)) kVar =
            subst
              (subst hb
                (Term.abs y tRet
                  (Term.handle epsH (plug E (Term.var y)) clauses)) kVar)
              v xVar := by
        exact subst_commute_closed hb v
          (Term.abs y tRet
            (Term.handle epsH (plug E (Term.var y)) clauses))
          xVar kVar hxk hClosedV hKClosed
      simpa [y, hswap] using
        (HasType.subEff [] Sigma [] [] _ _ (EffectRow.removeOps epsB epsH) eps hFinal hSub)
  | handleOpCtxs s op v epsH Es clauses xVar kVar hb tRet hv hsig hmem hop hEs =>
      refine ⟨Sigma, ?_, h_wf⟩
      rcases HasType.handle_inv_strong_bridge h_typ with
        ⟨GammaBody, epsB, hPlugPerform, hHsubB, hClIn, hClCov, hClauses, hSub⟩
      have hGammaBody : GammaBody = [] := has_type_closed_output_of_closed_input hPlugPerform
      subst hGammaBody
      rcases HasType.multiPlug_inner_closed hPlugPerform with ⟨_tPerf, _epsPerf, hPerform⟩
      rcases HasType.perform_inv_bridge hPerform with
        ⟨tArgV, epsV, hV, hPerfSig, _hSubPerf⟩
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
        ⟨hxk, _hxNotHb, _hkNotHb, _hHbScope⟩
      let y := capturedContName (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses)
      rcases capturedContName_fresh_selected_clause
          (epsH := epsH) (body := multiPlug Es (Term.perform op v))
          (clauses := clauses) (op := op) (x := xVar) (k := kVar) (hb := hb) hmem with
        ⟨_hyx, _hyk, _hyFreshHb⟩
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
      have hK :
          HasType [] Sigma [(xVar, some tArgV)]
            (Term.abs y tRet
              (Term.handle epsH (multiPlug Es (Term.var y)) clauses))
            (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) []
            [(xVar, some tArgV)] := by
        simpa [y] using
          (hasType_prefix_weaken hK0 [(xVar, some tArgV)])
      have hKClosed :
          Closed
            (Term.abs y tRet
              (Term.handle epsH (multiPlug Es (Term.var y)) clauses)) :=
        has_type_closed_term_of_closed_input hK0
      rcases subst_preserves_typing [] Sigma [(xVar, some tArgV)] [(xVar, slotX), (kVar, slotK)]
          kVar (Typ.arrow tRet t (EffectRow.removeOps epsB epsH)) t
          (EffectRow.removeOps epsB epsH) hb
          (Term.abs y tRet
            (Term.handle epsH (multiPlug Es (Term.var y)) clauses))
          hBody hK hKClosed with
        ⟨GammaAfterK, hAfterK⟩
      rcases subst_preserves_typing [] Sigma [] GammaAfterK
          xVar tArgV t (EffectRow.removeOps epsB epsH)
          (subst hb
            (Term.abs y tRet
              (Term.handle epsH (multiPlug Es (Term.var y)) clauses)) kVar)
          v
          hAfterK hVNil hClosedV with
        ⟨GammaFinal, hFinal⟩
      have hGammaFinal : GammaFinal = [] := has_type_closed_output_of_closed_input hFinal
      subst hGammaFinal
      have hswap :
          subst (subst hb v xVar)
              (Term.abs y tRet
                (Term.handle epsH (multiPlug Es (Term.var y)) clauses)) kVar =
            subst
              (subst hb
                (Term.abs y tRet
                  (Term.handle epsH (multiPlug Es (Term.var y)) clauses)) kVar)
              v xVar := by
        exact subst_commute_closed hb v
          (Term.abs y tRet
            (Term.handle epsH (multiPlug Es (Term.var y)) clauses))
          xVar kVar hxk hClosedV hKClosed
      simpa [y, hswap] using
        (HasType.subEff [] Sigma [] [] _ _ (EffectRow.removeOps epsB epsH) eps hFinal hSub)
  | tgrad s x tv tOut body =>
      sorry
  | tvmap s x tv body d =>
      sorry
  | ctx _ _ E e0 e0' h_inner =>
      -- Blocked on a frame-local store-agreement theorem. The natural
      -- closed-context route wants `StoreTypSub Sigma Sigma2` so the
      -- untouched sibling sub-derivations can be weakened, but inner
      -- numeric/store steps remove consumed locations and therefore do
      -- not preserve global `StoreTypSub`. `preservation_ctx_counterexample`
      -- shows this is a real theorem-shape gap, not just missing local
      -- proof search. Closing `ctx` needs a theorem that tracks agreement
      -- only on the locations mentioned by the outer frame, or a stronger
      -- store/term invariant tying loc mentions to linear ownership.
      sorry
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
  | fst =>
      -- plug fst e = Term.fst e.
      have h' : HasType Delta Sigma Gamma (Term.fst e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨t2, h_pair⟩ := HasType.fst_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_pair
      refine ⟨Sigma2, ?_, h_sub⟩
      show HasType Delta Sigma2 Gamma (Term.fst e') t eps Gamma'
      exact HasType.fst Delta Sigma2 Gamma Gamma' e' t t2 eps h_e'
  | snd =>
      have h' : HasType Delta Sigma Gamma (Term.snd e) t eps Gamma' := by
        simpa [plug] using h
      obtain ⟨t1, h_pair⟩ := HasType.snd_inv h'
      obtain ⟨Sigma2, h_e', h_sub⟩ := h_inner h_pair
      refine ⟨Sigma2, ?_, h_sub⟩
      show HasType Delta Sigma2 Gamma (Term.snd e') t eps Gamma'
      exact HasType.snd Delta Sigma2 Gamma Gamma' e' t1 t eps h_e'
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
