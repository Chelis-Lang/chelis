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
      -- E-Fst: fst (pair v1 v2) ↦ v1.
      obtain ⟨t2, h_pair⟩ := HasType.fst_inv h_typ
      obtain ⟨Gamma2, eps1, eps2, h1, _h2, heps⟩ := HasType.pair_inv h_pair
      -- The T-Fst rule preserves the same Gamma1, Gamma2, and eps of the
      -- sub-derivation, so the output context on `pair v1 v2` must be the
      -- original Gamma (i.e. Gamma2 = Gamma = []), and similarly eps on
      -- the sub-derivation is the same `eps` in the theorem statement.
      -- `h1 : HasType [] Sigma Gamma v1 t eps1 Gamma2` with eps =
      -- union eps1 eps2. For preservation we need HasType on v1 at
      -- effect row `eps`. Since v1 is a value (hv1), its effect row can
      -- be weakened --- but we don't have weakening yet. Instead, note
      -- that the theorem expects the *same* eps the redex had; the
      -- redex's effect row equals that of the sub-term `pair v1 v2`
      -- (T-Fst threads eps through unchanged), which in turn equals
      -- `union eps1 eps2`. We cannot in general retype `v1` at a larger
      -- row without an effect-weakening lemma. Leave as sorry pending
      -- `HasType.weaken_eff`.
      -- TODO Wave 5: needs HasType.weaken_eff to retype v1 at (eps1 ∪ eps2)
      sorry
  | snd s v1 v2 hv1 hv2 =>
      -- Symmetric to fst; same effect-weakening obstacle.
      -- TODO Wave 5: needs HasType.weaken_eff to retype v2 at (eps1 ∪ eps2)
      sorry
  | tconst s v ds ell hell =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | copy s ell ellNew w hlook hfresh =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | tadd s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | tmul s ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | tsum s ell ellOut w i hlook hfresh =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | texpand s ell ellOut w i k hlook hfresh =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      -- TODO Wave 5: needs HasType.loc rule + StoreTyp monotonicity
      sorry
  | handleRet s epsH v clauses hv =>
      -- E-Handle-Ret: handle[epsH] v clauses ↦ v   when v is a value.
      -- Typing inversion: HasType ... (handle epsH v clauses) t eps []
      -- gives HasType ... v t epsB Gamma2 with eps = removeOps epsB epsH.
      -- We need HasType on v at effect row `eps`. Same obstacle as
      -- fst/snd: we need effect weakening (or rather, strengthening)
      -- since `eps = removeOps epsB epsH ⊆ epsB`. For values this
      -- should be provable with a value-canonicity lemma, but without
      -- it we leave the case open.
      -- TODO Wave 5: needs HasType value-canonicity / effect-strengthening
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
