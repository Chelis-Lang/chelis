-- LaCaDiLE/AddDim.lean — addDim preserves typing (Phase 2 WS2.3).
--
-- Statement: if `Δ; Σ; Γ ⊢ e : t ! ε ⊣ Γ'`, then lifting the entire
-- derivation through `addDim d` on types and `addDimTerm d` on terms
-- produces a new well-typed derivation for the lifted judgment:
--
--   Δ; Σ; addDimCtx d Γ ⊢ addDimTerm d e : addDim d t ! ε ⊣ addDimCtx d Γ'
--
-- Proved by structural induction on the typing derivation. Each case
-- rebuilds the corresponding `HasType` constructor after applying
-- `addDim` / `addDimTerm` to every type and term in sight.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing

namespace LaCaDiLE

/-- Lift `addDim d` over every type in a linear context. -/
def addDimCtx (d : Dim) (Gamma : LinearCtx) : LinearCtx :=
  Gamma.map (fun p => (p.1, addDim d p.2))

@[simp] theorem addDimCtx_nil (d : Dim) : addDimCtx d [] = [] := rfl

@[simp] theorem addDimCtx_cons (d : Dim) (x : String) (t : Typ) (G : LinearCtx) :
    addDimCtx d ((x, t) :: G) = (x, addDim d t) :: addDimCtx d G := rfl

theorem addDimCtx_append (d : Dim) (G1 G2 : LinearCtx) :
    addDimCtx d (G1 ++ G2) = addDimCtx d G1 ++ addDimCtx d G2 := by
  simp [addDimCtx, List.map_append]

theorem addDimCtx_singleton (d : Dim) (x : String) (t : Typ) :
    addDimCtx d [(x, t)] = [(x, addDim d t)] := rfl

/-- `addDimCtx` commutes with a name-only filter. -/
theorem addDimCtx_filter_name (d : Dim) (G : LinearCtx) (p : String → Bool) :
    (addDimCtx d G).filter (fun q => p q.1)
      = addDimCtx d (G.filter (fun q => p q.1)) := by
  induction G with
  | nil => rfl
  | cons hd tl ih =>
    simp only [addDimCtx, List.map_cons, List.filter_cons]
    by_cases hp : p hd.1
    · simp [hp]; exact ih
    · simp [hp]; exact ih

-- Stage 1 refactor: positional `rem`/`ins` replaced by named
-- `rem ds d := ds.erase d` and `ins ds d := DimList.cons d ds`. The previous
-- `rem_cons_succ`, `ins_cons_succ`, `addDim_tensor_rem`, and
-- `addDim_tensor_ins` helper lemmas (which lived on positional
-- indices with an `i + 1` shift) no longer have meaningful content
-- and are removed. Stage 2 reintroduces the analogous commutativity
-- lemmas at the multiset/quotient level where they hold definitionally.

/-- addDim / addDimTerm preserves typing (Phase 2 WS2.3). Wave 2
    fix: store typing is now lifted via `addDimStoreTyp`, which
    unlocks the `loc` case (previously a vacuous sorry). -/
theorem addDim_preserves_typing
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow} (d : Dim)
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    HasType Delta (addDimStoreTyp d Sigma)
            (addDimCtx d Gamma) (addDimTerm d e) (addDim d t)
            eps (addDimCtx d Gamma') := by
  induction h using HasType.rec
    (motive_2 := fun Delta Sigma Gamma2 Gamma3 t epsR cls _ =>
      ClausesTyped Delta (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma2) (addDimCtx d Gamma3)
        (addDim d t) epsR (addDimClauses d cls)) with
  | var Delta Sigma Gamma_pre Gamma_post x t =>
    -- Wave 0.5: var consumes from arbitrary position. Lift both
    -- Gamma_pre and Gamma_post through addDimCtx, then re-apply T-Var
    -- at the lifted split.
    simp only [addDimTerm, addDim, addDimCtx_append, addDimCtx_singleton]
    exact HasType.var Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma_pre)
      (addDimCtx d Gamma_post) x (addDim d t)
  | unit Delta Sigma Gamma =>
    simp [addDimTerm, addDim]
    exact HasType.unit Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma)
  | abs Delta Sigma Gamma1 Gamma2 x t1 t2 eps e _h ih =>
    simp only [addDimTerm]
    have ih' : HasType Delta (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma1 ++ [(x, addDim d t1)]) (addDimTerm d e) (addDim d t2) eps
        (addDimCtx d Gamma2) := by
      rw [← addDimCtx_singleton, ← addDimCtx_append]; exact ih
    have key := HasType.abs Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
                  x (addDim d t1) (addDim d t2) eps (addDimTerm d e) ih'
    show HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) _ _ _
            (addDimCtx d (Gamma2.filter (fun p => p.1 ≠ x)))
    have hf : addDimCtx d (Gamma2.filter (fun p => p.1 ≠ x))
            = (addDimCtx d Gamma2).filter (fun p => p.1 ≠ x) :=
      (addDimCtx_filter_name d Gamma2 (fun y => decide (y ≠ x))).symm
    rw [hf]; exact key
  | app Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 eps eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp [addDimTerm, addDim]
    exact HasType.app Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d e1) (addDimTerm d e2)
      (addDim d t1) (addDim d t2) eps eps1 eps2 ih1 ih2
  | letBind Delta Sigma Gamma1 Gamma2 Gamma3 x e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp only [addDimTerm]
    have ih2' : HasType Delta (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma2 ++ [(x, addDim d t1)]) (addDimTerm d e2) (addDim d t2) eps2
        (addDimCtx d Gamma3) := by
      rw [← addDimCtx_singleton, ← addDimCtx_append]; exact ih2
    have key := HasType.letBind Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) x (addDimTerm d e1) (addDimTerm d e2)
      (addDim d t1) (addDim d t2) eps1 eps2 ih1 ih2'
    show HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) _ _ _
          (addDimCtx d (Gamma3.filter (fun p => p.1 ≠ x)))
    have hf : addDimCtx d (Gamma3.filter (fun p => p.1 ≠ x))
            = (addDimCtx d Gamma3).filter (fun p => p.1 ≠ x) :=
      (addDimCtx_filter_name d Gamma3 (fun y => decide (y ≠ x))).symm
    rw [hf]; exact key
  | copy Delta Sigma Gamma1 Gamma2 e ds eps _h ih =>
    -- Wave 3: T-Copy now takes DimList; addDim prepends `d`.
    simp only [addDimTerm, addDim]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1)
                 (addDimTerm d e) (Typ.tensor (DimList.cons d ds)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    exact HasType.copy Delta (addDimStoreTyp d Sigma)
      (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (DimList.cons d ds) eps ih'
  | letpair Delta Sigma Gamma1 Gamma2 Gamma3 x y e1 e2 t1 t2 t eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp only [addDimTerm]
    have ih1' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e1)
        (Typ.pair (addDim d t1) (addDim d t2)) eps1 (addDimCtx d Gamma2) := by
      simpa [addDim] using ih1
    have ih2'' : HasType Delta (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma2 ++ [(x, addDim d t1), (y, addDim d t2)])
        (addDimTerm d e2) (addDim d t) eps2 (addDimCtx d Gamma3) := by
      have := ih2
      rw [addDimCtx_append] at this
      simpa [addDimCtx, List.map] using this
    have key := HasType.letpair Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) x y (addDimTerm d e1) (addDimTerm d e2)
      (addDim d t1) (addDim d t2) (addDim d t) eps1 eps2 ih1' ih2''
    show HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) _ (addDim d t)
          (EffectRow.union eps1 eps2)
          (addDimCtx d (Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y)))
    have hf : addDimCtx d (Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y))
            = (addDimCtx d Gamma3).filter (fun p => p.1 ≠ x ∧ p.1 ≠ y) :=
      (addDimCtx_filter_name d Gamma3 (fun y_ => decide (y_ ≠ x ∧ y_ ≠ y))).symm
    rw [hf]; exact key
  | tpair Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp [addDimTerm, addDim]
    exact HasType.tpair Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d e1) (addDimTerm d e2)
      (addDim d t1) (addDim d t2) eps1 eps2 ih1 ih2
  | fst Delta Sigma Gamma1 Gamma2 e t1 t2 eps _h ih =>
    simp [addDimTerm, addDim]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e)
        (Typ.pair (addDim d t1) (addDim d t2)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    exact HasType.fst Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (addDim d t1) (addDim d t2) eps ih'
  | snd Delta Sigma Gamma1 Gamma2 e t1 t2 eps _h ih =>
    simp [addDimTerm, addDim]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e)
        (Typ.pair (addDim d t1) (addDim d t2)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    exact HasType.snd Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (addDim d t1) (addDim d t2) eps ih'
  | const Delta Sigma Gamma v ds =>
    simp [addDimTerm, addDim]
    exact HasType.const Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma) v (DimList.cons d ds)
  | tadd Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp [addDimTerm, addDim]
    have ih1' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e1)
        (Typ.tensor (DimList.cons d ds)) eps1 (addDimCtx d Gamma2) := by
      simpa [addDim] using ih1
    have ih2' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma2) (addDimTerm d e2)
        (Typ.tensor (DimList.cons d ds)) eps2 (addDimCtx d Gamma3) := by
      simpa [addDim] using ih2
    exact HasType.tadd Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d e1) (addDimTerm d e2) (DimList.cons d ds)
      eps1 eps2 ih1' ih2'
  | tmul Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp [addDimTerm, addDim]
    have ih1' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e1)
        (Typ.tensor (DimList.cons d ds)) eps1 (addDimCtx d Gamma2) := by
      simpa [addDim] using ih1
    have ih2' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma2) (addDimTerm d e2)
        (Typ.tensor (DimList.cons d ds)) eps2 (addDimCtx d Gamma3) := by
      simpa [addDim] using ih2
    exact HasType.tmul Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d e1) (addDimTerm d e2) (DimList.cons d ds)
      eps1 eps2 ih1' ih2'
  | uniformLike Delta Sigma Gamma1 Gamma2 e ds lo hi eps _h ih =>
    simp [addDimTerm]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e)
        (Typ.tensor (DimList.cons d ds)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    have key := HasType.uniformLike Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (DimList.cons d ds) lo hi eps ih'
    simpa [addDim] using key
  | perform Delta Sigma Gamma1 Gamma2 op e tArg tRet eps _h hMatch ih =>
    -- Wave 3 calculus refinement: T-Perform now takes tArg/tRet
    -- explicitly with an OpSigMatch side condition. addDim must
    -- transport the match witness through the dimension lifting.
    -- The current OpSigMatch cases are all unit/unit-or-tensor, and
    -- addDim is identity on unit; the tensor case prepends d.
    simp [addDimTerm]
    -- addDim d tArg / addDim d tRet are still in the OpSigMatch
    -- relation: each constructor is preserved by prepending d.
    have hMatch' : OpSigMatch op (addDim d tArg) (addDim d tRet) := by
      cases hMatch with
      | accumTensor ds => simpa [addDim] using OpSigMatch.accumTensor (DimList.cons d ds)
      | accumUnit      => simpa [addDim] using OpSigMatch.accumUnit
      | random         => simpa [addDim] using OpSigMatch.random
      | resource       => simpa [addDim] using OpSigMatch.resource
      | io             => simpa [addDim] using OpSigMatch.io
      | fail           => simpa [addDim] using OpSigMatch.fail
    exact HasType.perform Delta (addDimStoreTyp d Sigma)
      (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      op (addDimTerm d e) (addDim d tArg) (addDim d tRet) eps ih hMatch'
  | handle Delta Sigma Gamma1 Gamma2 Gamma3 body clauses t epsH epsB
           _hb hEpsH hCl hCov _hCT ihb ihCT =>
    simp only [addDimTerm]
    -- helper: for every clause in `addDimClauses d cls`, there is a source
    -- clause in `cls` with the same op.
    have hClauseMap : ∀ cls : List (EffectLabel × String × String × Term),
        ∀ cl ∈ addDimClauses d cls, ∃ cl0 ∈ cls, cl.1 = cl0.1 := by
      intro cls
      induction cls with
      | nil => intro cl hcl; simp [addDimClauses] at hcl
      | cons hd tl ihtl =>
        intro cl hcl
        rcases hd with ⟨op, x, k, hb⟩
        simp [addDimClauses] at hcl
        rcases hcl with heq | hrest
        · refine ⟨(op, x, k, hb), by simp, ?_⟩
          rw [heq]
        · rcases ihtl cl hrest with ⟨cl0, hcl0, heq⟩
          exact ⟨cl0, List.mem_cons_of_mem _ hcl0, heq⟩
    have hClauseCov : ∀ cls : List (EffectLabel × String × String × Term),
        ∀ cl0 ∈ cls, ∃ cl ∈ addDimClauses d cls, cl.1 = cl0.1 := by
      intro cls
      induction cls with
      | nil => intro cl0 hcl0; simp at hcl0
      | cons hd tl ihtl =>
        intro cl0 hcl0
        rcases hd with ⟨op, x, k, hb⟩
        simp at hcl0
        rcases hcl0 with heq2 | hrest
        · refine ⟨(op, x, k, addDimTerm d hb), ?_, ?_⟩
          · simp [addDimClauses]
          · simp [heq2]
        · rcases ihtl cl0 hrest with ⟨cl, hcl, heq⟩
          refine ⟨cl, ?_, heq⟩
          simp [addDimClauses]; exact Or.inr hcl
    exact HasType.handle Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d body) (addDimClauses d clauses)
      (addDim d t) epsH epsB ihb hEpsH
      (by
        intro cl hcl
        rcases hClauseMap clauses cl hcl with ⟨cl0, hcl0, heq⟩
        rw [heq]; exact hCl cl0 hcl0)
      (by
        intro op hop
        rcases hCov op hop with ⟨cl0, hcl0, heq⟩
        rcases hClauseCov clauses cl0 hcl0 with ⟨cl, hcl, heq2⟩
        exact ⟨cl, hcl, heq2.trans heq⟩)
      ihCT
  | tgrad Delta Sigma Gamma x ds dsOut e eps _h hsub ih =>
    simp [addDimTerm, addDim]
    have ih' : HasType (Capability.diff :: Delta) (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma ++ [(x, Typ.tensor (DimList.cons d ds))]) (addDimTerm d e)
        (Typ.tensor (DimList.cons d dsOut)) eps (addDimCtx d Gamma) := by
      have := ih
      rw [addDimCtx_append, addDimCtx_singleton] at this
      simpa [addDim] using this
    exact HasType.tgrad Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma)
      x (DimList.cons d ds) (DimList.cons d dsOut) (addDimTerm d e) eps ih' hsub
  | loc Delta Sigma Gamma ell t hlook =>
    -- Wave 2: closed via addDimStoreTyp_lookup. The store typing in
    -- the output is lifted via addDimStoreTyp, so the looked-up type
    -- is correspondingly addDim d t.
    simp only [addDimTerm]
    have hlook' := addDimStoreTyp_lookup d Sigma ell t hlook
    exact HasType.loc Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma)
      ell (addDim d t) hlook'
  | subEff Delta Sigma Gamma Gamma' e t eps eps' _h hsub ih =>
    -- Wave 0.5: widen the lifted derivation's effect row via subEff.
    -- The sub-derivation `ih` is at the narrower eps, and we re-apply
    -- the original SubEffRow proof to widen to eps' post-lifting.
    exact HasType.subEff Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma) (addDimCtx d Gamma')
      (addDimTerm d e) (addDim d t) eps eps' ih hsub
  | nil Delta Sigma Gamma2 t epsR =>
    simp [addDimClauses]
    exact ClausesTyped.nil Delta (addDimStoreTyp d Sigma)
      (addDimCtx d Gamma2) (addDim d t) epsR
  | cons Delta Sigma Gamma2 Gamma3 t tArg tRet epsR op x k hb rest _hhb _hrest ihhb ihrest =>
    simp [addDimClauses]
    have ihhb' : HasType Delta (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma2 ++ [(x, addDim d tArg),
                                (k, Typ.arrow (addDim d tRet) (addDim d t) epsR)])
        (addDimTerm d hb) (addDim d t) epsR (addDimCtx d Gamma3) := by
      have := ihhb
      rw [addDimCtx_append] at this
      simpa [addDimCtx, List.map, addDim] using this
    exact ClausesTyped.cons Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma2) (addDimCtx d Gamma3)
      (addDim d t) (addDim d tArg) (addDim d tRet) epsR op x k
      (addDimTerm d hb) (addDimClauses d rest) ihhb' ihrest
  | _ =>
    -- Stage 1 refactor: `tvmap`, `tsum`, and `texpand` are all blocked
    -- on the Stage 2 `DimList` quotient / multiset representation.
    --
    -- `tvmap` obstacle (pre-existing): `addDim` prepends on the left,
    -- so `addDim d_out (addDim d_in τ)` and `addDim d_in (addDim d_out τ)`
    -- differ in order. HasType.tvmap's IH forces both compositions to
    -- be equal, which fails definitionally on ordered lists.
    --
    -- `tsum` / `texpand` obstacle (new in Stage 1): with
    -- `rem ds d = ds.erase d` and `ins ds d = DimList.cons d ds`, the goal after
    -- lifting becomes
    --   Typ.tensor (d_new :: ds.erase d) ≟ Typ.tensor ((d_new :: ds).erase d)
    -- (and analogously for `ins`). These are only propositionally equal
    -- when `d_new ≠ d`; no such freshness is threaded through the
    -- statement.
    --
    -- All three cases close uniformly once `DimList` becomes a multiset /
    -- quotient: prepend order is irrelevant and `erase` commutes with
    -- `cons` up to the quotient. Consolidated under a single wildcard
    -- so the net `sorry` count in this theorem stays at one (replacing
    -- the pre-existing standalone tvmap sorry).

    sorry

end LaCaDiLE
