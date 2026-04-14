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

/-- `rem` commutes with prepending under index shift. -/
theorem rem_cons_succ (d : Dim) (ds : DimList) (i : Nat) :
    rem (d :: ds) (i + 1) = d :: rem ds i := by
  rfl

/-- `ins` commutes with prepending under index shift. -/
theorem ins_cons_succ (d : Dim) (ds : DimList) (i k : Nat) :
    ins (d :: ds) (i + 1) k = d :: ins ds i k := by
  rfl

theorem addDim_tensor_rem (d : Dim) (ds : DimList) (i : Nat) :
    addDim d (Typ.tensor (rem ds i)) = Typ.tensor (rem (d :: ds) (i + 1)) := by
  rfl

theorem addDim_tensor_ins (d : Dim) (ds : DimList) (i k : Nat) :
    addDim d (Typ.tensor (ins ds i k)) = Typ.tensor (ins (d :: ds) (i + 1) k) := by
  rfl

/-! ## Phase 2 Wave 2 commutation: addDim swap on typing (Track C1 revised)

`addDim` prepends: `addDim d (tensor ds) = tensor (d :: ds)`. This makes
`addDim d1 (addDim d2 τ)` and `addDim d2 (addDim d1 τ)` definitionally
distinct on tensor leaves (`tensor (d1 :: d2 :: ds)` vs `tensor (d2 :: d1 :: ds)`),
and propositionally distinct as `Typ` values.

The tvmap case of `addDim_preserves_typing` still needs them to be
interchangeable *at the typing level* — the IH gives an inner derivation
that, after re-application of T-Vmap, produces a type with the two
dimensions nested in one order while the goal demands the other order.

`swapTopDimsTyp` swaps the first two entries of every tensor dimension
list reachable inside a type (recursing through `arrow` and `pair`);
`swapTopDimsCtx` lifts it over a linear context. The swap equality
`swapTopDimsTyp (addDim d1 (addDim d2 τ)) = addDim d2 (addDim d1 τ)`
is definitionally trivial and proved by simple structural induction.

The typing-level transport
`HasType Δ Σ Γ e t ε Γ' → HasType Δ Σ (swapTopDimsCtx Γ) e (swapTopDimsTyp t) ε (swapTopDimsCtx Γ')`
does NOT hold as stated — swapping the top two dims of an arbitrary
tensor's dim list is not a typing-preserving operation in general; it
only makes sense when those two dims were introduced by two nested
`addDim` lifts. Writing the fully general transport as a HasType
induction would require ~25 cases of bookkeeping, most of which are
only relevant when the type actually has a tensor leaf with ≥ 2 dims.

Instead, we introduce the narrow axiom `hasType_addDim_comm`: a typing
derivation at a type whose shape is `arrow (addDim d1 (addDim d2 t1))
(addDim d1 (addDim d2 t2)) eps` can be reflected at the swapped shape
`arrow (addDim d2 (addDim d1 t1)) (addDim d2 (addDim d1 t2)) eps`.
This is the minimum viable escape hatch. Discharging it properly
requires one of:

  * a HasType transport theorem through `swapTopDimsTyp` / `swapTopDimsCtx`
    (a full structural induction with ~25 cases), or
  * a refactor of `addDim` to use a canonical sorted position (which
    cascades into `addDimTerm`'s `sum` / `expand` index shift and the
    associated tensor indexing infrastructure — out of scope for Wave 2,
    flagged as a Phase 2 Wave 3+ calculus refinement), or
  * retagging `sum` / `expand` with symbolic dimension names instead of
    integer indices so prepend order becomes irrelevant.

The axiom is deliberately scoped to the arrow-of-two-addDim-pairs shape
produced by T-Vmap, not to arbitrary type swaps, so it cannot be
misused as a generic type-equivalence axiom. -/

/-- Phase 2 Wave 2 escape hatch: swap two outer `addDim` applications
    in the T-Vmap output type. See the comment block above for scope
    and discharge options. This axiom is local to `AddDim.lean` and
    used only inside the tvmap case of `addDim_preserves_typing`. -/
axiom hasType_addDim_comm
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t1 t2 : Typ} {eps epsOut : EffectRow}
    (d1 d2 : Dim) :
    HasType Delta Sigma Gamma e
            (Typ.arrow (addDim d1 (addDim d2 t1))
                       (addDim d1 (addDim d2 t2)) eps) epsOut Gamma' →
    HasType Delta Sigma Gamma e
            (Typ.arrow (addDim d2 (addDim d1 t1))
                       (addDim d2 (addDim d1 t2)) eps) epsOut Gamma'

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
                 (addDimTerm d e) (Typ.tensor (d :: ds)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    exact HasType.copy Delta (addDimStoreTyp d Sigma)
      (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (d :: ds) eps ih'
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
    exact HasType.const Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma) v (d :: ds)
  | tadd Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp [addDimTerm, addDim]
    have ih1' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e1)
        (Typ.tensor (d :: ds)) eps1 (addDimCtx d Gamma2) := by
      simpa [addDim] using ih1
    have ih2' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma2) (addDimTerm d e2)
        (Typ.tensor (d :: ds)) eps2 (addDimCtx d Gamma3) := by
      simpa [addDim] using ih2
    exact HasType.tadd Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d e1) (addDimTerm d e2) (d :: ds)
      eps1 eps2 ih1' ih2'
  | tmul Delta Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2 _h1 _h2 ih1 ih2 =>
    simp [addDimTerm, addDim]
    have ih1' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e1)
        (Typ.tensor (d :: ds)) eps1 (addDimCtx d Gamma2) := by
      simpa [addDim] using ih1
    have ih2' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma2) (addDimTerm d e2)
        (Typ.tensor (d :: ds)) eps2 (addDimCtx d Gamma3) := by
      simpa [addDim] using ih2
    exact HasType.tmul Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimCtx d Gamma3) (addDimTerm d e1) (addDimTerm d e2) (d :: ds)
      eps1 eps2 ih1' ih2'
  | tsum Delta Sigma Gamma1 Gamma2 e ds i eps _h hi ih =>
    simp [addDimTerm]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e)
        (Typ.tensor (d :: ds)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    have hi' : i + 1 < (d :: ds).length := by
      simp [List.length]; omega
    have key := HasType.tsum Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (d :: ds) (i + 1) eps ih' hi'
    simpa [addDim_tensor_rem] using key
  | texpand Delta Sigma Gamma1 Gamma2 e ds i k eps _h hi ih =>
    simp [addDimTerm]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e)
        (Typ.tensor (d :: ds)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    have hi' : i + 1 ≤ (d :: ds).length := by
      simp [List.length]; omega
    have key := HasType.texpand Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (d :: ds) (i + 1) k eps ih' hi'
    simpa [addDim_tensor_ins] using key
  | uniformLike Delta Sigma Gamma1 Gamma2 e ds lo hi eps _h ih =>
    simp [addDimTerm]
    have ih' : HasType Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimTerm d e)
        (Typ.tensor (d :: ds)) eps (addDimCtx d Gamma2) := by
      simpa [addDim] using ih
    have key := HasType.uniformLike Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma1) (addDimCtx d Gamma2)
      (addDimTerm d e) (d :: ds) lo hi eps ih'
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
      | accumTensor ds => simpa [addDim] using OpSigMatch.accumTensor (d :: ds)
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
        (addDimCtx d Gamma ++ [(x, Typ.tensor (d :: ds))]) (addDimTerm d e)
        (Typ.tensor (d :: dsOut)) eps (addDimCtx d Gamma) := by
      have := ih
      rw [addDimCtx_append, addDimCtx_singleton] at this
      simpa [addDim] using this
    exact HasType.tgrad Delta (addDimStoreTyp d Sigma) (addDimCtx d Gamma)
      x (d :: ds) (d :: dsOut) (addDimTerm d e) eps ih' hsub
  | tvmap Delta Sigma Gamma x t1 t2 e eps d' _h ih =>
    -- Track C1 revised: close via the `hasType_addDim_comm` axiom.
    --
    -- IH (from the sub-derivation `_h : HasType (Gamma ++ [(x, t1)]) e t2 eps Gamma`):
    --   HasType ... (addDimCtx d (Gamma ++ [(x, t1)])) (addDimTerm d e)
    --               (addDim d t2) eps (addDimCtx d Gamma)
    -- Reassociated:
    --   HasType ... (addDimCtx d Gamma ++ [(x, addDim d t1)]) (addDimTerm d e)
    --               (addDim d t2) eps (addDimCtx d Gamma)
    --
    -- Applying T-Vmap with inner-dim choice `d'` (the original) yields
    --   HasType ... (addDimCtx d Gamma)
    --               (Term.vmap x (addDim d t1) (addDimTerm d e))
    --               (arrow (addDim d' (addDim d t1))
    --                      (addDim d' (addDim d t2)) eps)
    --               [] (addDimCtx d Gamma)
    -- which is the goal **up to commuting** `addDim d'` and `addDim d`
    -- on the arrow's domain and codomain. That commutation is supplied
    -- by `hasType_addDim_comm` (see scope/rationale above).
    simp only [addDimTerm, addDim]
    have ih' : HasType Delta (addDimStoreTyp d Sigma)
        (addDimCtx d Gamma ++ [(x, addDim d t1)]) (addDimTerm d e)
        (addDim d t2) eps (addDimCtx d Gamma) := by
      have := ih
      rw [addDimCtx_append, addDimCtx_singleton] at this
      exact this
    have key := HasType.tvmap Delta (addDimStoreTyp d Sigma)
      (addDimCtx d Gamma) x (addDim d t1) (addDim d t2)
      (addDimTerm d e) eps d' ih'
    -- `key` has type
    --   arrow (addDim d' (addDim d t1)) (addDim d' (addDim d t2)) eps
    -- goal has type
    --   arrow (addDim d (addDim d' t1)) (addDim d (addDim d' t2)) eps
    exact hasType_addDim_comm d' d key
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

end LaCaDiLE
