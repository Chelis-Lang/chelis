-- LaCaDiLE/AdjointTyping.lean — adjoint typing lemma (Phase 2 proof).
--
-- WS2.2 target: the adjoint transformation preserves typing. Per the
-- E-Grad reduction (figures/opsem.tex), `grad(λx:τ.e)` reduces to
--    λx:τ. λgs:τ_out. handle[{Accum}] (adjoint e x gs) with h_accum
-- so the *body of the handle* is `adjoint e x gs`. The handle's clause
-- assembles the parameter gradient and returns it as tensor[ds]; the
-- adjoint body itself is a sequence of `perform accum (...)` calls
-- whose head term has type `unit` and effect row `{accum}` (plus the
-- forwarded effects of the original body, which must lie in
-- DiffCompat).
--
-- Wave 3 calculus refinement originally expected an `OpSigMatch`
-- witness that lets `perform accum` take a tensor argument. The
-- current global `OpSigMatch` in `Syntax.lean` instead fixes `accum`
-- at `unit -> unit`, so the adjoint leaf skeleton now consumes the
-- tensor seed via a trivial `letBind` and emits `perform accum unit`.
-- This keeps the linear-context threading honest while the richer
-- tensor-carrying accum surface remains future work.
--
-- Wave 4 Track B: the helper `adjoint_typed_aux` is counter-threaded
-- (via `adjointFrom` rather than `adjoint`) and carries a freshness
-- premise `AdjointNamesFresh n Γ_s Γ_s'` stating that every
-- counter-indexed fresh name the adjoint transform could mint at
-- counter ≥ n is disjoint from both linear contexts. The freshness
-- discipline lets the `add` and `mul` cases close by building typed
-- `letpair` / `letBind` / `copy` derivations over sub-calls at the
-- incremented counter.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.AdjointTransform
import LaCaDiLE.StringHelpers
import LaCaDiLE.Operational
import LaCaDiLE.TranslationDB

namespace LaCaDiLE

/-- The nine base names the adjoint transform mints. Used to state the
    freshness premise of `adjoint_typed_aux` uniformly across cases. -/
def adjointBases : List String :=
  ["gA", "gB", "a", "aTape", "b", "bTape", "y", "adjA", "adjHb"]

/-- Every base name used by `adjointFrom` is hash-free, so
    `freshName_ne_of_base_ne` applies whenever two bases differ. -/
theorem adjointBases_noHash :
    ∀ b ∈ adjointBases, NoHash b := by
  intro b hb
  simp [adjointBases] at hb
  rcases hb with h | h | h | h | h | h | h | h | h <;>
    (subst h; show '#' ∉ _; decide)

/-- Freshness predicate: every counter-indexed fresh name
    `freshName base m` for `m ≥ n` and `base ∈ adjointBases` is
    disjoint from `Γ`. Monotone in `n`. -/
def AdjointNamesFresh (n : Nat) (Γ : LinearCtx) : Prop :=
  ∀ m, m ≥ n → ∀ base, base ∈ adjointBases →
    freshName base m ∉ linearCtxDom Γ

/-- Freshness is monotone in the counter: if all names at counters ≥ n
    are fresh, so are all names at counters ≥ n' for any n' ≥ n. -/
theorem AdjointNamesFresh.mono {n n' : Nat} {Γ : LinearCtx}
    (h : AdjointNamesFresh n Γ) (hle : n ≤ n') : AdjointNamesFresh n' Γ :=
  fun m hm base hb => h m (Nat.le_trans hle hm) base hb

/-- Adding a binder whose name is a `freshName` at a *strictly smaller*
    counter preserves freshness at the current counter. Used to extend
    `Γ_s'` with `(freshName b k, t)` when recursing on an inner
    sub-term at counter ≥ k+1. -/
theorem AdjointNamesFresh.cons_freshName
    {n k : Nat} {Γ : LinearCtx} {b : String} {t : Option Typ}
    (h : AdjointNamesFresh n Γ)
    (hb_adj : b ∈ adjointBases) (hk : k < n) :
    AdjointNamesFresh n (Γ ++ [(freshName b k, t)]) := by
  intro m hm base hbase
  have hne_m_k : m ≠ k := by
    intro heq
    exact (Nat.not_lt.mpr (heq ▸ hm) hk)
  have hne_name : freshName base m ≠ freshName b k := by
    by_cases hbb : base = b
    · subst hbb
      exact freshName_ne_of_nat_ne base m k hne_m_k
    · exact freshName_ne_of_base_ne base b m k
        (adjointBases_noHash base hbase)
        (adjointBases_noHash b hb_adj) hbb
  intro hmem
  -- hmem : freshName base m ∈ linearCtxDom (Γ ++ [(freshName b k, t)])
  simp only [linearCtxDom, List.map_append, List.map_cons, List.map_nil,
             List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hmem
  rcases hmem with hIn | hIn
  · exact h m hm base hbase (by simpa [linearCtxDom] using hIn)
  · exact hne_name hIn

-- linearCtx_filter_fresh_eq and linearCtx_filter_fresh_two_eq
-- deleted: obsolete under tombstone-style contexts (no more .filter
-- on output contexts).

/-- `SubEffRow` from `union eps [accum, accum]`-shaped rows back to
    `union [accum] eps`. Elementwise membership. -/
private theorem subEff_accum_accum_to_accum (epsSeed : EffectRow) :
    SubEffRow
      (EffectRow.union epsSeed
        (EffectRow.union
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
      (EffectRow.union [EffectLabel.accum] epsSeed) := by
  -- Both sides are sets equal to `{accum} ∪ epsSeed`. Reduce to a
  -- direct case split on whether `op = accum` or `op ∈ epsSeed`.
  intro op hop
  have hcases : op = EffectLabel.accum ∨ op ∈ epsSeed := by
    simp [EffectRow.union, List.mem_append, List.mem_filter] at hop
    rcases hop with hL | hR
    · exact Or.inr hL
    · exact Or.inl hR.1
  -- Goal: op ∈ union [accum] epsSeed = [accum] ++ epsSeed.filter ...
  show op ∈ EffectRow.union [EffectLabel.accum] epsSeed
  -- It suffices to show `op ∈ [accum] ∨ op ∈ epsSeed`.
  have : op ∈ [EffectLabel.accum] ∨ op ∈ epsSeed := by
    rcases hcases with h | h
    · exact Or.inl (by subst h; simp)
    · exact Or.inr h
  unfold EffectRow.union
  rcases this with hL | hR
  · exact List.mem_append_left _ hL
  · by_cases hmem : op ∈ ([EffectLabel.accum] : EffectRow)
    · exact List.mem_append_left _ hmem
    · apply List.mem_append_right
      refine List.mem_filter.mpr ⟨hR, ?_⟩
      have hop_ne : op ≠ EffectLabel.accum := by
        intro heq
        exact hmem (by subst heq; exact List.mem_singleton.mpr rfl)
      cases op <;> first | rfl | exact absurd rfl hop_ne

/-- SubEffRow for the outer `copy gSeed` union with the inner add body. -/
private theorem subEff_letpair_add (epsSeed : EffectRow) :
    SubEffRow
      (EffectRow.union epsSeed
        (EffectRow.union
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
      (EffectRow.union [EffectLabel.accum] epsSeed) :=
  subEff_accum_accum_to_accum epsSeed

/-- SubEffRow for the leaf skeleton
    `let _ = gSeed in perform accum unit`. -/
private theorem subEff_seed_accum (epsSeed : EffectRow) :
    SubEffRow
      (EffectRow.union epsSeed
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
      (EffectRow.union [EffectLabel.accum] epsSeed) := by
  intro op hop
  unfold EffectRow.union at hop ⊢
  simp only [List.mem_append, List.mem_filter, List.mem_singleton] at hop ⊢
  rcases hop with hop | hop
  · by_cases hEq : op = EffectLabel.accum
    · left
      exact hEq
    · right
      exact ⟨hop, by simp [List.mem_singleton, hEq]⟩
  · left
    simpa using hop.1

/-- `adjoint_typed_aux` instantiated with a pure seed variable emits
    at least `accum`, so the row can always be widened to the outer
    grad target row `eps ∪ {accum}`. -/
private theorem subEff_accum_into_grad (eps : EffectRow) :
    SubEffRow
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (EffectRow.union eps [EffectLabel.accum]) := by
  intro op hop
  have hopAccum : op = EffectLabel.accum := by
    simp [EffectRow.union] at hop
    simpa using hop
  subst hopAccum
  unfold EffectRow.union
  by_cases hmem : EffectLabel.accum ∈ eps
  · exact List.mem_append_left _ hmem
  · apply List.mem_append_right
    simp [hmem]

/-- Generic typing lemma for the unit-valued `adjointLeaf` skeleton. It
    linearly consumes an arbitrary supported cotangent seed, then emits
    `perform accum unit`, so the overall result is `unit` with exactly
    the incoming seed effects plus `accum`. -/
private theorem adjointLeaf_typed
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma_s Gamma_s' : LinearCtx)
    (seedTy : Typ) (epsSeed : EffectRow) (n : Nat) (gSeed : Term)
    (h_seed : HasType Delta Sigma Gamma_s gSeed seedTy epsSeed Gamma_s') :
    HasType Delta Sigma Gamma_s
      (adjointLeaf gSeed n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] epsSeed)
      Gamma_s' := by
  have hAccumSig : OpSigMatch EffectLabel.accum Typ.unit Typ.unit := by
    simp [OpSigMatch, opArgType, opRetType]
  have hPerform :
      HasType Delta Sigma
        (Gamma_s' ++ [(freshName "adjA" n, some seedTy)])
        (Term.perform EffectLabel.accum Term.unit)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma_s' ++ [(freshName "adjA" n, some seedTy)]) := by
    exact HasType.perform Delta Sigma
      (Gamma_s' ++ [(freshName "adjA" n, some seedTy)])
      (Gamma_s' ++ [(freshName "adjA" n, some seedTy)])
      EffectLabel.accum Term.unit Typ.unit Typ.unit []
      (HasType.unit Delta Sigma _)
      hAccumSig
  have hLet :
      HasType Delta Sigma Gamma_s
        (adjointLeaf gSeed n)
        Typ.unit
        (EffectRow.union epsSeed
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
        Gamma_s' := by
    simpa [adjointLeaf] using
      (HasType.letBind Delta Sigma Gamma_s Gamma_s' Gamma_s'
        (freshName "adjA" n) gSeed
        (Term.perform EffectLabel.accum Term.unit)
        seedTy Typ.unit epsSeed
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (some seedTy) h_seed hPerform)
  exact HasType.subEff Delta Sigma Gamma_s Gamma_s'
    (adjointLeaf gSeed n) Typ.unit
    (EffectRow.union epsSeed
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
    (EffectRow.union [EffectLabel.accum] epsSeed)
    hLet
    (subEff_seed_accum epsSeed)

/-- Zero cotangent seeds are well-typed in any linear context at the
    structural cotangent type chosen by `AdjointTransform`. -/
private theorem zeroCotangent_typed
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx) :
    ∀ t, HasType Delta Sigma Gamma (zeroCotangent t) (cotangentType t) [] Gamma
  | Typ.tensor ds =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.const Delta Sigma Gamma 0 ds)
  | Typ.pair t1 t2 =>
      by
        have h1 := zeroCotangent_typed Delta Sigma Gamma t1
        have h2 := zeroCotangent_typed Delta Sigma Gamma t2
        simpa [zeroCotangent, cotangentType] using
          (HasType.tpair Delta Sigma Gamma Gamma Gamma
            (zeroCotangent t1) (zeroCotangent t2)
            (cotangentType t1) (cotangentType t2) [] [] h1 h2)
  | Typ.unit =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.unit Delta Sigma Gamma)
  | Typ.arrow _ _ _ =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.unit Delta Sigma Gamma)
  | Typ.tyVar _ =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.unit Delta Sigma Gamma)

/-- CPS typing lemma for `splitCotangentSeedFrom` on pure seeds. This
    is the internal bridge needed by the typed clause transform:
    callers supply the typing proof for the continuation at the split
    point, and the theorem reconstructs the surrounding seed split. -/
private theorem splitCotangentSeedFrom_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    (Gamma_s Gamma_s' : LinearCtx)
    (t : Typ) (gSeed : Term) (n : Nat)
    (k : Nat → Term → Term → Term)
    (epsK : EffectRow) (Gamma_out : LinearCtx)
    (h_seed : HasType Delta Sigma Gamma_s gSeed (cotangentType t) [] Gamma_s')
    (h_k :
      match t with
      | Typ.tensor ds =>
          ∃ slotA slotB,
            HasType Delta Sigma
              (Gamma_s' ++
                [(freshName "gA" n, some (Typ.tensor ds)),
                 (freshName "gB" n, some (Typ.tensor ds))])
              (k (n + 2)
                (Term.var (freshName "gA" n))
                (Term.var (freshName "gB" n)))
              Typ.unit
              epsK
              (Gamma_out ++
                [(freshName "gA" n, slotA),
                 (freshName "gB" n, slotB)])
      | Typ.pair t1 t2 =>
          ∃ slotA slotB,
            HasType Delta Sigma
              (Gamma_s' ++
                [(freshName "gA" n, some (cotangentType t1)),
                 (freshName "gB" n, some (cotangentType t2))])
              (splitCotangentSeedFrom t1 (Term.var (freshName "gA" n)) (n + 2)
                (fun n' gA1 gA2 =>
                  splitCotangentSeedFrom t2 (Term.var (freshName "gB" n)) n'
                    (fun n'' gB1 gB2 =>
                      k n''
                        (Term.pair gA1 gB1)
                        (Term.pair gA2 gB2))))
              Typ.unit
              epsK
              (Gamma_out ++
                [(freshName "gA" n, slotA),
                 (freshName "gB" n, slotB)])
      | Typ.unit =>
          HasType Delta Sigma Gamma_s
            (k n Term.unit Term.unit) Typ.unit epsK Gamma_out
      | Typ.arrow _ _ _ =>
          HasType Delta Sigma Gamma_s
            (k n Term.unit Term.unit) Typ.unit epsK Gamma_out
      | Typ.tyVar _ =>
          HasType Delta Sigma Gamma_s
            (k n Term.unit Term.unit) Typ.unit epsK Gamma_out) :
    HasType Delta Sigma Gamma_s
      (splitCotangentSeedFrom t gSeed n k)
      Typ.unit
      epsK
      Gamma_out := by
  cases t with
  | tensor ds =>
      rcases h_k with ⟨slotA, slotB, hBody⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letpair Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "gA" n) (freshName "gB" n)
          (Term.copy gSeed)
          (k (n + 2)
            (Term.var (freshName "gA" n))
            (Term.var (freshName "gB" n)))
          (Typ.tensor ds) (Typ.tensor ds) Typ.unit
          []
          epsK
          slotA slotB
          (HasType.copy Delta Sigma Gamma_s Gamma_s' gSeed ds [] h_seed)
          hBody)
  | pair t1 t2 =>
      rcases h_k with ⟨slotA, slotB, hBody⟩
      simpa [splitCotangentSeedFrom, cotangentType] using
        (HasType.letpair Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "gA" n) (freshName "gB" n)
          gSeed
          (splitCotangentSeedFrom t1 (Term.var (freshName "gA" n)) (n + 2)
            (fun n' gA1 gA2 =>
              splitCotangentSeedFrom t2 (Term.var (freshName "gB" n)) n'
                (fun n'' gB1 gB2 =>
                  k n''
                    (Term.pair gA1 gB1)
                    (Term.pair gA2 gB2))))
          (cotangentType t1) (cotangentType t2) Typ.unit
          []
          epsK
          slotA slotB
          h_seed
          hBody)
  | unit =>
      simpa [splitCotangentSeedFrom] using h_k
  | arrow _ _ _ =>
      simpa [splitCotangentSeedFrom] using h_k
  | tyVar _ =>
      simpa [splitCotangentSeedFrom] using h_k

/-- On quotient-based dimension multisets, re-inserting an erased member
    recovers the original multiset. This is the key shape fact for the
    typed `sum` adjoint branch. -/
private theorem ins_rem_eq_of_mem
    {ds : DimList} {d : Dim}
    (h : d ∈ ds) :
    ins (rem ds d) d = ds := by
  refine Quotient.inductionOn ds ?_ h
  intro l hmem
  have hmem' : d ∈ l := by
    simpa [DimList.mem] using hmem
  show Quotient.mk ListDimSetoid (d :: l.erase d) = Quotient.mk ListDimSetoid l
  exact Quotient.sound (s := ListDimSetoid) (List.perm_cons_erase hmem').symm

/-- Erasing the freshly inserted dimension cancels propositionally on the
    permutation quotient. This is the companion fact for `expand`. -/
private theorem rem_ins_eq
    {ds : DimList} {d : Dim} :
    rem (ins ds d) d = ds := by
  refine Quotient.inductionOn ds ?_
  intro l
  show Quotient.mk ListDimSetoid ((d :: l).erase d) = Quotient.mk ListDimSetoid l
  simp [rem, ins, DimList.erase, DimList.cons, List.erase_cons]

mutual

/-- Structural type-shape witness for `adjointTypedFrom`. This records
    exactly the local result-type information the typed transform needs
    to avoid falling back to the legacy tensor-only surface. The only
    constructor carrying extra typing payload is `mul`, where the
    transform embeds the source operands directly via `copy e1` / `copy e2`.
    Everything else is syntax-directed. -/
inductive AdjointTypedShape (Delta : CapCtx) (Sigma : StoreTyp) : Typ → Term → Prop
  | var {t : Typ} {x : String} :
      AdjointTypedShape Delta Sigma t (Term.var x)
  | const {t : Typ} {v : Float} {ds : DimList} :
      AdjointTypedShape Delta Sigma t (Term.const v ds)
  | unit {t : Typ} :
      AdjointTypedShape Delta Sigma t Term.unit
  | loc {t : Typ} {ell : Nat} :
      AdjointTypedShape Delta Sigma t (Term.loc ell)
  | letBind {t : Typ} {x : String} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma t e2 →
      AdjointTypedShape Delta Sigma t (Term.letBind x e1 e2)
  | letpair {t : Typ} {x y : String} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma t e2 →
      AdjointTypedShape Delta Sigma t (Term.letpair x y e1 e2)
  | add {ds : DimList} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e1 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e2 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) (Term.add e1 e2)
  | mul {ds : DimList} {e1 e2 : Term} :
      (∃ Γ1 Γ2 Γ3 eps1 eps2,
         HasType Delta Sigma Γ1 e1 (Typ.tensor ds) eps1 Γ2 ∧
         HasType Delta Sigma Γ2 e2 (Typ.tensor ds) eps2 Γ3) →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e1 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e2 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) (Term.mul e1 e2)
  | sum {ds : DimList} {d : Dim} {e : Term} :
      d ∈ ds →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e →
      AdjointTypedShape Delta Sigma (Typ.tensor (rem ds d)) (Term.sum e d)
  | expand {ds : DimList} {d : Dim} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e →
      AdjointTypedShape Delta Sigma (Typ.tensor (ins ds d)) (Term.expand e d)
  | copy {ds : DimList} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e →
      AdjointTypedShape Delta Sigma (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) (Term.copy e)
  | pair {t1 t2 : Typ} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma t1 e1 →
      AdjointTypedShape Delta Sigma t2 e2 →
      AdjointTypedShape Delta Sigma (Typ.pair t1 t2) (Term.pair e1 e2)
  | fst {t1 t2 : Typ} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.pair t1 t2) e →
      AdjointTypedShape Delta Sigma t1 (Term.fst t2 e)
  | snd {t1 t2 : Typ} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.pair t1 t2) e →
      AdjointTypedShape Delta Sigma t2 (Term.snd t1 e)
  | handle {t : Typ} {epsH : EffectRow} {body : Term}
      {clauses : List (EffectLabel × String × String × Term)} :
      AdjointTypedShape Delta Sigma t body →
      AdjointTypedClausesShape Delta Sigma t clauses →
      AdjointTypedShape Delta Sigma t (Term.handle epsH body clauses)
  | perform {op : EffectLabel} {e : Term} :
      AdjointTypedShape Delta Sigma (opArgType op) e →
      AdjointTypedShape Delta Sigma Typ.unit (Term.perform op e)

/-- Clause companion to `AdjointTypedShape`. Each handler clause body is
    transformed against the handled result type. -/
inductive AdjointTypedClausesShape
    (Delta : CapCtx) (Sigma : StoreTyp) :
    Typ → List (EffectLabel × String × String × Term) → Prop
  | nil :
      AdjointTypedClausesShape Delta Sigma t []
  | cons {op : EffectLabel} {x k : String} {hb : Term}
      {rest : List (EffectLabel × String × String × Term)} :
      AdjointTypedShape Delta Sigma t hb →
      AdjointTypedClausesShape Delta Sigma t rest →
      AdjointTypedClausesShape Delta Sigma t ((op, x, k, hb) :: rest)

end

mutual

/-- Extract the typed-transform shape witness from a source typing
    derivation plus the supported-fragment premise required by `T-Grad`. -/
private theorem adjointTypedShape_of_typed
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 e t eps Gamma2)
    (hSupp : AdjointSupported e) :
    AdjointTypedShape Delta Sigma t e := by
  induction h using HasType.rec
    (motive_2 := fun Delta Sigma Gamma2 Gamma3 t epsR cls _hcls =>
      AdjointSupportedClauses cls → AdjointTypedClausesShape Delta Sigma t cls) with
  | var =>
      exact .var
  | unit =>
      exact .unit
  | fst _ _ _ _ e t1 t2 _ hBody ih =>
      exact .fst (ih hSupp)
  | snd _ _ _ _ e t1 t2 _ hBody ih =>
      exact .snd (ih hSupp)
  | const =>
      exact .const
  | tadd _ _ _ _ _ e1 e2 ds _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨hSupp1, hSupp2⟩
      exact .add (ih1 hSupp1) (ih2 hSupp2)
  | tmul _ _ _ _ _ e1 e2 ds _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨hSupp1, hSupp2⟩
      exact .mul ⟨_, _, _, _, _, h1, h2⟩ (ih1 hSupp1) (ih2 hSupp2)
  | tsum _ _ _ _ e ds d _ hBody hmem ih =>
      exact .sum hmem (ih hSupp)
  | texpand _ _ _ _ e ds d _ hBody ih =>
      exact .expand (ih hSupp)
  | uniformLike _ _ _ _ _ _ _ _ _ hBody ih =>
      cases hSupp
  | copy _ _ _ _ e ds _ hBody ih =>
      exact .copy (ih hSupp)
  | letBind _ _ _ _ _ _ e1 e2 _ _ _ _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨_hSupp1, hSupp2⟩
      exact .letBind (ih2 hSupp2)
  | letpair _ _ _ _ _ _ _ e1 e2 _ _ _ _ _ _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨_hSupp1, hSupp2⟩
      exact .letpair (ih2 hSupp2)
  | tpair _ _ _ _ _ e1 e2 _ _ _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨hSupp1, hSupp2⟩
      exact .pair (ih1 hSupp1) (ih2 hSupp2)
  | loc =>
      exact .loc
  | perform _ _ _ _ op e tArg tRet _ hBody hMatch ih =>
      rcases hSupp with ⟨_hop, hSuppBody⟩
      have hArgEq : tArg = opArgType op := hMatch.1
      have hRetEq : tRet = Typ.unit := by
        cases op <;> simpa [OpSigMatch, opRetType] using hMatch.2
      cases hArgEq
      cases hRetEq
      exact .perform (ih hSuppBody)
  | handle _ _ _ _ _ body clauses _ _ _ hBody _ _ _ hClauses ihBody ihClauses =>
      rcases hSupp with ⟨hSuppBody, hSuppClauses⟩
      exact .handle (ihBody hSuppBody) (ihClauses hSuppClauses)
  | tgrad =>
      cases hSupp
  | tvmap =>
      cases hSupp
  | abs =>
      cases hSupp
  | app =>
      cases hSupp
  | subEff _ _ _ _ _ _ _ _ hBody _ ih =>
      exact ih hSupp
  | nil _ _ _ _ _ hSupp =>
      exact AdjointTypedClausesShape.nil
  | cons _ _ _ _ _ _ _ _ _ _ _ hb rest _ _ hMatch hBody hRest ihBody ihRest hSupp =>
      rcases hSupp with ⟨hSuppBody, hSuppRest⟩
      exact AdjointTypedClausesShape.cons (ihBody hSuppBody) (ihRest hSuppRest)

end

/-- A pair term can never type-check at a tensor result type. Used by
    the structured-seed counterexample below. -/
private theorem hasType_pair_tensor_absurd
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e1 e2 : Term} {ds : DimList} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) (Typ.tensor ds) eps Gamma2) :
    False := by
  generalize heq : Term.pair e1 e2 = e_in at h
  generalize hteq : Typ.tensor ds = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      cases heq
      cases hteq
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq hteq
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Local let-pair inversion used by the handled product-seed
    counterexample. `Progress.lean` has the public theorem, but
    importing it here would create a cycle through `Preservation`. -/
private theorem hasType_letpair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps GammaOut) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 t2 : Typ) (eps1 eps2 : EffectRow)
      (slotX slotY : Option Typ),
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
              (Gamma3 ++ [(x, slotX), (y, slotY)]) ∧
      GammaOut = Gamma3 := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1 t2 _ eps1 eps2 slotX slotY h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Local copy inversion used by the handled product-seed
    counterexample. `Preservation.lean` has the public theorem, but
    importing it here would create a cycle. -/
private theorem hasType_copy_inv
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
  | subEff Δ S Γ Γ' _ t' eps0 eps' _ hSub ih =>
      obtain ⟨ds, hteq, hInv⟩ := ih heq
      refine ⟨ds, hteq, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' hInv hSub
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Local variable inversion used by the higher-order product witness.
    Importing the public inversion surface from `Preservation.lean`
    would create a cycle. -/
private theorem hasType_var_ctx_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma GammaOut : LinearCtx}
    {x : String} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (Term.var x) t eps GammaOut) :
    ∃ GammaPre GammaPost,
      Gamma = GammaPre ++ [(x, some t)] ++ GammaPost ∧
      GammaOut = GammaPre ++ [(x, none)] ++ GammaPost := by
  generalize heq : Term.var x = e_in at h
  generalize hteq : t = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | var _ _ GammaPre GammaPost y ty =>
      cases heq
      cases hteq
      exact ⟨GammaPre, GammaPost, rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq hteq
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Structural predicate: every `Term.mul` sub-expression of `e` has
    its two operands well-typed as `tensor dsE` in *some* linear
    context chain. Threaded as the body-typing premise of
    `adjoint_typed_aux`: the seed-polymorphic helper does not itself
    carry a typing for `e`, so the `mul` case's recursive tape layer
    (which emits `copy e1` / `copy e2`) needs external witnesses for
    the operand typings. Structurally recursive on `Term` so Lean's
    termination checker accepts it; each constructor case recurses
    into the sub-terms the adjoint transform itself visits. The
    `handle` case only tracks the body (clauses are parked with
    `sum`/`expand`/`handle` in the catch-all). -/
def AdjointMulTyped (Delta : CapCtx) (Sigma : StoreTyp) (dsE : DimList) :
    Term → Prop
  | Term.var _           => True
  | Term.const _ _       => True
  | Term.unit            => True
  | Term.loc _           => True
  | Term.abs _ _ body    => AdjointMulTyped Delta Sigma dsE body
  | Term.app e1 e2       =>
      AdjointMulTyped Delta Sigma dsE e1 ∧ AdjointMulTyped Delta Sigma dsE e2
  | Term.letBind _ e1 e2 =>
      AdjointMulTyped Delta Sigma dsE e1 ∧ AdjointMulTyped Delta Sigma dsE e2
  | Term.copy e          => AdjointMulTyped Delta Sigma dsE e
  | Term.letpair _ _ e1 e2 =>
      AdjointMulTyped Delta Sigma dsE e1 ∧ AdjointMulTyped Delta Sigma dsE e2
  | Term.pair e1 e2      =>
      AdjointMulTyped Delta Sigma dsE e1 ∧ AdjointMulTyped Delta Sigma dsE e2
  | Term.fst _ e         => AdjointMulTyped Delta Sigma dsE e
  | Term.snd _ e         => AdjointMulTyped Delta Sigma dsE e
  | Term.add e1 e2       =>
      AdjointMulTyped Delta Sigma dsE e1 ∧ AdjointMulTyped Delta Sigma dsE e2
  | Term.mul e1 e2       =>
      (∃ Γ1 Γ2 Γ3 eps1 eps2,
         HasType Delta Sigma Γ1 e1 (Typ.tensor dsE) eps1 Γ2 ∧
         HasType Delta Sigma Γ2 e2 (Typ.tensor dsE) eps2 Γ3) ∧
      AdjointMulTyped Delta Sigma dsE e1 ∧
      AdjointMulTyped Delta Sigma dsE e2
  | Term.sum e _         => AdjointMulTyped Delta Sigma dsE e
  | Term.expand e _      => AdjointMulTyped Delta Sigma dsE e
  | Term.uniformLike e _ _ => AdjointMulTyped Delta Sigma dsE e
  | Term.grad _ _ _ body => AdjointMulTyped Delta Sigma dsE body
  | Term.vmap _ _ _ body   => AdjointMulTyped Delta Sigma dsE body
  | Term.handle _ body _ => AdjointMulTyped Delta Sigma dsE body
  | Term.perform _ e     => AdjointMulTyped Delta Sigma dsE e

mutual

/-- Seed-polymorphic helper for adjoint typing. The proof depends on
    the seed's typing, the structural shape of `e`, and the freshness
    of adjoint counter-indexed names at counters ≥ `n`. The closed
    structural cases recurse syntactically; the hard `mul` / `expand`
    cases are still parked below. After the Phase 1 placeholder was
    corrected to recurse through the result-producing `letBind` /
    `letpair` body, the old concrete `grad` / `expand` witness
    disappeared. But the current public surface is still too strong:
    Lean now also contains a typed `snd` / `pair` / `expand` witness
    showing that structural recursion can still route an incompatible
    seed into an `expand` subterm. So the remaining `expand` debt is a
    broader theorem-shape gap: `adjoint_typed_aux` only tracks the
    current seed typing, but the `expand` branch needs a source-side
    relation strong enough to show the incoming seed dimension list
    actually contains the summed dimension along every structural route
    that can reach that subterm. -/
private theorem adjoint_typed_aux
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma_s Gamma_s' : LinearCtx)
    (dsE : DimList) (epsSeed : EffectRow) (x : String) (n : Nat)
    (e : Term) (gSeed : Term)
    (h_seed : HasType Delta Sigma Gamma_s gSeed (Typ.tensor dsE) epsSeed Gamma_s')
    (h_fresh_s : AdjointNamesFresh n Gamma_s)
    (h_fresh_s' : AdjointNamesFresh n Gamma_s') :
    HasType Delta Sigma Gamma_s (adjointFrom e x gSeed n) Typ.unit
            (EffectRow.union [EffectLabel.accum] epsSeed) Gamma_s' := by
  -- Local witness: the leaf skeleton consumes `gSeed` linearly and
  -- then emits a unit-valued `accum`, so it has type `unit` with
  -- effect row `union [accum] epsSeed`.
  have leaf_perform :
      HasType Delta Sigma Gamma_s
        (adjointLeaf gSeed n) Typ.unit
        (EffectRow.union [EffectLabel.accum] epsSeed) Gamma_s' := by
    have hAccumSig : OpSigMatch EffectLabel.accum Typ.unit Typ.unit := by
      simp [OpSigMatch, opArgType, opRetType]
    have hPerform :
        HasType Delta Sigma
          (Gamma_s' ++ [(freshName "adjA" n, some (Typ.tensor dsE))])
          (Term.perform EffectLabel.accum Term.unit)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ [(freshName "adjA" n, some (Typ.tensor dsE))]) := by
      exact HasType.perform Delta Sigma
        (Gamma_s' ++ [(freshName "adjA" n, some (Typ.tensor dsE))])
        (Gamma_s' ++ [(freshName "adjA" n, some (Typ.tensor dsE))])
        EffectLabel.accum Term.unit Typ.unit Typ.unit []
        (HasType.unit Delta Sigma _)
        hAccumSig
    have hLet :
        HasType Delta Sigma Gamma_s (adjointLeaf gSeed n) Typ.unit
          (EffectRow.union epsSeed
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))) Gamma_s' := by
      simpa [adjointLeaf] using
        (HasType.letBind Delta Sigma Gamma_s Gamma_s' Gamma_s'
          (freshName "adjA" n) gSeed
          (Term.perform EffectLabel.accum Term.unit)
          (Typ.tensor dsE) Typ.unit epsSeed
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (some (Typ.tensor dsE)) h_seed hPerform)
    exact HasType.subEff Delta Sigma Gamma_s Gamma_s'
      (adjointLeaf gSeed n) Typ.unit
      (EffectRow.union epsSeed
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
      (EffectRow.union [EffectLabel.accum] epsSeed)
      hLet
      (subEff_seed_accum epsSeed)
  match e with
  -- Leaf cases.
  | Term.var _ =>
      simp only [adjointFrom]; split <;> exact leaf_perform
  | Term.const _ _ =>
      simp only [adjointFrom]; exact leaf_perform
  | Term.unit =>
      simp only [adjointFrom]; exact leaf_perform
  | Term.loc _ =>
      simp only [adjointFrom]; exact leaf_perform
  -- Vestigial Phase-1 structural cases: adjoint recurses on a single
  -- sub-term with the same seed and counter.
  | Term.letBind _ _ e2 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e2 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.letpair _ _ _ e2 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e2 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.abs _ _ e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.app e1 _ =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.grad _ _ _ e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.vmap _ _ _ e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.perform _ e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.uniformLike e1 _ _ =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  -- Real adjoint cases (add, mul).
  | Term.add e1 e2 =>
      simp only [adjointFrom]
      let gA : String := freshName "gA" n
      let gB : String := freshName "gB" n
      let adjA : String := freshName "adjA" n
      have hFreshBase : AdjointNamesFresh (n + 3) Gamma_s' :=
        h_fresh_s'.mono (Nat.le_add_right n 3)
      have hkFresh : n < n + 3 := Nat.lt_add_of_pos_right (by decide : 0 < 3)
      have hFreshPairIn :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh)
      have hFreshPairOut :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := none)
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh)
      have hFreshLetIn :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++
              [(gA, none), (gB, some (Typ.tensor dsE)), (adjA, some Typ.unit)]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := none)
            (by simp [adjointBases]) hkFresh
        have h2 :=
          AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, adjA, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h2) (b := "adjA") (t := some Typ.unit)
            (by simp [adjointBases]) hkFresh)
      have hFreshLetOut :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++
              [(gA, none), (gB, none), (adjA, some Typ.unit)]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := none)
            (by simp [adjointBases]) hkFresh
        have h2 :=
          AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := none)
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, adjA, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h2) (b := "adjA") (t := some Typ.unit)
            (by simp [adjointBases]) hkFresh)
      have hSeedCopy :
          HasType Delta Sigma Gamma_s (Term.copy gSeed)
            (Typ.pair (Typ.tensor dsE) (Typ.tensor dsE)) epsSeed Gamma_s' := by
        exact HasType.copy Delta Sigma Gamma_s Gamma_s' gSeed dsE epsSeed h_seed
      have hSeedA :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (Term.var gA) (Typ.tensor dsE) []
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))]) := by
        simpa [gA, gB, List.append_assoc] using
          (HasType.var Delta Sigma Gamma_s'
            ([(gB, some (Typ.tensor dsE))] : LinearCtx) gA (Typ.tensor dsE))
      have hAdj1 :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (adjointFrom e1 x (Term.var gA) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))]) := by
        exact adjoint_typed_aux Delta Sigma
          (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
          (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))])
          dsE [] x (n + 3) e1 (Term.var gA)
          hSeedA hFreshPairIn hFreshPairOut
      have hSeedB :
          HasType Delta Sigma
            (((Gamma_s' ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor dsE))]) ++
              [(adjA, some Typ.unit)])
            (Term.var gB) (Typ.tensor dsE) []
            (((Gamma_s' ++ [(gA, none)]) ++
              [(gB, none)]) ++
              [(adjA, some Typ.unit)]) := by
        have hRaw :
            HasType Delta Sigma
              (((Gamma_s' ++ [(gA, none)]) ++
                [(gB, some (Typ.tensor dsE))]) ++
                [(adjA, some Typ.unit)])
              (Term.var gB) (Typ.tensor dsE) []
              (((Gamma_s' ++ [(gA, none)]) ++
                [(gB, none)]) ++
                [(adjA, some Typ.unit)]) :=
          HasType.var Delta Sigma
            (Gamma_s' ++ [(gA, none)])
            ([(adjA, some Typ.unit)] : LinearCtx) gB (Typ.tensor dsE)
        simpa [gA, gB, adjA, List.append_assoc] using hRaw
      have hAdj2 :
          HasType Delta Sigma
            (((Gamma_s' ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor dsE))]) ++
              [(adjA, some Typ.unit)])
            (adjointFrom e2 x (Term.var gB) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ [(gA, none)]) ++
              [(gB, none)]) ++
              [(adjA, some Typ.unit)]) := by
        exact adjoint_typed_aux Delta Sigma
          ((((Gamma_s' ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor dsE))]) ++
              [(adjA, some Typ.unit)]))
          ((((Gamma_s' ++ [(gA, none)]) ++
              [(gB, none)]) ++
              [(adjA, some Typ.unit)]))
          dsE [] x (n + 3) e2 (Term.var gB)
          hSeedB
          (by simpa [List.append_assoc] using hFreshLetIn)
          (by simpa [List.append_assoc] using hFreshLetOut)
      have hBody :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (Term.letBind adjA
              (adjointFrom e1 x (Term.var gA) (n + 3))
              (adjointFrom e2 x (Term.var gB) (n + 3)))
            Typ.unit
            (EffectRow.union
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
            (Gamma_s' ++ [(gA, none), (gB, none)]) := by
        simpa [adjA, gA, gB, List.append_assoc] using
          (HasType.letBind Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))])
            (Gamma_s' ++ [(gA, none), (gB, none)])
            adjA
            (adjointFrom e1 x (Term.var gA) (n + 3))
            (adjointFrom e2 x (Term.var gB) (n + 3))
            Typ.unit Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (some Typ.unit)
            hAdj1
            (by simpa [List.append_assoc] using hAdj2))
      have hAddRaw :
          HasType Delta Sigma Gamma_s
            (Term.letpair gA gB (Term.copy gSeed)
              (Term.letBind adjA
                (adjointFrom e1 x (Term.var gA) (n + 3))
                (adjointFrom e2 x (Term.var gB) (n + 3))))
            Typ.unit
            (EffectRow.union epsSeed
              (EffectRow.union
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
            Gamma_s' := by
        simpa [gA, gB, adjA, List.append_assoc] using
          (HasType.letpair Delta Sigma Gamma_s Gamma_s' Gamma_s'
            gA gB
            (Term.copy gSeed)
            (Term.letBind adjA
              (adjointFrom e1 x (Term.var gA) (n + 3))
              (adjointFrom e2 x (Term.var gB) (n + 3)))
            (Typ.tensor dsE) (Typ.tensor dsE) Typ.unit
            epsSeed
            (EffectRow.union
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
            none none
            hSeedCopy hBody)
      exact HasType.subEff Delta Sigma Gamma_s Gamma_s'
        (Term.letpair gA gB (Term.copy gSeed)
          (Term.letBind adjA
            (adjointFrom e1 x (Term.var gA) (n + 3))
            (adjointFrom e2 x (Term.var gB) (n + 3))))
        Typ.unit
        (EffectRow.union epsSeed
          (EffectRow.union
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
        (EffectRow.union [EffectLabel.accum] epsSeed)
        hAddRaw
        (subEff_letpair_add epsSeed)
  | Term.sum e1 d =>
      simp only [adjointFrom]
      have hSeedExpand :
          HasType Delta Sigma Gamma_s
            (Term.expand gSeed d)
            (Typ.tensor (ins dsE d))
            epsSeed
            Gamma_s' := by
        exact HasType.texpand Delta Sigma Gamma_s Gamma_s' gSeed dsE d epsSeed h_seed
      exact adjoint_typed_aux Delta Sigma
        Gamma_s Gamma_s'
        (ins dsE d) epsSeed x n e1 (Term.expand gSeed d)
        hSeedExpand h_fresh_s h_fresh_s'
  | Term.handle _ body clauses =>
      simpa [adjointFrom] using
        (adjointClauses_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed
          x n clauses body gSeed h_seed h_fresh_s h_fresh_s')
  -- `mul` and `expand` remain under the explicit admitted branches.
  --
  -- Wave 5 Track B: `AdjointTransform` reshaped the `mul` case so
  -- `copy gSeed` is the outermost letpair (before any operand tape
  -- layer). That aligns the seed's Γ_s → Γ_s' threading with the
  -- outer derivation — the originally-identified blocker. The
  -- reshape is semantically equivalent (copy gSeed, copy e1, copy e2,
  -- mul a b all commute as pure values on disjoint bindings).
  --
  -- Two structural obstacles remain for closing the `mul` case,
  -- neither of which is surmountable without touching files outside
  -- this one:
  --
  --  (i) Operand rebasing. `AdjointMulTyped.mul` supplies an
  --      existential typing chain `Γ1 → Γ2 → Γ3` for `e1`/`e2` at
  --      `tensor dsE`. After the reshape, the operand tape layers
  --      `copy e1` / `copy e2` must type at the adjoint's extended
  --      linear context (`Γ_s' ++ [(gA,τ),(gB,τ)]` and deeper), which
  --      is not the existential `Γ1` the predicate supplies.
  --      Strengthening the predicate to `∀ Γ, ∃ Γ' eps, ...` would
  --      close the rebasing — but it would also require the caller
  --      to supply a no-consumption witness, and the effect rows
  --      become existential, reintroducing obstacle (ii).
  --
  -- (ii) Effect-row bridging. The nested letpair/letBind chain
  --      produces a compound row `union epsSeed (union eps_e1 (union
  --      eps_e2 (... accum ... accum ...)))` where `eps_e1`, `eps_e2`
  --      come from the operand typings. To bridge via `subEff` to
  --      the target `union [accum] epsSeed`, the operand effect rows
  --      must be empty (or subsumed by `epsSeed`). That cannot be
  --      derived from `AdjointMulTyped` without adding a purity
  --      premise to the predicate. The purity premise, in turn, is
  --      too strong for realistic user programs that use `copy` /
  --      `perform` on operands.
  --
  -- Named substitution is no longer the blocker here. The remaining
  -- `mul` work needs a stronger theorem-level invariant for rebasing
  -- operand tape typings and their effect rows.
  --
  -- `expand` is no longer blocked by the old structural `letBind` /
  -- `letpair` routing bug. The remaining issue is theorem shape:
  -- this helper only carries the seed typing, but the `expand` branch
  -- needs the source-side relation that witnesses the summed dimension
  -- is actually present in the current seed shape.
  -- `handle` is now typed for the repaired seed-threading shape; the
  -- transform still eagerly sequences every clause adjoint, but that is
  -- a known Phase 1 semantic caveat in `AdjointTransform`, not a typing
  -- blocker local to this file.
  | _ => sorry
termination_by (sizeOf e, 1)
decreasing_by
  all_goals
    simp_wf
    omega

private theorem adjointClauses_typed_aux
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma_s Gamma_s' : LinearCtx)
    (dsE : DimList) (epsSeed : EffectRow) (x : String) (n : Nat)
    (clauses : List (EffectLabel × String × String × Term))
    (body : Term) (gSeed : Term)
    (h_seed : HasType Delta Sigma Gamma_s gSeed (Typ.tensor dsE) epsSeed Gamma_s')
    (h_fresh_s : AdjointNamesFresh n Gamma_s)
    (h_fresh_s' : AdjointNamesFresh n Gamma_s') :
    HasType Delta Sigma Gamma_s (adjointClausesFrom clauses x body gSeed n) Typ.unit
            (EffectRow.union [EffectLabel.accum] epsSeed) Gamma_s' := by
  match clauses with
  | [] =>
      simpa [adjointClausesFrom] using
        (adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n body gSeed
          h_seed h_fresh_s h_fresh_s')
  | (_op, _xv, _kv, hb) :: rest =>
      let gA : String := freshName "gA" n
      let gB : String := freshName "gB" n
      let adjHb : String := freshName "adjHb" n
      have hFreshBase : AdjointNamesFresh (n + 3) Gamma_s' :=
        h_fresh_s'.mono (Nat.le_add_right n 3)
      have hkFresh : n < n + 3 := Nat.lt_add_of_pos_right (by decide : 0 < 3)
      have hFreshPairIn :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh)
      have hFreshPairOut :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := none)
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh)
      have hFreshLetIn :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE)), (adjHb, some Typ.unit)]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := none)
            (by simp [adjointBases]) hkFresh
        have h2 :=
          AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := some (Typ.tensor dsE))
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, adjHb, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h2) (b := "adjHb") (t := some Typ.unit)
            (by simp [adjointBases]) hkFresh)
      have hFreshLetOut :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, none), (gB, none), (adjHb, some Typ.unit)]) := by
        have h1 :=
          AdjointNamesFresh.cons_freshName
            (h := hFreshBase) (b := "gA") (t := none)
            (by simp [adjointBases]) hkFresh
        have h2 :=
          AdjointNamesFresh.cons_freshName
            (h := h1) (b := "gB") (t := none)
            (by simp [adjointBases]) hkFresh
        simpa [gA, gB, adjHb, List.append_assoc] using
          (AdjointNamesFresh.cons_freshName
            (h := h2) (b := "adjHb") (t := some Typ.unit)
            (by simp [adjointBases]) hkFresh)
      have hSeedCopy :
          HasType Delta Sigma Gamma_s (Term.copy gSeed)
            (Typ.pair (Typ.tensor dsE) (Typ.tensor dsE)) epsSeed Gamma_s' := by
        exact HasType.copy Delta Sigma Gamma_s Gamma_s' gSeed dsE epsSeed h_seed
      have hSeedA :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (Term.var gA) (Typ.tensor dsE) []
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))]) := by
        simpa [gA, gB, List.append_assoc] using
          (HasType.var Delta Sigma Gamma_s'
            ([(gB, some (Typ.tensor dsE))] : LinearCtx) gA (Typ.tensor dsE))
      have hAdjHb :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (adjointFrom hb x (Term.var gA) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))]) := by
        exact adjoint_typed_aux Delta Sigma
          (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
          (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))])
          dsE [] x (n + 3) hb (Term.var gA)
          hSeedA hFreshPairIn hFreshPairOut
      have hSeedB :
          HasType Delta Sigma
            (((Gamma_s' ++ [(gA, none)]) ++ [(gB, some (Typ.tensor dsE))]) ++ [(adjHb, some Typ.unit)])
            (Term.var gB) (Typ.tensor dsE) []
            (((Gamma_s' ++ [(gA, none)]) ++ [(gB, none)]) ++ [(adjHb, some Typ.unit)]) := by
        have hRaw :
            HasType Delta Sigma
              (((Gamma_s' ++ [(gA, none)]) ++ [(gB, some (Typ.tensor dsE))]) ++ [(adjHb, some Typ.unit)])
              (Term.var gB) (Typ.tensor dsE) []
              (((Gamma_s' ++ [(gA, none)]) ++ [(gB, none)]) ++ [(adjHb, some Typ.unit)]) :=
          HasType.var Delta Sigma
            (Gamma_s' ++ [(gA, none)])
            ([(adjHb, some Typ.unit)] : LinearCtx) gB (Typ.tensor dsE)
        simpa [gA, gB, adjHb, List.append_assoc] using hRaw
      have hAdjRest :
          HasType Delta Sigma
            (((Gamma_s' ++ [(gA, none)]) ++ [(gB, some (Typ.tensor dsE))]) ++ [(adjHb, some Typ.unit)])
            (adjointClausesFrom rest x body (Term.var gB) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ [(gA, none)]) ++ [(gB, none)]) ++ [(adjHb, some Typ.unit)]) := by
        exact adjointClauses_typed_aux Delta Sigma
          ((((Gamma_s' ++ [(gA, none)]) ++ [(gB, some (Typ.tensor dsE))]) ++ [(adjHb, some Typ.unit)]))
          ((((Gamma_s' ++ [(gA, none)]) ++ [(gB, none)]) ++ [(adjHb, some Typ.unit)]))
          dsE [] x (n + 3) rest body (Term.var gB)
          hSeedB
          (by simpa [List.append_assoc] using hFreshLetIn)
          (by simpa [List.append_assoc] using hFreshLetOut)
      have hBody :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (Term.letBind adjHb
              (adjointFrom hb x (Term.var gA) (n + 3))
              (adjointClausesFrom rest x body (Term.var gB) (n + 3)))
            Typ.unit
            (EffectRow.union
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
            (Gamma_s' ++ [(gA, none), (gB, none)]) := by
        simpa [gA, gB, adjHb, List.append_assoc] using
          (HasType.letBind Delta Sigma
            (Gamma_s' ++ [(gA, some (Typ.tensor dsE)), (gB, some (Typ.tensor dsE))])
            (Gamma_s' ++ [(gA, none), (gB, some (Typ.tensor dsE))])
            (Gamma_s' ++ [(gA, none), (gB, none)])
            adjHb
            (adjointFrom hb x (Term.var gA) (n + 3))
            (adjointClausesFrom rest x body (Term.var gB) (n + 3))
            Typ.unit Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (some Typ.unit)
            hAdjHb
            (by simpa [List.append_assoc] using hAdjRest))
      have hRaw :
          HasType Delta Sigma Gamma_s
            (Term.letpair gA gB (Term.copy gSeed)
              (Term.letBind adjHb
                (adjointFrom hb x (Term.var gA) (n + 3))
                (adjointClausesFrom rest x body (Term.var gB) (n + 3))))
            Typ.unit
            (EffectRow.union epsSeed
              (EffectRow.union
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
            Gamma_s' := by
        simpa [gA, gB, adjHb, List.append_assoc] using
          (HasType.letpair Delta Sigma Gamma_s Gamma_s' Gamma_s'
            gA gB
            (Term.copy gSeed)
            (Term.letBind adjHb
              (adjointFrom hb x (Term.var gA) (n + 3))
              (adjointClausesFrom rest x body (Term.var gB) (n + 3)))
            (Typ.tensor dsE) (Typ.tensor dsE) Typ.unit
            epsSeed
            (EffectRow.union
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
            none none
            hSeedCopy hBody)
      simpa [adjointClausesFrom, gA, gB, adjHb] using
        (HasType.subEff Delta Sigma Gamma_s Gamma_s'
          (Term.letpair gA gB (Term.copy gSeed)
            (Term.letBind adjHb
              (adjointFrom hb x (Term.var gA) (n + 3))
              (adjointClausesFrom rest x body (Term.var gB) (n + 3))))
          Typ.unit
          (EffectRow.union epsSeed
            (EffectRow.union
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
          (EffectRow.union [EffectLabel.accum] epsSeed)
          hRaw
          (subEff_letpair_add epsSeed))
termination_by (sizeOf clauses + sizeOf body + 1, 0)
decreasing_by
  all_goals
    simp_wf
    omega

end

private theorem hasType_letBind_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps Gamma_out) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 : Typ) (eps1 eps2 : EffectRow)
      (slot : Option Typ),
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t eps2
              (Gamma3 ++ [(x, slot)]) ∧
      Gamma_out = Gamma3 := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Γ2 Γ3 _ _ _ t1 _ eps1 eps2 slot h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, eps1, eps2, slot, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem hasType_pair_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) t eps GammaOut) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 t2 : Typ) (eps1 eps2 : EffectRow),
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 ∧
      t = Typ.pair t1 t2 ∧
      GammaOut = Gamma3 := by
  generalize heq : Term.pair e1 e2 = e_in at h
  generalize hteq : t = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ Gamma2 Gamma3 _ _ t1 t2 eps1 eps2 h1 h2 _ _ =>
      cases heq
      cases hteq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, h1, h2, rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      obtain ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, h1, h2, ht, hOut⟩ := ih heq hteq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, h1, h2, ht, hOut⟩
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem hasType_sum_inv_local
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
  | subEff Δ S Γ Γ' _ _ eps0 eps' _hSub hSub ih =>
      obtain ⟨ds, hteq, hmem, hInv⟩ := ih heq
      refine ⟨ds, hteq, hmem, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' hInv hSub
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

private def adjointExpandGapDim : Dim :=
  Dim.named "dGap"

private def adjointExpandGapBody : Term :=
  Term.letBind "y"
    (Term.expand (Term.const 0 DimList.empty) adjointExpandGapDim)
    (Term.const 0 DimList.empty)

private theorem adjointExpandGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
      adjointExpandGapBody
      (Typ.tensor DimList.empty)
      []
      ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
  have hExpandArg :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Typ.tensor DimList.empty)
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.const (Capability.diff :: []) Sigma _ 0 DimList.empty
  have hExpand :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.expand (Term.const 0 DimList.empty) adjointExpandGapDim)
        (Typ.tensor (ins DimList.empty adjointExpandGapDim))
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.texpand (Capability.diff :: []) Sigma
      _ _ _ DimList.empty adjointExpandGapDim []
      hExpandArg
  have hBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty)),
          ("y", some (Typ.tensor (ins DimList.empty adjointExpandGapDim)))] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Typ.tensor DimList.empty)
        []
        ([("x", some (Typ.tensor DimList.empty)),
          ("y", some (Typ.tensor (ins DimList.empty adjointExpandGapDim)))] : LinearCtx) := by
    exact HasType.const (Capability.diff :: []) Sigma _ 0 DimList.empty
  exact HasType.letBind (Capability.diff :: []) Sigma
    _ _ _
    "y"
    (Term.expand (Term.const 0 DimList.empty) adjointExpandGapDim)
    (Term.const 0 DimList.empty)
    (Typ.tensor (ins DimList.empty adjointExpandGapDim))
    (Typ.tensor DimList.empty)
    []
    []
    (some (Typ.tensor (ins DimList.empty adjointExpandGapDim)))
    hExpand
    hBody

private theorem adjointExpandGapAdjoint_typed_gen
    {Sigma : StoreTyp} {x gs : String} {n : Nat} :
    HasType [] Sigma
      ([(x, some (Typ.tensor DimList.empty)),
        (gs, some (Typ.tensor DimList.empty))] : LinearCtx)
      (adjointFrom adjointExpandGapBody x (Term.var gs) n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      ([(x, some (Typ.tensor DimList.empty)),
        (gs, none)] : LinearCtx) := by
  have hSeed :
      HasType [] Sigma
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.var gs)
        (Typ.tensor DimList.empty)
        []
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, none)] : LinearCtx) := by
    simpa [List.append_assoc] using
      (HasType.var [] Sigma
        ([(x, some (Typ.tensor DimList.empty))] : LinearCtx)
        ([] : LinearCtx)
        gs
        (Typ.tensor DimList.empty))
  have hAccumSig : OpSigMatch EffectLabel.accum Typ.unit Typ.unit := by
    simp [OpSigMatch, opArgType, opRetType]
  have hPerform :
      HasType [] Sigma
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, none),
          (freshName "adjA" n, some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.perform EffectLabel.accum Term.unit)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, none),
          (freshName "adjA" n, some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.perform [] Sigma
      _ _ EffectLabel.accum Term.unit Typ.unit Typ.unit []
      (HasType.unit [] Sigma _)
      hAccumSig
  have hLet :
      HasType [] Sigma
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, some (Typ.tensor DimList.empty))] : LinearCtx)
        (adjointLeaf (Term.var gs) n)
        Typ.unit
        (EffectRow.union []
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, none)] : LinearCtx) := by
    simpa [adjointLeaf] using
      (HasType.letBind [] Sigma
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, some (Typ.tensor DimList.empty))] : LinearCtx)
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, none)] : LinearCtx)
        ([(x, some (Typ.tensor DimList.empty)),
          (gs, none)] : LinearCtx)
        (freshName "adjA" n)
        (Term.var gs)
        (Term.perform EffectLabel.accum Term.unit)
        (Typ.tensor DimList.empty)
        Typ.unit
        []
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (some (Typ.tensor DimList.empty))
        hSeed
        hPerform)
  have hSub :=
    HasType.subEff [] Sigma
      ([(x, some (Typ.tensor DimList.empty)),
        (gs, some (Typ.tensor DimList.empty))] : LinearCtx)
      ([(x, some (Typ.tensor DimList.empty)),
        (gs, none)] : LinearCtx)
      (adjointLeaf (Term.var gs) n)
      Typ.unit
      (EffectRow.union []
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      hLet
      (subEff_seed_accum [])
  simpa [adjointExpandGapBody, adjointFrom] using hSub

private def adjointExpandGapGradT : Typ :=
  Typ.tensor DimList.empty

private def adjointExpandGapGradType : Typ :=
  Typ.arrow adjointExpandGapGradT
    (Typ.arrow adjointExpandGapGradT adjointExpandGapGradT []) []

private def adjointExpandGapGradX : String :=
  "x"

private def adjointExpandGapGradSeed : String :=
  gradSeedName adjointExpandGapGradX adjointExpandGapBody

private def adjointExpandGapGradTmp : String :=
  gradResultName adjointExpandGapGradX adjointExpandGapBody

private def adjointExpandGapGradCounter : Nat :=
  gradAdjointCounter adjointExpandGapGradX adjointExpandGapGradSeed
    adjointExpandGapBody

private def adjointExpandGapGradClauseBody : Term :=
  Term.app (Term.var "k") (Term.var "p")

private def adjointExpandGapGradClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.accum, "p", "k", adjointExpandGapGradClauseBody)]

private def adjointExpandGapGradTerm : Term :=
  Term.grad adjointExpandGapGradX adjointExpandGapGradT
    adjointExpandGapGradT adjointExpandGapBody

private def adjointExpandGapGradReduct : Term :=
  Term.abs adjointExpandGapGradX adjointExpandGapGradT
    (Term.abs adjointExpandGapGradSeed adjointExpandGapGradT
      (Term.letBind adjointExpandGapGradTmp
        (Term.handle [EffectLabel.accum]
          (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
            (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
          adjointExpandGapGradClauses)
        (Term.var adjointExpandGapGradX)))

private theorem adjointExpandGapGradSeed_ne_x :
    adjointExpandGapGradSeed ≠ adjointExpandGapGradX := by
  intro hEq
  have hs :
      freshNameAvoiding
          ("x" :: (freeVars adjointExpandGapBody ++ boundVars adjointExpandGapBody)) = "x" := by
    simpa [adjointExpandGapGradSeed, adjointExpandGapGradX, gradSeedName] using hEq
  have hnot :=
    freshNameAvoiding_not_mem
      ("x" :: (freeVars adjointExpandGapBody ++ boundVars adjointExpandGapBody))
  apply hnot
  simp [hs]

private theorem adjointExpandGapGradTerm_typed
    {Sigma : StoreTyp} :
    HasType [] Sigma [] adjointExpandGapGradTerm adjointExpandGapGradType [] [] := by
  unfold adjointExpandGapGradTerm adjointExpandGapGradType adjointExpandGapGradT
  exact HasType.tgrad [] Sigma [] "x" DimList.empty DimList.empty adjointExpandGapBody []
    (some (Typ.tensor DimList.empty))
    adjointExpandGapBody_typed
    (by simp [subsetEffRow, DiffCompat])
    (by simp [AdjointSupported, adjointExpandGapBody])
    (by
      intro z t hz
      simp [adjointExpandGapBody, freeVars] at hz)

private theorem adjointExpandGapGradReduct_typed
    {Sigma : StoreTyp} :
    HasType [] Sigma [] adjointExpandGapGradReduct adjointExpandGapGradType [] [] := by
  let epsAdj : EffectRow := EffectRow.union [EffectLabel.accum] ([] : EffectRow)
  let epsHandle : EffectRow := EffectRow.removeOps epsAdj [EffectLabel.accum]
  have hAdj :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, some (Typ.tensor DimList.empty))] : LinearCtx)
        (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
          (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
        Typ.unit
        epsAdj
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx) := by
    simpa [epsAdj, adjointExpandGapGradT, adjointTypedFrom, adjointExpandGapBody, adjointFrom] using
      (adjointExpandGapAdjoint_typed_gen
        (Sigma := Sigma)
        (x := adjointExpandGapGradX)
        (gs := adjointExpandGapGradSeed)
        (n := adjointExpandGapGradCounter))
  have hClauseSig : OpSigMatch EffectLabel.accum Typ.unit Typ.unit := by
    simp [OpSigMatch, opArgType, opRetType]
  have hVarK :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
        (Term.var "k")
        (Typ.arrow Typ.unit Typ.unit epsHandle)
        []
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", none)] : LinearCtx) := by
    simpa [List.append_assoc, epsHandle] using
      (HasType.var [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit)] : LinearCtx)
        ([] : LinearCtx)
        "k"
        (Typ.arrow Typ.unit Typ.unit epsHandle))
  have hVarP :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", none)] : LinearCtx)
        (Term.var "p")
        Typ.unit
        []
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", none),
          ("k", none)] : LinearCtx) := by
    simpa [List.append_assoc] using
      (HasType.var [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        ([("k", none)] : LinearCtx)
        "p"
        Typ.unit)
  have hClauseBodyRaw :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
        adjointExpandGapGradClauseBody
        Typ.unit
        (EffectRow.union (EffectRow.union [] []) epsHandle)
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", none),
          ("k", none)] : LinearCtx) := by
    simpa [adjointExpandGapGradClauseBody] using
      (HasType.app [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", none)] : LinearCtx)
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", none),
          ("k", none)] : LinearCtx)
        (Term.var "k")
        (Term.var "p")
        Typ.unit
        Typ.unit
        epsHandle
        []
        []
        hVarK
        hVarP)
  have hClauseBodySub :
      SubEffRow (EffectRow.union (EffectRow.union [] []) epsHandle) epsHandle := by
    intro op hop
    simpa [EffectRow.union, List.mem_filter] using hop
  have hClauseBody :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", some Typ.unit),
          ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
        adjointExpandGapGradClauseBody
        Typ.unit
        epsHandle
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          ("p", none),
          ("k", none)] : LinearCtx) := by
    exact HasType.subEff [] Sigma
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none),
        ("p", some Typ.unit),
        ("k", some (Typ.arrow Typ.unit Typ.unit epsHandle))] : LinearCtx)
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none),
        ("p", none),
        ("k", none)] : LinearCtx)
      adjointExpandGapGradClauseBody
      Typ.unit
      (EffectRow.union (EffectRow.union [] []) epsHandle)
      epsHandle
      hClauseBodyRaw
      hClauseBodySub
  have hClausesNil :
      ClausesTyped [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        Typ.unit
        epsHandle
        [] :=
    ClausesTyped.nil [] Sigma
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none)] : LinearCtx)
      Typ.unit
      epsHandle
  have hClauses :
      ClausesTyped [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        Typ.unit
        epsHandle
        adjointExpandGapGradClauses := by
    exact ClausesTyped.cons [] Sigma
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none)] : LinearCtx)
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none)] : LinearCtx)
      Typ.unit
      Typ.unit
      Typ.unit
      epsHandle
      EffectLabel.accum
      "p"
      "k"
      adjointExpandGapGradClauseBody
      []
      none
      none
      hClauseSig
      hClauseBody
      hClausesNil
  have hHandleRaw :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.handle [EffectLabel.accum]
          (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
            (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
          adjointExpandGapGradClauses)
        Typ.unit
        epsHandle
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx) := by
    refine HasType.handle [] Sigma
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, some (Typ.tensor DimList.empty))] : LinearCtx)
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none)] : LinearCtx)
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
        (adjointExpandGapGradSeed, none)] : LinearCtx)
      (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
        (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
      adjointExpandGapGradClauses
      Typ.unit
      [EffectLabel.accum]
      epsAdj
      hAdj
      ?_
      ?_
      ?_
      hClauses
    · intro op hop
      simp [epsAdj] at hop ⊢
      rcases hop with rfl
      exact List.mem_append_left _ (by simp)
    · intro cl hmem
      simp [adjointExpandGapGradClauses] at hmem ⊢
      rcases hmem with rfl
      simp
    · intro op hop
      simp at hop
      rcases hop with rfl
      refine ⟨(EffectLabel.accum, "p", "k", adjointExpandGapGradClauseBody), ?_, rfl⟩
      simp [adjointExpandGapGradClauses]
  have hVarX :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none),
          (adjointExpandGapGradTmp, some Typ.unit)] : LinearCtx)
        (Term.var adjointExpandGapGradX)
        (Typ.tensor DimList.empty)
        []
        ([(adjointExpandGapGradX, none),
          (adjointExpandGapGradSeed, none),
          (adjointExpandGapGradTmp, some Typ.unit)] : LinearCtx) := by
    simpa [List.append_assoc] using
      (HasType.var [] Sigma
        ([] : LinearCtx)
        ([(adjointExpandGapGradSeed, none),
          (adjointExpandGapGradTmp, some Typ.unit)] : LinearCtx)
        adjointExpandGapGradX
        (Typ.tensor DimList.empty))
  have hLet :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.letBind adjointExpandGapGradTmp
          (Term.handle [EffectLabel.accum]
            (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
              (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
            adjointExpandGapGradClauses)
          (Term.var adjointExpandGapGradX))
        (Typ.tensor DimList.empty)
        (EffectRow.union epsHandle [])
        ([(adjointExpandGapGradX, none),
          (adjointExpandGapGradSeed, none)] : LinearCtx) := by
    simpa [epsHandle, List.append_assoc] using
      (HasType.letBind [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, some (Typ.tensor DimList.empty))] : LinearCtx)
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        ([(adjointExpandGapGradX, none),
          (adjointExpandGapGradSeed, none)] : LinearCtx)
        adjointExpandGapGradTmp
        (Term.handle [EffectLabel.accum]
          (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
            (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
          adjointExpandGapGradClauses)
        (Term.var adjointExpandGapGradX)
        Typ.unit
        (Typ.tensor DimList.empty)
        epsHandle
        []
        (some Typ.unit)
        hHandleRaw
        hVarX)
  have hAbsInner :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.abs adjointExpandGapGradSeed (Typ.tensor DimList.empty)
          (Term.letBind adjointExpandGapGradTmp
            (Term.handle [EffectLabel.accum]
              (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
                (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
              adjointExpandGapGradClauses)
            (Term.var adjointExpandGapGradX)))
        (Typ.arrow (Typ.tensor DimList.empty) (Typ.tensor DimList.empty) [])
        []
        ([(adjointExpandGapGradX, none)] : LinearCtx) := by
    exact HasType.abs [] Sigma
      ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty))] : LinearCtx)
      ([(adjointExpandGapGradX, none)] : LinearCtx)
      adjointExpandGapGradSeed
      (Typ.tensor DimList.empty)
      (Typ.tensor DimList.empty)
      []
      (Term.letBind adjointExpandGapGradTmp
        (Term.handle [EffectLabel.accum]
          (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
            (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
          adjointExpandGapGradClauses)
        (Term.var adjointExpandGapGradX))
      none
      hLet
  unfold adjointExpandGapGradReduct adjointExpandGapGradType adjointExpandGapGradT
  exact HasType.abs [] Sigma
    ([] : LinearCtx)
    ([] : LinearCtx)
    adjointExpandGapGradX
    (Typ.tensor DimList.empty)
    (Typ.arrow (Typ.tensor DimList.empty) (Typ.tensor DimList.empty) [])
    []
    (Term.abs adjointExpandGapGradSeed (Typ.tensor DimList.empty)
      (Term.letBind adjointExpandGapGradTmp
        (Term.handle [EffectLabel.accum]
          (adjointTypedFrom adjointExpandGapBody adjointExpandGapGradT adjointExpandGapGradX
            (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
          adjointExpandGapGradClauses)
        (Term.var adjointExpandGapGradX)))
    none
    hAbsInner

/-- Regression witness for the repaired Phase 1 `letBind` / `letpair`
    routing: the same concrete `grad` / `expand` body that previously
    produced an untypable `tgrad` reduct now steps to a reduct that is
    typable for every store typing. This witnesses that the old bug was
    the structural recursion target, not the concrete handler packaging
    around `E-Grad`. -/
theorem grad_preservation_expand_regression :
    HasType [] [] [] adjointExpandGapGradTerm adjointExpandGapGradType [] [] ∧
    Step ⟨([] : Store), adjointExpandGapGradTerm⟩
      ⟨([] : Store), adjointExpandGapGradReduct⟩ ∧
    (∀ Sigma2, HasType [] Sigma2 [] adjointExpandGapGradReduct adjointExpandGapGradType [] []) := by
  refine ⟨adjointExpandGapGradTerm_typed, ?_, ?_⟩
  · simpa [adjointExpandGapGradTerm, adjointExpandGapGradReduct,
      adjointExpandGapGradX, adjointExpandGapGradT, adjointExpandGapGradSeed,
      adjointExpandGapGradCounter, adjointExpandGapGradTmp,
      adjointExpandGapGradClauses, adjointExpandGapGradClauseBody]
      using
        (Step.tgrad ([] : Store) adjointExpandGapGradX
          adjointExpandGapGradT adjointExpandGapGradT adjointExpandGapBody
          (by simp [AdjointSupported, adjointExpandGapBody]))
  · intro Sigma2
    exact adjointExpandGapGradReduct_typed (Sigma := Sigma2)

private def adjointHigherOrderGapTensorT : Typ :=
  Typ.tensor DimList.empty

private def adjointHigherOrderGapFnT : Typ :=
  Typ.arrow adjointHigherOrderGapTensorT adjointHigherOrderGapTensorT []

private def adjointHigherOrderGapBody : Term :=
  Term.snd adjointHigherOrderGapFnT
    (Term.pair
      (Term.abs "y" adjointHigherOrderGapTensorT
        (Term.add (Term.var "x") (Term.var "y")))
      (Term.const 0 DimList.empty))

private theorem adjointHigherOrderGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
      adjointHigherOrderGapBody
      adjointHigherOrderGapTensorT
      []
      ([("x", none)] : LinearCtx) := by
  have hVarX :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.var "x")
        adjointHigherOrderGapTensorT
        []
        ([("x", none), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx) := by
    simpa [adjointHigherOrderGapTensorT, List.append_assoc] using
      (HasType.var (Capability.diff :: []) Sigma
        ([] : LinearCtx)
        ([("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        "x"
        adjointHigherOrderGapTensorT)
  have hVarY :
      HasType (Capability.diff :: []) Sigma
        ([("x", none), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.var "y")
        adjointHigherOrderGapTensorT
        []
        ([("x", none), ("y", none)] : LinearCtx) := by
    simpa [adjointHigherOrderGapTensorT, List.append_assoc] using
      (HasType.var (Capability.diff :: []) Sigma
        ([("x", none)] : LinearCtx)
        ([] : LinearCtx)
        "y"
        adjointHigherOrderGapTensorT)
  have hAdd :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.add (Term.var "x") (Term.var "y"))
        adjointHigherOrderGapTensorT
        []
        ([("x", none), ("y", none)] : LinearCtx) := by
    simpa [EffectRow.union, adjointHigherOrderGapTensorT] using
      (HasType.tadd (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none), ("y", none)] : LinearCtx)
        (Term.var "x")
        (Term.var "y")
        DimList.empty
        []
        []
        hVarX
        hVarY)
  have hAbs :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.abs "y" adjointHigherOrderGapTensorT
          (Term.add (Term.var "x") (Term.var "y")))
        adjointHigherOrderGapFnT
        []
        ([("x", none)] : LinearCtx) := by
    simpa [adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT] using
      (HasType.abs (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none)] : LinearCtx)
        "y"
        adjointHigherOrderGapTensorT
        adjointHigherOrderGapTensorT
        []
        (Term.add (Term.var "x") (Term.var "y"))
        none
        hAdd)
  have hConst :
      HasType (Capability.diff :: []) Sigma
        ([("x", none)] : LinearCtx)
        (Term.const 0 DimList.empty)
        adjointHigherOrderGapTensorT
        []
        ([("x", none)] : LinearCtx) := by
    simpa [adjointHigherOrderGapTensorT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("x", none)] : LinearCtx)
        0
        DimList.empty)
  have hPair :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.pair
          (Term.abs "y" adjointHigherOrderGapTensorT
            (Term.add (Term.var "x") (Term.var "y")))
          (Term.const 0 DimList.empty))
        (Typ.pair adjointHigherOrderGapFnT adjointHigherOrderGapTensorT)
        []
        ([("x", none)] : LinearCtx) := by
    simpa [EffectRow.union, adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none)] : LinearCtx)
        ([("x", none)] : LinearCtx)
        (Term.abs "y" adjointHigherOrderGapTensorT
          (Term.add (Term.var "x") (Term.var "y")))
        (Term.const 0 DimList.empty)
        adjointHigherOrderGapFnT
        adjointHigherOrderGapTensorT
        []
        []
        hAbs
        hConst)
  simpa [adjointHigherOrderGapBody, adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT] using
    (HasType.snd (Capability.diff :: []) Sigma
      ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
      ([("x", none)] : LinearCtx)
      (Term.pair
        (Term.abs "y" adjointHigherOrderGapTensorT
          (Term.add (Term.var "x") (Term.var "y")))
        (Term.const 0 DimList.empty))
      adjointHigherOrderGapFnT
      adjointHigherOrderGapTensorT
      []
      hPair)

private theorem adjointHigherOrderGap_head :
    adjointTypedFrom adjointHigherOrderGapBody adjointHigherOrderGapTensorT "x"
      (Term.var "gs") 0 =
      Term.letpair (freshName "gA" 0) (freshName "gB" 0)
        (Term.pair (zeroCotangent adjointHigherOrderGapFnT) (Term.var "gs"))
        (Term.letBind (freshName "adjA" 0)
          (adjointTypedFrom
            (Term.abs "y" adjointHigherOrderGapTensorT
              (Term.add (Term.var "x") (Term.var "y")))
            adjointHigherOrderGapFnT
            "x"
            (Term.var (freshName "gA" 0))
            (0 + 3))
          (adjointTypedFrom
            (Term.const 0 DimList.empty)
            adjointHigherOrderGapTensorT
            "x"
            (Term.var (freshName "gB" 0))
            (0 + 3))) := by
  simp [adjointHigherOrderGapBody, adjointHigherOrderGapFnT,
    adjointHigherOrderGapTensorT, adjointTypedFrom, zeroCotangent]

private theorem adjointHigherOrderGap_abs_head :
    adjointTypedFrom
      (Term.abs "y" adjointHigherOrderGapTensorT
        (Term.add (Term.var "x") (Term.var "y")))
      adjointHigherOrderGapFnT
      "x"
      (Term.var (freshName "gA" 0))
      (0 + 3) =
      Term.letpair (freshName "gA" (0 + 3)) (freshName "gB" (0 + 3))
        (Term.copy (Term.var (freshName "gA" 0)))
        (Term.letBind (freshName "adjA" ((0 + 3) + 2))
          (adjointTypedFrom
            (Term.var "x")
            adjointHigherOrderGapTensorT
            "x"
            (Term.var (freshName "gA" (0 + 3)))
            (((0 + 3) + 2) + 1))
          (adjointTypedFrom
            (Term.var "y")
            adjointHigherOrderGapTensorT
            "x"
            (Term.var (freshName "gB" (0 + 3)))
            (((0 + 3) + 2) + 1))) := by
  simp [adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT,
    adjointTypedFrom, splitCotangentSeedFrom]

private theorem adjointHigherOrderGap_ctx_no_tensor_gA
    {GammaPre GammaPost : LinearCtx} {ds : DimList}
    (h :
      ([("x", some adjointHigherOrderGapTensorT), ("gs", none),
        (freshName "gA" 0, some Typ.unit),
        (freshName "gB" 0, some adjointHigherOrderGapTensorT)] : LinearCtx) =
        GammaPre ++ [(freshName "gA" 0, some (Typ.tensor ds))] ++ GammaPost) :
    False := by
  cases GammaPre with
  | nil =>
      simp [adjointHigherOrderGapTensorT, freshName] at h
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          simp [adjointHigherOrderGapTensorT, freshName] at h
      | cons b GammaPre =>
          cases GammaPre with
          | nil =>
              simp [adjointHigherOrderGapTensorT] at h
          | cons c GammaPre =>
              cases GammaPre with
              | nil =>
                  simp [adjointHigherOrderGapTensorT, freshName] at h
              | cons d GammaPre =>
                  simp [adjointHigherOrderGapTensorT, freshName] at h

private theorem adjointHigherOrderGap_copy_unit_absurd
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([( "x", some adjointHigherOrderGapTensorT), ("gs", none),
        (freshName "gA" 0, some Typ.unit),
        (freshName "gB" 0, some adjointHigherOrderGapTensorT)] : LinearCtx)
      (Term.copy (Term.var (freshName "gA" 0)))
      t
      eps
      GammaOut) :
    False := by
  obtain ⟨ds, _hteq, hVar⟩ := hasType_copy_inv h
  obtain ⟨GammaPre, GammaPost, hCtx, _hOut⟩ := hasType_var_ctx_inv hVar
  exact adjointHigherOrderGap_ctx_no_tensor_gA hCtx

private theorem hasType_unit_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma GammaOut : LinearCtx}
    {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma Term.unit t eps GammaOut) :
    t = Typ.unit ∧ GammaOut = Gamma := by
  generalize heq : Term.unit = e_in at h
  generalize hteq : t = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | unit _ _ Gamma =>
      cases heq
      cases hteq
      exact ⟨rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq hteq
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem adjointHigherOrderGap_seed_var_inv
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([("x", some adjointHigherOrderGapTensorT),
        ("gs", some adjointHigherOrderGapTensorT)] : LinearCtx)
      (Term.var "gs")
      t
      eps
      GammaOut) :
    t = adjointHigherOrderGapTensorT ∧
      GammaOut =
        ([("x", some adjointHigherOrderGapTensorT),
          ("gs", none)] : LinearCtx) := by
  obtain ⟨GammaPre, GammaPost, hCtx, hOut⟩ := hasType_var_ctx_inv h
  cases GammaPre with
  | nil =>
      simp [adjointHigherOrderGapTensorT] at hCtx
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          cases GammaPost with
          | nil =>
              simp [adjointHigherOrderGapTensorT] at hCtx
              rcases hCtx with ⟨rfl, rfl⟩
              exact ⟨rfl, by simpa [adjointHigherOrderGapTensorT] using hOut⟩
          | cons b GammaPost =>
              simp [adjointHigherOrderGapTensorT] at hCtx
      | cons b GammaPre =>
          simp [adjointHigherOrderGapTensorT] at hCtx

private theorem adjointHigherOrderGap_seed_pair_inv
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([("x", some adjointHigherOrderGapTensorT),
        ("gs", some adjointHigherOrderGapTensorT)] : LinearCtx)
      (Term.pair (zeroCotangent adjointHigherOrderGapFnT) (Term.var "gs"))
      t
      eps
      GammaOut) :
    t = Typ.pair Typ.unit adjointHigherOrderGapTensorT ∧
      GammaOut =
        ([("x", some adjointHigherOrderGapTensorT),
          ("gs", none)] : LinearCtx) := by
  rcases hasType_pair_inv_local h with
    ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, hZero, hSeed, ht, hOut⟩
  have hUnitInv := hasType_unit_inv_local (by
    simpa [zeroCotangent, cotangentType, adjointHigherOrderGapFnT] using hZero)
  have hSeedInv := adjointHigherOrderGap_seed_var_inv (by simpa [hUnitInv.2] using hSeed)
  cases hUnitInv.1
  cases hSeedInv.1
  cases ht
  exact ⟨rfl, by simpa [hSeedInv.2] using hOut⟩

/-- Even after switching products/projections to typed cotangent seeds,
    the typed transform is still false on higher-order `grad` bodies:
    the current `abs` branch passes a non-tensor seed straight into the
    function body, so a later tensor primitive can still force an
    ill-typed `copy`. This witnesses that the next honest theorem
    surface must carry an explicit supported-fragment premise, not just
    a typed seed. -/
theorem adjointTyped_higherOrder_counterexample :
    (∀ Sigma,
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        adjointHigherOrderGapBody
        adjointHigherOrderGapTensorT
        []
        ([("x", none)] : LinearCtx)) ∧
    (∀ Sigma, ¬ ∃ eps GammaOut,
      HasType [] Sigma
        ([("x", some adjointHigherOrderGapTensorT),
          ("gs", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (adjointTypedFrom adjointHigherOrderGapBody
          adjointHigherOrderGapTensorT "x" (Term.var "gs") 0)
        Typ.unit
        eps
        GammaOut) := by
  refine ⟨?_, ?_⟩
  · intro Sigma
    exact adjointHigherOrderGapBody_typed (Sigma := Sigma)
  · intro Sigma
    intro h
    rcases h with ⟨eps, GammaOut, hAdj⟩
    rw [adjointHigherOrderGap_head] at hAdj
    rcases hasType_letpair_inv hAdj with
      ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY,
        hSeedPair, hBody, _hOut⟩
    have hSeedInv := adjointHigherOrderGap_seed_pair_inv hSeedPair
    cases hSeedInv.1
    cases hSeedInv.2
    rcases hasType_letBind_inv_local hBody with
      ⟨GammaMid, GammaEnd, tAdj, epsAdj, epsRest, slotAdj,
        hAdjAbs, _hRest, _hOut⟩
    rw [adjointHigherOrderGap_abs_head] at hAdjAbs
    rcases hasType_letpair_inv hAdjAbs with
      ⟨GammaCopy, GammaCopyOut, tCopy1, tCopy2, epsCopy, epsBody,
        slotCopy1, slotCopy2, hCopy, _hBody, _hOut⟩
    exact adjointHigherOrderGap_copy_unit_absurd hCopy

private def adjointSndGapDim : Dim :=
  Dim.named "dSnd"

private def adjointSndGapBody : Term :=
  Term.snd (Typ.tensor (ins DimList.empty adjointSndGapDim))
    (Term.pair
      (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
      (Term.const 0 DimList.empty))

private theorem adjointSndGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
      adjointSndGapBody
      (Typ.tensor DimList.empty)
      []
      ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
  have hExpandArg :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Typ.tensor DimList.empty)
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.const (Capability.diff :: []) Sigma _ 0 DimList.empty
  have hExpand :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
        (Typ.tensor (ins DimList.empty adjointSndGapDim))
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.texpand (Capability.diff :: []) Sigma
      _ _ _ DimList.empty adjointSndGapDim []
      hExpandArg
  have hConst :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Typ.tensor DimList.empty)
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.const (Capability.diff :: []) Sigma _ 0 DimList.empty
  have hPair :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.pair
          (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
          (Term.const 0 DimList.empty))
        (Typ.pair
          (Typ.tensor (ins DimList.empty adjointSndGapDim))
          (Typ.tensor DimList.empty))
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    simpa [EffectRow.union] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
        ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
        ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
        (Term.const 0 DimList.empty)
        (Typ.tensor (ins DimList.empty adjointSndGapDim))
        (Typ.tensor DimList.empty)
        []
        []
        hExpand
        hConst)
  exact HasType.snd (Capability.diff :: []) Sigma
    ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
    ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
    (Term.pair
      (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
      (Term.const 0 DimList.empty))
    (Typ.tensor (ins DimList.empty adjointSndGapDim))
    (Typ.tensor DimList.empty)
    []
    hPair

private def adjointHandleSeedGapT : Typ :=
  Typ.tensor DimList.empty

private def adjointHandleSeedGapClauseBody : Term :=
  Term.pair
    (Term.const 0 DimList.empty)
    (Term.const 0 DimList.empty)

private def adjointHandleSeedGapForwardBody : Term :=
  Term.letBind "u"
    (Term.perform EffectLabel.resource Term.unit)
    adjointHandleSeedGapClauseBody

private def adjointHandleSeedGapClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.resource, "p", "k", adjointHandleSeedGapClauseBody)]

private def adjointHandleSeedGapBody : Term :=
  Term.snd adjointHandleSeedGapT
    (Term.handle [EffectLabel.resource]
      adjointHandleSeedGapForwardBody
      adjointHandleSeedGapClauses)

private theorem adjointHandleSeedGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      adjointHandleSeedGapBody
      adjointHandleSeedGapT
      []
      ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
  let tPair : Typ := Typ.pair adjointHandleSeedGapT adjointHandleSeedGapT
  let tK : Typ := Typ.arrow Typ.unit tPair []
  have hUnit :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        Term.unit Typ.unit []
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    exact HasType.unit (Capability.diff :: []) Sigma _
  have hPerform :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        (Term.perform EffectLabel.resource Term.unit)
        Typ.unit
        [EffectLabel.resource]
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    exact HasType.perform (Capability.diff :: []) Sigma _ _
      EffectLabel.resource Term.unit Typ.unit Typ.unit [] hUnit
      (by simp [OpSigMatch, opArgType, opRetType])
  have hConstX :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        []
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx) := by
    simpa [adjointHandleSeedGapT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        0 DimList.empty)
  have hPairBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        adjointHandleSeedGapClauseBody
        tPair
        []
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx) := by
    simpa [adjointHandleSeedGapClauseBody, tPair, EffectRow.union] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        adjointHandleSeedGapT
        []
        []
        hConstX
        hConstX)
  have hForwardBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        adjointHandleSeedGapForwardBody
        tPair
        [EffectLabel.resource]
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    simpa [adjointHandleSeedGapForwardBody, tPair] using
      (HasType.letBind (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        "u"
        (Term.perform EffectLabel.resource Term.unit)
        adjointHandleSeedGapClauseBody
        Typ.unit
        tPair
        [EffectLabel.resource]
        []
        (some Typ.unit)
        hPerform
        hPairBody)
  have hConstP :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        []
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx) := by
    simpa [adjointHandleSeedGapT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        0 DimList.empty)
  have hClauseBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        adjointHandleSeedGapClauseBody
        tPair
        []
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx) := by
    simpa [adjointHandleSeedGapClauseBody, tPair, EffectRow.union] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        adjointHandleSeedGapT
        []
        []
        hConstP
        hConstP)
  have hClauses :
      ClausesTyped (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        tPair
        []
        adjointHandleSeedGapClauses := by
    exact ClausesTyped.cons (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      tPair
      Typ.unit
      Typ.unit
      []
      EffectLabel.resource
      "p"
      "k"
      adjointHandleSeedGapClauseBody
      []
      (some Typ.unit)
      (some tK)
      (by simp [OpSigMatch, opArgType, opRetType])
      hClauseBody
      (ClausesTyped.nil (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        tPair
        [])
  have hHandle :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        (Term.handle [EffectLabel.resource]
          adjointHandleSeedGapForwardBody
          adjointHandleSeedGapClauses)
        tPair
        []
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    exact HasType.handle (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      adjointHandleSeedGapForwardBody
      adjointHandleSeedGapClauses
      tPair
      [EffectLabel.resource]
      [EffectLabel.resource]
      hForwardBody
      (by intro op hop; simp at hop; rcases hop with rfl; simp)
      (by intro cl hmem; simp [adjointHandleSeedGapClauses] at hmem ⊢; rcases hmem with rfl; simp)
      (by
        intro op hop
        simp at hop
        rcases hop with rfl
        exact ⟨(EffectLabel.resource, "p", "k", adjointHandleSeedGapClauseBody),
          by simp [adjointHandleSeedGapClauses], rfl⟩)
      hClauses
  simpa [adjointHandleSeedGapBody, adjointHandleSeedGapT] using
    (HasType.snd (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      (Term.handle [EffectLabel.resource]
        adjointHandleSeedGapForwardBody
        adjointHandleSeedGapClauses)
      adjointHandleSeedGapT
      adjointHandleSeedGapT
      []
      hHandle)

private theorem adjointHandleSeedGap_head :
    adjointFrom adjointHandleSeedGapBody "x" (Term.var "gs") 0 =
      Term.letpair (freshName "gA" 0) (freshName "gB" 0)
        (Term.copy (Term.pair (Term.const 0 DimList.empty) (Term.var "gs")))
        (Term.letBind (freshName "adjHb" 0)
          (adjointFrom adjointHandleSeedGapClauseBody "x"
            (Term.var (freshName "gA" 0)) (0 + 3))
          (adjointFrom adjointHandleSeedGapForwardBody "x"
            (Term.var (freshName "gB" 0)) (0 + 3))) := by
  simp [adjointHandleSeedGapBody, adjointHandleSeedGapT,
    adjointHandleSeedGapForwardBody, adjointHandleSeedGapClauses,
    adjointHandleSeedGapClauseBody, adjointFrom, adjointClausesFrom,
    zeroCotangent]

/-- The old product/projection false witness is gone from the transform
    itself, but the current public theorem surface is still false:
    `adjointFrom`'s legacy handler path splits every clause seed with
    tensor-only `copy`, so feeding a structured cotangent seed through
    `snd` into `handle` produces an untypable adjoint term before the
    `mul` case is even in play. The staged `adjointTypedFrom` /
    `adjointTypedClausesFrom` path in `AdjointTransform.lean` is the
    intended repair. -/
theorem adjoint_handle_product_seed_counterexample :
    (∀ Sigma,
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        adjointHandleSeedGapBody
        adjointHandleSeedGapT
        []
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)) ∧
    (∀ Sigma, ¬ ∃ eps GammaOut,
      HasType [] Sigma
        ([("x", some adjointHandleSeedGapT), ("gs", some adjointHandleSeedGapT)] : LinearCtx)
        (adjointFrom adjointHandleSeedGapBody "x" (Term.var "gs") 0)
        Typ.unit
        eps
        GammaOut) := by
  refine ⟨?_, ?_⟩
  · intro Sigma
    exact adjointHandleSeedGapBody_typed (Sigma := Sigma)
  · intro Sigma
    intro h
    rcases h with ⟨eps, GammaOut, hAdj⟩
    rw [adjointHandleSeedGap_head] at hAdj
    rcases hasType_letpair_inv hAdj with
      ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY,
        hCopy, _hBody, _hOut, _hSub⟩
    obtain ⟨ds, _hteq, hPairAsTensor⟩ := hasType_copy_inv hCopy
    exact hasType_pair_tensor_absurd hPairAsTensor

/-- Counter-threaded public typing theorem for `adjointFrom`. This is
    the theorem preservation should use when the operational rule picks
    a start counter above the exposed binder-name lengths.

    The freshness premise requires that no counter-indexed adjoint
    name (`freshName base m` for `m ≥ n` and `base ∈ adjointBases`)
    collides with a name already in `Γ ++ [(x, _), (gs, _)]`.

    Current branch note: the old monomorphic tensor-seed mismatch for
    `pair` / `fst` / `snd` is repaired in `AdjointTransform.lean`, but
    the current public surface here is still false for handled product
    paths: `adjointFrom` still routes `handle` through the legacy
    tensor-only `adjointClausesFrom`, and
    `adjoint_handle_product_seed_counterexample` shows that can force an
    ill-typed `copy` on a structured seed before `mul` is involved. The
    next real proof step is therefore stronger than “generalize the
    helper”: switch the public surface to the staged typed companion
    transform (`adjointTypedFrom` / `adjointTypedClausesFrom`), then
    reprove the helper there while separately closing the remaining
    `mul` operand rebasing/effect-row debt. -/
theorem adjointFrom_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (n : Nat) (slot : Option Typ)
    (_h_e : HasType (Capability.diff :: Delta) Sigma
                    (Gamma ++ [(x, some (Typ.tensor ds))])
                    e (Typ.tensor dsOut) eps
                    (Gamma ++ [(x, slot)]))
    (_h_compat : subsetEffRow eps DiffCompat = true)
    (h_fresh_full : AdjointNamesFresh n
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (_h_fresh_small : AdjointNamesFresh n (Gamma ++ [(x, some (Typ.tensor ds))])) :
    HasType Delta Sigma
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
            (adjointFrom e x (Term.var gs) n)
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
  have hSeed :
      HasType Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (Term.var gs) (Typ.tensor dsOut) []
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    simpa [List.append_assoc] using
      (HasType.var Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds))]) ([] : LinearCtx)
        gs (Typ.tensor dsOut))
  have hFreshOut : AdjointNamesFresh n
      (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    simpa [AdjointNamesFresh, linearCtxDom] using h_fresh_full
  have hRaw :
      HasType Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (adjointFrom e x (Term.var gs) n)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    exact
      (adjoint_typed_aux Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)])
        dsOut [] x n e (Term.var gs)
        hSeed h_fresh_full hFreshOut)
  exact HasType.subEff Delta Sigma
    (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
    (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)])
    (adjointFrom e x (Term.var gs) n)
    Typ.unit
    (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
    (EffectRow.union eps [EffectLabel.accum])
    hRaw
    (subEff_accum_into_grad eps)

/-- The adjoint transformation preserves typing. Closed as the
    `n = 0` corollary of `adjointFrom_preserves_typing`.

    The freshness premise requires that no counter-indexed adjoint
    name (`freshName base m` for `m ≥ 0` and `base ∈ adjointBases`)
    collides with a name already in `Γ ++ [(x, _), (gs, _)]`. In
    practice the public-surface caller chooses `x`, `gs`, and `Γ`
    outside the adjoint name namespace, so the premise is discharged
    at each call site. -/
theorem adjoint_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (slot : Option Typ)
    (h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, some (Typ.tensor ds))])
                   e (Typ.tensor dsOut) eps
                   (Gamma ++ [(x, slot)]))
    (h_compat : subsetEffRow eps DiffCompat = true)
    (h_fresh_full : AdjointNamesFresh 0
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (h_fresh_small : AdjointNamesFresh 0 (Gamma ++ [(x, some (Typ.tensor ds))])) :
    HasType Delta Sigma
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
            (adjoint e x (Term.var gs))
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
  simpa [adjoint] using
    (adjointFrom_preserves_typing Delta Sigma Gamma
      x gs ds dsOut e eps 0 slot h_e h_compat h_fresh_full h_fresh_small)

/-- Typed public surface for the staged adjoint transform. This is the
    theorem `T-Grad` now needs because the operational reduct uses
    `adjointTypedFrom`, not the legacy `adjointFrom` shim. -/
theorem adjointTypedFrom_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (n : Nat) (slot : Option Typ)
    (_h_e : HasType (Capability.diff :: Delta) Sigma
                    (Gamma ++ [(x, some (Typ.tensor ds))])
                    e (Typ.tensor dsOut) eps
                    (Gamma ++ [(x, slot)]))
    (_h_compat : subsetEffRow eps DiffCompat = true)
    (_h_supp : AdjointSupported e)
    (_h_fresh_full : AdjointNamesFresh n
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (_h_fresh_small : AdjointNamesFresh n (Gamma ++ [(x, some (Typ.tensor ds))])) :
    HasType Delta Sigma
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
            (adjointTypedFrom e (Typ.tensor dsOut) x (Term.var gs) n)
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
  sorry

theorem adjointTyped_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (slot : Option Typ)
    (h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, some (Typ.tensor ds))])
                   e (Typ.tensor dsOut) eps
                   (Gamma ++ [(x, slot)]))
    (h_compat : subsetEffRow eps DiffCompat = true)
    (h_supp : AdjointSupported e)
    (h_fresh_full : AdjointNamesFresh 0
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (h_fresh_small : AdjointNamesFresh 0 (Gamma ++ [(x, some (Typ.tensor ds))])) :
    HasType Delta Sigma
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
            (adjointTyped e (Typ.tensor dsOut) x (Term.var gs))
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
  simpa [adjointTyped] using
    (adjointTypedFrom_preserves_typing Delta Sigma Gamma
      x gs ds dsOut e eps 0 slot h_e h_compat h_supp h_fresh_full h_fresh_small)

end LaCaDiLE
