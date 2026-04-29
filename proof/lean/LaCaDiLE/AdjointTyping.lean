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

/- The legacy `adjointFrom` theorem family was removed from the live
   proof surface once handled product-seed counterexamples showed that
   its tensor-only clause threading is false. `Preservation` now uses
   the staged typed companion below. -/

private theorem hasType_var_with_suffix
    (Delta : CapCtx) (Sigma : StoreTyp)
    (GammaPre GammaPost : LinearCtx)
    (x : String) (t : Typ) :
    HasType Delta Sigma
      (GammaPre ++ [(x, some t)] ++ GammaPost)
      (Term.var x)
      t
      []
      (GammaPre ++ [(x, none)] ++ GammaPost) := by
  simpa [List.append_assoc] using
    (HasType.var Delta Sigma GammaPre GammaPost x t)

private theorem adjointTypedShape_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) :
    ∀ {t : Typ} {e : Term},
      AdjointTypedShape (Capability.diff :: Delta) Sigma t e →
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat},
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t) [] (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedFrom e t x gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix) := by
  intro t e hShape
  induction hShape using AdjointTypedShape.rec
    (motive_2 := fun t clauses _ =>
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {body : Term} {gSeed : Term} {n : Nat},
        (∀ {Gamma_s0 Gamma_s0' suffix0 : LinearCtx} {x0 : String} {gSeed0 : Term} {n0 : Nat},
          HasType Delta Sigma
            (Gamma_s0 ++ suffix0) gSeed0 (cotangentType t) [] (Gamma_s0' ++ suffix0) →
          HasType Delta Sigma
            (Gamma_s0 ++ suffix0)
            (adjointTypedFrom body t x0 gSeed0 n0)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s0' ++ suffix0)) →
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t) [] (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedClausesFrom clauses x body t gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
  with
  | var =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | const =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | unit =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | loc =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | letBind hBody ih =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := gSeed) (n := n) h_seed)
  | letpair hBody ih =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := gSeed) (n := n) h_seed)
  | add =>
      rename_i ds e1 e2 h1 h2 ih1 ih2
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      sorry
  | mul =>
      rename_i ds e1 e2 hMul h1 h2 ih1 ih2
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      sorry
  | sum =>
      rename_i ds d e hmem hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hExpand :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.expand gSeed d)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma_s' ++ suffix) := by
        have := HasType.texpand Delta Sigma
          (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          gSeed
          (rem ds d)
          d
          []
          h_seed
        simpa [cotangentType, ins_rem_eq_of_mem hmem] using this
      simpa [adjointTypedFrom, ins_rem_eq_of_mem hmem] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := Term.expand gSeed d) (n := n) hExpand)
  | expand =>
      rename_i ds d e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hmem : d ∈ ins ds d := by
        refine Quotient.inductionOn ds ?_
        intro l
        change d ∈ (d :: l)
        simp
      have hSum :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.sum gSeed d)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma_s' ++ suffix) := by
        have := HasType.tsum Delta Sigma
          (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          gSeed
          (ins ds d)
          d
          []
          h_seed
          hmem
        simpa [cotangentType, rem_ins_eq] using this
      simpa [adjointTypedFrom, rem_ins_eq] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := Term.sum gSeed d) (n := n) hSum)
  | copy =>
      rename_i ds e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      sorry
  | pair =>
      rename_i t1 t2 e1 e2 h1 h2 ih1 ih2
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      sorry
  | fst =>
      rename_i t1 t2 e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hZero :
          HasType Delta Sigma
            (Gamma_s' ++ suffix)
            (zeroCotangent t2)
            (cotangentType t2)
            []
            (Gamma_s' ++ suffix) :=
        zeroCotangent_typed Delta Sigma (Gamma_s' ++ suffix) t2
      have hPairSeed :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.pair gSeed (zeroCotangent t2))
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma_s' ++ suffix) := by
        simpa [cotangentType] using
          (HasType.tpair Delta Sigma
            (Gamma_s ++ suffix) (Gamma_s' ++ suffix) (Gamma_s' ++ suffix)
            gSeed (zeroCotangent t2)
            (cotangentType t1) (cotangentType t2)
            [] [] h_seed hZero)
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x)
          (gSeed := Term.pair gSeed (zeroCotangent t2))
          (n := n) hPairSeed)
  | snd =>
      rename_i t1 t2 e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hZero :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (zeroCotangent t1)
            (cotangentType t1)
            []
            (Gamma_s ++ suffix) :=
        zeroCotangent_typed Delta Sigma (Gamma_s ++ suffix) t1
      have hPairSeed :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.pair (zeroCotangent t1) gSeed)
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma_s' ++ suffix) := by
        simpa [cotangentType] using
          (HasType.tpair Delta Sigma
            (Gamma_s ++ suffix) (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
            (zeroCotangent t1) gSeed
            (cotangentType t1) (cotangentType t2)
            [] [] hZero h_seed)
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x)
          (gSeed := Term.pair (zeroCotangent t1) gSeed)
          (n := n) hPairSeed)
  | handle =>
      rename_i t epsH body clauses hBody hClauses ihBody ihClauses
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (ihClauses (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (body := body) (gSeed := gSeed) (n := n)
          (fun {Gamma_s0 Gamma_s0' suffix0 x0 gSeed0 n0} h_seed0 =>
            ihBody (Gamma_s := Gamma_s0) (Gamma_s' := Gamma_s0')
              (suffix := suffix0) (x := x0) (gSeed := gSeed0) (n := n0) h_seed0)
          h_seed)
  | perform =>
      rename_i op e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hSeedArg :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            gSeed
            (cotangentType (opArgType op))
            []
            (Gamma_s' ++ suffix) := by
        cases op <;> simpa [opArgType, cotangentType] using h_seed
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := gSeed) (n := n) hSeedArg)
  | nil =>
      rename_i Gamma_s Gamma_s' suffix x body gSeed n ihBody h_seed
      simpa [adjointTypedClausesFrom] using
        (ihBody (Gamma_s0 := Gamma_s) (Gamma_s0' := Gamma_s')
          (suffix0 := suffix) (x0 := x) (gSeed0 := gSeed) (n0 := n) h_seed)
  | cons =>
      rename_i t op xv kv hb rest hHead hRest ihHead ihRest
      rename_i Gamma_s Gamma_s' suffix x body gSeed n ihBody h_seed
      sorry

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
  have hShape : AdjointTypedShape (Capability.diff :: Delta) Sigma (Typ.tensor dsOut) e :=
    adjointTypedShape_of_typed _h_e _h_supp
  have hAdj :
      HasType Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (adjointTypedFrom e (Typ.tensor dsOut) x (Term.var gs) n)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    simpa using
      (adjointTypedShape_preserves_typing Delta Sigma hShape
        (Gamma_s := Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (Gamma_s' := Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)])
        (suffix := ([] : LinearCtx))
        (x := x) (gSeed := Term.var gs) (n := n)
        (by
          simpa [List.append_assoc] using
            (HasType.var Delta Sigma
              (Gamma ++ [(x, some (Typ.tensor ds))])
              ([] : LinearCtx)
              gs
              (Typ.tensor dsOut))))
  exact HasType.subEff Delta Sigma
    (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
    (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)])
    (adjointTypedFrom e (Typ.tensor dsOut) x (Term.var gs) n)
    Typ.unit
    (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
    (EffectRow.union eps [EffectLabel.accum])
    hAdj
    (subEff_accum_into_grad eps)

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
