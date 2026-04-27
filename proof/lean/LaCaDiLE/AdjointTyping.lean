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
  | Term.fst e           => AdjointMulTyped Delta Sigma dsE e
  | Term.snd e           => AdjointMulTyped Delta Sigma dsE e
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
  | Term.pair e1 _ =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.fst e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.snd e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.copy e1 =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
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
          (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
    (some (Typ.tensor DimList.empty)) adjointExpandGapBody_typed (by
      simp [subsetEffRow, DiffCompat])

private theorem adjointExpandGapGradReduct_typed
    {Sigma : StoreTyp} :
    HasType [] Sigma [] adjointExpandGapGradReduct adjointExpandGapGradType [] [] := by
  let epsAdj : EffectRow := EffectRow.union [EffectLabel.accum] ([] : EffectRow)
  let epsHandle : EffectRow := EffectRow.removeOps epsAdj [EffectLabel.accum]
  have hAdj :
      HasType [] Sigma
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, some (Typ.tensor DimList.empty))] : LinearCtx)
        (adjointFrom adjointExpandGapBody adjointExpandGapGradX
          (Term.var adjointExpandGapGradSeed) adjointExpandGapGradCounter)
        Typ.unit
        epsAdj
        ([(adjointExpandGapGradX, some (Typ.tensor DimList.empty)),
          (adjointExpandGapGradSeed, none)] : LinearCtx) := by
    simpa [epsAdj] using
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
          (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
      (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
            (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
          (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
              (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
          (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
          (adjointFrom adjointExpandGapBody adjointExpandGapGradX
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
          adjointExpandGapGradT adjointExpandGapGradT adjointExpandGapBody)
  · intro Sigma2
    exact adjointExpandGapGradReduct_typed (Sigma := Sigma2)

private def adjointSndGapDim : Dim :=
  Dim.named "dSnd"

private def adjointSndGapBody : Term :=
  Term.snd
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

private theorem adjointSndGapVarInv
    {Delta : CapCtx} {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h :
      HasType Delta Sigma
        ([("x", some (Typ.tensor DimList.empty)),
          ("gs", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.var "gs")
        t
        eps
        GammaOut) :
    t = Typ.tensor DimList.empty := by
  generalize hctx :
      ([("x", some (Typ.tensor DimList.empty)),
        ("gs", some (Typ.tensor DimList.empty))] : LinearCtx) = GammaIn at h
  generalize heq : Term.var "gs" = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | var Delta Sigma Gamma_pre Gamma_post y ty =>
      cases heq
      cases Gamma_pre with
      | nil =>
          cases Gamma_post with
          | nil =>
              simp at hctx
          | cons hd tl =>
              cases tl with
              | nil =>
                  cases hd
                  simp at hctx
              | cons hd2 tl2 =>
                  simp at hctx
      | cons hd tl =>
          cases tl with
          | nil =>
              cases Gamma_post with
              | nil =>
                  cases hd
                  simp at hctx
                  subst_vars
                  exact hctx.2.symm
              | cons hd2 tl2 =>
                  simp at hctx
          | cons hd2 tl2 =>
              simp at hctx
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih hctx heq
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem adjointSndGapAdjoint_untypable
    {Sigma : StoreTyp} {n : Nat} :
    ¬ ∃ eps GammaOut,
      HasType [] Sigma
        ([("x", some (Typ.tensor DimList.empty)),
          ("gs", some (Typ.tensor DimList.empty))] : LinearCtx)
        (adjointFrom adjointSndGapBody "x" (Term.var "gs") n)
        Typ.unit
        eps
        GammaOut := by
  intro h
  rcases h with ⟨eps, GammaOut, hAdj⟩
  have hLet :
      HasType [] Sigma
        ([("x", some (Typ.tensor DimList.empty)),
          ("gs", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.letBind (freshName "adjA" n)
          (Term.sum (Term.var "gs") adjointSndGapDim)
          (Term.perform EffectLabel.accum Term.unit))
        Typ.unit
        eps
        GammaOut := by
    simpa [adjointSndGapBody, adjointFrom, adjointLeaf] using hAdj
  rcases hasType_letBind_inv_local hLet with
    ⟨Gamma2, _Gamma3, t1, eps1, _eps2, _slot, hSum, _hBody, _hOut⟩
  obtain ⟨ds, _htEq, hMem, hVar⟩ := hasType_sum_inv_local hSum
  have hDs : ds = DimList.empty := by
    simpa using adjointSndGapVarInv hVar
  subst hDs
  change adjointSndGapDim ∈ ([] : List Dim) at hMem
  simp at hMem

theorem adjoint_snd_expand_false_witness :
    HasType (Capability.diff :: []) [] [("x", some (Typ.tensor DimList.empty))]
      adjointSndGapBody (Typ.tensor DimList.empty) []
      [("x", some (Typ.tensor DimList.empty))] ∧
    ∀ n,
      ¬ ∃ eps GammaOut,
        HasType [] [] 
          ([("x", some (Typ.tensor DimList.empty)),
            ("gs", some (Typ.tensor DimList.empty))] : LinearCtx)
          (adjointFrom adjointSndGapBody "x" (Term.var "gs") n)
          Typ.unit
          eps
          GammaOut := by
  refine ⟨adjointSndGapBody_typed, ?_⟩
  intro n
  exact adjointSndGapAdjoint_untypable (Sigma := []) (n := n)

/-- Counter-threaded public typing theorem for `adjointFrom`. This is
    the theorem preservation should use when the operational rule picks
    a start counter above the exposed binder-name lengths.

    The freshness premise requires that no counter-indexed adjoint
    name (`freshName base m` for `m ≥ n` and `base ∈ adjointBases`)
    collides with a name already in `Γ ++ [(x, _), (gs, _)]`.

    Current branch note: this statement is still admitted. The old
    concrete `letBind` / `letpair` routing bug is repaired, but Lean
    now also contains a typed `snd` / `pair` / `expand` false witness.
    So the remaining proof debt is not only a missing `expand` premise:
    the current monomorphic tensor-seed transform is itself too weak for
    product/projection paths. The next honest fix is either a typed
    cotangent-seed transform or an explicit restriction/normalization of
    grad bodies to the product-free fragment, while `mul` still needs
    the operand rebasing/effect-row repair documented above. -/
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

end LaCaDiLE
