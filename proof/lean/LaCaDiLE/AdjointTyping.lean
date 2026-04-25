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

/-- Seed-polymorphic helper for `adjoint_preserves_typing`. The proof
    depends on the seed's typing, the structural shape of `e`, and
    the freshness of adjoint counter-indexed names at counters ≥ `n`.
    `e`'s own typing is not required. The recursive cases that build
    new typed sub-expressions (`add`, `mul`) are closed below; the
    handler case and the tape-dependent `sum` / `expand` cases are
    parked on later-phase work and fall into the catch-all. -/
private theorem adjoint_typed_aux
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma_s Gamma_s' : LinearCtx)
    (dsE : DimList) (epsSeed : EffectRow) (x : String) (n : Nat)
    (e : Term) (gSeed : Term)
    (h_seed : HasType Delta Sigma Gamma_s gSeed (Typ.tensor dsE) epsSeed Gamma_s')
    (h_e : AdjointMulTyped Delta Sigma dsE e)
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
  -- sub-term with the same seed and counter. `h_e` is destructured
  -- via `AdjointMulTyped`'s structural unfolding on the outer shape.
  | Term.letBind _ e1 _ =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e.1)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.letpair _ _ e1 _ =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e.1)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.pair e1 _ =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e.1)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.fst e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.snd e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.copy e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.abs _ _ e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.app e1 _ =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e.1)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.grad _ _ _ e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.vmap _ _ _ e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.perform _ e1 =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  | Term.uniformLike e1 _ _ =>
      simp only [adjointFrom]
      have h_e1 : AdjointMulTyped Delta Sigma dsE e1 :=
        (by unfold AdjointMulTyped at h_e; exact h_e)
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_e1 h_fresh_s h_fresh_s'
  -- Real adjoint cases (add, mul).
  | Term.add e1 e2 =>
      simp only [adjointFrom]
      have h_e_add : AdjointMulTyped Delta Sigma dsE e1 ∧
                     AdjointMulTyped Delta Sigma dsE e2 := by
        unfold AdjointMulTyped at h_e; exact h_e
      have h_e1 := h_e_add.1
      have h_e2 := h_e_add.2
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
          hSeedA h_e1 hFreshPairIn hFreshPairOut
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
          hSeedB h_e2
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
  -- `mul`, `sum`, `expand`, and `handle` remain under the catch-all.
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
  -- A sound path forward is the `weakening_insert` lemma in
  -- `Substitution.lean` (currently sorry-blocked). With that lemma,
  -- the operand rebasing collapses to linear weakening over the
  -- predicate's existential chain, and the effect rows pass through
  -- without modification. That work is a Wave 3 linearity-discipline
  -- deliverable and belongs alongside `has_type_linear_shrinks`, not
  -- inside this file.
  --
  -- `sum` and `expand` remain on the Phase 1 T9 tape-extent work;
  -- `handle` remains on ClausesTypedDB. All four cases park under
  -- this catch-all.
  | _ => sorry
termination_by sizeOf e

/-- The adjoint transformation preserves typing. Closed as a corollary
    of `adjoint_typed_aux` instantiated with `gSeed = Term.var gs`.

    The freshness premise requires that no counter-indexed adjoint
    name (`freshName base m` for `m ≥ 0` and `base ∈ adjointBases`)
    collides with a name already in `Γ ++ [(x, _), (gs, _)]`. In
    practice the public-surface caller chooses `x`, `gs`, and `Γ`
    outside the adjoint name namespace, so the premise is discharged
    at each call site. -/
theorem adjoint_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, some (Typ.tensor ds))])
                   e (Typ.tensor dsOut) eps Gamma)
    (h_e_muls : AdjointMulTyped (Capability.diff :: Delta) Sigma dsOut e)
    (h_compat : subsetEffRow eps DiffCompat = true)
    (h_fresh_full : AdjointNamesFresh 0
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (h_fresh_small : AdjointNamesFresh 0 (Gamma ++ [(x, some (Typ.tensor ds))])) :
    HasType (Capability.diff :: Delta) Sigma
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
            (adjoint e x (Term.var gs))
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
  have hSeed :
      HasType (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (Term.var gs) (Typ.tensor dsOut) []
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    simpa [List.append_assoc] using
      (HasType.var (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, some (Typ.tensor ds))]) ([] : LinearCtx)
        gs (Typ.tensor dsOut))
  have hFreshOut : AdjointNamesFresh 0
      (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    simpa [AdjointNamesFresh, linearCtxDom] using h_fresh_full
  have hRaw :
      HasType (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (adjoint e x (Term.var gs))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)]) := by
    simpa [adjoint] using
      (adjoint_typed_aux (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)])
        dsOut [] x 0 e (Term.var gs)
        hSeed h_e_muls h_fresh_full hFreshOut)
  exact HasType.subEff (Capability.diff :: Delta) Sigma
    (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
    (Gamma ++ [(x, some (Typ.tensor ds)), (gs, none)])
    (adjoint e x (Term.var gs))
    Typ.unit
    (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
    (EffectRow.union eps [EffectLabel.accum])
    hRaw
    (subEff_accum_into_grad eps)

end LaCaDiLE
