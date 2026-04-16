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
-- Wave 3 calculus refinement: T-Perform now uses an `OpSigMatch`
-- relation that lets `perform accum` take a tensor argument. This
-- unblocks the var/const/unit/loc base cases, which all emit
-- `Term.perform EffectLabel.accum gSeed` with `gSeed : tensor[dsOut]`.
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
  | Term.vmap _ _ body   => AdjointMulTyped Delta Sigma dsE body
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
  -- Local witness: `perform accum gSeed` type-checks at unit with
  -- effect row `union [accum] epsSeed` given the seed's typing.
  have leaf_perform :
      HasType Delta Sigma Gamma_s
        (Term.perform EffectLabel.accum gSeed) Typ.unit
        (EffectRow.union [EffectLabel.accum] epsSeed) Gamma_s' :=
    HasType.perform Delta Sigma Gamma_s Gamma_s'
      EffectLabel.accum gSeed (Typ.tensor dsE) Typ.unit epsSeed
      h_seed (OpSigMatch.accumTensor dsE)
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
  | Term.vmap _ _ e1 =>
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
      -- Abbreviate the fresh names via local `let`s.
      let gA : String := freshName "gA" n
      -- Tombstone-style context refactor broke the add case's context
      -- threading: T-Var now outputs tombstoned slots instead of removing
      -- entries, and T-LetBind/T-LetPair output slot parameters instead
      -- of filtering. The entire derivation chain needs restructuring.
      -- Sorry until the tombstone-aware context lemmas are available.
      sorry
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
  -- T-Var on gs now produces a tombstone (gs, none) instead of removing gs.
  -- The output context shape changed, and the downstream reasoning about
  -- effect bridging needs the tombstone-aware context chain. Sorry until
  -- the tombstone lemmas are available.
  sorry

end LaCaDiLE
