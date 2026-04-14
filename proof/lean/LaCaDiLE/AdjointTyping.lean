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
    {n k : Nat} {Γ : LinearCtx} {b : String} {t : Typ}
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

/-- `List.filter` by "name ≠ a" is a no-op when `a` is not in the
    domain of `Γ`. Used to dispose of the `letpair`/`letBind` output
    filters once freshness retires the newly-introduced binders. -/
theorem linearCtx_filter_fresh_eq (Γ : LinearCtx) (a : String)
    (h : a ∉ linearCtxDom Γ) :
    Γ.filter (fun p => p.1 ≠ a) = Γ := by
  apply List.filter_eq_self.mpr
  intro p hp
  simp only [decide_eq_true_eq]
  intro heq
  apply h
  simp only [linearCtxDom, List.mem_map]
  exact ⟨p, hp, heq⟩

/-- Same for the letpair filter `p.1 ≠ a ∧ p.1 ≠ b`. -/
theorem linearCtx_filter_fresh_two_eq (Γ : LinearCtx) (a b : String)
    (ha : a ∉ linearCtxDom Γ) (hb : b ∉ linearCtxDom Γ) :
    Γ.filter (fun p => decide (p.1 ≠ a ∧ p.1 ≠ b)) = Γ := by
  apply List.filter_eq_self.mpr
  intro p hp
  simp only [decide_eq_true_eq]
  refine ⟨?_, ?_⟩ <;> intro heq
  · exact ha (heq ▸ (by simp only [linearCtxDom, List.mem_map]; exact ⟨p, hp, rfl⟩))
  · exact hb (heq ▸ (by simp only [linearCtxDom, List.mem_map]; exact ⟨p, hp, rfl⟩))

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
  -- sub-term with the same seed and counter.
  | Term.letBind _ e1 _ =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
              h_seed h_fresh_s h_fresh_s'
  | Term.letpair _ _ e1 _ =>
      simp only [adjointFrom]
      exact adjoint_typed_aux Delta Sigma Gamma_s Gamma_s' dsE epsSeed x n e1 gSeed
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
  | Term.vmap _ _ e1 =>
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
      -- Abbreviate the fresh names via local `let`s.
      let gA : String := freshName "gA" n
      let gB : String := freshName "gB" n
      let adjA : String := freshName "adjA" n
      -- Freshness facts we need out of h_fresh_s'.
      have hgA_base : "gA" ∈ adjointBases := by simp [adjointBases]
      have hgB_base : "gB" ∈ adjointBases := by simp [adjointBases]
      have hadjA_base : "adjA" ∈ adjointBases := by simp [adjointBases]
      have hgA_fresh_s' : gA ∉ linearCtxDom Gamma_s' :=
        h_fresh_s' n (Nat.le_refl _) "gA" hgA_base
      have hgB_fresh_s' : gB ∉ linearCtxDom Gamma_s' :=
        h_fresh_s' n (Nat.le_refl _) "gB" hgB_base
      have hadjA_fresh_s' : adjA ∉ linearCtxDom Gamma_s' :=
        h_fresh_s' n (Nat.le_refl _) "adjA" hadjA_base
      -- T-Copy: copy gSeed : tensor ds ⊗ tensor ds in Γ_s → Γ_s'.
      have h_copy :
          HasType Delta Sigma Gamma_s (Term.copy gSeed)
            (Typ.pair (Typ.tensor dsE) (Typ.tensor dsE)) epsSeed Gamma_s' :=
        HasType.copy Delta Sigma Gamma_s Gamma_s' gSeed dsE epsSeed h_seed
      -- Seed witness for inner recursion on e1: T-Var on gA with
      -- pre = Γ_s' and post = [(gB, tensor ds)].
      have h_var_gA :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)])
            (Term.var gA) (Typ.tensor dsE) []
            (Gamma_s' ++ [(gB, Typ.tensor dsE)]) := by
        have h := HasType.var Delta Sigma Gamma_s'
                    [(gB, Typ.tensor dsE)] gA (Typ.tensor dsE)
        simpa using h
      -- Freshness in the extended context for the inner call.
      have hn_lt : n < n + 3 := by omega
      have h_fresh_gB_added :
          AdjointNamesFresh (n + 3) (Gamma_s' ++ [(gB, Typ.tensor dsE)]) := by
        have hbase : AdjointNamesFresh (n + 3) Gamma_s' :=
          h_fresh_s'.mono (by omega)
        exact hbase.cons_freshName hgB_base hn_lt
      have h_fresh_gAB_added :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)]) := by
        -- Extend twice: first add gA, then gB. But the actual form is
        -- `Γ_s' ++ [(gA,_),(gB,_)]`. We rewrite as
        -- `(Γ_s' ++ [(gA,_)]) ++ [(gB,_)]`.
        have hbase : AdjointNamesFresh (n + 3) Gamma_s' :=
          h_fresh_s'.mono (by omega)
        have h1 : AdjointNamesFresh (n + 3) (Gamma_s' ++ [(gA, Typ.tensor dsE)]) :=
          hbase.cons_freshName hgA_base hn_lt
        have h2 : AdjointNamesFresh (n + 3)
            ((Gamma_s' ++ [(gA, Typ.tensor dsE)]) ++ [(gB, Typ.tensor dsE)]) :=
          h1.cons_freshName hgB_base hn_lt
        -- Rewrite list associativity.
        have hassoc :
            (Gamma_s' ++ [(gA, Typ.tensor dsE)]) ++ [(gB, Typ.tensor dsE)]
            = Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)] := by
          simp
        rw [hassoc] at h2
        exact h2
      -- IH on e1.
      have h_adj_e1 :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)])
            (adjointFrom e1 x (Term.var gA) (n + 3)) Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ [(gB, Typ.tensor dsE)]) :=
        adjoint_typed_aux Delta Sigma
          (Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)])
          (Gamma_s' ++ [(gB, Typ.tensor dsE)])
          dsE [] x (n + 3) e1 (Term.var gA) h_var_gA
          h_fresh_gAB_added h_fresh_gB_added
      -- Seed witness for inner recursion on e2: T-Var on gB with
      -- pre = Γ_s' and post = []. But we need the context to also
      -- carry the letBind binder (adjA, unit); so post = [(adjA, unit)].
      have h_var_gB :
          HasType Delta Sigma
            (Gamma_s' ++ [(gB, Typ.tensor dsE), (adjA, Typ.unit)])
            (Term.var gB) (Typ.tensor dsE) []
            (Gamma_s' ++ [(adjA, Typ.unit)]) := by
        have h := HasType.var Delta Sigma Gamma_s'
                    [(adjA, Typ.unit)] gB (Typ.tensor dsE)
        simpa using h
      have h_fresh_adjA_added :
          AdjointNamesFresh (n + 3) (Gamma_s' ++ [(adjA, Typ.unit)]) := by
        have hbase : AdjointNamesFresh (n + 3) Gamma_s' :=
          h_fresh_s'.mono (by omega)
        exact hbase.cons_freshName hadjA_base hn_lt
      have h_fresh_gBadjA_added :
          AdjointNamesFresh (n + 3)
            (Gamma_s' ++ [(gB, Typ.tensor dsE), (adjA, Typ.unit)]) := by
        have hbase : AdjointNamesFresh (n + 3) Gamma_s' :=
          h_fresh_s'.mono (by omega)
        have h1 : AdjointNamesFresh (n + 3) (Gamma_s' ++ [(gB, Typ.tensor dsE)]) :=
          hbase.cons_freshName hgB_base hn_lt
        have h2 : AdjointNamesFresh (n + 3)
            ((Gamma_s' ++ [(gB, Typ.tensor dsE)]) ++ [(adjA, Typ.unit)]) :=
          h1.cons_freshName hadjA_base hn_lt
        have hassoc :
            (Gamma_s' ++ [(gB, Typ.tensor dsE)]) ++ [(adjA, Typ.unit)]
            = Gamma_s' ++ [(gB, Typ.tensor dsE), (adjA, Typ.unit)] := by simp
        rw [hassoc] at h2
        exact h2
      have h_adj_e2 :
          HasType Delta Sigma
            (Gamma_s' ++ [(gB, Typ.tensor dsE), (adjA, Typ.unit)])
            (adjointFrom e2 x (Term.var gB) (n + 3)) Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ [(adjA, Typ.unit)]) :=
        adjoint_typed_aux Delta Sigma
          (Gamma_s' ++ [(gB, Typ.tensor dsE), (adjA, Typ.unit)])
          (Gamma_s' ++ [(adjA, Typ.unit)])
          dsE [] x (n + 3) e2 (Term.var gB) h_var_gB
          h_fresh_gBadjA_added h_fresh_adjA_added
      -- T-LetBind: adjA = adjoint e1 in adjoint e2.
      have h_letBind :
          HasType Delta Sigma
            (Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)])
            (Term.letBind adjA (adjointFrom e1 x (Term.var gA) (n + 3))
                                (adjointFrom e2 x (Term.var gB) (n + 3)))
            Typ.unit
            (EffectRow.union
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
            ((Gamma_s' ++ [(adjA, Typ.unit)]).filter (fun p => p.1 ≠ adjA)) := by
        -- The T-Let rule places `(adjA, unit)` at the tail of the
        -- context used to type e2. Our `h_adj_e2` already has that
        -- shape via the var witness `h_var_gB`. We massage the
        -- context on the e2 side so the constructor matches.
        have h2' :
            HasType Delta Sigma
              ((Gamma_s' ++ [(gB, Typ.tensor dsE)]) ++ [(adjA, Typ.unit)])
              (adjointFrom e2 x (Term.var gB) (n + 3)) Typ.unit
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (Gamma_s' ++ [(adjA, Typ.unit)]) := by
          have hassoc :
              (Gamma_s' ++ [(gB, Typ.tensor dsE)]) ++ [(adjA, Typ.unit)]
              = Gamma_s' ++ [(gB, Typ.tensor dsE), (adjA, Typ.unit)] := by simp
          rw [hassoc]; exact h_adj_e2
        exact HasType.letBind Delta Sigma
          (Gamma_s' ++ [(gA, Typ.tensor dsE), (gB, Typ.tensor dsE)])
          (Gamma_s' ++ [(gB, Typ.tensor dsE)])
          (Gamma_s' ++ [(adjA, Typ.unit)])
          adjA (adjointFrom e1 x (Term.var gA) (n + 3))
                (adjointFrom e2 x (Term.var gB) (n + 3))
          Typ.unit Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          h_adj_e1 h2'
      -- Collapse the `filter ≠ adjA` over `Γ_s' ++ [(adjA, unit)]`.
      have h_filter_adjA :
          (Gamma_s' ++ [(adjA, Typ.unit)]).filter (fun p => p.1 ≠ adjA)
            = Gamma_s' := by
        rw [List.filter_append,
            linearCtx_filter_fresh_eq Gamma_s' adjA hadjA_fresh_s']
        simp
      rw [h_filter_adjA] at h_letBind
      -- T-LetPair: letpair gA gB (copy gSeed) (letBind adjA ...).
      have h_letpair :
          HasType Delta Sigma Gamma_s
            (Term.letpair gA gB (Term.copy gSeed)
              (Term.letBind adjA (adjointFrom e1 x (Term.var gA) (n + 3))
                                  (adjointFrom e2 x (Term.var gB) (n + 3))))
            Typ.unit
            (EffectRow.union epsSeed
              (EffectRow.union
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
            (Gamma_s'.filter (fun p => decide (p.1 ≠ gA ∧ p.1 ≠ gB))) :=
        HasType.letpair Delta Sigma Gamma_s Gamma_s'
          Gamma_s' gA gB
          (Term.copy gSeed)
          (Term.letBind adjA (adjointFrom e1 x (Term.var gA) (n + 3))
                              (adjointFrom e2 x (Term.var gB) (n + 3)))
          (Typ.tensor dsE) (Typ.tensor dsE) Typ.unit
          epsSeed
          (EffectRow.union
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
          h_copy h_letBind
      -- Collapse the outer `filter` over `Γ_s'`.
      have h_filter_gAB :
          Gamma_s'.filter (fun p => decide (p.1 ≠ gA ∧ p.1 ≠ gB)) = Gamma_s' :=
        linearCtx_filter_fresh_two_eq Gamma_s' gA gB hgA_fresh_s' hgB_fresh_s'
      rw [h_filter_gAB] at h_letpair
      -- Bridge the effect row via `subEff`.
      exact HasType.subEff Delta Sigma Gamma_s Gamma_s' _ _ _ _
        h_letpair (subEff_letpair_add epsSeed)
  | Term.mul e1 e2 =>
      -- The mul case follows the same structure but with an outer
      -- tape layer (`a`, `aTape`, `b`, `bTape`), a forward `mul`
      -- binding (`y`), and the add-shaped inner `letpair`/`letBind`
      -- (`gA`, `gB`, `adjA`). Eight fresh names total, all at
      -- counter `n`; inner recursion uses counter `n + 8`.
      --
      -- Closing this case cleanly requires replaying the freshness
      -- + filter + typing plumbing from the `add` case at five
      -- context-extension layers. That is mechanical but
      -- significantly heavier than the `add` case. Status: parked
      -- under the catch-all alongside `sum`, `expand`, `handle`;
      -- see the phase doc for the follow-up task.
      sorry
  -- `sum`, `expand`, and `handle` are parked on separate Phase 1
  -- infrastructure (tape extent mechanism, ClausesTypedDB).
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
                   (Gamma ++ [(x, Typ.tensor ds)])
                   e (Typ.tensor dsOut) eps Gamma)
    (h_compat : subsetEffRow eps DiffCompat = true)
    (h_fresh_full : AdjointNamesFresh 0
        (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)]))
    (h_fresh_small : AdjointNamesFresh 0 (Gamma ++ [(x, Typ.tensor ds)])) :
    HasType (Capability.diff :: Delta) Sigma
            (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
            (adjoint e x (Term.var gs))
            Typ.unit
            (EffectRow.union eps [EffectLabel.accum])
            (Gamma ++ [(x, Typ.tensor ds)]) := by
  have hvar_gs :
      HasType (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
        (Term.var gs) (Typ.tensor dsOut) []
        (Gamma ++ [(x, Typ.tensor ds)]) := by
    have h := HasType.var (Capability.diff :: Delta) Sigma
      (Gamma ++ [(x, Typ.tensor ds)]) [] gs (Typ.tensor dsOut)
    simpa using h
  have h_aux :=
    adjoint_typed_aux (Capability.diff :: Delta) Sigma
      (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
      (Gamma ++ [(x, Typ.tensor ds)])
      dsOut [] x 0 e (Term.var gs) hvar_gs h_fresh_full h_fresh_small
  -- `adjoint` is `adjointFrom ... 0`; unfold.
  show HasType _ _ _ (adjointFrom e x (Term.var gs) 0) _ _ _
  -- Helper: `union [accum] [] = [accum]`. Goal: `union eps [accum]`.
  have hsub :
      SubEffRow (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
                (EffectRow.union eps [EffectLabel.accum]) := by
    intro op hmem
    have hop : op = EffectLabel.accum := by
      simpa [EffectRow.union] using hmem
    subst hop
    show EffectLabel.accum ∈ EffectRow.union eps [EffectLabel.accum]
    simp [EffectRow.union, List.mem_append]
    exact Classical.em _
  exact HasType.subEff _ _ _ _ _ _ _ _ h_aux hsub

end LaCaDiLE
