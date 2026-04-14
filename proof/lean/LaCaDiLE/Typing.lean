-- LaCaDiLE/Typing.lean — CapCtx, LinearCtx, HasType inductive relation.
--
-- Every typing rule from proof/paper/figures/typing.tex is encoded as a
-- constructor of `HasType`. Phase 1 T7 skeleton, no proofs yet.
--
-- Judgment shape:
--   HasType Δ Σ Γ e τ ε Γ'
-- ↔ paper's
--   Δ; Σ; Γ ⊢ e : τ ! ε ⊣ Γ'
--
-- Phase 2 Wave 0 refinements that are now live:
--  * effect row union uses `EffectRow.union` with set semantics (P2).
--  * `T-Handle` is the multi-clause form, typed via the `ClausesTyped`
--    companion inductive in the mutual block below (P5).
--  * The typing judgment is parameterized by a store typing `Σ`
--    (threaded unchanged through every Phase 1 rule; Phase 2
--    preservation will grow it monotonically) (P3).
--  * `Term.grad` carries an explicit output type annotation `tOut`
--    so E-Grad can produce a preservation-matching reduced term (P4).
--  * `T-Perform`'s argument and result types come from `opSignature op`
--    instead of free constructor parameters (P7).
--
-- Phase 1 conventions that remain:
--  * context lookup for T-Var requires the variable at the tail of Γ;
--    this matches the paper's "Γ, x:τ" notation exactly and avoids a
--    permutation lemma.
--  * context-splitting premises for T-App / T-Let / T-Pair / T-LetPair
--    thread linear contexts left-to-right through the sub-derivations,
--    matching the paper rules.

import LaCaDiLE.Syntax
import LaCaDiLE.Store

namespace LaCaDiLE

mutual

/-- The typing judgment of LaCaDiLE as an inductive relation.
    Each constructor corresponds to a typing rule in `figures/typing.tex`.
    The arguments are: capability context, input linear context, term,
    type, effect row, output linear context.

    Phase 2 Wave 0 P5 places `HasType` in a mutual block with
    `ClausesTyped` so that the multi-clause `T-Handle` rule can
    reference handler-body typing via a companion inductive,
    sidestepping the `∃ tArg tRet` positivity restriction. -/
inductive HasType : CapCtx → StoreTyp → LinearCtx → Term → Typ → EffectRow → LinearCtx → Prop

  -- T-Var: consume the binding for `x` from any position in Γ.
  -- Paper: Δ; Γpre, x:τ, Γpost ⊢ x : τ ! ∅ ⊣ Γpre, Γpost
  --
  -- Wave 0.5: generalized from tail-only consumption to arbitrary
  -- position. The tail form (Γ ++ [(x,t)] with output Γ) is the
  -- special case where Γpost = []. Generalization is required for
  -- the substitution lemma's `var` and `binder` cases — a tail-only
  -- rule makes the statement of `exchange_tail`/`weakening_tail`/
  -- `subst_preserves_typing` uninhabited because consumption can't
  -- reach the middle of a swapped or weakened context.
  | var
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma_pre Gamma_post : LinearCtx) (x : String) (t : Typ) :
      HasType Delta Sigma (Gamma_pre ++ [(x, t)] ++ Gamma_post)
              (Term.var x) t [] (Gamma_pre ++ Gamma_post)

  -- T-Unit.
  -- Paper: Δ; Γ ⊢ () : unit ! ∅ ⊣ Γ
  | unit
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx) :
      HasType Delta Sigma Gamma Term.unit Typ.unit [] Gamma

  -- T-Abs.
  -- Paper: Δ; Γ₁, x:τ₁ ⊢ e : τ₂ ! ε ⊣ Γ₂
  --        ─────────────────────────────────
  --        Δ; Γ₁ ⊢ λx:τ₁.e : τ₁ → τ₂ ! ε ⊣ Γ₂ \ {x}
  | abs
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (x : String) (t1 t2 : Typ) (eps : EffectRow) (e : Term) :
      HasType Delta Sigma (Gamma1 ++ [(x, t1)]) e t2 eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.abs x t1 e) (Typ.arrow t1 t2 eps) []
              (Gamma2.filter (fun p => p.1 ≠ x))

  -- T-App.
  -- Paper: Δ; Γ₁ ⊢ e₁ : τ₁ → τ₂ ! ε ⊣ Γ₂    Δ; Γ₂ ⊢ e₂ : τ₁ ! ε₂ ⊣ Γ₃
  --        ─────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ e₁ e₂ : τ₂ ! ε₁ ∪ ε₂ ∪ ε ⊣ Γ₃
  | app
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (t1 t2 : Typ) (eps eps1 eps2 : EffectRow) :
      HasType Delta Sigma Gamma1 e1 (Typ.arrow t1 t2 eps) eps1 Gamma2 →
      HasType Delta Sigma Gamma2 e2 t1 eps2 Gamma3 →
      HasType Delta Sigma Gamma1 (Term.app e1 e2) t2 (EffectRow.union (EffectRow.union eps1 eps2) eps) Gamma3

  -- T-Let.
  -- Paper: Δ; Γ₁ ⊢ e₁ : τ₁ ! ε₁ ⊣ Γ₂     Δ; Γ₂, x:τ₁ ⊢ e₂ : τ₂ ! ε₂ ⊣ Γ₃
  --        ────────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ let x = e₁ in e₂ : τ₂ ! ε₁ ∪ ε₂ ⊣ Γ₃ \ {x}
  | letBind
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (x : String) (e1 e2 : Term) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 →
      HasType Delta Sigma (Gamma2 ++ [(x, t1)]) e2 t2 eps2 Gamma3 →
      HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t2 (EffectRow.union eps1 eps2)
              (Gamma3.filter (fun p => p.1 ≠ x))

  -- T-Copy.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂
  --        ───────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ copy(e) : tensor[d̄] ⊗ tensor[d̄] ! ε ⊣ Γ₂
  --
  -- Wave 3 restriction: T-Copy is now tensor-only (Phase 1 had it
  -- polymorphic, but the runtime Step.copy rule only fires on loc
  -- values, and only tensor values are represented as locs. The
  -- polymorphic form made Progress's copy case unprovable because
  -- copy of an abs/unit/pair value would be stuck. Restricting to
  -- tensors aligns the typing rule with the operational semantics.)
  | copy
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.copy e)
              (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) eps Gamma2

  -- T-LetPair.
  -- Paper: Δ; Γ₁ ⊢ e₁ : τ₁ ⊗ τ₂ ! ε₁ ⊣ Γ₂
  --        Δ; Γ₂, x:τ₁, x':τ₂ ⊢ e₂ : τ ! ε₂ ⊣ Γ₃
  --        ────────────────────────────────────────
  --        Δ; Γ₁ ⊢ let (x,x') = e₁ in e₂ : τ ! ε₁ ∪ ε₂ ⊣ Γ₃ \ {x, x'}
  | letpair
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (x y : String) (e1 e2 : Term) (t1 t2 t : Typ) (eps1 eps2 : EffectRow) :
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 →
      HasType Delta Sigma (Gamma2 ++ [(x, t1), (y, t2)]) e2 t eps2 Gamma3 →
      HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t (EffectRow.union eps1 eps2)
              (Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y))

  -- T-Pair.
  | tpair
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 →
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 →
      HasType Delta Sigma Gamma1 (Term.pair e1 e2) (Typ.pair t1 t2) (EffectRow.union eps1 eps2) Gamma3

  -- T-Fst.
  | fst
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (t1 t2 : Typ) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.pair t1 t2) eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.fst e) t1 eps Gamma2

  -- T-Snd.
  | snd
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (t1 t2 : Typ) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.pair t1 t2) eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.snd e) t2 eps Gamma2

  -- T-Const.
  -- Paper: v is a scalar literal
  --        ───────────────────────────────────────────
  --        Δ; Γ ⊢ const(v, d̄) : tensor[d̄] ! ∅ ⊣ Γ
  | const
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx) (v : Float) (ds : DimList) :
      HasType Delta Sigma Gamma (Term.const v ds) (Typ.tensor ds) [] Gamma

  -- T-Add.
  | tadd
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (ds : DimList) (eps1 eps2 : EffectRow) :
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 →
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 →
      HasType Delta Sigma Gamma1 (Term.add e1 e2) (Typ.tensor ds) (EffectRow.union eps1 eps2) Gamma3

  -- T-Mul.
  | tmul
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (ds : DimList) (eps1 eps2 : EffectRow) :
      HasType Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 →
      HasType Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 →
      HasType Delta Sigma Gamma1 (Term.mul e1 e2) (Typ.tensor ds) (EffectRow.union eps1 eps2) Gamma3

  -- T-Sum.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂    d ∈ d̄
  --        ─────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ sum(e, d) : tensor[rem(d̄, d)] ! ε ⊣ Γ₂
  -- Stage 1 refactor: positional index `i` replaced by named
  -- dimension `d`; membership premise replaces the old `i < |d̄|`.
  | tsum
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (d : Dim) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      d ∈ ds →
      HasType Delta Sigma Gamma1 (Term.sum e d) (Typ.tensor (rem ds d)) eps Gamma2

  -- T-Expand.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂
  --        ─────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ expand(e, d) : tensor[ins(d̄, d)] ! ε ⊣ Γ₂
  -- Stage 1 refactor: positional `(i, k)` replaced by named
  -- dimension `d`; for `Dim.lit k` the extent `k` is carried by
  -- `d` itself.
  | texpand
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (d : Dim) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.expand e d) (Typ.tensor (ins ds d)) eps Gamma2

  -- T-UniformLike.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂
  --        ──────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ uniform_like(e, lo, hi) : tensor[d̄] ! ε ∪ {Random} ⊣ Γ₂
  | uniformLike
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (lo hi : Float) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.uniformLike e lo hi) (Typ.tensor ds)
              (EffectRow.union eps [EffectLabel.random]) Gamma2

  -- T-Perform.
  -- Paper: Δ; Γ₁ ⊢ e : τ_arg ! ε ⊣ Γ₂   op : τ_arg → τ_ret
  --        ────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ perform op(e) : τ_ret ! ε ∪ {op} ⊣ Γ₂
  --
  -- Wave 0 P7 fix: `tArg` and `tRet` are no longer free constructor
  -- parameters. They are looked up via `opSignature op`, which Phase 1
  -- skeleton defines as uniform `(unit, unit)`. Phase 2 WS2.7 refines
  -- the signature table per-operation without touching this rule.
  | perform
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (op : EffectLabel) (e : Term) (tArg tRet : Typ) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e tArg eps Gamma2 →
      OpSigMatch op tArg tRet →
      HasType Delta Sigma Gamma1 (Term.perform op e) tRet
              (EffectRow.union [op] eps) Gamma2

  -- T-Handle (multi-clause, Wave 0 P5).
  -- Paper: Δ; Γ₁ ⊢ body : τ ! ε_b ⊣ Γ₂   (every op ∈ ε_h is in ε_b)
  --        (every clause's op is in ε_h — no spurious clauses)
  --        (every op ∈ ε_h has a matching clause)
  --        for each clause (op, x, k, hb):
  --          Δ; Γ₂, x:τ_arg_op, k:(τ_ret_op → τ ! ε_r) ⊢ hb : τ ! ε_r ⊣ Γ₃
  --        where ε_r = ε_b with every ε_h operation removed
  --        ──────────────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ handle[ε_h] body with {op_i(x_i,k_i) → hb_i}_i : τ ! ε_r ⊣ Γ₃
  --
  -- The per-clause typing obligation is delegated to the companion
  -- `ClausesTyped` inductive (part of the mutual block below). Each
  -- clause's `tArg`/`tRet` is absorbed into `ClausesTyped.cons` to
  -- avoid the existential-inside-constructor positivity restriction.
  | handle
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (body : Term)
      (clauses : List (EffectLabel × String × String × Term))
      (t : Typ) (epsH epsB : EffectRow) :
      HasType Delta Sigma Gamma1 body t epsB Gamma2 →
      (∀ op ∈ epsH, op ∈ epsB) →
      (∀ cl ∈ clauses, cl.1 ∈ epsH) →
      (∀ op ∈ epsH, ∃ cl ∈ clauses, cl.1 = op) →
      ClausesTyped Delta Sigma Gamma2 Gamma3 t
                   (EffectRow.removeOps epsB epsH) clauses →
      HasType Delta Sigma Gamma1
              (Term.handle epsH body clauses)
              t (EffectRow.removeOps epsB epsH)
              Gamma3

  -- T-Grad.
  -- Restricted to literal abstractions per the Wave 2 red-team round 2 fix.
  -- Paper: Δ ∪ {Diff}; Γ, x:tensor[d̄] ⊢ e : tensor[d̄'] ! ε ⊣ Γ
  --        ε ⊆ DiffCompat
  --        ────────────────────────────────────────────────────────────────
  --        Δ; Γ ⊢ grad(λx:tensor[d̄].e)
  --            : tensor[d̄] → tensor[d̄'] → tensor[d̄] ! ε ⊣ Γ
  --
  -- The result type is a two-argument (curried) function: parameter value
  -- plus output-gradient seed, producing the parameter gradient. The outer
  -- arrow carries the empty effect row (abstraction has no effects); the
  -- inner arrow carries `eps`.
  | tgrad
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
      (x : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow) :
      HasType (Capability.diff :: Delta) Sigma
              (Gamma ++ [(x, Typ.tensor ds)]) e (Typ.tensor dsOut) eps Gamma →
      subsetEffRow eps DiffCompat = true →
      HasType Delta Sigma Gamma
              (Term.grad x (Typ.tensor ds) (Typ.tensor dsOut) e)
              (Typ.arrow
                (Typ.tensor ds)
                (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) eps)
                [])
              []
              Gamma

  -- T-Vmap.
  -- Syntactic on literal abstractions (matches E-Vmap's reduction shape).
  -- Paper: Δ; Γ, x:τ₁ ⊢ e : τ₂ ! ε ⊣ Γ    d fresh
  --        ───────────────────────────────────────────────────
  --        Δ; Γ ⊢ vmap(λx:τ₁.e) : addDim(d, τ₁) → addDim(d, τ₂) ! ε ⊣ Γ
  --
  -- Phase 1 skeleton treats "d fresh" as a side condition checked at
  -- construction; Phase 2 will add a freshness lemma.
  | tvmap
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
      (x : String) (t1 t2 : Typ) (e : Term) (eps : EffectRow) (d : Dim) :
      HasType Delta Sigma (Gamma ++ [(x, t1)]) e t2 eps Gamma →
      HasType Delta Sigma Gamma (Term.vmap x t1 e)
              (Typ.arrow (addDim d t1) (addDim d t2) eps) [] Gamma

  -- T-Loc: runtime locations.
  -- Not in the paper's source language — introduced by reduction. The
  -- store typing records which type each live location holds; T-Loc
  -- retrieves it. Phase 2 Wave 1 structural fix: without this rule,
  -- every store-allocating Step (tconst, copy, tadd, ...) reduces to
  -- a Term.loc with no reachable typing derivation, blocking the
  -- Preservation proof.
  | loc
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
      (ell : Loc) (t : Typ) :
      storeTypLookup Sigma ell = some t →
      HasType Delta Sigma Gamma (Term.loc ell) t [] Gamma

  -- T-SubEff: effect-row subsumption (Wave 0.5).
  -- A derivation at effect row ε can be widened to any ε' that
  -- includes every operation of ε. Standard row-polymorphic
  -- subsumption (Koka-style). Required by Preservation's value-case
  -- sub-derivations (E-Fst, E-Snd, E-HandleRet) and by the effect-row
  -- widening that happens in E-Ctx congruence on effectful sub-terms.
  --
  -- The ripple: every inversion lemma must strip off trailing subEff
  -- applications. A helper `HasType.strip_subEff` (Wave 1+) does this
  -- once and every inversion reuses it.
  | subEff
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
      (e : Term) (t : Typ) (eps eps' : EffectRow) :
      HasType Delta Sigma Gamma e t eps Gamma' →
      SubEffRow eps eps' →
      HasType Delta Sigma Gamma e t eps' Gamma'

/-- Handler-clause typing judgment used by `HasType.handle` (Wave 0 P5).
    `ClausesTyped Δ Γ₂ Γ₃ τ ε_r cls` asserts that every clause in `cls`
    type-checks its handler body `hb` under `Γ₂` extended with the
    operation argument `x : tArg` and the linear continuation
    `k : τ_ret → τ ! ε_r`, producing result type `τ`, residual effect
    row `ε_r`, and output context `Γ₃`. Each clause supplies its own
    `tArg` and `tRet` as `cons` constructor parameters, avoiding the
    existential-in-premise positivity restriction that rules out an
    inline `∀ cl ∈ clauses, ∃ tArg tRet, HasType ...` premise on
    `HasType.handle`. -/
inductive ClausesTyped :
    CapCtx → StoreTyp → LinearCtx → LinearCtx → Typ → EffectRow →
    List (EffectLabel × String × String × Term) → Prop
  -- Wave 3 fix: nil case now requires Γ2 = Γ3 so that
  -- `has_type_linear_shrinks`'s ClausesTyped motive can discharge
  -- the handle case via an inductive shrinkage argument. An empty
  -- clause list has no binders, so the outer context threads through
  -- unchanged.
  | nil (Delta : CapCtx) (Sigma : StoreTyp) (Gamma2 : LinearCtx)
        (t : Typ) (epsR : EffectRow) :
        ClausesTyped Delta Sigma Gamma2 Gamma2 t epsR []
  | cons (Delta : CapCtx) (Sigma : StoreTyp) (Gamma2 Gamma3 : LinearCtx)
         (t tArg tRet : Typ) (epsR : EffectRow)
         (op : EffectLabel) (x k : String) (hb : Term)
         (rest : List (EffectLabel × String × String × Term)) :
         HasType Delta Sigma
                 (Gamma2 ++ [(x, tArg), (k, Typ.arrow tRet t epsR)])
                 hb t epsR Gamma3 →
         ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR rest →
         ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR ((op, x, k, hb) :: rest)

end

/-! ## Effect-row weakening (Wave 1 structural helper)

Preservation's value-case sub-derivations (E-Fst, E-Snd, E-HandleRet)
reduce a redex typed at effect row `union ε1 ε2` to a sub-value typed
at `ε1` — but the goal demands `union ε1 ε2`. Widening the effect row
of a typing derivation is the canonical fix. -/

/-! ### Effect row algebraic helpers -/

@[simp] theorem EffectRow.union_nil_left (eps : EffectRow) :
    EffectRow.union [] eps = eps := by
  simp [EffectRow.union]

/-- `eps` is a SubEffRow of its union with anything on the right.
    Direct consequence of `EffectRow.union` being list append +
    filter: every op of the left operand appears in the result. -/
theorem SubEffRow.union_left (eps eps_extra : EffectRow) :
    SubEffRow eps (EffectRow.union eps eps_extra) := by
  intro op hop
  show op ∈ eps ++ eps_extra.filter (fun o => !eps.contains o)
  exact List.mem_append_left _ hop

/-- `SubEffRow` is reflexive. -/
theorem SubEffRow.refl (eps : EffectRow) : SubEffRow eps eps :=
  fun _ h => h

/-- `SubEffRow` is transitive. -/
theorem SubEffRow.trans
    {eps1 eps2 eps3 : EffectRow}
    (h12 : SubEffRow eps1 eps2) (h23 : SubEffRow eps2 eps3) :
    SubEffRow eps1 eps3 :=
  fun op hop => h23 op (h12 op hop)

/-- Widen the effect row of a typing derivation. Wave 0.5 made this
    trivial: with `HasType.subEff` as a constructor, we apply it
    directly with a `SubEffRow.union_left` witness. -/
theorem HasType.weaken_eff
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (eps_extra : EffectRow)
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    HasType Delta Sigma Gamma e t (EffectRow.union eps eps_extra) Gamma' :=
  HasType.subEff Delta Sigma Gamma Gamma' e t eps
    (EffectRow.union eps eps_extra) h (SubEffRow.union_left eps eps_extra)

/-! ## Linear-context domain shrinkage (Wave 3)

Moved from Progress.lean so both Progress and Preservation can use
`has_type_linear_shrinks` without a circular import. -/

/-- Domain-subset relation on linear contexts. -/
def DomSub (G1 G2 : LinearCtx) : Prop :=
  ∀ x, x ∈ linearCtxDom G1 → x ∈ linearCtxDom G2

theorem DomSub.refl (G : LinearCtx) : DomSub G G := fun _ h => h

theorem DomSub.trans {G1 G2 G3 : LinearCtx}
    (h12 : DomSub G1 G2) (h23 : DomSub G2 G3) : DomSub G1 G3 :=
  fun x h => h23 x (h12 x h)

theorem DomSub.filter_self (G : LinearCtx) (p : (String × Typ) → Bool) :
    DomSub (G.filter p) G := by
  intro x hx
  simp only [linearCtxDom, List.mem_map] at hx ⊢
  obtain ⟨q, hq_mem, hq_eq⟩ := hx
  rw [List.mem_filter] at hq_mem
  exact ⟨q, hq_mem.1, hq_eq⟩

/-- Linear-context domain shrinks across every HasType derivation. -/
theorem has_type_linear_shrinks
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ x, x ∈ linearCtxDom Gamma' → x ∈ linearCtxDom Gamma := by
  induction h using HasType.rec
    (motive_2 := fun (_Δ : CapCtx) (_S : StoreTyp)
                     (Γ2 Γ3 : LinearCtx) (_ : Typ) (_ : EffectRow)
                     (_ : List (EffectLabel × String × String × Term))
                     (_ : _) => DomSub Γ3 Γ2) with
  | var _ _ Γpre Γpost y t_v =>
      intro z hz
      simp only [linearCtxDom, List.map_append, List.mem_append] at hz ⊢
      rcases hz with h1 | h1
      · exact Or.inl (Or.inl h1)
      · exact Or.inr h1
  | unit _ _ _ => intro z hz; exact hz
  | abs _ _ Γ1 Γ2 y t1 _ _ _ _ ih =>
      intro z hz
      have hz_ne_y : z ≠ y := by
        simp only [linearCtxDom, List.mem_map, List.mem_filter, decide_eq_true_eq] at hz
        obtain ⟨q, ⟨_, hne⟩, hqeq⟩ := hz
        rw [← hqeq]; exact hne
      have hz2 : z ∈ linearCtxDom Γ2 := DomSub.filter_self Γ2 _ z hz
      have hz3 := ih z hz2
      simp only [linearCtxDom, List.map_append, List.mem_append, List.map_cons,
        List.mem_cons, List.map_nil, List.not_mem_nil] at hz3
      rcases hz3 with hin | heq
      · exact hin
      · rcases heq with heq | hfalse
        · exact absurd heq hz_ne_y
        · exact hfalse.elim
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro z hz; exact ih1 z (ih2 z hz)
  | letBind _ _ _ _ Γ3 y _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro z hz
      have hz_ne_y : z ≠ y := by
        simp only [linearCtxDom, List.mem_map, List.mem_filter, decide_eq_true_eq] at hz
        obtain ⟨q, ⟨_, hne⟩, hqeq⟩ := hz
        rw [← hqeq]; exact hne
      have hz2 : z ∈ linearCtxDom Γ3 := DomSub.filter_self Γ3 _ z hz
      have hz3 := ih2 z hz2
      simp only [linearCtxDom, List.map_append, List.mem_append, List.map_cons,
        List.mem_cons, List.map_nil] at hz3
      rcases hz3 with hin | heq
      · exact ih1 z hin
      · rcases heq with heq | hfalse
        · exact absurd heq hz_ne_y
        · exact (List.not_mem_nil hfalse).elim
  | copy _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | letpair _ _ _ _ Γ3 x y _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro z hz
      have hz_neither : z ≠ x ∧ z ≠ y := by
        simp only [linearCtxDom, List.mem_map, List.mem_filter, decide_eq_true_eq] at hz
        obtain ⟨q, ⟨_, hne1, hne2⟩, hqeq⟩ := hz
        rw [← hqeq]; exact ⟨hne1, hne2⟩
      have hz2 : z ∈ linearCtxDom Γ3 := DomSub.filter_self Γ3 _ z hz
      have hz3 := ih2 z hz2
      simp only [linearCtxDom, List.map_append, List.mem_append, List.map_cons,
        List.mem_cons, List.map_nil] at hz3
      rcases hz3 with hin | heq
      · exact ih1 z hin
      · rcases heq with heqx | heq2
        · exact absurd heqx hz_neither.1
        · rcases heq2 with heqy | hfalse
          · exact absurd heqy hz_neither.2
          · exact (List.not_mem_nil hfalse).elim
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro z hz; exact ih1 z (ih2 z hz)
  | fst _ _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | snd _ _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | const _ _ _ _ _ => intro z hz; exact hz
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro z hz; exact ih1 z (ih2 z hz)
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      intro z hz; exact ih1 z (ih2 z hz)
  | tsum _ _ _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | texpand _ _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | perform _De _Si _G1 _G2 _op _e _tA _tR _eps _h _hM ih =>
      intro z hz; exact ih z hz
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih_body ih_clauses =>
      intro z hz; exact ih_body z (ih_clauses z hz)
  | tgrad _ _ _ _ _ _ _ _ _ _ _ => intro z hz; exact hz
  | tvmap _ _ _ _ _ _ _ _ _ _ => intro z hz; exact hz
  | loc _ _ _ _ _ _ => intro z hz; exact hz
  | subEff _ _ _ _ _ _ _ _ _ _ ih => intro z hz; exact ih z hz
  | nil _ _ _ _ _ => exact DomSub.refl _
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih_rest => exact ih_rest

/-- A well-typed value under a closed input context produces a
    closed output context. (Lighter form — the `IsValue` premise is
    not actually used because `has_type_linear_shrinks` gives
    shrinkage for any derivation.) -/
theorem has_type_closed_output_of_closed_input
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma [] e t eps Gamma') : Gamma' = [] := by
  have hshrink := has_type_linear_shrinks h
  cases hΓ' : Gamma' with
  | nil => rfl
  | cons hd tl =>
    exfalso
    have hmem : hd.1 ∈ linearCtxDom Gamma' := by
      rw [hΓ']; simp [linearCtxDom]
    have := hshrink hd.1 hmem
    simp [linearCtxDom] at this

/-! ## Effect-scoping lemma (Track C2)

Every derivation of `Term.perform op e` produces an outer effect row
that contains `op`. Absorbs `HasType.subEff` via membership
preservation. Used by the closed-form `progress` theorem to rule
out a top-level unhandled `perform` when `eps = []`. -/

theorem hasType_perform_eff_mem
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {op : EffectLabel} {e : Term} {tRet : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (Term.perform op e) tRet eps Gamma') :
    op ∈ eps := by
  generalize heq : Term.perform op e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | perform _ _ _ _ op' _ _ _ eps' _ _ _ =>
      cases heq
      -- Goal: op ∈ EffectRow.union [op] eps'
      show op ∈ [op] ++ eps'.filter (fun o => !([op] : EffectRow).contains o)
      exact List.mem_append_left _ (List.mem_singleton.mpr rfl)
  | subEff _ _ _ _ _ _ _ _ _ hsub ih =>
      exact hsub _ (ih heq)
  | _ => (try cases heq) <;>
         first | exact True.intro | (exfalso; contradiction)

end LaCaDiLE
