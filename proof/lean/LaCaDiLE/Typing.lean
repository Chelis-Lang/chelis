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
  -- Tombstone semantics: output context marks the slot as `none`
  -- rather than removing it, preserving context length.
  | var
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma_pre Gamma_post : LinearCtx) (x : String) (t : Typ) :
      HasType Delta Sigma (Gamma_pre ++ [(x, some t)] ++ Gamma_post)
              (Term.var x) t [] (Gamma_pre ++ [(x, none)] ++ Gamma_post)

  -- T-Unit.
  -- Paper: Δ; Γ ⊢ () : unit ! ∅ ⊣ Γ
  | unit
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx) :
      HasType Delta Sigma Gamma Term.unit Typ.unit [] Gamma

  -- T-Abs: body typed under Γ₁ ++ [(x, some τ₁)]. Outer rule strips
  -- the binder slot from the body's output context.
  | abs
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (x : String) (t1 t2 : Typ) (eps : EffectRow) (e : Term)
      (slot : Option Typ) :
      HasType Delta Sigma (Gamma1 ++ [(x, some t1)]) e t2 eps
              (Gamma2 ++ [(x, slot)]) →
      HasType Delta Sigma Gamma1 (Term.abs x t1 e) (Typ.arrow t1 t2 eps) []
              Gamma2

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
      (x : String) (e1 e2 : Term) (t1 t2 : Typ) (eps1 eps2 : EffectRow)
      (slot : Option Typ) :
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 →
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t2 eps2
              (Gamma3 ++ [(x, slot)]) →
      HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t2
              (EffectRow.union eps1 eps2) Gamma3

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
      (x y : String) (e1 e2 : Term) (t1 t2 t : Typ) (eps1 eps2 : EffectRow)
      (slotX slotY : Option Typ) :
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 →
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
              (Gamma3 ++ [(x, slotX), (y, slotY)]) →
      HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t
              (EffectRow.union eps1 eps2) Gamma3

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
      (x : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
      (slot : Option Typ) :
      HasType (Capability.diff :: Delta) Sigma
              (Gamma ++ [(x, some (Typ.tensor ds))]) e (Typ.tensor dsOut) eps
              (Gamma ++ [(x, slot)]) →
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
      (x : String) (t1 t2 : Typ) (e : Term) (eps : EffectRow) (d : Dim)
      (slot : Option Typ) :
      HasType Delta Sigma (Gamma ++ [(x, some t1)]) e t2 eps
              (Gamma ++ [(x, slot)]) →
      HasType Delta Sigma Gamma (Term.vmap x t1 d e)
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
         (rest : List (EffectLabel × String × String × Term))
         (slotX slotK : Option Typ) :
         OpSigMatch op tArg tRet →
         HasType Delta Sigma
                (Gamma2 ++ [(x, some tArg), (k, some (Typ.arrow tRet t epsR))])
                hb t epsR
                (Gamma3 ++ [(x, slotX), (k, slotK)]) →
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

theorem DomSub.append_left (G1 G2 : LinearCtx) :
    DomSub G1 (G1 ++ G2) := by
  intro x hx
  show x ∈ linearCtxDom (G1 ++ G2)
  simp only [linearCtxDom, List.map_append, List.mem_append]
  left; exact hx

/-- Names in LinearCtx are invariant across HasType derivations.
    Tombstoning only changes Option Typ values, not the String keys. -/
theorem hasType_names_preserved
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    Gamma.map Prod.fst = Gamma'.map Prod.fst := by
  induction h using HasType.rec
    (motive_2 := fun _ _ Γ2 Γ3 _ _ _ _ => Γ2.map Prod.fst = Γ3.map Prod.fst)
    with
  | var _ _ Γpre Γpost _ _ => simp [List.map_append]
  | unit _ _ _ => rfl
  | abs _ _ _ _ _ _ _ _ _ _ _ ih =>
      simp only [List.map_append, List.map_cons, List.map_nil] at ih
      exact List.append_cancel_right ih
  | letBind _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      simp only [List.map_append, List.map_cons, List.map_nil] at ih2
      exact ih1.trans (List.append_cancel_right ih2)
  | letpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      simp only [List.map_append, List.map_cons, List.map_nil] at ih2
      exact ih1.trans (List.append_cancel_right ih2)
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | copy _ _ _ _ _ _ _ _ ih => exact ih
  | fst _ _ _ _ _ _ _ _ _ ih => exact ih
  | snd _ _ _ _ _ _ _ _ _ ih => exact ih
  | const _ _ _ _ _ => rfl
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | tsum _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | texpand _ _ _ _ _ _ _ _ _ ih => exact ih
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | perform _ _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih_body ih_clauses =>
      exact ih_body.trans ih_clauses
  | tgrad => rfl
  | tvmap => rfl
  | loc _ _ _ _ _ _ => rfl
  | subEff _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | nil _ _ _ _ _ => rfl
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihRest => exact ihRest

/-- Linear-context domain is invariant across HasType derivations.
    Under tombstone semantics, linearCtxDom = map Prod.fst includes
    all names. Since names are preserved, domain is identical. -/
theorem has_type_linear_shrinks
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ x, x ∈ linearCtxDom Gamma' → x ∈ linearCtxDom Gamma := by
  simp only [linearCtxDom, hasType_names_preserved h]
  exact fun _ h => h

/-- Tombstone semantics preserves the ordered list of names. In
    particular, the output-domain list is an ordered sublist of the
    input-domain list. -/
theorem has_type_sublist
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    List.Sublist (linearCtxDom Gamma') (linearCtxDom Gamma) := by
  simpa [linearCtxDom, hasType_names_preserved h] using
    (List.Sublist.refl (linearCtxDom Gamma))

/-- Slotwise output/input relation for tombstone-preserving named
    contexts: each output binding keeps the same name, and each slot is
    either unchanged or tombstoned to `none`. -/
def SlotSub : LinearCtx → LinearCtx → Prop
  | [], [] => True
  | (xOut, out) :: outs, (xIn, inp) :: ins =>
      xOut = xIn ∧ (out = none ∨ out = inp) ∧ SlotSub outs ins
  | _, _ => False

theorem slotSub_refl :
    ∀ Gamma : LinearCtx, SlotSub Gamma Gamma
  | [] => by simp [SlotSub]
  | (x, slot) :: rest => by
      simp [SlotSub, slotSub_refl rest]

theorem slotSub_tail
    {out inp : String × Option Typ} {outs inps : LinearCtx}
    (h : SlotSub (out :: outs) (inp :: inps)) :
    SlotSub outs inps :=
  h.2.2

theorem slotSub_append_singleton_inv
    {outs inps : LinearCtx} {xOut xIn : String} {out inp : Option Typ}
    (h : SlotSub (outs ++ [(xOut, out)]) (inps ++ [(xIn, inp)])) :
    xOut = xIn ∧ (out = none ∨ out = inp) ∧ SlotSub outs inps := by
  induction outs generalizing inps with
  | nil =>
      cases inps with
      | nil =>
          simpa [SlotSub] using h
      | cons inpHd inpTl =>
          cases inpTl <;> simp [SlotSub] at h
  | cons outHd outTl ih =>
      cases inps with
      | nil =>
          cases outTl <;> simp [SlotSub] at h
      | cons inpHd inpTl =>
          simp [SlotSub] at h
          rcases h with ⟨hname, hslot, htail⟩
          rcases ih (inps := inpTl) htail with ⟨hx, hlast, hrest⟩
          exact ⟨hx, hlast, ⟨hname, hslot, hrest⟩⟩

theorem slotSub_append_pair_inv
    {outs inps : LinearCtx}
    {xOut xIn yOut yIn : String}
    {outX inpX outY inpY : Option Typ}
    (h : SlotSub
      (outs ++ [(xOut, outX), (yOut, outY)])
      (inps ++ [(xIn, inpX), (yIn, inpY)])) :
    xOut = xIn ∧ (outX = none ∨ outX = inpX) ∧
      yOut = yIn ∧ (outY = none ∨ outY = inpY) ∧
      SlotSub outs inps := by
  have h1 :
      yOut = yIn ∧ (outY = none ∨ outY = inpY) ∧
      SlotSub (outs ++ [(xOut, outX)]) (inps ++ [(xIn, inpX)]) := by
    have h' :
        SlotSub
          ((outs ++ [(xOut, outX)]) ++ [(yOut, outY)])
          ((inps ++ [(xIn, inpX)]) ++ [(yIn, inpY)]) := by
      simpa [List.append_assoc] using h
    simpa [List.append_assoc] using
      (slotSub_append_singleton_inv
        (outs := outs ++ [(xOut, outX)])
        (inps := inps ++ [(xIn, inpX)])
        (xOut := yOut) (xIn := yIn) (out := outY) (inp := inpY)
        h')
  rcases h1 with ⟨hy, hslotY, hrest⟩
  have h2 :
      xOut = xIn ∧ (outX = none ∨ outX = inpX) ∧ SlotSub outs inps := by
    simpa [List.append_assoc] using
      (slotSub_append_singleton_inv
        (outs := outs) (inps := inps)
        (xOut := xOut) (xIn := xIn) (out := outX) (inp := inpX)
        hrest)
  rcases h2 with ⟨hx, hslotX, hcore⟩
  exact ⟨hx, hslotX, hy, hslotY, hcore⟩

theorem slotSub_append
    {out1 out2 inp1 inp2 : LinearCtx}
    (h1 : SlotSub out1 inp1)
    (h2 : SlotSub out2 inp2) :
    SlotSub (out1 ++ out2) (inp1 ++ inp2) := by
  revert out2 inp2 h2
  induction out1 generalizing inp1 with
  | nil =>
      intro out2 inp2 h2
      cases inp1 with
      | nil =>
          simpa [SlotSub] using h2
      | cons i is =>
          cases h1
  | cons o os ih =>
      intro out2 inp2 h2
      cases inp1 with
      | nil =>
          cases h1
      | cons i is =>
          rcases h1 with ⟨hname, hslot, htail⟩
          exact ⟨hname, hslot, ih htail h2⟩

theorem slotSub_trans
    {out mid inp : LinearCtx}
    (h1 : SlotSub out mid)
    (h2 : SlotSub mid inp) :
    SlotSub out inp := by
  induction out generalizing mid inp with
  | nil =>
      cases mid <;> cases inp <;> simp [SlotSub] at h1 h2 ⊢
  | cons o os ih =>
      cases mid with
      | nil =>
          cases h1
      | cons m ms =>
          cases inp with
          | nil =>
              cases h2
          | cons i is =>
              rcases h1 with ⟨h1name, h1slot, h1tail⟩
              rcases h2 with ⟨h2name, h2slot, h2tail⟩
              refine ⟨h1name.trans h2name, ?_, ih h1tail h2tail⟩
              rcases h1slot with h1slot | h1slot
              · exact Or.inl h1slot
              · rw [h1slot]
                exact h2slot

theorem has_type_slotSub
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    SlotSub Gamma' Gamma := by
  induction h using HasType.rec
    (motive_2 := fun _ _ GammaIn GammaOut _ _ _ _ => SlotSub GammaOut GammaIn) with
  | var _ _ GammaPre GammaPost x tx =>
      have hpost : SlotSub GammaPost GammaPost := slotSub_refl _
      have hmid : SlotSub [(x, none)] [(x, some tx)] := by
        simp [SlotSub]
      have hpre : SlotSub GammaPre GammaPre := slotSub_refl _
      simpa [SlotSub] using slotSub_append hpre (slotSub_append hmid hpost)
  | unit _ _ Gamma =>
      simpa using slotSub_refl Gamma
  | abs _ _ Gamma1 Gamma2 x t1 _ _ _ slot _ ih =>
      have hbody :
          SlotSub (Gamma2 ++ [(x, slot)]) (Gamma1 ++ [(x, some t1)]) := by
        simpa [SlotSub] using ih
      exact (slotSub_append_singleton_inv hbody).2.2
  | app _ _ _ Gamma2 _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSub_trans ih2 ih1
  | letBind _ _ _ Gamma2 Gamma3 x _ _ t1 _ _ _ slot _ _ ih1 ih2 =>
      have hbody :
          SlotSub (Gamma3 ++ [(x, slot)]) (Gamma2 ++ [(x, some t1)]) := by
        simpa [SlotSub] using ih2
      exact slotSub_trans (slotSub_append_singleton_inv hbody).2.2 ih1
  | copy _ _ _ _ _ _ _ _ ih =>
      exact ih
  | letpair _ _ _ Gamma2 Gamma3 x y _ _ t1 t2 _ _ _ slotX slotY _ _ ih1 ih2 =>
      have hbody :
          SlotSub (Gamma3 ++ [(x, slotX), (y, slotY)])
            (Gamma2 ++ [(x, some t1), (y, some t2)]) := by
        simpa [SlotSub] using ih2
      exact slotSub_trans (slotSub_append_pair_inv hbody).2.2.2.2 ih1
  | tpair _ _ _ Gamma2 _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSub_trans ih2 ih1
  | fst _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | snd _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | const _ _ Gamma _ _ =>
      simpa using slotSub_refl Gamma
  | tadd _ _ _ Gamma2 _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSub_trans ih2 ih1
  | tmul _ _ _ Gamma2 _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact slotSub_trans ih2 ih1
  | tsum _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | texpand _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | perform _ _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihBody ihClauses =>
      exact slotSub_trans ihClauses ihBody
  | tgrad =>
      simpa using slotSub_refl _
  | tvmap =>
      simpa using slotSub_refl _
  | loc _ _ Gamma _ _ _ =>
      simpa using slotSub_refl Gamma
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | nil _ _ Gamma _ _ =>
      simpa using slotSub_refl Gamma
  | cons _ _ Gamma2 Gamma3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihRest =>
      exact ihRest

theorem slotSub_singleton_cases
    {out : LinearCtx} {x : String} {t : Typ}
    (h : SlotSub out [(x, some t)]) :
    out = [(x, none)] ∨ out = [(x, some t)] := by
  cases out with
  | nil =>
      cases h
  | cons hd rest =>
      cases rest with
      | nil =>
          rcases hd with ⟨xOut, slotOut⟩
          rcases h with ⟨hname, hslot, hnil⟩
          cases hnil
          subst hname
          rcases hslot with hslot | hslot
          · exact Or.inl (by cases hslot; rfl)
          · exact Or.inr (by cases hslot; rfl)
      | cons hd2 rest2 =>
          cases h.2.2

theorem slotSub_pair_cases
    {out : LinearCtx} {x y : String} {tx ty : Typ}
    (h : SlotSub out [(x, some tx), (y, some ty)]) :
    out = [(x, none), (y, none)] ∨
      out = [(x, none), (y, some ty)] ∨
      out = [(x, some tx), (y, none)] ∨
      out = [(x, some tx), (y, some ty)] := by
  cases out with
  | nil =>
      cases h
  | cons hd1 rest =>
      cases rest with
      | nil =>
          cases h.2.2
      | cons hd2 rest' =>
          cases rest' with
          | nil =>
              rcases hd1 with ⟨xOut, slotOut1⟩
              rcases hd2 with ⟨yOut, slotOut2⟩
              rcases h with ⟨hx, hslot1, htail⟩
              rcases htail with ⟨hy, hslot2, hnil⟩
              cases hnil
              subst hx
              subst hy
              rcases hslot1 with hslot1 | hslot1 <;>
                rcases hslot2 with hslot2 | hslot2
              · exact Or.inl (by cases hslot1; cases hslot2; rfl)
              · exact Or.inr (Or.inl (by cases hslot1; cases hslot2; rfl))
              · exact Or.inr (Or.inr (Or.inl (by cases hslot1; cases hslot2; rfl)))
              · exact Or.inr (Or.inr (Or.inr (by cases hslot1; cases hslot2; rfl)))
          | cons hd3 rest'' =>
              rcases h with ⟨_hx, _hslot1, htail⟩
              rcases htail with ⟨_hy, _hslot2, hrest⟩
              cases hrest

theorem slotSub_split_target
    {out GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (h : SlotSub out (GammaPre ++ [(x, some t)] ++ GammaPost)) :
    ∃ outPre slot outPost,
      out = outPre ++ [(x, slot)] ++ outPost ∧
      SlotSub outPre GammaPre ∧
      (slot = none ∨ slot = some t) ∧
      SlotSub outPost GammaPost := by
  induction GammaPre generalizing out with
  | nil =>
      cases out with
      | nil =>
          cases h
      | cons hd tl =>
          rcases hd with ⟨xOut, slotOut⟩
          rcases h with ⟨hname, hslot, htail⟩
          subst hname
          exact ⟨[], slotOut, tl, by simp, slotSub_refl [], hslot, htail⟩
  | cons hd rest ih =>
      cases out with
      | nil =>
          cases h
      | cons hdOut outTail =>
          rcases h with ⟨hname, hslot, htail⟩
          rcases ih htail with
            ⟨outPre, slot, outPost, hout, hpre, hslotX, hpost⟩
          exact ⟨hdOut :: outPre, slot, outPost, by simp [hout],
            ⟨hname, hslot, hpre⟩, hslotX, hpost⟩

theorem noDupNames_of_sublist
    {Gamma' Gamma : LinearCtx}
    (hsub : List.Sublist Gamma' Gamma)
    (hnd : NoDupNames Gamma) :
    NoDupNames Gamma' := by
  unfold NoDupNames at hnd ⊢
  simpa [linearCtxDom] using (hsub.map Prod.fst).nodup hnd

theorem noDupNames_middle_fresh_suffix
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, some t)] ++ GammaPost)) :
    x ∉ linearCtxDom GammaPost := by
  intro hx
  unfold NoDupNames at hnd
  simp [linearCtxDom, List.nodup_append] at hnd
  have hx' : ∃ ty, (x, ty) ∈ GammaPost := by
    simpa [linearCtxDom] using hx
  rcases hx' with ⟨ty, hmem⟩
  exact hnd.2.1.1 ty hmem

theorem noDupNames_middle_fresh_prefix
    {GammaPre GammaPost : LinearCtx} {x : String} {t : Typ}
    (hnd : NoDupNames (GammaPre ++ [(x, some t)] ++ GammaPost)) :
    x ∉ linearCtxDom GammaPre := by
  have hsub : List.Sublist (GammaPre ++ [(x, some t)])
      (GammaPre ++ [(x, some t)] ++ GammaPost) := by
    induction GammaPost with
    | nil =>
        simp
    | cons p rest ih =>
        simp [List.append_assoc]
  have hnd' : NoDupNames (GammaPre ++ [(x, some t)]) :=
    noDupNames_of_sublist hsub hnd
  intro hx
  have hnd'' := hnd'
  unfold NoDupNames at hnd''
  simp [linearCtxDom, List.nodup_append] at hnd''
  have hx' : ∃ ty, (x, ty) ∈ GammaPre := by
    simpa [linearCtxDom] using hx
  rcases hx' with ⟨ty, hmem⟩
  exact (hnd''.2 x ty hmem) rfl

theorem lexical_of_names_eq
    {Gamma Gamma' : LinearCtx} {e : Term}
    (hEq : Gamma.map Prod.fst = Gamma'.map Prod.fst)
    (hlex : LexicallyScoped Gamma e) :
    LexicallyScoped Gamma' e := by
  rcases hlex with ⟨hnd, hdom, hws⟩
  refine ⟨?_, ?_, hws⟩
  · unfold NoDupNames at hnd ⊢
    simpa [linearCtxDom, hEq] using hnd
  · intro x hx
    have hx' : x ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom, hEq] using hx
    exact hdom x hx'

theorem lexical_output_of_typing
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma Gamma' : LinearCtx} {e : Term} {e' : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (hlex : LexicallyScoped Gamma e') :
    LexicallyScoped Gamma' e' :=
  lexical_of_names_eq (hasType_names_preserved h) hlex

private theorem mem_dom_of_mem_append_singleton_ne
    {Gamma : LinearCtx} {z x : String} {t : Typ}
    (hz : z ∈ linearCtxDom (Gamma ++ ([(x, some t)] : LinearCtx)))
    (hne : z ≠ x) :
    z ∈ linearCtxDom Gamma := by
  simp only [linearCtxDom, List.map_append, List.mem_append,
    List.map_cons, List.mem_cons, List.map_nil, List.not_mem_nil] at hz
  rcases hz with hz | hz
  · exact hz
  · rcases hz with hz | hz
    · exfalso
      exact hne hz
    · exact False.elim hz

private theorem mem_dom_of_mem_append_pair_ne
    {Gamma : LinearCtx} {z x y : String} {tx ty : Typ}
    (hz : z ∈ linearCtxDom (Gamma ++ ([(x, some tx), (y, some ty)] : LinearCtx)))
    (hne_x : z ≠ x) (hne_y : z ≠ y) :
    z ∈ linearCtxDom Gamma := by
  simp only [linearCtxDom, List.map_append, List.mem_append,
    List.map_cons, List.mem_cons, List.map_nil, List.not_mem_nil] at hz
  rcases hz with hz | hz
  · exact hz
  · rcases hz with hz | hz
    · exfalso
      exact hne_x hz
    · rcases hz with hz | hz
      · exfalso
        exact hne_y hz
      · exact False.elim hz

mutual

def has_type_free_vars_aux
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ z, z ∈ freeVars e → z ∈ linearCtxDom Gamma := by
  match h with
  | HasType.var _ _ Γpre Γpost x _ =>
      intro z hz
      simp [freeVars, linearCtxDom] at hz ⊢
      subst z
      simp [linearCtxDom]
  | HasType.unit _ _ _ =>
      intro z hz
      simp [freeVars] at hz
  | HasType.abs _ _ Γ1 _ x t1 _ _ body _ hBody =>
      intro z hz
      have hz' : z ∈ freeVars body ∧ z ≠ x := by
        simpa [freeVars] using hz
      have hz_ctx : z ∈ linearCtxDom (Γ1 ++ ([(x, some t1)] : LinearCtx)) :=
        has_type_free_vars_aux hBody z hz'.1
      exact mem_dom_of_mem_append_singleton_ne hz_ctx hz'.2
  | HasType.app _ _ Γ1 Γ2 _ _ _ _ _ _ _ _ h1 h2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux h1 z hz
      · exact has_type_linear_shrinks h1 z (has_type_free_vars_aux h2 z hz)
  | HasType.letBind _ _ Γ1 Γ2 _ x _ _ t1 _ _ _ _ h1 h2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux h1 z hz
      · have hz_ctx : z ∈ linearCtxDom (Γ2 ++ ([(x, some t1)] : LinearCtx)) :=
          has_type_free_vars_aux h2 z hz.1
        have hz_mid : z ∈ linearCtxDom Γ2 :=
          mem_dom_of_mem_append_singleton_ne hz_ctx hz.2
        exact has_type_linear_shrinks h1 z hz_mid
  | HasType.copy _ _ _ _ _ _ _ hBody =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.letpair _ _ Γ1 Γ2 _ x y _ _ t1 t2 _ _ _ _ _ h1 h2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux h1 z hz
      · have hz_ctx :
            z ∈ linearCtxDom (Γ2 ++ ([(x, some t1), (y, some t2)] : LinearCtx)) :=
          has_type_free_vars_aux h2 z hz.1
        have hz_mid : z ∈ linearCtxDom Γ2 :=
          mem_dom_of_mem_append_pair_ne hz_ctx hz.2.1 hz.2.2
        exact has_type_linear_shrinks h1 z hz_mid
  | HasType.tpair _ _ Γ1 Γ2 _ _ _ _ _ _ _ h1 h2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux h1 z hz
      · exact has_type_linear_shrinks h1 z (has_type_free_vars_aux h2 z hz)
  | HasType.fst _ _ _ _ _ _ _ _ hBody =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.snd _ _ _ _ _ _ _ _ hBody =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.const _ _ _ _ _ =>
      intro z hz
      simp [freeVars] at hz
  | HasType.tadd _ _ Γ1 Γ2 _ _ _ _ _ _ h1 h2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux h1 z hz
      · exact has_type_linear_shrinks h1 z (has_type_free_vars_aux h2 z hz)
  | HasType.tmul _ _ Γ1 Γ2 _ _ _ _ _ _ h1 h2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux h1 z hz
      · exact has_type_linear_shrinks h1 z (has_type_free_vars_aux h2 z hz)
  | HasType.tsum _ _ _ _ _ _ _ _ hBody _ =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.texpand _ _ _ _ _ _ _ _ hBody =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.uniformLike _ _ _ _ _ _ _ _ _ hBody =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.perform _ _ _ _ _ _ _ _ _ hBody _ =>
      intro z hz
      simpa [freeVars] using has_type_free_vars_aux hBody z hz
  | HasType.handle _ _ Γ1 Γ2 _ _ _ _ _ _ hBody _ _ _ hClauses =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact has_type_free_vars_aux hBody z hz
      · exact has_type_linear_shrinks hBody z (clauses_typed_free_vars_aux hClauses z hz)
  | HasType.tgrad _ _ Γ x ds _ _ _ _ hBody _ =>
      intro z hz
      simp [freeVars] at hz
      have hz_ctx :
            z ∈ linearCtxDom (Γ ++ ([(x, some (Typ.tensor ds))] : LinearCtx)) :=
        has_type_free_vars_aux hBody z hz.1
      exact mem_dom_of_mem_append_singleton_ne hz_ctx hz.2
  | HasType.tvmap _ _ Γ x t1 _ _ _ _ _ hBody =>
      intro z hz
      simp [freeVars] at hz
      have hz_ctx : z ∈ linearCtxDom (Γ ++ ([(x, some t1)] : LinearCtx)) :=
        has_type_free_vars_aux hBody z hz.1
      exact mem_dom_of_mem_append_singleton_ne hz_ctx hz.2
  | HasType.loc _ _ _ _ _ _ =>
      intro z hz
      simp [freeVars] at hz
  | HasType.subEff _ _ _ _ _ _ _ _ hBody _ =>
      exact has_type_free_vars_aux hBody

def clauses_typed_free_vars_aux
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls) :
    ∀ z, z ∈ freeVarsClauses cls → z ∈ linearCtxDom Gamma2 := by
  match h with
  | ClausesTyped.nil _ _ _ _ _ =>
      intro z hz
      simp [freeVarsClauses] at hz
  | ClausesTyped.cons _ _ Γ2 _ t tArg tRet epsR op x k _ _ _ _ hMatch hBody hRest =>
      intro z hz
      simp [freeVarsClauses] at hz
      rcases hz with hz | hz
      · have hz_ctx :
            z ∈ linearCtxDom
              (Γ2 ++ ([(x, some tArg), (k, some (Typ.arrow tRet t epsR))] : LinearCtx)) :=
          has_type_free_vars_aux hBody z hz.1
        exact mem_dom_of_mem_append_pair_ne hz_ctx hz.2.1 hz.2.2
      · exact clauses_typed_free_vars_aux hRest z hz

end

theorem has_type_free_vars
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ z, z ∈ freeVars e → z ∈ linearCtxDom Gamma :=
  has_type_free_vars_aux h

theorem clauses_typed_free_vars
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls) :
    ∀ z, z ∈ freeVarsClauses cls → z ∈ linearCtxDom Gamma2 :=
  clauses_typed_free_vars_aux h

theorem clauses_typed_same_ctx
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls) :
    Gamma2 = Gamma3 := by
  match h with
  | ClausesTyped.nil _ _ _ _ _ =>
      rfl
  | ClausesTyped.cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hRest =>
      exact clauses_typed_same_ctx hRest

theorem has_type_closed_term_of_closed_input
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma [] e t eps Gamma') :
    Closed e := by
  unfold Closed
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro z hz
  have hzΓ : z ∈ linearCtxDom ([] : LinearCtx) := has_type_free_vars h z hz
  simpa [linearCtxDom] using hzΓ

/-- A well-typed value under a closed input context produces a
    closed output context. (Lighter form — the `IsValue` premise is
    not actually used because `has_type_linear_shrinks` gives
    shrinkage for any derivation.) -/
theorem hasType_length_preserved
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    Gamma.length = Gamma'.length := by
  induction h using HasType.rec
    (motive_2 := fun _ _ Γ2 Γ3 _ _ _ _ => Γ2.length = Γ3.length) with
  | var _ _ Γpre Γpost _ _ => simp [List.length_append]
  | unit _ _ _ => rfl
  | abs _ _ _ _ _ _ _ _ _ _ _ ih =>
      simp [List.length_append] at ih; omega
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => omega
  | letBind _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      simp [List.length_append] at ih2; omega
  | copy _ _ _ _ _ _ _ _ ih => exact ih
  | letpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      simp [List.length_append] at ih2; omega
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => omega
  | fst _ _ _ _ _ _ _ _ _ ih => exact ih
  | snd _ _ _ _ _ _ _ _ _ ih => exact ih
  | const _ _ _ _ _ => rfl
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => omega
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => omega
  | tsum _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | texpand _ _ _ _ _ _ _ _ _ ih => exact ih
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | perform _ _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih_body ih_clauses => omega
  | tgrad => rfl
  | tvmap => rfl
  | loc _ _ _ _ _ _ => rfl
  | subEff _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | nil _ _ _ _ _ => rfl
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihRest => exact ihRest

theorem has_type_closed_output_of_closed_input
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma [] e t eps Gamma') : Gamma' = [] := by
  have hlen := hasType_length_preserved h
  simp at hlen
  exact List.eq_nil_of_length_eq_zero hlen.symm

/-- Values are effect-row polymorphic: a value typed at any effect row
    can be re-typed at any other effect row with the same type and
    contexts. -/
theorem HasType.value_eff_polymorphic
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {v : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma v t eps Gamma')
    (hv : IsValue v) :
    ∀ eps', HasType Delta Sigma Gamma v t eps' Gamma' := by
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | unit Δ S G =>
      intro eps'
      exact HasType.subEff Δ S G G _ _ [] eps'
        (HasType.unit Δ S G) (by intro op hop; cases hop)
  | abs Δ S G1 G2 y t1 t2 epsBody body slot h_body =>
      intro eps'
      exact HasType.subEff Δ S G1 G2 _ _ [] eps'
        (HasType.abs Δ S G1 G2 y t1 t2 epsBody body slot h_body)
        (by intro op hop; cases hop)
  | tpair Δ S G1 G2 G3 v1 v2 t1 t2 eps1 eps2 _hv1 _hv2 ih1 ih2 =>
      intro eps'
      cases hv with
      | pair _ _ hp1 hp2 =>
          have h1' := ih1 hp1 ([] : EffectRow)
          have h2' := ih2 hp2 ([] : EffectRow)
          exact HasType.subEff Δ S G1 G3 _ _ _ eps'
            (HasType.tpair Δ S G1 G2 G3 v1 v2 t1 t2 [] [] h1' h2')
            (by intro op hop; cases hop)
  | loc Δ S G ell t' hlook =>
      intro eps'
      exact HasType.subEff Δ S G G _ _ [] eps'
        (HasType.loc Δ S G ell t' hlook)
        (by intro op hop; cases hop)
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      intro eps'
      exact ih hv eps'
  | nil =>
      exact True.intro
  | cons =>
      exact True.intro
  | _ =>
      intro _
      cases hv

/-- Prefix a common outer context onto a typing derivation. The extra
    slots sit outside every local binder introduced by the derivation,
    so the constructor shape is preserved structurally. -/
theorem hasType_prefix_weaken
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (outer : LinearCtx) :
    HasType Delta Sigma (outer ++ Gamma) e t eps (outer ++ Gamma') := by
  induction h using HasType.rec
    (motive_2 := fun Δ_ S_ Γ2_ Γ3_ t_ εR_ cls_ _ =>
      ClausesTyped Δ_ S_ (outer ++ Γ2_) (outer ++ Γ3_) t_ εR_ cls_) with
  | var Δ_ S_ Γpre Γpost x tx =>
      simpa [List.append_assoc] using
        (HasType.var Δ_ S_ (outer ++ Γpre) Γpost x tx)
  | unit Δ_ S_ Γ_ =>
      simpa [List.append_assoc] using (HasType.unit Δ_ S_ (outer ++ Γ_))
  | abs Δ_ S_ Γ1 Γ2 x t1 t2 eps_ body slot hbody ih =>
      have ih' :
          HasType Δ_ S_ ((outer ++ Γ1) ++ [(x, some t1)]) body t2 eps_
            ((outer ++ Γ2) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih
      exact HasType.abs Δ_ S_ (outer ++ Γ1) (outer ++ Γ2)
        x t1 t2 eps_ body slot ih'
  | app Δ_ S_ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.app Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 t1 t2 eps_ eps1 eps2 ih1 ih2
  | letBind Δ_ S_ Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 slot h1 h2 ih1 ih2 =>
      have ih2' :
          HasType Δ_ S_ ((outer ++ Γ2) ++ [(x, some t1)]) e2 t2 eps2
            ((outer ++ Γ3) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih2
      exact HasType.letBind Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        x e1 e2 t1 t2 eps1 eps2 slot ih1 ih2'
  | copy Δ_ S_ Γ1 Γ2 e0 ds ep hbody ih =>
      exact HasType.copy Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds ep ih
  | letpair Δ_ S_ Γ1 Γ2 Γ3 x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY
      h1 h2 ih1 ih2 =>
      have ih2' :
          HasType Δ_ S_ ((outer ++ Γ2) ++ [(x, some t1), (y, some t2)]) e2 tr eps2
            ((outer ++ Γ3) ++ [(x, slotX), (y, slotY)]) := by
        simpa [List.append_assoc] using ih2
      exact HasType.letpair Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        x y e1 e2 t1 t2 tr eps1 eps2 slotX slotY ih1 ih2'
  | tpair Δ_ S_ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.tpair Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 t1 t2 eps1 eps2 ih1 ih2
  | fst Δ_ S_ Γ1 Γ2 e0 t1 t2 ep hbody ih =>
      exact HasType.fst Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 t1 t2 ep ih
  | snd Δ_ S_ Γ1 Γ2 e0 t1 t2 ep hbody ih =>
      exact HasType.snd Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 t1 t2 ep ih
  | const Δ_ S_ Γ_ v ds =>
      simpa [List.append_assoc] using (HasType.const Δ_ S_ (outer ++ Γ_) v ds)
  | tadd Δ_ S_ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.tadd Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 ds eps1 eps2 ih1 ih2
  | tmul Δ_ S_ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 ih1 ih2 =>
      exact HasType.tmul Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        e1 e2 ds eps1 eps2 ih1 ih2
  | tsum Δ_ S_ Γ1 Γ2 e0 ds d ep hbody hmem ih =>
      exact HasType.tsum Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds d ep ih hmem
  | texpand Δ_ S_ Γ1 Γ2 e0 ds d ep hbody ih =>
      exact HasType.texpand Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds d ep ih
  | uniformLike Δ_ S_ Γ1 Γ2 e0 ds lo hi ep hbody ih =>
      exact HasType.uniformLike Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) e0 ds lo hi ep ih
  | perform Δ_ S_ Γ1 Γ2 op e0 tArg tRet ep hbody hM ih =>
      exact HasType.perform Δ_ S_ (outer ++ Γ1) (outer ++ Γ2)
        op e0 tArg tRet ep ih hM
  | handle Δ_ S_ Γ1 Γ2 Γ3 body clauses t_ epsH epsB hb hSubsH hClsH
      hCover hcls ih_body ih_clauses =>
      exact HasType.handle Δ_ S_ (outer ++ Γ1) (outer ++ Γ2) (outer ++ Γ3)
        body clauses t_ epsH epsB ih_body hSubsH hClsH hCover ih_clauses
  | tgrad Δ_ S_ Γ_ x ds dsOut body ep slot hbody hsub_eff ih =>
      have ih' :
          HasType (Capability.diff :: Δ_) S_
            ((outer ++ Γ_) ++ [(x, some (Typ.tensor ds))])
            body (Typ.tensor dsOut) ep
            ((outer ++ Γ_) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih
      exact HasType.tgrad Δ_ S_ (outer ++ Γ_) x ds dsOut body ep slot ih' hsub_eff
  | tvmap Δ_ S_ Γ_ x t1 t2 body ep d slot hbody ih =>
      have ih' :
          HasType Δ_ S_ ((outer ++ Γ_) ++ [(x, some t1)]) body t2 ep
            ((outer ++ Γ_) ++ [(x, slot)]) := by
        simpa [List.append_assoc] using ih
      exact HasType.tvmap Δ_ S_ (outer ++ Γ_) x t1 t2 body ep d slot ih'
  | loc Δ_ S_ Γ_ ell tv hlook =>
      simpa [List.append_assoc] using (HasType.loc Δ_ S_ (outer ++ Γ_) ell tv hlook)
  | subEff Δ_ S_ Γ_ Γ'' e0 tv eps0 eps1 hbody hSub ih =>
      exact HasType.subEff Δ_ S_ (outer ++ Γ_) (outer ++ Γ'')
        e0 tv eps0 eps1 ih hSub
  | nil Δ_ S_ Γ2 t_ epsR_ =>
      simpa [List.append_assoc] using (ClausesTyped.nil Δ_ S_ (outer ++ Γ2) t_ epsR_)
  | cons Δ_ S_ Γ2 Γ3 t_ tArg tRet epsR_ op x k hb rest slotX slotK
      hmatch h_body h_rest ih_body ih_rest =>
      have ih_body' :
          HasType Δ_ S_
            ((outer ++ Γ2) ++ [(x, some tArg), (k, some (Typ.arrow tRet t_ epsR_))])
            hb t_ epsR_
            ((outer ++ Γ3) ++ [(x, slotX), (k, slotK)]) := by
        simpa [List.append_assoc] using ih_body
      exact ClausesTyped.cons Δ_ S_ (outer ++ Γ2) (outer ++ Γ3)
        t_ tArg tRet epsR_ op x k hb rest slotX slotK
        hmatch ih_body' ih_rest

theorem clausesTyped_prefix_weaken
    {Delta : CapCtx} {Sigma : StoreTyp}
    {Gamma2 Gamma3 : LinearCtx} {t : Typ} {epsR : EffectRow}
    {clauses : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR clauses)
    (outer : LinearCtx) :
    ClausesTyped Delta Sigma (outer ++ Gamma2) (outer ++ Gamma3) t epsR clauses := by
  induction clauses generalizing Gamma2 Gamma3 with
  | nil =>
      cases h
      simpa [List.append_assoc] using
        (ClausesTyped.nil Delta Sigma (outer ++ Gamma2) t epsR)
  | cons cl rest ih =>
      cases h with
      | cons _ _ _ _ _ tArg tRet _ op x k hb rest slotX slotK hmatch hBody hRest =>
          exact ClausesTyped.cons Delta Sigma (outer ++ Gamma2) (outer ++ Gamma3)
            t tArg tRet epsR op x k hb rest slotX slotK hmatch
            (by
              simpa [List.append_assoc] using
                hasType_prefix_weaken hBody outer)
            (ih hRest)

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
