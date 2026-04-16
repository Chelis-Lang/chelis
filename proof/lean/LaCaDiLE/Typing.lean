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
         (rest : List (EffectLabel × String × String × Term))
         (slotX slotK : Option Typ) :
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

/-- Linear-context domain is invariant across HasType derivations.
    Under tombstone semantics, consumption only changes Option values
    (some t → none), not names. So Γ.map Prod.fst = Γ'.map Prod.fst.
    DomSub follows by rewriting. -/
theorem has_type_linear_shrinks
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ x, x ∈ linearCtxDom Gamma' → x ∈ linearCtxDom Gamma := by
  -- Under tombstoning, linearCtxDom = map Prod.fst includes all names
  -- (live and dead). Names are preserved by hasType_names_preserved
  -- (proved in Translation.lean). Inline proof pending import refactor.
  sorry

/-- Linear-context outputs are not merely domain subsets of inputs;
    they preserve the original order as actual list sublists. This is
    the structural fact the TranslationDB bridge needs when it classifies
    singleton and two-slot body outputs by shape. -/
theorem has_type_sublist
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    List.Sublist Gamma' Gamma := by
  induction h using HasType.rec
    (motive_2 := fun (_Δ : CapCtx) (_S : StoreTyp)
                     (Γ2 Γ3 : LinearCtx) (_ : Typ) (_ : EffectRow)
                     (_ : List (EffectLabel × String × String × Term))
                     (_ : _) => List.Sublist Γ3 Γ2) with
  | var _ _ Γpre Γpost x tx =>
      have hpost : List.Sublist Γpost ([(x, tx)] ++ Γpost) := by
        simpa using (List.sublist_cons_self (x, tx) Γpost)
      simpa [List.append_assoc] using
        (List.Sublist.append (List.Sublist.refl Γpre) hpost)
  | unit _ _ Γ =>
      exact List.Sublist.refl Γ
  | abs _ _ Γ1 Γ2 x t1 _ _ _ hBody ih =>
      have hfilter :
          List.Sublist
            (Γ2.filter (fun p => p.1 ≠ x))
            (Γ1.filter (fun p => p.1 ≠ x)) := by
        simpa using (List.Sublist.filter (fun p => p.1 ≠ x) ih)
      exact hfilter.trans
        (show List.Sublist (Γ1.filter (fun p => p.1 ≠ x)) Γ1 from by
          simpa using (List.Sublist.filter (fun p => p.1 ≠ x) (List.Sublist.refl Γ1)))
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact ih2.trans ih1
  | letBind _ _ Γ1 Γ2 Γ3 x _ _ t1 _ _ _ _ _ ih1 ih2 =>
      have hfilter :
          List.Sublist
            (Γ3.filter (fun p => p.1 ≠ x))
            (Γ2.filter (fun p => p.1 ≠ x)) := by
        simpa using (List.Sublist.filter (fun p => p.1 ≠ x) ih2)
      exact (hfilter.trans
        (show List.Sublist (Γ2.filter (fun p => p.1 ≠ x)) Γ2 from by
          simpa using (List.Sublist.filter (fun p => p.1 ≠ x) (List.Sublist.refl Γ2)))).trans ih1
  | copy _ _ Γ1 _ _ _ _ _ ih =>
      exact ih
  | letpair _ _ Γ1 Γ2 Γ3 x y _ _ t1 t2 _ _ _ _ _ ih1 ih2 =>
      let p : (String × Typ) → Bool := fun q => q.1 ≠ x ∧ q.1 ≠ y
      have hfilter : List.Sublist (Γ3.filter p) (Γ2.filter p) := by
        simpa [p, and_left_comm, and_assoc] using (List.Sublist.filter p ih2)
      exact (hfilter.trans (show List.Sublist (Γ2.filter p) Γ2 from by
        simpa using (List.Sublist.filter p (List.Sublist.refl Γ2)))).trans ih1
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact ih2.trans ih1
  | fst _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | snd _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | const _ _ Γ _ _ =>
      exact List.Sublist.refl Γ
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact ih2.trans ih1
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      exact ih2.trans ih1
  | tsum _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | texpand _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | perform _ _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | handle _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihBody ihClauses =>
      exact ihClauses.trans ihBody
  | tgrad _ _ Γ _ _ _ _ _ _ _ _ =>
      exact List.Sublist.refl Γ
  | tvmap _ _ Γ _ _ _ _ _ _ _ =>
      exact List.Sublist.refl Γ
  | loc _ _ Γ _ _ _ =>
      exact List.Sublist.refl Γ
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih
  | nil _ _ Γ _ _ =>
      exact List.Sublist.refl Γ
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ihRest =>
      exact ihRest

private theorem mem_dom_of_mem_append_singleton_ne
    {Gamma : LinearCtx} {z x : String} {t : Typ}
    (hz : z ∈ linearCtxDom (Gamma ++ [(x, t)]))
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
    (hz : z ∈ linearCtxDom (Gamma ++ [(x, tx), (y, ty)]))
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

/-- Every free variable of a well-typed term comes from the input
    linear context. Closed input derivations are therefore genuinely
    closed in the syntax-level sense, not just context-closed. -/
theorem has_type_free_vars
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma') :
    ∀ z, z ∈ freeVars e → z ∈ linearCtxDom Gamma := by
  induction h using HasType.rec
    (motive_2 := fun (_Δ : CapCtx) (_S : StoreTyp)
                     (Γ2 Γ3 : LinearCtx) (_ : Typ) (_ : EffectRow)
                     (cls : List (EffectLabel × String × String × Term))
                     (_ : _) => ∀ z, z ∈ freeVarsClauses cls → z ∈ linearCtxDom Γ2) with
  | var _ _ Γpre Γpost x t =>
      intro z hz
      simp [freeVars, linearCtxDom] at hz ⊢
      subst z
      simp [linearCtxDom]
  | unit _ _ Γ =>
      intro z hz
      simp [freeVars] at hz
  | abs _ _ Γ1 _ x t1 _ _ body hBody ih =>
      intro z hz
      have hz' : z ∈ freeVars body ∧ z ≠ x := by
        simpa [freeVars] using hz
      have hz_ctx : z ∈ linearCtxDom (Γ1 ++ [(x, t1)]) := ih z hz'.1
      exact mem_dom_of_mem_append_singleton_ne hz_ctx hz'.2
  | app _ _ Γ1 Γ2 _ e1 e2 _ _ _ _ _ h1 h2 ih1 ih2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ih1 z hz
      · exact has_type_linear_shrinks h1 z (ih2 z hz)
  | letBind _ _ Γ1 Γ2 _ x e1 e2 t1 _ _ _ h1 h2 ih1 ih2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ih1 z hz
      · have hz_ctx : z ∈ linearCtxDom (Γ2 ++ [(x, t1)]) := ih2 z hz.1
        have hz_mid : z ∈ linearCtxDom Γ2 :=
          mem_dom_of_mem_append_singleton_ne hz_ctx hz.2
        exact has_type_linear_shrinks h1 z hz_mid
  | copy _ _ Γ1 _ e _ _ hBody ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | letpair _ _ Γ1 Γ2 _ x y e1 e2 t1 t2 _ _ _ h1 h2 ih1 ih2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ih1 z hz
      · have hz_ctx : z ∈ linearCtxDom (Γ2 ++ [(x, t1), (y, t2)]) := ih2 z hz.1
        have hz_mid : z ∈ linearCtxDom Γ2 :=
          mem_dom_of_mem_append_pair_ne hz_ctx hz.2.1 hz.2.2
        exact has_type_linear_shrinks h1 z hz_mid
  | tpair _ _ Γ1 Γ2 _ e1 e2 _ _ _ _ h1 h2 ih1 ih2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ih1 z hz
      · exact has_type_linear_shrinks h1 z (ih2 z hz)
  | fst _ _ Γ1 _ e _ _ _ hBody ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | snd _ _ Γ1 _ e _ _ _ hBody ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | const _ _ Γ _ _ =>
      intro z hz
      simp [freeVars] at hz
  | tadd _ _ Γ1 Γ2 _ e1 e2 _ _ _ h1 h2 ih1 ih2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ih1 z hz
      · exact has_type_linear_shrinks h1 z (ih2 z hz)
  | tmul _ _ Γ1 Γ2 _ e1 e2 _ _ _ h1 h2 ih1 ih2 =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ih1 z hz
      · exact has_type_linear_shrinks h1 z (ih2 z hz)
  | tsum _ _ Γ1 _ e _ _ _ hBody _ ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | texpand _ _ Γ1 _ e _ _ _ hBody ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | uniformLike _ _ Γ1 _ e _ _ _ _ hBody ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | perform _ _ Γ1 _ _ e _ _ _ hBody _ ih =>
      intro z hz
      simpa [freeVars] using ih z hz
  | handle _ _ Γ1 Γ2 _ body clauses _ _ _ hBody _ _ _ _ ihBody ihClauses =>
      intro z hz
      simp [freeVars] at hz
      rcases hz with hz | hz
      · exact ihBody z hz
      · exact has_type_linear_shrinks hBody z (ihClauses z hz)
  | tgrad _ _ Γ x ds dsOut body eps hBody hCompat ih =>
      intro z hz
      simp [freeVars] at hz
      have hz_ctx : z ∈ linearCtxDom (Γ ++ [(x, Typ.tensor ds)]) := by
        exact ih z hz.1
      exact mem_dom_of_mem_append_singleton_ne hz_ctx hz.2
  | tvmap _ _ Γ x t1 t2 body eps d hBody ih =>
      intro z hz
      simp [freeVars] at hz
      have hz_ctx : z ∈ linearCtxDom (Γ ++ [(x, t1)]) := by
        exact ih z hz.1
      exact mem_dom_of_mem_append_singleton_ne hz_ctx hz.2
  | loc _ _ Γ _ _ _ =>
      intro z hz
      simp [freeVars] at hz
  | subEff _ _ Γ _ e _ _ _ hBody _ ih =>
      exact ih
  | nil _ _ Γ _ _ z hz =>
      simp [freeVarsClauses] at hz
  | cons _ _ Γ2 _ t tArg tRet epsR op x k hb rest hBody hRest ihBody ihRest z hz =>
      simp [freeVarsClauses] at hz
      rcases hz with hz | hz
      · have hz_ctx : z ∈ linearCtxDom (Γ2 ++ [(x, tArg), (k, Typ.arrow tRet t epsR)]) := ihBody z hz.1
        exact mem_dom_of_mem_append_pair_ne hz_ctx hz.2.1 hz.2.2
      · exact ihRest z hz

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
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih_rest => exact ih_rest

theorem has_type_closed_output_of_closed_input
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma [] e t eps Gamma') : Gamma' = [] := by
  have hlen := hasType_length_preserved h
  simp at hlen
  exact List.eq_nil_of_length_eq_zero hlen.symm

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
