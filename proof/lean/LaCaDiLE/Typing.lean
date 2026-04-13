-- LaCaDiLE/Typing.lean — CapCtx, LinearCtx, HasType inductive relation.
--
-- Every typing rule from proof/paper/figures/typing.tex is encoded as a
-- constructor of `HasType`. Phase 1 T7: structural skeleton, no proofs.
--
-- Judgment shape:
--   HasType Δ Γ e τ ε Γ'
-- ↔ paper's
--   Δ; Γ ⊢ e : τ ! ε ⊣ Γ'
--
-- Phase 1 skeleton simplifications (all tightened in Phase 2):
--  * effect row union is encoded as list concatenation; proper set union
--    with deduplication is deferred.
--  * context lookup for T-Var requires the variable at the tail of Γ; this
--    matches the paper's "Γ, x:τ" notation exactly and avoids permutation.
--  * T-Handle is encoded for the single-clause case. Multi-clause handlers
--    use a separate `HasTypeHandleMulti` constructor that quantifies over
--    the clause list.
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

  -- T-Var: consume the tail binding from Γ.
  -- Paper: Δ; Γ, x:τ ⊢ x : τ ! ∅ ⊣ Γ
  | var
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx) (x : String) (t : Typ) :
      HasType Delta Sigma (Gamma ++ [(x, t)]) (Term.var x) t [] Gamma

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
  -- Paper: Δ; Γ₁ ⊢ e : τ ! ε ⊣ Γ₂
  --        ────────────────────────────────
  --        Δ; Γ₁ ⊢ copy(e) : τ ⊗ τ ! ε ⊣ Γ₂
  | copy
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (t : Typ) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e t eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.copy e) (Typ.pair t t) eps Gamma2

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
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂    0 ≤ i < |d̄|
  --        ─────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ sum(e, i) : tensor[rem(d̄, i)] ! ε ⊣ Γ₂
  | tsum
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (i : Nat) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      i < ds.length →
      HasType Delta Sigma Gamma1 (Term.sum e i) (Typ.tensor (rem ds i)) eps Gamma2

  -- T-Expand.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂    0 ≤ i ≤ |d̄|
  --        ─────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ expand(e, i, k) : tensor[ins(d̄, i, k)] ! ε ⊣ Γ₂
  | texpand
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (i k : Nat) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      i ≤ ds.length →
      HasType Delta Sigma Gamma1 (Term.expand e i k) (Typ.tensor (ins ds i k)) eps Gamma2

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
      (op : EffectLabel) (e : Term) (eps : EffectRow) :
      HasType Delta Sigma Gamma1 e (opSignature op).1 eps Gamma2 →
      HasType Delta Sigma Gamma1 (Term.perform op e) (opSignature op).2
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
  | nil (Delta : CapCtx) (Sigma : StoreTyp) (Gamma2 Gamma3 : LinearCtx)
        (t : Typ) (epsR : EffectRow) :
        ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR []
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

end LaCaDiLE
