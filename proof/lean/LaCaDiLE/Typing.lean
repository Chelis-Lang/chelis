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

/-- The typing judgment of LaCaDiLE as an inductive relation.
    Each constructor corresponds to a typing rule in `figures/typing.tex`.
    The arguments are: capability context, input linear context, term,
    type, effect row, output linear context. -/
inductive HasType : CapCtx → LinearCtx → Term → Typ → EffectRow → LinearCtx → Prop

  -- T-Var: consume the tail binding from Γ.
  -- Paper: Δ; Γ, x:τ ⊢ x : τ ! ∅ ⊣ Γ
  | var
      (Delta : CapCtx) (Gamma : LinearCtx) (x : String) (t : Typ) :
      HasType Delta (Gamma ++ [(x, t)]) (Term.var x) t [] Gamma

  -- T-Unit.
  -- Paper: Δ; Γ ⊢ () : unit ! ∅ ⊣ Γ
  | unit
      (Delta : CapCtx) (Gamma : LinearCtx) :
      HasType Delta Gamma Term.unit Typ.unit [] Gamma

  -- T-Abs.
  -- Paper: Δ; Γ₁, x:τ₁ ⊢ e : τ₂ ! ε ⊣ Γ₂
  --        ─────────────────────────────────
  --        Δ; Γ₁ ⊢ λx:τ₁.e : τ₁ → τ₂ ! ε ⊣ Γ₂ \ {x}
  | abs
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (x : String) (t1 t2 : Typ) (eps : EffectRow) (e : Term) :
      HasType Delta (Gamma1 ++ [(x, t1)]) e t2 eps Gamma2 →
      HasType Delta Gamma1 (Term.abs x t1 e) (Typ.arrow t1 t2 eps) []
              (Gamma2.filter (fun p => p.1 ≠ x))

  -- T-App.
  -- Paper: Δ; Γ₁ ⊢ e₁ : τ₁ → τ₂ ! ε ⊣ Γ₂    Δ; Γ₂ ⊢ e₂ : τ₁ ! ε₂ ⊣ Γ₃
  --        ─────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ e₁ e₂ : τ₂ ! ε₁ ∪ ε₂ ∪ ε ⊣ Γ₃
  | app
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (t1 t2 : Typ) (eps eps1 eps2 : EffectRow) :
      HasType Delta Gamma1 e1 (Typ.arrow t1 t2 eps) eps1 Gamma2 →
      HasType Delta Gamma2 e2 t1 eps2 Gamma3 →
      HasType Delta Gamma1 (Term.app e1 e2) t2 (eps1 ++ eps2 ++ eps) Gamma3

  -- T-Let.
  -- Paper: Δ; Γ₁ ⊢ e₁ : τ₁ ! ε₁ ⊣ Γ₂     Δ; Γ₂, x:τ₁ ⊢ e₂ : τ₂ ! ε₂ ⊣ Γ₃
  --        ────────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ let x = e₁ in e₂ : τ₂ ! ε₁ ∪ ε₂ ⊣ Γ₃ \ {x}
  | letBind
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (x : String) (e1 e2 : Term) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasType Delta Gamma1 e1 t1 eps1 Gamma2 →
      HasType Delta (Gamma2 ++ [(x, t1)]) e2 t2 eps2 Gamma3 →
      HasType Delta Gamma1 (Term.letBind x e1 e2) t2 (eps1 ++ eps2)
              (Gamma3.filter (fun p => p.1 ≠ x))

  -- T-Copy.
  -- Paper: Δ; Γ₁ ⊢ e : τ ! ε ⊣ Γ₂
  --        ────────────────────────────────
  --        Δ; Γ₁ ⊢ copy(e) : τ ⊗ τ ! ε ⊣ Γ₂
  | copy
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (t : Typ) (eps : EffectRow) :
      HasType Delta Gamma1 e t eps Gamma2 →
      HasType Delta Gamma1 (Term.copy e) (Typ.pair t t) eps Gamma2

  -- T-LetPair.
  -- Paper: Δ; Γ₁ ⊢ e₁ : τ₁ ⊗ τ₂ ! ε₁ ⊣ Γ₂
  --        Δ; Γ₂, x:τ₁, x':τ₂ ⊢ e₂ : τ ! ε₂ ⊣ Γ₃
  --        ────────────────────────────────────────
  --        Δ; Γ₁ ⊢ let (x,x') = e₁ in e₂ : τ ! ε₁ ∪ ε₂ ⊣ Γ₃ \ {x, x'}
  | letpair
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (x y : String) (e1 e2 : Term) (t1 t2 t : Typ) (eps1 eps2 : EffectRow) :
      HasType Delta Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 →
      HasType Delta (Gamma2 ++ [(x, t1), (y, t2)]) e2 t eps2 Gamma3 →
      HasType Delta Gamma1 (Term.letpair x y e1 e2) t (eps1 ++ eps2)
              (Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ y))

  -- T-Pair.
  | tpair
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasType Delta Gamma1 e1 t1 eps1 Gamma2 →
      HasType Delta Gamma2 e2 t2 eps2 Gamma3 →
      HasType Delta Gamma1 (Term.pair e1 e2) (Typ.pair t1 t2) (eps1 ++ eps2) Gamma3

  -- T-Fst.
  | fst
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (t1 t2 : Typ) (eps : EffectRow) :
      HasType Delta Gamma1 e (Typ.pair t1 t2) eps Gamma2 →
      HasType Delta Gamma1 (Term.fst e) t1 eps Gamma2

  -- T-Snd.
  | snd
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (t1 t2 : Typ) (eps : EffectRow) :
      HasType Delta Gamma1 e (Typ.pair t1 t2) eps Gamma2 →
      HasType Delta Gamma1 (Term.snd e) t2 eps Gamma2

  -- T-Const.
  -- Paper: v is a scalar literal
  --        ───────────────────────────────────────────
  --        Δ; Γ ⊢ const(v, d̄) : tensor[d̄] ! ∅ ⊣ Γ
  | const
      (Delta : CapCtx) (Gamma : LinearCtx) (v : Float) (ds : DimList) :
      HasType Delta Gamma (Term.const v ds) (Typ.tensor ds) [] Gamma

  -- T-Add.
  | tadd
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (ds : DimList) (eps1 eps2 : EffectRow) :
      HasType Delta Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 →
      HasType Delta Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 →
      HasType Delta Gamma1 (Term.add e1 e2) (Typ.tensor ds) (eps1 ++ eps2) Gamma3

  -- T-Mul.
  | tmul
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (e1 e2 : Term) (ds : DimList) (eps1 eps2 : EffectRow) :
      HasType Delta Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 →
      HasType Delta Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 →
      HasType Delta Gamma1 (Term.mul e1 e2) (Typ.tensor ds) (eps1 ++ eps2) Gamma3

  -- T-Sum.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂    0 ≤ i < |d̄|
  --        ─────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ sum(e, i) : tensor[rem(d̄, i)] ! ε ⊣ Γ₂
  | tsum
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (i : Nat) (eps : EffectRow) :
      HasType Delta Gamma1 e (Typ.tensor ds) eps Gamma2 →
      i < ds.length →
      HasType Delta Gamma1 (Term.sum e i) (Typ.tensor (rem ds i)) eps Gamma2

  -- T-Expand.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂    0 ≤ i ≤ |d̄|
  --        ─────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ expand(e, i, k) : tensor[ins(d̄, i, k)] ! ε ⊣ Γ₂
  | texpand
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (i k : Nat) (eps : EffectRow) :
      HasType Delta Gamma1 e (Typ.tensor ds) eps Gamma2 →
      i ≤ ds.length →
      HasType Delta Gamma1 (Term.expand e i k) (Typ.tensor (ins ds i k)) eps Gamma2

  -- T-UniformLike.
  -- Paper: Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂
  --        ──────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ uniform_like(e, lo, hi) : tensor[d̄] ! ε ∪ {Random} ⊣ Γ₂
  | uniformLike
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (e : Term) (ds : DimList) (lo hi : Float) (eps : EffectRow) :
      HasType Delta Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasType Delta Gamma1 (Term.uniformLike e lo hi) (Typ.tensor ds)
              (eps ++ [EffectLabel.random]) Gamma2

  -- T-Perform.
  -- Paper: Δ; Γ₁ ⊢ e : τ_arg ! ε ⊣ Γ₂   op : τ_arg → τ_ret
  --        ────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ perform op(e) : τ_ret ! ε ∪ {op} ⊣ Γ₂
  --
  -- Phase 1 skeleton: `tArg` and `tRet` are free parameters rather than
  -- constrained by a per-operation signature table. Phase 2 will add a
  -- `def opSignature : EffectLabel → Typ × Typ` (e.g., `fail : unit → α`,
  -- `random : tensor[ds] → tensor[ds]`, `accum : (Loc × TensorVal) → unit`)
  -- and a premise requiring `(tArg, tRet) = opSignature op`. Round-3
  -- review (H1) flagged this weakness; tightening is a Phase 2 task.
  | perform
      (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
      (op : EffectLabel) (e : Term) (tArg tRet : Typ) (eps : EffectRow) :
      HasType Delta Gamma1 e tArg eps Gamma2 →
      HasType Delta Gamma1 (Term.perform op e) tRet (op :: eps) Gamma2

  -- T-Handle (single-clause form).
  -- Paper: Δ; Γ₁ ⊢ body : τ ! ε_b ⊣ Γ₂       op ∈ ε_b
  --        Δ; Γ₂, x:τ_arg, k:(τ_ret → τ ! ε_r) ⊢ handlerBody : τ ! ε_r ⊣ Γ₃
  --        where ε_r = ε_b.erase op
  --        ──────────────────────────────────────────────────────────────────
  --        Δ; Γ₁ ⊢ handle[{op}] body with {op(x,k) → handlerBody} : τ ! ε_r ⊣ Γ₃
  --
  -- Round-3 fix (M6): the body's effect row `epsB` is passed in
  -- unrestricted and the residual is computed via `List.erase`. The
  -- previous shape `op :: epsR` only matched when `op` was literally
  -- the first element, silently rejecting most well-formed programs.
  --
  -- Multi-clause handlers are deferred to Phase 2 (they quantify over
  -- the clause list).
  | handleSingle
      (Delta : CapCtx) (Gamma1 Gamma2 Gamma3 : LinearCtx)
      (body handlerBody : Term) (op : EffectLabel)
      (x k : String) (t tArg tRet : Typ) (epsB : EffectRow) :
      HasType Delta Gamma1 body t epsB Gamma2 →
      op ∈ epsB →
      HasType Delta
              (Gamma2 ++ [(x, tArg), (k, Typ.arrow tRet t (epsB.erase op))])
              handlerBody t (epsB.erase op) Gamma3 →
      HasType Delta Gamma1
              (Term.handle [op] body [(op, x, k, handlerBody)])
              t (epsB.erase op)
              (Gamma3.filter (fun p => p.1 ≠ x ∧ p.1 ≠ k))

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
      (Delta : CapCtx) (Gamma : LinearCtx)
      (x : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow) :
      HasType (Capability.diff :: Delta)
              (Gamma ++ [(x, Typ.tensor ds)]) e (Typ.tensor dsOut) eps Gamma →
      subsetEffRow eps DiffCompat = true →
      HasType Delta Gamma (Term.grad x (Typ.tensor ds) e)
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
      (Delta : CapCtx) (Gamma : LinearCtx)
      (x : String) (t1 t2 : Typ) (e : Term) (eps : EffectRow) (d : Dim) :
      HasType Delta (Gamma ++ [(x, t1)]) e t2 eps Gamma →
      HasType Delta Gamma (Term.vmap x t1 e)
              (Typ.arrow (addDim d t1) (addDim d t2) eps) [] Gamma

end LaCaDiLE
