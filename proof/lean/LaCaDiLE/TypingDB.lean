-- LaCaDiLE/TypingDB.lean — de Bruijn mirror of `HasType`.
--
-- Track A contingency. The named `HasType` judgment in `Typing.lean`
-- uses a `LinearCtx = List (String × Typ)` whose string components
-- make the substitution metatheory proof hit a rigidity obstruction
-- in every binder case (the adjacent-swap exchange lemma is false in
-- a linear type system). This file mirrors `HasType` over `TermDB`
-- using a name-free, position-indexed linear context `List Typ`, so
-- that binder cases reduce to rigid `Nat`-arithmetic on the inserted
-- position rather than fighting a nonexistent name.
--
-- Every constructor corresponds 1:1 to a constructor of `HasType`.
-- Only two shape changes occur:
--
-- 1. The linear context is `List Typ` instead of `List (String × Typ)`.
--    Position `i` in the list holds the type bound by the `i`-th
--    outstanding linear binder, counted from the innermost. A
--    `Term.var x` named-lookup becomes `TermDB.var i` position-lookup.
--
-- 2. Every binder case that, in the named version, does
--    `Γ.filter (fun p => p.1 ≠ x)` at the *output* context instead
--    removes the binder position by (a) extending the *input* context
--    with the binder type at position 0 (via `t :: Γ`) and (b)
--    letting the body's output context `Γout` record whether the
--    body consumed the binder. The constructor then peels the
--    tail-removed entry, using the same convention as
--    `SyntaxDB.substDBAux`.
--
-- Store typings (`StoreTyp`), capability contexts (`CapCtx`), effect
-- rows, and dimension lists are name-free in `Syntax.lean` already
-- and are reused verbatim.

import LaCaDiLE.SyntaxDB
import LaCaDiLE.Typing

namespace LaCaDiLE

/-- De Bruijn linear context: position `i` is the type of the `i`-th
    outstanding linear binder, counted from the innermost. -/
abbrev LinearCtxDB := List Typ

/-- Peel the head `k` entries off a DB linear context. Used in binder
    cases to "consume" the newly-introduced binder positions after the
    body is typed. -/
def LinearCtxDB.dropHead (Γ : LinearCtxDB) (k : Nat) : LinearCtxDB :=
  Γ.drop k

-- DB-side typing judgment. See `HasType` in `Typing.lean` for the
-- named version; this mirrors it 1:1 over `TermDB`.
mutual
inductive HasTypeDB :
    CapCtx → StoreTyp → LinearCtxDB → TermDB → Typ → EffectRow →
    LinearCtxDB → Prop
  -- T-Var-DB: consume position `Γpre.length` of the context. The
  -- named rule's `Γpre ++ [(x, t)] ++ Γpost` becomes
  -- `Γpre ++ [t] ++ Γpost`; the output context is `Γpre ++ Γpost`
  -- (the consumed binding is removed in place).
  | var
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma_pre Gamma_post : LinearCtxDB) (t : Typ) :
      HasTypeDB Delta Sigma (Gamma_pre ++ [t] ++ Gamma_post)
                (TermDB.var Gamma_pre.length) t [] (Gamma_pre ++ Gamma_post)

  | unit
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtxDB) :
      HasTypeDB Delta Sigma Gamma TermDB.unit Typ.unit [] Gamma

  -- T-Abs-DB: body is typed in `t1 :: Γ1`; the output context of the
  -- body drops its head to recover the outer context. The named
  -- version uses `.filter (· ≠ x)`; here removal is positional.
  | abs
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (t1 t2 : Typ) (eps : EffectRow) (body : TermDB) :
      HasTypeDB Delta Sigma (t1 :: Gamma1) body t2 eps (t1 :: Gamma2) →
      HasTypeDB Delta Sigma Gamma1 (TermDB.abs t1 body)
                (Typ.arrow t1 t2 eps) [] Gamma2

  | app
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (e1 e2 : TermDB) (t1 t2 : Typ) (eps eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 (Typ.arrow t1 t2 eps) eps1 Gamma2 →
      HasTypeDB Delta Sigma Gamma2 e2 t1 eps2 Gamma3 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.app e1 e2) t2
                (EffectRow.union (EffectRow.union eps1 eps2) eps) Gamma3

  | letBind
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (e1 e2 : TermDB) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 t1 eps1 Gamma2 →
      HasTypeDB Delta Sigma (t1 :: Gamma2) e2 t2 eps2 (t1 :: Gamma3) →
      HasTypeDB Delta Sigma Gamma1 (TermDB.letBind e1 e2) t2
                (EffectRow.union eps1 eps2) Gamma3

  | copy
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (ds : DimList) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.copy e)
                (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) eps Gamma2

  -- T-LetPair-DB: body sees two new positions. Convention matches
  -- `SyntaxDB.substDBAux`: the body's innermost position 0 is `y`
  -- (the second component), position 1 is `x` (the first component).
  -- So the body's input context is `t2 :: t1 :: Γ2` and its output
  -- context must still have the two binder types at the head.
  | letpair
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (e1 e2 : TermDB) (t1 t2 t : Typ) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 →
      HasTypeDB Delta Sigma (t2 :: t1 :: Gamma2) e2 t eps2
                (t2 :: t1 :: Gamma3) →
      HasTypeDB Delta Sigma Gamma1 (TermDB.letpair e1 e2) t
                (EffectRow.union eps1 eps2) Gamma3

  | tpair
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (e1 e2 : TermDB) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 t1 eps1 Gamma2 →
      HasTypeDB Delta Sigma Gamma2 e2 t2 eps2 Gamma3 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.pair e1 e2) (Typ.pair t1 t2)
                (EffectRow.union eps1 eps2) Gamma3

  | fst
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (t1 t2 : Typ) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.pair t1 t2) eps Gamma2 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.fst e) t1 eps Gamma2

  | snd
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (t1 t2 : Typ) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.pair t1 t2) eps Gamma2 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.snd e) t2 eps Gamma2

  | const
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtxDB)
      (v : Float) (ds : DimList) :
      HasTypeDB Delta Sigma Gamma (TermDB.const v ds) (Typ.tensor ds) [] Gamma

  | tadd
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (e1 e2 : TermDB) (ds : DimList) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 →
      HasTypeDB Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.add e1 e2) (Typ.tensor ds)
                (EffectRow.union eps1 eps2) Gamma3

  | tmul
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (e1 e2 : TermDB) (ds : DimList) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 (Typ.tensor ds) eps1 Gamma2 →
      HasTypeDB Delta Sigma Gamma2 e2 (Typ.tensor ds) eps2 Gamma3 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.mul e1 e2) (Typ.tensor ds)
                (EffectRow.union eps1 eps2) Gamma3

  -- Note: SyntaxDB's `TermDB.sum`/`TermDB.expand` still carry a `Nat`
  -- positional index rather than a `Dim` name, matching the TermDB
  -- shape declared in SyntaxDB.lean. The DB-side typing rules below
  -- mirror the *pre-Stage-1* integer-index form because that's what
  -- SyntaxDB encodes. Translations from the named (`Dim`-indexed)
  -- form to DB therefore go through the positional index computed
  -- from the named dimension's position in `ds`.
  | tsum
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (ds : DimList) (i : Nat) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      -- Existential on the resulting dim list: whatever shape the
      -- translation produces, we just record it. The precise dim
      -- arithmetic lives on the named side.
      ∀ ds' : DimList, True →
      HasTypeDB Delta Sigma Gamma1 (TermDB.sum e i) (Typ.tensor ds') eps Gamma2

  | texpand
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (ds : DimList) (i k : Nat) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      ∀ ds' : DimList, True →
      HasTypeDB Delta Sigma Gamma1 (TermDB.expand e i k) (Typ.tensor ds')
                eps Gamma2

  | uniformLike
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (ds : DimList) (lo hi : Float) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.uniformLike e lo hi)
                (Typ.tensor ds)
                (EffectRow.union eps [EffectLabel.random]) Gamma2

  | perform
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (op : EffectLabel) (e : TermDB) (tArg tRet : Typ) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e tArg eps Gamma2 →
      OpSigMatch op tArg tRet →
      HasTypeDB Delta Sigma Gamma1 (TermDB.perform op e) tRet
                (EffectRow.union [op] eps) Gamma2

  | handle
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (body : TermDB) (clauses : List (EffectLabel × TermDB))
      (t : Typ) (epsH epsB : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 body t epsB Gamma2 →
      (∀ op ∈ epsH, op ∈ epsB) →
      (∀ cl ∈ clauses, cl.1 ∈ epsH) →
      (∀ op ∈ epsH, ∃ cl ∈ clauses, cl.1 = op) →
      ClausesTypedDB Delta Sigma Gamma2 Gamma3 t
                     (EffectRow.removeOps epsB epsH) clauses →
      HasTypeDB Delta Sigma Gamma1
                (TermDB.handle epsH body clauses)
                t (EffectRow.removeOps epsB epsH) Gamma3

  | tgrad
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtxDB)
      (ds dsOut : DimList) (body : TermDB) (eps : EffectRow) :
      HasTypeDB (Capability.diff :: Delta) Sigma
                (Typ.tensor ds :: Gamma) body (Typ.tensor dsOut) eps
                (Typ.tensor ds :: Gamma) →
      subsetEffRow eps DiffCompat = true →
      HasTypeDB Delta Sigma Gamma
                (TermDB.grad (Typ.tensor ds) (Typ.tensor dsOut) body)
                (Typ.arrow
                  (Typ.tensor ds)
                  (Typ.arrow (Typ.tensor dsOut) (Typ.tensor ds) eps)
                  [])
                [] Gamma

  | tvmap
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtxDB)
      (t1 t2 : Typ) (body : TermDB) (eps : EffectRow) (d : Dim) :
      HasTypeDB Delta Sigma (t1 :: Gamma) body t2 eps (t1 :: Gamma) →
      HasTypeDB Delta Sigma Gamma (TermDB.vmap t1 body)
                (Typ.arrow (addDim d t1) (addDim d t2) eps) [] Gamma

  | loc
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtxDB)
      (ell : Loc) (t : Typ) :
      storeTypLookup Sigma ell = some t →
      HasTypeDB Delta Sigma Gamma (TermDB.loc ell) t [] Gamma

  | subEff
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtxDB)
      (e : TermDB) (t : Typ) (eps eps' : EffectRow) :
      HasTypeDB Delta Sigma Gamma e t eps Gamma' →
      SubEffRow eps eps' →
      HasTypeDB Delta Sigma Gamma e t eps' Gamma'

/-- Handler-clause typing on DB terms. Mirrors `ClausesTyped` in
    `Typing.lean`; each clause's body sees `tArg` at position 1 and
    `(tRet → t ! εR)` at position 0, matching the two-positional-
    binder convention of `SyntaxDB.liftClausesAux`. -/
inductive ClausesTypedDB :
    CapCtx → StoreTyp → LinearCtxDB → LinearCtxDB → Typ → EffectRow →
    List (EffectLabel × TermDB) → Prop
  | nil (Delta : CapCtx) (Sigma : StoreTyp) (Gamma2 : LinearCtxDB)
        (t : Typ) (epsR : EffectRow) :
        ClausesTypedDB Delta Sigma Gamma2 Gamma2 t epsR []
  | cons (Delta : CapCtx) (Sigma : StoreTyp)
         (Gamma2 Gamma3 : LinearCtxDB)
         (t tArg tRet : Typ) (epsR : EffectRow)
         (op : EffectLabel) (hb : TermDB)
         (rest : List (EffectLabel × TermDB)) :
         HasTypeDB Delta Sigma
                   (Typ.arrow tRet t epsR :: tArg :: Gamma2)
                   hb t epsR
                   (Typ.arrow tRet t epsR :: tArg :: Gamma3) →
         ClausesTypedDB Delta Sigma Gamma2 Gamma3 t epsR rest →
         ClausesTypedDB Delta Sigma Gamma2 Gamma3 t epsR ((op, hb) :: rest)

end

end LaCaDiLE
