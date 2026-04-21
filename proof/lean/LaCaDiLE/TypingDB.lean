-- LaCaDiLE/TypingDB.lean — de Bruijn mirror of `HasType`.
--
-- Track A Option C. The linear context is
-- `List (Option Typ)`: position `i` is either `some t` (the slot is
-- "live" and holds a linear binding of type `t`) or `none` (the slot
-- is "dead", already consumed). Consumption marks a slot as `none`
-- instead of removing it, so every typing rule preserves the length
-- of the linear context end-to-end. Under this shape, `shiftAt j`
-- is uniformly correct and the substitution lemmas have clean
-- uniform signatures.
--
-- Binder rules add new `some t` slots at the *head* of the context
-- (innermost de Bruijn index 0). The body's output context has a
-- head slot in either state (`some t` if unused, `none` if consumed)
-- and the outer rule peels it off. Head-cons is chosen so that
-- `insertAt (j+1)` on a cons naturally peels the head, which makes
-- weakening under binders structurally trivial.

import LaCaDiLE.SyntaxDB
import LaCaDiLE.Typing

namespace LaCaDiLE

/-- De Bruijn linear context: position `i` is an optional type.
    `some t` means the slot is live; `none` means consumed. Every
    typing rule preserves the length of the context end-to-end. -/
abbrev LinearCtxDB := List (Option Typ)

/-- Drop the first `k` entries off a DB linear context. Used in
    binder cases to peel the binder slots added at the head after
    the body is typed. -/
def LinearCtxDB.dropHead (Γ : LinearCtxDB) (k : Nat) : LinearCtxDB :=
  Γ.drop k

-- DB-side typing judgment. See `HasType` in `Typing.lean` for the
-- named version; this mirrors it 1:1 over `TermDB`.
mutual
inductive HasTypeDB :
    CapCtx → StoreTyp → LinearCtxDB → TermDB → Typ → EffectRow →
    LinearCtxDB → Prop
  -- T-Var-DB: the slot at position `i` must be live (`some t`); the
  -- output context marks that slot `none`. Length preserved.
  | var
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma : LinearCtxDB) (i : Nat) (t : Typ) :
      Gamma[i]? = some (some t) →
      HasTypeDB Delta Sigma Gamma (TermDB.var i) t [] (Gamma.set i none)

  | unit
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtxDB) :
      HasTypeDB Delta Sigma Gamma TermDB.unit Typ.unit [] Gamma

  -- T-Abs-DB: body is typed under a new slot at the head (innermost
  -- de Bruijn index 0). The body's output may leave it `some t1` or
  -- `none`; the outer rule discards that head slot.
  | abs
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (slot : Option Typ)
      (t1 t2 : Typ) (eps : EffectRow) (body : TermDB) :
      HasTypeDB Delta Sigma (some t1 :: Gamma1) body t2 eps
                (slot :: Gamma2) →
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
      (slot : Option Typ)
      (e1 e2 : TermDB) (t1 t2 : Typ) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 t1 eps1 Gamma2 →
      HasTypeDB Delta Sigma (some t1 :: Gamma2) e2 t2 eps2
                (slot :: Gamma3) →
      HasTypeDB Delta Sigma Gamma1 (TermDB.letBind e1 e2) t2
                (EffectRow.union eps1 eps2) Gamma3

  | copy
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (ds : DimList) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
      HasTypeDB Delta Sigma Gamma1 (TermDB.copy e)
                (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) eps Gamma2

  -- T-LetPair-DB: body sees two new head slots. Convention matches
  -- `SyntaxDB.substDBAux`: innermost (pos 0) is the second component,
  -- next (pos 1) is the first component.
  | letpair
      (Delta : CapCtx) (Sigma : StoreTyp)
      (Gamma1 Gamma2 Gamma3 : LinearCtxDB)
      (slot1 slot2 : Option Typ)
      (e1 e2 : TermDB) (t1 t2 t : Typ) (eps1 eps2 : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 →
      HasTypeDB Delta Sigma (some t2 :: some t1 :: Gamma2) e2 t eps2
                (slot1 :: slot2 :: Gamma3) →
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

  | tsum
      (Delta : CapCtx) (Sigma : StoreTyp) (Gamma1 Gamma2 : LinearCtxDB)
      (e : TermDB) (ds : DimList) (i : Nat) (eps : EffectRow) :
      HasTypeDB Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 →
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
      (slot : Option Typ)
      (ds dsOut : DimList) (body : TermDB) (eps : EffectRow) :
      HasTypeDB (Capability.diff :: Delta) Sigma
                (some (Typ.tensor ds) :: Gamma) body (Typ.tensor dsOut) eps
                (slot :: Gamma) →
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
      (slot : Option Typ)
      (t1 t2 : Typ) (body : TermDB) (eps : EffectRow) (d : Dim) :
      HasTypeDB Delta Sigma (some t1 :: Gamma) body t2 eps
                (slot :: Gamma) →
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

/-- Handler-clause typing on DB terms. Each clause's body sees `tArg`
    and the resumption `(tRet → t ! εR)` as two new slots at the tail. -/
inductive ClausesTypedDB :
    CapCtx → StoreTyp → LinearCtxDB → LinearCtxDB → Typ → EffectRow →
    List (EffectLabel × TermDB) → Prop
  | nil (Delta : CapCtx) (Sigma : StoreTyp) (Gamma2 : LinearCtxDB)
        (t : Typ) (epsR : EffectRow) :
        ClausesTypedDB Delta Sigma Gamma2 Gamma2 t epsR []
  | cons (Delta : CapCtx) (Sigma : StoreTyp)
         (Gamma2 Gamma3 : LinearCtxDB)
         (slot1 slot2 : Option Typ)
         (t tArg tRet : Typ) (epsR : EffectRow)
         (op : EffectLabel) (hb : TermDB)
         (rest : List (EffectLabel × TermDB)) :
         OpSigMatch op tArg tRet →
         HasTypeDB Delta Sigma
                   (some (Typ.arrow tRet t epsR) :: some tArg :: Gamma2)
                   hb t epsR
                   (slot1 :: slot2 :: Gamma3) →
         ClausesTypedDB Delta Sigma Gamma2 Gamma3 t epsR rest →
         ClausesTypedDB Delta Sigma Gamma2 Gamma3 t epsR ((op, hb) :: rest)

end

end LaCaDiLE
