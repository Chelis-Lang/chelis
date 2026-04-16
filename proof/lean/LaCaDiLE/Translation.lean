-- LaCaDiLE/Translation.lean — Term ↔ TermDB translation bridge.
--
-- The named substitution theorem in `Substitution.lean` is blocked on
-- the adjacent-swap exchange lemma (provably false). The de Bruijn
-- metatheory in `SubstitutionDB.lean` proves the same theorem cleanly
-- via `HasTypeDB.rec` with the trajectory-parameterized motive. To
-- reuse that result, this file defines a forward translation
-- `termToDB` that maps `Term` to `TermDB` against a naming environment,
-- plus the context translation `ctxToDB` that maps `LinearCtx` to
-- `LinearCtxDB`. Downstream waves prove that
-- `HasType → HasTypeDB ∘ termToDB` and derive the named
-- `subst_preserves_typing` as a corollary of the DB theorem.
--
-- Wave 5o: forward translation skeleton (this file).
-- Wave 5p: `hasType_to_hasTypeDB` forward preservation theorem.
-- Wave 5q: `subst` commutes with `termToDB`, closing the named
--          `subst_preserves_typing` via the DB theorem.
--
-- Conventions:
--
-- * `env : List String` is an innermost-to-outermost binding list.
--   A new binder `x` prepends: `x :: env`. The de Bruijn index of a
--   variable `x` is its position in `env` (0 = innermost).
-- * `ctxToDB Γ` maps each `(_, t) : String × Typ` in the named linear
--   context to `some t` in the DB context. The names are dropped;
--   positions are preserved.
-- * For `Term.sum e d` and `Term.expand e d`, the DB representation
--   uses a `Nat` index. Since `HasTypeDB.tsum`/`HasTypeDB.texpand`
--   both quantify the output `DimList` existentially (`∀ ds' : DimList,
--   True → ...`), the index value does not constrain typing. We pick
--   `0` as a placeholder — it does not affect
--   `subst_preserves_typing`'s closure.
-- * `Term.handle` clauses are 4-tuples `(op, xArg, kCont, body)`. The
--   DB encoding drops `xArg` and `kCont`, turning the body into a
--   position-2 term with the continuation at index 0 and the argument
--   at index 1. `clauseToDB` emits this positional shape.
-- * `Term.grad x t tOut body` discards `x` and runs the body under a
--   new head binder. `Term.vmap x t body` discards `x` similarly.

import LaCaDiLE.Syntax
import LaCaDiLE.SyntaxDB
import LaCaDiLE.Typing
import LaCaDiLE.TypingDB

namespace LaCaDiLE

/-- De Bruijn index of `x` in the innermost-to-outermost binding
    list `env`. Returns `env.length` (out-of-range) if `x` is not
    bound — callers should only invoke `termToDB` on closed terms
    w.r.t. the passed environment. -/
def envIndex : List String → String → Nat
  | [], _ => 0
  | y :: rest, x => if x = y then 0 else envIndex rest x + 1

/-- Context translation: drop names, wrap each type as a live slot. -/
def ctxToDB : LinearCtx → LinearCtxDB
  | [] => []
  | (_, t) :: rest => some t :: ctxToDB rest

@[simp] theorem ctxToDB_length (Γ : LinearCtx) :
    (ctxToDB Γ).length = Γ.length := by
  induction Γ with
  | nil => rfl
  | cons p rest ih =>
    cases p with
    | mk x t =>
      simp [ctxToDB, ih]

@[simp] theorem ctxToDB_nil : ctxToDB [] = [] := rfl

@[simp] theorem ctxToDB_cons (x : String) (t : Typ) (Γ : LinearCtx) :
    ctxToDB ((x, t) :: Γ) = some t :: ctxToDB Γ := rfl

mutual

/-- Forward translation of a named Term into a de Bruijn TermDB
    against the environment env. Mutual-recursive with
    clausesToDB to handle handler-clause bodies. -/
def termToDB (env : List String) : Term → TermDB
  | Term.var x => TermDB.var (envIndex env x)
  | Term.abs x t body => TermDB.abs t (termToDB (x :: env) body)
  | Term.app e1 e2 => TermDB.app (termToDB env e1) (termToDB env e2)
  | Term.letBind x e1 e2 =>
      TermDB.letBind (termToDB env e1) (termToDB (x :: env) e2)
  | Term.copy e => TermDB.copy (termToDB env e)
  | Term.letpair x y e1 e2 =>
      -- HasTypeDB.letpair's body sees `some t2 :: some t1 :: Γ`, so the
      -- innermost binder (index 0) corresponds to the *second*
      -- component — which is `y` in the named `letpair x y e1 e2`
      -- shape. Environment prepend order: y then x.
      TermDB.letpair (termToDB env e1) (termToDB (y :: x :: env) e2)
  | Term.pair e1 e2 => TermDB.pair (termToDB env e1) (termToDB env e2)
  | Term.fst e => TermDB.fst (termToDB env e)
  | Term.snd e => TermDB.snd (termToDB env e)
  | Term.unit => TermDB.unit
  | Term.const v ds => TermDB.const v ds
  | Term.add e1 e2 => TermDB.add (termToDB env e1) (termToDB env e2)
  | Term.mul e1 e2 => TermDB.mul (termToDB env e1) (termToDB env e2)
  | Term.sum e _ => TermDB.sum (termToDB env e) 0
  | Term.expand e _ => TermDB.expand (termToDB env e) 0 0
  | Term.uniformLike e lo hi =>
      TermDB.uniformLike (termToDB env e) lo hi
  | Term.grad x t tOut body =>
      TermDB.grad t tOut (termToDB (x :: env) body)
  | Term.vmap x t body =>
      TermDB.vmap t (termToDB (x :: env) body)
  | Term.handle epsH body clauses =>
      TermDB.handle epsH (termToDB env body) (clausesToDB env clauses)
  | Term.perform op e => TermDB.perform op (termToDB env e)
  | Term.loc ell => TermDB.loc ell

/-- Clause-list translation. Each clause `(op, xArg, kCont, body)`
    becomes `(op, termToDB (kCont :: xArg :: env) body)`. The prepend
    order matches `HasTypeDB.ClausesTypedDB.cons`, which types the
    body under `some (arrow tRet t epsR) :: some tArg :: Γ` — so the
    innermost binder (index 0) is the continuation (`kCont`) and
    index 1 is the argument (`xArg`). -/
def clausesToDB (env : List String) :
    List (EffectLabel × String × String × Term) →
    List (EffectLabel × TermDB)
  | [] => []
  | (op, xArg, kCont, body) :: rest =>
      (op, termToDB (kCont :: xArg :: env) body) :: clausesToDB env rest

end

@[simp] theorem clausesToDB_nil (env : List String) :
    clausesToDB env [] = [] := rfl

@[simp] theorem clausesToDB_cons (env : List String)
    (op : EffectLabel) (xArg kCont : String) (body : Term)
    (rest : List (EffectLabel × String × String × Term)) :
    clausesToDB env ((op, xArg, kCont, body) :: rest) =
    (op, termToDB (kCont :: xArg :: env) body) :: clausesToDB env rest :=
  rfl

end LaCaDiLE
