-- LaCaDiLE/SyntaxDB.lean — de Bruijn mirror of `Syntax.Term`.
--
-- Track A contingency from the Phase 2 plan's risk register: the named
-- `subst_preserves_typing` proof hits a rigidity obstruction in every
-- binder case (the adjacent-swap exchange lemma is provably false in
-- this linear type system, so induction on `HasType` cannot thread a
-- shifting inserted binding through a binder). The standard resolution
-- is to move the substitution metatheory to a de Bruijn indexed
-- representation where binder cases reduce to rigid `Nat`-arithmetic.
--
-- This file provides the de Bruijn *term* representation plus the
-- standard `lift` / `subst_db` operations. It is deliberately
-- self-contained with respect to the named `Term` inductive in
-- `Syntax.lean`: the translation Term ↔ TermDB and the DB-side typing
-- judgment live in separate files (`TypingDB.lean`) so that the
-- untyped term layer can be re-used by any future metatheory work.
--
-- Convention: bound variables are represented as de Bruijn *indices*,
-- so every binder constructor drops the `String` name that the named
-- `Term` uses. Variable references `TermDB.var i` count lambdas from
-- the innermost enclosing binder outward, starting at 0. The types,
-- dimension lists, effect rows, and capability contexts in `Syntax.lean`
-- are name-free already (they have no variable references) and are
-- re-used verbatim.

import LaCaDiLE.Syntax

namespace LaCaDiLE

/-! ## Terms with de Bruijn indices -/

/-- De Bruijn-indexed mirror of `Term`. Constructors correspond 1:1 with
    `Term` constructors in `Syntax.lean`; the only shape change is that
    every binder drops its `String` name and every `Term.var x` becomes
    `TermDB.var i` where `i` is the de Bruijn index. Handler clauses
    are inlined as 3-tuples `(op, body)` rather than 4-tuples `(op, x, k,
    body)` because the argument and continuation binders are represented
    positionally: the body uses index 0 for the continuation and index
    1 for the argument. -/
inductive TermDB where
  -- core λ-calculus
  | var     (i : Nat)
  | abs     (t : Typ) (body : TermDB)
  | app     (e1 : TermDB) (e2 : TermDB)
  | letBind (e1 : TermDB) (e2 : TermDB)
  -- linearity primitives
  | copy    (e : TermDB)
  | letpair (e1 : TermDB) (e2 : TermDB)
  -- pair constructors / projections / unit
  | pair    (e1 : TermDB) (e2 : TermDB)
  | fst     (e : TermDB)
  | snd     (e : TermDB)
  | unit    : TermDB
  -- RISC primitives
  | const       (v : Float) (ds : DimList)
  | add         (e1 : TermDB) (e2 : TermDB)
  | mul         (e1 : TermDB) (e2 : TermDB)
  | sum         (e : TermDB) (i : Nat)
  | expand      (e : TermDB) (i : Nat) (k : Nat)
  | uniformLike (e : TermDB) (lo : Float) (hi : Float)
  -- AD / vectorization transforms
  | grad    (t : Typ) (tOut : Typ) (body : TermDB)
  | vmap    (t : Typ) (body : TermDB)
  -- effects: handler clauses as (op, body) pairs; body sees arg at index
  -- 1 and continuation at index 0 under its two positional binders
  | handle  (epsH : EffectRow) (body : TermDB)
            (clauses : List (EffectLabel × TermDB))
  | perform (op : EffectLabel) (e : TermDB)
  -- runtime location
  | loc     (ell : Loc)
  deriving Repr

-- ## Lifting (shift)
--
-- `liftAux c d e` adds `d` to every free variable index in `e` that
-- is at least `c` (the cutoff). The cutoff tracks how many binders
-- have been entered while traversing `e`; variables below `c` refer to
-- those binders and must not be shifted. This is the standard "shift"
-- of de Bruijn calculi, generalized to a `d`-place shift so we can
-- reuse it for both single-binder lifting (`d = 1`) and multi-binder
-- lifting (`d = 2` for handler clauses). Handler clauses are handled
-- by the companion `liftClausesAux`.

mutual

def liftAux (c d : Nat) : TermDB → TermDB
  | TermDB.var i =>
      if i < c then TermDB.var i else TermDB.var (i + d)
  | TermDB.abs t body => TermDB.abs t (liftAux (c + 1) d body)
  | TermDB.app e1 e2 => TermDB.app (liftAux c d e1) (liftAux c d e2)
  | TermDB.letBind e1 e2 =>
      TermDB.letBind (liftAux c d e1) (liftAux (c + 1) d e2)
  | TermDB.copy e => TermDB.copy (liftAux c d e)
  | TermDB.letpair e1 e2 =>
      TermDB.letpair (liftAux c d e1) (liftAux (c + 2) d e2)
  | TermDB.pair e1 e2 => TermDB.pair (liftAux c d e1) (liftAux c d e2)
  | TermDB.fst e => TermDB.fst (liftAux c d e)
  | TermDB.snd e => TermDB.snd (liftAux c d e)
  | TermDB.unit => TermDB.unit
  | TermDB.const v ds => TermDB.const v ds
  | TermDB.add e1 e2 => TermDB.add (liftAux c d e1) (liftAux c d e2)
  | TermDB.mul e1 e2 => TermDB.mul (liftAux c d e1) (liftAux c d e2)
  | TermDB.sum e i => TermDB.sum (liftAux c d e) i
  | TermDB.expand e i k => TermDB.expand (liftAux c d e) i k
  | TermDB.uniformLike e lo hi => TermDB.uniformLike (liftAux c d e) lo hi
  | TermDB.grad t tOut body =>
      TermDB.grad t tOut (liftAux (c + 1) d body)
  | TermDB.vmap t body => TermDB.vmap t (liftAux (c + 1) d body)
  | TermDB.handle epsH body clauses =>
      TermDB.handle epsH (liftAux c d body) (liftClausesAux c d clauses)
  | TermDB.perform op e => TermDB.perform op (liftAux c d e)
  | TermDB.loc ell => TermDB.loc ell

def liftClausesAux (c d : Nat) :
    List (EffectLabel × TermDB) → List (EffectLabel × TermDB)
  | [] => []
  | (op, hb) :: rest =>
      (op, liftAux (c + 2) d hb) :: liftClausesAux c d rest

end

/-- Single-binder lift: `lift e` shifts every free index in `e` by 1.
    Used when substituting under a binder: the substituted term, which
    originally lived in the outer context, needs every free variable
    shifted up to account for the new binder. -/
def lift (e : TermDB) : TermDB := liftAux 0 1 e

-- ## Capture-avoiding substitution
--
-- `substDBAux j v e` replaces every occurrence of `TermDB.var j` in `e`
-- with `v`, shifting `v` under each binder to keep indices consistent.
-- Occurrences of `TermDB.var i` with `i ≠ j` are renumbered so that
-- indices greater than `j` (which originally referred to the
-- now-removed binder `j` or outer variables) drop by one. The cutoff
-- `j` increases by the number of binders crossed, and `v` is lifted by
-- the same amount so that its free indices continue to point at the
-- correct outer bindings.

mutual

def substDBAux (j : Nat) (v : TermDB) : TermDB → TermDB
  | TermDB.var i =>
      if i = j then v
      else if i < j then TermDB.var i
      else TermDB.var (i - 1)
  | TermDB.abs t body =>
      TermDB.abs t (substDBAux (j + 1) (lift v) body)
  | TermDB.app e1 e2 =>
      TermDB.app (substDBAux j v e1) (substDBAux j v e2)
  | TermDB.letBind e1 e2 =>
      TermDB.letBind (substDBAux j v e1) (substDBAux (j + 1) (lift v) e2)
  | TermDB.copy e => TermDB.copy (substDBAux j v e)
  | TermDB.letpair e1 e2 =>
      TermDB.letpair (substDBAux j v e1)
                     (substDBAux (j + 2) (lift (lift v)) e2)
  | TermDB.pair e1 e2 =>
      TermDB.pair (substDBAux j v e1) (substDBAux j v e2)
  | TermDB.fst e => TermDB.fst (substDBAux j v e)
  | TermDB.snd e => TermDB.snd (substDBAux j v e)
  | TermDB.unit => TermDB.unit
  | TermDB.const c ds => TermDB.const c ds
  | TermDB.add e1 e2 =>
      TermDB.add (substDBAux j v e1) (substDBAux j v e2)
  | TermDB.mul e1 e2 =>
      TermDB.mul (substDBAux j v e1) (substDBAux j v e2)
  | TermDB.sum e i => TermDB.sum (substDBAux j v e) i
  | TermDB.expand e i k => TermDB.expand (substDBAux j v e) i k
  | TermDB.uniformLike e lo hi =>
      TermDB.uniformLike (substDBAux j v e) lo hi
  | TermDB.grad t tOut body =>
      TermDB.grad t tOut (substDBAux (j + 1) (lift v) body)
  | TermDB.vmap t body =>
      TermDB.vmap t (substDBAux (j + 1) (lift v) body)
  | TermDB.handle epsH body clauses =>
      TermDB.handle epsH (substDBAux j v body)
                    (substClausesDBAux j v clauses)
  | TermDB.perform op e => TermDB.perform op (substDBAux j v e)
  | TermDB.loc ell => TermDB.loc ell

def substClausesDBAux (j : Nat) (v : TermDB) :
    List (EffectLabel × TermDB) → List (EffectLabel × TermDB)
  | [] => []
  | (op, hb) :: rest =>
      (op, substDBAux (j + 2) (lift (lift v)) hb)
        :: substClausesDBAux j v rest

end

/-- Top-level substitution: replace the innermost-bound variable
    (index 0) of `e` with `v`. This is the de Bruijn analogue of the
    named `subst e v x` used by the operational semantics. -/
def substDB (v : TermDB) (e : TermDB) : TermDB := substDBAux 0 v e

/-! ## Elementary lemmas -/

/-- The cutoff never fires a shift at a strictly-smaller index. -/
theorem liftAux_var_lt {c d i : Nat} (h : i < c) :
    liftAux c d (TermDB.var i) = TermDB.var i := by
  simp [liftAux, h]

/-- Above the cutoff, `liftAux c d` adds `d` to the index. -/
theorem liftAux_var_ge {c d i : Nat} (h : ¬ i < c) :
    liftAux c d (TermDB.var i) = TermDB.var (i + d) := by
  simp [liftAux, h]

/-- A zero-shift at any cutoff is the identity on variables. -/
theorem liftAux_var_zero (c i : Nat) :
    liftAux c 0 (TermDB.var i) = TermDB.var i := by
  by_cases h : i < c
  · simp [liftAux, h]
  · simp [liftAux, h]

/-- Substitution at `var 0` returns the substituted term, regardless of
    what `v` looks like. -/
theorem substDBAux_var_eq (j : Nat) (v : TermDB) :
    substDBAux j v (TermDB.var j) = v := by
  simp [substDBAux]

/-- Substitution leaves smaller-index variables untouched. -/
theorem substDBAux_var_lt {j i : Nat} (v : TermDB) (h : i < j) :
    substDBAux j v (TermDB.var i) = TermDB.var i := by
  have hne : i ≠ j := Nat.ne_of_lt h
  simp [substDBAux, hne, h]

/-- Substitution decrements larger-index variables by one, reflecting
    the removal of the bound position. -/
theorem substDBAux_var_gt {j i : Nat} (v : TermDB) (h : j < i) :
    substDBAux j v (TermDB.var i) = TermDB.var (i - 1) := by
  have hne : i ≠ j := (Nat.ne_of_lt h).symm
  have hnlt : ¬ i < j := Nat.not_lt.mpr (Nat.le_of_lt h)
  simp [substDBAux, hne, hnlt]

end LaCaDiLE
