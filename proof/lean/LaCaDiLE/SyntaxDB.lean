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

/-! ## Runtime-location references

The DB term language preserves runtime locations verbatim under erasure
from named terms, so we mirror the named `locRefs` accounting here.
This is the right surface for substitution-side runtime-linearity
arguments, because the DB metatheory already tracks the exact slot
trajectory of substituted binders. -/
mutual

def locRefsDB : TermDB → List Loc
  | TermDB.var _ => []
  | TermDB.abs _ body => locRefsDB body
  | TermDB.app e1 e2 => locRefsDB e1 ++ locRefsDB e2
  | TermDB.letBind e1 e2 => locRefsDB e1 ++ locRefsDB e2
  | TermDB.copy e => locRefsDB e
  | TermDB.letpair e1 e2 => locRefsDB e1 ++ locRefsDB e2
  | TermDB.pair e1 e2 => locRefsDB e1 ++ locRefsDB e2
  | TermDB.fst e => locRefsDB e
  | TermDB.snd e => locRefsDB e
  | TermDB.unit => []
  | TermDB.const _ _ => []
  | TermDB.add e1 e2 => locRefsDB e1 ++ locRefsDB e2
  | TermDB.mul e1 e2 => locRefsDB e1 ++ locRefsDB e2
  | TermDB.sum e _ => locRefsDB e
  | TermDB.expand e _ _ => locRefsDB e
  | TermDB.uniformLike e _ _ => locRefsDB e
  | TermDB.grad _ _ body => locRefsDB body
  | TermDB.vmap _ body => locRefsDB body
  | TermDB.handle _ body clauses => locRefsDB body ++ locRefsClausesDB clauses
  | TermDB.perform _ e => locRefsDB e
  | TermDB.loc ell => [ell]

def locRefsClausesDB : List (EffectLabel × TermDB) → List Loc
  | [] => []
  | (_, hb) :: rest => locRefsDB hb ++ locRefsClausesDB rest

end

mutual

/-- DB-side active runtime-location surface. This mirrors
    `Syntax.activeLocRefs`: handler clause bodies remain dormant and do
    not contribute to the active footprint until a matching `perform`
    selects them. -/
def activeLocRefsDB : TermDB → List Loc
  | TermDB.var _ => []
  | TermDB.abs _ body => activeLocRefsDB body
  | TermDB.app e1 e2 => activeLocRefsDB e1 ++ activeLocRefsDB e2
  | TermDB.letBind e1 e2 => activeLocRefsDB e1 ++ activeLocRefsDB e2
  | TermDB.copy e => activeLocRefsDB e
  | TermDB.letpair e1 e2 => activeLocRefsDB e1 ++ activeLocRefsDB e2
  | TermDB.pair e1 e2 => activeLocRefsDB e1 ++ activeLocRefsDB e2
  | TermDB.fst e => activeLocRefsDB e
  | TermDB.snd e => activeLocRefsDB e
  | TermDB.unit => []
  | TermDB.const _ _ => []
  | TermDB.add e1 e2 => activeLocRefsDB e1 ++ activeLocRefsDB e2
  | TermDB.mul e1 e2 => activeLocRefsDB e1 ++ activeLocRefsDB e2
  | TermDB.sum e _ => activeLocRefsDB e
  | TermDB.expand e _ _ => activeLocRefsDB e
  | TermDB.uniformLike e _ _ => activeLocRefsDB e
  | TermDB.grad _ _ body => activeLocRefsDB body
  | TermDB.vmap _ body => activeLocRefsDB body
  | TermDB.handle _ body clauses => activeLocRefsDB body ++ activeLocRefsClausesDB clauses
  | TermDB.perform _ e => activeLocRefsDB e
  | TermDB.loc ell => [ell]

def activeLocRefsClausesDB : List (EffectLabel × TermDB) → List Loc
  | [] => []
  | _ :: rest => activeLocRefsClausesDB rest

end

@[simp] theorem activeLocRefsClausesDB_eq_nil
    (clauses : List (EffectLabel × TermDB)) :
    activeLocRefsClausesDB clauses = [] := by
  induction clauses with
  | nil =>
      simp [activeLocRefsClausesDB]
  | cons _ rest ih =>
      simp [activeLocRefsClausesDB, ih]

/-- DB-side active runtime linearity: no active explicit location is
    mentioned twice. -/
def ActiveRuntimeLinearDB (e : TermDB) : Prop :=
  (activeLocRefsDB e).Nodup

def RuntimeLinearDB (e : TermDB) : Prop :=
  (locRefsDB e).Nodup

mutual

/-- DB-side recursive closure of `ActiveRuntimeLinearDB`. This mirrors
    `Syntax.DeepActiveRuntimeLinear` and checks dormant handler clause
    bodies recursively rather than folding them into the active
    footprint directly. -/
def DeepActiveRuntimeLinearDB : TermDB → Prop
  | TermDB.var _ => True
  | TermDB.abs t body =>
      ActiveRuntimeLinearDB (TermDB.abs t body) ∧
      DeepActiveRuntimeLinearDB body
  | TermDB.app e1 e2 =>
      ActiveRuntimeLinearDB (TermDB.app e1 e2) ∧
      DeepActiveRuntimeLinearDB e1 ∧
      DeepActiveRuntimeLinearDB e2
  | TermDB.letBind e1 e2 =>
      ActiveRuntimeLinearDB (TermDB.letBind e1 e2) ∧
      DeepActiveRuntimeLinearDB e1 ∧
      DeepActiveRuntimeLinearDB e2
  | TermDB.copy e =>
      ActiveRuntimeLinearDB (TermDB.copy e) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.letpair e1 e2 =>
      ActiveRuntimeLinearDB (TermDB.letpair e1 e2) ∧
      DeepActiveRuntimeLinearDB e1 ∧
      DeepActiveRuntimeLinearDB e2
  | TermDB.pair e1 e2 =>
      ActiveRuntimeLinearDB (TermDB.pair e1 e2) ∧
      DeepActiveRuntimeLinearDB e1 ∧
      DeepActiveRuntimeLinearDB e2
  | TermDB.fst e =>
      ActiveRuntimeLinearDB (TermDB.fst e) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.snd e =>
      ActiveRuntimeLinearDB (TermDB.snd e) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.unit => True
  | TermDB.const _ _ => True
  | TermDB.add e1 e2 =>
      ActiveRuntimeLinearDB (TermDB.add e1 e2) ∧
      DeepActiveRuntimeLinearDB e1 ∧
      DeepActiveRuntimeLinearDB e2
  | TermDB.mul e1 e2 =>
      ActiveRuntimeLinearDB (TermDB.mul e1 e2) ∧
      DeepActiveRuntimeLinearDB e1 ∧
      DeepActiveRuntimeLinearDB e2
  | TermDB.sum e i =>
      ActiveRuntimeLinearDB (TermDB.sum e i) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.expand e i k =>
      ActiveRuntimeLinearDB (TermDB.expand e i k) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.uniformLike e lo hi =>
      ActiveRuntimeLinearDB (TermDB.uniformLike e lo hi) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.grad t tOut body =>
      ActiveRuntimeLinearDB (TermDB.grad t tOut body) ∧
      DeepActiveRuntimeLinearDB body
  | TermDB.vmap t body =>
      ActiveRuntimeLinearDB (TermDB.vmap t body) ∧
      DeepActiveRuntimeLinearDB body
  | TermDB.handle epsH body clauses =>
      ActiveRuntimeLinearDB (TermDB.handle epsH body clauses) ∧
      DeepActiveRuntimeLinearDB body ∧
      DeepActiveRuntimeLinearClausesDB clauses
  | TermDB.perform op e =>
      ActiveRuntimeLinearDB (TermDB.perform op e) ∧
      DeepActiveRuntimeLinearDB e
  | TermDB.loc _ => True

def DeepActiveRuntimeLinearClausesDB :
    List (EffectLabel × TermDB) → Prop
  | [] => True
  | (_, hb) :: rest =>
      DeepActiveRuntimeLinearDB hb ∧
      DeepActiveRuntimeLinearClausesDB rest

end

theorem mem_activeLocRefsDB_subset
    {e : TermDB} {ell : Loc}
    (h : ell ∈ activeLocRefsDB e) :
    ell ∈ locRefsDB e := by
  match e with
  | TermDB.var i =>
      simp [activeLocRefsDB] at h
  | TermDB.abs t body =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := body) h
  | TermDB.app e1 e2 =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefsDB_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefsDB_subset (e := e2) h)
  | TermDB.letBind e1 e2 =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefsDB_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefsDB_subset (e := e2) h)
  | TermDB.copy e =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.letpair e1 e2 =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefsDB_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefsDB_subset (e := e2) h)
  | TermDB.pair e1 e2 =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefsDB_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefsDB_subset (e := e2) h)
  | TermDB.fst e =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.snd e =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.unit =>
      simp [activeLocRefsDB] at h
  | TermDB.const c ds =>
      simp [activeLocRefsDB] at h
  | TermDB.add e1 e2 =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefsDB_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefsDB_subset (e := e2) h)
  | TermDB.mul e1 e2 =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefsDB_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefsDB_subset (e := e2) h)
  | TermDB.sum e i =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.expand e i k =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.uniformLike e lo hi =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.grad t tOut body =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := body) h
  | TermDB.vmap t body =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := body) h
  | TermDB.handle epsH body clauses =>
      simp [activeLocRefsDB, locRefsDB] at h ⊢
      exact Or.inl (mem_activeLocRefsDB_subset (e := body) h)
  | TermDB.perform op e =>
      simpa [activeLocRefsDB, locRefsDB] using
        mem_activeLocRefsDB_subset (e := e) h
  | TermDB.loc ell' =>
      simpa [activeLocRefsDB, locRefsDB] using h

theorem runtimeLinearDB_active
    {e : TermDB}
    (h : RuntimeLinearDB e) :
    ActiveRuntimeLinearDB e := by
  match e with
  | TermDB.var i =>
      simp [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] at h ⊢
  | TermDB.abs t body =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := body) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.app e1 e2 =>
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinearDB_active (e := e1) h1
      have hAct2 := runtimeLinearDB_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinearDB] using hAct1,
        by simpa [ActiveRuntimeLinearDB] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefsDB_subset hmem1) ell
        (mem_activeLocRefsDB_subset hmem2) rfl
  | TermDB.letBind e1 e2 =>
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinearDB_active (e := e1) h1
      have hAct2 := runtimeLinearDB_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinearDB] using hAct1,
        by simpa [ActiveRuntimeLinearDB] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefsDB_subset hmem1) ell
        (mem_activeLocRefsDB_subset hmem2) rfl
  | TermDB.copy e =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.letpair e1 e2 =>
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinearDB_active (e := e1) h1
      have hAct2 := runtimeLinearDB_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinearDB] using hAct1,
        by simpa [ActiveRuntimeLinearDB] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefsDB_subset hmem1) ell
        (mem_activeLocRefsDB_subset hmem2) rfl
  | TermDB.pair e1 e2 =>
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinearDB_active (e := e1) h1
      have hAct2 := runtimeLinearDB_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinearDB] using hAct1,
        by simpa [ActiveRuntimeLinearDB] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefsDB_subset hmem1) ell
        (mem_activeLocRefsDB_subset hmem2) rfl
  | TermDB.fst e =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.snd e =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.unit =>
      simp [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] at h ⊢
  | TermDB.const c ds =>
      simp [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] at h ⊢
  | TermDB.add e1 e2 =>
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinearDB_active (e := e1) h1
      have hAct2 := runtimeLinearDB_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinearDB] using hAct1,
        by simpa [ActiveRuntimeLinearDB] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefsDB_subset hmem1) ell
        (mem_activeLocRefsDB_subset hmem2) rfl
  | TermDB.mul e1 e2 =>
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinearDB_active (e := e1) h1
      have hAct2 := runtimeLinearDB_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinearDB] using hAct1,
        by simpa [ActiveRuntimeLinearDB] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefsDB_subset hmem1) ell
        (mem_activeLocRefsDB_subset hmem2) rfl
  | TermDB.sum e i =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.expand e i k =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.uniformLike e lo hi =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.grad t tOut body =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := body) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.vmap t body =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := body) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.handle epsH body clauses =>
      have hsplit : (locRefsDB body ++ locRefsClausesDB clauses).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨hBody, _hClauses, _hsep⟩
      have hActBody := runtimeLinearDB_active (e := body) hBody
      simpa [ActiveRuntimeLinearDB, activeLocRefsDB, activeLocRefsClausesDB] using hActBody
  | TermDB.perform op e =>
      simpa [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] using
        runtimeLinearDB_active (e := e) (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.loc ell =>
      simp [RuntimeLinearDB, ActiveRuntimeLinearDB, locRefsDB, activeLocRefsDB] at h ⊢

mutual

theorem runtimeLinearDB_deepActive
    : ∀ {e : TermDB}, RuntimeLinearDB e -> DeepActiveRuntimeLinearDB e
  | TermDB.var _, _ => by
      simp [DeepActiveRuntimeLinearDB]
  | TermDB.abs t body, h => by
      refine ⟨runtimeLinearDB_active h, ?_⟩
      exact runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)
  | TermDB.app e1 e2, h => by
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive h1,
        runtimeLinearDB_deepActive h2⟩
  | TermDB.letBind e1 e2, h => by
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive h1,
        runtimeLinearDB_deepActive h2⟩
  | TermDB.copy e, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.letpair e1 e2, h => by
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive h1,
        runtimeLinearDB_deepActive h2⟩
  | TermDB.pair e1 e2, h => by
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive h1,
        runtimeLinearDB_deepActive h2⟩
  | TermDB.fst e, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.snd e, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.unit, _ => by
      simp [DeepActiveRuntimeLinearDB]
  | TermDB.const _ _, _ => by
      simp [DeepActiveRuntimeLinearDB]
  | TermDB.add e1 e2, h => by
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive h1,
        runtimeLinearDB_deepActive h2⟩
  | TermDB.mul e1 e2, h => by
      have hsplit : (locRefsDB e1 ++ locRefsDB e2).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive h1,
        runtimeLinearDB_deepActive h2⟩
  | TermDB.sum e _, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.expand e _ _, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.uniformLike e _ _, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.grad _ _ body, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.vmap _ body, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.handle _ body clauses, h => by
      have hsplit : (locRefsDB body ++ locRefsClausesDB clauses).Nodup := by
        simpa [RuntimeLinearDB, locRefsDB] using h
      rcases List.nodup_append.mp hsplit with ⟨hBody, hClauses, _hsep⟩
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive hBody,
        runtimeLinearClausesDB_deepActive hClauses⟩
  | TermDB.perform _ e, h => by
      exact ⟨runtimeLinearDB_active h,
        runtimeLinearDB_deepActive (by simpa [RuntimeLinearDB, locRefsDB] using h)⟩
  | TermDB.loc _, _ => by
      simp [DeepActiveRuntimeLinearDB]
termination_by
  e _ => sizeOf e

theorem runtimeLinearClausesDB_deepActive
    : ∀ {clauses : List (EffectLabel × TermDB)},
      (locRefsClausesDB clauses).Nodup ->
      DeepActiveRuntimeLinearClausesDB clauses
  | [], _ => by
      simp [DeepActiveRuntimeLinearClausesDB]
  | (op, hb) :: rest, h => by
      have hsplit : (locRefsDB hb ++ locRefsClausesDB rest).Nodup := by
        simpa [locRefsClausesDB] using h
      rcases List.nodup_append.mp hsplit with ⟨hHead, hTail, _hsep⟩
      exact ⟨runtimeLinearDB_deepActive hHead, runtimeLinearClausesDB_deepActive hTail⟩
termination_by
  clauses _ => sizeOf clauses

end

/-! ## Lifting (shift)
--
--
-- `liftAux c d e` adds `d` to every free variable index in `e` that
-- is at least `c` (the cutoff). The cutoff tracks how many binders
-- have been entered while traversing `e`; variables below `c` refer to
-- those binders and must not be shifted. This is the standard "shift"
-- of de Bruijn calculi, generalized to a `d`-place shift so we can
-- reuse it for both single-binder lifting (`d = 1`) and multi-binder
-- lifting (`d = 2` for handler clauses). Handler clauses are handled
-- by the companion `liftClausesAux`. -/

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

/-! ## Runtime-location flow through lifting and substitution -/

mutual

theorem locRefsDB_liftAux
    (c d : Nat) (e : TermDB) :
    locRefsDB (liftAux c d e) = locRefsDB e := by
  match e with
  | TermDB.var i =>
      by_cases h : i < c <;> simp [liftAux, locRefsDB, h]
  | TermDB.abs t body =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux (c + 1) d body]
  | TermDB.app e1 e2 =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e1, locRefsDB_liftAux c d e2]
  | TermDB.letBind e1 e2 =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e1, locRefsDB_liftAux (c + 1) d e2]
  | TermDB.copy e =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.letpair e1 e2 =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e1, locRefsDB_liftAux (c + 2) d e2]
  | TermDB.pair e1 e2 =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e1, locRefsDB_liftAux c d e2]
  | TermDB.fst e =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.snd e =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.unit =>
      simp [liftAux, locRefsDB]
  | TermDB.const v ds =>
      simp [liftAux, locRefsDB]
  | TermDB.add e1 e2 =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e1, locRefsDB_liftAux c d e2]
  | TermDB.mul e1 e2 =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e1, locRefsDB_liftAux c d e2]
  | TermDB.sum e i =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.expand e i k =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.uniformLike e lo hi =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.grad t tOut body =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux (c + 1) d body]
  | TermDB.vmap t body =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux (c + 1) d body]
  | TermDB.handle epsH body clauses =>
      simp [liftAux, locRefsDB,
        locRefsDB_liftAux c d body, locRefsClausesDB_liftClausesAux c d clauses]
  | TermDB.perform op e =>
      simp [liftAux, locRefsDB, locRefsDB_liftAux c d e]
  | TermDB.loc ell =>
      simp [liftAux, locRefsDB]

theorem locRefsClausesDB_liftClausesAux
    (c d : Nat) (clauses : List (EffectLabel × TermDB)) :
    locRefsClausesDB (liftClausesAux c d clauses) = locRefsClausesDB clauses := by
  match clauses with
  | [] =>
      simp [liftClausesAux, locRefsClausesDB]
  | (_, hb) :: rest =>
      simp [liftClausesAux, locRefsClausesDB,
        locRefsDB_liftAux (c + 2) d hb, locRefsClausesDB_liftClausesAux c d rest]

end

@[simp] theorem locRefsDB_lift (e : TermDB) :
    locRefsDB (lift e) = locRefsDB e := by
  simpa [lift] using locRefsDB_liftAux 0 1 e

mutual

theorem activeLocRefsDB_liftAux
    (c d : Nat) (e : TermDB) :
    activeLocRefsDB (liftAux c d e) = activeLocRefsDB e := by
  match e with
  | TermDB.var i =>
      by_cases h : i < c <;> simp [liftAux, activeLocRefsDB, h]
  | TermDB.abs t body =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux (c + 1) d body]
  | TermDB.app e1 e2 =>
      simp [liftAux, activeLocRefsDB,
        activeLocRefsDB_liftAux c d e1, activeLocRefsDB_liftAux c d e2]
  | TermDB.letBind e1 e2 =>
      simp [liftAux, activeLocRefsDB,
        activeLocRefsDB_liftAux c d e1, activeLocRefsDB_liftAux (c + 1) d e2]
  | TermDB.copy e =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.letpair e1 e2 =>
      simp [liftAux, activeLocRefsDB,
        activeLocRefsDB_liftAux c d e1, activeLocRefsDB_liftAux (c + 2) d e2]
  | TermDB.pair e1 e2 =>
      simp [liftAux, activeLocRefsDB,
        activeLocRefsDB_liftAux c d e1, activeLocRefsDB_liftAux c d e2]
  | TermDB.fst e =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.snd e =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.unit =>
      simp [liftAux, activeLocRefsDB]
  | TermDB.const v ds =>
      simp [liftAux, activeLocRefsDB]
  | TermDB.add e1 e2 =>
      simp [liftAux, activeLocRefsDB,
        activeLocRefsDB_liftAux c d e1, activeLocRefsDB_liftAux c d e2]
  | TermDB.mul e1 e2 =>
      simp [liftAux, activeLocRefsDB,
        activeLocRefsDB_liftAux c d e1, activeLocRefsDB_liftAux c d e2]
  | TermDB.sum e i =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.expand e i k =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.uniformLike e lo hi =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.grad t tOut body =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux (c + 1) d body]
  | TermDB.vmap t body =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux (c + 1) d body]
  | TermDB.handle epsH body clauses =>
      simp [liftAux, activeLocRefsDB, activeLocRefsClausesDB,
        activeLocRefsDB_liftAux c d body,
        activeLocRefsClausesDB_liftClausesAux c d clauses]
  | TermDB.perform op e =>
      simp [liftAux, activeLocRefsDB, activeLocRefsDB_liftAux c d e]
  | TermDB.loc ell =>
      simp [liftAux, activeLocRefsDB]

theorem activeLocRefsClausesDB_liftClausesAux
    (c d : Nat) (clauses : List (EffectLabel × TermDB)) :
    activeLocRefsClausesDB (liftClausesAux c d clauses) = activeLocRefsClausesDB clauses := by
  match clauses with
  | [] =>
      simp [liftClausesAux, activeLocRefsClausesDB]
  | _ :: rest =>
      simp [liftClausesAux, activeLocRefsClausesDB,
        activeLocRefsClausesDB_liftClausesAux c d rest]

end

@[simp] theorem activeLocRefsDB_lift (e : TermDB) :
    activeLocRefsDB (lift e) = activeLocRefsDB e := by
  simpa [lift] using activeLocRefsDB_liftAux 0 1 e

mutual

theorem mem_locRefsDB_substDBAux
    (j : Nat) (v e : TermDB) (ell : Loc)
    (hmem : ell ∈ locRefsDB (substDBAux j v e)) :
    ell ∈ locRefsDB e ∨ ell ∈ locRefsDB v := by
  match e with
  | TermDB.var i =>
      by_cases hij : i = j
      · simp [substDBAux, locRefsDB, hij] at hmem ⊢
        exact hmem
      · by_cases hlt : i < j
        · simp [substDBAux, locRefsDB, hij, hlt] at hmem
        · simp [substDBAux, locRefsDB, hij, hlt] at hmem
  | TermDB.abs t body =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact Or.elim (mem_locRefsDB_substDBAux (j + 1) (lift v) body ell hmem)
        Or.inl (fun h => Or.inr (by simpa using h))
  | TermDB.app e1 e2 =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v e1 ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsDB_substDBAux j v e2 ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | TermDB.letBind e1 e2 =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v e1 ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsDB_substDBAux (j + 1) (lift v) e2 ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr (by simpa using h))
  | TermDB.copy e =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.letpair e1 e2 =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v e1 ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsDB_substDBAux (j + 2) (lift (lift v)) e2 ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr (by simpa using h))
  | TermDB.pair e1 e2 =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v e1 ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsDB_substDBAux j v e2 ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | TermDB.fst e =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.snd e =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.unit =>
      simp [substDBAux, locRefsDB] at hmem
  | TermDB.const c ds =>
      simp [substDBAux, locRefsDB] at hmem
  | TermDB.add e1 e2 =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v e1 ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsDB_substDBAux j v e2 ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | TermDB.mul e1 e2 =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v e1 ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsDB_substDBAux j v e2 ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | TermDB.sum e i =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.expand e i k =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.uniformLike e lo hi =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.grad t tOut body =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact Or.elim (mem_locRefsDB_substDBAux (j + 1) (lift v) body ell hmem)
        Or.inl (fun h => Or.inr (by simpa using h))
  | TermDB.vmap t body =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact Or.elim (mem_locRefsDB_substDBAux (j + 1) (lift v) body ell hmem)
        Or.inl (fun h => Or.inr (by simpa using h))
  | TermDB.handle epsH body clauses =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim (mem_locRefsDB_substDBAux j v body ell hmem)
          (fun h => Or.inl (Or.inl h)) (fun h => Or.inr h)
      · exact Or.elim (mem_locRefsClausesDB_substClausesDBAux j v clauses ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)
  | TermDB.perform op e =>
      simp [substDBAux, locRefsDB] at hmem ⊢
      exact mem_locRefsDB_substDBAux j v e ell hmem
  | TermDB.loc ell' =>
      simp [substDBAux, locRefsDB] at hmem
      exact Or.inl (by simp [locRefsDB, hmem])

theorem mem_locRefsClausesDB_substClausesDBAux
    (j : Nat) (v : TermDB) (clauses : List (EffectLabel × TermDB)) (ell : Loc)
    (hmem : ell ∈ locRefsClausesDB (substClausesDBAux j v clauses)) :
    ell ∈ locRefsClausesDB clauses ∨ ell ∈ locRefsDB v := by
  match clauses with
  | [] =>
      simp [substClausesDBAux, locRefsClausesDB] at hmem
  | (_, hb) :: rest =>
      simp [substClausesDBAux, locRefsClausesDB] at hmem ⊢
      rcases hmem with hmem | hmem
      · exact Or.elim
          (mem_locRefsDB_substDBAux (j + 2) (lift (lift v)) hb ell hmem)
          (fun h => Or.inl (Or.inl h))
          (fun h => Or.inr (by simpa using h))
      · exact Or.elim (mem_locRefsClausesDB_substClausesDBAux j v rest ell hmem)
          (fun h => Or.inl (Or.inr h)) (fun h => Or.inr h)

end

end LaCaDiLE
