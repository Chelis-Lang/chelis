-- LaCaDiLE/SubstitutionDB.lean — substitution metatheory on HasTypeDB.
--
-- Under Option C (see TypingDB.lean), the linear context is
-- `List (Option Typ)` whose length is preserved by every typing
-- rule. In this representation `shiftAt j` is uniformly correct and
-- the substitution lemmas admit clean uniform signatures.
--
-- This file establishes the Option C helper infrastructure
-- (positional `insertAt` on `List (Option Typ)`, plus the six
-- structural helpers the weakening proof needs) and states both
-- `weakening_insert_db` and `subst_preserves_typing_db` as proof
-- obligations in the closing doc block.
--
-- ### Why doc obligations and not inline proofs
--
-- Both theorems are mutually recursive with `ClausesTypedDB`
-- weakening / substitution — the `handle` case of each requires a
-- ClausesTypedDB-level analog that is proved by the same induction.
-- A clean discharge uses a mutual `theorem` block with explicit
-- termination; that is mechanical but bulky. For the non-handle
-- subset the proofs are direct structural inductions whose shape is
-- sketched in the doc block below.

import LaCaDiLE.SyntaxDB
import LaCaDiLE.TypingDB

namespace LaCaDiLE

/-! ## Positional insertion on DB linear contexts -/

/-- Insert an optional slot at position `j` of `Γ`. If `j ≥ Γ.length`
    the slot is appended. Recursion is primary on `j` so that `j = 0`
    reduces definitionally. -/
def LinearCtxDB.insertAt : Nat → Option Typ → LinearCtxDB → LinearCtxDB
  | 0, s, Γ => s :: Γ
  | _ + 1, s, [] => [s]
  | j + 1, s, x :: xs => x :: LinearCtxDB.insertAt j s xs

@[simp] theorem LinearCtxDB.insertAt_zero (Γ : LinearCtxDB) (s : Option Typ) :
    LinearCtxDB.insertAt 0 s Γ = s :: Γ := rfl

@[simp] theorem LinearCtxDB.insertAt_cons_succ
    (x : Option Typ) (xs : LinearCtxDB) (j : Nat) (s : Option Typ) :
    LinearCtxDB.insertAt (j + 1) s (x :: xs) =
      x :: LinearCtxDB.insertAt j s xs := rfl

@[simp] theorem LinearCtxDB.insertAt_nil_succ
    (j : Nat) (s : Option Typ) :
    LinearCtxDB.insertAt (j + 1) s [] = [s] := rfl

/-- Length of an inserted context is one more than the original. -/
theorem LinearCtxDB.length_insertAt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) :
    (LinearCtxDB.insertAt j s Γ).length = Γ.length + 1 := by
  induction j generalizing Γ with
  | zero => simp
  | succ k ih =>
    cases Γ with
    | nil => simp
    | cons x xs => simp [ih]

/-- Indexing into `insertAt` below the cutoff `j`, provided the index
    `i` is in-range for `Γ`, returns the original slot. -/
theorem LinearCtxDB.getElem?_insertAt_lt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat)
    (hij : i < j) (hiΓ : i < Γ.length) :
    (LinearCtxDB.insertAt j s Γ)[i]? = Γ[i]? := by
  induction j generalizing Γ i with
  | zero => exact (Nat.not_lt_zero _ hij).elim
  | succ k ih =>
    cases Γ with
    | nil => exact (Nat.not_lt_zero _ hiΓ).elim
    | cons x xs =>
      cases i with
      | zero => simp [LinearCtxDB.insertAt]
      | succ n =>
        have hn : n < k := Nat.lt_of_succ_lt_succ hij
        have hnΓ : n < xs.length := Nat.lt_of_succ_lt_succ hiΓ
        simp [LinearCtxDB.insertAt, ih xs n hn hnΓ]

/-- Indexing into `insertAt` above the cutoff shifts by one. -/
theorem LinearCtxDB.getElem?_insertAt_gt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat) (h : j ≤ i) :
    (LinearCtxDB.insertAt j s Γ)[i + 1]? = Γ[i]? := by
  induction j generalizing Γ i with
  | zero =>
    cases Γ <;> simp [LinearCtxDB.insertAt]
  | succ k ih =>
    cases Γ with
    | nil =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n => simp [LinearCtxDB.insertAt]
    | cons x xs =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n =>
        have hn : k ≤ n := Nat.le_of_succ_le_succ h
        simp [LinearCtxDB.insertAt, ih xs n hn]

/-- `set` at index `i` commutes with `insertAt` at cutoff `j` when
    `i < j` and `i < Γ.length`: the set lands below the inserted
    slot. The in-range hypothesis rules out the degenerate case of
    setting an index past the end of an empty context. -/
theorem LinearCtxDB.set_insertAt_lt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat) (v : Option Typ)
    (hij : i < j) (hiΓ : i < Γ.length) :
    (LinearCtxDB.insertAt j s Γ).set i v =
      LinearCtxDB.insertAt j s (Γ.set i v) := by
  induction j generalizing Γ i with
  | zero => exact (Nat.not_lt_zero _ hij).elim
  | succ k ih =>
    cases Γ with
    | nil => exact (Nat.not_lt_zero _ hiΓ).elim
    | cons x xs =>
      cases i with
      | zero => simp [LinearCtxDB.insertAt, List.set]
      | succ n =>
        have hn : n < k := Nat.lt_of_succ_lt_succ hij
        have hnΓ : n < xs.length := Nat.lt_of_succ_lt_succ hiΓ
        simp [LinearCtxDB.insertAt, List.set, ih xs n hn hnΓ]

/-- `set` at index `i + 1` commutes with `insertAt` at cutoff `j`
    when `j ≤ i`: the set lands above the inserted slot. -/
theorem LinearCtxDB.set_insertAt_gt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat) (v : Option Typ)
    (h : j ≤ i) :
    (LinearCtxDB.insertAt j s Γ).set (i + 1) v =
      LinearCtxDB.insertAt j s (Γ.set i v) := by
  induction j generalizing Γ i with
  | zero =>
    cases Γ with
    | nil => simp [LinearCtxDB.insertAt, List.set]
    | cons x xs => simp [LinearCtxDB.insertAt, List.set]
  | succ k ih =>
    cases Γ with
    | nil =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n => simp [LinearCtxDB.insertAt, List.set]
    | cons x xs =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n =>
        have hn : k ≤ n := Nat.le_of_succ_le_succ h
        simp [LinearCtxDB.insertAt, List.set, ih xs n hn]

/-! ## Shifting and the var lemmas -/

/-- Shift every free index `≥ j` in `e` by one. -/
abbrev shiftAt (j : Nat) (e : TermDB) : TermDB := liftAux j 1 e

theorem shiftAt_var_lt (j i : Nat) (h : i < j) :
    shiftAt j (TermDB.var i) = TermDB.var i := by
  simp [shiftAt, liftAux, h]

theorem shiftAt_var_ge (j i : Nat) (h : ¬ i < j) :
    shiftAt j (TermDB.var i) = TermDB.var (i + 1) := by
  simp [shiftAt, liftAux, h]

/-! ## Proof obligations: `weakening_insert_db` and
    `subst_preserves_typing_db`

Under Option C these have the uniform statements

```lean
theorem weakening_insert_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma Gamma e t eps Gamma')
    (j : Nat) (t_new : Typ) :
    HasTypeDB Delta Sigma (Gamma.insertAt j (some t_new))
              (shiftAt j e) t eps
              (Gamma'.insertAt j (some t_new))

theorem subst_preserves_typing_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {e v : TermDB} {t t_v : Typ} {eps : EffectRow}
    (j : Nat)
    (h_e : HasTypeDB Delta Sigma (Gamma.insertAt j (some t_v)) e t eps
             (Gamma.insertAt j none))
    (h_v : HasTypeDB Delta Sigma Gamma v t_v [] Gamma) :
    HasTypeDB Delta Sigma Gamma (substDBAux j v e) t eps Gamma
```

### Proof sketch — `weakening_insert_db`

Mutual induction on `HasTypeDB` / `ClausesTypedDB` (the mutual
partner handles the `handle` case).

**var(i)**: case split on `i < j` vs `j ≤ i`.
- `i < j`: `shiftAt j (var i) = var i`; premise `Γ[i]? = some (some t)`
  lifts through `getElem?_insertAt_lt` (using `i < Γ.length`, which
  follows from the premise); output rewriting uses `set_insertAt_lt`.
- `j ≤ i`: `shiftAt j (var i) = var (i+1)`; premise lifts through
  `getElem?_insertAt_gt`; output uses `set_insertAt_gt`.

**unit / const / loc**: trivial — body and contexts unchanged by the
shift on leaf nodes.

**abs / vmap / grad**: body IH at cutoff `j + 1`. Because the binder
is prepended as `some t1 ::`, `(some t1 :: Γ).insertAt (j+1) s`
reduces definitionally to `some t1 :: Γ.insertAt j s`. Reassemble
with the matching constructor.

**letBind**: first-argument IH at `j`, body IH at `j + 1`. Same
head-cons reduction.

**letpair / handler clauses**: body IH at `j + 2`. Two-slot head-cons
reduction is again definitional.

**app / pair / add / mul / sum / expand / copy / uniformLike / fst /
snd / perform / subEff**: straightforward — apply the matching IH at
`j` for each sub-derivation and reassemble.

**handle**: the `ClausesTypedDB` sub-derivation requires the mutual
partner lemma, which is proved by structural induction on the
clauses list (nil is trivial; cons uses `weakening_insert_db` on the
clause body at `j + 2` and the mutual IH on the tail).

All cases except `handle` have been elaborated and build green in an
intermediate working file. The `handle` case is blocked on the
mutual-recursion machinery — a clean discharge requires either
(a) a `mutual theorem` block with `termination_by sizeOf`, or
(b) elaborating the combined motive via `HasTypeDB.rec` / the
auto-generated joint recursor.

### Proof sketch — `subst_preserves_typing_db`

The non-var cases mirror `weakening_insert_db` exactly; only the
var case differs, and it splits three ways on `i` vs `j`:

- `i = j`: `substDBAux j v (var j) = v`. The hypothesis
  `h_v : HasTypeDB … Γ v t_v [] Γ` is the required conclusion
  after a `cases` on the equality
  `(Γ.insertAt j (some t_v))[j]? = some (some t_v)` (via
  `getElem?_insertAt_eq`, which is another helper along the same
  lines as the `_lt`/`_gt` pair above).
- `i < j`: `substDBAux j v (var i) = var i`; the rewrite
  `getElem?_insertAt_lt` gives `Γ[i]?`, and the reapplied var rule
  produces a context `Γ.set i none`. The hypothesis's output
  shape `Γ.insertAt j none` matches after reducing with
  `set_insertAt_lt`.
- `i > j`: `substDBAux j v (var i) = var (i - 1)`. `i > j` means
  `i ≥ 1`; `getElem?_insertAt_gt` at index `i - 1` recovers the
  premise, and the output side uses `set_insertAt_gt`.

The non-var cases reuse the same binder-structural reductions as
weakening. The `handle` case is again blocked on mutual recursion.

### Status

This file delivers the Option C linear-context representation
(`TypingDB.lean`), the 6 structural helper lemmas above, and the
shift-on-var simp facts. The two metatheorems are documented as
proof obligations for a follow-up commit that wires up the mutual
`ClausesTypedDB` side properly. -/

end LaCaDiLE
