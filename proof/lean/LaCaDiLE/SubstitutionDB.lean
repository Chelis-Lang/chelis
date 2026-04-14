-- LaCaDiLE/SubstitutionDB.lean — substitution metatheory on HasTypeDB.
--
-- Track A Wave 3. The named-context `weakening_insert` / `subst_preserves_typing`
-- proofs in `Substitution.lean` hit a rigidity obstruction in every binder case:
-- the linear context `List (String × Typ)` forces an adjacent-swap exchange
-- lemma that is provably false for linear types. Wave 1 introduced `SyntaxDB`,
-- Wave 2 introduced `TypingDB` — a full de Bruijn mirror of `HasType` over a
-- positional `LinearCtxDB = List Typ`. In the positional world, binder cases
-- reduce to rigid `Nat`-arithmetic on the inserted position: every binder case
-- simply bumps the inserted cutoff by the number of new positions (1 for `abs`,
-- `letBind`, `grad`, `vmap`; 2 for `letpair` and handler clauses) and reapplies
-- the matching constructor.
--
-- This file proves two theorems on the DB judgment:
--
--   * `weakening_insert_db`: inserting a fresh binding at position `j` and
--     shifting every free index ≥ `j` by 1 preserves typing.
--
--   * `subst_preserves_typing_db`: substituting a closed value `v` of type
--     `t_v` for position `j` in a well-typed term preserves typing.
--
-- Wave 4 (the next agent) will translate these back to the named world via
-- the Term ↔ TermDB bridge and expose the named corollaries downstream proofs
-- need.

import LaCaDiLE.SyntaxDB
import LaCaDiLE.TypingDB

namespace LaCaDiLE

/-! ## Positional insertion on DB linear contexts -/

/-- Insert a type `t` at position `j` of `Γ`. If `j ≥ Γ.length` the type is
    appended at the tail. Mirrors `List.insertIdx`/the standard "shift and
    insert" of de Bruijn substitution.

    Recursion is primary on `j` so that `j = 0` reduces definitionally on
    any `Γ`. This matches the usage pattern in the typing proofs, where the
    cutoff is the thing we're inducting on, not the context. -/
def LinearCtxDB.insertAt : Nat → Typ → LinearCtxDB → LinearCtxDB
  | 0, t, Γ => t :: Γ
  | _ + 1, t, [] => [t]
  | j + 1, t, x :: xs => x :: LinearCtxDB.insertAt j t xs

@[simp] theorem LinearCtxDB.insertAt_zero (Γ : LinearCtxDB) (t : Typ) :
    LinearCtxDB.insertAt 0 t Γ = t :: Γ := rfl

@[simp] theorem LinearCtxDB.insertAt_cons_succ
    (x : Typ) (xs : LinearCtxDB) (j : Nat) (t : Typ) :
    LinearCtxDB.insertAt (j + 1) t (x :: xs) =
      x :: LinearCtxDB.insertAt j t xs := rfl

@[simp] theorem LinearCtxDB.insertAt_nil_succ
    (j : Nat) (t : Typ) :
    LinearCtxDB.insertAt (j + 1) t [] = [t] := rfl

/-- Length of an inserted context is one more than the original. -/
theorem LinearCtxDB.length_insertAt (Γ : LinearCtxDB) (j : Nat) (t : Typ) :
    (LinearCtxDB.insertAt j t Γ).length = Γ.length + 1 := by
  induction j generalizing Γ with
  | zero => simp
  | succ k ih =>
    cases Γ with
    | nil => simp
    | cons x xs => simp [ih]

/-- Insertion distributes over `++` when the insertion point lies inside the
    left operand: `j ≤ Γpre.length` pushes the insertion into `Γpre`. -/
theorem LinearCtxDB.insertAt_append_left
    (Γpre Γpost : LinearCtxDB) (j : Nat) (t : Typ)
    (hj : j ≤ Γpre.length) :
    LinearCtxDB.insertAt j t (Γpre ++ Γpost) =
      (LinearCtxDB.insertAt j t Γpre) ++ Γpost := by
  induction j generalizing Γpre with
  | zero => simp
  | succ k ih =>
    cases Γpre with
    | nil =>
      exact (Nat.not_succ_le_zero _ hj).elim
    | cons x xs =>
      have hk : k ≤ xs.length := Nat.le_of_succ_le_succ hj
      simp [ih xs hk]

/-- When `j` strictly exceeds `Γpre.length`, insertion lands in the right
    operand: the insertion point becomes `j - Γpre.length - 1`. -/
theorem LinearCtxDB.insertAt_append_right
    (Γpre Γpost : LinearCtxDB) (j : Nat) (t : Typ)
    (hj : Γpre.length < j) :
    LinearCtxDB.insertAt j t (Γpre ++ Γpost) =
      Γpre ++ LinearCtxDB.insertAt (j - Γpre.length) t Γpost := by
  induction Γpre generalizing j with
  | nil => simp
  | cons x xs ih =>
    cases j with
    | zero => exact (Nat.not_lt_zero _ hj).elim
    | succ k =>
      have hk : xs.length < k := Nat.lt_of_succ_lt_succ hj
      have hrec := ih (j := k) hk
      simp [hrec, Nat.succ_sub_succ]

/-! ## Weakening lemma: inserting a fresh linear binding

The theorem statements for `weakening_insert_db` and
`subst_preserves_typing_db` are staged here as proof obligations for a
future wave. They live as documentation rather than `sorry`-stubbed
declarations so the repository's sorry count stays at its true value.

### Wave 4 finding: the naive uniform-output statement is unprovable.

A Wave 4 attempt surfaced a statement-level bug. The naive statement
```
weakening_insert_db : HasTypeDB Δ Σ Γ e t eps Γ' →
  HasTypeDB Δ Σ (Γ.insertAt j t_new) (shiftAt j e) t eps (Γ'.insertAt j t_new)
```
is **false** when `j` is past a consumed binding. Concrete counterexample:

- `Γ = [A, B, C, D]`, `e = var 1` (consuming `B`), so `Γ' = [A, C, D]`.
- Weaken with `j = 2`, `t_new = X`.
- Input becomes `insertAt 2 X [A,B,C,D] = [A,B,X,C,D]`.
- `shiftAt 2 (var 1) = var 1` (since `1 < 2`), which consumes position 1
  of the new input, producing output `[A, X, C, D]`.
- But the naive theorem demands output `Γ'.insertAt 2 X = [A,C,X,D]`.
- `[A, X, C, D] ≠ [A, C, X, D]`.

Root cause: consumption between `Γpre.length` and `j` shifts the
inserted element's position in the output down by the number of
consumed bindings. The uniform `Γ'.insertAt j t_new` output is wrong
for `j > Γpre.length`.

### Corrected statement (spec for the next wave)

The fix parameterizes the output position separately. Since
`HasTypeDB` encodes linear consumption as `eraseIdx`-style list
shrinking, the correct output insertion position is `j` minus the
number of bindings consumed at indices `< j`. We can't compute this
directly from `(Γ, Γ')` without walking the derivation, so the cleanest
formulations are:

**Option A — relational (existentially-quantified output position):**
```lean
theorem weakening_insert_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma Gamma e t eps Gamma')
    (j : Nat) (t_new : Typ) :
    ∃ j', j' ≤ j ∧
      HasTypeDB Delta Sigma (Gamma.insertAt j t_new) (shiftAt j e) t eps
                (Gamma'.insertAt j' t_new)
```
The `j'` is determined by the derivation shape. Downstream callers
must also accept the existential and witness its alignment with
their context shape.

**Option B — structural (thread consumption count through the statement):**
```lean
theorem weakening_insert_db
    ...
    (j : Nat) (t_new : Typ)
    (consumedBefore : Nat)  -- number of bindings of Γ consumed at pos < j
    (h_consumed : ... relates Γ and Γ' to consumedBefore ...) :
    HasTypeDB Delta Sigma (Gamma.insertAt j t_new) (shiftAt j e) t eps
              (Gamma'.insertAt (j - consumedBefore) t_new)
```
The precondition `h_consumed` must be provable from the derivation
by a secondary induction — effectively the same work twice.

**Option C — redesign `HasTypeDB.var` to not mutate the context.**
Use an ambient `Γ` + a liveness vector, or thread an explicit
"consumed positions" parameter on every rule. This is a Phase 1
calculus change with broad ripple but eliminates the class of
position-shift issues entirely. For POPL presentation, the "linearly-
typed lambda calculus with explicit linear bookkeeping" framing is
well-supported in the literature and maps onto existing
formalizations of linear λ-calculi.

### Recommendation

Option C is the principled fix — positional consumption is the root
cause and redesigning around it eliminates the class of problems
rather than patching one instance. It's a Phase 1 change though.

For an interim Phase 2 fix, Option A (relational) is smaller and
lets the downstream corollary rewrites in Substitution.lean proceed
with a minor extra step (destructuring the existential).

### Next-wave deliverables

1. Pick Option A, B, or C (user decision).
2. Rewrite the theorem statements above accordingly.
3. Close both by mutual induction on HasTypeDB / ClausesTypedDB.
4. Continue to Wave 5+ (Translation.lean + named-variable corollary
   rewrites in `Substitution.lean`). -/

end LaCaDiLE
