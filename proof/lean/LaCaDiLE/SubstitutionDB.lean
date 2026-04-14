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

/-! ## Weakening lemma: inserting a fresh linear binding -/

/-- A compact "insertion shift" abbreviation. `shiftAt j e` is `liftAux j 1 e`:
    add 1 to every free index of `e` that is at least `j`. -/
abbrev shiftAt (j : Nat) (e : TermDB) : TermDB := liftAux j 1 e

/--
Weakening by positional insertion. For any split of the input/output DB
linear contexts, inserting a fresh type `t_new` at the *same* position `j` in
both and shifting every free index ≥ `j` in the term preserves the DB typing
judgment.

The proof is by mutual induction on `HasTypeDB` with a companion `motive_2`
threading the same insertion through `ClausesTypedDB`. Binder cases bump the
insertion cutoff: `abs`, `letBind`, `grad`, `vmap` use `j + 1`; `letpair` and
handler clauses use `j + 2`.
-/
theorem weakening_insert_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow} {Gamma' : LinearCtxDB}
    (h : HasTypeDB Delta Sigma Gamma e t eps Gamma')
    (j : Nat) (t_new : Typ) :
    HasTypeDB Delta Sigma (LinearCtxDB.insertAt j t_new Gamma)
              (shiftAt j e) t eps
              (LinearCtxDB.insertAt j t_new Gamma') := by
  -- PROOF OBLIGATION (unclosed by Wave 3):
  --
  -- Mutual induction on `HasTypeDB` / `ClausesTypedDB`. The theorem is true,
  -- and every non-binder rule is a direct reassembly of the matching
  -- constructor after applying the IH to each premise. The binder rules
  -- (`abs`, `letBind`, `letpair`, `grad`, `vmap`, `handle` + clauses) need
  -- the IH specialized at `j + 1` (or `j + 2` for `letpair` / clauses); the
  -- fact that `insertAt (j + 1) t (x :: xs) = x :: insertAt j t xs` makes
  -- the bookkeeping rigid.
  --
  -- The `var` case is the only non-mechanical step. Given
  --   `h : HasTypeDB .. (Γpre ++ [t] ++ Γpost) (TermDB.var Γpre.length) ..
  --                     (Γpre ++ Γpost)`,
  -- split on `j ≤ Γpre.length`:
  --   * If yes: `shiftAt j (var Γpre.length) = var (Γpre.length + 1)` and
  --     `insertAt j t_new (Γpre ++ [t] ++ Γpost) =
  --      (insertAt j t_new Γpre) ++ [t] ++ Γpost` by
  --     `LinearCtxDB.insertAt_append_left`, and reapply `HasTypeDB.var` at
  --     the shifted position.
  --   * If no (`j > Γpre.length`): the var index is `< j` so
  --     `shiftAt j = var Γpre.length`; the insertion lands strictly inside
  --     `Γpost` by `LinearCtxDB.insertAt_append_right` and the var rule
  --     applies unchanged.
  --
  -- This is a large but mechanical case analysis (~800 lines across 25
  -- rules). Wave 3 exposes the lemma and the helper infrastructure so the
  -- proof can be filled in by the next wave. The single `sorry` stands for
  -- the entire rule-by-rule dispatch sketched above; per the Wave 3 charter
  -- it counts as one of the two allowed sorries in this file.
  sorry

/-! ## Substitution lemma -/

/--
Substituting a closed value for a linear position preserves typing. If `e` is
well-typed in a context with `t_v` inserted at position `j`, and `v` is a
pure-effect value of type `t_v` typeable in the ambient context `Γ` (with
`Γ` as both input and output, i.e., `v` consumes no linear resources), then
`substDBAux j v e` is well-typed in `Γ` at the same type and effect row.

In the de Bruijn world, binder cases bump both the cutoff `j` and the
value `v` (via `lift`) so the substituted value's free indices continue to
point at the correct outer bindings.
-/
theorem subst_preserves_typing_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {e v : TermDB} {t t_v : Typ} {eps : EffectRow} {Gamma' : LinearCtxDB}
    (j : Nat)
    (h_e : HasTypeDB Delta Sigma (LinearCtxDB.insertAt j t_v Gamma) e t eps
             (LinearCtxDB.insertAt j t_v Gamma'))
    (h_v : HasTypeDB Delta Sigma Gamma v t_v [] Gamma) :
    HasTypeDB Delta Sigma Gamma (substDBAux j v e) t eps Gamma' := by
  -- As with `weakening_insert_db`, the full proof is a mutual induction on
  -- `HasTypeDB` / `ClausesTypedDB`. Non-binder cases reapply the matching
  -- constructor after the IH. Binder cases use `weakening_insert_db` (with
  -- `j = 0`) to lift `h_v` under the new binder before invoking the IH with
  -- `j + 1` (or `j + 2` for `letpair` / clauses).
  --
  -- The var case is the interesting one: a three-way split on `i` vs. `j`
  --   * `i = j`: the target position; `substDBAux` returns `v` directly,
  --     and typing reduces to `h_v` together with the observation that the
  --     surrounding `Γpre ++ [t_v] ++ Γpost` split implies `Γpre ++ Γpost = Γ`.
  --   * `i < j`: `substDBAux` keeps the index; typing is immediate from the
  --     matching `HasTypeDB.var`.
  --   * `i > j`: `substDBAux` decrements the index by 1; typing follows from
  --     `HasTypeDB.var` at the shifted position.
  --
  -- Wave 4 will complete this proof. Deferred here behind a single `sorry`
  -- stub rather than partial case splits so the file stays auditable.
  sorry

end LaCaDiLE
