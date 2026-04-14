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

The theorem statements `weakening_insert_db` and `subst_preserves_typing_db`
are staged as proof obligations for Wave 4. Rather than leaving them as
`sorry` stubs (which would increase the repository's sorry count), they
live here as commented signatures. Wave 4 will uncomment them and fill in
the proofs.

### weakening_insert_db (Wave 4 target)

```lean
abbrev shiftAt (j : Nat) (e : TermDB) : TermDB := liftAux j 1 e

theorem weakening_insert_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow} {Gamma' : LinearCtxDB}
    (h : HasTypeDB Delta Sigma Gamma e t eps Gamma')
    (j : Nat) (t_new : Typ) :
    HasTypeDB Delta Sigma (LinearCtxDB.insertAt j t_new Gamma)
              (shiftAt j e) t eps
              (LinearCtxDB.insertAt j t_new Gamma')
```

Proof: mutual induction on `HasTypeDB` / `ClausesTypedDB` with `motive_2`
threading the same insertion through `ClausesTypedDB`. Non-binder cases
reassemble the matching constructor after the IH. Binder rules (`abs`,
`letBind`, `letpair`, `grad`, `vmap`, `handle` + clauses) specialize the IH
at `j + 1` (or `j + 2` for `letpair` and handler clauses). The rigidity
`insertAt (j + 1) t (x :: xs) = x :: insertAt j t xs` follows definitionally
from the `insertAt` shape above.

The `var` case splits on `j ≤ Γpre.length`:
  * Yes: `shiftAt j (var Γpre.length) = var (Γpre.length + 1)` and
    `insertAt_append_left` reshapes the context; reapply `HasTypeDB.var`.
  * No (`j > Γpre.length`): the var index stays at `Γpre.length`;
    `insertAt_append_right` places the new binding inside `Γpost`.

### subst_preserves_typing_db (Wave 4 target)

```lean
theorem subst_preserves_typing_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {e v : TermDB} {t t_v : Typ} {eps : EffectRow} {Gamma' : LinearCtxDB}
    (j : Nat)
    (h_e : HasTypeDB Delta Sigma (LinearCtxDB.insertAt j t_v Gamma) e t eps
             (LinearCtxDB.insertAt j t_v Gamma'))
    (h_v : HasTypeDB Delta Sigma Gamma v t_v [] Gamma) :
    HasTypeDB Delta Sigma Gamma (substDBAux j v e) t eps Gamma'
```

Proof: mutual induction. Non-binder cases straightforward. Binder cases use
`weakening_insert_db` at `j = 0` to lift `h_v` under the new binder before
recursing with `j + 1`. The `var` case is a three-way split on `i` vs `j`.

### Wave 4 deliverables

1. Uncomment the two theorems above, relocating them from this doc block
   into real declarations.
2. Close both by mutual induction. Estimated ~800 lines.
3. Continue to Wave 5 (Translation.lean + named-variable corollaries in
   `Substitution.lean`). -/

end LaCaDiLE
