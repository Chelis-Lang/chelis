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
--   new head binder. `Term.vmap x t d body` discards `x` similarly.

import LaCaDiLE.Syntax
import LaCaDiLE.SyntaxDB
import LaCaDiLE.Typing
import LaCaDiLE.TypingDB
import LaCaDiLE.TranslationDB

namespace LaCaDiLE

/-- De Bruijn index of `x` in the innermost-to-outermost binding
    list `env`. Returns `env.length` (out-of-range) if `x` is not
    bound — callers should only invoke `termToDB` on closed terms
    w.r.t. the passed environment. -/
def envIndex : List String → String → Nat
  | [], _ => 0
  | y :: rest, x => if x = y then 0 else envIndex rest x + 1

/-- Context translation: reverse the named context (which puts new
    binders at the TAIL via `Γ ++ [(x,t)]`) so that the translated DB
    context has the innermost binder at position 0 (HEAD via
    `some t :: Γ`). Then drop names.

    Under tombstone semantics `p.2 : Option Typ` is already the right
    shape for `LinearCtxDB = List (Option Typ)` — no wrapping needed.
    This makes `ctxToDB (Γ ++ [(x, ot)]) = ot :: ctxToDB Γ`, which
    aligns with HasTypeDB's binder convention. -/
def ctxToDB (Γ : LinearCtx) : LinearCtxDB :=
  (Γ.reverse).map (fun p => p.2)

@[simp] theorem ctxToDB_length (Γ : LinearCtx) :
    (ctxToDB Γ).length = Γ.length := by
  simp [ctxToDB]

@[simp] theorem ctxToDB_nil : ctxToDB [] = [] := rfl

theorem ctxToDB_append_singleton (Γ : LinearCtx) (x : String) (ot : Option Typ) :
    ctxToDB (Γ ++ [(x, ot)]) = ot :: ctxToDB Γ := by
  simp [ctxToDB, List.reverse_append, List.map_append, List.map_cons, List.map_nil]

theorem ctxToDB_append_pair (Γ : LinearCtx) (x y : String)
    (ot1 ot2 : Option Typ) :
    ctxToDB (Γ ++ [(x, ot1), (y, ot2)]) = ot2 :: ot1 :: ctxToDB Γ := by
  simp [ctxToDB, List.reverse_append, List.map_append, List.map_cons, List.map_nil]

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
  | Term.fst _ e => TermDB.fst (termToDB env e)
  | Term.snd _ e => TermDB.snd (termToDB env e)
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
  | Term.vmap x t d body =>
      TermDB.vmap t d (termToDB (x :: env) body)
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

/-! ## Naming environment extraction -/

/-- Extract the name environment from a LinearCtx, reversed to match
    the de Bruijn convention (innermost binder first). -/
def envOfCtx (Γ : LinearCtx) : List String :=
  (Γ.map Prod.fst).reverse

@[simp] theorem envOfCtx_nil : envOfCtx [] = [] := rfl

theorem envOfCtx_append_singleton (Γ : LinearCtx) (x : String) (ot : Option Typ) :
    envOfCtx (Γ ++ [(x, ot)]) = x :: envOfCtx Γ := by
  simp [envOfCtx, List.map_append, List.reverse_append]

theorem envOfCtx_append_pair (Γ : LinearCtx) (x y : String)
    (ot1 ot2 : Option Typ) :
    envOfCtx (Γ ++ [(x, ot1), (y, ot2)]) = y :: x :: envOfCtx Γ := by
  simp [envOfCtx, List.map_append, List.reverse_append]

theorem envIndex_zero (x : String) (rest : List String) :
    envIndex (x :: rest) x = 0 := by
  simp [envIndex]

theorem envIndex_cons_ne (x y : String) (rest : List String) (h : x ≠ y) :
    envIndex (y :: rest) x = envIndex rest x + 1 := by
  simp [envIndex, h]

-- hasType_to_hasTypeDB moved to after termToDB mutual block below

/-! ## Var-case structural lemmas for the forward translation -/

theorem ctxToDB_split (Γpre Γpost : LinearCtx) (x : String) (ot : Option Typ) :
    ctxToDB (Γpre ++ [(x, ot)] ++ Γpost) =
    ctxToDB Γpost ++ [ot] ++ ctxToDB Γpre := by
  simp only [ctxToDB, List.reverse_append, List.map_append, List.map_reverse,
    List.map_cons, List.map_nil, List.reverse_cons, List.reverse_nil, List.nil_append,
    List.singleton_append, List.append_assoc]

theorem ctxToDB_split_getElem (Γpre Γpost : LinearCtx) (x : String) (ot : Option Typ) :
    (ctxToDB (Γpre ++ [(x, ot)] ++ Γpost))[Γpost.length]? = some ot := by
  rw [ctxToDB_split]
  simp [ctxToDB, List.getElem?_append]

theorem ctxToDB_split_set (Γpre Γpost : LinearCtx) (x : String)
    (ot1 ot2 : Option Typ) :
    (ctxToDB (Γpre ++ [(x, ot1)] ++ Γpost)).set Γpost.length ot2 =
    ctxToDB (Γpre ++ [(x, ot2)] ++ Γpost) := by
  simp only [ctxToDB_split]
  simp [ctxToDB, List.set_append]

theorem envOfCtx_split (Γpre Γpost : LinearCtx) (x : String) (ot : Option Typ) :
    envOfCtx (Γpre ++ [(x, ot)] ++ Γpost) =
    (Γpost.map Prod.fst).reverse ++ [x] ++ (Γpre.map Prod.fst).reverse := by
  simp [envOfCtx, List.map_append, List.reverse_append]

theorem envIndex_not_mem (xs : List String) (x : String) (rest : List String)
    (h : x ∉ xs) :
    envIndex (xs ++ [x] ++ rest) x = xs.length := by
  induction xs with
  | nil => simp [envIndex]
  | cons y ys ih =>
    have hne : x ≠ y := fun heq => h (heq ▸ List.mem_cons_self ..)
    simp only [List.cons_append, List.length_cons, envIndex, hne, ite_false]
    congr 1
    exact ih (fun hm => h (List.mem_cons_of_mem y hm))

/-! ## Name preservation under tombstoning

Since `HasType` only changes `Option Typ` values (tombstoning `some t →
none`), the name component of every context entry is preserved.
`envOfCtx` maps `Prod.fst` and reverses, so it's invariant across any
`HasType` derivation. This means `termToDB (envOfCtx Γ) e` uses the
same environment at every intermediate context in the derivation. -/

-- hasType_names_preserved now lives in Typing.lean

theorem hasType_envOfCtx_eq
    {Δ : CapCtx} {S : StoreTyp} {Γ Γ' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Δ S Γ e t eps Γ') :
    envOfCtx Γ = envOfCtx Γ' := by
  simp [envOfCtx, hasType_names_preserved h]

/-! ## clausesToDB op-preservation helpers -/

theorem exists_orig_of_clausesToDB
    {env : List String}
    {clauses : List (EffectLabel × String × String × Term)}
    {cl : EffectLabel × TermDB}
    (h : cl ∈ clausesToDB env clauses) :
    ∃ orig ∈ clauses, orig.1 = cl.1 := by
  induction clauses with
  | nil => exact absurd h (by simp [clausesToDB])
  | cons hd rest ih =>
    simp only [clausesToDB, List.mem_cons] at h
    obtain ⟨op, xArg, kCont, body⟩ := hd
    rcases h with heq | hrest
    · refine ⟨(op, xArg, kCont, body), List.mem_cons_self .., ?_⟩
      cases cl; simp only [Prod.mk.injEq] at heq; exact heq.1.symm
    · obtain ⟨orig, hmem, hop⟩ := ih hrest
      exact ⟨orig, List.mem_cons_of_mem _ hmem, hop⟩

theorem exists_subst_of_clausesToDB
    {env : List String}
    {clauses : List (EffectLabel × String × String × Term)}
    (cl : EffectLabel × String × String × Term)
    (h : cl ∈ clauses) :
    ∃ hb', (cl.1, hb') ∈ clausesToDB env clauses := by
  induction clauses with
  | nil => exact absurd h (by simp)
  | cons hd rest ih =>
    obtain ⟨op, xArg, kCont, body⟩ := hd
    simp only [List.mem_cons] at h
    rcases h with heq | hrest
    · subst heq
      exact ⟨_, List.mem_cons_self ..⟩
    · obtain ⟨hb', hmem⟩ := ih hrest
      exact ⟨hb', List.mem_cons_of_mem _ hmem⟩

/-! ## Forward typing preservation (Wave 5p)

The main theorem `hasType_to_hasTypeDB` maps a named `HasType`
derivation to a DB typing derivation.

The original exact statement

`HasType Γ e ⟹ HasTypeDB (ctxToDB Γ) (termToDB (envOfCtx Γ) e)`

is not sound for arbitrary named derivations: named `HasType.var` may
consume any same-named slot in the context, while `termToDB` always
chooses the innermost lexical binder. `TranslationDB.lean` closes the
sound bridge by switching to derivation-guided erasure under a lexical
scoping premise. This file re-exports that honest boundary instead of
repeating the unsound total theorem shape. -/

@[simp] theorem ctxToDB_eq_eraseCtx (Γ : LinearCtx) :
    ctxToDB Γ = eraseCtx Γ := rfl

@[simp] theorem envOfCtx_eq_ctxEnv (Γ : LinearCtx) :
    envOfCtx Γ = ctxEnv Γ := rfl

theorem hasType_to_hasTypeDB
    {Δ : CapCtx} {S : StoreTyp} {Γ Γ' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Δ S Γ e t eps Γ')
    (hlex : LexicallyScoped Γ e) :
    ∃ eDB,
      eraseTerm (ctxEnv Γ) e = some eDB ∧
      HasTypeDB Δ S (ctxToDB Γ) eDB t eps (ctxToDB Γ') := by
  rcases transport_typing_lexical h hlex with ⟨eDB, hErase, hTy⟩
  exact ⟨eDB, hErase, by simpa [ctxToDB_eq_eraseCtx] using hTy⟩

end LaCaDiLE
