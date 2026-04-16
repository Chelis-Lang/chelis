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

theorem hasType_names_preserved
    {Δ : CapCtx} {S : StoreTyp} {Γ Γ' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Δ S Γ e t eps Γ') :
    Γ.map Prod.fst = Γ'.map Prod.fst := by
  induction h using HasType.rec
    (motive_2 := fun _ _ Γ2 Γ3 _ _ _ _ => Γ2.map Prod.fst = Γ3.map Prod.fst)
    with
  | var _ _ Γpre Γpost x t_ =>
      simp [List.map_append]
  | unit _ _ _ => rfl
  | abs _ _ _ _ _ _ _ _ _ _ _ ih =>
      simp only [List.map_append, List.map_cons, List.map_nil] at ih
      exact List.append_cancel_right ih
  | letBind _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      simp only [List.map_append, List.map_cons, List.map_nil] at ih2
      exact ih1.trans (List.append_cancel_right ih2)
  | letpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 =>
      simp only [List.map_append, List.map_cons, List.map_nil] at ih2
      exact ih1.trans (List.append_cancel_right ih2)
  | app _ _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | copy _ _ _ _ _ _ _ _ ih => exact ih
  | fst _ _ _ _ _ _ _ _ _ ih => exact ih
  | snd _ _ _ _ _ _ _ _ _ ih => exact ih
  | const _ _ _ _ _ => rfl
  | tadd _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | tmul _ _ _ _ _ _ _ _ _ _ _ _ ih1 ih2 => exact ih1.trans ih2
  | tsum _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | texpand _ _ _ _ _ _ _ _ _ ih => exact ih
  | uniformLike _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | perform _ _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | handle _ _ _ _ _ _ _ _ _ _ h_body _ _ _ _ ih_body ih_clauses =>
      exact ih_body.trans ih_clauses
  | tgrad _ _ _ _ _ _ _ _ _ _ _ _ => rfl
  | tvmap _ _ _ _ _ _ _ _ _ _ _ => rfl
  | loc _ _ _ _ _ _ => rfl
  | subEff _ _ _ _ _ _ _ _ _ _ ih => exact ih
  | nil _ _ _ _ _ => rfl
  | cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _h_body _h_rest _ih_body ih_rest =>
      exact ih_rest

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
derivation to a `HasTypeDB` derivation over the translated context
and term. The induction is on the `HasType` derivation using the
mutual recursor `HasType.rec` (with a motive_2 for clauses), and
each case builds the corresponding `HasTypeDB` constructor. -/

theorem hasType_to_hasTypeDB
    {Δ : CapCtx} {S : StoreTyp} {Γ Γ' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Δ S Γ e t eps Γ') :
    HasTypeDB Δ S (ctxToDB Γ) (termToDB (envOfCtx Γ) e)
              t eps (ctxToDB Γ') := by
  induction h using HasType.rec
    (motive_2 := fun Δ' S' Γ2 Γ3 t' epsR cls _ =>
      ClausesTypedDB Δ' S' (ctxToDB Γ2) (ctxToDB Γ3) t' epsR
                     (clausesToDB (envOfCtx Γ2) cls)) with
  | unit Δ' S' Γ_ =>
      simp only [termToDB]
      exact HasTypeDB.unit Δ' S' (ctxToDB Γ_)
  | const Δ' S' Γ_ v ds =>
      simp only [termToDB]
      exact HasTypeDB.const Δ' S' (ctxToDB Γ_) v ds
  | loc Δ' S' Γ_ ell t' hlook =>
      simp only [termToDB]
      exact HasTypeDB.loc Δ' S' (ctxToDB Γ_) ell t' hlook
  | subEff Δ' S' Γ_ Γ'_ e_ t_ eps_ eps'_ _h hSub ih =>
      exact HasTypeDB.subEff Δ' S' (ctxToDB Γ_) (ctxToDB Γ'_)
        (termToDB (envOfCtx Γ_) e_) t_ eps_ eps'_ ih hSub
  | fst Δ' S' Γ1 Γ2 e_ t1 t2 eps_ _h ih =>
      simp only [termToDB]
      exact HasTypeDB.fst Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        (termToDB (envOfCtx Γ1) e_) t1 t2 eps_ ih
  | snd Δ' S' Γ1 Γ2 e_ t1 t2 eps_ _h ih =>
      simp only [termToDB]
      exact HasTypeDB.snd Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        (termToDB (envOfCtx Γ1) e_) t1 t2 eps_ ih
  | copy Δ' S' Γ1 Γ2 e_ ds eps_ _h ih =>
      simp only [termToDB]
      exact HasTypeDB.copy Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        (termToDB (envOfCtx Γ1) e_) ds eps_ ih
  | app Δ' S' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps_ eps1 eps2 h1 _h2 ih1 ih2 =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h1] at ih1 ⊢
      exact HasTypeDB.app Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        (termToDB (envOfCtx Γ2) e1) (termToDB (envOfCtx Γ2) e2)
        t1 t2 eps_ eps1 eps2 ih1 ih2
  | tpair Δ' S' Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 h1 _h2 ih1 ih2 =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h1] at ih1 ⊢
      exact HasTypeDB.tpair Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        (termToDB (envOfCtx Γ2) e1) (termToDB (envOfCtx Γ2) e2)
        t1 t2 eps1 eps2 ih1 ih2
  | tadd Δ' S' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 _h2 ih1 ih2 =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h1] at ih1 ⊢
      exact HasTypeDB.tadd Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        (termToDB (envOfCtx Γ2) e1) (termToDB (envOfCtx Γ2) e2)
        ds eps1 eps2 ih1 ih2
  | tmul Δ' S' Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 _h2 ih1 ih2 =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h1] at ih1 ⊢
      exact HasTypeDB.tmul Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        (termToDB (envOfCtx Γ2) e1) (termToDB (envOfCtx Γ2) e2)
        ds eps1 eps2 ih1 ih2
  | tsum Δ' S' Γ1 Γ2 e_ ds d eps_ _h _hmem ih =>
      simp only [termToDB]
      exact HasTypeDB.tsum Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        (termToDB (envOfCtx Γ1) e_) ds 0 eps_ ih (rem ds d) (trivial)
  | texpand Δ' S' Γ1 Γ2 e_ ds d eps_ _h ih =>
      simp only [termToDB]
      exact HasTypeDB.texpand Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        (termToDB (envOfCtx Γ1) e_) ds 0 0 eps_ ih (ins ds d) (trivial)
  | uniformLike Δ' S' Γ1 Γ2 e_ ds lo hi eps_ _h ih =>
      simp only [termToDB]
      exact HasTypeDB.uniformLike Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        (termToDB (envOfCtx Γ1) e_) ds lo hi eps_ ih
  | perform Δ' S' Γ1 Γ2 op e_ tArg tRet eps_ _h hM ih =>
      simp only [termToDB]
      exact HasTypeDB.perform Δ' S' (ctxToDB Γ1) (ctxToDB Γ2)
        op (termToDB (envOfCtx Γ1) e_) tArg tRet eps_ ih hM
  | abs Δ' S' Γ1 Γ2 x t1 t2 eps_ body slot _h ih =>
      simp only [termToDB]
      rw [ctxToDB_append_singleton, envOfCtx_append_singleton] at ih
      rw [ctxToDB_append_singleton] at ih
      exact HasTypeDB.abs Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) slot t1 t2 eps_
        (termToDB (x :: envOfCtx Γ1) body) ih
  | letBind Δ' S' Γ1 Γ2 Γ3 x e1 e2 t1 t2 eps1 eps2 slot h1 _h2 ih1 ih2 =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h1] at ih1 ⊢
      rw [ctxToDB_append_singleton, envOfCtx_append_singleton] at ih2
      rw [ctxToDB_append_singleton] at ih2
      exact HasTypeDB.letBind Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        slot (termToDB (envOfCtx Γ2) e1) (termToDB (x :: envOfCtx Γ2) e2)
        t1 t2 eps1 eps2 ih1 ih2
  | letpair Δ' S' Γ1 Γ2 Γ3 x y e1 e2 t1 t2 t_ eps1 eps2 slotX slotY
            h1 _h2 ih1 ih2 =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h1] at ih1 ⊢
      rw [ctxToDB_append_pair, envOfCtx_append_pair] at ih2
      rw [ctxToDB_append_pair] at ih2
      exact HasTypeDB.letpair Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        slotY slotX
        (termToDB (envOfCtx Γ2) e1) (termToDB (y :: x :: envOfCtx Γ2) e2)
        t1 t2 t_ eps1 eps2 ih1 ih2
  | tgrad Δ' S' Γ_ x ds dsOut body eps_ slot _h hsub ih =>
      simp only [termToDB]
      rw [ctxToDB_append_singleton, envOfCtx_append_singleton] at ih
      rw [ctxToDB_append_singleton] at ih
      exact HasTypeDB.tgrad Δ' S' (ctxToDB Γ_) slot ds dsOut
        (termToDB (x :: envOfCtx Γ_) body) eps_ ih hsub
  | tvmap Δ' S' Γ_ x t1 t2 body eps_ d slot _h ih =>
      simp only [termToDB]
      rw [ctxToDB_append_singleton, envOfCtx_append_singleton] at ih
      rw [ctxToDB_append_singleton] at ih
      exact HasTypeDB.tvmap Δ' S' (ctxToDB Γ_) slot t1 t2
        (termToDB (x :: envOfCtx Γ_) body) eps_ d ih
  | nil Δ' S' Γ2 t_ epsR =>
      simp only [clausesToDB]
      exact ClausesTypedDB.nil Δ' S' (ctxToDB Γ2) t_ epsR
  | cons Δ' S' Γ2 Γ3 t_ tArg tRet epsR op x k hb rest slotX slotK
         _h_body _h_rest ih_body ih_rest =>
      simp only [clausesToDB]
      rw [ctxToDB_append_pair, envOfCtx_append_pair] at ih_body
      rw [ctxToDB_append_pair] at ih_body
      exact ClausesTypedDB.cons Δ' S' (ctxToDB Γ2) (ctxToDB Γ3)
        slotK slotX t_ tArg tRet epsR op
        (termToDB (k :: x :: envOfCtx Γ2) hb) (clausesToDB (envOfCtx Γ2) rest)
        ih_body ih_rest
  | handle Δ' S' Γ1 Γ2 Γ3 body clauses t_ epsH epsB
           h_body hSubsH hClsH hCover _h_cls ih_body ih_cls =>
      simp only [termToDB]
      rw [hasType_envOfCtx_eq h_body] at ih_body ⊢
      refine HasTypeDB.handle Δ' S' (ctxToDB Γ1) (ctxToDB Γ2) (ctxToDB Γ3)
        (termToDB (envOfCtx Γ2) body)
        (clausesToDB (envOfCtx Γ2) clauses) t_ epsH epsB
        ih_body hSubsH ?_ ?_ ih_cls
      · intro cl hmem
        obtain ⟨orig, horig, hop⟩ := exists_orig_of_clausesToDB hmem
        rw [← hop]; exact hClsH orig horig
      · intro op hop
        obtain ⟨cl, hmem, hcl_eq⟩ := hCover op hop
        obtain ⟨hb', hmem'⟩ := exists_subst_of_clausesToDB cl hmem
        exact ⟨(cl.1, hb'), hmem', hcl_eq⟩
  | var Δ' S' Γpre Γpost x t_ =>
      -- The var case requires showing that envIndex finds x at position
      -- |Γpost| in the reversed name list, which corresponds to the
      -- position of the (some t_) entry in ctxToDB. This holds when
      -- x ∉ names(Γpost) (linearity/uniqueness of names in context).
      -- Full closure requires a NoDup premise or a proof that HasType
      -- derivations only inhabit contexts with unique names. Sorry'd
      -- pending that premise; the rest of the translation chain
      -- (subst_preserves_typing corollary) works modulo this case.
      sorry

end LaCaDiLE
