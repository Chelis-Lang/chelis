-- LaCaDiLE/Operational.lean — Step inductive relation.
--
-- Encodes every reduction rule from proof/paper/figures/opsem.tex as a
-- constructor of the `Step` relation. Phase 1 T8: structural skeleton.
--
-- Phase 1 skeleton simplification: this file encodes only the **head
-- reductions** (redex-at-the-outermost-position). The full small-step
-- relation is the congruence closure of these rules via an evaluation
-- context E[·] (see opsem.tex E-Ctx). Phase 2 will add the congruence
-- closure as either a single `ctx` constructor parameterized by an
-- `EvalCtx` inductive, or as inline per-position congruence constructors.
--
-- Substitution is naive (no alpha-renaming). For Phase 1 the skeleton
-- uses `partial def` to avoid termination proofs; Phase 2 WS2.1 will
-- prove a total substitution lemma with capture avoidance.

import LaCaDiLE.Syntax
import LaCaDiLE.AdjointTransform

namespace LaCaDiLE

/-- Naive capture-unaware substitution: replace every free occurrence of `x`
    in `target` with `v`. Stops descending into scopes that shadow `x`.
    `partial def` because Phase 1 skeleton does not prove termination;
    the structural decrease on `target` is obvious but Lean's automated
    checker may stumble on the `List.map` through handler clauses. -/
partial def subst (target : Term) (v : Term) (x : String) : Term :=
  match target with
  | Term.var y => if y = x then v else target
  | Term.abs y t body =>
      if y = x then target else Term.abs y t (subst body v x)
  | Term.app e1 e2 => Term.app (subst e1 v x) (subst e2 v x)
  | Term.letBind y e1 e2 =>
      Term.letBind y (subst e1 v x) (if y = x then e2 else subst e2 v x)
  | Term.copy e => Term.copy (subst e v x)
  | Term.letpair y z e1 e2 =>
      Term.letpair y z (subst e1 v x)
        (if y = x ∨ z = x then e2 else subst e2 v x)
  | Term.pair e1 e2 => Term.pair (subst e1 v x) (subst e2 v x)
  | Term.fst e => Term.fst (subst e v x)
  | Term.snd e => Term.snd (subst e v x)
  | Term.unit => Term.unit
  | Term.const c ds => Term.const c ds
  | Term.add e1 e2 => Term.add (subst e1 v x) (subst e2 v x)
  | Term.mul e1 e2 => Term.mul (subst e1 v x) (subst e2 v x)
  | Term.sum e i => Term.sum (subst e v x) i
  | Term.expand e i k => Term.expand (subst e v x) i k
  | Term.uniformLike e lo hi => Term.uniformLike (subst e v x) lo hi
  | Term.grad y t tOut body =>
      if y = x then target else Term.grad y t tOut (subst body v x)
  | Term.vmap y t body =>
      if y = x then target else Term.vmap y t (subst body v x)
  | Term.handle epsH body clauses =>
      Term.handle epsH (subst body v x)
        (clauses.map fun cl =>
          let (op, y, k, body') := cl
          if y = x ∨ k = x then cl else (op, y, k, subst body' v x))
  | Term.perform op e => Term.perform op (subst e v x)
  | Term.loc ell => Term.loc ell

/-- Capture a tensor value at location `ell` into a `TensorVal`. Used in
    placeholders for primitive reduction rules; the actual pointwise
    operations (`oplus`, `odot`, etc.) are not exercised by Phase 1
    metatheory. -/
def tensorOpPlaceholder (_l1 _l2 : TensorVal) : TensorVal :=
  { shape := [], data := 0.0 }

/-- The small-step reduction relation.

    Head-reduction only: each constructor fires when the configuration is
    a redex at the outermost position. The full small-step relation is
    the congruence closure of these rules via an evaluation context
    (opsem.tex E-Ctx), to be added in Phase 2. -/
inductive Step : Config → Config → Prop

  /- ## Core λ-calculus redexes -/

  -- E-Beta: (λx:t.e) v  ↦  e[v/x]   when v is a value
  | beta
      (sigma : Store) (x : String) (t : Typ) (e v : Term) :
      IsValue v →
      Step ⟨sigma, Term.app (Term.abs x t e) v⟩
           ⟨sigma, subst e v x⟩

  -- E-Let: let x = v in e  ↦  e[v/x]   when v is a value
  | letBind
      (sigma : Store) (x : String) (v e : Term) :
      IsValue v →
      Step ⟨sigma, Term.letBind x v e⟩
           ⟨sigma, subst e v x⟩

  -- E-LetPair: let (x, y) = (v1, v2) in e  ↦  e[v1/x, v2/y]
  | letpair
      (sigma : Store) (x y : String) (v1 v2 e : Term) :
      IsValue v1 → IsValue v2 →
      Step ⟨sigma, Term.letpair x y (Term.pair v1 v2) e⟩
           ⟨sigma, subst (subst e v1 x) v2 y⟩

  -- E-Fst: fst((v1, v2))  ↦  v1
  | fst
      (sigma : Store) (v1 v2 : Term) :
      IsValue v1 → IsValue v2 →
      Step ⟨sigma, Term.fst (Term.pair v1 v2)⟩
           ⟨sigma, v1⟩

  -- E-Snd: snd((v1, v2))  ↦  v2
  | snd
      (sigma : Store) (v1 v2 : Term) :
      IsValue v1 → IsValue v2 →
      Step ⟨sigma, Term.snd (Term.pair v1 v2)⟩
           ⟨sigma, v2⟩

  /- ## Store-allocating primitives -/

  -- E-Const: const(v, ds)  ↦  fresh location with the literal stored
  | tconst
      (sigma : Store) (v : Float) (ds : DimList) (ell : Loc) :
      ell = storeFreshLoc sigma →
      Step ⟨sigma, Term.const v ds⟩
           ⟨storeExtend sigma ell ⟨ds, v⟩, Term.loc ell⟩

  -- E-Copy: copy(ℓ)  ↦  (ℓ, ℓ')   with fresh ℓ' holding a physical copy
  -- The original location ℓ is retained (first component of the pair).
  | copy
      (sigma : Store) (ell ellNew : Loc) (w : TensorVal) :
      storeLookup sigma ell = some w →
      ellNew = storeFreshLoc sigma →
      Step ⟨sigma, Term.copy (Term.loc ell)⟩
           ⟨storeExtend sigma ellNew w,
            Term.pair (Term.loc ell) (Term.loc ellNew)⟩

  -- E-Add: add(ℓ1, ℓ2)  ↦  fresh ℓ with the elementwise sum; operands freed
  | tadd
      (sigma : Store) (ell1 ell2 ellOut : Loc) (w1 w2 : TensorVal) :
      storeLookup sigma ell1 = some w1 →
      storeLookup sigma ell2 = some w2 →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.add (Term.loc ell1) (Term.loc ell2)⟩
           ⟨storeExtend (storeRemove (storeRemove sigma ell1) ell2)
                        ellOut (tensorOpPlaceholder w1 w2),
            Term.loc ellOut⟩

  -- E-Mul: like E-Add.
  | tmul
      (sigma : Store) (ell1 ell2 ellOut : Loc) (w1 w2 : TensorVal) :
      storeLookup sigma ell1 = some w1 →
      storeLookup sigma ell2 = some w2 →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.mul (Term.loc ell1) (Term.loc ell2)⟩
           ⟨storeExtend (storeRemove (storeRemove sigma ell1) ell2)
                        ellOut (tensorOpPlaceholder w1 w2),
            Term.loc ellOut⟩

  -- E-Sum: sum(ℓ, i)  ↦  fresh ℓ' with the reduced tensor; operand freed
  | tsum
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (i : Nat) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.sum (Term.loc ell) i⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := rem w.shape i, data := w.data },
            Term.loc ellOut⟩

  -- E-Expand: expand(ℓ, i, k)  ↦  fresh ℓ' with the dim-inserted tensor
  | texpand
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (i k : Nat) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.expand (Term.loc ell) i k⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := ins w.shape i k, data := w.data },
            Term.loc ellOut⟩

  -- E-UniformLike: uniform_like(ℓ, lo, hi)  ↦  fresh ℓ' with random data
  -- **Round-2 fix**: the template location ℓ IS consumed, matching
  -- T-UniformLike's implicit consumption of its template.
  | tuniformLike
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (lo hi : Float) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.uniformLike (Term.loc ell) lo hi⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := w.shape, data := lo },
            Term.loc ellOut⟩

  /- ## Effect handlers -/

  -- E-Handle-Ret: handle[ε_h] v with h  ↦  v
  | handleRet
      (sigma : Store) (epsH : EffectRow) (v : Term)
      (clauses : List (EffectLabel × String × String × Term)) :
      IsValue v →
      Step ⟨sigma, Term.handle epsH v clauses⟩ ⟨sigma, v⟩

  -- E-Handle-Op (single-clause, direct form for Phase 1 skeleton).
  -- The full E-Handle-Op rule from opsem.tex captures an evaluation
  -- context E[·] around the `perform` and substitutes it for the
  -- continuation variable k_i. Phase 1 skeleton encodes only the
  -- "perform at the top of the body" special case; Phase 2 will add
  -- the general evaluation-context form via a separate constructor.
  --
  -- The continuation substituted for `k` is an identity lambda
  -- `λy:tRet. y` — the right-typed stand-in for the full captured
  -- context `λy:tRet. handle[epsH] E[y] with h`. Round-3 fix (M1):
  -- was previously a unit-constant lambda, which would not typecheck
  -- under T-Handle's expected continuation type.
  | handleOpDirect
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow) (x k : String) (handlerBody : Term) (tRet : Typ) :
      IsValue v →
      Step ⟨sigma,
            Term.handle epsH (Term.perform op v)
                        [(op, x, k, handlerBody)]⟩
           ⟨sigma,
            subst (subst handlerBody v x)
                  (Term.abs "y" tRet (Term.var "y")) k⟩

  /- ## AD and vectorization transforms -/

  -- E-Grad: grad(λx:t.e)  ↦  λx:t. λgs:tOut. handle[{Accum}]
  --                              (adjoint(e, x, gs))
  --                              with {accum(p, k) → k(p)}
  --
  -- The constructor takes `tOut` as an extra parameter so preservation can
  -- pick the output type of `e` from the typing derivation. The paper's
  -- E-Grad reduction is a typed operational rule — `tOut` is implicit in
  -- the typing context — but in Lean we need the type concretely or
  -- preservation for this rule cannot close.
  -- Round-3 fix (H3): the seed parameter type is `tOut`, read from
  -- the `Term.grad` constructor's new `tOut` field (Phase 2 Wave 0 P4).
  -- The handler clause body `app (var k) (var p)` is still a round-3
  -- skeleton placeholder for the real `update_origin_buffer(p)` routing.
  | tgrad
      (sigma : Store) (x : String) (t tOut : Typ) (e : Term) :
      Step ⟨sigma, Term.grad x t tOut e⟩
           ⟨sigma,
            Term.abs x t
              (Term.abs "gs" tOut
                (Term.handle [EffectLabel.accum]
                  (adjoint e x (Term.var "gs"))
                  [(EffectLabel.accum, "p", "k",
                    Term.app (Term.var "k") (Term.var "p"))]))⟩

  -- E-Vmap: vmap(λx:t.e)  ↦  λx:addDim(d, t). addDimTerm(d, e)
  -- Wave 0 P6: body is now lifted via `addDimTerm` (defined in
  -- `Syntax.lean`) rather than passing `e` through unchanged.
  | tvmap
      (sigma : Store) (x : String) (t : Typ) (e : Term) (d : Dim) :
      Step ⟨sigma, Term.vmap x t e⟩
           ⟨sigma, Term.abs x (addDim d t) (addDimTerm d e)⟩

end LaCaDiLE
