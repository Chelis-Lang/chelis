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

-- Naive capture-unaware substitution: replace every free occurrence of
-- `x` in `target` with `v`. Stops descending into scopes that shadow
-- `x`. Wave 1 makes this a total `def` (no `partial`) via mutual
-- recursion with `substClauses` so Lean's structural checker accepts
-- the handler-clause case without `List.map`, and equation lemmas are
-- available for unfolding in proofs.
mutual

def subst (target : Term) (v : Term) (x : String) : Term :=
  match target with
  | Term.var y => if y = x then v else Term.var y
  | Term.abs y t body =>
      if y = x then Term.abs y t body else Term.abs y t (subst body v x)
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
  | Term.sum e d => Term.sum (subst e v x) d
  | Term.expand e d => Term.expand (subst e v x) d
  | Term.uniformLike e lo hi => Term.uniformLike (subst e v x) lo hi
  | Term.grad y t tOut body =>
      if y = x then Term.grad y t tOut body
      else Term.grad y t tOut (subst body v x)
  | Term.vmap y t body =>
      if y = x then Term.vmap y t body
      else Term.vmap y t (subst body v x)
  | Term.handle epsH body clauses =>
      Term.handle epsH (subst body v x) (substClauses clauses v x)
  | Term.perform op e => Term.perform op (subst e v x)
  | Term.loc ell => Term.loc ell

def substClauses
    (clauses : List (EffectLabel × String × String × Term))
    (v : Term) (x : String) :
    List (EffectLabel × String × String × Term) :=
  match clauses with
  | [] => []
  | (op, y, k, body) :: rest =>
      let body' := if y = x ∨ k = x then body else subst body v x
      (op, y, k, body') :: substClauses rest v x

end

/-- Capture a tensor value at location `ell` into a `TensorVal`. Used in
    placeholders for primitive reduction rules; the actual pointwise
    operations (`oplus`, `odot`, etc.) are not exercised by Phase 1
    metatheory. -/
def tensorOpPlaceholder (_l1 _l2 : TensorVal) : TensorVal :=
  { shape := DimList.empty, data := 0.0 }

/-! ## Evaluation contexts (Wave 0.5)

Call-by-value left-to-right evaluation contexts. `EvalCtx` enumerates
the positions where reduction can happen under a compound term; each
constructor names one such position, and `plug E e` produces the term
with `e` substituted at the hole. The `Step.ctx` congruence rule
reduces `plug E e` to `plug E e'` whenever `e` reduces to `e'`. This
is the standard workaround for needing one structural reduction rule
per (term-former × sub-position) combination. -/

inductive EvalCtx where
  | hole                                            : EvalCtx
  | appL       (e2 : Term)                          : EvalCtx
  | appR       (v1 : Term)                          : EvalCtx
  | letBind    (x : String) (e2 : Term)             : EvalCtx
  | copy                                            : EvalCtx
  | letpair    (x y : String) (e2 : Term)           : EvalCtx
  | pairL      (e2 : Term)                          : EvalCtx
  | pairR      (v1 : Term)                          : EvalCtx
  | fst                                             : EvalCtx
  | snd                                             : EvalCtx
  | addL       (e2 : Term)                          : EvalCtx
  | addR       (v1 : Term)                          : EvalCtx
  | mulL       (e2 : Term)                          : EvalCtx
  | mulR       (v1 : Term)                          : EvalCtx
  | sum        (d : Dim)                            : EvalCtx
  | expand     (d : Dim)                            : EvalCtx
  | uniformLike (lo hi : Float)                     : EvalCtx
  | handle     (epsH : EffectRow)
               (clauses : List (EffectLabel × String × String × Term))
                                                    : EvalCtx
  | perform    (op : EffectLabel)                   : EvalCtx
  deriving Repr

/-- Plug a term into an evaluation context, producing the compound
    term with the hole replaced by `e`. -/
def plug : EvalCtx → Term → Term
  | EvalCtx.hole, e                => e
  | EvalCtx.appL e2, e             => Term.app e e2
  | EvalCtx.appR v1, e             => Term.app v1 e
  | EvalCtx.letBind x e2, e        => Term.letBind x e e2
  | EvalCtx.copy, e                => Term.copy e
  | EvalCtx.letpair x y e2, e      => Term.letpair x y e e2
  | EvalCtx.pairL e2, e            => Term.pair e e2
  | EvalCtx.pairR v1, e            => Term.pair v1 e
  | EvalCtx.fst, e                 => Term.fst e
  | EvalCtx.snd, e                 => Term.snd e
  | EvalCtx.addL e2, e             => Term.add e e2
  | EvalCtx.addR v1, e             => Term.add v1 e
  | EvalCtx.mulL e2, e             => Term.mul e e2
  | EvalCtx.mulR v1, e             => Term.mul v1 e
  | EvalCtx.sum d, e               => Term.sum e d
  | EvalCtx.expand d, e            => Term.expand e d
  | EvalCtx.uniformLike lo hi, e   => Term.uniformLike e lo hi
  | EvalCtx.handle epsH clauses, e => Term.handle epsH e clauses
  | EvalCtx.perform op, e          => Term.perform op e

/-- `EvalCtx.noHandleFor op E` holds when the single-step evaluation
    context `E` is not itself a `handle` whose effect row catches `op`.
    Because `EvalCtx` is a one-step (non-recursive) context, this is a
    simple case split: only the `handle` constructor can catch an
    operation, and it does so exactly when `op ∈ epsH`. All other
    contexts trivially do not catch anything at their own level. -/
def EvalCtx.noHandleFor (op : EffectLabel) : EvalCtx → Prop
  | EvalCtx.handle epsH _ => op ∉ epsH
  | _                     => True

/-- The small-step reduction relation. Constructors cover the head
    reductions (redex at top position); `Step.ctx` provides the
    congruence closure via evaluation contexts. -/
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

  -- E-Sum: sum(ℓ, d)  ↦  fresh ℓ' with the reduced tensor; operand freed
  | tsum
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (d : Dim) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.sum (Term.loc ell) d⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := rem w.shape d, data := w.data },
            Term.loc ellOut⟩

  -- E-Expand: expand(ℓ, d)  ↦  fresh ℓ' with the dim-inserted tensor
  | texpand
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (d : Dim) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.expand (Term.loc ell) d⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := ins w.shape d, data := w.data },
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

  -- E-Handle-Op (multi-clause direct form, Wave 0 P5 + W0 audit fix).
  -- When the body is a top-level `perform op v`, the reduction looks
  -- up the matching clause `(op, x, k, handlerBody)` in the clause
  -- list and substitutes `v` for `x` and an identity continuation
  -- `λy:tRet. y` for `k`. Multiple clauses are supported: the
  -- membership premise picks whichever clause's op matches.
  --
  -- `handleOpDirect` is the `E = hole` special case of the full
  -- E-Handle-Op rule (see `handleOpCtx` below). It is kept as a
  -- standalone constructor because it uses the identity continuation
  -- `λy:tRet. y` rather than the general reified context, and
  -- Preservation's hand-closed cases refer to it by name.
  -- Wave 1 P1 update: the Phase 1 "top-of-body only" limitation that
  -- this rule used to document is now lifted by the companion
  -- `handleOpCtx` constructor, which fires when `perform` sits inside
  -- a non-trivial evaluation context.
  -- Wave 0 W0-audit M-W0-3 fix: the rule now matches any clause list
  -- in which the matching clause appears, not just a singleton list.
  | handleOpDirect
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow)
      (clauses : List (EffectLabel × String × String × Term))
      (x k : String) (handlerBody : Term) (tRet : Typ) :
      IsValue v →
      (op, x, k, handlerBody) ∈ clauses →
      Step ⟨sigma,
            Term.handle epsH (Term.perform op v) clauses⟩
           ⟨sigma,
            subst (subst handlerBody v x)
                  (Term.abs "y" tRet (Term.var "y")) k⟩

  -- E-Handle-Op (paper-accurate, captured-context form). When the
  -- handle body has the shape `plug E (perform op v)` with `E` an
  -- evaluation context that does NOT itself catch `op`, the step
  -- picks the matching clause `(op, xVar, kVar, hb)` from the clause
  -- list and reduces to
  --   `hb[v / xVar][λ _kArg : tRet. handle epsH (plug E (var _kArg))
  --                     clauses / kVar]`.
  -- The reified continuation re-enters the same handle around the
  -- captured context `E`, which is the standard delimited-control
  -- semantics for algebraic effects. `tRet` is the return type of
  -- the operation's signature (threaded in as a parameter, matching
  -- `handleOpDirect`'s convention). Freshness of `_kArg` is provided
  -- by a hardcoded sentinel name; a proper fresh-name discipline is
  -- a separate Wave 1 task (see Phase 1 WS2.1 notes).
  | handleOpCtx
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow) (E : EvalCtx)
      (clauses : List (EffectLabel × String × String × Term))
      (xVar kVar : String) (hb : Term) (tRet : Typ) :
      IsValue v →
      (op, xVar, kVar, hb) ∈ clauses →
      op ∈ epsH →
      EvalCtx.noHandleFor op E →
      Step ⟨sigma,
            Term.handle epsH (plug E (Term.perform op v)) clauses⟩
           ⟨sigma,
            subst (subst hb v xVar)
                  (Term.abs "_kArg" tRet
                    (Term.handle epsH
                      (plug E (Term.var "_kArg")) clauses)) kVar⟩

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

  -- E-Ctx: congruence closure via evaluation contexts (Wave 0.5).
  -- If `e` steps to `e'`, then `plug E e` steps to `plug E e'` for
  -- any evaluation context `E`. This collapses what would otherwise
  -- be one congruence constructor per (term-former × position) pair
  -- into a single parameterized rule.
  | ctx
      (sigma sigma' : Store) (E : EvalCtx) (e e' : Term) :
      Step ⟨sigma, e⟩ ⟨sigma', e'⟩ →
      Step ⟨sigma, plug E e⟩ ⟨sigma', plug E e'⟩

end LaCaDiLE
