-- LaCaDiLE/Syntax.lean — Term, Typ, Dim, DimList, EffectRow, Capability, Value, Config.
-- Also houses the meta-level `def`s (addDim, DiffCompat, ins, rem, contextSplit)
-- referenced by the paper's typing rules (proof/paper/figures/typing.tex).
--
-- Phase 1 T6: structural skeleton matching the T2/T3 figures. No proofs.
--
-- Convention: the paper writes `d̄` for a dimension list; Lean identifiers
-- cannot contain combining marks, so we use `ds` (plural of `d`) instead.

namespace LaCaDiLE

/-! ## Dimensions and dimension lists -/

/-- Named dimensions, dimension variables, and literal extents.
    Corresponds to `d ::= n | δ | k` in figures/syntax.tex. -/
inductive Dim where
  | named (n : String)
  | var   (delta : String)
  | lit   (k : Nat)
  deriving DecidableEq, Repr

/-- A dimension list is an ordered sequence of dimensions (`d̄` in the paper). -/
abbrev DimList := List Dim

/-- Insert a dimension `d` into a dimension list. Stage 1 refactor:
    `ins` now takes a `Dim` argument instead of a `(position, extent)`
    pair. Semantics: prepend. Stage 2 will switch `DimList` to a
    multiset representation; at that point the prepend order becomes
    irrelevant. Paper notation: `ins(d̄, d)`. Used by T-Expand and
    E-Expand. -/
def ins (ds : DimList) (d : Dim) : DimList := d :: ds

/-- Remove a dimension `d` from a dimension list. Stage 1 refactor:
    `rem` now takes a `Dim` argument instead of a positional index.
    Semantics: `List.erase` — remove the first occurrence by equality.
    Paper notation: `rem(d̄, d)`. Used by T-Sum and E-Sum. -/
def rem (ds : DimList) (d : Dim) : DimList := ds.erase d

/-! ## Effect rows and capabilities -/

/-- Effect labels. `accum` is an internal effect introduced only by `grad`
    reductions and never appears in user-facing effect rows. -/
inductive EffectLabel where
  | random
  | resource
  | io
  | fail
  | accum
  deriving DecidableEq, Repr

/-- Effect rows are lists of effect labels. Phase 1 uses closed rows (no row
    variable `ρ`); full row polymorphism is a Phase 2+ extension. -/
abbrev EffectRow := List EffectLabel

/-- Check whether every label in `eps1` appears in `eps2`.
    Paper notation: `ε₁ ⊆ ε₂`. List-based but set-semantic: duplicates
    in `eps1` or `eps2` don't affect the result. -/
def subsetEffRow (eps1 eps2 : EffectRow) : Bool :=
  eps1.all (fun op => eps2.contains op)

/-- Propositional effect-row subset. Used by the `HasType.subEff`
    subsumption rule (Wave 0.5). Membership-preserving: every
    operation in the narrower row appears in the wider row. -/
def SubEffRow (eps1 eps2 : EffectRow) : Prop :=
  ∀ op, op ∈ eps1 → op ∈ eps2

/-- Union of two effect rows. Wave 0 P2: set semantics — duplicates
    from the right operand are dropped if already present in the left.
    `List.union` is not in Lean 4 core, so we spell it out: append
    `eps1` with the `eps2` elements that `eps1` does not contain. -/
def EffectRow.union (eps1 eps2 : EffectRow) : EffectRow :=
  eps1 ++ eps2.filter (fun op => !eps1.contains op)

/-- Remove every occurrence of `op` from an effect row. Wave 0 P2:
    unlike `List.erase` (which removes only the first occurrence),
    this removes all occurrences, so handling an effect leaves the
    residual effect row free of that operation regardless of
    accidental duplicates. -/
def EffectRow.removeOp (eps : EffectRow) (op : EffectLabel) : EffectRow :=
  eps.filter (fun o => o ≠ op)

/-- Remove every occurrence of every operation in `ops` from `eps`.
    Used by the multi-clause `T-Handle` rule (Wave 0 P5) to compute
    the residual effect row after a handle removes a set of
    operations. -/
def EffectRow.removeOps (eps : EffectRow) (ops : EffectRow) : EffectRow :=
  ops.foldr (fun op r => EffectRow.removeOp r op) eps

/-- The set of effects compatible with differentiation.
    Per decisions.md: `DiffCompat = {Resource, Accum}`. -/
def DiffCompat : EffectRow := [EffectLabel.resource, EffectLabel.accum]

/-- Capabilities. Phase 1 has a single capability `diff` tracked in a separate
    context `Delta` that threads unchanged through typing rules. -/
inductive Capability where
  | diff
  deriving DecidableEq, Repr

/-- Capability context (the `Δ` in `Δ; Γ ⊢ e : τ ! ε ⊣ Γ'`). -/
abbrev CapCtx := List Capability

/-! ## Types -/

/-- LaCaDiLE types.
    Corresponds to the `τ ::= ...` production in figures/syntax.tex. -/
inductive Typ where
  | tensor (ds : DimList)
  | arrow  (t1 : Typ) (t2 : Typ) (eps : EffectRow)
  | pair   (t1 : Typ) (t2 : Typ)
  | unit   : Typ
  | tyVar  (alpha : String)
  deriving Repr

/-- The `addDim` meta-function from figures/typing.tex. Prepends dimension `d`
    to every tensor inside `t`, traversing function, pair, and type-variable
    cases structurally. Paper definition:
    ```
    addDim(d, tensor[d̄])           = tensor[d, d̄]
    addDim(d, τ₁ → τ₂ ! ε)          = addDim(d, τ₁) → addDim(d, τ₂) ! ε
    addDim(d, τ₁ ⊗ τ₂)              = addDim(d, τ₁) ⊗ addDim(d, τ₂)
    addDim(d, unit)                 = unit
    addDim(d, α)                    = α
    ``` -/
def addDim (d : Dim) : Typ → Typ
  | Typ.tensor ds       => Typ.tensor (d :: ds)
  | Typ.arrow t1 t2 eps => Typ.arrow (addDim d t1) (addDim d t2) eps
  | Typ.pair t1 t2      => Typ.pair (addDim d t1) (addDim d t2)
  | Typ.unit            => Typ.unit
  | Typ.tyVar a         => Typ.tyVar a

/-- Per-operation signature lookup for effect operations (Wave 0 P7).
    `T-Perform op e` requires `e` to have type `(opSignature op).1` and
    produces a term of type `(opSignature op).2`.

    Phase 1 skeleton uses a uniform `(unit, unit)` signature for every
    op. Phase 2 WS2.7 must refine this per-operation before Phase 2
    WS2.2 (`adjoint_preserves_typing`) can discharge: the `adjoint`
    function emits `Term.perform EffectLabel.accum gSeed` where
    `gSeed : tensor[dsOut]`, but the current `opSignature` claims
    `accum` takes a `unit` argument. That mismatch will block WS2.2.

    Concrete Phase 2 refinements:
      * `accum : (Loc × TensorVal) → unit` (via a location-tagged
        pair or a type variable over tensor shapes)
      * `fail : unit → α` (via type variables once polymorphism is
        introduced)
      * `random` / `resource` / `io` signatures as needed.

    The refinement does not touch any typing rule — Phase 2 WS2.7 just
    edits this `def` and re-runs `lake build`. -/
def opSignature (_op : EffectLabel) : Typ × Typ :=
  (Typ.unit, Typ.unit)

/-- Wave 3 calculus refinement: a relational signature for effect
    operations. `OpSigMatch op tArg tRet` says that the operation `op`
    can be performed on an argument of type `tArg` producing a result
    of type `tRet`. The default cases match the original
    `opSignature`-based behavior; the `accum` case is polymorphic over
    tensor shapes so that the adjoint transformation can emit
    `perform accum gSeed` with `gSeed : tensor[ds]`.

    This is the calculus change unblocking `adjoint_preserves_typing`
    (Wave 2 WS2.2): the old `opSignature accum = (unit, unit)` made
    `perform accum gSeed` ill-typed whenever `gSeed` was a tensor. The
    `accumTensor` constructor below allows accum to take any
    `tensor[ds]` argument and produce `unit`. -/
inductive OpSigMatch : EffectLabel → Typ → Typ → Prop where
  | accumTensor (ds : DimList) :
      OpSigMatch EffectLabel.accum (Typ.tensor ds) Typ.unit
  | accumUnit :
      OpSigMatch EffectLabel.accum Typ.unit Typ.unit
  | random :
      OpSigMatch EffectLabel.random Typ.unit Typ.unit
  | resource :
      OpSigMatch EffectLabel.resource Typ.unit Typ.unit
  | io :
      OpSigMatch EffectLabel.io Typ.unit Typ.unit
  | fail :
      OpSigMatch EffectLabel.fail Typ.unit Typ.unit

/-! ## Locations and tensor values -/

/-- Store locations. Phase 1 uses `Nat`; uniqueness of fresh locations is a
    runtime invariant checked by the store well-formedness predicate. -/
abbrev Loc := Nat

/-- Opaque tensor values. Phase 1 treats `TensorVal` as a record of shape
    and payload. The metatheory does not inspect payloads; it only tracks
    shapes. -/
structure TensorVal where
  shape : DimList
  data  : Float
  deriving Repr

/-! ## Terms -/

/-- Terms of LaCaDiLE.
    Corresponds to the `e ::= ...` production in figures/syntax.tex.

    Handler clauses are inlined as 4-tuples `(op, arg-var, cont-var, body)`
    to avoid mutual recursion with `Term`. Every clause's continuation
    variable is a linear binding; the one-shot property follows from
    linearity via the T-Handle rule. -/
inductive Term where
  -- core λ-calculus
  | var     (x : String)
  | abs     (x : String) (t : Typ) (body : Term)
  | app     (e1 : Term) (e2 : Term)
  | letBind (x : String) (e1 : Term) (e2 : Term)
  -- linearity primitives
  | copy    (e : Term)
  | letpair (x : String) (y : String) (e1 : Term) (e2 : Term)
  -- pair constructors / projections / unit
  | pair    (e1 : Term) (e2 : Term)
  | fst     (e : Term)
  | snd     (e : Term)
  | unit    : Term
  -- RISC primitives
  | const       (v : Float) (ds : DimList)
  | add         (e1 : Term) (e2 : Term)
  | mul         (e1 : Term) (e2 : Term)
  | sum         (e : Term) (d : Dim)
  | expand      (e : Term) (d : Dim)
  | uniformLike (e : Term) (lo : Float) (hi : Float)
  -- AD / vectorization transforms (restricted to literal abstractions per T-Grad)
  | grad    (x : String) (t : Typ) (tOut : Typ) (body : Term)
  | vmap    (x : String) (t : Typ) (body : Term)
  -- effects: clauses are (op, arg-var, cont-var, body) tuples
  | handle  (epsH : EffectRow) (body : Term)
            (clauses : List (EffectLabel × String × String × Term))
  | perform (op : EffectLabel) (e : Term)
  -- runtime location (introduced by reduction, not in source programs)
  | loc     (ell : Loc)
  deriving Repr

/-! ## Term-level dimension lifting (Wave 0 P6) -/

-- The term-level `vmap` lifting. `addDimTerm d e` structurally walks
-- `e` and rewrites every tensor-producing subterm to operate over a
-- new fresh batch dimension `d`. Companion to `addDim : Dim → Typ → Typ`
-- and referenced by `E-Vmap` in `Operational.lean`. The handler-clause
-- case is delegated to `addDimClauses` in a `mutual` block so the
-- structural recursion checker sees each recursive call lands on a
-- strictly-smaller sub-term. Phase 2 WS2.3 proves `addDim_preserves_typing`.
mutual

/-- Term-level vmap lifting; see the comment above the `mutual` block. -/
def addDimTerm (d : Dim) : Term → Term
  | Term.var x => Term.var x
  | Term.abs x t body => Term.abs x (addDim d t) (addDimTerm d body)
  | Term.app e1 e2 => Term.app (addDimTerm d e1) (addDimTerm d e2)
  | Term.letBind x e1 e2 => Term.letBind x (addDimTerm d e1) (addDimTerm d e2)
  | Term.copy e => Term.copy (addDimTerm d e)
  | Term.letpair x y e1 e2 =>
      Term.letpair x y (addDimTerm d e1) (addDimTerm d e2)
  | Term.pair e1 e2 => Term.pair (addDimTerm d e1) (addDimTerm d e2)
  | Term.fst e => Term.fst (addDimTerm d e)
  | Term.snd e => Term.snd (addDimTerm d e)
  | Term.unit => Term.unit
  | Term.const v ds => Term.const v (d :: ds)
  | Term.add e1 e2 => Term.add (addDimTerm d e1) (addDimTerm d e2)
  | Term.mul e1 e2 => Term.mul (addDimTerm d e1) (addDimTerm d e2)
  -- Stage 1 refactor: `sum`/`expand` now reference dimensions by
  -- name, not by positional index. The new batch dim `d` is prepended
  -- at the type level but the reduced/expanded axis is unchanged.
  | Term.sum e d' => Term.sum (addDimTerm d e) d'
  | Term.expand e d' => Term.expand (addDimTerm d e) d'
  | Term.uniformLike e lo hi => Term.uniformLike (addDimTerm d e) lo hi
  | Term.grad x t tOut body =>
      Term.grad x (addDim d t) (addDim d tOut) (addDimTerm d body)
  | Term.vmap x t body => Term.vmap x (addDim d t) (addDimTerm d body)
  | Term.handle epsH body clauses =>
      Term.handle epsH (addDimTerm d body) (addDimClauses d clauses)
  | Term.perform op e => Term.perform op (addDimTerm d e)
  | Term.loc ell => Term.loc ell

/-- Companion to `addDimTerm`: lift a handler-clause list through
    `addDimTerm`, preserving clause structure and recursing on bodies. -/
def addDimClauses (d : Dim) :
    List (EffectLabel × String × String × Term) →
    List (EffectLabel × String × String × Term)
  | [] => []
  | (op, x, k, hb) :: rest => (op, x, k, addDimTerm d hb) :: addDimClauses d rest

end

/-! ## Free variables / closedness (Wave 1) -/

-- `freeVars e` is the list of free-variable name occurrences in `e`.
-- Binder cases filter out the bound name(s); handler clauses delegate
-- to `freeVarsClauses` in the mutual block so structural recursion
-- lands on strictly-smaller sub-terms. `Closed e` means `e` has no
-- free variables. Wave 1 Substitution uses `Closed v` as the side
-- condition that makes naive capture-unaware `subst` sound: a closed
-- value has no variables that could be captured by binders in `e`.
mutual

def freeVars : Term → List String
  | Term.var x => [x]
  | Term.abs x _ body => (freeVars body).filter (· != x)
  | Term.app e1 e2 => freeVars e1 ++ freeVars e2
  | Term.letBind x e1 e2 =>
      freeVars e1 ++ (freeVars e2).filter (· != x)
  | Term.copy e => freeVars e
  | Term.letpair x y e1 e2 =>
      freeVars e1 ++ (freeVars e2).filter (fun z => z != x && z != y)
  | Term.pair e1 e2 => freeVars e1 ++ freeVars e2
  | Term.fst e => freeVars e
  | Term.snd e => freeVars e
  | Term.unit => []
  | Term.const _ _ => []
  | Term.add e1 e2 => freeVars e1 ++ freeVars e2
  | Term.mul e1 e2 => freeVars e1 ++ freeVars e2
  | Term.sum e _ => freeVars e
  | Term.expand e _ => freeVars e
  | Term.uniformLike e _ _ => freeVars e
  | Term.grad x _ _ body => (freeVars body).filter (· != x)
  | Term.vmap x _ body => (freeVars body).filter (· != x)
  | Term.handle _ body clauses => freeVars body ++ freeVarsClauses clauses
  | Term.perform _ e => freeVars e
  | Term.loc _ => []

def freeVarsClauses :
    List (EffectLabel × String × String × Term) → List String
  | [] => []
  | (_, x, k, hb) :: rest =>
      (freeVars hb).filter (fun z => z != x && z != k) ++ freeVarsClauses rest

end

/-- A term is closed iff it has no free variables. Wave 1 Substitution
    uses `Closed v` as the side condition on the substituted value. -/
def Closed (e : Term) : Prop := freeVars e = []

/-! ## Bound-variable set (Wave 2 freshness predicate)

`boundVars e` lists every binder occurrence inside `e`. Combined with
`freeVars e`, this gives a sufficient freshness predicate for the
position-indexed `weakening_insert` lemma: a variable `y` that is
not in `freeVars e ∪ boundVars e` cannot be consumed by a T-Var
inside the derivation (free freshness) and cannot collide with any
binder (bound freshness), so weakening commutes through every
binder without needing an exchange lemma. -/
mutual

def boundVars : Term → List String
  | Term.var _ => []
  | Term.abs x _ body => x :: boundVars body
  | Term.app e1 e2 => boundVars e1 ++ boundVars e2
  | Term.letBind x e1 e2 => x :: (boundVars e1 ++ boundVars e2)
  | Term.copy e => boundVars e
  | Term.letpair x y e1 e2 => x :: y :: (boundVars e1 ++ boundVars e2)
  | Term.pair e1 e2 => boundVars e1 ++ boundVars e2
  | Term.fst e => boundVars e
  | Term.snd e => boundVars e
  | Term.unit => []
  | Term.const _ _ => []
  | Term.add e1 e2 => boundVars e1 ++ boundVars e2
  | Term.mul e1 e2 => boundVars e1 ++ boundVars e2
  | Term.sum e _ => boundVars e
  | Term.expand e _ => boundVars e
  | Term.uniformLike e _ _ => boundVars e
  | Term.grad x _ _ body => x :: boundVars body
  | Term.vmap x _ body => x :: boundVars body
  | Term.handle _ body clauses => boundVars body ++ boundVarsClauses clauses
  | Term.perform _ e => boundVars e
  | Term.loc _ => []

def boundVarsClauses :
    List (EffectLabel × String × String × Term) → List String
  | [] => []
  | (_, x, k, hb) :: rest => x :: k :: boundVars hb ++ boundVarsClauses rest

end

/-- Freshness of `y` with respect to term `e`: `y` is neither free nor
    bound anywhere inside `e`. Sufficient precondition for the
    position-indexed weakening lemma. -/
def freshInTerm (y : String) (e : Term) : Prop :=
  y ∉ freeVars e ∧ y ∉ boundVars e

/-! ## Values -/

/-- Value predicate on `Term`. A term is a value iff it is a location, a
    closure (`abs`), a pair of values, or unit. -/
inductive IsValue : Term → Prop
  | loc  (ell : Loc)                  : IsValue (Term.loc ell)
  | abs  (x : String) (t : Typ) (e : Term) : IsValue (Term.abs x t e)
  | pair (v1 v2 : Term)
         (h1 : IsValue v1) (h2 : IsValue v2) : IsValue (Term.pair v1 v2)
  | unit                                : IsValue Term.unit

/-! ## Store and configurations -/

/-- Store: association list from locations to tensor values. Phase 1 uses
    a simple list; an efficient representation is not needed for metatheory. -/
abbrev Store := List (Loc × TensorVal)

/-- Look up a location in the store. -/
def storeLookup (sigma : Store) (ell : Loc) : Option TensorVal :=
  (sigma.find? (fun p => p.1 = ell)).map Prod.snd

/-- Remove a location from the store. -/
def storeRemove (sigma : Store) (ell : Loc) : Store :=
  sigma.filter (fun p => p.1 ≠ ell)

/-- Extend the store with a fresh location. Callers must ensure `ell` is fresh. -/
def storeExtend (sigma : Store) (ell : Loc) (w : TensorVal) : Store :=
  (ell, w) :: sigma

/-- Pick a fresh location: one greater than the maximum currently in use. -/
def storeFreshLoc (sigma : Store) : Loc :=
  (sigma.map Prod.fst).foldl max 0 + 1

/-- An operational configuration is a store paired with a term. -/
structure Config where
  store : Store
  term  : Term
  deriving Repr

/-! ## Linear context and context splitting -/

/-- Linear typing context. A list of (name, type) pairs; entries are consumed
    by successful derivations and removed from the output context. -/
abbrev LinearCtx := List (String × Typ)

/-! ### Phase 2 scaffolding: context splitting

The definitions below implement the paper's `Γ = Γ₁ + Γ₂` splitting
relation. They are NOT used by any Phase 1 `HasType` constructor — every
multi-premise rule in `Typing.lean` instead threads linear contexts
left-to-right (`Γ₁ → Γ₂ → Γ₃`), which is equivalent to a chain of
splits but avoids a standalone predicate. Phase 2 WS2.1 (the
substitution lemma) will need a permutation-aware split relation, and
these stubs are the starting point for that work.

The current `contextSplit` is a length-and-disjointness weakening of
the full permutation invariant; Phase 2 will either strengthen this
definition or derive it as a corollary of the threading form. Flagged
in the Phase 1 → Phase 2 gate audit (H-A1) as latent scaffolding;
kept in place rather than deleted because the Phase 2 substitution
proof will need something in this shape. -/

/-- Domain of a linear context (the names of the bindings). -/
def linearCtxDom (G : LinearCtx) : List String :=
  G.map Prod.fst

/-- Disjointness of two linear contexts' domains. Phase 2 scaffolding. -/
def linearCtxDisjoint (G1 G2 : LinearCtx) : Prop :=
  ∀ x, x ∈ linearCtxDom G1 → x ∉ linearCtxDom G2

/-- Context splitting: `G = G1 + G2` when every binding of `G` appears in
    exactly one of `G1` or `G2`, with no shared bindings. Paper notation:
    `Γ = Γ₁ + Γ₂`. The full permutation invariant (that the union is a
    reordering of `G`) is deferred to Phase 2 substitution-lemma work.
    Phase 2 scaffolding — not referenced by any Phase 1 typing rule. -/
def contextSplit (G G1 G2 : LinearCtx) : Prop :=
  linearCtxDisjoint G1 G2 ∧ (G1 ++ G2).length = G.length

end LaCaDiLE
