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

/-- Insert a literal extent `k` at position `i` in a dimension list.
    Paper notation: `ins(d̄, i, k)`. Used by T-Expand and E-Expand. -/
def ins (ds : DimList) (i : Nat) (k : Nat) : DimList :=
  let dNew := Dim.lit k
  match i, ds with
  | 0, rest => dNew :: rest
  | Nat.succ _, [] => [dNew]
  | Nat.succ n, d :: rest => d :: ins rest n k

/-- Remove the dimension at position `i`. Paper notation: `rem(d̄, i)`.
    Used by T-Sum and E-Sum. -/
def rem (ds : DimList) (i : Nat) : DimList :=
  match i, ds with
  | _, [] => []
  | 0, _ :: rest => rest
  | Nat.succ n, d :: rest => d :: rem rest n

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
    Paper notation: `ε₁ ⊆ ε₂`. -/
def subsetEffRow (eps1 eps2 : EffectRow) : Bool :=
  eps1.all (fun op => eps2.contains op)

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
  | sum         (e : Term) (i : Nat)
  | expand      (e : Term) (i : Nat) (k : Nat)
  | uniformLike (e : Term) (lo : Float) (hi : Float)
  -- AD / vectorization transforms (restricted to literal abstractions per T-Grad)
  | grad    (x : String) (t : Typ) (body : Term)
  | vmap    (x : String) (t : Typ) (body : Term)
  -- effects: clauses are (op, arg-var, cont-var, body) tuples
  | handle  (epsH : EffectRow) (body : Term)
            (clauses : List (EffectLabel × String × String × Term))
  | perform (op : EffectLabel) (e : Term)
  -- runtime location (introduced by reduction, not in source programs)
  | loc     (ell : Loc)
  deriving Repr

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

/-- Domain of a linear context (the names of the bindings). -/
def linearCtxDom (G : LinearCtx) : List String :=
  G.map Prod.fst

/-- Disjointness of two linear contexts' domains. -/
def linearCtxDisjoint (G1 G2 : LinearCtx) : Prop :=
  ∀ x, x ∈ linearCtxDom G1 → x ∉ linearCtxDom G2

/-- Context splitting: `G = G1 + G2` when every binding of `G` appears in
    exactly one of `G1` or `G2`, with no shared bindings. Paper notation:
    `Γ = Γ₁ + Γ₂`. The full permutation invariant (that the union is a
    reordering of `G`) is deferred to Phase 2 substitution-lemma work. -/
def contextSplit (G G1 G2 : LinearCtx) : Prop :=
  linearCtxDisjoint G1 G2 ∧ (G1 ++ G2).length = G.length

end LaCaDiLE
