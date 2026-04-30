-- LaCaDiLE/Syntax.lean — Term, Typ, Dim, DimList, EffectRow, Capability, Value, Config.
-- Also houses the meta-level `def`s (addDim, DiffCompat, ins, rem, contextSplit)
-- referenced by the paper's typing rules (proof/paper/figures/typing.tex).
--
-- Phase 1 T6: structural skeleton matching the T2/T3 figures. No proofs.
--
-- Convention: the paper writes `d̄` for a dimension list; Lean identifiers
-- cannot contain combining marks, so we use `ds` (plural of `d`) instead.

import LaCaDiLE.StringHelpers

namespace LaCaDiLE

/-! ## Dimensions and dimension lists -/

/-- Named dimensions, dimension variables, and literal extents.
    Corresponds to `d ::= n | δ | k` in figures/syntax.tex. -/
inductive Dim where
  | named (n : String)
  | var   (delta : String)
  | lit   (k : Nat)
  deriving DecidableEq, Repr

/-- Setoid on `List Dim` via `List.Perm`. `DimList` is a quotient of
    `List Dim` under permutation equivalence — a "dimension multiset".
    Stage 2 refactor: this replaces the Stage 1 `abbrev DimList := List Dim`
    so that `d1 :: d2 :: ds` and `d2 :: d1 :: ds` are identified at the
    type level, making `addDim_comm` provable via a one-line
    `Quotient.sound (List.Perm.swap ...)`. -/
def ListDimSetoid : Setoid (List Dim) :=
  { r := List.Perm
  , iseqv := ⟨List.Perm.refl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩ }

/-- A multiset of dimensions. Currently `Quotient ListDimSetoid`; could
    be swapped for a sorted-list normal form in a future refinement
    without changing any client code (clients only use `DimList.mk`,
    `cons`, `erase`, `mem`, `length`, and opaque equality). Paper
    notation: `d̄`. -/
def DimList : Type := Quotient ListDimSetoid

namespace DimList

/-- Inject a concrete `List Dim` into the quotient. -/
def mk (l : List Dim) : DimList := Quotient.mk ListDimSetoid l

/-- The empty dimension list. -/
def empty : DimList := mk []

/-- Cons a dimension onto a `DimList`. Stage 2 note:
    `cons d1 (cons d2 ds) = cons d2 (cons d1 ds)` holds propositionally
    (by `Quotient.sound (List.Perm.swap ...)`), which is what unlocks
    the tvmap/tsum/texpand cases of `addDim_preserves_typing`. -/
def cons (d : Dim) (ds : DimList) : DimList :=
  Quotient.lift (fun l => mk (d :: l))
    (fun a b (h : List.Perm a b) =>
      Quotient.sound (s := ListDimSetoid) (List.Perm.cons d h)) ds

/-- Erase the first occurrence of `d` from a `DimList`. Respects the
    permutation quotient because `List.Perm.erase` carries permutation
    through `List.erase`. -/
def erase (ds : DimList) (d : Dim) : DimList :=
  Quotient.lift (fun l => mk (l.erase d))
    (fun a b (h : List.Perm a b) =>
      Quotient.sound (s := ListDimSetoid) (List.Perm.erase d h)) ds

/-- Membership on `DimList`. Permutation preserves membership, so this
    descends to the quotient. -/
def mem (d : Dim) (ds : DimList) : Prop :=
  Quotient.lift (fun l => d ∈ l)
    (fun a b (h : List.Perm a b) => propext h.mem_iff) ds

instance : Membership Dim DimList := ⟨fun ds d => mem d ds⟩

/-- Length of a `DimList`. Permutations preserve length. -/
def length (ds : DimList) : Nat :=
  Quotient.lift List.length (fun a b (h : List.Perm a b) => h.length_eq) ds

end DimList

/-- Trivial `Repr` for `DimList`: a quotient has no canonical
    representative to pretty-print, so we emit a placeholder. Required
    so that the `deriving Repr` on `Typ` and `TensorVal` can discharge
    their dependency on a `Repr DimList` instance. -/
instance : Repr DimList where
  reprPrec _ _ := "«DimList»"

/-- Insert a dimension `d` into a dimension list. Stage 2 refactor:
    operates on the quotient `DimList` via `DimList.cons`. Paper
    notation: `ins(d̄, d)`. Used by T-Expand and E-Expand. -/
def ins (ds : DimList) (d : Dim) : DimList := DimList.cons d ds

/-- Remove a dimension `d` from a dimension list. Stage 2 refactor:
    operates on the quotient `DimList` via `DimList.erase`. Paper
    notation: `rem(d̄, d)`. Used by T-Sum and E-Sum. -/
def rem (ds : DimList) (d : Dim) : DimList := DimList.erase ds d

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
  | Typ.tensor ds       => Typ.tensor (DimList.cons d ds)
  | Typ.arrow t1 t2 eps => Typ.arrow (addDim d t1) (addDim d t2) eps
  | Typ.pair t1 t2      => Typ.pair (addDim d t1) (addDim d t2)
  | Typ.unit            => Typ.unit
  | Typ.tyVar a         => Typ.tyVar a

/-- Canonical argument type for each built-in effect label.
    Handler dispatch is by label only, so each label must determine a
    unique argument type. -/
def opArgType : EffectLabel → Typ
  | EffectLabel.accum => Typ.unit
  | EffectLabel.random => Typ.unit
  | EffectLabel.resource => Typ.unit
  | EffectLabel.io => Typ.unit
  | EffectLabel.fail => Typ.unit

/-- Canonical return type for each built-in effect label. -/
def opRetType : EffectLabel → Typ
  | EffectLabel.accum => Typ.unit
  | EffectLabel.random => Typ.unit
  | EffectLabel.resource => Typ.unit
  | EffectLabel.io => Typ.unit
  | EffectLabel.fail => Typ.unit

/-- Per-operation signature lookup for effect operations. -/
def opSignature (op : EffectLabel) : Typ × Typ :=
  (opArgType op, opRetType op)

/-- Functional operation-signature witness used by the typing and
    operational rules. The witness is proof-irrelevant bookkeeping:
    a clause-local `(tArg, tRet)` pair is accepted exactly when it
    matches the canonical signature lookup for `op`. -/
def OpSigMatch (op : EffectLabel) (tArg tRet : Typ) : Prop :=
  tArg = opArgType op ∧ tRet = opRetType op

theorem opSignature_arg_unique
    {op : EffectLabel} {tArg1 tArg2 : Typ}
    (h1 : tArg1 = opArgType op)
    (h2 : tArg2 = opArgType op) :
    tArg1 = tArg2 := by
  rw [h1, h2]

theorem opSignature_ret_unique
    {op : EffectLabel} {tRet1 tRet2 : Typ}
    (h1 : tRet1 = opRetType op)
    (h2 : tRet2 = opRetType op) :
    tRet1 = tRet2 := by
  rw [h1, h2]

theorem OpSigMatch.arg_unique
    {op : EffectLabel} {tArg1 tArg2 tRet1 tRet2 : Typ}
    (h1 : OpSigMatch op tArg1 tRet1)
    (h2 : OpSigMatch op tArg2 tRet2) :
    tArg1 = tArg2 := by
  exact opSignature_arg_unique h1.1 h2.1

theorem OpSigMatch.ret_unique
    {op : EffectLabel} {tArg1 tArg2 tRet1 tRet2 : Typ}
    (h1 : OpSigMatch op tArg1 tRet1)
    (h2 : OpSigMatch op tArg2 tRet2) :
    tRet1 = tRet2 := by
  exact opSignature_ret_unique h1.2 h2.2

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
  | fst     (tRight : Typ) (e : Term)
  | snd     (tLeft : Typ) (e : Term)
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
  | vmap    (x : String) (t : Typ) (d : Dim) (body : Term)
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
  | Term.fst tRight e => Term.fst (addDim d tRight) (addDimTerm d e)
  | Term.snd tLeft e => Term.snd (addDim d tLeft) (addDimTerm d e)
  | Term.unit => Term.unit
  | Term.const v ds => Term.const v (DimList.cons d ds)
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
  | Term.vmap x t dMap body => Term.vmap x (addDim d t) dMap (addDimTerm d body)
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
  | Term.fst _ e => freeVars e
  | Term.snd _ e => freeVars e
  | Term.unit => []
  | Term.const _ _ => []
  | Term.add e1 e2 => freeVars e1 ++ freeVars e2
  | Term.mul e1 e2 => freeVars e1 ++ freeVars e2
  | Term.sum e _ => freeVars e
  | Term.expand e _ => freeVars e
  | Term.uniformLike e _ _ => freeVars e
  | Term.grad x _ _ body => (freeVars body).filter (· != x)
  | Term.vmap x _ _ body => (freeVars body).filter (· != x)
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

/-! ## Runtime-location references

`locRefs e` collects every runtime location mentioned explicitly inside
`e`. This is a meta-level runtime invariant surface, separate from
named-variable scoping: source terms never contain `Term.loc`, but
reduction introduces them. Preservation's remaining `ctx` blocker needs
this kind of term-level location accounting, not just `StoreWf`. -/
mutual

def locRefs : Term → List Loc
  | Term.var _ => []
  | Term.abs _ _ body => locRefs body
  | Term.app e1 e2 => locRefs e1 ++ locRefs e2
  | Term.letBind _ e1 e2 => locRefs e1 ++ locRefs e2
  | Term.copy e => locRefs e
  | Term.letpair _ _ e1 e2 => locRefs e1 ++ locRefs e2
  | Term.pair e1 e2 => locRefs e1 ++ locRefs e2
  | Term.fst _ e => locRefs e
  | Term.snd _ e => locRefs e
  | Term.unit => []
  | Term.const _ _ => []
  | Term.add e1 e2 => locRefs e1 ++ locRefs e2
  | Term.mul e1 e2 => locRefs e1 ++ locRefs e2
  | Term.sum e _ => locRefs e
  | Term.expand e _ => locRefs e
  | Term.uniformLike e _ _ => locRefs e
  | Term.grad _ _ _ body => locRefs body
  | Term.vmap _ _ _ body => locRefs body
  | Term.handle _ body clauses => locRefs body ++ locRefsClauses clauses
  | Term.perform _ e => locRefs e
  | Term.loc ell => [ell]

def locRefsClauses :
    List (EffectLabel × String × String × Term) → List Loc
  | [] => []
  | (_, _, _, hb) :: rest => locRefs hb ++ locRefsClauses rest

end

mutual

/-- Lifting a term through `addDimTerm` changes tensor shapes but does
    not change which runtime locations the term mentions. -/
theorem locRefs_addDimTerm (d : Dim) :
    ∀ e : Term, locRefs (addDimTerm d e) = locRefs e
  | Term.var _ => rfl
  | Term.abs _ _ body => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d body
  | Term.app e1 e2 => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d e1, locRefs_addDimTerm d e2]
  | Term.letBind _ e1 e2 => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d e1, locRefs_addDimTerm d e2]
  | Term.copy e => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.letpair _ _ e1 e2 => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d e1, locRefs_addDimTerm d e2]
  | Term.pair e1 e2 => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d e1, locRefs_addDimTerm d e2]
  | Term.fst _ e => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.snd _ e => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.unit => rfl
  | Term.const _ _ => rfl
  | Term.add e1 e2 => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d e1, locRefs_addDimTerm d e2]
  | Term.mul e1 e2 => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d e1, locRefs_addDimTerm d e2]
  | Term.sum e _ => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.expand e _ => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.uniformLike e _ _ => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.grad _ _ _ body => by
      simpa [addDimTerm, locRefs] using locRefs_addDimTerm d body
  | Term.vmap _ _ _ body => by
      simpa [addDimTerm, locRefs] using locRefs_addDimTerm d body
  | Term.handle _ body clauses => by
      simp [addDimTerm, locRefs, locRefs_addDimTerm d body, locRefs_addDimClauses d clauses]
  | Term.perform _ e => by simpa [addDimTerm, locRefs] using locRefs_addDimTerm d e
  | Term.loc _ => rfl

/-- Companion to `locRefs_addDimTerm` for handler clauses. -/
theorem locRefs_addDimClauses (d : Dim) :
    ∀ clauses : List (EffectLabel × String × String × Term),
      locRefsClauses (addDimClauses d clauses) = locRefsClauses clauses
  | [] => rfl
  | (_, _, _, hb) :: rest => by
      simp [addDimClauses, locRefsClauses, locRefs_addDimTerm d hb, locRefs_addDimClauses d rest]

end

mutual

/-- `activeLocRefs e` collects only the explicit runtime locations that
    are active in the current evaluation surface of `e`. Unlike
    `locRefs`, this does not descend into dormant handler clause bodies:
    a clause body only becomes active after a matching `perform`
    selects it. Lambda/grad/vmap bodies are still counted because the
    next head step can activate them immediately via substitution. -/
def activeLocRefs : Term → List Loc
  | Term.var _ => []
  | Term.abs _ _ body => activeLocRefs body
  | Term.app e1 e2 => activeLocRefs e1 ++ activeLocRefs e2
  | Term.letBind _ e1 e2 => activeLocRefs e1 ++ activeLocRefs e2
  | Term.copy e => activeLocRefs e
  | Term.letpair _ _ e1 e2 => activeLocRefs e1 ++ activeLocRefs e2
  | Term.pair e1 e2 => activeLocRefs e1 ++ activeLocRefs e2
  | Term.fst _ e => activeLocRefs e
  | Term.snd _ e => activeLocRefs e
  | Term.unit => []
  | Term.const _ _ => []
  | Term.add e1 e2 => activeLocRefs e1 ++ activeLocRefs e2
  | Term.mul e1 e2 => activeLocRefs e1 ++ activeLocRefs e2
  | Term.sum e _ => activeLocRefs e
  | Term.expand e _ => activeLocRefs e
  | Term.uniformLike e _ _ => activeLocRefs e
  | Term.grad _ _ _ body => activeLocRefs body
  | Term.vmap _ _ _ body => activeLocRefs body
  | Term.handle _ body clauses => activeLocRefs body ++ activeLocRefsClauses clauses
  | Term.perform _ e => activeLocRefs e
  | Term.loc ell => [ell]

def activeLocRefsClauses :
    List (EffectLabel × String × String × Term) → List Loc
  | [] => []
  | _ :: rest => activeLocRefsClauses rest

end

@[simp] theorem activeLocRefsClauses_eq_nil
    (clauses : List (EffectLabel × String × String × Term)) :
    activeLocRefsClauses clauses = [] := by
  induction clauses with
  | nil =>
      simp [activeLocRefsClauses]
  | cons _ rest ih =>
      simp [activeLocRefsClauses, ih]

/-- Runtime linearity discipline for explicit locations: no location may
    appear more than once in the residual runtime term. This is stronger
    than `WellScoped` and is only meaningful after reduction has
    introduced `Term.loc`. -/
def ActiveRuntimeLinear (e : Term) : Prop :=
  (activeLocRefs e).Nodup

def RuntimeLinear (e : Term) : Prop :=
  (locRefs e).Nodup

mutual

/-- Recursive closure of `ActiveRuntimeLinear`: every subterm must have
    a duplicate-free active runtime footprint, and dormant handler
    clause bodies are checked recursively instead of being ignored
    wholesale. This is a candidate invariant for iterating
    preservation across handler steps. -/
def DeepActiveRuntimeLinear : Term → Prop
  | Term.var _ => True
  | Term.abs x t body =>
      ActiveRuntimeLinear (Term.abs x t body) ∧
      DeepActiveRuntimeLinear body
  | Term.app e1 e2 =>
      ActiveRuntimeLinear (Term.app e1 e2) ∧
      DeepActiveRuntimeLinear e1 ∧
      DeepActiveRuntimeLinear e2
  | Term.letBind x e1 e2 =>
      ActiveRuntimeLinear (Term.letBind x e1 e2) ∧
      DeepActiveRuntimeLinear e1 ∧
      DeepActiveRuntimeLinear e2
  | Term.copy e =>
      ActiveRuntimeLinear (Term.copy e) ∧
      DeepActiveRuntimeLinear e
  | Term.letpair x y e1 e2 =>
      ActiveRuntimeLinear (Term.letpair x y e1 e2) ∧
      DeepActiveRuntimeLinear e1 ∧
      DeepActiveRuntimeLinear e2
  | Term.pair e1 e2 =>
      ActiveRuntimeLinear (Term.pair e1 e2) ∧
      DeepActiveRuntimeLinear e1 ∧
      DeepActiveRuntimeLinear e2
  | Term.fst tRight e =>
      ActiveRuntimeLinear (Term.fst tRight e) ∧
      DeepActiveRuntimeLinear e
  | Term.snd tLeft e =>
      ActiveRuntimeLinear (Term.snd tLeft e) ∧
      DeepActiveRuntimeLinear e
  | Term.unit => True
  | Term.const _ _ => True
  | Term.add e1 e2 =>
      ActiveRuntimeLinear (Term.add e1 e2) ∧
      DeepActiveRuntimeLinear e1 ∧
      DeepActiveRuntimeLinear e2
  | Term.mul e1 e2 =>
      ActiveRuntimeLinear (Term.mul e1 e2) ∧
      DeepActiveRuntimeLinear e1 ∧
      DeepActiveRuntimeLinear e2
  | Term.sum e d =>
      ActiveRuntimeLinear (Term.sum e d) ∧
      DeepActiveRuntimeLinear e
  | Term.expand e d =>
      ActiveRuntimeLinear (Term.expand e d) ∧
      DeepActiveRuntimeLinear e
  | Term.uniformLike e lo hi =>
      ActiveRuntimeLinear (Term.uniformLike e lo hi) ∧
      DeepActiveRuntimeLinear e
  | Term.grad x t tOut body =>
      ActiveRuntimeLinear (Term.grad x t tOut body) ∧
      DeepActiveRuntimeLinear body
  | Term.vmap x t d body =>
      ActiveRuntimeLinear (Term.vmap x t d body) ∧
      DeepActiveRuntimeLinear body
  | Term.handle epsH body clauses =>
      ActiveRuntimeLinear (Term.handle epsH body clauses) ∧
      DeepActiveRuntimeLinear body ∧
      DeepActiveRuntimeLinearClauses clauses
  | Term.perform op e =>
      ActiveRuntimeLinear (Term.perform op e) ∧
      DeepActiveRuntimeLinear e
  | Term.loc _ => True

def DeepActiveRuntimeLinearClauses :
    List (EffectLabel × String × String × Term) → Prop
  | [] => True
  | (_, _, _, hb) :: rest =>
      DeepActiveRuntimeLinear hb ∧
      DeepActiveRuntimeLinearClauses rest

end

theorem mem_activeLocRefs_subset
    {e : Term} {ell : Loc}
    (h : ell ∈ activeLocRefs e) :
    ell ∈ locRefs e := by
  match e with
  | Term.var x =>
      simp [activeLocRefs] at h
  | Term.abs x t body =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := body) h
  | Term.app e1 e2 =>
      simp [activeLocRefs, locRefs] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefs_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefs_subset (e := e2) h)
  | Term.letBind x e1 e2 =>
      simp [activeLocRefs, locRefs] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefs_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefs_subset (e := e2) h)
  | Term.copy e =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.letpair x y e1 e2 =>
      simp [activeLocRefs, locRefs] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefs_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefs_subset (e := e2) h)
  | Term.pair e1 e2 =>
      simp [activeLocRefs, locRefs] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefs_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefs_subset (e := e2) h)
  | Term.fst _ e =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.snd _ e =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.unit =>
      simp [activeLocRefs] at h
  | Term.const c ds =>
      simp [activeLocRefs] at h
  | Term.add e1 e2 =>
      simp [activeLocRefs, locRefs] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefs_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefs_subset (e := e2) h)
  | Term.mul e1 e2 =>
      simp [activeLocRefs, locRefs] at h ⊢
      rcases h with h | h
      · exact Or.inl (mem_activeLocRefs_subset (e := e1) h)
      · exact Or.inr (mem_activeLocRefs_subset (e := e2) h)
  | Term.sum e d =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.expand e d =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.uniformLike e lo hi =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.grad x t tOut body =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := body) h
  | Term.vmap x t d body =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := body) h
  | Term.handle epsH body clauses =>
      have hbody : ell ∈ activeLocRefs body := by
        simpa [activeLocRefs, activeLocRefsClauses, activeLocRefsClauses_eq_nil] using h
      have hbody' : ell ∈ locRefs body := mem_activeLocRefs_subset (e := body) hbody
      simpa [locRefs] using Or.inl hbody'
  | Term.perform op e =>
      simpa [activeLocRefs, locRefs] using
        mem_activeLocRefs_subset (e := e) h
  | Term.loc ell' =>
      simpa [activeLocRefs, locRefs] using h

theorem runtimeLinear_active
    {e : Term}
    (h : RuntimeLinear e) :
    ActiveRuntimeLinear e := by
  match e with
  | Term.var x =>
      simp [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] at h ⊢
  | Term.abs x t body =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := body) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.app e1 e2 =>
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinear_active (e := e1) h1
      have hAct2 := runtimeLinear_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinear] using hAct1,
        by simpa [ActiveRuntimeLinear] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefs_subset hmem1) ell
        (mem_activeLocRefs_subset hmem2) rfl
  | Term.letBind x e1 e2 =>
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinear_active (e := e1) h1
      have hAct2 := runtimeLinear_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinear] using hAct1,
        by simpa [ActiveRuntimeLinear] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefs_subset hmem1) ell
        (mem_activeLocRefs_subset hmem2) rfl
  | Term.copy e =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.letpair x y e1 e2 =>
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinear_active (e := e1) h1
      have hAct2 := runtimeLinear_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinear] using hAct1,
        by simpa [ActiveRuntimeLinear] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefs_subset hmem1) ell
        (mem_activeLocRefs_subset hmem2) rfl
  | Term.pair e1 e2 =>
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinear_active (e := e1) h1
      have hAct2 := runtimeLinear_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinear] using hAct1,
        by simpa [ActiveRuntimeLinear] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefs_subset hmem1) ell
        (mem_activeLocRefs_subset hmem2) rfl
  | Term.fst _ e =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.snd _ e =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.unit =>
      simp [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] at h ⊢
  | Term.const c ds =>
      simp [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] at h ⊢
  | Term.add e1 e2 =>
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinear_active (e := e1) h1
      have hAct2 := runtimeLinear_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinear] using hAct1,
        by simpa [ActiveRuntimeLinear] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefs_subset hmem1) ell
        (mem_activeLocRefs_subset hmem2) rfl
  | Term.mul e1 e2 =>
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, hsep⟩
      have hAct1 := runtimeLinear_active (e := e1) h1
      have hAct2 := runtimeLinear_active (e := e2) h2
      refine List.nodup_append.mpr ?_
      refine ⟨by simpa [ActiveRuntimeLinear] using hAct1,
        by simpa [ActiveRuntimeLinear] using hAct2, ?_⟩
      intro ell hmem1 ell' hmem2 hEq
      subst ell'
      exact hsep ell (mem_activeLocRefs_subset hmem1) ell
        (mem_activeLocRefs_subset hmem2) rfl
  | Term.sum e d =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.expand e d =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.uniformLike e lo hi =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.grad x t tOut body =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := body) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.vmap x t d body =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := body) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.handle epsH body clauses =>
      have hsplit : (locRefs body ++ locRefsClauses clauses).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨hBody, _hClauses, _hsep⟩
      have hActBody := runtimeLinear_active (e := body) hBody
      simpa [ActiveRuntimeLinear, activeLocRefs, activeLocRefsClauses,
        activeLocRefsClauses_eq_nil] using hActBody
  | Term.perform op e =>
      simpa [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] using
        runtimeLinear_active (e := e) (by simpa [RuntimeLinear, locRefs] using h)
  | Term.loc ell =>
      simp [RuntimeLinear, ActiveRuntimeLinear, locRefs, activeLocRefs] at h ⊢

mutual

theorem runtimeLinear_deepActive
    : ∀ {e : Term}, RuntimeLinear e -> DeepActiveRuntimeLinear e
  | Term.var _, _ => by
      simp [DeepActiveRuntimeLinear]
  | Term.abs x t body, h => by
      refine ⟨runtimeLinear_active h, ?_⟩
      exact runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)
  | Term.app e1 e2, h => by
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive h1,
        runtimeLinear_deepActive h2⟩
  | Term.letBind x e1 e2, h => by
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive h1,
        runtimeLinear_deepActive h2⟩
  | Term.copy e, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.letpair x y e1 e2, h => by
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive h1,
        runtimeLinear_deepActive h2⟩
  | Term.pair e1 e2, h => by
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive h1,
        runtimeLinear_deepActive h2⟩
  | Term.fst _ e, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.snd _ e, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.unit, _ => by
      simp [DeepActiveRuntimeLinear]
  | Term.const _ _, _ => by
      simp [DeepActiveRuntimeLinear]
  | Term.add e1 e2, h => by
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive h1,
        runtimeLinear_deepActive h2⟩
  | Term.mul e1 e2, h => by
      have hsplit : (locRefs e1 ++ locRefs e2).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨h1, h2, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive h1,
        runtimeLinear_deepActive h2⟩
  | Term.sum e _, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.expand e _, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.uniformLike e _ _, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.grad _ _ _ body, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.vmap _ _ _ body, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.handle _ body clauses, h => by
      have hsplit : (locRefs body ++ locRefsClauses clauses).Nodup := by
        simpa [RuntimeLinear, locRefs] using h
      rcases List.nodup_append.mp hsplit with ⟨hBody, hClauses, _hsep⟩
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive hBody,
        runtimeLinearClauses_deepActive hClauses⟩
  | Term.perform _ e, h => by
      exact ⟨runtimeLinear_active h,
        runtimeLinear_deepActive (by simpa [RuntimeLinear, locRefs] using h)⟩
  | Term.loc _, _ => by
      simp [DeepActiveRuntimeLinear]
termination_by
  e _ => sizeOf e

theorem runtimeLinearClauses_deepActive
    : ∀ {clauses : List (EffectLabel × String × String × Term)},
      (locRefsClauses clauses).Nodup ->
      DeepActiveRuntimeLinearClauses clauses
  | [], _ => by
      simp [DeepActiveRuntimeLinearClauses]
  | (_, _, _, hb) :: rest, h => by
      have hsplit : (locRefs hb ++ locRefsClauses rest).Nodup := by
        simpa [locRefsClauses] using h
      rcases List.nodup_append.mp hsplit with ⟨hHead, hTail, _hsep⟩
      exact ⟨runtimeLinear_deepActive hHead, runtimeLinearClauses_deepActive hTail⟩
termination_by
  clauses _ => sizeOf clauses

end

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
  | Term.fst _ e => boundVars e
  | Term.snd _ e => boundVars e
  | Term.unit => []
  | Term.const _ _ => []
  | Term.add e1 e2 => boundVars e1 ++ boundVars e2
  | Term.mul e1 e2 => boundVars e1 ++ boundVars e2
  | Term.sum e _ => boundVars e
  | Term.expand e _ => boundVars e
  | Term.uniformLike e _ _ => boundVars e
  | Term.grad x _ _ body => x :: boundVars body
  | Term.vmap x _ _ body => x :: boundVars body
  | Term.handle _ body clauses => boundVars body ++ boundVarsClauses clauses
  | Term.perform _ e => boundVars e
  | Term.loc _ => []

def boundVarsClauses :
    List (EffectLabel × String × String × Term) → List String
  | [] => []
  | (_, x, k, hb) :: rest => x :: k :: boundVars hb ++ boundVarsClauses rest

end

mutual

/-- All free and bound source names are hash-free. This keeps the
    generated `freshName ...` namespace disjoint from user/source names
    whenever the transform replays forward source structure. -/
def NoHashTerm : Term → Prop
  | Term.var x => NoHash x
  | Term.abs x _ body => NoHash x ∧ NoHashTerm body
  | Term.app e1 e2 => NoHashTerm e1 ∧ NoHashTerm e2
  | Term.letBind x e1 e2 => NoHash x ∧ NoHashTerm e1 ∧ NoHashTerm e2
  | Term.copy e => NoHashTerm e
  | Term.letpair x y e1 e2 => NoHash x ∧ NoHash y ∧ NoHashTerm e1 ∧ NoHashTerm e2
  | Term.pair e1 e2 => NoHashTerm e1 ∧ NoHashTerm e2
  | Term.fst _ e => NoHashTerm e
  | Term.snd _ e => NoHashTerm e
  | Term.unit => True
  | Term.const _ _ => True
  | Term.add e1 e2 => NoHashTerm e1 ∧ NoHashTerm e2
  | Term.mul e1 e2 => NoHashTerm e1 ∧ NoHashTerm e2
  | Term.sum e _ => NoHashTerm e
  | Term.expand e _ => NoHashTerm e
  | Term.uniformLike e _ _ => NoHashTerm e
  | Term.grad x _ _ body => NoHash x ∧ NoHashTerm body
  | Term.vmap x _ _ body => NoHash x ∧ NoHashTerm body
  | Term.handle _ body clauses => NoHashTerm body ∧ NoHashClauses clauses
  | Term.perform _ e => NoHashTerm e
  | Term.loc _ => True

def NoHashClauses :
    List (EffectLabel × String × String × Term) → Prop
  | [] => True
  | (_, x, k, hb) :: rest => NoHash x ∧ NoHash k ∧ NoHashTerm hb ∧ NoHashClauses rest

end

/-- Freshness of `y` with respect to term `e`: `y` is neither free nor
    bound anywhere inside `e`. Sufficient precondition for the
    position-indexed weakening lemma. -/
def freshInTerm (y : String) (e : Term) : Prop :=
  y ∉ freeVars e ∧ y ∉ boundVars e

theorem freshInTerm_of_closed_not_bound
    {y : String} {e : Term}
    (hclosed : Closed e)
    (hbound : y ∉ boundVars e) :
    freshInTerm y e := by
  unfold freshInTerm
  refine ⟨?_, hbound⟩
  unfold Closed at hclosed
  rw [hclosed]
  simp

mutual

theorem freshName_freshInTerm_of_noHashTerm
    {base : String} {n : Nat} :
    ∀ {e : Term}, NoHashTerm e → freshInTerm (freshName base n) e
  | Term.var x, hx => by
      refine ⟨?_, by simp [boundVars]⟩
      intro hmem
      simp [freeVars] at hmem
      exact freshName_ne_of_noHash_name base x n hx hmem
  | Term.abs x _ body, hNoHash => by
      rcases hNoHash with ⟨hx, hbody⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hbody with ⟨hfBody, hbBody⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_filter] at hmem
        exact hfBody hmem.1
      · intro hmem
        simp [boundVars] at hmem
        rcases hmem with hmem | hmem
        · exact freshName_ne_of_noHash_name base x n hx hmem
        · exact hbBody hmem
  | Term.app e1 e2, hNoHash => by
      rcases hNoHash with ⟨h1, h2⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h1 with ⟨hf1, hb1⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h2 with ⟨hf2, hb2⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        exact hmem.elim hf1 hf2
      · intro hmem
        rw [boundVars, List.mem_append] at hmem
        exact hmem.elim hb1 hb2
  | Term.letBind x e1 e2, hNoHash => by
      rcases hNoHash with ⟨hx, h1, h2⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h1 with ⟨hf1, hb1⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h2 with ⟨hf2, hb2⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        rcases hmem with hmem | hmem
        · exact hf1 hmem
        · rw [List.mem_filter] at hmem
          exact hf2 hmem.1
      · intro hmem
        simp [boundVars, List.mem_append] at hmem
        rcases hmem with hmem | hmem
        · exact freshName_ne_of_noHash_name base x n hx hmem
        · exact hmem.elim hb1 hb2
  | Term.copy e, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.letpair x y e1 e2, hNoHash => by
      rcases hNoHash with ⟨hx, hy, h1, h2⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h1 with ⟨hf1, hb1⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h2 with ⟨hf2, hb2⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        rcases hmem with hmem | hmem
        · exact hf1 hmem
        · rw [List.mem_filter] at hmem
          exact hf2 hmem.1
      · intro hmem
        simp [boundVars, List.mem_append] at hmem
        rcases hmem with hmem | hmem
        · exact freshName_ne_of_noHash_name base x n hx hmem
        · rcases hmem with hmem | hmem
          · exact freshName_ne_of_noHash_name base y n hy hmem
          · exact hmem.elim hb1 hb2
  | Term.pair e1 e2, hNoHash => by
      rcases hNoHash with ⟨h1, h2⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h1 with ⟨hf1, hb1⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h2 with ⟨hf2, hb2⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        exact hmem.elim hf1 hf2
      · intro hmem
        rw [boundVars, List.mem_append] at hmem
        exact hmem.elim hb1 hb2
  | Term.fst _ e, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.snd _ e, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.unit, _ => by
      simp [freshInTerm, freeVars, boundVars]
  | Term.const _ _, _ => by
      simp [freshInTerm, freeVars, boundVars]
  | Term.add e1 e2, hNoHash => by
      rcases hNoHash with ⟨h1, h2⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h1 with ⟨hf1, hb1⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h2 with ⟨hf2, hb2⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        exact hmem.elim hf1 hf2
      · intro hmem
        rw [boundVars, List.mem_append] at hmem
        exact hmem.elim hb1 hb2
  | Term.mul e1 e2, hNoHash => by
      rcases hNoHash with ⟨h1, h2⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h1 with ⟨hf1, hb1⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) h2 with ⟨hf2, hb2⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        exact hmem.elim hf1 hf2
      · intro hmem
        rw [boundVars, List.mem_append] at hmem
        exact hmem.elim hb1 hb2
  | Term.sum e _, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.expand e _, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.uniformLike e _ _, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.grad x _ _ body, hNoHash => by
      rcases hNoHash with ⟨hx, hbody⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hbody with ⟨hfBody, hbBody⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_filter] at hmem
        exact hfBody hmem.1
      · intro hmem
        simp [boundVars] at hmem
        rcases hmem with hmem | hmem
        · exact freshName_ne_of_noHash_name base x n hx hmem
        · exact hbBody hmem
  | Term.vmap x _ _ body, hNoHash => by
      rcases hNoHash with ⟨hx, hbody⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hbody with ⟨hfBody, hbBody⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_filter] at hmem
        exact hfBody hmem.1
      · intro hmem
        simp [boundVars] at hmem
        rcases hmem with hmem | hmem
        · exact freshName_ne_of_noHash_name base x n hx hmem
        · exact hbBody hmem
  | Term.handle _ body clauses, hNoHash => by
      rcases hNoHash with ⟨hbody, hclauses⟩
      rcases freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hbody with ⟨hfBody, hbBody⟩
      refine ⟨?_, ?_⟩
      · intro hmem
        rw [freeVars, List.mem_append] at hmem
        exact hmem.elim hfBody
          (freshName_not_mem_freeVarsClauses_of_noHashClauses (base := base) (n := n) hclauses)
      · intro hmem
        rw [boundVars, List.mem_append] at hmem
        exact hmem.elim hbBody
          (freshName_not_mem_boundVarsClauses_of_noHashClauses (base := base) (n := n) hclauses)
  | Term.perform _ e, hNoHash => by
      have hInner : NoHashTerm e := by
        simpa [NoHashTerm] using hNoHash
      simpa [freshInTerm, freeVars, boundVars] using
        (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hInner)
  | Term.loc _, _ => by
      simp [freshInTerm, freeVars, boundVars]

termination_by
  e _ => sizeOf e

theorem freshName_not_mem_freeVarsClauses_of_noHashClauses
    {base : String} {n : Nat} :
    ∀ {clauses : List (EffectLabel × String × String × Term)},
      NoHashClauses clauses → freshName base n ∉ freeVarsClauses clauses
  | [], _ => by
      simp [freeVarsClauses]
  | (_, x, k, hb) :: rest, hNoHash => by
      rcases hNoHash with ⟨hx, hk, hhb, hrest⟩
      intro hmem
      rw [freeVarsClauses, List.mem_append] at hmem
      rcases hmem with hmem | hmem
      · rw [List.mem_filter] at hmem
        exact (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hhb).1 hmem.1
      · exact freshName_not_mem_freeVarsClauses_of_noHashClauses (base := base) (n := n) hrest hmem

termination_by
  clauses _ => sizeOf clauses

theorem freshName_not_mem_boundVarsClauses_of_noHashClauses
    {base : String} {n : Nat} :
    ∀ {clauses : List (EffectLabel × String × String × Term)},
      NoHashClauses clauses → freshName base n ∉ boundVarsClauses clauses
  | [], _ => by
      simp [boundVarsClauses]
  | (_, x, k, hb) :: rest, hNoHash => by
      rcases hNoHash with ⟨hx, hk, hhb, hrest⟩
      intro hmem
      simp [boundVarsClauses, List.mem_append] at hmem
      rcases hmem with hmem | hmem
      · exact freshName_ne_of_noHash_name base x n hx hmem
      · rcases hmem with hmem | hmem
        · exact freshName_ne_of_noHash_name base k n hk hmem
        · rcases hmem with hmem | hmem
          · exact (freshName_freshInTerm_of_noHashTerm (base := base) (n := n) hhb).2 hmem
          · exact freshName_not_mem_boundVarsClauses_of_noHashClauses (base := base) (n := n) hrest hmem

termination_by
  clauses _ => sizeOf clauses

end

/-- Global binder-distinctness for named terms. This is the scoping
    side condition used by the TranslationDB bridge to align lexical
    named substitution with positional DB substitution under the current
    non-lexical `HasType.var` rule. -/
def WellScoped (e : Term) : Prop :=
  (boundVars e).Nodup

@[simp] theorem wellScoped_var (x : String) : WellScoped (Term.var x) := by
  simp [WellScoped, boundVars]

@[simp] theorem wellScoped_unit : WellScoped Term.unit := by
  simp [WellScoped, boundVars]

@[simp] theorem wellScoped_loc (ell : Loc) : WellScoped (Term.loc ell) := by
  simp [WellScoped, boundVars]

theorem wellScoped_abs_iff (x : String) (t : Typ) (body : Term) :
    WellScoped (Term.abs x t body) ↔ x ∉ boundVars body ∧ WellScoped body := by
  simp [WellScoped, boundVars]

theorem wellScoped_grad_iff (x : String) (t tOut : Typ) (body : Term) :
    WellScoped (Term.grad x t tOut body) ↔
      x ∉ boundVars body ∧ WellScoped body := by
  simp [WellScoped, boundVars]

theorem wellScoped_vmap_iff (x : String) (t : Typ) (d : Dim) (body : Term) :
    WellScoped (Term.vmap x t d body) ↔
      x ∉ boundVars body ∧ WellScoped body := by
  simp [WellScoped, boundVars]

theorem wellScoped_abs_body
    {x : String} {t : Typ} {body : Term}
    (h : WellScoped (Term.abs x t body)) :
    x ∉ boundVars body ∧ WellScoped body := by
  exact (wellScoped_abs_iff x t body).mp h

theorem wellScoped_grad_body
    {x : String} {t tOut : Typ} {body : Term}
    (h : WellScoped (Term.grad x t tOut body)) :
    x ∉ boundVars body ∧ WellScoped body := by
  exact (wellScoped_grad_iff x t tOut body).mp h

theorem wellScoped_vmap_body
    {x : String} {t : Typ} {d : Dim} {body : Term}
    (h : WellScoped (Term.vmap x t d body)) :
    x ∉ boundVars body ∧ WellScoped body := by
  exact (wellScoped_vmap_iff x t d body).mp h

private theorem nodup_append_left_not_mem
    {α : Type} [DecidableEq α]
    {xs ys : List α}
    (h : (xs ++ ys).Nodup) :
    ∀ z, z ∈ xs → z ∉ ys := by
  simp [List.nodup_append] at h
  intro z hz hy
  exact h.2.2 z hz z hy rfl

private theorem nodup_append_right_not_mem
    {α : Type} [DecidableEq α]
    {xs ys : List α}
    (h : (xs ++ ys).Nodup) :
    ∀ z, z ∈ ys → z ∉ xs := by
  simp [List.nodup_append] at h
  intro z hz hx
  exact h.2.2 z hx z hz rfl

theorem wellScoped_app_parts
    {e1 e2 : Term}
    (h : WellScoped (Term.app e1 e2)) :
    WellScoped e1 ∧ WellScoped e2 ∧
      (∀ z, z ∈ boundVars e1 → z ∉ boundVars e2) ∧
      (∀ z, z ∈ boundVars e2 → z ∉ boundVars e1) := by
  have hNodup : (boundVars e1 ++ boundVars e2).Nodup := by
    simpa only [WellScoped, boundVars] using h
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact ⟨h.1, h.2.1, nodup_append_left_not_mem hNodup, nodup_append_right_not_mem hNodup⟩

theorem wellScoped_letpair_names_ne
    {x y : String} {e1 e2 : Term}
    (h : WellScoped (Term.letpair x y e1 e2)) :
    x ≠ y := by
  simp [WellScoped, boundVars] at h
  exact h.1.1

theorem wellScoped_handle_body
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    (h : WellScoped (Term.handle epsH body clauses)) :
    WellScoped body := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact h.1

theorem wellScoped_letBind_body
    {x : String} {e1 e2 : Term}
    (h : WellScoped (Term.letBind x e1 e2)) :
    WellScoped e1 ∧ x ∉ boundVars e2 ∧ WellScoped e2 := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact ⟨h.2.1, h.1.2, h.2.2.1⟩

theorem wellScoped_letBind_parts
    {x : String} {e1 e2 : Term}
    (h : WellScoped (Term.letBind x e1 e2)) :
    WellScoped e1 ∧ x ∉ boundVars e1 ∧ x ∉ boundVars e2 ∧ WellScoped e2 ∧
      (∀ z, z ∈ boundVars e1 → z ∉ boundVars e2) ∧
      (∀ z, z ∈ boundVars e2 → z ∉ boundVars e1) := by
  have hNodup : (x :: (boundVars e1 ++ boundVars e2)).Nodup := by
    simpa only [WellScoped, boundVars] using h
  have hxRest : x ∉ boundVars e1 ++ boundVars e2 := (List.nodup_cons.mp hNodup).1
  have hRest : (boundVars e1 ++ boundVars e2).Nodup := (List.nodup_cons.mp hNodup).2
  simp [WellScoped, boundVars, List.nodup_append] at h
  refine ⟨h.2.1, ?_, h.1.2, h.2.2.1,
    nodup_append_left_not_mem hRest, nodup_append_right_not_mem hRest⟩
  intro hx
  exact hxRest (by simp [List.mem_append, hx])

theorem wellScoped_pair_left
    {e1 e2 : Term}
    (h : WellScoped (Term.pair e1 e2)) :
    WellScoped e1 := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact h.1

theorem wellScoped_pair_right
    {e1 e2 : Term}
    (h : WellScoped (Term.pair e1 e2)) :
    WellScoped e2 := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact h.2.1

theorem wellScoped_pair_parts
    {e1 e2 : Term}
    (h : WellScoped (Term.pair e1 e2)) :
    WellScoped e1 ∧ WellScoped e2 ∧
      (∀ z, z ∈ boundVars e1 → z ∉ boundVars e2) ∧
      (∀ z, z ∈ boundVars e2 → z ∉ boundVars e1) := by
  have hNodup : (boundVars e1 ++ boundVars e2).Nodup := by
    simpa only [WellScoped, boundVars] using h
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact ⟨h.1, h.2.1, nodup_append_left_not_mem hNodup, nodup_append_right_not_mem hNodup⟩

theorem wellScoped_perform_body
    {op : EffectLabel} {e : Term}
    (h : WellScoped (Term.perform op e)) :
    WellScoped e := by
  simpa [WellScoped, boundVars] using h

theorem wellScoped_letpair_body
    {x y : String} {e1 e2 : Term}
    (h : WellScoped (Term.letpair x y e1 e2)) :
    WellScoped e1 ∧ x ≠ y ∧ x ∉ boundVars e2 ∧ y ∉ boundVars e2 ∧ WellScoped e2 := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact ⟨h.2.2.1, h.1.1, h.1.2.2, h.2.1.2, h.2.2.2.1⟩

theorem wellScoped_letpair_parts
    {x y : String} {e1 e2 : Term}
    (h : WellScoped (Term.letpair x y e1 e2)) :
    WellScoped e1 ∧ x ≠ y ∧ x ∉ boundVars e1 ∧ x ∉ boundVars e2 ∧
      y ∉ boundVars e1 ∧ y ∉ boundVars e2 ∧ WellScoped e2 ∧
      (∀ z, z ∈ boundVars e1 → z ∉ boundVars e2) ∧
      (∀ z, z ∈ boundVars e2 → z ∉ boundVars e1) := by
  have hNodup : (x :: y :: (boundVars e1 ++ boundVars e2)).Nodup := by
    simpa only [WellScoped, boundVars] using h
  have hxRest : x ∉ y :: (boundVars e1 ++ boundVars e2) := (List.nodup_cons.mp hNodup).1
  have hTail : (y :: (boundVars e1 ++ boundVars e2)).Nodup := (List.nodup_cons.mp hNodup).2
  have hyRest : y ∉ boundVars e1 ++ boundVars e2 := (List.nodup_cons.mp hTail).1
  have hRest : (boundVars e1 ++ boundVars e2).Nodup := (List.nodup_cons.mp hTail).2
  simp [WellScoped, boundVars, List.nodup_append] at h
  refine ⟨h.2.2.1, h.1.1, ?_, h.1.2.2, ?_, h.2.1.2, h.2.2.2.1,
    nodup_append_left_not_mem hRest, nodup_append_right_not_mem hRest⟩
  · intro hx
    have hx' : x ∈ y :: (boundVars e1 ++ boundVars e2) := by
      simp [List.mem_append, hx]
    exact hxRest hx'
  · intro hy
    have hy' : y ∈ boundVars e1 ++ boundVars e2 := by
      simp [List.mem_append, hy]
    exact hyRest hy'

theorem boundVarsClauses_mem_scoped
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hnd : (boundVarsClauses clauses).Nodup)
    (hmem : (op, x, k, hb) ∈ clauses) :
    x ≠ k ∧ x ∉ boundVars hb ∧ k ∉ boundVars hb ∧ WellScoped hb := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      cases cl with
      | mk op' tail =>
          cases tail with
          | mk x' tail =>
              cases tail with
              | mk k' hb' =>
                  simp [boundVarsClauses, List.nodup_append] at hnd
                  rcases List.mem_cons.mp hmem with hhd | htl
                  · cases hhd
                    exact ⟨hnd.1.1, hnd.1.2.1, hnd.2.1.1, hnd.2.2.1⟩
                  · have hrest : (boundVarsClauses rest).Nodup := by
                      exact hnd.2.2.2.1
                    exact ih hrest htl

private theorem mem_boundVarsClauses_arg
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses) :
    x ∈ boundVarsClauses clauses := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [boundVarsClauses]
      · simp [boundVarsClauses, ih htl]

private theorem mem_boundVarsClauses_cont
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses) :
    k ∈ boundVarsClauses clauses := by
  induction clauses with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [boundVarsClauses]
      · simp [boundVarsClauses, ih htl]

private theorem mem_boundVarsClauses_body
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k z : String} {hb : Term}
    (hmem : (op, x, k, hb) ∈ clauses)
    (hz : z ∈ boundVars hb) :
    z ∈ boundVarsClauses clauses := by
  induction clauses generalizing op x k hb z with
  | nil =>
      cases hmem
  | cons cl rest ih =>
      rcases List.mem_cons.mp hmem with hhd | htl
      · cases hhd
        simp [boundVarsClauses, hz]
      · simp [boundVarsClauses, ih htl hz]

theorem wellScoped_handle_clause
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (h : WellScoped (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    x ≠ k ∧ x ∉ boundVars hb ∧ k ∉ boundVars hb ∧ WellScoped hb := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  have hcls : (boundVarsClauses clauses).Nodup := h.2.1
  exact boundVarsClauses_mem_scoped hcls hmem

theorem wellScoped_handle_clause_not_in_body
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (h : WellScoped (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    x ∉ boundVars body ∧ k ∉ boundVars body := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  refine ⟨?_, ?_⟩
  · intro hx
    exact h.2.2 x hx _ (mem_boundVarsClauses_arg hmem) rfl
  · intro hk
    exact h.2.2 k hk _ (mem_boundVarsClauses_cont hmem) rfl

theorem wellScoped_handle_clause_body_disjoint
    {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    (h : WellScoped (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    ∀ z, z ∈ boundVars hb → z ∉ boundVars body := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  intro z hz hbz
  exact h.2.2 z hbz z (mem_boundVarsClauses_body hmem hz) rfl

theorem wellScoped_handle_rest
    {epsH : EffectRow} {body : Term}
    {op : EffectLabel} {x k : String} {hb : Term}
    {rest : List (EffectLabel × String × String × Term)}
    (h : WellScoped (Term.handle epsH body ((op, x, k, hb) :: rest))) :
    WellScoped (Term.handle epsH body rest) := by
  unfold WellScoped at h ⊢
  have hsubRest :
      List.Sublist (boundVarsClauses rest)
        (x :: k :: (boundVars hb ++ boundVarsClauses rest)) := by
    simpa [List.append_assoc] using
      (List.sublist_append_right
        (x :: k :: boundVars hb) (boundVarsClauses rest))
  have hsub :
      List.Sublist (boundVars body ++ boundVarsClauses rest)
        (boundVars body ++ boundVarsClauses ((op, x, k, hb) :: rest)) := by
    simpa [boundVars, boundVarsClauses, List.append_assoc] using
      (List.append_sublist_append_left (boundVars body)).2 hsubRest
  exact hsub.nodup h

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

/-- Linear typing context. A list of named slots; live slots hold
    `some t`, consumed (tombstoned) slots hold `none`. Variable
    consumption marks a slot as `none` rather than removing it, keeping
    the context length invariant across a derivation. This aligns with
    `LinearCtxDB := List (Option Typ)`. -/
abbrev LinearCtx := List (String × Option Typ)

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

/-- Rightmost tombstone-aware lookup in a linear context. This matches
    the operational interpretation of the tail as the innermost live
    binder: later entries shadow earlier ones, including when the later
    entry is a tombstone. -/
def lookupLinearCtx : LinearCtx → String → Option Typ
  | [], _ => none
  | (y, slot) :: rest, x =>
      match lookupLinearCtx rest x with
      | some t => some t
      | none => if x = y then slot else none

theorem lookupLinearCtx_append (pref tail : LinearCtx) (x : String) :
    lookupLinearCtx (pref ++ tail) x =
      match lookupLinearCtx tail x with
      | some t => some t
      | none => lookupLinearCtx pref x := by
  induction pref with
  | nil =>
      cases htail : lookupLinearCtx tail x with
      | none =>
          simp [lookupLinearCtx, htail]
      | some t =>
          simp [lookupLinearCtx, htail]
  | cons hd tl ih =>
      cases hd with
      | mk y slot =>
          cases htail : lookupLinearCtx tail x with
          | none =>
              simp [lookupLinearCtx, ih, htail]
          | some t =>
              simp [lookupLinearCtx, ih, htail]

theorem lookupLinearCtx_append_left_of_some
    {pref tail : LinearCtx} {x : String} {t : Typ}
    (h : lookupLinearCtx tail x = some t) :
    lookupLinearCtx (pref ++ tail) x = some t := by
  rw [lookupLinearCtx_append]
  simp [h]

theorem lookupLinearCtx_some_of_mem_live
    {Γ : LinearCtx} {x : String} {t : Typ}
    (hMem : (x, some t) ∈ Γ) :
    ∃ t', lookupLinearCtx Γ x = some t' := by
  induction Γ with
  | nil =>
      cases hMem
  | cons hd tl ih =>
      cases hd with
      | mk y slot =>
          simp at hMem
          rcases hMem with hHead | hTail
          · rcases hHead with ⟨rfl, rfl⟩
            cases htl : lookupLinearCtx tl x with
            | none =>
                exact ⟨t, by simp [lookupLinearCtx, htl]⟩
            | some t' =>
                exact ⟨t', by simp [lookupLinearCtx, htl]⟩
          · rcases ih hTail with ⟨t', hLook⟩
            exact ⟨t', lookupLinearCtx_append_left_of_some (pref := [(y, slot)]) hLook⟩

/-! ## Adjoint support fragment

These predicates are syntax-level side conditions for the current typed
adjoint transform. They live here, rather than in `AdjointTransform`,
so both `Typing` and the transform/theorem files can depend on them
without creating an import cycle. -/

mutual

/-- Syntax-only domain predicate for the currently supported adjoint
    transform fragment. This is the first-order fragment the typed
    transform actually implements: product structure, primitive tensor
    operators, handlers, and `perform` of differentiation-compatible
    effects are allowed; higher-order / staged constructs whose current
    transform equations are still placeholders are excluded. -/
def AdjointSupported : Term → Prop
  | Term.var _ => True
  | Term.const _ _ => True
  | Term.unit => True
  | Term.loc _ => True
  | Term.add e1 e2 => AdjointSupported e1 ∧ AdjointSupported e2
  | Term.mul e1 e2 => AdjointSupported e1 ∧ AdjointSupported e2
  | Term.sum e _ => AdjointSupported e
  | Term.expand e _ => AdjointSupported e
  | Term.copy e => AdjointSupported e
  | Term.letBind _ e1 e2 => AdjointSupported e1 ∧ AdjointSupported e2
  | Term.letpair _ _ e1 e2 => AdjointSupported e1 ∧ AdjointSupported e2
  | Term.pair e1 e2 => AdjointSupported e1 ∧ AdjointSupported e2
  | Term.fst _ e => AdjointSupported e
  | Term.snd _ e => AdjointSupported e
  | Term.handle _ body clauses =>
      AdjointSupported body ∧ AdjointSupportedClauses clauses
  | Term.perform op e => op ∈ DiffCompat ∧ AdjointSupported e
  | Term.abs _ _ _ => False
  | Term.app _ _ => False
  | Term.uniformLike _ _ _ => False
  | Term.grad _ _ _ _ => False
  | Term.vmap _ _ _ _ => False

/-- Clause-list companion to `AdjointSupported`. -/
def AdjointSupportedClauses :
    List (EffectLabel × String × String × Term) → Prop
  | [] => True
  | (_, _, _, hb) :: rest =>
      AdjointSupported hb ∧ AdjointSupportedClauses rest

end

/-- The fragment of source/result types whose cotangent seeds are
    currently represented explicitly by the typed adjoint transform. -/
def AdjointTypeSupported : Typ → Prop
  | Typ.tensor _ => True
  | Typ.pair t1 t2 => AdjointTypeSupported t1 ∧ AdjointTypeSupported t2
  | Typ.unit => True
  | Typ.arrow _ _ _ => False
  | Typ.tyVar _ => False

/-- Every live entry in the linear context lies in the currently
    supported first-order adjoint fragment. Tombstones are ignored. -/
def AdjointCtxSupported : LinearCtx → Prop
  | [] => True
  | (_, none) :: rest => AdjointCtxSupported rest
  | (_, some t) :: rest => AdjointTypeSupported t ∧ AdjointCtxSupported rest

/-- Successful rightmost lookup in a supported context returns a
    supported first-order type. -/
theorem adjointCtxSupported_lookup
    {Γ : LinearCtx} {x : String} {t : Typ}
    (h : AdjointCtxSupported Γ)
    (hLook : lookupLinearCtx Γ x = some t) :
    AdjointTypeSupported t := by
  let rec go :
      ∀ (Γ : LinearCtx), AdjointCtxSupported Γ →
        ∀ {x : String} {t : Typ},
          lookupLinearCtx Γ x = some t →
          AdjointTypeSupported t
    | [], _, _, _, hLook => by
        simp [lookupLinearCtx] at hLook
    | (y, none) :: tl, h, x, t, hLook => by
        simp [AdjointCtxSupported] at h
        cases htl : lookupLinearCtx tl x with
        | none =>
            simp [lookupLinearCtx, htl] at hLook
        | some t' =>
            simp [lookupLinearCtx, htl] at hLook
            cases hLook
            exact go tl h htl
    | (y, some t0) :: tl, h, x, t, hLook => by
        simp [AdjointCtxSupported] at h
        cases htl : lookupLinearCtx tl x with
        | none =>
            simp [lookupLinearCtx, htl] at hLook
            rcases hLook with ⟨rfl, rfl⟩
            exact h.1
        | some t' =>
            simp [lookupLinearCtx, htl] at hLook
            cases hLook
            exact go tl h.2 htl
  exact go Γ h hLook

/-- A successful rightmost lookup exposes a matching live binding split
    of the context, together with the fact that the tail to its right no
    longer contains a live binding for the same name. -/
theorem lookupLinearCtx_some_split
    {Γ : LinearCtx} {x : String} {t : Typ}
    (h : lookupLinearCtx Γ x = some t) :
    ∃ pre post,
      Γ = pre ++ [(x, some t)] ++ post ∧
      lookupLinearCtx post x = none := by
  induction Γ with
  | nil =>
      simp [lookupLinearCtx] at h
  | cons hd tl ih =>
      cases hd with
      | mk y slot =>
          cases htail : lookupLinearCtx tl x with
          | none =>
              cases slot with
              | none =>
                  simp [lookupLinearCtx, htail] at h
              | some t0 =>
                  simp [lookupLinearCtx, htail] at h
                  rcases h with ⟨rfl, rfl⟩
                  exact ⟨[], tl, by simp, by simpa [lookupLinearCtx] using htail⟩
          | some t' =>
              simp [lookupLinearCtx, htail] at h
              cases h
              rcases ih htail with ⟨pre, post, hEq, hPost⟩
              exact ⟨(y, slot) :: pre, post, by simp [hEq], hPost⟩

/-- Body-local support side condition for `grad`: every outer live
    binding that the body actually mentions, when resolved by the
    rightmost binder lookup, lies in the supported first-order
    fragment. This is weaker than `AdjointCtxSupported Γ`, so it
    survives weakening by fresh unused variables while still ruling out
    substitutions that would inject unsupported values into the AD
    fragment. -/
def AdjointFreeCtxSupported (Γ : LinearCtx) (e : Term) : Prop :=
  ∀ z t,
    z ∈ freeVars e →
    lookupLinearCtx Γ z = some t →
    AdjointTypeSupported t

/-- Full context support implies the body-local support premise. -/
theorem adjointFreeCtxSupported_of_ctxSupported
    {Γ : LinearCtx} {e : Term}
    (h : AdjointCtxSupported Γ) :
    AdjointFreeCtxSupported Γ e := by
  intro z t _hz hlook
  exact adjointCtxSupported_lookup h hlook

/-- Linear-context names are pairwise distinct. This is the scoping
    well-formedness predicate the TranslationDB bridge needs in order to
    align lexical named substitution with positional DB substitution. -/
def NoDupNames (G : LinearCtx) : Prop :=
  (linearCtxDom G).Nodup

@[simp] theorem noDupNames_nil : NoDupNames ([] : LinearCtx) := by
  simp [NoDupNames, linearCtxDom]

@[simp] theorem noDupNames_singleton (x : String) (t : Typ) :
    NoDupNames ([(x, t)] : LinearCtx) := by
  simp [NoDupNames, linearCtxDom]

theorem noDupNames_pair_iff
    (x y : String) (tx ty : Typ) :
    NoDupNames ([(x, tx), (y, ty)] : LinearCtx) ↔ x ≠ y := by
  simp [NoDupNames, linearCtxDom]

/-- Named scoping invariant that rules out binder/context shadowing and
    keeps binder names globally distinct inside the term. This is the
    honest boundary needed to align lexical named substitution with the
    positional DB metatheory. -/
def LexicallyScoped (Gamma : LinearCtx) (e : Term) : Prop :=
  NoDupNames Gamma ∧
    (∀ x, x ∈ linearCtxDom Gamma → x ∉ boundVars e) ∧
    WellScoped e

theorem lexical_nil
    {e : Term}
    (hws : WellScoped e) :
    LexicallyScoped [] e := by
  refine ⟨noDupNames_nil, ?_, hws⟩
  intro x hx
  simpa [linearCtxDom] using hx

theorem lexical_singleton
    {x : String} {tx : Typ} {e : Term}
    (hx : x ∉ boundVars e)
    (hws : WellScoped e) :
    LexicallyScoped [(x, some tx)] e := by
  refine ⟨noDupNames_singleton x tx, ?_, hws⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with rfl
  exact hx

theorem lexical_pair
    {x y : String} {tx ty : Typ} {e : Term}
    (hxy : x ≠ y)
    (hx : x ∉ boundVars e)
    (hy : y ∉ boundVars e)
    (hws : WellScoped e) :
    LexicallyScoped [(x, some tx), (y, some ty)] e := by
  refine ⟨(noDupNames_pair_iff x y tx ty).2 hxy, ?_, hws⟩
  intro z hz
  simp [linearCtxDom] at hz
  rcases hz with rfl | rfl
  · exact hx
  · exact hy

theorem noDupNames_append_singleton
    {Gamma : LinearCtx} {x : String} {t : Typ}
    (h : NoDupNames Gamma)
    (hx : x ∉ linearCtxDom Gamma) :
    NoDupNames (Gamma ++ [(x, some t)]) := by
  unfold NoDupNames at h ⊢
  simp [linearCtxDom, List.nodup_append]
  refine ⟨h, ?_⟩
  intro a ta hmem heq
  subst heq
  have hdommem : a ∈ linearCtxDom Gamma := by
    unfold linearCtxDom
    exact List.mem_map.mpr ⟨(a, ta), hmem, rfl⟩
  exact hx hdommem

theorem lexical_app_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.app e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_app_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.app e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_copy_body
    {Gamma : LinearCtx} {e : Term}
    (h : LexicallyScoped Gamma (Term.copy e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_pair_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.pair e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_pair_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.pair e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_letBind_bound
    {Gamma : LinearCtx} {x : String} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letBind x e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hbody := wellScoped_letBind_body hws
  refine ⟨hnd, ?_, ?_⟩
  · intro y hy
    have hnot := hdom y hy
    simp [boundVars] at hnot
    exact hnot.2.1
  · exact hbody.1

theorem lexical_letpair_bound
    {Gamma : LinearCtx} {x y : String} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letpair x y e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hbody := wellScoped_letpair_body hws
  refine ⟨hnd, ?_, ?_⟩
  · intro z hz
    have hnot := hdom z hz
    simp [boundVars] at hnot
    exact hnot.2.2.1
  · exact hbody.1

theorem lexical_fst_body
    {Gamma : LinearCtx} {e : Term}
    {tRight : Typ}
    (h : LexicallyScoped (Gamma := Gamma) (Term.fst tRight e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_snd_body
    {Gamma : LinearCtx} {e : Term}
    {tLeft : Typ}
    (h : LexicallyScoped (Gamma := Gamma) (Term.snd tLeft e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_add_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.add e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_add_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.add e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_mul_left
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.mul e1 e2)) :
    LexicallyScoped Gamma e1 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.1
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.1

theorem lexical_mul_right
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.mul e1 e2)) :
    LexicallyScoped Gamma e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
    exact hnot.2
  · simp [WellScoped, boundVars, List.nodup_append] at hws
    exact hws.2.1

theorem lexical_sum_body
    {Gamma : LinearCtx} {e : Term} {d : Dim}
    (h : LexicallyScoped (Gamma := Gamma) (Term.sum e d)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_expand_body
    {Gamma : LinearCtx} {e : Term} {d : Dim}
    (h : LexicallyScoped (Gamma := Gamma) (Term.expand e d)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_uniformLike_body
    {Gamma : LinearCtx} {e : Term} {lo hi : Float}
    (h : LexicallyScoped (Gamma := Gamma) (Term.uniformLike e lo hi)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_perform_body
    {Gamma : LinearCtx} {op : EffectLabel} {e : Term}
    (h : LexicallyScoped (Gamma := Gamma) (Term.perform op e)) :
    LexicallyScoped Gamma e := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, ?_⟩
  · intro x hx
    simpa [boundVars] using hdom x hx
  · simpa [WellScoped, boundVars] using hws

theorem lexical_handle_body
    {Gamma : LinearCtx} {epsH : EffectRow}
    {body : Term} {clauses : List (EffectLabel × String × String × Term)}
    (h : LexicallyScoped Gamma (Term.handle epsH body clauses)) :
    LexicallyScoped Gamma body := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, wellScoped_handle_body hws⟩
  intro x hx
  have hnot := hdom x hx
  simp [boundVars, List.mem_append, not_or] at hnot
  exact hnot.1

theorem lexical_abs_body
    {Gamma : LinearCtx} {x : String} {tx t : Typ} {body : Term}
    (h : LexicallyScoped Gamma (Term.abs x t body)) :
    LexicallyScoped (Gamma ++ [(x, some tx)]) body := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_abs_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2
  · subst y
    exact hbody.1

theorem lexical_letBind_body
    {Gamma : LinearCtx} {x : String} {tx : Typ} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letBind x e1 e2)) :
    LexicallyScoped (Gamma ++ [(x, some tx)]) e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_letBind_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2.2
  · subst y
    exact hbody.2.1

theorem lexical_letpair_body
    {Gamma : LinearCtx} {x y : String} {tx ty : Typ} {e1 e2 : Term}
    (h : LexicallyScoped Gamma (Term.letpair x y e1 e2)) :
    LexicallyScoped (Gamma ++ [(x, some tx), (y, some ty)]) e2 := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hyGamma : y ∉ linearCtxDom Gamma := by
    intro hy
    have hnot := hdom y hy
    simp [boundVars] at hnot
  have hbody := wellScoped_letpair_body hws
  have hnd' : NoDupNames (Gamma ++ [(x, some tx)]) :=
    noDupNames_append_singleton hnd hxGamma
  have hyGamma' : y ∉ linearCtxDom (Gamma ++ [(x, some tx)]) := by
    intro hy
    simp [linearCtxDom] at hy
    rcases hy with hy | hy
    · exact hyGamma (by simpa [linearCtxDom] using hy)
    · exact hbody.2.1 hy.symm
  have hnd'' : NoDupNames (Gamma ++ [(x, some tx)] ++ [(y, some ty)]) :=
    noDupNames_append_singleton hnd' hyGamma'
  simpa [List.append_assoc] using
    (show LexicallyScoped (Gamma ++ [(x, some tx)] ++ [(y, some ty)]) e2 from by
      refine ⟨hnd'', ?_, hbody.2.2.2.2⟩
      intro z hz
      simp [linearCtxDom] at hz
      rcases hz with hz | hz
      · have hz' : z ∈ linearCtxDom Gamma := by
          simpa [linearCtxDom] using hz
        have hnot := hdom z hz'
        simp [boundVars] at hnot
        exact hnot.2.2.2
      · rcases hz with hz | hz
        · subst z
          exact hbody.2.2.1
        · subst z
          exact hbody.2.2.2.1)

theorem lexical_grad_body
    {Gamma : LinearCtx} {x : String} {tx tOut : Typ} {body : Term}
    (h : LexicallyScoped Gamma (Term.grad x tx tOut body)) :
    LexicallyScoped (Gamma ++ [(x, some tx)]) body := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_grad_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2
  · subst y
    exact hbody.1

theorem lexical_vmap_body
    {Gamma : LinearCtx} {x : String} {tx : Typ} {d : Dim} {body : Term}
    (h : LexicallyScoped Gamma (Term.vmap x tx d body)) :
    LexicallyScoped (Gamma ++ [(x, some tx)]) body := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars] at hnot
  have hbody := wellScoped_vmap_body hws
  refine ⟨noDupNames_append_singleton hnd hxGamma, ?_, hbody.2⟩
  intro y hy
  simp [linearCtxDom] at hy
  rcases hy with hy | hy
  · have hy' : y ∈ linearCtxDom Gamma := by
      simpa [linearCtxDom] using hy
    have hnot := hdom y hy'
    simp [boundVars] at hnot
    exact hnot.2
  · subst y
    exact hbody.1

theorem lexical_handle_clause
    {Gamma : LinearCtx} {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    {op : EffectLabel} {x k : String} {hb : Term}
    {tx tk : Typ}
    (h : LexicallyScoped Gamma (Term.handle epsH body clauses))
    (hmem : (op, x, k, hb) ∈ clauses) :
    LexicallyScoped (Gamma ++ [(x, some tx), (k, some tk)]) hb := by
  rcases h with ⟨hnd, hdom, hws⟩
  have hclause : x ≠ k ∧ x ∉ boundVars hb ∧ k ∉ boundVars hb ∧ WellScoped hb :=
    wellScoped_handle_clause hws hmem
  have hxGamma : x ∉ linearCtxDom Gamma := by
    intro hx
    have hnot := hdom x hx
    simp [boundVars, List.mem_append, not_or] at hnot
    exact hnot.2 (mem_boundVarsClauses_arg hmem)
  have hkGamma : k ∉ linearCtxDom Gamma := by
    intro hk
    have hnot := hdom k hk
    simp [boundVars, List.mem_append, not_or] at hnot
    exact hnot.2 (mem_boundVarsClauses_cont hmem)
  have hnd' : NoDupNames (Gamma ++ [(x, some tx)]) :=
    noDupNames_append_singleton hnd hxGamma
  have hkGamma' : k ∉ linearCtxDom (Gamma ++ [(x, some tx)]) := by
    intro hk
    simp [linearCtxDom] at hk
    rcases hk with hk | hk
    · exact hkGamma (by simpa [linearCtxDom] using hk)
    · exact hclause.1 hk.symm
  have hnd'' : NoDupNames (Gamma ++ [(x, some tx)] ++ [(k, some tk)]) :=
    noDupNames_append_singleton hnd' hkGamma'
  simpa [List.append_assoc] using
    (show LexicallyScoped (Gamma ++ [(x, some tx)] ++ [(k, some tk)]) hb from by
      refine ⟨hnd'', ?_, hclause.2.2.2⟩
      intro z hz
      simp [linearCtxDom] at hz
      rcases hz with hz | hz
      · have hz' : z ∈ linearCtxDom Gamma := by
          simpa [linearCtxDom] using hz
        have hnot := hdom z hz'
        simp [boundVars, List.mem_append, not_or] at hnot
        intro hzb
        exact hnot.2 (mem_boundVarsClauses_body hmem hzb)
      · rcases hz with hz | hz
        · subst z
          exact hclause.2.1
        · subst z
          exact hclause.2.2.1)

theorem lexical_handle_rest
    {Gamma : LinearCtx} {epsH : EffectRow} {body : Term}
    {op : EffectLabel} {x k : String} {hb : Term}
    {rest : List (EffectLabel × String × String × Term)}
    (h : LexicallyScoped Gamma (Term.handle epsH body ((op, x, k, hb) :: rest))) :
    LexicallyScoped Gamma (Term.handle epsH body rest) := by
  rcases h with ⟨hnd, hdom, hws⟩
  refine ⟨hnd, ?_, wellScoped_handle_rest hws⟩
  intro z hz
  have hnot := hdom z hz
  intro hzb
  have hzb' : z ∈ boundVars (Term.handle epsH body ((op, x, k, hb) :: rest)) := by
    rcases List.mem_append.mp hzb with hzBody | hzRest
    · exact List.mem_append.mpr (Or.inl hzBody)
    · refine List.mem_append.mpr ?_
      exact Or.inr (by simp [boundVarsClauses, hzRest])
  exact hnot hzb'

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
