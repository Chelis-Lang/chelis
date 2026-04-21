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

theorem wellScoped_vmap_iff (x : String) (t : Typ) (body : Term) :
    WellScoped (Term.vmap x t body) ↔
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
    {x : String} {t : Typ} {body : Term}
    (h : WellScoped (Term.vmap x t body)) :
    x ∉ boundVars body ∧ WellScoped body := by
  exact (wellScoped_vmap_iff x t body).mp h

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

theorem wellScoped_letpair_body
    {x y : String} {e1 e2 : Term}
    (h : WellScoped (Term.letpair x y e1 e2)) :
    WellScoped e1 ∧ x ≠ y ∧ x ∉ boundVars e2 ∧ y ∉ boundVars e2 ∧ WellScoped e2 := by
  simp [WellScoped, boundVars, List.nodup_append] at h
  exact ⟨h.2.2.1, h.1.1, h.1.2.2, h.2.1.2, h.2.2.2.1⟩

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
