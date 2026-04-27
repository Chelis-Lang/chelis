-- LaCaDiLE/AdjointTransform.lean — adjoint : Term → String → Term → Term.
--
-- `adjoint body x gSeed` is the T0 §4 adjoint transformation: given a
-- differentiated function's body, the parameter name `x`, and the
-- incoming output-gradient seed `gSeed`, produce a transformed term
-- that:
--   (a) inserts tape copies (`let (a, a') = copy(a)`) before primitives
--       whose adjoint rule requires operand values (mul),
--   (b) emits `perform accum(loc, contribution)` for each operand,
--   (c) returns a term whose result is the parameter gradient.
--
-- The function is **total** (no `partial def`) via explicit structural
-- recursion on every `Term` former. The handler case delegates to
-- `adjointClauses` inside a `mutual` block so Lean's structural checker
-- sees each recursive call lands on a strictly smaller sub-term.
-- Phase 2 WS2.2 will state and prove the adjoint typing lemma.
--
-- Phase 1 semantic caveats that Phase 2 T9 will tighten (these are
-- Phase 2 prerequisites, not structural bugs):
--
--   * `sum` adjoint still hardcodes extent 0 in the generated
--     `expand`; real extent must come from a shape-tape read (P8
--     is subsumed into this comment and gets closed by the T0 §4
--     structural recursion in Phase 2 when the tape mechanism ships).
--   * `var y` when `y ≠ x` routes to `perform accum` — over-approximates
--     the real rule (non-parameter free variables should not contribute,
--     but the handler's origin filter drops them so the net effect is
--     correct).
--   * `letBind y e1 e2` now recurses on the result-producing body `e2`
--     rather than the bound expression `e1`. This is still only a
--     structural Phase 1 placeholder, but it at least matches the
--     forward-pass sequencing shape from the tape design notes. Phase 2
--     T9 still needs the real forward/backward ordering.
--   * `pair`, `fst`, `snd`, and `copy` now use typed cotangent seeds:
--     products split their incoming seed structurally, and
--     projections pad the inactive component with a zero cotangent.
--     This repairs the old monomorphic tensor-seed mismatch for
--     product/projection paths.
--   * `abs`, `app`, `uniformLike`, `grad`, `vmap`, `perform` still
--     use structural placeholders. Phase 2 T9 still needs to replace
--     those with the real T0 §4 adjoint rules or make the theorem
--     surface explicit about their transform domain.
--   * `loc` and `unit` are leaves.
--   * `handle` recurses into clause bodies via `adjointClauses`.

import LaCaDiLE.Syntax

namespace LaCaDiLE

/-- Fresh variable name suffix for tape-binding generation.
    Phase 1 uses a static name; Phase 2 threads a counter via
    `freshName` below. `tapeName` is kept as a legacy helper. -/
def tapeName (base : String) : String := base ++ "_tape"

/-- Counter-based fresh name. Attaches the counter `n` to the given
    prefix so two recursion levels with different counters produce
    disjoint names. Wave-4 Track B: threading this through
    `adjointFrom` makes freshness provable in the inner linear
    context of the `add`/`mul`/`handle` cases of
    `adjoint_typed_aux`. -/
def freshName (base : String) (n : Nat) : String :=
  String.ofList (base.toList ++ '#' :: List.replicate n 'x')

-- The adjoint transformation. Phase-1-note historical summary:
--   * `mul(e1, e2)` gets full tape treatment (two copies of each operand,
--     a forward mul, and backward contributions via `copy(gSeed)`).
--   * `add(e1, e2)` gets the backward `copy(gSeed)` + two accum calls.
--   * `sum(e, i)` routes gSeed through `expand(gSeed, i, 0)` and recurses.
--   * `expand(e, i, k)` routes gSeed through `sum(gSeed, i)` and recurses.
--   * `const(_, _)` emits an accum that the handler's origin filter drops.
--   * `var y` emits an accum for gSeed regardless of whether `y = x`; the
--     handler's origin filter picks the right routing.
--   * Structural recursion covers every other term former with a vestigial
--     `perform accum` leaf. Phase 2 T9 replaces these with the proper
--     T0 §4 adjoint rules.

/-- Leaf adjoint skeleton under the current monomorphic
    `accum : unit -> unit` signature. Consume the incoming seed
    linearly, then emit a unit-valued `accum` operation. This keeps
    the seed's context threading available to `AdjointTyping` without
    pretending the current calculus can transport tensor payloads
    through `accum`. -/
def adjointLeaf (gSeed : Term) (n : Nat) : Term :=
  Term.letBind (freshName "adjA" n) gSeed
    (Term.perform EffectLabel.accum Term.unit)

/-- Internal cotangent payload shape for the current reverse-mode
    transform. Tensor cotangents stay tensors, product cotangents
    distribute over products, and currently non-differentiable
    components collapse to `unit`, matching `spec/06-transformations.md`. -/
def cotangentType : Typ → Typ
  | Typ.tensor ds       => Typ.tensor ds
  | Typ.pair t1 t2      => Typ.pair (cotangentType t1) (cotangentType t2)
  | Typ.unit            => Typ.unit
  | Typ.arrow _ _ _     => Typ.unit
  | Typ.tyVar _         => Typ.unit

/-- Zero cotangent seed used to pad inactive branches in product /
    projection adjoints. -/
def zeroCotangent : Typ → Term
  | Typ.tensor ds       => Term.const 0 ds
  | Typ.pair t1 t2      => Term.pair (zeroCotangent t1) (zeroCotangent t2)
  | Typ.unit            => Term.unit
  | Typ.arrow _ _ _     => Term.unit
  | Typ.tyVar _         => Term.unit

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

/-- Split a cotangent seed for primal type `t` into two cotangent seeds
    of type `cotangentType t`. Tensor leaves use `copy`; product
    cotangents recurse structurally; currently non-differentiable leaves
    duplicate `unit`. The continuation receives the next fresh counter
    after the split's binder names so callers can keep later adjoint
    binders disjoint from the split scaffold. -/
def splitCotangentSeedFrom
    (t : Typ) (gSeed : Term) (n : Nat)
    (k : Nat → Term → Term → Term) : Term :=
  match t with
  | Typ.tensor _ =>
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      Term.letpair gA gB (Term.copy gSeed)
        (k (n + 2) (Term.var gA) (Term.var gB))
  | Typ.pair t1 t2 =>
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      Term.letpair gA gB gSeed
        (splitCotangentSeedFrom t1 (Term.var gA) (n + 2)
          (fun n' gA1 gA2 =>
            splitCotangentSeedFrom t2 (Term.var gB) n'
              (fun n'' gB1 gB2 =>
                k n'' (Term.pair gA1 gB1) (Term.pair gA2 gB2))))
  | Typ.unit =>
      k n Term.unit Term.unit
  | Typ.arrow _ _ _ =>
      k n Term.unit Term.unit
  | Typ.tyVar _ =>
      k n Term.unit Term.unit
termination_by sizeOf t
decreasing_by
  all_goals
    simp_wf
    omega

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

mutual

/-- `addDimTerm` preserves the supported adjoint fragment. This keeps
    `vmap`'s type-preservation transport aligned with the `T-Grad`
    fragment restriction. -/
theorem adjointSupported_addDimTerm (d : Dim) :
    ∀ e, AdjointSupported e → AdjointSupported (addDimTerm d e)
  | Term.var x, _ => by simp [AdjointSupported, addDimTerm]
  | Term.const v ds, _ => by simp [AdjointSupported, addDimTerm]
  | Term.unit, _ => by simp [AdjointSupported, addDimTerm]
  | Term.loc ell, _ => by simp [AdjointSupported, addDimTerm]
  | Term.add e1 e2, h => by
      rcases h with ⟨h1, h2⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro
          (adjointSupported_addDimTerm d e1 h1)
          (adjointSupported_addDimTerm d e2 h2)
  | Term.mul e1 e2, h => by
      rcases h with ⟨h1, h2⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro
          (adjointSupported_addDimTerm d e1 h1)
          (adjointSupported_addDimTerm d e2 h2)
  | Term.sum e d', h => by
      simpa [AdjointSupported, addDimTerm] using
        adjointSupported_addDimTerm d e h
  | Term.expand e d', h => by
      simpa [AdjointSupported, addDimTerm] using
        adjointSupported_addDimTerm d e h
  | Term.copy e, h => by
      simpa [AdjointSupported, addDimTerm] using
        adjointSupported_addDimTerm d e h
  | Term.letBind x e1 e2, h => by
      rcases h with ⟨h1, h2⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro
          (adjointSupported_addDimTerm d e1 h1)
          (adjointSupported_addDimTerm d e2 h2)
  | Term.letpair x y e1 e2, h => by
      rcases h with ⟨h1, h2⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro
          (adjointSupported_addDimTerm d e1 h1)
          (adjointSupported_addDimTerm d e2 h2)
  | Term.pair e1 e2, h => by
      rcases h with ⟨h1, h2⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro
          (adjointSupported_addDimTerm d e1 h1)
          (adjointSupported_addDimTerm d e2 h2)
  | Term.fst tRight e, h => by
      simpa [AdjointSupported, addDimTerm] using
        adjointSupported_addDimTerm d e h
  | Term.snd tLeft e, h => by
      simpa [AdjointSupported, addDimTerm] using
        adjointSupported_addDimTerm d e h
  | Term.handle epsH body clauses, h => by
      rcases h with ⟨hBody, hClauses⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro
          (adjointSupported_addDimTerm d body hBody)
          (adjointSupportedClauses_addDimClauses d clauses hClauses)
  | Term.perform op e, h => by
      rcases h with ⟨hop, hBody⟩
      simpa [AdjointSupported, addDimTerm] using
        And.intro hop (adjointSupported_addDimTerm d e hBody)
  | Term.abs _ _ _, h => by cases h
  | Term.app _ _, h => by cases h
  | Term.uniformLike _ _ _, h => by cases h
  | Term.grad _ _ _ _, h => by cases h
  | Term.vmap _ _ _ _, h => by cases h

/-- Clause-list companion to `adjointSupported_addDimTerm`. -/
theorem adjointSupportedClauses_addDimClauses (d : Dim) :
    ∀ clauses,
      AdjointSupportedClauses clauses →
      AdjointSupportedClauses (addDimClauses d clauses)
  | [], _ => by simp [AdjointSupportedClauses, addDimClauses]
  | (_op, _x, _k, hb) :: rest, h => by
      rcases h with ⟨hHead, hRest⟩
      simpa [AdjointSupportedClauses, addDimClauses] using
        And.intro
          (adjointSupported_addDimTerm d hb hHead)
          (adjointSupportedClauses_addDimClauses d rest hRest)

end

theorem adjointTypeSupported_addDim (d : Dim) :
    ∀ t, AdjointTypeSupported t → AdjointTypeSupported (addDim d t)
  | Typ.tensor ds, _ => by simp [AdjointTypeSupported, addDim]
  | Typ.pair t1 t2, h => by
      rcases h with ⟨h1, h2⟩
      simpa [AdjointTypeSupported, addDim] using
        And.intro
          (adjointTypeSupported_addDim d t1 h1)
          (adjointTypeSupported_addDim d t2 h2)
  | Typ.unit, _ => by simp [AdjointTypeSupported, addDim]
  | Typ.arrow _ _ _, h => by cases h
  | Typ.tyVar _, h => by cases h

theorem adjointCtxSupported_addDimCtx (d : Dim) :
    ∀ Γ, AdjointCtxSupported Γ → AdjointCtxSupported (Γ.map (fun p => (p.1, p.2.map (addDim d))))
  | [], _ => by simp [AdjointCtxSupported]
  | (x, none) :: rest, h => by
      simpa [AdjointCtxSupported] using
        adjointCtxSupported_addDimCtx d rest h
  | (x, some t) :: rest, h => by
      rcases h with ⟨ht, hrest⟩
      simpa [AdjointCtxSupported] using
        And.intro
          (adjointTypeSupported_addDim d t ht)
          (adjointCtxSupported_addDimCtx d rest hrest)

mutual

/-- The adjoint term-to-term transformation, counter-threaded form.
    Wave 0 P1: total function (no `partial`) via explicit structural
    recursion on `body`. Every recursive call lands on a strictly
    smaller sub-term, so Lean's structural recursion checker accepts
    the definition.

    Wave 4 Track B: `n` is a fresh-name counter. Each recursion level
    that introduces bindings mints them via `freshName prefix n` and
    increases the counter it hands to sub-calls by the number of
    bindings introduced locally, so sub-calls produce names disjoint
    from everything in the enclosing linear context. The public
    entry point `adjoint` below starts the counter at `0`. -/
def adjointFrom (body : Term) (x : String) (gSeed : Term) (n : Nat) : Term :=
  match body with
  | Term.var y =>
      -- Parameter gradient: if y is the differentiated parameter, the
      -- current skeleton consumes the incoming seed and emits a unit
      -- accum marker. Either way, let the handler's origin filter decide.
      if y = x then
        adjointLeaf gSeed n
      else
        adjointLeaf gSeed n
  | Term.const _ _ =>
      -- No inputs; the handler's const-origin filter drops this.
      adjointLeaf gSeed n
  | Term.unit =>
      -- Unit has no gradient; linearity preserved by accum-emission.
      adjointLeaf gSeed n
  | Term.loc _ =>
      -- Runtime locations don't appear in source programs under grad.
      adjointLeaf gSeed n
  | Term.add e1 e2 =>
      -- No tape; copy(gSeed) and route to each operand. Sub-adjoints
      -- have type `unit` (each emits `perform accum`); sequence them
      -- via `letBind` so the compound result is also `unit` rather
      -- than `pair unit unit`. Three fresh names at this level
      -- (gA, gB, adjA), so inner calls use `n + 3`.
      Term.letpair (freshName "gA" n) (freshName "gB" n) (Term.copy gSeed)
        (Term.letBind (freshName "adjA" n)
          (adjointFrom e1 x (Term.var (freshName "gA" n)) (n + 3))
          (adjointFrom e2 x (Term.var (freshName "gB" n)) (n + 3)))
  | Term.mul e1 e2 =>
      -- Tape both operands, forward mul, backward via copy(gSeed).
      -- Sub-adjoints sequenced via `letBind` (see `add` note).
      -- Eight fresh names at this level (a, aTape, b, bTape, y,
      -- gA, gB, adjA), so inner calls use `n + 8`.
      --
      -- Wave 5 reshape: `copy gSeed` hoisted to the outermost layer
      -- so the seed's Γ_s → Γ_s' threading aligns with the linear
      -- context the outer letpair expects. `copy gSeed`, `copy e1`,
      -- `copy e2`, and the forward `mul a b` act on independent
      -- values, so reordering is semantically equivalent and
      -- preserves linear consumption.
      Term.letpair (freshName "gA" n) (freshName "gB" n) (Term.copy gSeed)
        (Term.letpair (freshName "a" n) (freshName "aTape" n) (Term.copy e1)
          (Term.letpair (freshName "b" n) (freshName "bTape" n) (Term.copy e2)
            (Term.letBind (freshName "y" n)
              (Term.mul (Term.var (freshName "a" n))
                        (Term.var (freshName "b" n)))
              (Term.letBind (freshName "adjA" n)
                (adjointFrom e1 x
                  (Term.mul (Term.var (freshName "gA" n))
                            (Term.var (freshName "bTape" n))) (n + 8))
                (adjointFrom e2 x
                  (Term.mul (Term.var (freshName "gB" n))
                            (Term.var (freshName "aTape" n))) (n + 8))))))
  | Term.sum e d =>
      -- Backward for sum is expand at the same dim `d` (Stage 1:
      -- dimensions are now named, so extent comes along with `d`).
      adjointFrom e x (Term.expand gSeed d) n
  | Term.expand e d =>
      adjointFrom e x (Term.sum gSeed d) n
  | Term.uniformLike e _ _ =>
      -- `uniform_like` is rejected by T-Grad's DiffCompat premise, so
      -- this case is vacuous — but we structurally recurse anyway to
      -- keep `adjointFrom` total.
      adjointFrom e x gSeed n
  | Term.letBind _ _ e2 =>
      -- Phase 1 placeholder: recurse on the result-producing body.
      adjointFrom e2 x gSeed n
  | Term.letpair _ _ _ e2 =>
      adjointFrom e2 x gSeed n
  | Term.pair e1 e2 =>
      -- Split the cotangent product and route one branch to each
      -- component.
      Term.letpair (freshName "gA" n) (freshName "gB" n) gSeed
        (Term.letBind (freshName "adjA" n)
          (adjointFrom e1 x (Term.var (freshName "gA" n)) (n + 3))
          (adjointFrom e2 x (Term.var (freshName "gB" n)) (n + 3)))
  | Term.fst tRight e =>
      adjointFrom e x (Term.pair gSeed (zeroCotangent tRight)) n
  | Term.snd tLeft e =>
      adjointFrom e x (Term.pair (zeroCotangent tLeft) gSeed) n
  | Term.copy e =>
      -- `copy : a -> a ⊗ a`, so the input cotangent is the sum of the
      -- two output cotangents.
      Term.letpair (freshName "gA" n) (freshName "gB" n) gSeed
        (adjointFrom e x
          (Term.add (Term.var (freshName "gA" n))
                    (Term.var (freshName "gB" n))) (n + 3))
  | Term.abs _ _ e => adjointFrom e x gSeed n
  | Term.app e1 _ => adjointFrom e1 x gSeed n
  | Term.grad _ _ _ e =>
      -- Nested grad is out of T0 §0 scope; structurally recurse so
      -- adjointFrom stays total. Phase 2 T9 may reject this case.
      adjointFrom e x gSeed n
  | Term.vmap _ _ _ e =>
      -- vmap-in-grad-body is out of T0 §0 scope; same treatment.
      adjointFrom e x gSeed n
  | Term.perform _ e => adjointFrom e x gSeed n
  | Term.handle _ body clauses =>
      adjointClausesFrom clauses x body gSeed n
termination_by (sizeOf body, 1)
decreasing_by
  all_goals
    simp_wf
    omega

/-- Companion to `adjointFrom`: walks a handler-clause list, recursing
    on each clause body and threading the residual seed all the way to
    the handled body. Each clause splits the incoming seed with `copy`,
    feeds one branch to the current clause body, and passes the
    residual branch to the tail; the base case runs the handled body's
    adjoint on the final residual seed. Three fresh names are
    introduced per clause (`gA`, `gB`, `adjHb`), so the tail call uses
    `n + 3`. -/
def adjointClausesFrom
    (clauses : List (EffectLabel × String × String × Term))
    (x : String) (body : Term) (gSeed : Term) (n : Nat) : Term :=
  match clauses with
  | [] => adjointFrom body x gSeed n
  | (_op, _xv, _kv, hb) :: rest =>
      -- Recurse on `hb` with one seed branch, then pass the residual
      -- branch to the tail, which eventually feeds the handled body.
      Term.letpair (freshName "gA" n) (freshName "gB" n) (Term.copy gSeed)
        (Term.letBind (freshName "adjHb" n)
          (adjointFrom hb x (Term.var (freshName "gA" n)) (n + 3))
          (adjointClausesFrom rest x body (Term.var (freshName "gB" n)) (n + 3)))
termination_by (sizeOf clauses + sizeOf body + 1, 0)
decreasing_by
  all_goals
    simp_wf
    omega

end

/-- Compatibility shim: the original fresh-name-free entry point.
    Starts the counter at `0`. Callers in Operational/Preservation use
    this form; Wave-4 Track B proofs that need freshness discipline
    call `adjointFrom` with an explicit counter instead. -/
def adjoint (body : Term) (x : String) (gSeed : Term) : Term :=
  adjointFrom body x gSeed 0

/-- Compatibility shim for `adjointClausesFrom`. -/
def adjointClauses (clauses : List (EffectLabel × String × String × Term))
                   (x : String) (body : Term) (gSeed : Term) : Term :=
  adjointClausesFrom clauses x body gSeed 0

mutual

/-- Typed companion to `adjointFrom`. The legacy transform stays
    untyped for compatibility with the current proof surface; this
    companion threads the primal result type explicitly so handler
    clauses can split structured cotangent seeds recursively rather than
    relying on tensor-only `copy`. Branches whose current structural
    placeholder still lacks enough type information fall back to the
    legacy transform. -/
def adjointTypedFrom
    (body : Term) (bodyTy : Typ) (x : String) (gSeed : Term) (n : Nat) : Term :=
  match body with
  | Term.var y =>
      if y = x then
        adjointLeaf gSeed n
      else
        adjointLeaf gSeed n
  | Term.const _ _ =>
      adjointLeaf gSeed n
  | Term.unit =>
      adjointLeaf gSeed n
  | Term.loc _ =>
      adjointLeaf gSeed n
  | Term.add e1 e2 =>
      match bodyTy with
      | Typ.tensor _ =>
          splitCotangentSeedFrom bodyTy gSeed n
            (fun n' gA gB =>
              let adjA := freshName "adjA" n'
              Term.letBind adjA
                (adjointTypedFrom e1 bodyTy x gA (n' + 1))
                (adjointTypedFrom e2 bodyTy x gB (n' + 1)))
      | _ =>
          adjointFrom body x gSeed n
  | Term.mul e1 e2 =>
      match bodyTy with
      | Typ.tensor _ =>
          splitCotangentSeedFrom bodyTy gSeed n
            (fun n' gA gB =>
              let a := freshName "a" n'
              let aTape := freshName "aTape" n'
              let b := freshName "b" (n' + 1)
              let bTape := freshName "bTape" (n' + 1)
              let y := freshName "y" (n' + 2)
              let adjA := freshName "adjA" (n' + 3)
              Term.letpair a aTape (Term.copy e1)
                (Term.letpair b bTape (Term.copy e2)
                  (Term.letBind y
                    (Term.mul (Term.var a) (Term.var b))
                    (Term.letBind adjA
                      (adjointTypedFrom e1 bodyTy x
                        (Term.mul gA (Term.var bTape)) (n' + 4))
                      (adjointTypedFrom e2 bodyTy x
                        (Term.mul gB (Term.var aTape)) (n' + 4))))))
      | _ =>
          adjointFrom body x gSeed n
  | Term.sum e d =>
      match bodyTy with
      | Typ.tensor ds =>
          adjointTypedFrom e (Typ.tensor (ins ds d)) x (Term.expand gSeed d) n
      | _ =>
          adjointFrom body x gSeed n
  | Term.expand e d =>
      match bodyTy with
      | Typ.tensor ds =>
          adjointTypedFrom e (Typ.tensor (rem ds d)) x (Term.sum gSeed d) n
      | _ =>
          adjointFrom body x gSeed n
  | Term.uniformLike e _ _ =>
      match bodyTy with
      | Typ.tensor _ =>
          adjointTypedFrom e bodyTy x gSeed n
      | _ =>
          adjointFrom body x gSeed n
  | Term.letBind _ _ e2 =>
      adjointTypedFrom e2 bodyTy x gSeed n
  | Term.letpair _ _ _ e2 =>
      adjointTypedFrom e2 bodyTy x gSeed n
  | Term.pair e1 e2 =>
      match bodyTy with
      | Typ.pair t1 t2 =>
          Term.letpair (freshName "gA" n) (freshName "gB" n) gSeed
            (Term.letBind (freshName "adjA" n)
              (adjointTypedFrom e1 t1 x (Term.var (freshName "gA" n)) (n + 3))
              (adjointTypedFrom e2 t2 x (Term.var (freshName "gB" n)) (n + 3)))
      | _ =>
          adjointFrom body x gSeed n
  | Term.fst tRight e =>
      adjointTypedFrom e (Typ.pair bodyTy tRight) x
        (Term.pair gSeed (zeroCotangent tRight)) n
  | Term.snd tLeft e =>
      adjointTypedFrom e (Typ.pair tLeft bodyTy) x
        (Term.pair (zeroCotangent tLeft) gSeed) n
  | Term.copy e =>
      match bodyTy with
      | Typ.pair (Typ.tensor ds) (Typ.tensor _) =>
          Term.letpair (freshName "gA" n) (freshName "gB" n) gSeed
            (adjointTypedFrom e (Typ.tensor ds) x
              (Term.add (Term.var (freshName "gA" n))
                        (Term.var (freshName "gB" n))) (n + 3))
      | _ =>
          adjointFrom body x gSeed n
  | Term.abs _ _ e =>
      match bodyTy with
      | Typ.arrow _ tOut _ =>
          adjointTypedFrom e tOut x gSeed n
      | _ =>
          adjointFrom body x gSeed n
  | Term.app _ _ =>
      adjointFrom body x gSeed n
  | Term.grad _ _ tOut e =>
      adjointTypedFrom e tOut x gSeed n
  | Term.vmap _ _ _ _ =>
      adjointFrom body x gSeed n
  | Term.perform op e =>
      match bodyTy with
      | Typ.unit =>
          adjointTypedFrom e (opArgType op) x gSeed n
      | _ =>
          adjointFrom body x gSeed n
  | Term.handle _ body clauses =>
      adjointTypedClausesFrom clauses x body bodyTy gSeed n
termination_by (sizeOf body, 1)
decreasing_by
  all_goals
    simp_wf
    omega

/-- Typed companion to `adjointClausesFrom`. The handled result type
    `bodyTy` lets clause threading reuse `splitCotangentSeedFrom`, so
    structured cotangent seeds no longer have to go through tensor-only
    `copy` at this surface. -/
def adjointTypedClausesFrom
    (clauses : List (EffectLabel × String × String × Term))
    (x : String) (body : Term) (bodyTy : Typ) (gSeed : Term) (n : Nat) : Term :=
  match clauses with
  | [] =>
      adjointTypedFrom body bodyTy x gSeed n
  | (_op, _xv, _kv, hb) :: rest =>
      splitCotangentSeedFrom bodyTy gSeed n
        (fun n' gA gB =>
          let adjHb := freshName "adjHb" n'
          Term.letBind adjHb
            (adjointTypedFrom hb bodyTy x gA (n' + 1))
            (adjointTypedClausesFrom rest x body bodyTy gB (n' + 1)))
termination_by (sizeOf clauses + sizeOf body + 1, 0)
decreasing_by
  all_goals
    simp_wf
    omega

end

/-- Compatibility shim for `adjointTypedFrom`. -/
def adjointTyped (body : Term) (bodyTy : Typ) (x : String) (gSeed : Term) : Term :=
  adjointTypedFrom body bodyTy x gSeed 0

/-- Compatibility shim for `adjointTypedClausesFrom`. -/
def adjointTypedClauses (clauses : List (EffectLabel × String × String × Term))
    (x : String) (body : Term) (bodyTy : Typ) (gSeed : Term) : Term :=
  adjointTypedClausesFrom clauses x body bodyTy gSeed 0

end LaCaDiLE
