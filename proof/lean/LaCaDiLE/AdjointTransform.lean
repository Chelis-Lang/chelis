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
--   * `letBind y e1 e2` recurses on `e2` with `e1` passing through —
--     Phase 2 T9 will invert the forward/backward order properly.
--   * `pair`, `fst`, `snd`, `abs`, `app`, `uniformLike`, `grad`, `vmap`,
--     `perform` all structurally recurse on sub-terms with a vestigial
--     `perform accum` at the base; the T0 §4 rules for these are
--     Phase 2 T9 work.
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
  base ++ "_" ++ toString n

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
      -- incoming seed IS the contribution. Either way, emit accum and
      -- let the handler's origin filter decide.
      if y = x then
        Term.perform EffectLabel.accum gSeed
      else
        Term.perform EffectLabel.accum gSeed
  | Term.const _ _ =>
      -- No inputs; the handler's const-origin filter drops this.
      Term.perform EffectLabel.accum gSeed
  | Term.unit =>
      -- Unit has no gradient; linearity preserved by accum-emission.
      Term.perform EffectLabel.accum gSeed
  | Term.loc _ =>
      -- Runtime locations don't appear in source programs under grad.
      Term.perform EffectLabel.accum gSeed
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
      Term.letpair (freshName "a" n) (freshName "aTape" n) (Term.copy e1)
        (Term.letpair (freshName "b" n) (freshName "bTape" n) (Term.copy e2)
          (Term.letBind (freshName "y" n)
            (Term.mul (Term.var (freshName "a" n))
                      (Term.var (freshName "b" n)))
            (Term.letpair (freshName "gA" n) (freshName "gB" n) (Term.copy gSeed)
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
  | Term.letBind _ e1 _ =>
      -- Phase 2 T9 will invert the forward/backward order; Phase 1
      -- skeleton recurses into `e1` to preserve the structural measure.
      adjointFrom e1 x gSeed n
  | Term.letpair _ _ e1 _ =>
      adjointFrom e1 x gSeed n
  | Term.pair e1 _ =>
      adjointFrom e1 x gSeed n
  | Term.fst e => adjointFrom e x gSeed n
  | Term.snd e => adjointFrom e x gSeed n
  | Term.copy e => adjointFrom e x gSeed n
  | Term.abs _ _ e => adjointFrom e x gSeed n
  | Term.app e1 _ => adjointFrom e1 x gSeed n
  | Term.grad _ _ _ e =>
      -- Nested grad is out of T0 §0 scope; structurally recurse so
      -- adjointFrom stays total. Phase 2 T9 may reject this case.
      adjointFrom e x gSeed n
  | Term.vmap _ _ e =>
      -- vmap-in-grad-body is out of T0 §0 scope; same treatment.
      adjointFrom e x gSeed n
  | Term.perform _ e => adjointFrom e x gSeed n
  | Term.handle _ body clauses =>
      adjointClausesFrom clauses x gSeed (adjointFrom body x gSeed n) n

/-- Companion to `adjointFrom`: walks a handler-clause list, recursing
    on each clause body. Takes an accumulator `acc` (the
    `adjointFrom body` result) so the final term threads the body's
    adjoint with each clause's adjoint. For Phase 1 skeleton, each
    clause's adjoint emits a vestigial `perform accum` via
    `adjointFrom` on the clause body. One fresh binder per clause;
    the tail call uses `n + 1`. -/
def adjointClausesFrom
    (clauses : List (EffectLabel × String × String × Term))
    (x : String) (gSeed : Term) (acc : Term) (n : Nat) : Term :=
  match clauses with
  | [] => acc
  | (_op, _xv, _kv, hb) :: rest =>
      -- Recurse on hb then on the tail. For Phase 1 skeleton, the
      -- produced term just chains the sub-adjoints; Phase 2 T9 will
      -- restructure to match the handler's semantic reduction.
      Term.letBind (freshName "adjHb" n)
                   (adjointFrom hb x gSeed (n + 1))
                   (adjointClausesFrom rest x gSeed acc (n + 1))

end

/-- Compatibility shim: the original fresh-name-free entry point.
    Starts the counter at `0`. Callers in Operational/Preservation use
    this form; Wave-4 Track B proofs that need freshness discipline
    call `adjointFrom` with an explicit counter instead. -/
def adjoint (body : Term) (x : String) (gSeed : Term) : Term :=
  adjointFrom body x gSeed 0

/-- Compatibility shim for `adjointClausesFrom`. -/
def adjointClauses (clauses : List (EffectLabel × String × String × Term))
                   (x : String) (gSeed : Term) (acc : Term) : Term :=
  adjointClausesFrom clauses x gSeed acc 0

end LaCaDiLE
