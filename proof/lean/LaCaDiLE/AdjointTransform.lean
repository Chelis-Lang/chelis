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
    Phase 1 uses a static name; Phase 2 will thread a counter. -/
def tapeName (base : String) : String := base ++ "_tape"

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

/-- The adjoint term-to-term transformation. Wave 0 P1: total function
    (no `partial`) via explicit structural recursion on `body`. Every
    recursive call lands on a strictly smaller sub-term, so Lean's
    structural recursion checker accepts the definition. -/
def adjoint (body : Term) (x : String) (gSeed : Term) : Term :=
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
      -- No tape; copy(gSeed) and route to each operand.
      Term.letpair "gA" "gB" (Term.copy gSeed)
        (Term.pair (adjoint e1 x (Term.var "gA"))
                   (adjoint e2 x (Term.var "gB")))
  | Term.mul e1 e2 =>
      -- Tape both operands, forward mul, backward via copy(gSeed).
      Term.letpair "a" (tapeName "a") (Term.copy e1)
        (Term.letpair "b" (tapeName "b") (Term.copy e2)
          (Term.letBind "y" (Term.mul (Term.var "a") (Term.var "b"))
            (Term.letpair "gA" "gB" (Term.copy gSeed)
              (Term.pair
                (adjoint e1 x (Term.mul (Term.var "gA")
                                         (Term.var (tapeName "b"))))
                (adjoint e2 x (Term.mul (Term.var "gB")
                                         (Term.var (tapeName "a"))))))))
  | Term.sum e i =>
      -- Backward for sum is expand at the same axis. The extent
      -- placeholder 0 is replaced in Phase 2 T9 when the shape tape
      -- ships (subsumes P8).
      adjoint e x (Term.expand gSeed i 0)
  | Term.expand e i _k =>
      adjoint e x (Term.sum gSeed i)
  | Term.uniformLike e _ _ =>
      -- `uniform_like` is rejected by T-Grad's DiffCompat premise, so
      -- this case is vacuous — but we structurally recurse anyway to
      -- keep `adjoint` total.
      adjoint e x gSeed
  | Term.letBind _ e1 _ =>
      -- Phase 2 T9 will invert the forward/backward order; Phase 1
      -- skeleton recurses into `e1` to preserve the structural measure.
      adjoint e1 x gSeed
  | Term.letpair _ _ e1 _ =>
      adjoint e1 x gSeed
  | Term.pair e1 _ =>
      adjoint e1 x gSeed
  | Term.fst e => adjoint e x gSeed
  | Term.snd e => adjoint e x gSeed
  | Term.copy e => adjoint e x gSeed
  | Term.abs _ _ e => adjoint e x gSeed
  | Term.app e1 _ => adjoint e1 x gSeed
  | Term.grad _ _ _ e =>
      -- Nested grad is out of T0 §0 scope; structurally recurse so
      -- adjoint stays total. Phase 2 T9 may reject this case.
      adjoint e x gSeed
  | Term.vmap _ _ e =>
      -- vmap-in-grad-body is out of T0 §0 scope; same treatment.
      adjoint e x gSeed
  | Term.perform _ e => adjoint e x gSeed
  | Term.handle _ body clauses =>
      adjointClauses clauses x gSeed (adjoint body x gSeed)

/-- Companion to `adjoint`: walks a handler-clause list, recursing on
    each clause body. Takes an accumulator `acc` (the `adjoint body`
    result) so the final term threads the body's adjoint with each
    clause's adjoint. For Phase 1 skeleton, each clause's adjoint
    emits a vestigial `perform accum` via `adjoint` on the clause
    body. -/
def adjointClauses (clauses : List (EffectLabel × String × String × Term))
                   (x : String) (gSeed : Term) (acc : Term) : Term :=
  match clauses with
  | [] => acc
  | (_op, _xv, _kv, hb) :: rest =>
      -- Recurse on hb then on the tail. For Phase 1 skeleton, the
      -- produced term just chains the sub-adjoints; Phase 2 T9 will
      -- restructure to match the handler's semantic reduction.
      Term.letBind "_adj_hb" (adjoint hb x gSeed)
                   (adjointClauses rest x gSeed acc)

end

end LaCaDiLE
