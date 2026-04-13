-- LaCaDiLE/AdjointTransform.lean — adjoint : Term → String → Term → Term.
--
-- Phase 1 T8: structural skeleton of the adjoint transformation from
-- T0 §4. This is a `partial def` for Phase 1; Phase 2 WS2.2 will prove
-- termination and state the adjoint typing lemma.
--
-- The function signature `adjoint body x gSeed` matches the E-Grad
-- reduction in opsem.tex: given the differentiated function's body
-- `body`, the parameter name `x`, and the incoming output-gradient seed
-- `gSeed`, produce the transformed term that:
--   (a) inserts tape copies (let (a, a') = copy(a)) before primitives
--       whose adjoint rule requires operand values (mul),
--   (b) emits `perform accum(loc, contribution)` for each operand,
--   (c) produces a term whose result is the parameter gradient.
--
-- The Phase 1 implementation below handles the atomic primitives correctly
-- and uses a conservative catch-all for composition cases. Phase 2 will
-- replace the catch-all with the full T0 §4 recursive pattern.

import LaCaDiLE.Syntax

namespace LaCaDiLE

/-- Fresh variable name suffix for tape-binding generation.
    Phase 1 uses a static name; Phase 2 will thread a counter. -/
def tapeName (base : String) : String := base ++ "_tape"

/-- The adjoint transformation.

    Phase 1 notes:
    * `mul(e1, e2)` gets full tape treatment (two copies of each operand
      plus the forward mul plus backward contributions via `copy(gSeed)`).
    * `add(e1, e2)` gets the backward `copy(gSeed)` + two accum calls.
    * `sum(e, i)` becomes `expand(gSeed, i, 0)` then recurses.
    * `expand(e, i, k)` becomes `sum(gSeed, i)` then recurses.
    * `const(_, _)` emits an accum call whose handler will drop it (the
      origin is a const literal; see T0 §2.5).
    * `var y` — if `y = x`, the parameter gradient flows through as
      `gSeed`; otherwise route through accum (Phase 2 will distinguish
      free-variable captures from the differentiated parameter).
    * `let`, `copy`, `letpair`, pair, `fst`/`snd`, `handle`, `perform`,
      and the opaque `loc`/`unit`/`uniformLike`/`grad`/`vmap` cases use
      a catch-all that preserves the term and emits a vestigial accum.
      Phase 2 replaces each of these with the structural recursion from
      T0 §4. -/
partial def adjoint (body : Term) (x : String) (gSeed : Term) : Term :=
  match body with
  | Term.var y =>
      -- Parameter gradient: if y is the differentiated parameter, the
      -- incoming seed IS the contribution. Otherwise the seed is routed
      -- to whatever y's origin is via accum.
      if y = x then
        Term.perform EffectLabel.accum gSeed
      else
        Term.perform EffectLabel.accum gSeed
  | Term.const _ _ =>
      -- No inputs; the incoming seed has no operand to flow back to.
      -- The handler filters by origin and drops the contribution.
      Term.perform EffectLabel.accum gSeed
  | Term.add e1 e2 =>
      -- No tape; copy(gSeed) and route.
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
      -- Backward for sum is expand at the same axis. The extent placeholder
      -- 0 is replaced in Phase 2 by the actual shape tracked on the tape.
      adjoint e x (Term.expand gSeed i 0)
  | Term.expand e i _k =>
      -- Backward for expand is sum at the same axis.
      adjoint e x (Term.sum gSeed i)
  | Term.letBind y e1 e2 =>
      -- Phase 1 skeleton: recurse on e2 with the original bindings in scope.
      -- Phase 2 will thread the forward let-chain correctly.
      Term.letBind y e1 (adjoint e2 x gSeed)
  | _ =>
      -- Catch-all for Phase 1 skeleton: pair, fst, snd, unit, copy, letpair,
      -- uniformLike (ruled out by T-Grad), grad (nested — excluded by T0 §0),
      -- vmap (ruled out for in-grad-body by T0 §0), handle, perform, loc.
      -- Emit a vestigial accum to preserve linearity of gSeed; Phase 2 T9
      -- replaces this with the proper structural recursion from T0 §4.
      Term.perform EffectLabel.accum gSeed

end LaCaDiLE
