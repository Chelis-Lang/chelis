# Phase 2 Wave 2 Status — Preservation And Sorry Inventory

Snapshot on branch `chelis-proof` after restoring a green root build, closing named substitution, repairing the handler seed-threading transform, closing the executable `handle` typing branch in `AdjointTyping.lean`, fixing the concrete `letBind` / `letpair` adjoint routing bug, landing typed cotangent-seed routing for `pair` / `fst` / `snd` / `copy`, and then classifying the remaining handled product-seed blocker in the legacy public AD theorem surface.

## Acceptance Oracle

Use `cd proof/lean && lake build` as the authoritative branch-health check.

`proof/scripts/check_toolchain.py` is useful for workstation setup diagnostics, but it is not the proof-status oracle for this branch because it currently depends on extra local Vibe/Leanstral configuration that is not required for a repo-local Lean build.

## Current State

- `cd proof/lean && lake build` is green again on the current branch head.
- `TranslationDB.lean`'s four DB-backed preservation wrappers are closed.
- `Preservation.lean` syntactically closes the substitution-dependent cases, `handleOpCtx`, `handleOpCtxs`, `ctx`, `tgrad`, and `tvmap`.
- `Substitution.lean` is closed under the honest lexical-scoped theorem shape.
- `AddDim.lean` is no longer the `tvmap` upstream blocker, and the earlier theorem-shape bug has been repaired by making `vmap`'s batch dimension explicit in both the named and DB syntax.
- `AdjointTyping.lean`'s repaired `handle` branch is now typed.
- `AdjointTyping.lean` now has the repaired structural transform substrate in place:
  - a concrete typed regression showing the repaired `letBind` / `letpair` recursion fixes the old `grad` / `expand` routing bug
  - typed cotangent-seed routing in `AdjointTransform.lean` for `pair` / `fst` / `snd` / `copy`
- Lean now also contains a handled product-seed counterexample showing that the current public `adjointFrom` theorem surface is still false: `handle` still routes through the legacy tensor-only clause path, so a structured cotangent seed can hit a tensor-only `copy` before the `mul` case is even in play
- The branch still contains 1 executable `sorry` on the main proof path:
  - `LaCaDiLE/AdjointTyping.lean`: one consolidated catch-all admission; the next AD repair is to move the public theorem surface from `adjointFrom` to `adjointTypedFrom` / `adjointTypedClausesFrom`, then close the typed helper and the separate `mul` case

The current theorem-shape boundary is now explicit in Lean:

- plain `RuntimeLinear` is too weak as a generic preservation premise
- `ActiveRuntimeLinear` repairs the captured-handler context counterexamples
- `ActiveRuntimeLinear` is still too weak for direct handled operations, because a dormant clause body can become active in one step and expose duplicated locations
- the recursive `DeepActiveRuntimeLinear` repair is still too weak globally: a typed deep-active beta step can expose a dormant clause beside an active sibling with the same location, and one contextual `handleOpDirect` step later that overlap becomes active and breaks the invariant
- `tvmap` is no longer blocked on theorem shape: typing and stepping now share the batch dimension through the explicit `vmap` term annotation, and the preservation case is closed

## What Closed Recently

- `LinearitySoundness.lean`'s current theorem is the store-wellformedness preservation result, not the final Theorem 4 package.
- The AddDim multiset refactor landed, including the `Quotient.sound` proof path.
- `plug_preserves_typing` is fully proved across all 19 evaluation-context cases.
- `Progress` was restructured around the three-way outcome (`value / steps / stuck-on-perform`) and is down to a single remaining admission outside this file.
- Store weakening infrastructure and the effect-scoping lemma `hasType_perform_eff_mem` are proved.
- `SubstitutionDB.lean`'s last remaining admission is closed.
- `Substitution.lean`'s public theorem surface is now closed under the lexical-scoped substitution statement.
- The old `SlotCorr` / `SlotKillsOnly` bridge has been deleted; `CtxCorr` is now the only translation correspondence layer.
- `OpSigMatch` has been tightened to a functional operation-signature witness, which closed the direct handler wrapper in `TranslationDB.lean`.
- The operational semantics now mint a fresh captured continuation binder instead of hardcoding `_kArg`.
- `Preservation.lean` now closes the generic `ctx` case by using frame-local store agreement over untouched context locations instead of global store monotonicity.
- `Preservation.lean` now also contains explicit counterexamples showing:
  - why unconditional runtime preservation is false
  - why the first captured-handler `ActiveRuntimeLinear` repair is not the final invariant
- `AdjointTransform.lean` now threads handler seeds linearly through the clause chain, and `AdjointTyping.lean` closes the matching executable `handle` typing proof.
- `AdjointTransform.lean` now also routes typed cotangent seeds through `pair` / `fst` / `snd` / `copy`, so the old monomorphic product/projection mismatch has been removed from the transform surface itself.
- `LinearitySoundness.lean` now packages the current bridge from `RuntimeLinear` to the handler-aware runtime debt classification instead of leaving that branch implicit.

## Remaining Critical Path

1. Handler-aware runtime invariant
   The next theorem cannot be “one-step `RuntimeLinear` preservation,” and it also cannot be plain “one-step `ActiveRuntimeLinear` preservation.” The new deep-active two-step counterexample shows the final invariant must relate dormant handler clauses to the surrounding active context, not just recurse into clause bodies locally.

2. `AdjointTyping.lean`
   The repaired `handle` branch is closed, the old concrete `E-Grad` routing bug is repaired, and the transform now routes typed cotangent seeds through product/projection/copy. But the legacy public `adjointFrom` theorem surface is still false for handled product seeds, because `handle` still uses the tensor-only clause path. `tgrad` preservation is therefore still not honestly settled. The remaining AD blockers are:
   - switch the public theorem surface and downstream `tgrad` plumbing to `adjointTypedFrom` / `adjointTypedClausesFrom`
   - reprove the private typing helper on that typed surface
   - separately, solve the `mul` tape/effect-row case

## Next Moves

1. State the correct stronger cross-boundary handler-aware runtime invariant, using the direct-handler and deep-active two-step counterexamples as the acceptance tests for theorem shape.
2. Sync the paper-facing workstreams so §5 no longer promises a preservation premise that the Lean tree now falsifies.
3. Re-close the public AD surface against the now-updated transform domain by moving it onto `adjointTypedFrom` / `adjointTypedClausesFrom`, then reprove the private helper there, finish the remaining admitted `mul` case, and return to the AD-correctness wave.
