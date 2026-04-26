# Phase 2 Wave 2 Status — Preservation And Sorry Inventory

Snapshot on branch `chelis-proof` after restoring a green root build, closing named substitution, repairing the handler seed-threading transform, and closing the executable `handle` typing branch in `AdjointTyping.lean`.

## Acceptance Oracle

Use `cd proof/lean && lake build` as the authoritative branch-health check.

`proof/scripts/check_toolchain.py` is useful for workstation setup diagnostics, but it is not the proof-status oracle for this branch because it currently depends on extra local Vibe/Leanstral configuration that is not required for a repo-local Lean build.

## Current State

- `cd proof/lean && lake build` is green again on the current branch head.
- `TranslationDB.lean`'s four DB-backed preservation wrappers are closed.
- `Preservation.lean` closes the substitution-dependent cases, `handleOpCtx`, `handleOpCtxs`, `ctx`, `tgrad`, and `tvmap`.
- `Substitution.lean` is closed under the honest lexical-scoped theorem shape.
- `AddDim.lean` is no longer the `tvmap` upstream blocker, and the earlier theorem-shape bug has been repaired by making `vmap`'s batch dimension explicit in both the named and DB syntax.
- `AdjointTyping.lean`'s repaired `handle` branch is now typed.
- The branch still contains 1 executable `sorry` on the main proof path:
  - `LaCaDiLE/AdjointTyping.lean`: one consolidated catch-all admission, now reduced to the `mul` / `sum` / `expand` adjoint cases

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

## Remaining Critical Path

1. Handler-aware runtime invariant
   The next theorem cannot be “one-step `RuntimeLinear` preservation,” and it also cannot be plain “one-step `ActiveRuntimeLinear` preservation.” The new deep-active two-step counterexample shows the final invariant must relate dormant handler clauses to the surrounding active context, not just recurse into clause bodies locally.

2. `AdjointTyping.lean`
   `tgrad` preservation is now closed, the repaired `handle` branch is closed, and the remaining admitted adjoint cases still block the final AD-correctness wave.

## Next Moves

1. State the correct stronger cross-boundary handler-aware runtime invariant, using the direct-handler and deep-active two-step counterexamples as the acceptance tests for theorem shape.
2. Sync the paper-facing workstreams so §5 no longer promises a preservation premise that the Lean tree now falsifies.
3. Finish the remaining admitted `mul` / `sum` / `expand` adjoint cases and then return to the AD-correctness wave.
