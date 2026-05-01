# Phase 2 Wave 2 Status — Preservation And Sorry Inventory

Snapshot on branch `chelis-proof` after restoring a green root build, closing named substitution, repairing the handler seed-threading transform, closing the executable `handle` typing branch in `AdjointTyping.lean`, fixing the concrete `letBind` / `letpair` adjoint routing bug, landing typed cotangent-seed routing for `pair` / `fst` / `snd` / `copy`, proving the supported-fragment closure lemmas, and then classifying both the legacy handled product-seed blocker and the newer higher-order blocker on the typed AD surface.

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
- `AdjointTransform.lean` now also names the current honest first-order AD fragment:
  - `AdjointTypeSupported`
  - `AdjointCtxSupported`
  - `AdjointSupported`
  - `AdjointSupportedClauses`
- `Substitution.lean` now proves the key closure facts for that fragment:
  - `adjointSupported_subst`
  - `adjointSupportedClauses_substClauses`
  - `adjointSupported_of_typed_value`
- Lean now contains the current AD counterexample classes that still constrain the private theorem surface:
  - a handled product-seed counterexample showing that the legacy public `adjointFrom` theorem surface is still false: `handle` still routes through the tensor-only clause path, so a structured cotangent seed can hit a tensor-only `copy` before the `mul` case is even in play
  - a higher-order counterexample showing that unrestricted `adjointTypedFrom` is also false on raw terms, because the typed transform still needs an explicit supported-fragment premise to exclude `abs` / `app` / nested `grad` / nested `vmap`
  - a concrete `mul` context-gap counterexample showing that the current private shape-only helper is false on arbitrary seed-only contexts, because the transformed `mul` term replays raw source operands and therefore needs source-context information in the theorem statement
- The branch still contains 2 executable `sorry`s on the main proof path, both in `LaCaDiLE/AdjointTyping.lean`:
  - the `mul` branch of the private typed helper
  - the clause-list `cons` pair-seed branch of the same helper stack
  The public typed theorem and the `tgrad` preservation plumbing are already on `adjointTypedFrom` / `adjointTypedClausesFrom` with the slot-threaded existential output shape. The remaining AD repair is therefore to replace the private shape-only helper with an honest source-typed supported-domain induction, then close the remaining `mul` / `cons` cases.

The current theorem-shape boundary is now explicit in Lean:

- plain `RuntimeLinear` is too weak as a generic preservation premise
- `ActiveRuntimeLinear` repairs the captured-handler context counterexamples
- `ActiveRuntimeLinear` is still too weak for direct handled operations, because a dormant clause body can become active in one step and expose duplicated locations
- the recursive `DeepActiveRuntimeLinear` repair is still too weak globally: a typed deep-active beta step can expose a dormant clause beside an active sibling with the same location, and one contextual `handleOpDirect` step later that overlap becomes active and breaks the invariant
- `Operational.lean` now packages the next stronger candidate as `HandlerAwareRuntimeLinear`, but the public config-level boundary has moved on: `LinearitySoundness.lean` now exports `runtimeSafeConfig_step_or_debt`, and `Preservation.lean` now exports `preservation_runtimeSafeConfig_or_debt`, which preserves typing/store/runtime safety or enters an explicit `RuntimeSafeDebt` case as an interim checkpoint only
- a newer config-level `RuntimeSafeConfig` wrapper is also not yet final: there is now a concrete `handleOpCtx` counterexample showing that generic closure for that public surface is false
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
- `LinearitySoundness.lean` now packages the current config-level runtime-safety split as `runtimeSafeConfig_step_or_debt` instead of leaving the runtime debt boundary implicit.
- `Operational.lean` still contains the stronger internal candidate predicate `HandlerAwareRuntimeLinear`, and `Preservation.lean` now uses the config-level split to export the combined `preservation_runtimeSafeConfig_or_debt` theorem.

## Remaining Critical Path

1. Runtime-safe config boundary
   The next public theorem still cannot be “one-step `RuntimeLinear` preservation,” and it also cannot be plain “one-step `ActiveRuntimeLinear` preservation.” The internal term-level candidate remains `HandlerAwareRuntimeLinear`, but the exported boundary is now config-level: `runtimeSafeConfig_step_or_debt` plus `preservation_runtimeSafeConfig_or_debt`. The remaining work is to shrink the explicit `RuntimeSafeDebt` families, especially beta/let-style substitution, direct/captured handler substitution, `tgrad`, and contextual replugging across store-changing inner steps.

2. `AdjointTyping.lean`
   The repaired `handle` branch is closed, the old concrete `E-Grad` routing bug is repaired, and the transform now routes typed cotangent seeds through product/projection/copy. The public typed theorem is already on the honest slot-threaded existential output surface, but the private helper is still shape-only and therefore still false on arbitrary seed-only `mul` contexts. `tgrad` preservation is therefore still not honestly settled. The remaining AD blockers are:
   - replace the admitted private shape-only helper with a source-typed supported-domain induction
   - thread the supported-fragment premises through typing, substitution, and stepping
   - solve the remaining local `mul` and clause-list `cons` cases

## Next Moves

1. Refine and close the current `RuntimeSafeDebt` boundary, using the direct-handler and deep-active two-step counterexamples as the acceptance tests for the remaining side conditions.
2. Sync the paper-facing workstreams so §5 no longer promises a preservation premise that the Lean tree now falsifies and so it names the current handler-aware boundary honestly.
3. Re-close the public AD surface against the now-updated transform domain by keeping it on `adjointTypedFrom` / `adjointTypedClausesFrom`, redesigning the `mul` replay/tape boundary and public output-context claim so that surface is honest, threading the supported-fragment premises through `T-Grad` / `E-Grad` / the public theorem, then reproving the private helper there, finishing the remaining admitted `mul` and clause-list `cons` cases, and returning to the AD-correctness wave.
