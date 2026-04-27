# Chelis Proof — Project Plan

## Current Snapshot

- Branch: `chelis-proof`
- Proof build oracle: `cd proof/lean && lake build`
- Current proof state: the branch builds cleanly, the four DB-backed preservation wrappers are closed, `Translation.lean` now exposes the honest lexical forward bridge without admits, `Substitution.lean` is closed under its honest lexical-scoped theorem shape, and the only remaining executable admit is the consolidated catch-all in `AdjointTyping.lean`
- `Preservation.lean` is syntactically admit-free on the executable branch, including `tgrad` and `tvmap`, but the `tgrad` branch is not yet an honest closed result because it still imports an admitted public adjoint theorem surface. The AD transform now has the two necessary structural repairs on branch: the old concrete `E-Grad` `letBind` / `letpair` routing bug is fixed, and product/projection/copy nodes now route typed cotangent seeds rather than a monomorphic tensor seed. But Lean now also contains a handled product-seed counterexample showing that the current public `adjointFrom` surface is still false because `handle` delegates to the legacy tensor-only clause path. So the AD critical path is: move the public theorem surface and `tgrad` plumbing onto `adjointTypedFrom` / `adjointTypedClausesFrom`, reprove the helper there, and then close the separate `mul` tape/effect-row case. Separately, the stronger runtime-invariant story for iterating preservation is still unsettled: plain `RuntimeLinear` is false, `ActiveRuntimeLinear` is too weak for direct handled operations, and the current recursive `DeepActiveRuntimeLinear` candidate is still not compositional across a beta step followed by contextual `handleOpDirect`. `Operational.lean` now names the stronger cross-boundary candidate `HandlerAwareRuntimeLinear`, and `LinearitySoundness.lean` proves a partial one-step `HandlerAwareRuntimeLinear c2.term ∨ HandlerAwareRuntimeDebt c1 c2` boundary, but raw beta/direct-handler substitution and context-freshness counterexamples show that the remaining theorem still needs typing and store-side side conditions.
- `AdjointTyping.lean`'s repaired `handle` branch is now closed, but the current public `adjointFrom` theorem surface is still false for handled product seeds. The remaining AD gap is therefore localized to:
  - switching the public theorem surface and downstream `tgrad` plumbing onto `adjointTypedFrom` / `adjointTypedClausesFrom`
  - reproving the private typing helper on that typed surface
  - the `mul` tape/effect-row case
- `AddDim.lean` is closed, the earlier `tvmap` dimension-choice mismatch was repaired by making the batch dimension explicit in the term syntax, and the remaining `tvmap` preservation proof is now closed
- Immediate critical path:
  1. repair the admitted public adjoint theorem surface in `AdjointTyping.lean` by moving it off legacy `adjointFrom`, so the `tgrad` preservation branch rests on an honest theorem again
  2. close the current `HandlerAwareRuntimeLinear` / `HandlerAwareRuntimeDebt` runtime-invariant boundary into the public preservation story
  3. return to the AD-correctness wave once both the adjoint surface and runtime invariant are reduced further

This file tracks the real branch state, not the original project plan as imagined before the mechanization work started landing.

## Dependency Graph

```
WS1 (Core Calculus on Paper)
 ├─→ WS2 (Paper Proofs, all 5 theorems)
 │    ├─→ WS3 (Lean, all 5 theorems)  [parallel per-theorem: paper first, then encode]
 │    │    └─→ WS5.2 (Lean tarball)
 │    └─→ WS5.1 (Proof appendix)
 │
 ├─→ WS4 (Paper Draft)
 │    ├── §1-§2 can begin during WS1 (intro, examples)
 │    ├── §3-§5 require WS1 + WS2
 │    ├── §6-§7 can proceed independently (implementation, related work)
 │    └─→ WS5.3 (Case studies)
 │
 └─→ WS6 (Submission logistics) [thin tail]
```

WS1 is the root. WS4 §1/§2/§6/§7 are parallelizable with WS1 and WS2.

## Critical Path

WS1 → WS2 + WS3 (parallel per-theorem) → WS4 §3–§5 → WS5 + WS6

The critical path runs through WS3, specifically:

- **WS3.7 (`AdjointTyping.lean`):** the remaining executable admit is still the consolidated AD catch-all, but the blocker is sharper than “generalize the helper and solve `mul`.” `sum` is closed, the old concrete `letBind` / `letpair` routing bug is repaired, and the transform now routes typed cotangent seeds through `pair` / `fst` / `snd` / `copy`. Lean now also contains a handled product-seed counterexample showing that the legacy public `adjointFrom` theorem surface is false because `handle` still uses the tensor-only clause path. So the real path is: switch the public theorem and `tgrad` plumbing to `adjointTypedFrom` / `adjointTypedClausesFrom`, reprove the helper there, then discharge `mul`.
- **WS3.10 (`Preservation.lean`):** the substitution-dependent and context cases are now closed, but the runtime-side invariant is still not in its final form: the current named candidate is `HandlerAwareRuntimeLinear`, and its explicit debt cases are still open. The `tgrad` branch is only as honest as the admitted AD surface it currently imports.
- **WS3.14 (`ADCorrectness.lean`):** the hardest individual file. Requires denotational semantics layer, and full mechanization remains required.

## Workstream Index

| ID | File | Title | Size | Status |
|---|---|---|---|---|
| WS1 | [workstreams/ws1-core-calculus.md](workstreams/ws1-core-calculus.md) | Core Calculus on Paper | Medium | done |
| WS2 | [workstreams/ws2-paper-proofs.md](workstreams/ws2-paper-proofs.md) | Metatheory (Paper Proofs) | Large | in progress |
| WS3 | [workstreams/ws3-lean-mechanization.md](workstreams/ws3-lean-mechanization.md) | Lean 4 Formalization | Large | in progress |
| WS4 | [workstreams/ws4-paper-draft.md](workstreams/ws4-paper-draft.md) | Paper Draft | Large | in progress |
| WS5 | [workstreams/ws5-supplementary.md](workstreams/ws5-supplementary.md) | Supplementary Material | Medium | in progress |
| WS6 | [workstreams/ws6-submission-logistics.md](workstreams/ws6-submission-logistics.md) | Submission Logistics | Small | not started |

Update the status column as work progresses. Suggested values: `not started`, `in progress`, `blocked`, `done`.
