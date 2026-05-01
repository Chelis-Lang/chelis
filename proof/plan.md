# Chelis Proof — Project Plan

## Current Snapshot

- Branch: `chelis-proof`
- Proof build oracle: `cd proof/lean && lake build`
- Current proof state: the branch builds cleanly, the four DB-backed preservation wrappers are closed, `Translation.lean` now exposes the honest lexical forward bridge without admits, `Substitution.lean` is closed under its honest lexical-scoped theorem shape, and the remaining admitted proof surface is concentrated in `AdjointTyping.lean`. The live executable path is still blocked by the `mul` and clause-list `cons` pair-seed holes inside the private typed helper, but the file currently contains four `sorry` sites total because two legacy exact-output proof blocks remain commented rather than deleted.
- `Preservation.lean` is syntactically admit-free on the executable branch, including `tgrad` and `tvmap`, but the `tgrad` branch is not yet an honest closed result because it still imports admitted AD helper cases. The AD transform now has the two necessary structural repairs on branch: the old concrete `E-Grad` `letBind` / `letpair` routing bug is fixed, and product/projection/copy nodes now route typed cotangent seeds rather than a monomorphic tensor seed. Lean still contains three live AD theorem-shape blockers: a handled product-seed counterexample showing that legacy `adjointFrom` is still false because `handle` delegates to the tensor-only clause path, a higher-order counterexample showing that unrestricted `adjointTypedFrom` is false on raw terms, and a concrete `mul` context-gap counterexample showing that the current private shape-only helper is false on arbitrary seed-only contexts because the generated `mul` term replays raw source operands without enough source-context information in the theorem statement. The public theorem surface has already been moved onto the honest existential slot-threaded output shape and now carries explicit `AdjointFreeCtxSupported` and `AdjointTermFresh` premises, with local destructor lemmas for both on the AD branch. The AD critical path is therefore narrower: replace the private shape-only helper with a source-typed supported-domain induction, then close the remaining local `mul` and clause-list `cons` proofs. Separately, the stronger runtime-invariant story for iterating preservation is still unsettled: plain `RuntimeLinear` is false, `ActiveRuntimeLinear` is too weak for direct handled operations, the current recursive `DeepActiveRuntimeLinear` candidate is still not compositional across a beta step followed by contextual `handleOpDirect`, and `RuntimeSafeConfig` still has explicit generic counterexamples. The runtime side is now packaged honestly as an interim boundary: `LinearitySoundness.lean` exports `runtimeSafeConfig_step_or_residualDebt`, and `Preservation.lean` exports `preservation_runtimeSafeConfig_or_residualDebt`, with the older `_or_debt` theorems retained as compatibility wrappers. The remaining theorem-shape work is therefore to eliminate or prove unreachable that explicit residual debt surface, not to rediscover where the runtime proof still breaks.
- `AdjointTyping.lean`'s repaired `handle` branch is now closed, and the public theorem surface is already on the honest existential slot-threaded shape. The remaining AD gap is therefore localized to:
  - replacing the admitted private shape-only helper with a source-typed supported-domain induction that remembers enough source-context information to type raw source operands replayed by `mul`
  - threading the supported-fragment premises (`AdjointTypeSupported`, `AdjointCtxSupported`, `AdjointSupported`) through that honest typed surface
  - reproving the private helper there and closing the remaining local `mul` and clause-list `cons` cases
- `AddDim.lean` is closed, the earlier `tvmap` dimension-choice mismatch was repaired by making the batch dimension explicit in the term syntax, and the remaining `tvmap` preservation proof is now closed
- Immediate critical path:
  1. repair the `mul` replay/tape boundary and the admitted typed AD helper in `AdjointTyping.lean`, so the public typed theorem surface and the `tgrad` preservation branch become honest
  2. reduce the current residual runtime debt surface so the combined `preservation_runtimeSafeConfig_or_residualDebt` theorem can shed more explicit runtime cases, or replace `RuntimeSafeConfig` with a yet-stronger config theorem surface if the debt families prove irreducible
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

- **WS3.7 (`AdjointTyping.lean`):** the remaining executable admits are the `mul` and clause-list `cons` holes inside the private typed helper, but the blocker is sharper than “generalize the helper and solve `mul`.” `sum` is closed, the old concrete `letBind` / `letpair` routing bug is repaired, and the transform now routes typed cotangent seeds through `pair` / `fst` / `snd` / `copy`. Lean now contains the handled product-seed counterexample against legacy `adjointFrom`, the higher-order counterexample against unrestricted `adjointTypedFrom`, and a concrete `mul` context-gap counterexample against the current private shape-only helper. The public typed theorem surface and `tgrad` plumbing are already on `adjointTypedFrom` / `adjointTypedClausesFrom`; the real path is to replace the admitted helper with a source-typed supported-domain induction and then discharge the remaining local `mul` and `cons` proofs.
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
