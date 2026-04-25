# Chelis Proof — Project Plan

## Current Snapshot

- Branch: `chelis-proof`
- Proof build oracle: `cd proof/lean && lake build`
- Current proof state: the branch builds cleanly, the four DB-backed preservation wrappers are closed, `Translation.lean` now exposes the honest lexical forward bridge without admits, and the remaining executable admits are concentrated in `Substitution.lean`, `AdjointTyping.lean`, and `Preservation.lean`
- `Preservation.lean` is no longer blocked on generic context/store plumbing or `tvmap`; the remaining executable preservation admit is `tgrad`. Separately, the stronger runtime-invariant story for iterating preservation is still unsettled: plain `RuntimeLinear` is false, `ActiveRuntimeLinear` is too weak for direct handled operations, and the current recursive `DeepActiveRuntimeLinear` candidate is still not compositional across a beta step followed by contextual `handleOpDirect`
- `Substitution.lean`'s current named substitution theorem shape is now also known to be false on arbitrary named derivations without a lexical or derivation-guided restriction
- `AddDim.lean` is closed, the earlier `tvmap` dimension-choice mismatch was repaired by making the batch dimension explicit in the term syntax, and the remaining `tvmap` preservation proof is now closed
- Immediate critical path:
  1. settle the stronger cross-boundary handler-aware runtime invariant
  2. close named substitution (`weakening_insert`, `subst_preserves_typing`)
  3. resolve the `Accum` signature / adjoint-typing contradiction for `tgrad`

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

- **WS3.6 (`Substitution.lean`):** always the most painful mechanized PL proof. Linear context splitting makes it worse.
- **WS3.10 (`Preservation.lean`):** the substitution-dependent and context cases are now closed, but the runtime-side invariant is still not in its final form.
- **WS3.14 (`ADCorrectness.lean`):** the hardest individual file. Requires denotational semantics layer.

If WS3.14 proves intractable, the fallback is: state Theorem 5 in Lean, `sorry` the proof, provide the paper proof in the appendix, and note this in the README. This is honest and still a strong paper — four fully mechanized theorems plus a paper proof of the fifth is a substantial contribution. But attempt the full mechanization first.

## Workstream Index

| ID | File | Title | Size | Status |
|---|---|---|---|---|
| WS1 | [workstreams/ws1-core-calculus.md](workstreams/ws1-core-calculus.md) | Core Calculus on Paper | Medium | done |
| WS2 | [workstreams/ws2-paper-proofs.md](workstreams/ws2-paper-proofs.md) | Metatheory (Paper Proofs) | Large | in progress |
| WS3 | [workstreams/ws3-lean-mechanization.md](workstreams/ws3-lean-mechanization.md) | Lean 4 Formalization | Large | in progress |
| WS4 | [workstreams/ws4-paper-draft.md](workstreams/ws4-paper-draft.md) | Paper Draft | Large | in progress |
| WS5 | [workstreams/ws5-supplementary.md](workstreams/ws5-supplementary.md) | Supplementary Material | Medium | not started |
| WS6 | [workstreams/ws6-submission-logistics.md](workstreams/ws6-submission-logistics.md) | Submission Logistics | Small | not started |

Update the status column as work progresses. Suggested values: `not started`, `in progress`, `blocked`, `done`.
