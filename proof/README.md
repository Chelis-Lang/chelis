# Chelis Proof

A subproject of the Chelis repo. Distinct from the main Chelis language work; lives entirely under `proof/`.

**Goal:** publish **LaCaDiLE** (Lambda Calculus for Differentiable Linear Effects) at POPL 2027. The target proof package is a core tensor calculus with algebraic effects, linear types, and named-dimension indexing, with five theorems mechanized in Lean 4: type soundness, dimension safety, effect correctness, linearity soundness, and AD correctness.

**Current state:** the Lean root build is green (`cd proof/lean && lake build`), but the metatheory is not complete yet. The active branch's executable admits are now concentrated in exactly one place: `AdjointTyping.lean`. `Substitution.lean` is closed under its honest lexical-scoped theorem shape, and `Preservation.lean` is syntactically admit-free on the executable branch, including the `tgrad` and `tvmap` step cases. On the AD side, two structural corrections are now in place: the Phase 1 `letBind` / `letpair` placeholder recurses through the result-producing body, and `pair` / `fst` / `snd` / `copy` now route typed cotangent seeds rather than a monomorphic tensor seed. That removes the old product/projection false-theorem shape from the transform itself. But the current public AD theorem surface is still not honest: Lean now contains a handled product-seed counterexample showing that `adjointFrom` still routes `handle` through the legacy tensor-only clause path. The next AD step is therefore not just “generalize the helper and solve `mul`.” It is: switch the public theorem surface and `tgrad` plumbing from `adjointFrom` to the staged `adjointTypedFrom` / `adjointTypedClausesFrom`, reprove typing on that typed surface, and then close the separate `mul` tape/effect-row case. On the runtime side, the executable preservation theorem is honest only with an explicit `RuntimeLinear` premise. `Operational.lean` now names the stronger cross-boundary candidate `HandlerAwareRuntimeLinear`, and `LinearitySoundness.lean` proves a partial one-step boundary that either preserves it or exposes explicit debt, but raw beta, direct-handler substitution, and context-freshness counterexamples show that the paper-facing invariant still needs extra typing and store-side side conditions.

**Target venue:** PACMPL Issue POPL 2027
**Submission deadline:** July 9, 2026 (AoE)
**Format:** 25 pages max (excl. bibliography), `acmart` / `acmsmall`, double-blind
**Submission site:** popl27.hotcrp.com
**Post-deadline dates:** Author response Sep 7–10, notification Oct 5, revision Oct 26

## Thesis

Ehrhard and Regnier's differential linear logic established that differentiation and linearity are fundamentally connected — the differential operator is the dual of the exponential modality. LaCaDiLE is the first computational type system that realizes this connection for tensor programs, with algebraic effects providing the additional structure for stochasticity and device management. Chelis is the implementation evidence.

## Where things live

| File | Purpose |
|---|---|
| [`plan.md`](plan.md) | Workstream dependency graph, critical path, and index of all six workstreams with status. Start here. |
| [`decisions.md`](decisions.md) | Locked design decisions with rationale. Every workstream cites this. |
| [`risks.md`](risks.md) | Cross-cutting risk register with mitigations. |
| [`toolchain.md`](toolchain.md) | Local toolchain required for WS3 (elan/Lean, vibe, Leanstral, lean-lsp-mcp). Run `python3 proof/scripts/check_toolchain.py` to verify. |
| [`anonymization.md`](anonymization.md) | Submission-time anonymization checklist. |
| [`post-submission.md`](post-submission.md) | Author response prep, revision plan, artifact evaluation. |
| [`workstreams/`](workstreams/) | One file per workstream (WS1–WS6). |
| [`scripts/`](scripts/) | Python helpers (toolchain checker and its tests). |
| [`paper/`](paper/) | LaTeX sources plus draft section and supplement-outline scaffolds for the POPL paper. |
| [`lean/`](lean/) | Houses the Lean 4 mechanization. Pinned to `leanprover/lean4:v4.29.0` via [`lean/lean-toolchain`](lean/lean-toolchain). This tree is active and is the branch proof oracle. |

## Naming

The **project** is "Chelis Proof". The **calculus** being formalized is "LaCaDiLE". Chelis (the implementation) is referenced in §6 of the paper as evidence that LaCaDiLE's guarantees hold in practice.
