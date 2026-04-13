# Chelis Proof

A subproject of the Chelis repo. Distinct from the main Chelis language work; lives entirely under `proof/`.

**Goal:** publish **LaCaDiLE** (Lambda Calculus for Differentiable Linear Effects) at POPL 2027. A core tensor calculus with algebraic effects, linear types, and named-dimension indexing, with five theorems mechanized in Lean 4: type soundness, dimension safety, effect correctness, linearity soundness, and AD correctness.

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
| [`paper/`](paper/) | Reserved for LaTeX sources. Empty until WS4 begins. |
| [`lean/`](lean/) | Houses the Lean 4 mechanization. Pinned to `leanprover/lean4:v4.29.0` via [`lean/lean-toolchain`](lean/lean-toolchain). Actual `.lean` sources land here during WS3. |

## Naming

The **project** is "Chelis Proof". The **calculus** being formalized is "LaCaDiLE". Chelis (the implementation) is referenced in §6 of the paper as evidence that LaCaDiLE's guarantees hold in practice.
