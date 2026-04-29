# Chelis Proof

A subproject of the Chelis repo. Distinct from the main Chelis language work; lives entirely under `proof/`.

**Goal:** publish **LaCaDiLE** (Lambda Calculus for Differentiable Linear Effects) at POPL 2027. The target proof package is a core tensor calculus with algebraic effects, linear types, and named-dimension indexing, with five theorems mechanized in Lean 4: type soundness, dimension safety, effect correctness, linearity soundness, and AD correctness.

**Current state:** the Lean root build is green (`cd proof/lean && lake build`), but the metatheory is not complete yet. The active branch's executable admits are now concentrated in exactly one file: `AdjointTyping.lean`, with two live holes on the executable path (`mul` and clause-list `cons`) inside the private typed helper that feeds the public AD theorem. `Substitution.lean` is closed under its honest lexical-scoped theorem shape, and `Preservation.lean` is syntactically admit-free on the executable branch, including the `tgrad` and `tvmap` step cases. On the AD side, two structural corrections are now in place: the Phase 1 `letBind` / `letpair` placeholder recurses through the result-producing body, and `pair` / `fst` / `snd` / `copy` now route typed cotangent seeds rather than a monomorphic tensor seed. That removes the old product/projection false-theorem shape from the transform itself. But the live typed theorem surface is still not honest yet: Lean contains a handled product-seed counterexample showing that legacy `adjointFrom` still routes `handle` through the tensor-only clause path, a higher-order counterexample showing that unrestricted `adjointTypedFrom` is false on raw terms, a separate `mul` context-gap counterexample showing that the current private typed helper is false on arbitrary seed-only contexts because the transformed `mul` term replays raw source operands, and now a stronger public `mul` output counterexample showing that the current `adjointTypedFrom_preserves_typing` statement is itself false on the real `grad`-style input context because `copy x` tombstones `x` before the theorem's claimed output. The branch now therefore carries an explicit first-order supported fragment (`AdjointTypeSupported`, `AdjointCtxSupported`, `AdjointSupported`) together with substitution/value closure lemmas in `Substitution.lean`. The public typed theorem surface and the `tgrad` preservation plumbing are already on `adjointTypedFrom` / `adjointTypedClausesFrom`; the remaining AD repair is no longer just “restate the helper.” The `mul` replay/tape boundary and the public theorem's output-context claim both need an honest redesign, then the helper can be restated on a source-typed/supported domain and the remaining local `mul` / clause-threading cases can close. On the runtime side, the executable preservation theorem is honest only with an explicit `RuntimeLinear` premise. `Operational.lean` now names the stronger cross-boundary candidate `HandlerAwareRuntimeLinear`, and `LinearitySoundness.lean` proves a partial one-step boundary that either preserves it or exposes explicit debt, but the new config-level `handleOpCtx` counterexample shows that `RuntimeSafeConfig` is also not yet the final public theorem surface. Together with the existing beta, direct-handler substitution, and context-freshness witnesses, that means the runtime story still needs a stronger theorem shape and extra typing/store side conditions before preservation can iterate honestly.

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
