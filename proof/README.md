# Chelis Proof

A subproject of the Chelis repo. Distinct from the main Chelis language work; lives entirely under `proof/`.

**Goal:** publish **LaCaDiLE** (Lambda Calculus for Differentiable Linear Effects) at POPL 2027. The target proof package is a core tensor calculus with algebraic effects, linear types, and named-dimension indexing, with five theorems mechanized in Lean 4: type soundness, dimension safety, effect correctness, linearity soundness, and AD correctness.

**Current state:** the Lean root build is green (`cd proof/lean && lake build`), but the metatheory is not complete yet. The remaining admitted proof surface is now concentrated in exactly one file, `AdjointTyping.lean`, and it is best described as two blocker families rather than “two literal `sorry`s”: there are four `sorry` sites total, with the live executable path blocked by the `mul` branch of the private typed helper and the clause-list `cons` pair-seed branch inside that same slot-threaded helper stack, plus two parallel legacy exact-output `sorry`s in commented proof blocks that are no longer on the active theorem path. `Substitution.lean` is closed under its honest lexical-scoped theorem shape, and `Preservation.lean` is syntactically admit-free on the executable branch, including the `tgrad` and `tvmap` step cases. On the AD side, two structural corrections are now in place: the Phase 1 `letBind` / `letpair` placeholder recurses through the result-producing body, and `pair` / `fst` / `snd` / `copy` now route typed cotangent seeds rather than a monomorphic tensor seed. That removes the old product/projection false-theorem shape from the transform itself. The branch also now carries an explicit first-order supported fragment (`AdjointTypeSupported`, `AdjointCtxSupported`, `AdjointSupported`) together with substitution/value closure lemmas in `Substitution.lean`, fresh-name hygiene facts, reusable `AdjointFreeCtxSupported` / `AdjointTermFresh` destructors, the explicit `AdjointFreeCtxSupported` and `AdjointTermFresh` premises on the public typed adjoint theorem, and the slot-threaded public theorem surface for `adjointTypedFrom`. The live AD blocker is now sharper than the earlier public-output counterexample story: Lean still contains a handled product-seed counterexample against legacy `adjointFrom`, a higher-order counterexample against unrestricted `adjointTypedFrom`, and a concrete `mul` context-gap counterexample showing that the current private shape-only helper is false on arbitrary seed-only contexts because the transformed `mul` term replays raw source operands without carrying the source derivation it needs. The public typed theorem surface and the `tgrad` preservation plumbing are already on `adjointTypedFrom` / `adjointTypedClausesFrom`; the remaining AD repair is therefore to replace the private shape-only induction with an honest source-typed/supported one, then close the remaining local `mul` and clause-threading proofs. On the runtime side, the executable typing theorem is still honest only with an explicit `RuntimeLinear` premise, and the public config-level boundary has already been narrowed to `runtimeSafeConfig_step_or_residualDebt` plus `preservation_runtimeSafeConfig_or_residualDebt`, with the older `_or_debt` theorems retained only as compatibility wrappers. That residual split is an interim checkpoint, not a final paper theorem: the remaining runtime work is to eliminate those residual debt families outright or prove them unreachable from well-typed source-reachable configurations, especially for beta/let-style substitution, direct or captured handler substitution, `tgrad`, and contextual replugging across store-changing inner steps.

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
