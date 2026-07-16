## Why

The active FCIS changes define strong architectural laws, 79 requirements, 200 scenarios, 309 tasks, prerequisite relationships, focused slices, and one final oracle per change. Today those relationships are repeated in prose. OpenSpec strict validation checks artifact shape, but it cannot prove that every scenario has executable coverage, that prerequisite and slice graphs are acyclic and complete, that architecture boundaries match checked manifests, or that a named acceptance command exists. Every active change currently names `scripts/fcis_gate.py`, but the runner and its canonical registry do not yet exist.

Without a machine-readable evidence contract, the migrations can drift independently, omit a negative fixture, accept a stale prerequisite claim, or report a focused slice as completion despite the prose forbidding that conclusion. The evidence machinery should land before domain implementation so every later slice is constrained by the same fail-closed rules.

## What Changes

- Add a versioned machine-readable FCIS change manifest covering stable change/capability IDs, change kind, prerequisite DAG, exact core and adapter boundaries, forbidden capabilities, architecture enforcement and blind spots, protocol/identity scope, public surface matrices, focused slices, final oracle, and requirement/scenario/test traceability.
- Add stable requirement and scenario IDs with explicit positive/negative polarity and require every active scenario to map to an executable test or compile-fail fixture owned by an oracle slice.
- Add a dependency-minimal Rust contract checker that parses the manifests, validates graph and coverage invariants deterministically, and emits a canonically ordered structured report.
- Add `scripts/fcis_gate.py` as the single Python orchestration entry point. It uses one registry of supported oracles and slices, executes argv without a shell, invokes the Rust contract checker and declared evidence commands, and returns a structured report whose successful state has an empty error list.
- Add shared test-only architecture evidence for Cargo dependency allowlists, exact mixed-module manifests, resolved forbidden-API checks where available, negative fixtures, threat-model metadata, and public capability-leak checks. This supplements rather than replaces behavioral determinism, denial, replay, and parity tests.
- Add tripwire tests that keep OpenSpec proposal/design/spec/task claims, capability matrices, prerequisite links, identity domains, and oracle commands in lockstep with the machine-readable manifests.
- Register all active FCIS changes and their planned slices before domain implementation. Unimplemented evidence fails explicitly; it is never treated as skipped or green.

## Capabilities

### New Capabilities

- `fcis-contract-evidence`: Defines the typed, deterministic, fail-closed manifest, traceability, architecture-evidence, and acceptance-oracle contract shared by FCIS migrations.

### Modified Capabilities

None.

## Impact

This change affects `openspec/config.yaml`, `openspec/FCIS_ARCHITECTURE.md`, every active FCIS change artifact, a new dependency-minimal Rust evidence/checker crate, `scripts/fcis_gate.py`, Python tests for the orchestrator, Rust contract/fixture tests, and CI/developer documentation. It does not change Chelis language semantics, compiler outputs, evaluator behavior, proof verdicts, Reef behavior, lint rule meaning, or any production runtime API. The shared machinery is evidence infrastructure only; evaluator, proof, Reef, compiler, and lint retain domain-owned production algebras and do not gain a generic effect/workflow framework.
