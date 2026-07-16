## Why

Every active FCIS change currently assumes a pinned Dylint layer, but creation of that layer is bundled into the larger FCIS contract-mechanics change. That makes later boundary specifications depend on tooling whose compilation, diagnostic behavior, fixture coverage, and nightly isolation have not yet been established by an independently accepted prerequisite.

## What Changes

- Add a dependency-locked nested Rust workspace at `tools/dylint` with its own dated nightly toolchain, leaving the repository root and normal gate on stable Rust.
- Add the initial `tools/dylint/fcis-boundaries` Dylint category library, package `chelis-fcis-boundaries`, for config-driven resolved forbidden-reference, core-to-adapter-reference, ambient-state, and capability-bearing-interface diagnostics.
- Define one typed registry for the library's supported diagnostic IDs, configuration schema version, detector classes, and fix capability. Same-owner diagnostics remain passes in this library; later sibling categories require a distinct policy owner or compilation/evidence lifecycle.
- Add allowed-positive and violating-negative UI fixtures for every syntax form the initial diagnostics claim, plus live configuration probes, deterministic JSON diagnostics, malformed/stale configuration failures, and zero-match failures. Fixture evidence remains explicitly scope-bounded and is not a transitive purity proof.
- Establish the reusable fix-safety contract. Initial architectural diagnostics are `NoFix`; any future machine-applicable suggestion must add disposable positive exact-rewrite and negative no-machine-fix fixtures, formatting/compilation, post-fix lint cleanliness, domain parity, second-run idempotence, and tracked-checkout immutability evidence before it can be registered.
- Add a narrow bootstrap acceptance runner, `.venv/bin/python scripts/dylint_gate.py`, with unit tests. Its independently owned `dylint-tooling` oracle verifies the pinned workspace, UI/live probes, JSON output, fix-safety harness, root stable isolation, and checkout immutability without depending on the not-yet-created FCIS manifest runner.
- Make `establish-dylint-tooling` the first prerequisite. `establish-fcis-contract-mechanics` will consume its versioned configuration/diagnostic contract and include its current-revision oracle; all domain FCIS changes remain downstream of contract mechanics.

## Capabilities

### New Capabilities
- `dylint-architecture-enforcement`: Isolated pinned Dylint tooling, category and diagnostic ownership, config-driven architecture diagnostics, adversarial detector/fix fixtures, and the prerequisite acceptance oracle.

### Modified Capabilities

None.

## Impact

This change adds only evidence tooling under `tools/dylint`, its fixture corpus, `scripts/dylint_gate.py`, runner tests, and developer documentation. It introduces no production dependency, language semantic change, runtime API, or user-facing compiler behavior. Root workspace builds remain on stable Rust. The later `establish-fcis-contract-mechanics` change owns FCIS manifest projection, traceability, and domain command plans; this prerequisite owns the Dylint library and proves that its generic detector surface is safe to consume.
