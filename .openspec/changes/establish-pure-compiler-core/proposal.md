## Why

Chelis documents each compilation stage as a pure input-to-output function, but the type checker and lowerer still consume ambient state, perform diagnostics I/O, and mutate process-wide behavior. Build semantics are also duplicated across the CLI and compiler API, allowing public surfaces to drift and making future incremental compilation harder.

## What Changes

- Introduce explicit compiler-stage option and context values for source provenance, limits, and optional instrumentation.
- Remove semantic dependence on thread-local or process-global compiler state.
- Replace core-stage stderr profiling with returned structured observations consumed by outer adapters.
- Remove process-wide panic-hook installation and panic-based expected control flow from fallible lowering APIs.
- Introduce one canonical pure compilation facade used by the CLI, compiler API, Python, Tide, and acceptance harness, with algebraic accepted/rejected outcomes and a separate host-execution-failure channel.
- Move orchestration of backend admissibility, target capability checks, optimization ordering, root selection, and build diagnostics into the canonical compilation core.
- Define C, HIP, and Metal target policy once in dependency-lower pure backend policy modules shared by the core and emitters, avoiding a compiler-core/backend dependency cycle, and publish a capability matrix for public surfaces that intentionally support only a subset of those targets.
- Separate target-independent emission requirements from host-specific native command planning; host discovery remains an outer adapter and cannot determine whether source emission succeeds. Runtime payload bytes or content-addressed compiler resource bundles are explicit build inputs rather than ambient files selected by the shell.
- Make compiler build capabilities and the pinned v1 deterministic semantic resource limits explicit, including removal of native remaining-stack state from semantic acceptance decisions, explicit lower service limits for untrusted callers, and resource-regression coverage for the pinned ceilings.
- Add architecture tests that reject filesystem, environment, process, clock, terminal-output, mutable-global, and hidden host-resource dependencies in designated compiler-core modules.
- Preserve existing accepted programs, diagnostics, and emitted artifacts unless a separately specified correctness fix requires a visible change.

## Capabilities

### New Capabilities

- `compiler-stage-purity`: Defines explicit-input, side-effect-free contracts for post-preparation checking, lowering, optimization, and code-generation stages. Surf/Deep parsing remains governed by the canonical pure-stage rule but is not newly claimed by this change's boundary.
- `canonical-build-planning`: Defines the single target-aware build planning surface shared by every Chelis frontend.

### Modified Capabilities

None.

## Impact

Primary impact is in `chelis-types`, `chelis-ir`, `chelis-backend-c`, `chelis-backend-hip`, `chelis-backend-metal`, the new `chelis-compiler-core`, `chelis-compiler-api`, and `chelis-cli`. Public Rust APIs that currently use ambient guards or perform toolchain discovery may be deprecated in favor of explicit context and adapter APIs. CLI output and generated artifacts require parity tests across Surf and Deep ingestion and across each caller/target pair declared supported by the public capability matrix. The matrix also records style/preflight differences so canonical post-preparation parity is not confused with CLI-only style rejection. Span and provenance differences are compared through an explicit semantic normalization rather than assumed byte-identical. Before implementation, `establish-fcis-contract-mechanics` registers stable coverage IDs, the typed capability matrix, exact result projection, core-computed identity encodings, boundary/threat model, and fail-closed slice/final oracles.
