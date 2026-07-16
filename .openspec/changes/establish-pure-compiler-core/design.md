## Context

The canonical reference requires every compilation stage to remain a pure function, forbids global mutable compiler state, and expects crate APIs to take inputs and return outputs. Most parsing and transformation code follows that model, but several cross-cutting mechanisms do not:

- linked-program provenance, opacity metadata, annotation metadata, recursion state, stack-exhaustion state, and host inlining state use thread-local storage;
- checker acceptance can depend on native remaining-stack state;
- type checking and lowering read profiling environment variables and write timing or fallback messages to stderr;
- fallible lowering catches panics and installs a process-wide panic hook;
- target admission and build sequencing are duplicated in `chelis-cli` and `chelis-compiler-api`;
- the CLI supports Metal while machine APIs currently expose only C and HIP;
- native command construction reads environment variables and probes executables.

The change must retain the existing language, diagnostics, cache compatibility where practical, and generated artifact behavior. It must also keep the default build free of heavyweight optional solver or GPU prerequisites. It follows the boundary, decision-algebra, identity, ordering, and parity laws in `.openspec/FCIS_ARCHITECTURE.md`.

## Goals / Non-Goals

**Goals:**

- Make all semantic compiler inputs explicit and make compiler outputs independent of ambient process and native-stack state.
- Establish one canonical compilation core with distinct checking and building operations.
- Keep measurement, filesystem mutation, environment discovery, process execution, native-command construction, and terminal reporting in adapters.
- Define C, HIP, and Metal target policy once and expose an explicit caller/target capability matrix.
- Preserve exact or mechanically normalized parity for diagnostics and generated artifacts.
- Lock the architecture with executable positive and negative tests.

**Non-Goals:**

- Adopting Salsa as part of this change.
- Rewriting the Surf or Deep parser, type system, IR, or backend algorithms.
- Changing Chelis language semantics or target support intentionally.
- Requiring every public surface to expose every target; intentional capability differences remain explicit.
- Making `chelis build` invoke a native compiler.
- Removing caches; only their placement and identity relative to the pure core change.

## Decisions

### 1. Explicit stage inputs and pinned limits replace ambient guards and host-resource decisions

Each stage receives an owned or borrowed context containing only semantic inputs required by that stage. The type-check context includes source provenance, the base type environment, deterministic recursion/fuel limits, and precomputed program metadata. Lowering contexts own recursion, inlining, fallback, and diagnostic state.

`SemanticLimits::V1` is pinned to `max_type_recursion_depth = 16_384`, `max_lower_recursion_depth = 16_384`, `max_stage_steps = 100_000_000`, and `max_diagnostics = 10_000` for trusted local CLI, compiler API, Python, and acceptance callers. Tide's untrusted default is the explicit `ServiceSemanticLimits::V1`: type and lowering depth 4,096, stage steps 10,000,000, and diagnostics 1,000. A caller may request smaller limits, but compatibility adapters never silently raise them and untrusted service configuration cannot exceed its deployment ceiling. The counters advance by documented structural events rather than allocation count, wall time, or native frames. Limit diagnostics name the exhausted counter and configured value. Focused resource tests bound time and peak working memory for near-limit fixtures so the deterministic ceiling does not become an unmeasured denial-of-service allowance.

Raw and Reef-linked programs enter through distinct preparation paths. Public command and machine APIs do not expose a user-controlled boolean that can mark arbitrary source as trusted linker output. The linked preparation path constructs the provenance value consumed by the checker.

Native remaining-stack measurements remain implementation safety mechanisms only. They must not decide whether a program is semantically accepted. Every recursive path checks the deterministic depth/step counter before the implementation red zone. Legitimately bounded input is accepted or rejected according to explicit limits; a reportable inability to allocate stack growth or fallible working storage is a host execution failure outside the semantic diagnostic set and is never cached as source rejection. Rust allocator aborts that cannot be returned remain outside the API guarantee and are not relabeled through panic catching.

Alternative considered: keep scoped thread-local guards because they restore on drop. This was rejected because results still depend on hidden state and callers can omit guards.

### 2. Core observations are data, not I/O

Compiler stages may return deterministic structural observations such as node counts, pass names, and rejection classifications. They do not read profiling environment variables, inspect clocks, or print. An outer measured adapter may time stage calls and render observations when profiling is enabled.

Alternative considered: inject a logger or clock trait into every stage. This would hide effects behind an interface and complicate caching, so returned observations are preferred.

### 3. Fallible lowering uses typed errors

Expected unsupported or unrepresentable input returns `LowerDiagnostic` or a more specific typed error directly. Panic remains reserved for violated internal invariants. Public fallible lowering APIs do not install or replace the process panic hook. Inlining and recursion bookkeeping are invocation-owned, so every exit path restores state.

Existing infallible convenience functions may remain temporarily as compatibility wrappers that panic on error, but canonical production paths use fallible APIs.

### 4. One compilation core exposes algebraic check and build outcomes

A new dependency-minimal `crates/chelis-compiler-core` crate is the canonical orchestration boundary rather than `chelis-cli`. It accepts prepared in-memory programs plus explicit compiler resource bundles, build capabilities, and semantic options.

- `check_prepared` performs checking and target-independent effect validation and returns `Execution<CheckSuccess, CheckRejection, CompilerHostFailure>`.
- `build_prepared` performs checking, target-effect validation, host/DAG lowering selection, entry-root selection, target admissibility validation, optimization ordering, and backend emission and returns `Execution<BuildSuccess, BuildRejection, CompilerHostFailure>`.

A check operation never performs backend emission and does not require a target. `BuildSuccess` contains a complete artifact manifest and cannot contain error diagnostics. `BuildRejection` contains structured diagnostics and cannot contain a complete artifact manifest. `CompilerHostFailure` is separate from semantic diagnostics and cache entries. These sum types make success/error/artifact contradictions unrepresentable.

Surf and Deep differ only in preparation. Span-attributed inputs may retain different source-map or comment data, so parity uses a specified semantic projection rather than unconditional byte equality.

### 5. Target policy and public capability support are explicit

C, HIP, and Metal each expose one authoritative validator in a pure `policy` module inside its owning backend crate. The policy module is dependency-lower than that backend's emitter: it depends only on shared type/effect/IR data and never on `chelis-compiler-core`. The canonical core depends on and calls those validators before emission; each emitter may defensively call its own crate's same validator. Frontends do not implement target-admissibility conditionals. This yields `compiler-core -> backend::{policy,emit}` with no backend-to-core edge.

The current v1 surface matrix is:

| Surface | Check | C build | HIP build | Metal build | Preflight |
|---|---:|---:|---:|---:|---|
| CLI | yes | yes | yes | yes | formatter + blocking lint before preparation |
| Rust compiler API | yes | yes | yes | no | none; unknown/Metal capability is explicit unsupported |
| Python | yes | yes | yes | no | none; returns structured unsupported target |
| Tide HTTP/MCP | yes | yes | yes | no | none; returns structured unsupported target |
| Acceptance harness | yes | yes | yes | yes | drives prepared core directly and every supported adapter pair |

The acceptance harness is evidence infrastructure, not a public caller. Adding Metal to Rust compiler API, Python, or Tide is out of scope. Their wire/deserialization adapters return stable `unsupported_target` diagnostics for `metal` instead of silently falling back; the Rust capability query reports the exclusion even though its closed v1 target enum cannot construct Metal.

Post-preparation semantic parity is required on the intersection of supported capabilities. End-to-end CLI parity uses canonically formatted, lint-clean source. Style-failing input intentionally stops in the CLI preflight and is covered by separate style-gate tests rather than compared as a core semantic rejection.

### 6. Emission resources and native command planning are separate pure steps

Backend emission returns platform-neutral requirements. It does not read environment variables, probe executables, inspect runtime artifact paths, or decide whether source emission succeeded based on installed host tools.

A `CompilerResourceBundle` supplies every runtime/header/library payload that may enter the artifact manifest as immutable relative-name/bytes/digest data. The bundle is created at the packaging or adapter boundary, is path-traversal and duplicate-name validated before compilation, and participates in the build query key. `BuildSuccess` either contains the required payload bytes in its complete artifact manifest or content-addressed references whose bytes and digests came from that explicit bundle; the filesystem executor cannot substitute ambient runtime files.

Outer discovery adapters resolve `NativeToolchainFacts`. A separate pure `plan_native_command(requirements, facts)` operation may produce compile/link commands or structured unmet-host requirements. Missing host facts cannot erase or invalidate otherwise successful source emission because Chelis build emits source rather than invoking the native compiler.

Compile-time backend capabilities that change emitted bytes, such as math-library support, are explicit `CompilerBuildCapabilities` and participate in the build query key.

### 7. Determinism, query keys, and output digests are explicit

Diagnostics, roots, notices, requirements, and artifact manifests use canonical ordering. A core-computed `PreparedProgramFingerprint` covers the exact prepared input consumed by the core—including canonical module/expression structure, source/provenance attribution capable of changing returned diagnostics, and declared preparation version—under `chelis-fcis/compiler-prepared/v1\0`. It uses the same tagged, length-prefixed, fixed-width encoding discipline and independent golden fixtures as the query key. A caller-supplied prepared fingerprint is never authoritative: the core computes it from the prepared value or validates equality before cache lookup.

`CompilationQueryKey` includes every pre-execution input capable of changing core outputs, including that core-validated prepared-program fingerprint, compiler version, build capabilities, resource-bundle digests, target specification, options, provenance, and deterministic limits. `CompilationOutcomeDigest` exists only for normalized accepted or rejected core outcomes; host failures are neither given a semantic outcome digest nor stored as source decisions. Query and outcome identities use SHA-256 over distinct `chelis-fcis/compiler-query/v1\0` and `chelis-fcis/compiler-outcome/v1\0` domains followed by tagged, length-prefixed canonical fields, fixed-width big-endian integers, and canonically ordered vectors, with checked-in golden vectors decoded by an implementation-independent fixture. Measurement configuration, elapsed duration, cache location, style-preflight reporting, and host discovery results used only for command rendering are excluded from all three identities.

### 8. Exact architecture boundaries are executable but not treated as a purity proof

The designated boundary is all production source in the new `crates/chelis-compiler-core`, `crates/chelis-types`, `crates/chelis-effects`, and `crates/chelis-ir`; the pure target `policy` modules in each backend; C code generation in `chelis-backend-c::{blas,emit,host_emit,memory}` and the production codegen portions of its `lib.rs`; and all production code generation in `chelis-backend-hip` and `chelis-backend-metal`. Surf/Deep parsing, desugaring, style preflight, and standalone structural `chelis validate` preparation are outside this change's newly proven boundary; they remain subject to the canonical pure-stage rule and may receive a separate boundary audit. `chelis-backend-c::toolchain`, compiler API source/wire/cache/filesystem adapters, CLI, Python, Tide, measured wrappers, and native discovery/command execution are outside the core. The core crate may depend on backend policy/codegen APIs but cannot import backend adapter modules, and no backend crate may depend on `chelis-compiler-core`.

Cargo dependency allowlists on `chelis-compiler-core` and the already separated stage/backend crates are the primary architecture enforcement. A checked-in manifest records the exact mixed backend module set that cannot yet be expressed as a crate boundary. Resolved-API checks and fixtures reject the specifically documented direct, alias, re-export, qualified, function-pointer, callback, trait, macro, and `cfg(test)` forms for filesystem, environment, process, network, clock, terminal output, entropy, semantic thread-scheduling, unsafe FFI, global panic hooks, hidden native-resource decisions, and mutable static or thread-local semantic state. The checked-in threat model records the actual mechanism and blind spots and does not claim complete detection of arbitrary procedural-macro expansion, dynamic dispatch, or future Rust syntax.

The gate also checks public core interfaces so effects cannot be hidden behind logger, clock, filesystem, or process traits. Behavioral determinism and parity suites remain authoritative; source/dependency checks are guardrails, not a proof of referential transparency.

### 9. Capability matrices, stage transitions, and evidence have typed owners

One typed surface/operation/target/preflight matrix owns the v1 rows above. CLI, compiler API, Python, Tide, wire schemas, docs, and parity-corpus generation are projections or tripwire-checked consumers. In particular, the machine API's closed target enum and its explicit Metal exclusion are checked against the matrix rather than maintained as unrelated frontend conditionals.

Stage-specific input and output newtypes make canonical transitions explicit. `BuildSuccess` has private construction from a validated complete, duplicate-free, traversal-safe artifact manifest and cannot carry error diagnostics. `CheckRejection`/`BuildRejection` carry nonempty diagnostic collections. Host failure is a separate variant. Limit constructors enforce the trusted and service ceilings, and linked provenance can be constructed only by the trusted preparation path.

Before stage implementation, this change registers stable requirement/scenario/fixture IDs, the exact result projection table, core/adapter/module boundaries, the capability matrix owner, all identity domains/field tags/golden vectors, and the failing `compiler-core --slice stage-purity` runner in the FCIS manifest. Missing fields or a caller-asserted identity block the slice.

## Risks / Trade-offs

- **Risk: Moving checks changes diagnostics or target behavior.** Mitigation: capture caller/target and Surf/Deep parity before extraction and compare through the declared semantic projection.
- **Risk: Explicit contexts become oversized parameter bags.** Mitigation: define stage-specific context types and keep orchestration facts out of lower layers.
- **Risk: Deterministic limits reject previously host-dependent edge cases.** Mitigation: capture deep-input behavior first and specify the supported limit independently from native stack size.
- **Risk: Cache formats embed current context structures.** Mitigation: version cache envelopes, provide clean misses for old formats, and test cached versus uncached semantic parity.
- **Risk: Removing panic recovery exposes latent invariant failures.** Mitigation: convert expected panic sites first and retain a top-level CLI crash boundary only as a last-resort adapter.
- **Risk: One compilation facade becomes a monolith.** Mitigation: compose small pure stages behind check/build operations and keep target validators independently testable.

## Migration Plan

This change lands in independently reviewable slices with compatibility adapters:

1. Add failing architecture, deterministic-limit, capability-matrix, and normalized cross-surface parity tests.
2. Replace checker and lowerer thread-local semantic state with explicit contexts and deterministic limits; move profiling out and remove panic-hook/expected-panic paths. This slice must pass `.venv/bin/python scripts/fcis_gate.py compiler-core --slice stage-purity` before any facade is described as pure.
3. Extract dependency-lower C, HIP, and Metal backend policy modules and lock the v1 target/preflight capability matrix with `.venv/bin/python scripts/fcis_gate.py compiler-core --slice target-policy`.
4. Introduce algebraic check/build outcomes, compiler resource bundles, canonical query/outcome identities, and the canonical facade; migrate Surf and Deep CLI paths. This slice uses `.venv/bin/python scripts/fcis_gate.py compiler-core --slice facade`.
5. Migrate compiler API, Python, and Tide caller/target pairs declared supported and pass `.venv/bin/python scripts/fcis_gate.py compiler-core --slice surfaces`.
6. Move compiler-build capability selection and native toolchain discovery to adapters and make resource-limit regression evidence green.
7. Delete deprecated paths only after every focused command passes compatibility and the final acceptance oracle passes.

Rollback is slice-scoped. A focused command is supporting evidence, not a change-completion claim. Cache schema changes roll back by cleanly missing and rebuilding.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py compiler-core
```

The runner must execute exact-boundary fixtures, pinned local/service deterministic-limit and checker/lowerer isolation tests, near-limit time/working-memory regression checks, the acyclic target-policy dependency check, the target/preflight matrix, resource-bundle substitution tests, algebraic outcome invariants, canonical query-key/outcome-digest and cached/uncached parity, and the declared CLI/compiler API/Python/Tide Surf/Deep corpus against one revision. Success means exit status 0, an empty reported error list, no contradictory success/error/artifact state, no backend-to-core dependency edge, and no unsupported caller/target pair or CLI-only style failure falsely counted as core parity evidence. Crate-local or focused-slice green tests do not replace this oracle.

## Resolved Compatibility Decisions

- Diagnostic codes, stages, severity, expected/got fields, roots, requirements, and stable artifact bytes are exact; free-form explanatory text and source spans compare through the documented normalization until separately promoted to byte-stable fields.
- Infallible lowering wrappers remain for one documented minor-version compatibility window, are not used by production orchestration, and then become test-only or are removed in a separately reviewed API change.
- Metal remains an explicit compiler API, Python, and Tide exclusion in this change; adding it is a separate public API proposal.
