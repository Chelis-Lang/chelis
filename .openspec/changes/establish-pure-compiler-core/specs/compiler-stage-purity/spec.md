## ADDED Requirements

### Requirement: Compiler stages use explicit semantic inputs
Every designated compiler-core stage SHALL receive all values capable of changing its semantic output through function arguments, immutable input objects, or an invocation-owned stage context. Source provenance, linked provenance, type environments, deterministic limits, recursion state, inlining state, target specifications, and compiler build capabilities MUST NOT be inferred from process-global or thread-local state.

#### Scenario: Repeated execution is deterministic
- **WHEN** a stage is invoked twice with structurally equal semantic inputs and context values
- **THEN** it produces equal canonically ordered semantic outputs and diagnostics

#### Scenario: Provenance changes only through explicit context
- **WHEN** two otherwise equal prepared programs have different explicit provenance values
- **THEN** any checker behavior difference is attributable to that input value and not to an ambient guard

### Requirement: Semantic acceptance is independent of native stack state
Compiler acceptance and user-facing semantic diagnostics SHALL depend on explicit deterministic complexity limits rather than native remaining-stack measurements, thread stack size, or build profile. Trusted local `SemanticLimits::V1` SHALL set type recursion depth to 16,384, lowering recursion depth to 16,384, stage steps to 100,000,000, and diagnostics to 10,000. Tide's untrusted `ServiceSemanticLimits::V1` SHALL set both depths to 4,096, stage steps to 10,000,000, and diagnostics to 1,000; deployment configuration MAY lower but MUST NOT silently exceed its ceiling. Native stack growth and allocation MAY protect implementation execution but MUST NOT silently redefine semantic acceptance within the selected explicit limits.

#### Scenario: Different thread stacks preserve semantics
- **WHEN** the same program and deterministic limits are checked on threads with different stack sizes
- **THEN** they produce equal semantic acceptance and diagnostics

#### Scenario: Explicit complexity limit rejects deterministically
- **WHEN** a program exceeds an explicit supported recursion or complexity limit
- **THEN** every execution rejects it with the same structured limit diagnostic before native stack exhaustion

#### Scenario: Untrusted service uses its lower ceiling
- **WHEN** Tide receives a program accepted under trusted local limits but exceeding `ServiceSemanticLimits::V1`
- **THEN** Tide returns the named deterministic service-limit rejection without raising the ceiling or depending on wall-clock timeout

#### Scenario: Host resource failure is not a semantic rejection
- **WHEN** a fallible host allocation or stack-growth request cannot provide an implementation resource needed to execute an otherwise in-limit stage
- **THEN** the failure is classified through the host-execution-failure channel and is not cached as a semantic source diagnostic

#### Scenario: Unreturnable allocator abort is not relabeled
- **WHEN** Rust terminates the process for an allocation failure that cannot be returned
- **THEN** no panic-catching path relabels it as an ordinary semantic rejection and the returned-result contract makes no false claim to have handled it

### Requirement: Stage state is isolated per invocation
Mutable working state used by checking, lowering, optimization, or code generation SHALL be owned by the current invocation. A failed, nested, or concurrent invocation MUST NOT contaminate another invocation.

#### Scenario: Failure does not leak state
- **WHEN** one lowering invocation exits with a typed error after entering recursion or inlining bookkeeping
- **THEN** a subsequent invocation behaves as if no prior invocation had run

#### Scenario: Concurrent invocations remain independent
- **WHEN** two programs are checked or lowered concurrently with different contexts
- **THEN** each result matches an isolated execution of that program and context

### Requirement: Compiler-core execution has no host side effects
Designated post-preparation compiler-core modules SHALL NOT directly or indirectly obtain capabilities to read or write the filesystem, inspect process environment, execute subprocesses, access a wall clock, emit terminal output, install panic hooks, access a network, obtain entropy, make semantic decisions from thread scheduling, use unsafe FFI, or mutate static or thread-local semantic state. Surf/Deep parsing and desugaring are not newly certified by this change and MUST NOT be counted as acceptance evidence for this boundary.

#### Scenario: Core compilation leaves the host unchanged
- **WHEN** a prepared in-memory program is compiled by a core stage
- **THEN** the only observable result is its returned value, diagnostic, or deterministic structural observation

#### Scenario: Architecture gate rejects a hidden effect
- **WHEN** a negative fixture adds a direct, aliased, re-exported, qualified, callback/macro/trait-hidden forbidden host capability to a designated core module
- **THEN** the architecture gate fails and identifies the forbidden dependency class

#### Scenario: Test-only code is classified intentionally
- **WHEN** a designated source file contains a `cfg(test)` fixture using host I/O
- **THEN** the gate applies its documented test-code policy rather than silently accepting or falsely classifying the production boundary

#### Scenario: Local implementation mutation remains allowed
- **WHEN** a stage mutates invocation-owned local collections without performing a host effect
- **THEN** the architecture gate accepts the implementation

### Requirement: Instrumentation is separated from semantics
Core stages MAY return deterministic structural observations, but SHALL NOT read profiling configuration, inspect elapsed time, or print profiling output. An outer adapter MAY measure a stage and render its returned observations.

#### Scenario: Profiling environment cannot change semantics
- **WHEN** a core stage runs under different values of profiling-related process environment variables
- **THEN** its semantic output and diagnostics are equal

#### Scenario: Outer adapter reports a measurement
- **WHEN** profiling is enabled in an outer adapter
- **THEN** the adapter measures the stage and renders timing without modifying the stage result, compilation query key, or outcome digest

### Requirement: Expected lowering failures are typed results
Every production path that accepts unsupported or potentially invalid lowering input SHALL use a fallible lowering API returning structured diagnostics. Expected user-input failure MUST NOT be implemented by panic and catch, and lowering MUST NOT install or replace the process panic hook.

#### Scenario: Unsupported input returns a diagnostic
- **WHEN** validly parsed input cannot be represented by the selected lowering lane
- **THEN** lowering returns a structured diagnostic without panicking

#### Scenario: Internal invariant violation is not relabeled
- **WHEN** a true internal invariant violation occurs
- **THEN** it is not silently converted into an ordinary unsupported-input diagnostic

### Requirement: Pure and measured adapters preserve semantic results
Compatibility, cache, instrumentation, and CLI adapters SHALL preserve the semantic result of the core operation they wrap. A core-computed or core-validated `PreparedProgramFingerprint` SHALL cover the exact prepared program and source/provenance attribution capable of changing returned core outputs under `chelis-fcis/compiler-prepared/v1\0`; a caller-asserted fingerprint MUST NOT authorize cache lookup without equality validation against the value. `CompilationQueryKey` SHALL include that fingerprint, compiler version/build capabilities, compiler resource-bundle digests, target specification, options, provenance, and deterministic limits while excluding elapsed duration, style-preflight reporting, cache location, and rendering configuration. `CompilationOutcomeDigest` SHALL cover only the normalized accepted or rejected core outcome; host failures SHALL have no semantic outcome digest. Prepared, query, and outcome identities SHALL use SHA-256 with distinct domains followed by tagged, length-prefixed canonical fields, fixed-width big-endian integers, and canonically ordered vectors. Checked-in golden vectors SHALL pin each encoding independently of the Rust implementation.

#### Scenario: Cached and uncached stages agree
- **WHEN** the same prepared program is compiled through cache-hit and cache-miss paths
- **THEN** normalized diagnostics and emitted artifact bytes are identical

#### Scenario: Measurement does not affect cache identity
- **WHEN** only elapsed timing or reporting configuration differs
- **THEN** compilation query keys, outcome digests, and core outputs remain unchanged

#### Scenario: Capability-changing input affects identity
- **WHEN** an explicit compiler build capability or resource-bundle digest can change emitted bytes
- **THEN** changing that input changes the compilation query key
