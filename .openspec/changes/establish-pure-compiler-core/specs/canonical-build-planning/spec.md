## ADDED Requirements

### Requirement: One core owns canonical compilation semantics
Chelis SHALL provide one canonical compilation facade for prepared in-memory programs. CLI, compiler API, Python, Tide, and test entry points MUST delegate shared checking and build policy to this facade rather than implementing parallel target validation, lowering-lane selection, root selection, optimization ordering, or backend policy.

#### Scenario: Equivalent supported callers produce equivalent outcomes
- **WHEN** two public callers that declare support for the same operation and target submit semantically equal prepared programs and options
- **THEN** they receive equivalent outcomes under the declared semantic projection

#### Scenario: Frontend cannot bypass target policy
- **WHEN** a caller submits a program containing a target-inadmissible construct
- **THEN** every public frontend supporting that target receives the authoritative target rejection and none obtains an artifact by bypassing the core

### Requirement: Check and build are distinct canonical operations
The compilation facade SHALL expose a target-independent check operation and a target-aware build operation. Checking MUST NOT perform backend emission or require native toolchain facts. Building SHALL use the checked result and canonical build pipeline. Both operations SHALL return algebraic accepted or rejected decisions with a separate host-execution-failure channel so contradictory success, error, and artifact states are unrepresentable.

#### Scenario: Check stops before emission
- **WHEN** a caller requests checking for a valid prepared program
- **THEN** it receives a `CheckOutcome` without backend artifacts, target probing, or native command requirements

#### Scenario: Build performs target policy
- **WHEN** a caller requests a C, HIP, or Metal build
- **THEN** the core applies the selected target's authoritative policy before backend emission

### Requirement: Surf and Deep share post-preparation semantics
Surf and Deep ingestion SHALL have format-specific preparation, after which equivalent prepared programs use the same check or build operation. Equivalence SHALL be evaluated through a documented semantic projection that separates source spans, provenance comments, and source maps from stable semantic outputs.

#### Scenario: Equivalent Surf and Deep inputs agree semantically
- **WHEN** a Surf program and its canonical Deep representation prepare to semantically equal programs
- **THEN** they agree on acceptance, diagnostic codes, selected roots, requirements, and normalized emitted artifacts

#### Scenario: Span metadata remains attributable
- **WHEN** a span-attributed Deep input and an equivalent Surf input differ only in source metadata
- **THEN** parity permits the documented source-map or span-comment difference while preserving equal semantic artifacts and diagnostics

### Requirement: Target capabilities have one authoritative validator
C, HIP, and Metal SHALL each expose one authoritative capability validator or matrix used by every public build surface supporting that target. Validators SHALL return structured acceptance or rejection data and MUST NOT be duplicated in frontend-specific conditionals.

#### Scenario: Supported restricted type path is accepted consistently
- **WHEN** a type or operation satisfies a documented target restriction
- **THEN** every public surface supporting that target accepts it

#### Scenario: Unsupported restricted type path is rejected consistently
- **WHEN** the same type or operation violates the target restriction
- **THEN** every public surface supporting that target rejects it with equivalent structured diagnostics

### Requirement: Public target and preflight support is explicit
Chelis SHALL publish the v1 capability matrix in which CLI supports check and C/HIP/Metal build with formatter/lint preflight; Rust compiler API, Python, and Tide support check and C/HIP build without that CLI preflight and explicitly exclude Metal; and the acceptance harness drives every core target plus supported adapter pair. Cross-surface parity MUST be required only for declared supported operation/target pairs after declared preparation/preflight differences, and unsupported pairs MUST fail explicitly rather than silently falling back.

#### Scenario: Supported pair participates in parity
- **WHEN** the capability matrix declares that a surface supports a target
- **THEN** the acceptance corpus exercises that pair for positive and negative inputs

#### Scenario: Unsupported pair is not counted as evidence
- **WHEN** Rust compiler API, Python, or Tide receives or queries Metal capability
- **THEN** the surface returns or reports stable `unsupported_target` behavior and the acceptance runner does not count that pair as target parity

#### Scenario: CLI style failure is preflight, not core rejection
- **WHEN** noncanonical or blocking-lint source is submitted to CLI and an otherwise equivalent in-memory program is submitted to a machine API
- **THEN** the CLI stops at its documented preflight and the run is not misreported as post-preparation core parity evidence

### Requirement: Build emission is free of host discovery and mutation
The canonical build operation SHALL NOT read environment variables, probe executables, execute subprocesses, inspect the filesystem, write artifacts, or print. Target semantics, compiler build capabilities, and immutable `CompilerResourceBundle` payloads/digests required to change emitted bytes SHALL be supplied explicitly.

#### Scenario: Build does not create artifacts on disk
- **WHEN** a valid program is built successfully by the core
- **THEN** emitted artifact data is returned in memory and no output path is created or modified

#### Scenario: Compiler capability difference is explicit
- **WHEN** two compiler builds can emit different code because of an optional backend capability or runtime resource payload
- **THEN** the capability or resource-bundle digest is explicit and participates in the compilation query key

#### Scenario: Ambient runtime file cannot change output
- **WHEN** a host runtime header or library file changes but the supplied compiler resource bundle does not
- **THEN** the core build outcome remains unchanged and the filesystem adapter cannot substitute the ambient file

### Requirement: Native command planning is separate from build success
Backend emission SHALL return platform-neutral compile/link requirements. Host discovery adapters MAY resolve `NativeToolchainFacts`, and a separate pure command planner MAY combine those facts with requirements. Missing native tools MUST NOT invalidate an otherwise successful source-emission outcome.

#### Scenario: Missing compiler preserves emitted source
- **WHEN** source emission succeeds but no native compiler is discovered
- **THEN** the `BuildOutcome` retains complete emitted artifacts and the command planner reports a structured unmet host requirement

#### Scenario: Equal facts produce equal commands
- **WHEN** native command planning receives equal requirements and equal resolved host facts
- **THEN** it returns equal command arguments and unmet requirements without probing the host

### Requirement: Build outcomes describe complete outputs
A `BuildSuccess` SHALL describe all generated relative artifact names and bytes or content-addressed compiler-resource references, runtime artifact requirements, platform-neutral compile/link requirements, target notices, selected entry roots, and semantic observations needed by outer adapters. Relative names SHALL be duplicate-free and traversal-safe. `BuildSuccess` MUST contain no error diagnostics; `BuildRejection` MUST NOT contain a complete artifact manifest; `CompilerHostFailure` MUST remain distinct from both.

#### Scenario: Successful C outcome is complete
- **WHEN** a C-target build succeeds
- **THEN** the outcome contains C source, header and required runtime payload bytes or verified content references, roots, notices, requirements, and no error diagnostics

#### Scenario: Rejected build has no complete artifact set
- **WHEN** checking, target validation, lowering, optimization, or backend emission rejects the program
- **THEN** `BuildRejection` contains diagnostics and cannot represent a complete successful artifact set

#### Scenario: Host failure is not source rejection
- **WHEN** a reportable implementation resource failure prevents execution of an otherwise in-limit build
- **THEN** `CompilerHostFailure` is returned separately and is not cached as a source diagnostic

### Requirement: Build pipeline ordering is canonical
The build operation SHALL define one ordering for checking, target-effect validation, lowering-lane selection, root selection, target validation, optimization, and backend emission. Public callers MUST NOT reorder or omit required build stages.

#### Scenario: Optimization sees canonical roots
- **WHEN** root-sensitive optimization is enabled
- **THEN** it receives roots selected by the canonical core before backend emission

#### Scenario: Earlier failure prevents later emission
- **WHEN** target validation rejects a prepared program
- **THEN** backend emission is not performed

### Requirement: Cross-surface parity uses an explicit projection
The acceptance suite SHALL compare every supported caller/target pair for successful and failing Surf and Deep programs after declared preparation and preflight differences. The suite SHALL compare diagnostic codes/stages/fields, roots, requirements, and emitted bytes through the documented semantic projection. Stable artifact bytes use byte equality; free-form explanatory text and source-attribution fields use the documented normalization until separately promoted to byte-stable contracts.

#### Scenario: Successful corpus member has parity
- **WHEN** a supported corpus member is built through each available supporting surface
- **THEN** all surfaces agree on acceptance, roots, requirements, and normalized stable artifacts

#### Scenario: Failing corpus member has parity
- **WHEN** an unsupported corpus member is built through each available supporting surface
- **THEN** all surfaces agree on rejection stage, diagnostic code, and structured reason
