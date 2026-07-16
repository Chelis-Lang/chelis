## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, exact core/adapter/mixed-module boundaries, stable requirement/scenario IDs, explicit polarity, and complete fixture-to-slice-to-final-oracle traces in the FCIS contract manifest
- [ ] 0.2 Register one typed surface/operation/target/preflight matrix owner and tripwire CLI, compiler API, Python, Tide, schema, docs, and acceptance projections
- [ ] 0.3 Register the exact accepted/rejected/host-failure projection table, stage transition types, trusted/service limit constants, and architecture threat-model layers
- [ ] 0.4 Register `compiler-prepared`, `compiler-query`, and `compiler-outcome` identity domains, canonical field tags/order, core-computation/validation rules, and independent golden vectors
- [ ] 0.5 Make `.venv/bin/python scripts/fcis_gate.py compiler-core --slice stage-purity` exist and fail on current ambient state/resource/architecture fixtures
- [ ] 0.6 Pass the owning FCIS contract registration/traceability slice before checker/lowerer implementation

## 1. Lock Current Semantics And Boundaries With Tests

- [ ] 1.1 Add successful and failing Surf/Deep fixtures that compare normalized diagnostics, roots, requirements, and stable artifact bytes
- [ ] 1.2 Add positive and negative C, HIP, and Metal capability tests and the pinned CLI/Rust/Python/Tide/acceptance-harness target and preflight matrix
- [ ] 1.3 Add check-vs-build tests proving check performs no backend emission and requires no target or toolchain facts
- [ ] 1.4 Add checker/lowerer tests for sequential failure cleanup, nested invocation isolation, and concurrent context isolation
- [ ] 1.5 Add equal-input tests across different thread stack sizes plus trusted-local and untrusted-service limit boundary fixtures
- [ ] 1.6 Add near-limit time and peak-working-memory regression fixtures so the 16,384-depth and 100,000,000-step local ceilings are measured rather than treated as free capacity
- [ ] 1.7 Add profiling, compiler-capability, resource-bundle, implementation-independent canonical query-key/outcome-digest golden vectors, cached/uncached, and ambient-runtime-file tests
- [ ] 1.8 Add architecture fixtures for allowed local mutation and forbidden direct, aliased, re-exported, qualified, callback-hidden, trait-hidden, unsafe-FFI, entropy, scheduling, and intentionally classified `cfg(test)` host dependencies
- [ ] 1.9 Add a dependency fixture that fails on any backend-to-`chelis-compiler-core` edge
- [ ] 1.10 Commit all test stubs and verify intended negative cases fail before implementation

## 2. Make Checker And Lowerer State Explicit First

- [ ] 2.1 Introduce stage-specific checker contexts for provenance, type environments, deterministic limits, and precomputed metadata
- [ ] 2.2 Replace linked-program, opacity, annotation, recursion, and stack-exhaustion thread-local semantic state with invocation-owned fields
- [ ] 2.3 Restrict linked provenance construction to trusted preparation paths instead of exposing a user-controlled trust switch
- [ ] 2.4 Decouple semantic acceptance from `stacker::remaining_stack`, enforce selected structural counters before native red zones, and classify reportable allocation/stack-growth failure as host execution failure
- [ ] 2.5 Implement trusted-local `SemanticLimits::V1` and Tide `ServiceSemanticLimits::V1` without silent ceiling increases
- [ ] 2.6 Move host inlining, recursion, fallback, and other lowering bookkeeping into invocation-owned contexts
- [ ] 2.7 Replace expected lowerer panic sites with typed diagnostics and migrate production callers to fallible APIs
- [ ] 2.8 Remove process panic-hook installation and panic-catching from fallible lowering paths
- [ ] 2.9 Retain or deprecate infallible compatibility wrappers without using them in production orchestration
- [ ] 2.10 Replace core-stage profiling environment reads and stderr writes with deterministic structural observations and measured outer adapters
- [ ] 2.11 Make limit, resource-regression, unsupported-input, invariant, cleanup, nested, concurrency, and measured/unmeasured tests pass
- [ ] 2.12 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py compiler-core --slice stage-purity`; no facade may be described as pure before this command is green

## 3. Single-Source Acyclic Target And Surface Policy

- [ ] 3.1 Define structured target capability results in dependency-lower pure `policy` modules inside the C, HIP, and Metal backend crates
- [ ] 3.2 Move C target policy from CLI and compiler API branches to the authoritative backend policy module
- [ ] 3.3 Move HIP target policy from CLI and compiler API branches to the authoritative backend policy module
- [ ] 3.4 Move Metal target policy from CLI branches to the authoritative backend policy module
- [ ] 3.5 Make compiler-core and each emitter call the same owning-backend validator without a backend-to-core dependency edge
- [ ] 3.6 Encode the specified CLI/compiler API/Python/Tide/acceptance-harness target/preflight capability matrix and stable `unsupported_target` behavior
- [ ] 3.7 Route every supported caller-target pair through shared validators and delete duplicated frontend policy conditionals
- [ ] 3.8 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py compiler-core --slice target-policy` and require the complete positive/negative matrix and acyclic dependency check

## 4. Introduce Canonical Check And Build Outcomes

- [ ] 4.1 Create `crates/chelis-compiler-core`, enforce Cargo dependency allowlists, and check in the exact mixed-backend production-module manifest that cannot yet be represented by a crate boundary
- [ ] 4.2 Add explicit prepared input, `CompilerResourceBundle`, compiler build capabilities, semantic options, selected limits, diagnostics, notices, and observations
- [ ] 4.3 Implement target-independent algebraic check success/rejection plus separate `CompilerHostFailure` without backend emission
- [ ] 4.4 Implement algebraic build success/rejection plus separate host failure with canonical checking, target effects, lowering-lane selection, root selection, validation, optimization, and emission ordering
- [ ] 4.5 Include duplicate-free traversal-safe relative artifact names and bytes/content references, runtime requirements, platform-neutral compile/link requirements, roots, notices, and structural success invariants
- [ ] 4.6 Define domain-separated canonical `CompilationQueryKey` and `CompilationOutcomeDigest`, with no semantic digest for host failures
- [ ] 4.7 Define the Surf/Deep and CLI-preflight semantic parity projection for source spans, provenance comments, source maps, diagnostic fields/text, and stable bytes
- [ ] 4.8 Add a filesystem executor adapter that writes successful outcomes without ambient runtime substitution or mutation in the core
- [ ] 4.9 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py compiler-core --slice facade`

## 5. Separate Native Command Planning

- [ ] 5.1 Make backend emission return platform-neutral requirements and explicit resource references without environment, runtime-path, or executable discovery
- [ ] 5.2 Add outer adapters that resolve explicit native toolchain and platform facts
- [ ] 5.3 Add a pure native command planner over requirements and resolved facts
- [ ] 5.4 Prove missing native tools preserve successful emitted source while returning structured unmet host requirements
- [ ] 5.5 Remove environment reads and executable probes from compiler-core and backend-emission paths

## 6. Migrate Public Compilation Surfaces

- [ ] 6.1 Migrate Surf and Deep CLI check paths to canonical checking without target emission
- [ ] 6.2 Migrate Surf and Deep CLI build paths for C, HIP, and Metal
- [ ] 6.3 Migrate compiler API check/build callers for every declared supported target
- [ ] 6.4 Migrate Python and Tide callers for every declared supported target, apply Tide's explicit service limits, and preserve explicit unsupported results for exclusions
- [ ] 6.5 Remove superseded root selection, optimization ordering, backend validation, artifact assembly, and command planning from frontend modules
- [ ] 6.6 Make the normalized cross-surface success/rejection corpus and separate CLI style-preflight corpus pass without counting exclusions as parity
- [ ] 6.7 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py compiler-core --slice surfaces`

## 7. Lock The Architecture

- [ ] 7.1 Implement crate-dependency checks as the primary boundary plus the exact mixed-backend module manifest, resolved forbidden-API checks where available, and documented lexical/dynamic-dispatch/macro blind spots
- [ ] 7.2 Reject the specifically claimed alias, re-export, qualified-path, callback/function-pointer, forbidden-trait, unsafe-FFI, entropy/scheduling, adapter-import, macro, and `cfg(test)` fixtures without claiming arbitrary procedural-macro or dynamic-dispatch completeness
- [ ] 7.3 Verify backend policy/emission modules have no dependency on `chelis-compiler-core`
- [ ] 7.4 Verify measured/unmeasured and cached/uncached paths preserve normalized semantic outputs

## 8. Documentation And Acceptance

- [ ] 8.1 Update active compiler, target, API, cache, and current-state documentation with algebraic outcomes, explicit contexts/resource bundles, local/service v1 limits, target/preflight matrix, identity encoding, parity projection, compatibility windows, and adapter boundaries
- [ ] 8.2 Update executable examples and machine-output fixtures for planner-owned notices or requirements
- [ ] 8.3 Consolidate the already landed focused commands and implement the final `.venv/bin/python scripts/fcis_gate.py compiler-core` acceptance oracle
- [ ] 8.4 Run every focused command before landing its slice; do not claim change completion until the final oracle exits 0 with an empty error list, bounded near-limit resources, and an acyclic core/backend graph
- [ ] 8.5 Run the repository local gate and record any CI-owned workspace evidence required for completion
