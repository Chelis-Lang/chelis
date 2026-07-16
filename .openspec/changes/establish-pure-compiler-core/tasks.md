## 1. Lock Current Semantics And Boundaries With Tests

- [ ] 1.1 Add successful and failing Surf/Deep fixtures that compare normalized diagnostics, roots, requirements, and stable artifact bytes
- [ ] 1.2 Add positive and negative C, HIP, and Metal capability tests and the pinned CLI/Rust/Python/Tide/acceptance-harness target and preflight matrix
- [ ] 1.3 Add check-vs-build tests proving check performs no backend emission and requires no target or toolchain facts
- [ ] 1.4 Add checker/lowerer tests for sequential failure cleanup, nested invocation isolation, and concurrent context isolation
- [ ] 1.5 Add equal-input tests across different thread stack sizes plus v1 16,384-depth, 100,000,000-step, and 10,000-diagnostic limit boundary fixtures
- [ ] 1.6 Add profiling, compiler-capability, resource-bundle, query-key, outcome-digest, cached/uncached, and ambient-runtime-file tests
- [ ] 1.7 Add architecture fixtures for allowed local mutation and forbidden direct, aliased, re-exported, qualified, callback-hidden, trait-hidden, and intentionally classified `cfg(test)` host dependencies
- [ ] 1.8 Commit all test stubs and verify intended negative cases fail before implementation

## 2. Single-Source Target And Surface Policy

- [ ] 2.1 Define structured target capability results for C, HIP, and Metal beside their owning backend or core target modules
- [ ] 2.2 Move C target policy from CLI and compiler API branches to the authoritative validator
- [ ] 2.3 Move HIP target policy from CLI and compiler API branches to the authoritative validator
- [ ] 2.4 Move Metal target policy from CLI branches to the authoritative validator
- [ ] 2.5 Encode the specified CLI/compiler API/Python/Tide/acceptance-harness target/preflight capability matrix and stable `unsupported_target` behavior
- [ ] 2.6 Route every supported caller-target pair through shared validators and make unsupported pairs fail explicitly
- [ ] 2.7 Delete duplicated frontend policy conditionals and make the complete positive/negative matrix pass

## 3. Introduce Canonical Check And Build Outcomes

- [ ] 3.1 Create `crates/chelis-compiler-core`, enforce the specified dependency boundary, and check in the exact transitive production-module manifest
- [ ] 3.2 Add explicit prepared input, `CompilerResourceBundle`, compiler build capabilities, semantic options, pinned v1 limits, diagnostics, notices, and observations
- [ ] 3.3 Implement target-independent algebraic check success/rejection plus separate `CompilerHostFailure` without backend emission
- [ ] 3.4 Implement algebraic build success/rejection plus separate host failure with canonical checking, target effects, lowering-lane selection, root selection, validation, optimization, and emission ordering
- [ ] 3.5 Include duplicate-free traversal-safe relative artifact names and bytes/content references, runtime requirements, platform-neutral compile/link requirements, roots, notices, and structural success invariants
- [ ] 3.6 Define `CompilationQueryKey` and normalized `CompilationOutcomeDigest`
- [ ] 3.7 Define the Surf/Deep and CLI-preflight semantic parity projection for source spans, provenance comments, source maps, diagnostic fields/text, and stable bytes
- [ ] 3.8 Add a filesystem executor adapter that writes successful outcomes without ambient runtime substitution or mutation in the core

## 4. Separate Native Command Planning

- [ ] 4.1 Make backend emission return platform-neutral requirements and explicit resource references without environment, runtime-path, or executable discovery
- [ ] 4.2 Add outer adapters that resolve explicit native toolchain and platform facts
- [ ] 4.3 Add a pure native command planner over requirements and resolved facts
- [ ] 4.4 Prove missing native tools preserve successful emitted source while returning structured unmet host requirements
- [ ] 4.5 Remove environment reads and executable probes from compiler-core and backend-emission paths

## 5. Migrate Public Compilation Surfaces

- [ ] 5.1 Migrate Surf and Deep CLI check paths to canonical checking without target emission
- [ ] 5.2 Migrate Surf and Deep CLI build paths for C, HIP, and Metal
- [ ] 5.3 Migrate compiler API check/build callers for every declared supported target
- [ ] 5.4 Migrate Python and Tide callers for every declared supported target and preserve explicit unsupported results for exclusions
- [ ] 5.5 Remove superseded root selection, optimization ordering, backend validation, artifact assembly, and command planning from frontend modules
- [ ] 5.6 Make the normalized cross-surface success/rejection corpus and separate CLI style-preflight corpus pass without counting exclusions as parity

## 6. Make Checker And Lowerer State Explicit

- [ ] 6.1 Introduce stage-specific checker contexts for provenance, type environments, deterministic limits, and precomputed metadata
- [ ] 6.2 Replace linked-program, opacity, annotation, recursion, and stack-exhaustion thread-local semantic state with invocation-owned fields
- [ ] 6.3 Restrict linked provenance construction to trusted preparation paths instead of exposing a user-controlled trust switch
- [ ] 6.4 Decouple semantic acceptance from `stacker::remaining_stack`, enforce the pinned structural counters before native red zones, and classify reportable allocation/stack-growth failure as host execution failure
- [ ] 6.5 Move host inlining, recursion, fallback, and other lowering bookkeeping into invocation-owned contexts
- [ ] 6.6 Replace expected lowerer panic sites with typed diagnostics and migrate production callers to fallible APIs
- [ ] 6.7 Remove process panic-hook installation and panic-catching from fallible lowering paths
- [ ] 6.8 Retain or deprecate infallible compatibility wrappers without using them in production orchestration
- [ ] 6.9 Make limit, unsupported-input, invariant, cleanup, nested, and concurrency tests pass

## 7. Isolate Instrumentation And Lock Architecture

- [ ] 7.1 Replace core-stage profiling environment reads and stderr writes with deterministic structural observations
- [ ] 7.2 Add measured outer adapters that inspect clocks and render observations
- [ ] 7.3 Implement architecture checks for source aliases, re-exports, qualified paths, callbacks/function pointers, forbidden capability traits, crate dependency boundaries, and intentional `cfg(test)` body policy
- [ ] 7.4 Verify measured/unmeasured and cached/uncached paths preserve normalized semantic outputs

## 8. Documentation And Acceptance

- [ ] 8.1 Update active compiler, target, API, cache, and current-state documentation with algebraic outcomes, explicit contexts/resource bundles, v1 limits, target/preflight matrix, identity taxonomy, parity projection, compatibility windows, and adapter boundaries
- [ ] 8.2 Update executable examples and machine-output fixtures for planner-owned notices or requirements
- [ ] 8.3 Implement `.venv/bin/python scripts/fcis_gate.py compiler-core` as the named acceptance runner
- [ ] 8.4 Run `.venv/bin/python scripts/fcis_gate.py compiler-core` and require exit 0 with an empty error list
- [ ] 8.5 Run the repository local gate and record any CI-owned workspace evidence required for completion
