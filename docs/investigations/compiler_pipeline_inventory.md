# Compiler Pipeline Inventory

This inventory records the current production front-end sequences and ownership boundaries.

The scope covers these consumers:

- compiler API
- CLI
- edit tools
- caches
- contexts
- E2E tools
- shared Deep verification consumers
- the structural Deep validator
- Deep lint and trace consumers
- Deep authoring and opaque-value decode consumers
- lenient Deep diagnostics and typed macro expansion

The guarded scope also includes `chelis-pipeline-core` and `chelis-reef`. Tests can call individual stages for focused evidence.

## Baseline Sequences

| Consumer | Baseline phase boundary | Prior owner |
|---|---|---|
| Compiler API `check` | Parse, expand, fitness | `compiler.rs` |
| Compiler API compile, lower, and eval | Parse, expand, type, effects, linearity, lower | `compiler.rs` |
| Contextual eval and check | Rewrite, expand, contextual type, effects, linearity, contextual lower | `compiler.rs` |
| Whole-module edit validation | Fitness, typed check, effects, linearity | `fragment.rs` |
| Reef context construction | Expand, one-session context type check, effects, linearity, library lower | `context.rs` |
| Standard library cache construction | Expand, one-session context type check, effects, linearity, optional library lower | `stdlib_cache.rs` |
| Layered CLI check | Expand, contextual type, effects, linearity | `layered.rs` |
| Layered CLI build | Expand, contextual type, effects, linearity, compose | `layered.rs` |
| CLI Surf check fallback | Expand, fitness, typed check, effects, linearity | `main.rs` |
| CLI Deep check | Strict parse, fitness, typed check, effects, linearity | `main.rs` |
| CLI build helper | Type, effects, linearity | `main.rs` |
| CLI typed lint gate | Parse, expand, fitness, typed check, effects, linearity | `main.rs` |
| E2E `compile_surf` | Parse, expand, type, effects, linearity, lower, root reconstruction | `pipeline.rs` |
| E2E snippet checker | Parse, expand, fitness | `check_snippet.rs` |

The CLI check paths ran inference twice. They called `check_ir_fitness` and `check_typed_program` for the same Deep program.

## Canonical Ownership

`chelis-pipeline-core` owns these transitions in the guarded scope:

- isolated and contextual type analysis
- effect checks
- linearity checks
- isolated and contextual lowering
- root metadata

`chelis_compiler_api::pipeline` owns source preparation, dynamic goal selection, cancellation, host policy, and backend policy.

Adapters retain presentation and policy. The CLI retains style checks, Reef preparation, JSON, exit codes, target selection, and backend emission.

Shared Deep verification consumers retain typed parser ingress. One complete transitional bridge supplies their list-based dispatch.

The Deep validator uses one complete tagged-list view for transitional and typed representations.

Deep lint uses the same representation discipline. Trace tests visit metadata and children in every typed expression variant.

Deep authoring reads typed declaration tags, binders, and metadata. Opaque-value decode reads typed field types, invariants, constants, and field names.

Valid generic ADTs retain typed declarations. The lenient fragment boundary retains malformed input for checker diagnostics.

The macro expander reads internal definitions and typed variable calls.

The source guard checks production functions in exactly these trees:

- `crates/chelis-pipeline-core/src`
- `crates/chelis-compiler-api/src`
- `crates/chelis-reef/src`
- `crates/chelis-cli/src`
- `crates/chelis-e2e/src`

The guard builds one callable inventory for all guarded files. It propagates stage sets through crate-local helpers, imports, aliases, and higher-order calls.

Canonical stage identities include the owning module and function. Typed receiver bindings resolve crate-local methods without matches for unrelated receiver types.

The import map includes `use` and `extern crate` aliases. Local type aliases resolve before receiver method lookup.

Each execution path retains its abstract values. Callable branch results, stable Boolean values, and mutually exclusive assignments remain separate.

The guard calculates a fixed point across unknown loop iterations. It retains exact values and counts for known array iterables.

The path model tracks zero-iteration exits, returns, labeled breaks, and continues.

Branch and loop patterns create lexical bindings. Local macros resolve canonical imports from their definition scope and lexical block.

The guard ignores uninvoked callable bodies. Mutually exclusive execution paths remain valid.

The guard classifies direct lower-stage calls. It treats core transition calls as adapter delegation.

## Dependency Boundary

`chelis-pipeline-core` is below `chelis-compiler-api` and `chelis-reef` in the dependency graph.

The core depends directly on `chelis-deep`, `chelis-types`, `chelis-effects`, and `chelis-ir`. A dependency guard checks the manifest and resolved graph.

The approved transitive closure includes `chelis-axis-core`, a leaf containing
ordered-axis algorithms shared by the checker and IR. Its ordinary Rust build
uses the proof-erasing `vstd` macros; the Verus, Kani, and Vermilion executables
remain opt-in development tools. This leaf does not authorize additional direct
core dependencies or dependencies on compiler adapters, shells, or backends.

The dependency guard, the documentation guard, and the pipeline-artifact compile-fail fixture run in the per-PR gate `lint-and-unit` stage, so hosted CI enforces the dependency boundary, the no_std documentation contract, and the facade artifact boundary. The manual `compiler_pipeline_oracle.py` still runs the same three controls.

The package artifact and schema paths pass fully linked expanded Deep to the core. `checked_program_with_effects` retains only the Reef error adapter.

The Reef exception no longer exists. This inventory does not claim whole-workspace source-guard coverage.

## Final Artifact Boundaries

| Artifact | Constructor owner | Consumers |
|---|---|---|
| `ValidatedModule` | `check_whole_module_edit` | Edit tools and compiler API schema adapters |
| `CheckedLibrary` | Core library semantic completion or cache parser | Contextual compiler paths and layered checks |
| `ContextualTypeAnalysis` | Core contextual type check | Contextual semantic completion |
| `ContextualLibraryTypeAnalysis` | Core library-extension type check | Bound library composition |
| `AllRootNames` | Core root analysis | Compiler API evaluation and host output adapters |
| `TensorRootNames` | Core root analysis | Lower policy, CLI root selection, and compiler API evaluation |
| `NamedRoots` | Core exact name-to-DAG alignment | Compiler API schema adapters, evaluation, and E2E tools |
| `ForwardNodeIndex` | Core `NamedRoots` plus DAG load aliases | Compiler API gradient adapters |
| `LoweredParts` | Core `LoweredCompilation::into_parts` | Compiler API and E2E consumers |
| `SemanticRejection` | Core `complete_checks` | Full pipeline adapters, edit validation, layered checks, and CLI checks |
| `DiagnosticCheckpoint` | `DiagnosticSink::checkpoint` | Body inference in `chelis-types` |
| `LayeredCheck` | Layered compiler checks | CLI report assembly and linked proof checks |

`ValidatedModule` replaces a public Boolean and a raw edited module. The proof owns the exact expressions that passed all semantic checks.

The four root products use `IrName`. Every successful DAG-backed result uses one exact alignment constructor.

A selected host-backend result or nonfatal lower rejection uses one explicit empty constructor.

`LoweredParts` gives names to the checked state, DAG, declared roots, and forward index. This product removes positional root-map swaps.

`SemanticRejection` contains only effect and linearity failures. Full pipeline owners convert it to `PipelineRejection` at their error boundary.

`LayeredCheck` uses exclusive variants. A value cannot contain effect errors and linearity errors at the same time.

## Rust API Migration

| Old API | New API |
|---|---|
| `report.checks_clean` | Successful construction of `ValidatedModule` |
| `report.rewritten_module` | `report.validated_module.as_exprs()` |
| `check_whole_module_edit(...).rewritten_module` | `check_whole_module_edit(...).as_exprs()` |
| `RootMetadata::all_names() -> &[String]` | `RootMetadata::all_names() -> &AllRootNames` |
| `RootMetadata::tensor_names() -> &[String]` | `RootMetadata::tensor_names() -> &TensorRootNames` |
| Raw root maps | `NamedRoots` and `ForwardNodeIndex` |
| Four-element lowered tuple | `LoweredParts` with named fields |
| `complete_checks -> PipelineRejection` | `complete_checks -> SemanticRejection` |
| Parallel layered error vectors | Exclusive `LayeredCheck` variants |
| `iter_from(usize)` | `iter_since(DiagnosticCheckpoint)` |

Call `IrName::as_str` or `IrName::into_string` only at an existing string boundary. Keep schema maps and JSON fields unchanged.

## Adversarial Review

A fresh local red-team reviewed the specifications, code, tests, public API, and completed oracle evidence.

A later pull request review found three actionable defects:

- The branch did not compile against the target typed Deep AST.
- The root path did not distinguish selected host-backend success from a nonfatal fallback.
- Helper calls hid duplicated semantic stage sequences from the source guard.

The first remediation used `Atom::Name` and retained the #912 realizability manifest observation. It also added an explicit root-binding mode and helper propagation.

A second fresh red-team rejected the stale target evidence. It found these additional defects:

- The branch did not include the latest typed `.dp` ingestion and wire bridge.
- The root collector ignored typed `Node` values.
- Repeated calls to one conditional helper lost call-site multiplicity.
- Qualified local helper calls did not resolve.
- An unrelated receiver method with a stage name created a false finding.

The second rebase retained typed `.dp` ingestion, the complete wire bridge, and the #912 observation.

The root collector now reads `List` and `Node` values through one tagged-child view. A parity test covers ordered tuple-root names.

The second remediation preserved each call site and added module paths to function identities. Tests covered all supported local path qualifiers.

Direct stage classification required a known stage path or imported alias. An unrelated receiver or external path did not create a stage.

A secondary automated critique claimed that the host-backend and #912 positive tests were absent. This high-severity claim was rejected.

The artifact tests cover typed names, selected host success, and accepted nonfatal rejection. The target CLI test also builds the host artifact.

The critique also claimed that the helper graph affected root modes. This claim was rejected because the guard and lowering code share no data path.

A third fresh red-team found false negatives across files, aliases, traits, and macros. It also found false reports from returns and uninvoked callable bodies.

The third remediation added one workspace inventory and full import targets. It resolves named imports, glob imports, function aliases, trait defaults, and direct stage macros.

The path model now terminates returned branches. Invoked nested functions and closures contribute stages, but uninvoked bodies do not.

The review also found that the oracle omitted the real CLI host-backend test. The oracle now runs that test directly.

A fourth fresh red-team found typed and parenthesized alias bypasses. It also found lost branch assignments and higher-order callable arguments.

The same review found a stale alias after lexical shadowing. It also found missing compile-fail evidence for raw diagnostic offsets.

The fourth remediation moved alias bindings onto execution paths. It now substitutes callable arguments only when a local helper invokes its parameter.

The remediation also added the raw-offset compile fixture to the oracle. The CLI Deep-check documentation now names the current compiler-API pipeline.

A fifth fresh red-team found three source-guard bypasses:

- A shared function name merged type and effect stage identities.
- A typed local receiver hid a higher-order method call.
- A loop body return deleted the zero-iteration exit path.

The fifth remediation classifies full stage identities and resolves methods from typed receiver bindings. Qualified-method calls preserve callable argument positions.

The path model now keeps separate zero-iteration exits for `while` and `for` loops. A body return terminates only its applicable path.

A sixth fresh red-team found four source-guard defect classes:

- Repeated loop iterations did not compose reachable stages.
- Break and continue did not isolate unreachable statements.
- Branch patterns did not shadow outer aliases.
- Imported stage aliases inside local macros did not resolve.

The literal repeated-loop probes used an invariant Boolean. Those probes did not establish a real two-stage path.

Equivalent probes with a varying branch condition confirmed the loop defect. Permanent tests use the corrected varying condition.

The sixth remediation adds a loop fixed point and distinct flow states. It preserves labeled control targets and lexical pattern scopes.

The macro inventory now resolves module and block imports from each macro definition scope.

A seventh fresh red-team found four source-guard defect classes:

- Branch-result callables and callable iterable elements did not resolve.
- `extern crate` aliases and local receiver type aliases did not resolve.
- Nested macro definitions did not retain lexical block identity.
- Stable loop conditions and known array counts created impossible paths.

One reported macro positive control invoked a type-stage macro before an effect stage. That program contained a real duplicate sequence.

The corrected positive control does not invoke the inner type-stage macro. The negative control confirms that lookup restores the outer macro.

The seventh remediation adds one abstract-value resolver for expressions, arguments, conditions, and iterables.

It preserves callable branch results, stable Boolean values, and known array elements on each path.

The import inventory now includes `extern crate` aliases and local type aliases. Macro keys now include lexical block identity.

That historical post-remediation red-team remained pending at this checkpoint.

## Acceptance Evidence

### Local Evidence

The earlier oracle runs occurred before the latest target rebase. They do not satisfy final acceptance.

Current focused tests passed for typed roots, full stage identities, receiver methods, loop exits, aliases, higher-order calls, wire shape, and typed `.dp` rejection.

The format check, strict validation, target all-target check, and authoritative oracle passed after the second remediation.

The authoritative oracle passed again after the third source-guard remediation. It included the real CLI host-backend regression test.

The oracle passed after the fourth remediation. It included the path-alias tests, higher-order tests, and raw-offset compile fixture.

All 54 focused source-guard tests passed after the fifth remediation. The authoritative oracle also passed on that tree.

All 65 focused source-guard tests passed after the sixth remediation. The authoritative oracle passed after the current target rebase.

All 74 focused source-guard tests passed after the seventh remediation.

The authoritative oracle passed on target `ad63fa0600737dd656a4f3ece15f43695ce89902`. The implementation commit was `d7844ae5204a31127e4de91b50f392c8622a10cd`.

The current target then changed known Deep parser output from `List` to `Node`. The local gate found that the shared property runner silently skipped typed properties.

A crate run also found that a constructor probe lost all accepted samples. Current `origin/main` reproduced both defects.

The correction retains typed parsing and uses one heap worklist for complete transitional list conversion. Property, obligation, constant, and constructor paths use this boundary.

The next local gate found that structural identity checks accepted reopened modules and reserved linker names. Current `origin/main` reproduced both failures.

The validator now uses a complete view of typed `Node` values. Its nested-bind comment fixture now has the required typed arity.

A full CLI run then found four more target-integration defects. Deep lint skipped typed nodes, two formatter fixtures used invalid old arity, and trace collection skipped nested node spans.

Deep lint uses one borrowed representation view. Trace collection visits every typed expression variant.

The formatter fixtures now use valid typed declaration arity. CLI validation retains its existing fragment diagnostics.

The oracle runs typed property, verification, validation, lint, formatting, and trace tests. It also runs the CLI Deep-property parity test.

The next local gate found two typed authoring failures. Authoring rejected typed declarations, and insertion used a raw-list offset for a typed module.

A baseline run on current `origin/main` reproduced five authoring failures and six opaque-value decode failures.

The authoring boundary now accepts both representations. The insertion helper uses a declaration-relative index and then applies the representation offset.

The list-only decode collectors omitted field types, invariants, constants, and field names.

Opaque-value decode now uses one borrowed representation view. Its positive, negative, and fail-closed tests pass for typed `Node` programs.

The next gate found that strict parsing accepted invalid tags after typed fallback construction. The failures covered an unknown tag and a known tag without metadata.

The Deep validator now reports both invalid top-level forms. Its positive control retains nested structural lists.

The next gate found two generic ADT failures because the typed parser treated parameter lists as type expressions.

A complete type-checker run found 12 more failures. The failures covered malformed diagnostic fixtures, unknown-tag text, and typed macro calls.

Canonical ADT parameter lists now use syntax roles. The lenient fragment boundary retains malformed structures for checker diagnostics.

The macro expander now reads internal fallback definitions and typed variable callees.

All 1,013 `chelis-types` tests, 212 `chelis-deep` tests, six `chelis-macros` tests, and 149 `chelis-e2e` tests passed after these corrections.

The expanded authoritative oracle passed after all target corrections.

The current-target local gate passed for all nine changed crates.

That historical review remained pending at this checkpoint.

The first local gate found that blanket empty-DAG alignment broke a valid C host-backend program. The revised typed policy preserves that build.

The second gate passed that test. It then found a stale canary for the target typed-parser diagnostic.

The isolated canary passed after its accepted set included the target parser message. Later tests still expected pre-#912 unlabeled roots.

Those tests now require the target `[05-OBS-6]` labels. The earlier local gate passed for all four changed crates.

The earlier target stopped on three `infer_recursion_depth_guard` tests with signal 10. The old `origin/main` reproduced those failures.

Current `main` includes the stack-safe normalization fix from #1035. All five `infer_recursion_depth_guard` tests now pass after the rebase.

### Hosted Evidence

All 13 hosted checks passed for exact head `3c133f2f98dfa5420f845637e47570dcedac7c6e`.

The `Lint and Unit Tests (Linux)` job in run `30768292087` executed both new controls. The compiler-API command passed one regular doctest and five compile-fail doctests.

The raw-checkpoint command reported `diagnostic checkpoint compile-fail: PASS`. The same hosted run also passed macOS Smoke, Docs, and Integration Tests.

### PR Review Remediation

A PR review found that the active ownership requirement exceeded the source guard scope. It also found that hosted CI did not execute two compile-fail controls.

At that time, the ownership requirement named three guarded roots and the Reef dependency exception. A source-guard test locked that root set.

The canonical gate now runs compiler-API doctests and the raw-checkpoint fixture. Gate unit tests lock both commands in local and hosted stages.

The compiler-API doctests passed five compile-fail tests and one regular doctest. The checkpoint fixture passed with its exact diagnostic checks.

All 29 gate unit tests passed. The complete local gate and authoritative compiler pipeline oracle passed after the correction.

### Functional pipeline core extraction

The Reef parity baseline records source revision `e1065d94fbd7a41f086e0690929a66c2335accc5`.
The capture used the Reef pipeline before the core extraction.
The accepted capture records exact artifacts, schema output, and hashes.
The rejected capture records complete type, effect, and linearity errors.

The canonical gate now runs doctests for `chelis-compiler-api` and `chelis-pipeline-core`.
It also runs the raw checkpoint fixture.
Command-list tests lock all three controls in local and hosted stages.

A fresh local red team found three enforceable boundary defects.
The composition hook permitted safe construction with an unchecked library.
The dependency guard omitted direct build dependencies.
The documentation guard accepted broader false portability claims.

The first correction made the core composition hook an unsafe adapter boundary.
That correction blocked accidental safe calls, but it did not encode the library relationship.

The core now uses `CheckedLibrary`, `ContextualTypeAnalysis`, and `ContextCheckedCompilation`.
Each composable contextual product retains the exact library that produced it.
Safe composition consumes that bound product and accepts no replacement library.
Contextual lowering also requires the identity from that library.

`ContextualLibraryTypeAnalysis` also retains the combined type environment from the same type session.
Its completion function creates the composed `CheckedLibrary` without a caller-supplied environment.

The core no longer exports the unchecked `prepared_analysis_from_checked` adoption helper.
It also keeps raw contextual analysis and the library binding constructor private.
Public `complete_checks` accepts only `SemanticContext::Isolated`.

One opaque source identity binds each library type environment to its checked program and lowered library.
The core exposes the lowered library as an immutable artifact.
Only `lower_library(&CheckedLibrary)` constructs that artifact.
The standard-library, dependency and compiled-context cache parsers require the identities and the declared-type map to match.
The standard-library and compiled-context parsers rerun the lower phase and compare each canonical payload with its cache payload, and adopt the stored effect and linearity results instead of rerunning those checkers (chelis#2558).
The dependency parser, whose payload carries no lowering, and a standard-library parser whose optional lowering is absent rerun the effect and linearity checkers without another type-inference session.
The compiler API installs linked-program policy for linked cache payloads before this validation.

The stored library proof identity is a self-consistency check between the cached type environment and program, not a recomputation from the decoded source. The identity is bound before the effect and linearity passes, and a layered build's cached library is a composed program whose identity is the extension's identity over the base context, so the decoded program's own source cannot reproduce it. Cache trust rests on the identity agreement, the envelope's source and build identity, and either the re-lowered payload comparison or, where the payload carries no lowering, the effect and linearity reruns. Where the comparison runs, the effect and linearity results are the producer's, adopted rather than rerun, and `cached_program_is_a_checker_fixed_point` locks that rerunning them would reproduce them (chelis#2558). A canonical post-effects derivation that would let the boundary recompute the identity from source is tracked as the `canonicalize-library-proof-identity` follow-up.

Before chelis#2558 every decode reran the effect and linearity checkers, and this paragraph records what that cost; the dependency cache still pays it. The decode-time reruns are not free, but the warm path stays well below a cold build. A throwaway measurement on the `path_dep_fixture` context (macOS local, 10 iterations, 31,735-byte cached payload) recorded a cold `compile_reef_context` at ~30.0 ms and a warm `CompiledContext::decode` (bincode decode plus the effect/linearity rerun plus the re-lower) at ~5.2 ms, a warm/cold ratio of ~0.17. The rerun-everything decode therefore keeps roughly a 5-6x margin over a from-scratch build on this fixture because it skips parse, desugar, macro expansion, and type inference; the re-lower and semantic reruns are the residual cost. This is the trade-off a future recompute-the-identity decode (`canonicalize-library-proof-identity`) is weighed against. The measurement used a temporary `#[ignore]` timing test that was not retained; re-run by re-adding an equivalent timer around `compile_reef_context` and `CompiledContext::decode` on the `path_dep_fixture`.
Both pipeline crates forbid unsafe code.

The manifest guard now scans normal and build dependencies. It also scans target-specific tables.
Its closed direct set rejects unknown build dependencies.
The documentation guard now rejects current compatibility claims without the exact attribute spelling.

The authoritative compiler pipeline oracle passed after these corrections.
The local gate passed for `chelis-compiler-api`, `chelis-pipeline-core`, and `chelis-reef`.
Hosted run `30914264791` passed for implementation head `1cf91cf8190b4eb9fb4a4f3bf7684aa8a3300b70`.
The `macOS Smoke`, `Docs`, `Lint and Unit Tests (Linux)`, and `Integration Tests (Linux)` jobs passed.
The hosted workspace suite covered all four changed crates and passed.

The change was rebased onto target `e1065d94fbd7a41f086e0690929a66c2335accc5`.
The Reef baseline harness passed against that target before the extraction.
The extracted implementation produced the same accepted and rejected results.

A fresh local red team checked the rebased tree.
It verified rootless-definition behavior with three CLI controls and one external contextual probe.
It also verified the facade, cancellation, guards, compile-fail controls, and Reef parity.

The review found one source-guard bypass.
Production files named `tests.rs`, files under `tests/`, and other `source_arch.rs` files escaped the scan.
The collector now scans every Rust file except the canonical guard implementation.
A negative fixture locks all three former bypass paths.
All 77 source-guard tests passed after the correction.

The authoritative compiler pipeline oracle passed after the rebased-tree correction.
The local gate also passed after that correction.
