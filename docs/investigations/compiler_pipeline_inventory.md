# Compiler Pipeline Inventory

This inventory records production front-end sequences before the canonical pipeline migration.

The scope covers these consumers:

- compiler API
- CLI
- edit tools
- caches
- contexts
- E2E tools

Tests can call individual stages for focused evidence.

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

`chelis_compiler_api::pipeline` now owns these transitions:

- source preparation
- isolated and contextual type analysis
- effect checks
- linearity checks
- isolated and contextual lowering
- root metadata

Adapters retain presentation and policy. The CLI retains style checks, Reef preparation, JSON, exit codes, target selection, and backend emission.

The source guard checks production functions in these trees:

- `crates/chelis-compiler-api/src`
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

## Dependency Boundary

`chelis-reef` remains below `chelis-compiler-api` in the dependency graph. Its package artifact builder cannot call the compiler-API pipeline without a dependency cycle.

This change does not invert that dependency. The guard treats Reef linking and package artifact construction as the lower preparation boundary.

A later package-resolution split can move the remaining Reef artifact sequence behind the compiler API. That work is outside this change.

## Final Artifact Boundaries

| Artifact | Constructor owner | Consumers |
|---|---|---|
| `ValidatedModule` | `check_whole_module_edit` | Edit tools and compiler API schema adapters |
| `AllRootNames` | Compiler pipeline root analysis | Compiler API evaluation and host output adapters |
| `TensorRootNames` | Compiler pipeline root analysis | Lower policy, CLI root selection, and compiler API evaluation |
| `NamedRoots` | Exact name-to-DAG alignment | Compiler API schema adapters, evaluation, and E2E tools |
| `ForwardNodeIndex` | `NamedRoots` plus DAG load aliases | Compiler API gradient adapters |
| `LoweredParts` | `LoweredCompilation::into_parts` | Compiler API and E2E consumers |
| `SemanticRejection` | `complete_checks` | Full pipeline adapters, edit validation, layered checks, and CLI checks |
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

A final fresh post-remediation red-team remains pending.

## Acceptance Evidence

### Local Evidence

The earlier oracle runs occurred before the latest target rebase. They do not satisfy final acceptance.

Current focused tests passed for typed roots, full stage identities, receiver methods, loop exits, aliases, higher-order calls, wire shape, and typed `.dp` rejection.

The format check, strict validation, target all-target check, and authoritative oracle passed after the second remediation.

The authoritative oracle passed again after the third source-guard remediation. It included the real CLI host-backend regression test.

The oracle passed after the fourth remediation. It included the path-alias tests, higher-order tests, and raw-offset compile fixture.

All 54 focused source-guard tests passed after the fifth remediation. The authoritative oracle also passed on that tree.

All 65 focused source-guard tests passed after the sixth remediation. The authoritative oracle passed after the current target rebase.

All 74 focused source-guard tests passed after the seventh remediation. The final target rebase and authoritative oracle remain pending.

The first local gate found that blanket empty-DAG alignment broke a valid C host-backend program. The revised typed policy preserves that build.

The second gate passed that test. It then found a stale canary for the target typed-parser diagnostic.

The isolated canary passed after its accepted set included the target parser message. Later tests still expected pre-#912 unlabeled roots.

Those tests now require the target `[05-OBS-6]` labels. The earlier local gate passed for all four changed crates.

The earlier target stopped on three `infer_recursion_depth_guard` tests with signal 10. The old `origin/main` reproduced those failures.

Current `main` includes the stack-safe normalization fix from #1035. All five `infer_recursion_depth_guard` tests now pass after the rebase.

### Hosted Evidence

Hosted CI evidence is pending. Local results do not replace macOS Smoke, Docs, or the changed-crate jobs.
