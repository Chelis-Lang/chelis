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

The guard rejects a function that directly calls two or more canonical semantic stage primitives.

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

The four root products use `IrName`. Normal DAG output uses one exact alignment constructor, and host-only output uses one explicit empty constructor.

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

A fresh local red-team reviewed the specifications, code, tests, public API, and completed oracle evidence. The review found no actionable issue.

A final coverage audit added one runtime test for all three `LayeredCheck` variants. This test passed.

No high-severity finding was rejected. Thus, no rejected finding needs a separate disposition.

## Acceptance Evidence

### Local Evidence

The authoritative compiler pipeline oracle passed before and after the adversarial review. The format check and strict change validation also passed.

The local gate found one timeout test failure under concurrent load. The isolated rerun passed, and the full CLI rerun passed all 1,797 tests.

### Hosted Evidence

Hosted CI evidence is pending. Local results do not replace macOS Smoke, Docs, or the changed-crate jobs.
