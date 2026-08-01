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
