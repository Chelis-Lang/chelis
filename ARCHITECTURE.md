# Architecture

This document orients new developers and coding agents to the Chelis compiler as it
exists today.
For project-level decisions, read `spec/design/chelis_canonical_reference.md` first.
This file explains how the workspace is structured and how a source file moves through
the compiler.

## Current Status

Phase 0f (C backend codegen) is in progress.
Phases 0a-0e are complete.
The active work is backend emission, runtime support, BLAS integration, and numerical
validation.

## Compilation Pipeline

```text
  .ch source          .dp source
      |                    |
  [Surf Lexer]        [Deep Lexer]
      |                    |
  [Surf Parser]       [Deep Parser]
      |                    |
  Surf AST            Deep AST
      |                    |
  [Desugarer] -------->   |
                           |
                    [Type Checker]
                           |
                    Typed Deep AST
                           |
                    [Effect Checker]
                           |
                   [Linearity Checker]
                           |
                      [Lowering]
                           |
                       RISC DAG
                           |
              [Optimize / Transform / Evaluate]
                           |
                 +---------+----------+
                 |                    |
          [IR Evaluator]      [C Code Emitter]
                 |                    |
          Interactive result      C source code
                                       |
                                [gcc / clang]
                                       |
                                  Executable
```

Both source forms converge at Deep.
The type checker runs on Deep, lowering produces the RISC DAG, and then the pipeline
forks:

- interactive workflows use the IR evaluator
- production builds emit C and compile it with the system toolchain

This split is deliberate.
Chelis does not plan a second general-purpose JIT backend unless measured latency makes
it necessary.

### Canonical Pipeline Owner

`chelis-pipeline-core` owns type analysis, effect checks, linearity checks, root metadata, and lowering.

`chelis_compiler_api::pipeline` remains the public facade. It owns source preparation, dynamic goals, cancellation, host policy, and backend policy.

Consumers select one closed goal:

- `TypeAnalysis` returns fitness and one type-inference product.
- `FullCheck` adds effect and linearity checks.
- `Lower` adds DAG lowering and canonical root metadata.

`CheckedCompilation` exists only after all semantic checks accept the program.
`CheckedLibrary` binds one type environment to its semantically accepted program.
A composable contextual analysis retains the exact checked library that produced it.
A library-extension analysis also retains the combined type environment from its type session.
Contextual composition consumes the bound product and accepts no replacement library or environment.
Contextual lowering accepts only a core-bound lowered library. It rejects a library with another proof identity.
The core exports no function that adopts separate prepared, environment, or checked products.
Public semantic completion accepts only `SemanticContext::Isolated`.
Contextual completion requires an analysis that already retains its checked library.
Library type products share one opaque identity derived from accepted checked source.
A lowered library retains the same identity in an immutable core artifact.
Only `lower_library(&CheckedLibrary)` constructs that artifact.
Cache parsing requires the identities and the declared-type map to match.
It reruns semantic checks without another type-inference session.
It reruns the lower phase and compares the canonical payload with the cache payload.
`LoweredCompilation` contains one checked compilation and its DAG products.
A rejection does not contain a checked or lowered success product.
The core and compiler API crates forbid unsafe code.

The CLI retains style policy, Reef preparation, JSON, exit codes, target selection, and backend emission.
Backend emitters remain final target-specific correctness boundaries.

`chelis-reef` passes linked expanded Deep to the core. It retains package links, name policy, archives, and schemas.

The source guard checks five production roots. It does not claim coverage for all workspace crates.
The detailed inventory is in `docs/investigations/compiler_pipeline_inventory.md`.
The `std` blocker inventory is in `docs/investigations/pipeline_core_std_blockers.md`.

## Crate Dependency Graph

```text
chelis-cli ───────────────┐
chelis-e2e ───────────────┤
chelis-tide ──────────────┼──> chelis-compiler-api
chelis-python ────────────┘          │
                                     ├──> chelis-pipeline-core
                                     ├──> chelis-reef ──> chelis-pipeline-core
                                     ├──> chelis-surf and chelis-macros
                                     └──> target backends

chelis-pipeline-core
    ├──> chelis-deep
    ├──> chelis-types
    ├──> chelis-effects
    └──> chelis-ir
```

The dependency graph is a strict DAG.
The compiler API and Reef use the dependency-bottom semantic core.
The core has exactly four direct production dependencies.

## Crates

### `chelis-deep`

Parses and prints Deep, the canonical machine-facing syntax.
Deep is the compiler's common representation and the target produced by Surf
desugaring.

### `chelis-surf`

Parses Surf and desugars it into Deep.
It also owns best-effort decompilation back to Surf.

### `chelis-types`

Implements Hindley-Milner inference extended with named tensor dimensions, precision
tracking, and fitness-oriented error reporting.

#### Type-inference module map

`crates/chelis-types/src/infer/mod.rs` defines the public facade and the shared imports.
Its child modules have these roles:

- `stack.rs` owns stack growth and recursion protection.
- `checked.rs` owns checked-program construction, metadata, and totality finalization.
- `program.rs` owns the public check entry points and the inference schedules.
- `declarations.rs` owns declaration collection, dependency analysis, and signature schedules.
- `validate.rs`, `static_value.rs`, and `annotate.rs` own IR checks and type annotation.
- `expr.rs` owns expression dispatch and common expression helpers.
- `expr_function.rs` owns functions, definitions, local bindings, conditionals, and pipes.
- `expr_pattern.rs` owns match and pattern inference.
- `expr_record.rs` owns tuples, records, access, updates, and casts.
- `expr_transform.rs` owns gradient and vector-map inference.
- `app.rs` owns generic calls and dispatch to operation families.
- `app_numeric.rs` owns numeric rules and precision diagnostics.
- `app_tensor.rs` owns tensor signature checks.
- `app_shape.rs` and `app_shape_helpers.rs` own shape rules and static shape readers.
- `app_collection.rs` owns collection rules and constructor helpers.
- `app_post.rs` applies operation-family checks after generic unification.

The source guard is in `crates/chelis-types/src/source_arch.rs`.
It rejects the legacy `src/infer.rs` path, a missing role module, or a source file with more than 3,000 lines.

### `chelis-ir`

Defines the RISC DAG, lowering, verification, optimization passes, transform passes,
and the IR evaluator used for interactive execution.

### `chelis-backend-c`

Emits C from the DAG, performs BLAS-oriented lowering decisions, and manages runtime
and memory-planning concerns for the reference backend.

### `chelis-cli`

Exposes the compiler pipeline as commands such as `build`, `check`, `deep`, `surf`,
`eval`, and `tide`.

## Where to Start

If you are working on the current phase, start with:

1. `spec/design/chelis_canonical_reference.md`
2. `spec/05-risc-primitives.md`
3. `spec/08-backends.md`
4. `spec/12-roadmap.md`

Then inspect:

- `crates/chelis-ir`
- `crates/chelis-backend-c`
- `crates/chelis-cli`

If you are extending the language front end, read `spec/02-surf-syntax.md`,
`spec/03-deep-syntax.md`, and `spec/04-type-system.md` in order.

## Design Constraints

### Deep Is Canonical

Surf is for supervision.
Deep is the canonical compiler-facing syntax.
Every Deep node has the form `(tag {} children...)`, which keeps generation and
transformation regular for both agents and compiler passes.

### Small IR, Rich Surface

High-level language constructs lower into a compact RISC DAG rather than requiring a
large backend surface area.
That keeps `grad`, optimization, and backend emission tractable.

### Pure Stage Boundaries

Compilation stages should remain pure functions from inputs to outputs.
Avoid global mutable state and long-lived compiler objects with implicit sequencing.
This keeps the design easy to test now and makes a later `salsa` migration mechanical
rather than architectural.

### Evaluator First for Interactivity

Tide and `chelis eval` should go through the IR evaluator first.
If interactive latency later needs more work, the escalation order is:

1. cache compiled C artifacts
2. use a persistent compiler helper
3. only then evaluate a JIT backend
