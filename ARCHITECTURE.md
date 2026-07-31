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

## Crate Dependency Graph

```text
chelis-cli
  ├── chelis-surf
  │     └── chelis-deep
  ├── chelis-types
  │     └── chelis-deep
  ├── chelis-ir
  │     ├── chelis-deep
  │     └── chelis-types
  └── chelis-backend-c
        └── chelis-ir
```

The dependency graph is a strict DAG.
`chelis-deep` is the foundation, and `chelis-cli` only orchestrates library crates.

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
