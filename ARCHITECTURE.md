# Architecture

This document orients new developers and coding agents to the Chelis compiler. It describes the compilation pipeline, crate structure, and key design decisions.

## Compilation Pipeline

```
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
                   [Transformations]
                   (grad, optimize)
                           |
                    Optimized DAG
                           |
                    [C Code Emitter]
                           |
                      C source code
                           |
                     [gcc / clang]
                           |
                      Executable
```

Both entry points (Surf `.ch` files and Deep `.dp` files) converge at the Deep AST. The type checker operates on Deep AST, then lowering produces a RISC DAG of ~12 primitive operations. Transformations like `grad` (reverse-mode AD) and optimizations are DAG-to-DAG rewrites. Finally, a backend emits target code — currently C, with GPU backends planned.

## Crate Dependency Graph

```
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

The dependency graph is a strict DAG with no cycles. `chelis-deep` is the foundation that everything else depends on. The CLI crate at the top ties everything together but contains no compiler logic itself.

## Crates

### `chelis-deep` (foundation)

Parses the Deep syntax — s-expressions that serve as the canonical, machine-friendly representation of Chelis programs. Every Surf program desugars to Deep, and all compiler passes operate on the Deep AST. Key types will include `Expr` (the AST node enum), `Atom`, and the parser entry point.

### `chelis-surf`

Parses the surface syntax (Surf) that humans write, and desugars it to Deep AST. Surf adds syntactic conveniences like infix operators, `where` blocks, and indentation-sensitive syntax. Depends on `chelis-deep` because desugaring produces Deep AST nodes.

### `chelis-types`

Implements type checking over the Deep AST using Hindley-Milner inference extended with named tensor dimensions and precision tracking. Produces a typed AST where every node carries its inferred type. Also implements the fitness scoring system that rates type errors on a 0.0-1.0 scale and suggests repairs.

### `chelis-ir`

Defines the RISC DAG intermediate representation — approximately 12 primitive operations that every higher-level operation decomposes into. Handles lowering from typed Deep AST to DAG, and implements DAG-to-DAG transformations including `grad` (reverse-mode automatic differentiation) and optimization passes.

### `chelis-backend-c`

Generates C source code from an optimized RISC DAG. Handles memory planning (buffer allocation and reuse), BLAS integration for matrix operations, and emission of a self-contained C file that can be compiled with gcc or clang. Future backends (GPU, StableHLO) will follow the same interface.

### `chelis-cli`

The command-line binary. Orchestrates the pipeline from source file to compiled output. Contains no compiler logic — just argument parsing, file I/O, and calls into the library crates. Will eventually host the Tide interactive shell (Phase 0i).

## Where to Start

Current project status: Phase 0f (C backend codegen) is in progress. Phases 0a-0e are complete, and the core spec set is written and reviewed.

If you are working on the current phase, start with `spec/08-backends.md`, `spec/05-risc-primitives.md`, and `spec/12-roadmap.md`, then inspect `crates/chelis-backend-c` and `crates/chelis-ir`. The current implementation work is C emission, runtime support, BLAS integration, and numerical verification.

If you are an AI coding agent, the spec files in `spec/` are your primary reference. Each spec document is self-contained and numbered in dependency order.

## Key Design Decisions

### Virtual workspace

The project uses a Cargo virtual workspace (the root `Cargo.toml` has no `[package]`, only `[workspace]`). This keeps each compiler phase in its own crate with explicit dependencies, enabling parallel compilation and ensuring clean separation of concerns. A developer working on the type checker never accidentally depends on the C backend.

### S-expressions as the canonical form

Deep syntax uses s-expressions because they are trivial to parse, trivial to generate, and unambiguous. AI agents can emit Deep directly without worrying about operator precedence, indentation, or syntactic sugar. The surface syntax (Surf) exists purely for human ergonomics and desugars completely to Deep — the compiler never sees Surf after the desugaring pass.

### ~12 RISC primitives

Instead of having dedicated IR nodes for matmul, softmax, conv2d, attention, and every other operation, Chelis decomposes everything into approximately 12 primitives (elementwise ops, reduce, broadcast, reshape, index, etc.). This makes transformations like `grad` tractable — you only need differentiation rules for ~12 operations instead of hundreds. It also means new high-level operations can be added to Surf without changing the IR or backends.

### Named dimensions instead of positional

Tensor dimensions carry names (like `batch`, `hidden`, `seq_len`) rather than being identified by position (axis 0, axis 1, ...). This eliminates a large class of shape errors — transposing `[batch, hidden]` to `[hidden, batch]` is explicit, not a silent `transpose()`. It also makes dimension checking local: each operation specifies which named dimensions it expects, and the type checker verifies compatibility without needing to track axis ordering through an entire program.
