# Chelis Language Specification: Nomenclature

**Version:** 0.2.0-draft
**Status:** Authoritative specification draft

---

This document defines the current core terminology used throughout the Chelis specs.
It deliberately focuses on settled, active meanings.
Historical terminology and superseded proposals are kept out of this glossary.

## 1. Syntax Layers

### Surf

The human-facing syntax of Chelis.
Surf is stored in `.ch` files and is meant for reading, review, and supervision.
Surf includes sugar such as infix operators, pipes, pattern matching, and ergonomic type
annotations.

### Deep

The canonical machine-facing syntax of Chelis.
Deep is stored in `.dp` files and is the source of truth for the compiler.
Every Deep node has the canonical shape `(tag {} children...)`.

### Canonical Form

The unique printed form of a Deep program.
Canonical printing normalizes structure and metadata placement so the same AST always
prints the same way.

## 2. File and Artifact Terms

### `.ch`

Surf source file.

### `.dp`

Deep source file.

### `.chb`

Planned binary Chelis artifact.
The name is settled, but the full on-disk format is not frozen in this spec yet.

### Shell

A package or compiled unit in the Reef ecosystem.
The term is ecosystem-facing and broader than a single file format.

## 3. Ecosystem Names

### Reef

The future package ecosystem and registry for Chelis.

### Tide

The interactive developer-facing layer for Chelis.
In Phase 0i this means the REPL plus related CLI commands.
In later phases it expands to include agent-facing APIs and tooling.

### Cove

The planned terminal UI environment built on top of Tide-era compiler services.

### `reef.toml`

The planned project manifest for the Reef ecosystem.

## 4. Compiler Stages

Chelis uses the following stage names:

### Parse

Convert source text into Surf or Deep AST structures.

### Desugar

Translate Surf AST into Deep AST.
This is mechanical and syntax-directed.

### Check

Type-check Deep AST, infer types, validate dimensions and precision, and produce
fitness-oriented diagnostics.

### Lower

Translate typed Deep AST into the RISC DAG.

### Transform

Apply DAG-to-DAG rewrites such as `grad`, `vmap`, and optimization passes.

### Emit

Translate the DAG into backend-specific source code such as C or future HIP kernels.

### Compile

Invoke external toolchains where needed to produce runnable artifacts from emitted code.

### Evaluate

Execute the RISC DAG directly through the IR evaluator.
This is the planned default interactive execution path.

## 5. CLI Terms

The current command vocabulary used across docs is:

- `chelis build`
- `chelis check`
- `chelis deep`
- `chelis surf`
- `chelis eval`
- `chelis tide`
- `chelis fmt`
- `chelis validate`
- `chelis cove`

Some commands are planned rather than already implemented.
This glossary defines the names, not their implementation status.

### `chelis build`

Run the production compilation path through code generation.
Today that means Surf or Deep input through type checking, lowering, C emission, and
runtime artifact generation for external native compilation.

### `chelis check`

Run the front-end and type-checking path only, producing fitness-oriented diagnostics.

### `chelis deep`

Print the Deep form of Surf input.

### `chelis surf`

Best-effort decompile Deep back into Surf.

### `chelis eval`

Evaluate an expression through the IR evaluator without going through C compilation.

### `chelis tide`

Launch the interactive REPL-oriented mode.

### `chelis fmt`

Format Surf code using the compiler-owned canonical style.

### `chelis validate`

Validate syntax against executable grammar tooling once that Phase 1 tool exists.

### `chelis cove`

Launch the planned terminal coding environment.

## 6. Type-System Terms

### ADT

Algebraic data type with constructors and exhaustive pattern matching.

### HM

Hindley-Milner inference as the basis of the Chelis type system.

### Named Dimensions

Tensor dimensions identified by names such as `batch`, `seq`, or `hidden`.
They participate in type checking and do not silently reorder or broadcast.

### Precision Types

Explicit numeric base types such as `f32`, `f64`, `bf16`, `int32`, and `bool`.
Precision never changes implicitly.

### Fitness Score

A graded compiler output describing how close a program is to type-correctness, together
with structured diagnostics and repair suggestions.

## 7. IR and Transform Terms

### RISC DAG

Chelis's typed intermediate representation after lowering.
Programs become DAGs of a small primitive tensor-op set.

### RISC Primitive

One of the small set of tensor primitives that backends and transform passes operate on.
See `spec/05-risc-primitives.md` for the authoritative list and semantics.

### Adjoint

The reverse-mode differentiation rule for a primitive operation.

### `grad`

Reverse-mode automatic differentiation as a DAG rewrite.

### `vmap`

Vectorization as a DAG rewrite over an added batch dimension.

### `jit`

A future compilation-boundary marker.
It remains part of the language design, but it does not imply a standalone JIT backend
plan.

## 8. Backend Terms

### Backend

A code generation target for the lowered DAG.

### C Backend

The current reference backend.
It emits portable C, uses BLAS where appropriate, and parallelizes loops with OpenMP.

### HIP Backend

The planned Phase 1 GPU path.
Chelis will generate HIP kernel strings and compile them via `hiprtc`.

### StableHLO Backend

A later integration backend for TPU and XLA-family interoperability.

### FX Backend

A later integration backend for PyTorch ecosystem interoperability.
