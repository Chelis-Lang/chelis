# Concurrency

## 1. Implicit DAG Parallelism

Chelis is pure by default and lowers to a DAG.
Independent branches of that DAG are eligible for parallel execution without explicit
programmer annotation.

This is the default concurrency model.

## 2. Explicit Parallelism: `par`

Chelis reserves `par` for cases where the programmer wants to express explicit fork/join
structure that the compiler cannot recover automatically.

The semantics of `par` do not require a particular scheduler. A conforming
implementation may execute it sequentially or in parallel, but observable results must
be identical.

## 3. Backend Mapping

- the C backend uses OpenMP-parallel loops where appropriate
- BLAS-backed dense linear algebra uses library-managed parallelism
- GPU backends map parallel work onto kernels

These backend details do not change the language-level semantics.

## 4. Non-Goals

The concurrency model excludes:

- shared mutable-state concurrency
- actor systems
- stream-processing semantics
- scatter/gather-focused parallel runtime features

## 5. Related Features

Concurrency interacts with:

- `vmap`, which expresses structured data parallelism
- linear types, which clarify safe buffer reuse
- backend scheduling and fusion passes
