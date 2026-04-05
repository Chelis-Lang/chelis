# Concurrency

**Status:** Outline with settled Phase 0 / Phase 1 direction.
Detailed semantics can expand as the implementation matures.

## 1. Tier 1: Implicit DAG Parallelism

Chelis is pure by default and lowers to a DAG.
Independent branches of that DAG are eligible for parallel execution without explicit
programmer annotation.

This is the default concurrency story for v1.

## 2. Explicit Parallelism: `par`

Chelis reserves `par` for cases where the programmer wants to express explicit fork/join
structure that the compiler cannot recover automatically.

Phase 0 does not require a sophisticated runtime interpretation of `par`.
The language-level construct is settled before backend-specific scheduling is.

## 3. Backend Mapping

- the C backend uses OpenMP-parallel loops where appropriate
- BLAS-backed dense linear algebra uses library-managed parallelism
- the planned GPU backend maps parallel work onto HIP kernels

These backend details do not change the language-level semantics.

## 4. Non-Goals for v1

The following are not part of the initial concurrency story:

- shared mutable-state concurrency
- actor systems
- stream-processing semantics
- scatter/gather-focused parallel runtime features

## 5. Related Features

Concurrency interacts with:

- `vmap`, which expresses structured data parallelism
- future linear types, which will clarify safe buffer reuse
- backend scheduling and fusion passes
