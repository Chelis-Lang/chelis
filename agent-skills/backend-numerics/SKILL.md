---
name: backend-numerics
description: Use for Chelis IR, backend, and autodiff work where numerical correctness matters. Focuses on evaluator-vs-backend agreement, compile-run harnesses, tolerances, and finite-difference checks.
---

# Backend Numerics

Use this skill for IR lowering, backend codegen, numerical kernels, and autodiff work.

## Core Principle

A green string-emission test is not enough.
Numerical paths must be validated with execution.

## Validation Surfaces

- evaluator vs generated C agreement
- compile-and-run helpers
- finite-difference comparisons for gradients
- shape, stride, and reduction edge cases
- BLAS and non-BLAS agreement where both paths exist

## Chelis Expectations

- prefer real numeric assertions over structural-only checks
- include multi-op pipelines, not just single primitives
- exercise scalar, small tensor, and larger tensor cases where threading matters
- document any manual long-running numerical gate separately from default CI

