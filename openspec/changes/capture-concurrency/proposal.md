## Why

`spec/07-concurrency.md` records the settled Phase 0 / Phase 1 concurrency direction: implicit
DAG parallelism as the default story, the reserved `par` construct for explicit fork/join,
backend mapping, and the explicit v1 non-goals. No OpenSpec capability records these as
testable requirements.

## What Changes

- Introduce a `concurrency` capability recording implicit DAG parallelism, the reserved `par`
  construct, the backend concurrency mapping, and the v1 non-goals.

## Capabilities

### New Capabilities
- `concurrency`: implicit DAG parallelism, the `par` explicit-parallelism construct, backend
  concurrency mapping, and the v1 concurrency non-goals.

### Modified Capabilities

## Impact

- Source: `spec/07-concurrency.md` (read-only) is the authority for this capability.
- Surface: the C backend (OpenMP), BLAS-backed linear algebra, and the planned GPU backend.
- No code changes; this change records current behavior as an OpenSpec capability spec.
