# Concurrency

> **Status:** Stub. To be filled in Phase 1+.

Chelis concurrency model. Covers:
- Tier 1: DAG-implicit parallelism (independent DAG nodes can execute in parallel automatically)
- `par` construct for explicit data-parallel operations
- No shared mutable state — all parallelism is over immutable tensors
- Interaction with transformations (vmap naturally expresses data parallelism)

## Sections (planned)
- DAG-Level Parallelism
- The `par` Construct
- Memory Model
- Interaction with vmap
- Backend-Specific Parallelism (threads for C, warps for CUDA)
