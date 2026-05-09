# Cross-Function Specialization Workstream

**Status:** Proposed follow-up workstream.
This document records the intended path for closing the current cross-function
specialization gap without changing the Phase 2 completion claim.

## Problem

The shipped C backend recognizes BLAS-equivalent rank-2 `f32` matmul when the canonical
shape is visible in the function currently being emitted:

- direct `matmul(a, b)`
- inline hand-written `expand -> mul -> sum`

The same computation misses BLAS when it is hidden behind a user-defined helper called
from another user function. The helper is emitted as a separate generated C function,
and the BLAS detector runs on the caller's per-function DAG rather than on a verified
callee summary.

clang LTO is only a workaround. It can inline generated C helpers and improve ordinary
native optimization, but it cannot make Chelis select `cblas_sgemm`, hipBLAS, or future
vendor-library calls after the backend has already emitted the wrong code shape.

## Chosen Direction: Path (b)

This workstream follows path (b): explicit BLAS-equivalent function annotations with
compiler verification and callsite specialization.

Path (a), whole-program inlining before optimization, remains a possible implementation
technique for small helpers. It is not the design contract because it couples user
abstraction performance to an inliner heuristic and does not scale cleanly to packaged
library boundaries.

## BLAS-Equivalent Function Annotations

A top-level pure helper may be marked as equivalent to a compiler-known tensor
operation, initially `matmul` / GEMM.

The concrete Surf spelling is intentionally not fixed here. The semantic contract is:

- the annotation names the canonical operation family, such as `blas.matmul`
- the helper's typed body must verify against that operation's canonical DAG summary
- the verifier accepts both direct built-in calls and equivalent primitive expansions
- parameter order, transpose/layout constraints, dtype, rank, and result shape must be
  explicit in the verified summary
- the annotation does not introduce implicit broadcasting, implicit precision promotion,
  or new AD rules
- an annotation that cannot be proven equivalent is a compile-time error, not a hint

The initial accepted surface is deliberately narrow:

- rank-2 `f32` matmul with contiguous row-major operands
- helper body is pure tensor code with no `Random`, `IO`, or host-side effects
- no data-dependent control flow around the equivalent operation
- no hidden allocation or mutation semantics beyond the ordinary tensor DAG nodes

## Nested Helper Composition

Library authors should be able to decompose readable helpers without losing
specialization. The compiler therefore needs summaries, not only inlined bodies.

Required behavior:

- every verified annotation produces a typed specialization summary
- a helper summary may reference summaries of other verified helpers
- nested summaries are expanded logically for verification with a fixed recursion limit
- wrappers that only rename, reorder, or pass through arguments may preserve the summary
  when the verifier can prove the same canonical operation
- wrappers that add arithmetic around a BLAS-equivalent helper are not automatically
  BLAS-equivalent, though their internal helper call may still specialize at its
  callsite

This keeps ordinary helper composition viable while avoiding an unrestricted
cross-function pattern matcher.

## Dimension Specialization Interaction

Future dimension-specialization work may clone or specialize functions for concrete
symbolic dimension bindings. Cross-function specialization must compose with that path.

Rules:

- annotation summaries are shape-polymorphic over named dimensions unless the helper
  signature fixes concrete extents
- callsite specialization instantiates the summary after type checking has bound named
  dimensions but before backend emission selects BLAS, hipBLAS, or generic loops
- repeated named dimensions must still match by name and runtime value under the stable
  tensor ABI
- dimension-specialized clones may cache the instantiated summary, but they must not
  weaken the original named-dimension checks
- symbolic dimensions may block a backend library path only when that backend requires a
  statically known extent the runtime ABI cannot provide

The intended order is:

```text
type check -> verify annotations -> bind callsite dimensions -> instantiate summaries
-> backend specialization -> emit C/HIP
```

## Callsite Emission Rules

When a call to a verified BLAS-equivalent helper is encountered, the caller's backend
emission behaves as if the canonical operation appeared directly at the callsite.

Required rules:

- if the callsite satisfies the summary's dtype, rank, layout, and dimension
  constraints, emit the backend library path directly
- for the C backend, the generated C must contain the same `cblas_sgemm` /
  `chelis_blas_matmul` path used by inline `matmul`
- for the HIP backend, the same rule applies to the existing or future hipBLAS path
- required link flags and runtime artifacts are surfaced exactly as they are for direct
  built-in specialization
- the generated helper function may still exist for unspecialized calls, but the hot
  specialized callsite must not depend on native-compiler LTO to recover the BLAS call
- if constraints fail at a callsite, the compiler falls back to the ordinary helper call
  or generic lowered DAG and may emit a diagnostic explaining why specialization did not
  fire

Diagnostics should distinguish "annotation rejected" from "annotation accepted but this
callsite did not meet the backend constraints."

## Empirical Acceptance

The executable acceptance anchor is
`crates/chelis-cli/tests/cross_library_semantic_gap.rs`.

The workstream is accepted when `semantic_gap_inline_vs_user_def` flips the two
user-defined cases from expected BLAS misses to expected BLAS hits:

- `user_def_builtin` must emit at least one `cblas_sgemm` or `chelis_blas_matmul`
- `user_def_manual` must emit at least one `cblas_sgemm` or `chelis_blas_matmul`
- the existing direct and inline manual cases must continue to hit BLAS
- the assertion is based on generated Chelis C before invoking clang, gcc, or LTO

Manual acceptance command:

```sh
cargo test -p chelis-cli --test cross_library_semantic_gap semantic_gap_inline_vs_user_def -- --nocapture
```

Expected success condition: all four semantically equivalent matmul forms report BLAS
hits in the generated C, and the user-defined cases no longer rely on clang LTO to
recover performance.

The same acceptance condition is encoded as the ignored target-behavior test
`target_behavior_user_def_matmul_helpers_hit_blas`. While the workstream is open, run
it explicitly as a known-failing target check:

```sh
cargo test -p chelis-cli --test cross_library_semantic_gap target_behavior_user_def_matmul_helpers_hit_blas -- --ignored --nocapture
```
