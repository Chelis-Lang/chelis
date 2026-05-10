# Cross-Function Specialization Workstream

**Status:** Active implementation contract for the remaining compiler perf batch.
This document records the path for closing the current cross-function
specialization gap without changing source syntax.

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

## Chosen Direction: Auto-Derived Summaries

This workstream follows call-graph-aware specialization summaries. Summaries are
auto-derived by the compiler; there are no user annotations in v1.

Path (a), whole-program inlining before optimization, remains a possible implementation
technique for small helpers. It is not the design contract because it couples user
abstraction performance to an inliner heuristic and does not scale cleanly to packaged
library boundaries.

## Specialization Summaries

A top-level pure helper may be summarized as equivalent to a compiler-known tensor
operation, initially `matmul` / GEMM and sparse gather once the gather recognizer
ships.

The derived metadata shape is:

```text
chelis_specialization: { kind: "blas_matmul", m: <expr>, n: <expr>, k: <expr> }
chelis_specialization: { kind: "gather", axis: <axis>, ... }
chelis_specialization: { kind: "none" }
```

The semantic contract is:

- summary derivation runs after type checking and ordinary helper lowering
- summaries are compiler-derived only; v1 has no user annotations or annotation
  syntax
- direct built-in calls and equivalent primitive expansions may produce summaries
- parameter order, transpose/layout constraints, dtype, rank, and result shape are
  explicit in the summary
- summaries do not introduce implicit broadcasting, implicit precision promotion,
  or new AD rules
- helpers with effects, recursion, polymorphic ambiguity, or unsupported control
  flow get `kind: "none"` rather than a partial summary

The initial accepted surface is deliberately narrow:

- rank >= 2 `f32` matmul with the same constraints as the IR BLAS recognizer
- helper body is pure tensor code with no `Random`, `IO`, or host-side effects
- no data-dependent control flow around the equivalent operation
- no hidden allocation or mutation semantics beyond the ordinary tensor DAG nodes

## Nested Helper Composition

Library authors should be able to decompose readable helpers without losing
specialization. The compiler therefore needs summaries, not only inlined bodies.

Required behavior:

- every eligible helper produces a typed specialization summary
- a helper summary may reference summaries of other summarized helpers
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

- auto-derived summaries are shape-polymorphic over named dimensions unless the helper
  signature fixes concrete extents
- callsite specialization instantiates the summary after type checking has bound named
  dimensions but before backend emission selects BLAS, hipBLAS, or generic loops
- repeated named dimensions must still match by name and runtime value under the stable
  tensor ABI
- dimension-specialized clones may cache the instantiated summary, but they must not
  weaken the original named-dimension checks
- symbolic dimensions may block a backend library path only when that backend requires a
  statically known extent the runtime ABI cannot provide

The fixed compiler order is:

```text
AD -> closed-list no-op cleanup -> BLAS/gather/scatter recognizers
   -> cross-function callsite specialization -> DCE -> in-place fusion
   -> backend codegen
```

## Callsite Emission Rules

The rules in this section are the full summary-table target. The current branch
ships the first C path by specializing emitted tensor-helper DAG bodies; it does
not yet attach persistent summary metadata to the call graph or bypass the
helper at the caller.

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
- when summary metadata proves specialization, the callsite emits the specialized
  form unconditionally; there is no runtime fallback branch
- the generated helper function is always emitted; specialized callsites may bypass it,
  but non-specializable callers, missing metadata, older/dynamic paths, and debugging
  still have the ordinary helper body available
- if constraints fail before a summary can be proven, the helper gets `kind: "none"`
  and ordinary helper-call emission is used

Diagnostics should distinguish "no specialization summary could be derived" from
"summary derived but this callsite did not meet the backend constraints."

## Empirical Acceptance

The executable acceptance anchor is
`crates/chelis-cli/tests/cross_library_semantic_gap.rs`.

The current branch is accepted when the two-test pattern proves both the inline baseline
and the first user-defined helper closure:

- `inline_matmul_forms_hit_blas_and_allocate_only_result` proves the direct
  `matmul(a, b)` and inline `expand -> mul -> sum` cases still emit at least one
  `cblas_sgemm` or `chelis_blas_matmul` and allocate only the result tensor
- `user_def_matmul_helpers_hit_blas` proves `user_def_builtin` and `user_def_manual`
  emit at least one `cblas_sgemm` or `chelis_blas_matmul`
- `user_def_matmul_helpers_hit_blas` also proves the generated helper body remains
  emitted and called, so the test does not pass by deleting the helper surface
- assertions are based on generated Chelis C before invoking clang, gcc, or LTO

Manual acceptance command:

```sh
cargo test -p chelis-cli --test cross_library_semantic_gap -- --nocapture
```

Expected success condition: all four semantically equivalent matmul forms report BLAS
hits in the generated C, and the user-defined cases no longer rely on clang LTO to
recover performance. This is the current helper-body specialization gate, not the full
summary-table callsite-bypass gate.

Current implementation note: the first shipped C path specializes emitted tensor-helper
DAGs using the existing BLAS recognizer, which makes user-defined matmul helpers hit
BLAS while preserving helper emission. A complete summary-table implementation still
needs to move the proof metadata into the call graph so eligible callsites can emit the
specialized form directly and surface BLAS link requirements from host-program codegen
without source-string inference.
