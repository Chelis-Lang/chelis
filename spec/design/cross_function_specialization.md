# Cross-Function Specialization Workstream

**Status:** Active implementation contract for the remaining compiler perf batch.
This document records the path for closing the current cross-function
specialization gap without changing source syntax.

## Problem

The shipped C backend recognizes BLAS-equivalent rank-2 `f32` matmul when the canonical
shape is visible in the function currently being emitted:

- direct `matmul(a, b)`
- inline hand-written `expand -> mul -> sum`

Before the first C helper-summary slice, the same computation missed BLAS when it was
hidden behind a user-defined helper called from another user function. The helper was
emitted as a separate generated C function, and the BLAS detector ran on the caller's
per-function DAG rather than on a verified callee summary. The remaining problem is
broadening that narrow C path to HIP, gather/scatter summaries, richer helper shapes,
and explicit rejection diagnostics.

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
operation. The shipped surface covers `matmul` / GEMM (the first slice) and the three
sparse RISC primitives `gather`, `scatter_add`, and `scatter_replace`.

The derived metadata shape is:

```text
chelis_specialization: { kind: "blas_matmul", m: <expr>, n: <expr>, k: <expr> }
chelis_specialization: { kind: "sparse_gather", axis: <axis>, input_indices: [...] }
chelis_specialization: { kind: "sparse_scatter_add", axis: <axis>, input_indices: [...] }
chelis_specialization: { kind: "sparse_scatter_replace", axis: <axis>, input_indices: [...] }
chelis_specialization: { kind: "none" }
```

For sparse summaries the recognizer requires:

- exactly one DAG root, equal to a `RiscOp::Gather`, `RiscOp::ScatterAdd`, or
  `RiscOp::Scatter`
- every sparse-op operand is a direct `RiscOp::Load` referencing one of the
  helper's input parameters by name
- indices precision is `int32` or `int64`
- payload precisions match across `values`/`updates`/`target`/`output`
- no helper input or output carries a wildcard `Named("*", None)` dim (type-
  inference placeholder); wildcard helpers fall back to the marshaling shim
  rather than register a false-positive summary

`RiscOp::ScatterAdd` has no Surf surface form today (it is produced exclusively
by the AD adjoint of `gather`). The recognizer covers it mechanically for
symmetry with the other two sparse ops; IR-level coverage lives in
`crates/chelis-ir/tests/host_sparse_summary.rs`.

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

The current branch ships the first C summary path for BLAS-equivalent tensor
helpers and simple pure top-level wrappers. Summary metadata is derived from the
helper DAG, propagated through wrappers that only pass tensor parameters through,
and consumed by C host emission. The generated helper function is still emitted
so non-specialized callers and debugging paths remain available.

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
  emitted, so the test does not pass by deleting the helper surface. Specialized
  callsites may bypass that helper body; they do not emit a runtime fallback branch.
- assertions are based on generated Chelis C before invoking clang, gcc, or LTO

Manual acceptance command:

```sh
cargo test -p chelis-cli --test cross_library_semantic_gap -- --nocapture
```

Expected success condition: all four semantically equivalent matmul forms report BLAS
hits in the generated C, and the user-defined cases no longer rely on clang LTO to
recover performance.

Current implementation note: the shipped C path combines helper-body BLAS
specialization with compiler-derived host summaries for simple wrappers. The W3-B
batch broadens the recognizer to the three sparse RISC primitives (`Gather`,
`ScatterAdd`, `Scatter`/replace) with the same wrapper-propagation rules. The
sparse acceptance oracle is `crates/chelis-cli/tests/cross_library_sparse_summaries.rs`
plus the IR-level lock at `crates/chelis-ir/tests/host_sparse_summary.rs`. Remaining
work is to add explicit negative diagnostics for summary-derived-but-callsite-
rejected cases (W4-A), carry the same summary consumption into HIP (W3-A), and
surface BLAS link requirements from structured host-program metadata instead of
source-string inference.
