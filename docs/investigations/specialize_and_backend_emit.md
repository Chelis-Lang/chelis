# Specialize.rs + Backend `emit.rs` — Research Pass

A reference map of two compiler surfaces in the Chelis codebase:

1. `crates/chelis-ir/src/specialize.rs` — the IR-level backend specialization
   pass: recognizers, passes, pass order, input/output contracts, sibling
   dependencies.
2. Every backend's `emit.rs` — the `RiscOp`-by-`RiscOp` dispatch matrix across
   the three in-tree backends (C, HIP, Metal), plus the backend-specific
   quirks (phase gates, panics, `todo!()`, fallbacks).

This document is factual and citation-heavy by design (`file:line` for every
behavior claim). It is a map for navigating the code, not an opinion piece —
no recommendations or refactor proposals appear here. Verified against the
source on 2026-05-11.

A note on grouping: the `RiscOp` enum source comment puts `UniformLike` and
`Dropout` under the "Unary elementwise" header (`dag.rs:470–490`), but those
two ops carry state (seeds) and their backend dispatch is unlike the pure
unary ops, so this report treats them as a distinct **RNG** family in the
matrix sections.

## Table of contents

- [Part 1 — specialize.rs](#part-1--crateschelis-irsrcspecializers)
  - Overview
  - Public API
  - Recognizers
  - Passes
  - Pass order
  - Input / output contract
  - Sibling module dependencies
- [Part 2 — backend `emit.rs`](#part-2--backend-emitrs)
  - Backend inventory
  - RiscOp inventory
  - Section A — Binary elementwise
  - Section B — Unary elementwise
  - Section C — RNG
  - Section D — Reductions
  - Section E — Movement
  - Section F — Memory & lifecycle
  - Section G — Cast
  - Section H — Fusion
  - Section I — Matmul
  - Section J — Sparse
  - Backend-specific quirks
  - Matrix completeness summary
  - Appendix — RiscOp reference

## Part 1 — `crates/chelis-ir/src/specialize.rs`

### Overview

IR-level backend specialization pass for the Chelis compiler (`specialize.rs:1–5`). The module runs after autodiff and before DCE / fusion / codegen, replacing recognized subgraphs with explicit backend-specialized IR nodes. The module comment states the pass is "deliberately conservative: the no-op cleanup has a closed list" (`specialize.rs:3–5`). Total length: 1,250 lines.

The pipeline ordering contract is encoded as code rather than prose via `pub const SPECIALIZATION_PIPELINE_ORDER` (`specialize.rs:17–27`), which lists nine stages (`ad`, `no_op_cleanup`, `dense_gather_recognizer`, `one_hot_fallback_lowering`, `blas_matmul_recognizer`, `cross_function_specialization`, `dce`, `in_place_fusion`, `codegen`). The constant exists because "gather / scatter recognizers, cross-function specialization, and future in-place rewrites all depend on the same ordering contract" (`specialize.rs:14–16`).

### Public API

- `pub const SPECIALIZATION_PIPELINE_ORDER: &[&str]` (`specialize.rs:17–27`) — Declares the compiler pipeline ordering around backend specialization as a code-level contract.
- `pub fn specialize_for_blas(dag: &Dag) -> Dag` (`specialize.rs:30–36`) — Main entrypoint. Runs closed-list no-op cleanup, dense-gather recognition, OneHot fallback lowering, BLAS matmul recognition, then DCE.
- `pub fn eliminate_closed_list_noops(dag: &Dag) -> Dag` (`specialize.rs:40–78`) — Eliminates identity `Cast`, identity `Reshape`, and identity `Permute` nodes.

### Recognizers

| Function | Lines | Pattern | Returns |
|---|---|---|---|
| `identity_source` | `specialize.rs:80–107` | Identity `Cast` (same precision + same dims), identity `Reshape` (input dims == new_shape == output dims), or identity `Permute` (axes equal `0..n` and dims unchanged) | `Option<NodeId>` |
| `detect_dense_gather_pattern` | `specialize.rs:384–418` | `Sum`-over-vocab of `Mul(OneHot-expanded, Values-expanded)` subgraph | `Option<DenseGatherInfo>` |
| `detect_dense_gather_operands` | `specialize.rs:420–499` | Helper validating gather operand structure and dim alignment | `Option<DenseGatherInfo>` |
| `detect_matmul_pattern` | `specialize.rs:505–590` | `Sum`-over-K of `Mul(Expand(A), Expand(B))` with F32 precision and contiguous matrix slices | `Option<MatmulInfo>` |
| `matmul_dims` | `specialize.rs:592–610` | Validates and extracts `(m, n, k)` triple from typed operands and batch dims | `Option<(DimExpr, DimExpr, DimExpr)>` |
| `node_has_contiguous_matrix_slices` | `specialize.rs:612–671` | Recursive check that no `Permute`/stride/`Pad`/`Shrink` interferes at matrix rank | `bool` |
| `dims_equivalent` | `specialize.rs:501–503` | Structural equality of `DimExpr` values | `bool` |

### Passes

| Function | Lines | Input | Output | Recognizers used | Mutation |
|---|---|---|---|---|---|
| `eliminate_closed_list_noops` | `specialize.rs:40–78` | `Dag` containing identity `Cast`/`Reshape`/`Permute` | `Dag` with identities remapped away via `id_map` | `identity_source` | Builds new `Dag`; populates `id_map: HashMap<NodeId, NodeId>` |
| `replace_dense_gather_patterns` | `specialize.rs:161–206` | `Dag` containing `OneHot`+`Expand`+`Mul`+`Sum` subgraphs | `Dag` with matches rewritten to `RiscOp::Gather { axis: 0 }` | `detect_dense_gather_pattern` | Builds new `Dag` |
| `lower_unmatched_one_hot` | `specialize.rs:208–246` | `Dag` containing `OneHot` nodes (post-gather-match) | `Dag` with unmatched `OneHot` lowered to `Const`/`CmpLt`/`MaxElem`/`Cast`/`Expand`/`Pad`/`Add` chain | None directly; checks `RiscOp::OneHot` and delegates to helper | Builds new `Dag`; calls `lower_one_hot_node` |
| `lower_one_hot_node` | `specialize.rs:248–359` | Mutable `Dag`, indices `NodeId`, source `DagNode`, vocab size | `NodeId` of final accumulated result | None | Mutates output `Dag` directly |
| `replace_matmul_patterns` | `specialize.rs:109–159` | `Dag` containing `Sum(Mul(Expand(A), Expand(B)))` subgraphs | `Dag` with matches rewritten to `RiscOp::BlasMatmul { batch_dims, m, n, k }` | `detect_matmul_pattern` | Builds new `Dag` |

### Pass order

The pipeline executes in fixed order inside `specialize_for_blas` (`specialize.rs:30–36`):

1. `eliminate_closed_list_noops(dag)` (`specialize.rs:31`)
2. `replace_dense_gather_patterns(&cleaned)` (`specialize.rs:32`)
3. `lower_unmatched_one_hot(&gathered)` (`specialize.rs:33`)
4. `replace_matmul_patterns(&lowered)` (`specialize.rs:34`)
5. `crate::optimize::dead_code_eliminate(&specialized)` (`specialize.rs:35`) — final cleanup, lives in a sibling module

The sequence is unconditional: there are no branches, no fixed-point loops, and no skip flags. The ordering contract is documented as code at `specialize.rs:17–27`.

### Input / output contract

**Input preconditions.** `specialize_for_blas` accepts a `Dag` that has already been processed by autodiff (`specialize.rs:3`, `specialize.rs:18`). Implicit preconditions are well-formed nodes, valid `NodeId` references, and reachable roots — these are required by the `id_map[id]` indexing pattern used by every pass (e.g. `specialize.rs:45`, `specialize.rs:133`, `specialize.rs:180`), which panics on missing IDs.

**Output postconditions.**
- No identity `Cast`, `Reshape`, or `Permute` nodes remain (`specialize.rs:80–107` recognizer drives elimination at `specialize.rs:46–51`).
- No dense-gather subgraphs of the recognized shape remain; matches are replaced with `RiscOp::Gather { axis: 0 }` (`specialize.rs:169–174`).
- No unmatched `OneHot` nodes remain; they are lowered to a `Const`/`CmpLt`/`MaxElem`/`Cast`/`Expand`/`Pad`/`Add` primitive chain (`specialize.rs:208–246`, lowering body at `specialize.rs:248–359`).
- F32 contiguous matmul subgraphs are replaced by `RiscOp::BlasMatmul { batch_dims, m, n, k }` (`specialize.rs:117–127`).
- Provenance is preserved: `span_id` is passed through each `add_node` call, `merged_spans` is migrated to output nodes (`specialize.rs:64–68`, `specialize.rs:145–149`, `specialize.rs:189–203`), and `reusable_input` is remapped through `id_map` (`specialize.rs:59–63`, `specialize.rs:140–144`).

**Invariants preserved.**
- Roots are remapped via `id_map` so the output `Dag`'s roots correspond to the same logical outputs as the input (`specialize.rs:72–76`, `specialize.rs:153–157`).
- `Dag` validity holds because every pass rebuilds via `out.add_node(...)` rather than in-place mutation of the input.
- F32-only `BlasMatmul` guard at `specialize.rs:519–520` rejects non-F32 sum nodes; the comment at `specialize.rs:514–518` explains this prevents silent miscompilation since `BlasMatmul` dispatches to `cblas_sgemm` / `hipblas_sgemm` (single-precision BLAS). A redundant operand-precision check at `specialize.rs:557–559` enforces F32 on both A and B.

### Sibling module dependencies

**Production dependencies:**
- `crate::dag::{Dag, DagNode, DimExpr, DimInfo, NodeId, RiscOp, TensorType}` (`specialize.rs:9`) — Core IR types.
- `crate::optimize::dead_code_eliminate` (`specialize.rs:35`) — Final cleanup in the pipeline.
- `crate::span_merge::{append_span_to_node, append_spans_to_node}` (`specialize.rs:709–710`) — Provenance preservation helpers (used by the `append_*_provenance` helpers in this module).
- `chelis_types::types::Prim` (`specialize.rs:10`) — External crate; provides the `Prim::F32` discriminant used by the matmul guards (`specialize.rs:519`, `specialize.rs:557`).
- `std::collections::HashMap` (`specialize.rs:7`) — `id_map` storage in every pass.

**Test-only dependencies:**
- `crate::tier2::lower_matmul` (`specialize.rs:980`, `specialize.rs:1024`).
- `crate::fuse::fuse` (`specialize.rs:1035`).
- `crate::verify::verify` (`specialize.rs:1106`, `specialize.rs:1195`, `specialize.rs:1231`).
- `crate::eval::{eval_tensor, TensorValue::from_vec}` (`specialize.rs:1113`, `specialize.rs:1120`, `specialize.rs:1123`, `specialize.rs:1124`, `specialize.rs:1237`, `specialize.rs:1239`, `specialize.rs:1240`).

## Part 2 — backend `emit.rs`

This part covers the three backends in tree — C (`crates/chelis-backend-c`), HIP (`crates/chelis-backend-hip`), and Metal (`crates/chelis-backend-metal`) — against the 42 `RiscOp` variants defined in `crates/chelis-ir/src/dag.rs:463–618`. Each `emit.rs` carries a single top-level `match &node.op` that dispatches every variant; this section gives the full RiscOp × backend dispatch matrix grouped by op family. Terminology used in the cells: **library call** = a math runtime entry point (libm `expf`, CBLAS, etc.); **GPU kernel** = a launch of a pre-written or generated device-side kernel; **generic kernel** = inline emitted host C or MSL (loops/branches) compiled with the rest of the artifact; **error** = a panic / `unreachable!` / `todo!` / `Err(...)` arm with no actual codegen.

### Backend inventory

| Backend | File | Lines | RiscOp dispatch site |
|---|---|---|---|
| C | `crates/chelis-backend-c/src/emit.rs` | 4039 | `emit_node()` at `c/emit.rs:285–442` |
| HIP | `crates/chelis-backend-hip/src/emit.rs` | 2609 | `emit_node()` at `hip/emit.rs:856–1044` |
| Metal | `crates/chelis-backend-metal/src/emit.rs` | 970 | `emit_node()` at `metal/emit.rs:381–428` |

### RiscOp inventory

All 42 variants enumerated by the `RiscOp` enum at `crates/chelis-ir/src/dag.rs:463–618`:

- **Binary elementwise (4):** `Add`, `Mul`, `CmpLt`, `MaxElem`
- **Unary elementwise (11):** `Neg`, `Exp`, `Log`, `Sin`, `Sqrt`, `Cos`, `Tan`, `Atan`, `Abs`, `Floor`, `Ceil`
- **RNG (2):** `UniformLike { low, high, seed }`, `Dropout { rate, seed }`
- **Reduction (6):** `Sum { axis }`, `MaxReduce { axis }`, `MinReduce { axis }`, `ProdReduce { axis }`, `Argmax { axis }`, `Argmin { axis }`
- **Movement (7):** `Reshape { new_shape }`, `Permute { axes }`, `Expand { axis, size }`, `OneHot { vocab }`, `Pad { padding, fill }`, `Shrink { bounds }`, `Stride { strides }`
- **Memory (6):** `Const { value }`, `Load { name }`, `Store { name }`, `Copy`, `Drop`, `Realize`
- **Cast (1):** `Cast { new_precision }`
- **Fusion (1):** `FusedElem { ops }`
- **Matmul (1):** `BlasMatmul { batch_dims, m, n, k }`
- **Sparse (3):** `Gather { axis }`, `ScatterAdd { axis }`, `Scatter { axis }`

### Section A — Binary elementwise

| Op | C | HIP | Metal |
|---|---|---|---|
| `Add` | Generic kernel via `emit_binary(id, "+", ...)` at `c/emit.rs:290`; inline host loop `t{a}->data[i] + t{b}->data[i]`. | GPU kernel — `emit_binary_launch` of `kernel_add` at `hip/emit.rs:861–863`. | Generic kernel — M2 rank-1 contiguous path via `emit_binary` at `metal/emit.rs:409`; MSL body `out[tid] = a[tid] + b[tid];`. |
| `Mul` | Generic kernel via `emit_binary(id, "*", ...)` at `c/emit.rs:291`. | GPU kernel — `kernel_mul` at `hip/emit.rs:864–866`. | Generic kernel — same `emit_binary` arm at `metal/emit.rs:409`. |
| `CmpLt` | Generic kernel via `emit_cmplt(...)` at `c/emit.rs:295`; branching compare returning f32 0.0/1.0. | GPU kernel — `kernel_cmplt` at `hip/emit.rs:870–872`. | Error — falls into the `other =>` arm at `metal/emit.rs:422–426`; not in M2 cut. |
| `MaxElem` | Generic kernel via `emit_binary_func(id, "fmaxf", ...)` at `c/emit.rs:292–294`. | GPU kernel — `kernel_max_elem` at `hip/emit.rs:867–869`. | Error at `metal/emit.rs:422–426`. |

### Section B — Unary elementwise

| Op | C | HIP | Metal |
|---|---|---|---|
| `Neg` | Generic kernel via `emit_unary(id, "-", ...)` at `c/emit.rs:296`. | GPU kernel — `kernel_neg` at `hip/emit.rs:873–875`. | Generic kernel — dispatched through shared unary arm at `metal/emit.rs:396–406`; inline MSL body `out[tid] = -a[tid];` constructed at `metal/emit.rs:575`. |
| `Exp` / `Log` / `Sin` / `Sqrt` / `Cos` / `Tan` / `Atan` / `Abs` / `Floor` / `Ceil` | Library call — libm functions `expf`, `logf`, `sinf`, `sqrtf`, `cosf`, `tanf`, `atanf`, `fabsf`, `floorf`, `ceilf` routed through `emit_unary_func()` one arm per op at `c/emit.rs:297–306`; double-precision variants chosen by `double_math_fn()` at `c/emit.rs:869`. | GPU kernel — one `kernel_{op}` per op (`kernel_exp`, `kernel_log`, `kernel_sin`, `kernel_sqrt`, `kernel_cos`, `kernel_tan`, `kernel_atan`, `kernel_abs`, `kernel_floor`, `kernel_ceil`), launched via `emit_unary_launch` at `hip/emit.rs:876–905`. | Generic kernel — all ten share the unary arm at `metal/emit.rs:396–406`; MSL function name selected by `kernels::unary_func(other)` at `metal/emit.rs:577`, producing `out[tid] = <fn>(a[tid]);`. |

### Section C — RNG

| Op | C | HIP | Metal |
|---|---|---|---|
| `UniformLike { low, high, seed }` | Generic kernel via `emit_uniform_like(...)` at `c/emit.rs:307–309`; inline host loop calling `chelis_uniform_sample_f32()` (SplitMix64 helper at `c/emit.rs:99–109`). | GPU kernel — `emit_uniform_like_launch` of `kernel_uniform_like` at `hip/emit.rs:906–908`. | Error — falls into the `other =>` arm at `metal/emit.rs:422–426`; not in M2 cut. |
| `Dropout { rate, seed }` | Error — `unreachable!("dropout should be rejected before C code generation")` at `c/emit.rs:310–312`. | Error — `unreachable!("dropout should be rejected before HIP code generation")` at `hip/emit.rs:909–911`. | Error — falls into the `other =>` arm at `metal/emit.rs:422–426`. |

### Section D — Reductions

| Op | C | HIP | Metal |
|---|---|---|---|
| `Sum{axis}` | Generic kernel via `emit_reduce_sum()` at `c/emit.rs:329`; fused-input path `emit_fused_reduce(..., "sum")` at `c/emit.rs:319–327`. Nested-loop accumulation. | GPU kernel via `emit_reduce_launch(..., ReduceKind::Sum)` at `hip/emit.rs:921` (`kernel_sum_stage{n}` family); fused-input path at `hip/emit.rs:919`. | Inline MSL kernel via `emit_reduce(dag, node, ReduceKind::Sum)` at `metal/emit.rs:412`. M4 first-cut: full-axis rank-1 → scalar only, single-threadgroup. |
| `MaxReduce{axis}` | Generic kernel via `emit_reduce_max()` at `c/emit.rs:346` using `fmaxf()`; fused-input path at `c/emit.rs:332–344`. | GPU kernel via `emit_reduce_launch(..., ReduceKind::Max)` at `hip/emit.rs:936`; fused-input path at `hip/emit.rs:934`. | Inline MSL kernel `ReduceKind::Max` at `metal/emit.rs:413`; same M4 constraints as `Sum`. |
| `MinReduce{axis}` | Generic kernel via `emit_reduce_simple(..., "INFINITY", "acc = fminf(...)", "chelis_min_f32")` at `c/emit.rs:350–359`. | Panic: `"HIP backend: MinReduce is not yet supported (Phase 3j-pre ships C backend only)"` at `hip/emit.rs:946–949`. | Inline MSL kernel `ReduceKind::Min` at `metal/emit.rs:414`. |
| `ProdReduce{axis}` | Generic kernel via `emit_reduce_simple(..., "1.0f", "acc *= ...", None)` at `c/emit.rs:361–371`. | Panic: `"HIP backend: ProdReduce is not yet supported (Phase 3j-pre ships C backend only)"` at `hip/emit.rs:951–954`. | Unsupported — falls through default arm error at `metal/emit.rs:422`. |
| `Argmax{axis}` | Generic kernel via `emit_reduce_argcmp(..., true)` at `c/emit.rs:374`; tracks max-value plus running index as floats. | Panic: `"HIP backend: Argmax is not yet supported (Phase 3j-pre ships C backend only)"` at `hip/emit.rs:956–959`. | Unsupported — default-arm error at `metal/emit.rs:422`. |
| `Argmin{axis}` | Generic kernel via `emit_reduce_argcmp(..., false)` at `c/emit.rs:377`. | Panic: `"HIP backend: Argmin is not yet supported (Phase 3j-pre ships C backend only)"` at `hip/emit.rs:961–964`. | Unsupported — default-arm error at `metal/emit.rs:422`. |

### Section E — Movement

| Op | C | HIP | Metal |
|---|---|---|---|
| `Reshape{new_shape}` | Generic kernel via `emit_reshape()` at `c/emit.rs:380`; view-over-buffer when rank + element-count match, else error at the reinterpret boundary. | GPU kernel via `emit_reshape()` at `hip/emit.rs:972`; device-side view remapping. | Unsupported — default-arm error at `metal/emit.rs:422`; not in M2 first-cut (rank-1 only). |
| `Permute{axes}` | Generic kernel via `emit_permute()` at `c/emit.rs:383`; nested loops over new strides. | GPU kernel via `emit_permute()` at `hip/emit.rs:975`. | Unsupported — default-arm error at `metal/emit.rs:422`. |
| `Expand{axis, size}` | Generic kernel via `emit_expand()` at `c/emit.rs:386`; broadcasts by reusing input pointer across the repeated axis. | GPU kernel via `emit_expand()` at `hip/emit.rs:978`. | Unsupported — default-arm error at `metal/emit.rs:422`. |
| `OneHot{vocab}` | Panic: `"C backend: internal OneHot must be consumed by specialization before codegen"` at `c/emit.rs:388–391`. | Panic: `"HIP backend: internal OneHot must be consumed by specialization before codegen"` at `hip/emit.rs:966–969`. | Unsupported — default-arm error at `metal/emit.rs:422`. |
| `Pad{padding, fill}` | Generic kernel via `emit_pad()` at `c/emit.rs:394`. | `todo!("Pad on GPU requires a kernel. Deferred to Phase 1a iteration 2")` at `hip/emit.rs:980–981`. | Unsupported — default-arm error at `metal/emit.rs:422`. |
| `Shrink{bounds}` | Generic kernel via `emit_shrink()` at `c/emit.rs:397`; view over an offset pointer using strides. | `todo!("Shrink on GPU requires a kernel. Deferred to Phase 1a iteration 2")` at `hip/emit.rs:983–984`. | Unsupported — default-arm error at `metal/emit.rs:422`. |
| `Stride{strides}` | Generic kernel via `emit_stride()` at `c/emit.rs:400`; reinterprets strides on the same buffer. | GPU kernel via `emit_stride()` at `hip/emit.rs:987`. | Unsupported — default-arm error at `metal/emit.rs:422`. |

### Section F — Memory & lifecycle

| Op | C | HIP | Metal |
|---|---|---|---|
| `Const{value}` | Library call: `emit_const()` at `c/emit.rs:288` — allocates buffer, fills with CPU loop calling `chelis_fill_f32()` or `chelis_fill_f64()`. | GPU kernel: `emit_const()` at `hip/emit.rs:859` — `kernel_fill` with fill value + size args. | Generic kernel: `emit_const()` at `metal/emit.rs:392` — CPU loop fills MTL device buffer via `[buf contents]`. |
| `Load{name}` | Handled separately: `unreachable!` at `c/emit.rs:289`. Pre-processed in `emit_dag()` at `c/emit.rs:181–186` — emits host input view wrapper + symbolic dim declarations. | Handled separately: `unreachable!` at `hip/emit.rs:860`. Pre-processed in `emit_dag()` at `hip/emit.rs:158–163` and device entrypoint at `hip/emit.rs:297–301` — host→device transfer or alias. | Library call dispatched at `metal/emit.rs:390`; `emit_load()` allocates device buffer + `chelis_metal_host_to_device()` (helper at `metal/emit.rs:514`). |
| `Store{name}` | Dispatcher: `emit_store()` at `c/emit.rs:404` — routes output spec to writeback phase at `c/emit.rs:194–201`. | Dispatcher: `emit_store()` at `hip/emit.rs:996` — device-side tensor naming for output mapping. | Dispatcher: `emit_store()` at `metal/emit.rs:391` — allocates host tensor + device→host transfer (`metal/emit.rs:936`). |
| `Copy` | Generic kernel: `emit_realize()` for contiguous materialization at `c/emit.rs:313`. | GPU kernel: `emit_unary_launch(id, "kernel_cast", ...)` at `hip/emit.rs:913` — identity-cast kernel. | No-op: `Ok(())` at `metal/emit.rs:393`. |
| `Drop` | No-op: empty block at `c/emit.rs:314`. | No-op at `hip/emit.rs:915`. | No-op at `metal/emit.rs:393`. |
| `Realize` | Generic kernel: `emit_realize()` at `c/emit.rs:402` — materializes contiguous tensor. | GPU kernel: `emit_unary_launch(id, "kernel_cast", ...)` at `hip/emit.rs:990`. | Error at default arm `metal/emit.rs:422`. |

### Section G — Cast

| Op | C | HIP | Metal |
|---|---|---|---|
| `Cast{new_precision}` | Generic kernel via `emit_cast()` at `c/emit.rs:403` — element-wise conversion loop. Codegen-time validation of target precision (F32/F64/Int32/Int64) at `c/emit.rs:558–569`. | GPU kernel: `emit_unary_launch(id, "kernel_cast", ...)` at `hip/emit.rs:993` — same identity-cast kernel as `Copy` for same-type ops. | Error at default arm `metal/emit.rs:422`. |

### Section H — Fusion

| Op | C | HIP | Metal |
|---|---|---|---|
| `FusedElem{ops}` | Generic kernel + in-place optimization: `emit_fused_elem()` at `c/emit.rs:405–411`. Inlined loop chain per `ops` vector; in-place reuse detection via `fused_in_place_spec()` at `c/emit.rs:407`. Can be inlined into trailing `Sum` (`c/emit.rs:317–327`) or `MaxReduce` (`c/emit.rs:334–344`). | GPU kernel: `emit_fused_launch()` at `hip/emit.rs:998–1012`. Generates per-node `kernel_fused_{id}`. Same in-place detection + reduction-inlining. | Error at default arm `metal/emit.rs:422`. |

### Section I — Matmul

| Op | C | HIP | Metal |
|---|---|---|---|
| `BlasMatmul{batch_dims, m, n, k}` | Library call: `emit_blas_matmul()` dispatched at `c/emit.rs:413–430`. Routes to `cblas_sgemm()` (F32) or `cblas_dgemm()` (F64) with layout-aware strides for batched case (implementation at `c/emit.rs:1809`). | Library call: `emit_blas_matmul()` dispatched at `hip/emit.rs:1014–1032`. Routes to `hipblasGemmEx()` or strided-batch variant. | Inline MSL kernel: matmul subgraph detected in pre-walk at `metal/emit.rs:251–267`; `Sum{axis:1}` dispatches via `self.matmuls` lookup to `emit_matmul()` at `metal/emit.rs:417–420`. Tiled MSL kernel `k_matmul_{id}` via `kernels::matmul_tiled_kernel()` — M5 first-cut. |

### Section J — Sparse

| Op | C | HIP | Metal |
|---|---|---|---|
| `Gather{axis}` | Generic kernel via `emit_sparse_gather()` at `c/emit.rs:432–433`. Nested-loop indexing with bounds-check. Index-type validation (Int32/Int64) at `c/emit.rs:593–611`. | GPU kernel: `emit_gather_launch()` at `hip/emit.rs:1034–1035`. | Error at default arm `metal/emit.rs:422`. |
| `ScatterAdd{axis}` | Generic kernel via `emit_sparse_scatter_add()` at `c/emit.rs:435–436`. Index validation at `c/emit.rs:613–634`. | GPU kernel: `emit_scatter_add_launch()` at `hip/emit.rs:1037–1038` — atomic add. | Error at default arm `metal/emit.rs:422`. |
| `Scatter{axis}` | Generic kernel via `emit_sparse_scatter_replace()` at `c/emit.rs:438–439`. Last-write-wins. Index validation at `c/emit.rs:636–658`. | GPU kernel: `emit_scatter_replace_launch()` at `hip/emit.rs:1040–1041`. | Error at default arm `metal/emit.rs:422`. |

### Backend-specific quirks

#### C

- Reduction inlining via the `reduction_inlined` set, pre-computed by `chelis_ir::fuse::reduction_inlined_fused_elems()` at `c/emit.rs:70`. `Sum` / `MaxReduce` dispatch to `emit_fused_reduce()` if input is in the set, avoiding standalone materialization.
- `Dropout` rejection enforced at codegen via `unreachable!()` at `c/emit.rs:310`.
- `OneHot` rejection at `c/emit.rs:388–391` — specialization must lower before C emission.
- Symbolic dimensions: preamble validation at `c/emit.rs:737–772` binds symbolic dims from `Load` input shapes; rejects at runtime if mismatched across inputs.
- Supported precisions: F32 / F64 / Bool / Int32 / Int64 only — validated at `c/emit.rs:548–570`.
- BLAS integration: conditional on `use_blas` flag; CBLAS symbols routed via `chelis_blas.h` include guard at `c/emit.rs:92–94`.

#### HIP

- Reduction inlining mirrors C (`Sum` at `hip/emit.rs:917–929`, `MaxReduce` at `hip/emit.rs:931–944`) via `emit_fused_reduce_launch()`.
- Phase 3j-pre ships C backend only; GPU reductions for `MinReduce` / `ProdReduce` / `Argmax` / `Argmin` panic explicitly with phase callout at `hip/emit.rs:946–964`.
- `Pad` / `Shrink` are `todo!()` at `hip/emit.rs:980–984`, deferred to Phase 1a iteration 2.
- Separate `{func_name}_device()` entrypoint at `hip/emit.rs:250–325` for GPU-internal call chains (no host↔device transfer). Memory plan pre-allocates all slots at entrypoint scope at `hip/emit.rs:291–295`.
- `Dropout` `unreachable!()` at `hip/emit.rs:910`.

#### Metal

- M-phase gates: M2 = elementwise rank-1; M4 = full-axis reductions to scalar (rank-1 input); M5 = matmul.
- Unimplemented ops return `Err(...)` at the default arm `metal/emit.rs:422` rather than panicking — surfaces as a build-time codegen error to the caller, not silent failure and not a hard panic.
- Shape constraints: `require_static_rank1()` at `metal/emit.rs:430` enforces rank-1 + literal extent for most ops; `require_static_shape()` at `metal/emit.rs:455` allows rank-1 or rank-2 for Load/Store/matmul operands.
- Matmul subgraph pre-walk at `metal/emit.rs:251–267`: `Expand` / `Mul` intermediates marked in `matmul_consumed` set; `Sum{axis:1}` dispatches via `self.matmuls` lookup at `metal/emit.rs:417`.
- Single-threadgroup reduction cap `SINGLE_TG_LIMIT = 4096` at `metal/emit.rs:745`; larger inputs rejected at `metal/emit.rs:747` pending M4.next two-pass implementation.
- No slot reuse: every materialized node owns one device buffer; peak-bytes sum at `metal/emit.rs:949`.
- Span comments: host-side at `metal/emit.rs:218–221`, per-kernel MSL string prefix at `metal/emit.rs:228–240`.

### Matrix completeness summary

All counts below are over the 42 variants enumerated in `dag.rs:463–618`.

- **C (42 dispatched).** Every variant has a match arm. Two intentional rejections via `unreachable!()` (`Dropout` at `c/emit.rs:310`, `OneHot` at `c/emit.rs:388–391`) — both must be lowered by upstream passes before reaching the C backend. The remaining 40 variants have real implementations: `BlasMatmul` and `Const` route to libm/CBLAS-style library calls; the rest are inline generic kernels.
- **HIP (42 dispatched).** Intentional `unreachable!()` for `Dropout` (`hip/emit.rs:910`) and `OneHot` (`hip/emit.rs:966–969`) mirroring C. Phase-deferred panics for `MinReduce`, `ProdReduce`, `Argmax`, `Argmin` (`hip/emit.rs:946–964`). `todo!()` for `Pad` and `Shrink` (`hip/emit.rs:980–984`). The remaining 34 variants have implementations: `BlasMatmul` is a hipBLAS library call, `Load` is pre-processed in `emit_dag()`, `Store` and `Drop` are dispatcher / no-op respectively, and everything else is a GPU kernel launch.
- **Metal.** Implemented per M-phase gates: M2 covers `Add`, `Mul`, the 11-op unary family, `Const`, `Load`, `Store`, `Copy` (no-op), `Drop` (no-op); M4 adds `Sum`, `MaxReduce`, `MinReduce`; M5 adds `BlasMatmul` via subgraph detection — roughly 22 variants. The remaining 20 variants fall through to the structured `Err(...)` default arm at `metal/emit.rs:422`, surfaced as a codegen error to the caller.

### Appendix — RiscOp reference

- Canonical `RiscOp` enum definition lives at `crates/chelis-ir/src/dag.rs:463–618`.
- The matrix sections above (A–J) are organized in the same family order used by that enum, so a reader can scan the source enum top-to-bottom and find the corresponding backend cell in this report.
- Note: the `dag.rs` source places `UniformLike` and `Dropout` under the `// --- Unary elementwise ---` comment header (`dag.rs:470–490`), but this report categorizes them as a separate RNG family (Section C) because their backend dispatch — stateful seeds, RNG kernels, codegen rejection for `Dropout` — is unlike the pure unary ops in Section B.
