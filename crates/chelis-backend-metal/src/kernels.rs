//! MSL kernel templates for the Metal backend.
//!
//! Each function returns a complete MSL source string ready for runtime
//! compilation via `[MTLDevice newLibraryWithSource:options:error:]`. The
//! kernels assume same-shape, contiguous, rank-1 tensors as the M2 first
//! cut. Strided/broadcast/rank>1 layouts are tracked as M-phase follow-ups
//! (the emitter currently rejects those at the host site rather than
//! emitting a kernel that would silently miscompile the index math).
//!
//! ## Conventions
//!
//! Buffers are bound at `[[buffer(0)..n_inputs-1)]]` (inputs, in DAG-input
//! order), then `[[buffer(n_inputs..n_inputs+n_outputs-1)]]` (outputs), then
//! a final `constant uint& n [[buffer(n_inputs+n_outputs)]]` carrying the
//! element count. All elementwise kernels are 1D over `tid`.
//!
//! ## Mapping from HIP
//!
//! | Concept | HIP | MSL |
//! |---|---|---|
//! | kernel decl | `extern "C" __global__ void` | `kernel void` |
//! | tid | `blockIdx.x * blockDim.x + threadIdx.x` | `uint tid [[thread_position_in_grid]]` |
//! | in ptr | `const float*` | `device const float*` |
//! | out ptr | `float*` | `device float*` |
//! | scalar uniform | function arg | `constant uint& n [[buffer(N)]]` |
//! | math | `expf/logf/sqrtf/sinf` | `exp/log/sqrt/sin` (overloaded) |
//! | header | `#include <hip/hip_runtime.h>` | `#include <metal_stdlib>\nusing namespace metal;` |
//!
//! Bool tensors are emitted as `device const bool*` directly. Verified in
//! the M2 prelude on Apple Silicon (M3, GPU family 7+); no uchar
//! workaround needed.
//!
//! ## WS-M1: per-dtype parameterization
//!
//! All templates here that previously hardcoded `float` are now
//! parameterized through [`crate::dtype::msl_type`] / [`crate::dtype::kernel_suffix`].
//! See spec/04-type-system.md §1.1.3 for the per-backend matrix and
//! §5.7.1 for reduction-accumulator promotion. `bf16` kernels gate
//! their MSL `bfloat` use behind `#if __METAL_VERSION__ >= 320`
//! (Apple7+ requirement).

use chelis_types::types::Prim;

use crate::dtype;

/// Returns the MSL header block: `#include <metal_stdlib>` + namespace
/// import. Wraps the body in a `bfloat`-version `#if` guard when any
/// participating dtype requires MSL 3.2 / Apple7+ (bf16 today, per
/// spec/04-type-system.md §1.1.3). The caller composes the guarded
/// region; this helper only emits the unconditional header.
fn msl_header() -> &'static str {
    "#include <metal_stdlib>\nusing namespace metal;\n\n"
}

/// Wrap a kernel body in `#if __METAL_VERSION__ >= 320 ... #endif` when
/// any participating precision requires it (today: bf16 only). Returns
/// the body unchanged otherwise. The wrap puts the guard outside the
/// kernel function so a pre-Apple7 build still emits a syntactically
/// valid translation unit; the guarded region simply produces no
/// `kernel void` declaration on those builds, and the runtime's
/// pipeline-creation step surfaces a clean diagnostic when the user
/// dispatches a missing kernel.
fn maybe_wrap_msl_320(body: String, precs: &[Prim]) -> String {
    if precs.iter().any(|p| dtype::requires_msl_320_guard(*p)) {
        format!("#if __METAL_VERSION__ >= 320\n{body}#endif\n")
    } else {
        body
    }
}

/// Emit a complete elementwise kernel source string.
///
/// `kernel_name`: the MSL function name, also used as the cache key
/// `params`: list of parameter declarations (one per buffer) in order.
/// `body`: the per-element body, with `tid` already bounds-checked. The
///         body may reference parameter names directly.
///
/// Returns a self-contained MSL source string suitable for embedding in a
/// `static NSString *const ... = @R"MSL(...)MSL";` literal.
pub fn elementwise_kernel(kernel_name: &str, params: &[String], body: &str) -> String {
    let header = msl_header();
    let params_block = params.join(",\n    ");
    format!(
        "{header}kernel void {kernel_name}(
    {params_block},
    constant uint& n [[buffer({n_idx})]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= n) return;
{body}
}}
",
        n_idx = params.len()
    )
}

/// WS-M1: same as [`elementwise_kernel`] but wraps the body in the
/// `#if __METAL_VERSION__ >= 320` guard when any of `precs` requires
/// it (bf16 today). Preferred call shape for new emit sites.
pub fn elementwise_kernel_for(
    kernel_name: &str,
    params: &[String],
    body: &str,
    precs: &[Prim],
) -> String {
    let raw = elementwise_kernel(kernel_name, params, body);
    maybe_wrap_msl_320(raw, precs)
}

/// Build a parameter declaration for a `device const T*` input buffer at
/// `[[buffer(idx)]]`.
pub fn input_param(idx: usize, ty: &str, name: &str) -> String {
    format!("device const {ty}* {name} [[buffer({idx})]]")
}

/// Build a parameter declaration for a `device T*` output buffer at
/// `[[buffer(idx)]]`.
pub fn output_param(idx: usize, ty: &str, name: &str) -> String {
    format!("device {ty}* {name} [[buffer({idx})]]")
}

/// Map a Chelis precision to the MSL type spelling. Thin re-export of
/// [`crate::dtype::msl_type`] kept here so existing call sites compile
/// without churn.
pub fn msl_type(prec: Prim) -> &'static str {
    dtype::msl_type(prec)
}

/// MSL spelling for a unary math intrinsic.
///
/// Note: MSL's default `exp`/`log`/`sqrt`/`sin` are fast-math variants. M6
/// will widen the Metal-specific f32 tolerance for transcendental-heavy
/// kernels, and switch individual call sites to `precise::exp` etc. when
/// numerical agreement requires it.
pub fn unary_func(op: &chelis_ir::dag::RiscOp) -> Option<&'static str> {
    use chelis_ir::dag::RiscOp;
    match op {
        RiscOp::Neg => Some("-"),
        RiscOp::Exp => Some("exp"),
        RiscOp::Log => Some("log"),
        RiscOp::Sqrt => Some("sqrt"),
        RiscOp::Sin => Some("sin"),
        RiscOp::Cos => Some("cos"),
        RiscOp::Tan => Some("tan"),
        RiscOp::Atan => Some("atan"),
        RiscOp::Abs => Some("fabs"),
        RiscOp::Floor => Some("floor"),
        RiscOp::Ceil => Some("ceil"),
        _ => None,
    }
}

/// MSL spelling for a binary elementwise operator.
pub fn binary_op(op: &chelis_ir::dag::RiscOp) -> Option<&'static str> {
    use chelis_ir::dag::RiscOp;
    match op {
        RiscOp::Add => Some("+"),
        RiscOp::Mul => Some("*"),
        _ => None,
    }
}

/// Reduction kind for the M4 reduction kernel templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
}

impl ReduceKind {
    /// Short label used in kernel names (`reduce_sum`, `reduce_max`, …).
    pub fn label(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
        }
    }

    /// MSL identity element string for the reduction at the given
    /// accumulator precision. Sums use the dtype's `0`, max uses the
    /// dtype's minimum-representable value, min uses the maximum.
    /// Integer types use literal zero/INT*_MIN/INT*_MAX rather than
    /// `INFINITY` so the resulting kernel actually compiles.
    pub fn identity(self, prec: Prim) -> &'static str {
        match (self, prec) {
            (Self::Sum, Prim::F32) => "0.0f",
            (Self::Sum, Prim::F16) => "(half)0.0h",
            (Self::Sum, Prim::Bf16) => "(bfloat)0.0",
            (Self::Sum, Prim::Int8) => "(char)0",
            (Self::Sum, Prim::Int16) => "(short)0",
            (Self::Sum, Prim::Int32) => "0",
            (Self::Sum, Prim::Int64) => "0L",
            (Self::Sum, Prim::Bool) => "false",
            (Self::Max, Prim::F32) => "-INFINITY",
            (Self::Max, Prim::F16) => "(half)(-65504.0)",
            (Self::Max, Prim::Bf16) => "(bfloat)(-INFINITY)",
            (Self::Max, Prim::Int8) => "(char)(-128)",
            (Self::Max, Prim::Int16) => "(short)(-32768)",
            (Self::Max, Prim::Int32) => "(int)(-2147483647 - 1)",
            (Self::Max, Prim::Int64) => "(long)(-9223372036854775807L - 1L)",
            (Self::Max, Prim::Bool) => "false",
            (Self::Min, Prim::F32) => "INFINITY",
            (Self::Min, Prim::F16) => "(half)65504.0",
            (Self::Min, Prim::Bf16) => "(bfloat)INFINITY",
            (Self::Min, Prim::Int8) => "(char)127",
            (Self::Min, Prim::Int16) => "(short)32767",
            (Self::Min, Prim::Int32) => "2147483647",
            (Self::Min, Prim::Int64) => "9223372036854775807L",
            (Self::Min, Prim::Bool) => "true",
            (kind, prec) => panic!(
                "Metal reduction identity not defined for ({kind:?}, {}); see spec/04-type-system.md §5.7.1",
                prec.name()
            ),
        }
    }

    /// Combine expression at the accumulator dtype: given two MSL
    /// expressions `lhs` and `rhs`, produce the merged scalar.
    pub fn combine_at(self, prec: Prim, lhs: &str, rhs: &str) -> String {
        let _ = prec; // currently MSL handles the operator overloads itself.
        match self {
            Self::Sum => format!("({lhs}) + ({rhs})"),
            Self::Max => format!("max({lhs}, {rhs})"),
            Self::Min => format!("min({lhs}, {rhs})"),
        }
    }
}

/// Tile size for the M5 first-cut tiled matmul kernel. 16x16 is the
/// canonical tile size for an Apple-Silicon-friendly tiled GEMM and what
/// the upstream Metal-backend plan committed to.
pub const MATMUL_TILE: usize = 16;

/// Emit the tiled MSL matmul kernel parameterized over operand precision.
///
/// Computes `C[M, N] = A[M, K] @ B[K, N]` for row-major contiguous
/// matrices via a 16x16 tile. Threadgroup memory caches one tile of A
/// and one tile of B per outer iteration; each thread accumulates one
/// output element. The kernel is dispatched with an `MxN` grid in
/// 16x16 threadgroups; the host picks tg = (16, 16, 1).
///
/// `prec` parameterizes the buffer element type AND the threadgroup
/// tile element type (operand precision is preserved in shared memory
/// to keep the threadgroup footprint matched to the device buffer
/// footprint). The accumulator type is selected by
/// [`dtype::matmul_accumulator`] per spec/04-type-system.md §5.7.1:
/// bf16 and f16 promote to f32 for the per-thread accumulator and the
/// inner-product partial product is computed in f32 (operands are
/// explicitly cast at the multiply, not just at `acc +=`). f32 keeps
/// f32 accumulator with no cast.
///
/// `bf16` operand precision wraps the kernel in the
/// `#if __METAL_VERSION__ >= 320` guard required by the MSL `bfloat`
/// type. Integer matmul is rejected at type-check per
/// spec/04-type-system.md §5.7.2 and at the F1 codegen guard in
/// `emit::Emitter::emit_matmul`; this template still admits integer
/// dtypes defensively for future generalization (using same-type
/// accumulator for ints).
pub fn matmul_tiled_kernel_for(kernel_name: &str, prec: Prim) -> String {
    let acc_prec = dtype::matmul_accumulator(prec);
    matmul_tiled_kernel_with_acc(kernel_name, prec, acc_prec)
}

/// Variant of [`matmul_tiled_kernel_for`] that takes the accumulator
/// precision explicitly. Callers that already know the IR-pinned
/// accumulator (e.g., from a `BlasMatmul` node) should use this entry
/// point so the kernel template never has to re-derive the spec
/// promotion rules.
pub fn matmul_tiled_kernel_with_acc(
    kernel_name: &str,
    operand_prec: Prim,
    accumulator_prec: Prim,
) -> String {
    let tile = MATMUL_TILE;
    let ty = msl_type(operand_prec);
    let acc_ty = msl_type(accumulator_prec);
    let identity = ReduceKind::Sum.identity(operand_prec);
    let acc_identity = ReduceKind::Sum.identity(accumulator_prec);
    // When the accumulator differs from the operand, the partial product
    // tileA*tileB must compute in accumulator precision (spec §5.7.1) —
    // otherwise the multiplication happens at operand precision and
    // truncates before the accumulator widens it. Cast both operands at
    // the multiply, not just the result.
    let mul_expr = if accumulator_prec == operand_prec {
        "tileA[lid.y][i] * tileB[i][lid.x]".to_string()
    } else {
        format!("({acc_ty})tileA[lid.y][i] * ({acc_ty})tileB[i][lid.x]")
    };
    // Output is at operand precision; downcast `acc` at write-out time
    // when the accumulator widened.
    let writeback_expr = if accumulator_prec == operand_prec {
        "acc".to_string()
    } else {
        format!("({ty})acc")
    };
    // M, N, K are packed into a single uniform struct bound at buffer(3).
    // Binding them as three separate `constant uint&` parameters at
    // distinct buffer indices would require three `setBytes:atIndex:` calls
    // from the host, but `chelis_metal_launch2d` only binds one uniforms
    // blob, so a single struct keeps the kernel and host in lockstep.
    let body = format!(
        "#include <metal_stdlib>
using namespace metal;

struct ChelisMatmulDims {{
    uint M;
    uint N;
    uint K;
}};

kernel void {kernel_name}(
    device const {ty}* A [[buffer(0)]],
    device const {ty}* B [[buffer(1)]],
    device {ty}* C [[buffer(2)]],
    constant ChelisMatmulDims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {{
    const uint TILE = {tile};
    const uint M = dims.M;
    const uint N = dims.N;
    const uint K = dims.K;
    // Tiles cache operand bytes so the threadgroup memory footprint matches
    // the device buffer footprint. The accumulator promotion happens at
    // the multiply, not in the tile cache (spec/04-type-system.md §5.7.1).
    threadgroup {ty} tileA[{tile}][{tile}];
    threadgroup {ty} tileB[{tile}][{tile}];

    {acc_ty} acc = {acc_identity};
    uint num_tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {{
        uint aRow = gid.y;
        uint aCol = t * TILE + lid.x;
        uint bRow = t * TILE + lid.y;
        uint bCol = gid.x;
        tileA[lid.y][lid.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : ({ty}){identity};
        tileB[lid.y][lid.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : ({ty}){identity};
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {{
            acc += {mul_expr};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (gid.y < M && gid.x < N) {{
        C[gid.y * N + gid.x] = {writeback_expr};
    }}
}}
"
    );
    maybe_wrap_msl_320(body, &[operand_prec, accumulator_prec])
}

/// f32-only legacy entry point retained for any callers that haven't
/// migrated to [`matmul_tiled_kernel_for`]. New code should call the
/// `_for` variant.
pub fn matmul_tiled_kernel(kernel_name: &str) -> String {
    matmul_tiled_kernel_for(kernel_name, Prim::F32)
}

/// Threadgroup size for the single-threadgroup reduction kernel.
///
/// Must be a power of two so the tree-reduction halving loop terminates
/// without dropping upper-half entries. The wrap loop in the kernel
/// handles arbitrary `n` by striding `tg_size` at a time, so the
/// threadgroup size is decoupled from `n`. The emitter pads `shared[]`
/// with the reduction's identity element for `lid >= tg_size` (vacuously
/// true here since `tg_size == REDUCE_TG_SIZE` covers the array), but
/// per-iteration the kernel guards against out-of-bounds writes via
/// the `lid + stride < tg_size` check.
pub const REDUCE_TG_SIZE: usize = 256;

/// Emit a 1D reduction kernel parameterized on operand and accumulator
/// dtype per spec §5.7.1.
///
/// Reduces a contiguous `n`-element buffer of `operand_prec` to a single
/// scalar of `accumulator_prec` via threadgroup memory + tree reduction.
/// For `reduce_sum`, `accumulator_prec` is the spec-required widened
/// dtype (f32 for f16/bf16; i32 for i8/i16; otherwise same as operand).
/// For `max`/`min`, `accumulator_prec == operand_prec` per the spec.
///
/// Dispatched with a fixed power-of-two threadgroup size
/// ([`REDUCE_TG_SIZE`] = 256). The wrap loop strides `n` at `tg_size`
/// per iteration so any `n <= REDUCE_TG_SIZE * REDUCE_TG_SIZE` (with
/// the same single-threadgroup constraint) reduces in one pass.
///
/// Two-pass reduction for `n > REDUCE_TG_SIZE * REDUCE_TG_SIZE` is
/// M4.next; the emitter rejects oversized inputs at the host site.
pub fn reduce_full_kernel_for(
    kernel_name: &str,
    kind: ReduceKind,
    operand_prec: Prim,
    accumulator_prec: Prim,
) -> String {
    let in_ty = msl_type(operand_prec);
    let acc_ty = msl_type(accumulator_prec);
    let identity = kind.identity(accumulator_prec);
    let tg = REDUCE_TG_SIZE;
    // Promote each input element to the accumulator dtype before
    // combining so narrow-float / narrow-int sums don't silently
    // saturate (per spec/04-type-system.md §5.7.1).
    let elem_promoted = if operand_prec == accumulator_prec {
        "input[lid + i]".to_string()
    } else {
        format!("({acc_ty})input[lid + i]")
    };
    let combine_acc_input = kind.combine_at(accumulator_prec, "acc", &elem_promoted);
    let combine_pair = kind.combine_at(accumulator_prec, "shared[lid]", "shared[lid + stride]");
    let body = format!(
        "#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const {in_ty}* input [[buffer(0)]],
    device {acc_ty}* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    // Threadgroup size is fixed at compile time so the tree-reduce halving
    // terminates exactly. The host always dispatches a single threadgroup
    // of TG_SIZE threads regardless of n; the wrap loop below handles any
    // n by striding TG_SIZE elements per iteration.
    const uint TG_SIZE = {tg};
    threadgroup {acc_ty} shared[{tg}];

    {acc_ty} acc = {identity};
    // Threads with lid >= n skip the wrap loop and seed `acc` with the
    // identity, keeping the tree reduction well-defined for any n.
    for (uint i = 0; i + lid < n; i += TG_SIZE) {{
        acc = {combine_acc_input};
    }}
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Power-of-two tree reduction. Because TG_SIZE is a power of two,
    // every halved stride is exact and no shared[] entry is dropped.
    for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {{
        if (lid < stride) {{
            shared[lid] = {combine_pair};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (lid == 0) {{
        output[0] = shared[0];
    }}
}}
"
    );
    maybe_wrap_msl_320(body, &[operand_prec, accumulator_prec])
}

/// f32-only legacy entry point retained for callers that haven't
/// migrated to [`reduce_full_kernel_for`]. New code should call the
/// `_for` variant.
pub fn reduce_full_kernel(kernel_name: &str, kind: ReduceKind) -> String {
    reduce_full_kernel_for(kernel_name, kind, Prim::F32, Prim::F32)
}

/// Maximum tensor rank supported by the movement-op uniform structs.
/// Mirrors the HIP backend's `MAX_DIM` so both GPU paths cap rank
/// identically. Pad/shrink encode `src_shape`, `out_shape`, and the
/// per-axis offset vector as fixed-size arrays in the MSL uniform block.
pub const MOVEMENT_MAX_DIM: usize = 8;

/// Emit the shared MSL `ChelisMovementDims` uniform struct + the
/// contiguous-stride / flat-index helpers used by the pad and shrink
/// kernels. The Metal backend materializes every tensor contiguously
/// (it has no strided-view path), so the kernels reconstruct row-major
/// strides from the shape inline rather than receiving them as inputs.
fn movement_helpers() -> String {
    let max_dim = MOVEMENT_MAX_DIM;
    format!(
        "struct ChelisMovementDims {{
    uint ndim;
    uint total;
    uint src_shape[{max_dim}];
    uint out_shape[{max_dim}];
    uint offset[{max_dim}];
}};

// Decompose a flat row-major index into per-axis indices over `shape`.
static inline void chelis_flat_to_indices(uint flat, constant uint* shape, uint ndim, thread uint* out) {{
    for (uint d = ndim; d-- > 0; ) {{
        out[d] = flat % shape[d];
        flat /= shape[d];
    }}
}}

// Row-major flat index of per-axis `indices` over a contiguous `shape`.
static inline uint chelis_indices_to_flat(thread const uint* indices, constant uint* shape, uint ndim) {{
    uint flat = 0;
    for (uint d = 0; d < ndim; d++) {{
        flat = flat * shape[d] + indices[d];
    }}
    return flat;
}}
"
    )
}

/// MSL `pad` kernel (typed; one thread per output element).
///
/// The output buffer is contiguous; thread `tid` is the output flat
/// index. Recover the per-axis output indices, subtract the per-axis low
/// padding (`offset[d]`) to get the source index, and copy the source
/// element when every source index lies in `[0, src_shape[d])`; otherwise
/// write `fill`. Semantics mirror the C backend `emit_pad`, the evaluator
/// `pad`, and the HIP `pad_typed` kernel (spec/05-risc-primitives.md
/// §2.4). bf16 outputs wrap in the `__METAL_VERSION__ >= 320` guard.
pub fn pad_kernel(kernel_name: &str, prec: Prim) -> String {
    let ty = msl_type(prec);
    let max_dim = MOVEMENT_MAX_DIM;
    let helpers = movement_helpers();
    let body = format!(
        "#include <metal_stdlib>
using namespace metal;

{helpers}
kernel void {kernel_name}(
    device const {ty}* a [[buffer(0)]],
    device {ty}* out [[buffer(1)]],
    constant ChelisMovementDims& dims [[buffer(2)]],
    constant {ty}& fill [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= dims.total) return;
    uint out_idx[{max_dim}];
    chelis_flat_to_indices(tid, dims.out_shape, dims.ndim, out_idx);
    uint src_idx[{max_dim}];
    bool in_source = true;
    for (uint d = 0; d < dims.ndim; d++) {{
        int s = (int)out_idx[d] - (int)dims.offset[d];
        if (s < 0 || s >= (int)dims.src_shape[d]) {{
            in_source = false;
        }}
        src_idx[d] = (uint)(s < 0 ? 0 : s);
    }}
    if (in_source) {{
        uint flat = chelis_indices_to_flat(src_idx, dims.src_shape, dims.ndim);
        out[tid] = a[flat];
    }} else {{
        out[tid] = fill;
    }}
}}
"
    );
    maybe_wrap_msl_320(body, &[prec])
}

/// MSL `shrink` kernel (typed; one thread per output element).
///
/// One thread per (contiguous) output element reads the source at
/// `out_index + offset` where `offset[d]` is `bounds[d].0`. The shrink
/// output is always strictly inside the source, so no bounds margin
/// exists. Mirrors the C backend `emit_shrink`, the evaluator `shrink`,
/// and the HIP `shrink_typed` kernel (spec/05-risc-primitives.md §2.4).
pub fn shrink_kernel(kernel_name: &str, prec: Prim) -> String {
    let ty = msl_type(prec);
    let max_dim = MOVEMENT_MAX_DIM;
    let helpers = movement_helpers();
    let body = format!(
        "#include <metal_stdlib>
using namespace metal;

{helpers}
kernel void {kernel_name}(
    device const {ty}* a [[buffer(0)]],
    device {ty}* out [[buffer(1)]],
    constant ChelisMovementDims& dims [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= dims.total) return;
    uint out_idx[{max_dim}];
    chelis_flat_to_indices(tid, dims.out_shape, dims.ndim, out_idx);
    uint src_idx[{max_dim}];
    for (uint d = 0; d < dims.ndim; d++) {{
        src_idx[d] = out_idx[d] + dims.offset[d];
    }}
    uint flat = chelis_indices_to_flat(src_idx, dims.src_shape, dims.ndim);
    out[tid] = a[flat];
}}
"
    );
    maybe_wrap_msl_320(body, &[prec])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_template_uses_dtype_for_buffers() {
        let src = matmul_tiled_kernel_for("k_matmul_test", Prim::F16);
        assert!(src.contains("device const half* A"));
        assert!(src.contains("device half* C"));
        // Tiles cache operand bytes (operand precision), even though
        // the accumulator widens (spec §5.7.1).
        assert!(src.contains("threadgroup half tileA"));
    }

    #[test]
    fn matmul_bf16_wraps_in_msl_320_guard() {
        let src = matmul_tiled_kernel_for("k_matmul_bf16_test", Prim::Bf16);
        assert!(src.contains("#if __METAL_VERSION__ >= 320"));
        assert!(src.contains("device const bfloat* A"));
        assert!(src.trim_end().ends_with("#endif"));
    }

    /// F1 + F5 (spec/04-type-system.md §5.7.1): bf16 matmul accumulates
    /// in f32 and the partial product casts both operands to f32 at the
    /// multiply (operand-precision multiplication truncates before the
    /// accumulator widens).
    #[test]
    fn bf16_matmul_uses_f32_accumulator_per_spec_5_7_1() {
        let src = matmul_tiled_kernel_for("k_matmul_bf16_acc", Prim::Bf16);
        // Accumulator is f32 (spec §5.7.1).
        assert!(
            src.contains("float acc"),
            "bf16 matmul must declare `float acc` per spec §5.7.1: {src}"
        );
        assert!(
            !src.contains("bfloat acc"),
            "bf16 matmul must NOT declare `bfloat acc` (silently truncates): {src}"
        );
        // Tiles still hold operand precision (memory-bandwidth match).
        assert!(
            src.contains("threadgroup bfloat tileA") && src.contains("threadgroup bfloat tileB"),
            "bf16 matmul tile storage stays at operand precision: {src}"
        );
        // Partial product casts operands to f32 at the multiply.
        assert!(
            src.contains("(float)tileA[lid.y][i] * (float)tileB[i][lid.x]"),
            "bf16 matmul partial product must cast to f32 at multiply: {src}"
        );
        // Writeback downcasts back to bf16.
        assert!(
            src.contains("(bfloat)acc"),
            "bf16 matmul must downcast f32 acc to bfloat at write-out: {src}"
        );
    }

    /// f32 matmul keeps the f32 accumulator and emits no operand cast
    /// (cast would be a no-op).
    #[test]
    fn f32_matmul_no_operand_cast_at_multiply() {
        let src = matmul_tiled_kernel_for("k_matmul_f32_acc", Prim::F32);
        assert!(src.contains("float acc"));
        assert!(
            src.contains("tileA[lid.y][i] * tileB[i][lid.x]"),
            "f32 matmul should not insert a redundant cast: {src}"
        );
        assert!(
            !src.contains("(float)tileA"),
            "f32 matmul must not double-cast operands: {src}"
        );
    }

    /// f16 matmul: spec §5.7.1 promotes f16 operand to f32 accumulator
    /// just like bf16.
    #[test]
    fn f16_matmul_promotes_to_f32_accumulator() {
        let src = matmul_tiled_kernel_for("k_matmul_f16_acc", Prim::F16);
        assert!(src.contains("float acc"));
        assert!(src.contains("(float)tileA[lid.y][i] * (float)tileB[i][lid.x]"));
        assert!(src.contains("(half)acc"));
    }

    /// `matmul_tiled_kernel_with_acc` lets the caller pin the
    /// accumulator explicitly; verify the cast routing follows the
    /// same shape regardless of the helper used.
    #[test]
    fn matmul_with_explicit_accumulator() {
        let src = matmul_tiled_kernel_with_acc("k_mm_explicit", Prim::Bf16, Prim::F32);
        assert!(src.contains("float acc"));
        assert!(src.contains("(float)tileA"));
        assert!(src.contains("(bfloat)acc"));
    }

    #[test]
    fn matmul_f32_no_msl_320_guard() {
        let src = matmul_tiled_kernel_for("k_matmul_f32_test", Prim::F32);
        assert!(!src.contains("#if __METAL_VERSION__"));
    }

    #[test]
    fn reduce_sum_widens_narrow_floats_to_f32() {
        let src = reduce_full_kernel_for(
            "k_reduce_sum_f16_test",
            ReduceKind::Sum,
            Prim::F16,
            Prim::F32,
        );
        assert!(src.contains("device const half* input"));
        assert!(src.contains("device float* output"));
        assert!(src.contains("threadgroup float shared"));
        assert!(src.contains("(float)input[lid + i]"));
    }

    #[test]
    fn reduce_sum_widens_narrow_ints_to_i32() {
        let src = reduce_full_kernel_for(
            "k_reduce_sum_i8_test",
            ReduceKind::Sum,
            Prim::Int8,
            Prim::Int32,
        );
        assert!(src.contains("device const char* input"));
        assert!(src.contains("device int* output"));
        assert!(src.contains("(int)input[lid + i]"));
    }

    #[test]
    fn reduce_max_keeps_operand_precision() {
        let src = reduce_full_kernel_for(
            "k_reduce_max_f16_test",
            ReduceKind::Max,
            Prim::F16,
            Prim::F16,
        );
        assert!(src.contains("device const half* input"));
        assert!(src.contains("device half* output"));
        assert!(src.contains("max(acc, input[lid + i])"));
        assert!(!src.contains("(half)input[lid + i]"));
    }

    #[test]
    fn elementwise_kernel_for_bf16_wraps_guard() {
        let params = vec![
            input_param(0, "bfloat", "a"),
            output_param(1, "bfloat", "out"),
        ];
        let src = elementwise_kernel_for(
            "k_unary_bf16_test",
            &params,
            "    out[tid] = -a[tid];",
            &[Prim::Bf16],
        );
        assert!(src.contains("#if __METAL_VERSION__ >= 320"));
    }

    #[test]
    fn legacy_matmul_kernel_still_emits_f32() {
        let src = matmul_tiled_kernel("k_matmul_legacy");
        assert!(src.contains("device const float* A"));
    }
}
