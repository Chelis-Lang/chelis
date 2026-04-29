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
//! the M2 prelude on Apple Silicon (M3, GPU family 7+) — no uchar
//! workaround needed.

/// Emit a complete elementwise kernel source string.
///
/// `kernel_name`: the MSL function name, also used as the cache key
/// `params`: list of `(qualifier, type, name)` triples for the parameter
///           block, in order. The emitter is expected to follow the buffer
///           conventions documented above.
/// `body`: the per-element body, with `tid` already bounds-checked. The
///         body may reference parameter names directly.
///
/// Returns a self-contained MSL source string suitable for embedding in a
/// `static NSString *const ... = @R"MSL(...)MSL";` literal.
pub fn elementwise_kernel(kernel_name: &str, params: &[String], body: &str) -> String {
    let header = "#include <metal_stdlib>\nusing namespace metal;\n\n";
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

/// Map a Chelis precision to the MSL type spelling used in kernel params
/// and the body. M2 supports f32 and bool; everything else is rejected
/// upstream at the CLI by `reject_unsupported_metal_precisions`.
pub fn msl_type(prec: chelis_types::types::Prim) -> &'static str {
    match prec {
        chelis_types::types::Prim::F32 => "float",
        chelis_types::types::Prim::Bool => "bool",
        _ => panic!(
            "Metal backend M2 supports only f32 and bool element types; \
             reject_unsupported_metal_precisions should have caught the rest"
        ),
    }
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
    /// MSL identity element string for the reduction.
    pub fn identity(self) -> &'static str {
        match self {
            Self::Sum => "0.0f",
            Self::Max => "-INFINITY",
            Self::Min => "INFINITY",
        }
    }

    /// Short label used in kernel names (`reduce_sum`, `reduce_max`, …).
    pub fn label(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
        }
    }

    /// Combine expression: given two MSL float expressions `lhs` and `rhs`,
    /// produce the merged scalar.
    pub fn combine(self, lhs: &str, rhs: &str) -> String {
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

/// Emit the tiled MSL matmul kernel.
///
/// Computes `C[M, N] = A[M, K] @ B[K, N]` for row-major contiguous f32
/// matrices via a 16x16 tile. Threadgroup memory caches one tile of A
/// and one tile of B per outer iteration; each thread accumulates one
/// output element. The kernel is dispatched with an `MxN` grid in
/// 16x16 threadgroups; the host picks tg = (16, 16, 1).
pub fn matmul_tiled_kernel(kernel_name: &str) -> String {
    let tile = MATMUL_TILE;
    // M, N, K are packed into a single uniform struct bound at buffer(3).
    // Binding them as three separate `constant uint&` parameters at
    // distinct buffer indices would require three `setBytes:atIndex:` calls
    // from the host, but `chelis_metal_launch2d` only binds one uniforms
    // blob — so a single struct keeps the kernel and host in lockstep.
    format!(
        "#include <metal_stdlib>
using namespace metal;

struct ChelisMatmulDims {{
    uint M;
    uint N;
    uint K;
}};

kernel void {kernel_name}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant ChelisMatmulDims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {{
    const uint TILE = {tile};
    const uint M = dims.M;
    const uint N = dims.N;
    const uint K = dims.K;
    threadgroup float tileA[{tile}][{tile}];
    threadgroup float tileB[{tile}][{tile}];

    float acc = 0.0f;
    uint num_tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {{
        uint aRow = gid.y;
        uint aCol = t * TILE + lid.x;
        uint bRow = t * TILE + lid.y;
        uint bCol = gid.x;
        tileA[lid.y][lid.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        tileB[lid.y][lid.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {{
            acc += tileA[lid.y][i] * tileB[i][lid.x];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (gid.y < M && gid.x < N) {{
        C[gid.y * N + gid.x] = acc;
    }}
}}
"
    )
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

/// Emit a 1D reduction kernel that reduces a contiguous `n`-element f32
/// buffer to a single scalar via threadgroup memory + tree reduction.
///
/// Dispatched with a fixed power-of-two threadgroup size
/// ([`REDUCE_TG_SIZE`] = 256). The wrap loop strides `n` at `tg_size`
/// per iteration so any `n <= REDUCE_TG_SIZE * REDUCE_TG_SIZE` (with
/// the same single-threadgroup constraint) reduces in one pass. For
/// `n` smaller than `tg_size`, threads with `lid >= n` skip the wrap
/// loop entirely and seed `shared[lid]` with the identity, so the
/// tree-reduction is still well-defined.
///
/// Two-pass reduction for `n > REDUCE_TG_SIZE` is M4.next; the emitter
/// rejects oversized inputs at the host site.
pub fn reduce_full_kernel(kernel_name: &str, kind: ReduceKind) -> String {
    let identity = kind.identity();
    let tg = REDUCE_TG_SIZE;
    // The single-threadgroup reduction uses `lid` as both the per-thread
    // index into the input stride loop AND the threadgroup-local index for
    // tree reduction. Attribute bindings must not duplicate, so bind once
    // as `lid` and use it everywhere.
    let combine_acc_input = kind.combine("acc", "input[lid + i]");
    let combine_pair = kind.combine("shared[lid]", "shared[lid + stride]");
    format!(
        "#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {{
    // Threadgroup size is fixed at compile time so the tree-reduce halving
    // terminates exactly. The host always dispatches a single threadgroup
    // of TG_SIZE threads regardless of n; the wrap loop below handles any
    // n by striding TG_SIZE elements per iteration.
    const uint TG_SIZE = {tg};
    threadgroup float shared[{tg}];

    float acc = {identity};
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
    )
}
