//! HIP kernel source string templates for RISC operations.
//!
//! Each function returns a complete kernel source string ready for hiprtc
//! compilation. Device-side indexing helpers are prepended to every kernel.
//!
//! ## Parameter Convention (Phase 1a)
//!
//! Shapes and strides are passed as individual integer kernel parameters, NOT
//! device pointers. Each tensor passes `CHELIS_MAX_DIM` (8) ints for strides
//! and shapes. Inside the kernel, these are assembled into local arrays for
//! the device-side indexing helpers. This avoids per-launch hipMalloc/hipMemcpy
//! for small metadata arrays. Switch to device pointer arrays in Phase 1d.

/// Device-side helper functions included at the top of every kernel source.
pub const DEVICE_HELPERS: &str = "\
#if CHELIS_DEBUG_BOUNDS
__device__ int chelis_gpu_failure = 0;
__device__ void chelis_record_failure(int code) {
    if (code != 0) {
        atomicCAS(&chelis_gpu_failure, 0, code);
    }
}
__device__ int chelis_bounds_guard(int idx, int size, int code) {
    if (idx < 0 || idx >= size) {
        chelis_record_failure(code);
        return 0;
    }
    return idx;
}
#define CHELIS_GUARD_INDEX(idx, size, code) chelis_bounds_guard((idx), (size), (code))
#else
#define CHELIS_GUARD_INDEX(idx, size, code) (idx)
#endif
__device__ void chelis_flat_to_indices(int flat, const int *shape, int ndim, int *out) {
    for (int d = ndim - 1; d >= 0; d--) {
        out[d] = flat % shape[d];
        flat /= shape[d];
    }
}
__device__ int chelis_indices_to_flat(const int *indices, const int *strides, int ndim) {
    int flat = 0;
    for (int d = 0; d < ndim; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}
";

/// Maximum tensor dimensions (must match CHELIS_MAX_DIM in runtime).
pub const MAX_DIM: usize = 8;

/// Stride parameter names for one tensor: `{prefix}_s0, {prefix}_s1, ..., {prefix}_s7`.
fn stride_params(prefix: &str) -> String {
    (0..MAX_DIM)
        .map(|i| format!("int {prefix}_s{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Shape parameter names: `{prefix}_sh0, {prefix}_sh1, ..., {prefix}_sh7`.
fn shape_params(prefix: &str) -> String {
    (0..MAX_DIM)
        .map(|i| format!("int {prefix}_sh{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Emit code to build a local `int[]` array from individual params.
fn build_array(var_name: &str, prefix: &str, suffix: &str) -> String {
    let elems: Vec<String> = (0..MAX_DIM)
        .map(|i| format!("{prefix}_{suffix}{i}"))
        .collect();
    format!("  int {var_name}[] = {{ {} }};", elems.join(", "))
}

/// Generate kernel source for a binary elementwise op (add, mul).
pub fn binary_elementwise(kernel_name: &str, op: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    const float *b, {b_strides}, int b_ndim, int b_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  int idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = a[idx_a] {op} b[idx_b];
}}
",
        a_strides = stride_params("a"),
        b_strides = stride_params("b"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_b_s = build_array("b_s", "b", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for a binary function op (fmaxf for max_elem).
pub fn binary_func(kernel_name: &str, func: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    const float *b, {b_strides}, int b_ndim, int b_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  int idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = {func}(a[idx_a], b[idx_b]);
}}
",
        a_strides = stride_params("a"),
        b_strides = stride_params("b"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_b_s = build_array("b_s", "b", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for cmplt (returns 1.0f/0.0f).
pub fn cmplt(kernel_name: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    const float *b, {b_strides}, int b_ndim, int b_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  int idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = (a[idx_a] < b[idx_b]) ? 1.0f : 0.0f;
}}
",
        a_strides = stride_params("a"),
        b_strides = stride_params("b"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_b_s = build_array("b_s", "b", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for a unary prefix op (neg: `-`).
pub fn unary_prefix(kernel_name: &str, op: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {op}a[idx];
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for a unary function op (expf, logf, sinf, sqrtf).
pub fn unary_func(kernel_name: &str, func: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {func}(a[idx]);
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for sum reduction (naive: one thread per output element).
pub fn reduce_sum(kernel_name: &str, axis: usize) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  float acc = 0.0f;
  for (int k = 0; k < axis_size; k++) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    int src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc += a[src_idx];
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for max reduction (naive: one thread per output element).
pub fn reduce_max(kernel_name: &str, axis: usize) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  float acc = -3.402823466e+38F;
  for (int k = 0; k < axis_size; k++) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    int src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc = fmaxf(acc, a[src_idx]);
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

fn reduce_init(kind: ReduceKind) -> &'static str {
    match kind {
        ReduceKind::Sum => "0.0f",
        ReduceKind::Max => "-3.402823466e+38F",
    }
}

fn reduce_accumulate(kind: ReduceKind, lhs: &str, rhs: &str) -> String {
    match kind {
        ReduceKind::Sum => format!("{lhs} += {rhs};"),
        ReduceKind::Max => format!("{lhs} = fmaxf({lhs}, {rhs});"),
    }
}

fn reduce_combine_expr(kind: ReduceKind, lhs: &str, rhs: &str) -> String {
    match kind {
        ReduceKind::Sum => format!("{lhs} + {rhs}"),
        ReduceKind::Max => format!("fmaxf({lhs}, {rhs})"),
    }
}

pub fn segmented_reduce_small(
    kernel_name: &str,
    axis: usize,
    axis_size: usize,
    threads_per_segment: usize,
    segments_per_block: usize,
    kind: ReduceKind,
) -> String {
    let init = reduce_init(kind);
    let accumulate = reduce_accumulate(kind, "acc", "a[src_idx]");
    let combine = reduce_combine_expr(kind, "shared[base + lane]", "shared[base + lane + offset]");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  __shared__ float shared[256];
  int lane = threadIdx.x % {threads_per_segment};
  int seg_in_block = threadIdx.x / {threads_per_segment};
  int seg = blockIdx.x * {segments_per_block} + seg_in_block;
  if (seg >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(seg, out_sh, out_ndim, out_indices);
  float acc = {init};
  for (int k = lane; k < {axis_size}; k += {threads_per_segment}) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    int src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    {accumulate}
  }}
  int base = seg_in_block * {threads_per_segment};
  shared[base + lane] = acc;
  __syncthreads();
  for (int offset = {threads_per_segment} / 2; offset > 0; offset /= 2) {{
    if (lane < offset) {{
      shared[base + lane] = {combine};
    }}
    __syncthreads();
  }}
  if (lane == 0) {{
    out[seg] = shared[base];
  }}
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

pub fn segmented_reduce_large(
    kernel_name: &str,
    axis: usize,
    axis_size: usize,
    block_size: usize,
    kind: ReduceKind,
) -> String {
    let init = reduce_init(kind);
    let accumulate = reduce_accumulate(kind, "acc", "a[src_idx]");
    let combine = reduce_combine_expr(kind, "shared[lane]", "shared[lane + offset]");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  __shared__ float shared[{block_size}];
  int seg = blockIdx.x;
  int lane = threadIdx.x;
  if (seg >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(seg, out_sh, out_ndim, out_indices);
  float acc = {init};
  for (int k = lane; k < {axis_size}; k += blockDim.x) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    int src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    {accumulate}
  }}
  shared[lane] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
    if (lane < offset) {{
      shared[lane] = {combine};
    }}
    __syncthreads();
  }}
  if (lane == 0) out[seg] = shared[0];
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

pub fn staged_reduce_stage1(kernel_name: &str, block_size: usize, kind: ReduceKind) -> String {
    let init = reduce_init(kind);
    let accumulate = reduce_accumulate(kind, "acc", "a[idx]");
    let combine = reduce_combine_expr(kind, "shared[lane]", "shared[lane + offset]");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, float *partials, int total_size) {{
  __shared__ float shared[{block_size}];
  int lane = threadIdx.x;
  int start = blockIdx.x * blockDim.x + lane;
  float acc = {init};
  for (int idx = start; idx < total_size; idx += gridDim.x * blockDim.x) {{
    {accumulate}
  }}
  shared[lane] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
    if (lane < offset) {{
      shared[lane] = {combine};
    }}
    __syncthreads();
  }}
  if (lane == 0) partials[blockIdx.x] = shared[0];
}}
"
    )
}

pub fn staged_reduce_stage_n(kernel_name: &str, block_size: usize, kind: ReduceKind) -> String {
    let init = reduce_init(kind);
    let accumulate = reduce_accumulate(kind, "acc", "in_data[k]");
    let combine = reduce_combine_expr(kind, "shared[lane]", "shared[lane + offset]");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(const float *in_data, float *out_data, int total_size) {{
  __shared__ float shared[{block_size}];
  int lane = threadIdx.x;
  int start = blockIdx.x * blockDim.x + lane;
  float acc = {init};
  for (int k = start; k < total_size; k += gridDim.x * blockDim.x) {{
    {accumulate}
  }}
  shared[lane] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
    if (lane < offset) {{
      shared[lane] = {combine};
    }}
    __syncthreads();
  }}
  if (lane == 0) out_data[blockIdx.x] = shared[0];
}}
"
    )
}

#[allow(clippy::too_many_arguments)]
pub fn segmented_reduce_small_fused(
    kernel_name: &str,
    axis: usize,
    axis_size: usize,
    threads_per_segment: usize,
    segments_per_block: usize,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
    kind: ReduceKind,
) -> String {
    use chelis_ir::dag::{FusedInput, FusedStepOp};

    let mut params = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        params.push(format!("const float *{pfx}"));
        params.push(stride_params(&pfx));
        params.push(format!("int {pfx}_ndim"));
        params.push(format!("int {pfx}_size"));
    }
    params.push("float *out".into());
    params.push(shape_params("out"));
    params.push("int out_ndim".into());
    params.push("int out_size".into());

    let mut body_arrays = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        body_arrays.push(build_array(&format!("{pfx}_s"), &pfx, "s"));
    }
    body_arrays.push(build_array("out_sh", "out", "sh"));

    let resolve = |input: &FusedInput| -> String {
        match input {
            FusedInput::External(i) => format!("ext{i}[idx_ext{i}]"),
            FusedInput::PreviousStep(j) => format!("v{j}"),
        }
    };

    let mut step_lines = Vec::new();
    for (si, step) in steps.iter().enumerate() {
        let expr = match step.op {
            FusedStepOp::Add => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("fmaxf({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("({a} < {b}) ? 1.0f : 0.0f")
            }
            FusedStepOp::Neg => {
                let a = resolve(&step.input_indices[0]);
                format!("-{a}")
            }
            FusedStepOp::Exp => {
                let a = resolve(&step.input_indices[0]);
                format!("expf({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&step.input_indices[0]);
                format!("logf({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&step.input_indices[0]);
                format!("sinf({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&step.input_indices[0]);
                format!("sqrtf({a})")
            }
        };
        step_lines.push(format!("    float v{si} = {expr};"));
    }

    let last_step = steps.len() - 1;
    let init = reduce_init(kind);
    let accumulate = reduce_accumulate(kind, "acc", &format!("v{last_step}"));
    let combine = reduce_combine_expr(kind, "shared[base + lane]", "shared[base + lane + offset]");

    let mut index_lines = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        index_lines.push(format!(
            "    int idx_{pfx} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, {pfx}_s, {pfx}_ndim), {pfx}_size, 1);"
        ));
    }

    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  __shared__ float shared[256];
  int lane = threadIdx.x % {threads_per_segment};
  int seg_in_block = threadIdx.x / {threads_per_segment};
  int seg = blockIdx.x * {segments_per_block} + seg_in_block;
  if (seg >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(seg, out_sh, out_ndim, out_indices);
  float acc = {init};
  for (int k = lane; k < {axis_size}; k += {threads_per_segment}) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < out_ndim + 1; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
{index_lines}
{step_lines}
    {accumulate}
  }}
  int base = seg_in_block * {threads_per_segment};
  shared[base + lane] = acc;
  __syncthreads();
  for (int offset = {threads_per_segment} / 2; offset > 0; offset /= 2) {{
    if (lane < offset) {{
      shared[base + lane] = {combine};
    }}
    __syncthreads();
  }}
  if (lane == 0) {{
    out[seg] = shared[base];
  }}
}}
",
        params = params.join(",\n    "),
        arrays = body_arrays.join("\n"),
        index_lines = index_lines.join("\n"),
        step_lines = step_lines.join("\n"),
    )
}

pub fn segmented_reduce_large_fused(
    kernel_name: &str,
    axis: usize,
    axis_size: usize,
    block_size: usize,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
    kind: ReduceKind,
) -> String {
    use chelis_ir::dag::{FusedInput, FusedStepOp};

    let mut params = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        params.push(format!("const float *{pfx}"));
        params.push(stride_params(&pfx));
        params.push(format!("int {pfx}_ndim"));
        params.push(format!("int {pfx}_size"));
    }
    params.push("float *out".into());
    params.push(shape_params("out"));
    params.push("int out_ndim".into());
    params.push("int out_size".into());

    let mut body_arrays = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        body_arrays.push(build_array(&format!("{pfx}_s"), &pfx, "s"));
    }
    body_arrays.push(build_array("out_sh", "out", "sh"));

    let resolve = |input: &FusedInput| -> String {
        match input {
            FusedInput::External(i) => format!("ext{i}[idx_ext{i}]"),
            FusedInput::PreviousStep(j) => format!("v{j}"),
        }
    };

    let mut step_lines = Vec::new();
    for (si, step) in steps.iter().enumerate() {
        let expr = match step.op {
            FusedStepOp::Add => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("fmaxf({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("({a} < {b}) ? 1.0f : 0.0f")
            }
            FusedStepOp::Neg => {
                let a = resolve(&step.input_indices[0]);
                format!("-{a}")
            }
            FusedStepOp::Exp => {
                let a = resolve(&step.input_indices[0]);
                format!("expf({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&step.input_indices[0]);
                format!("logf({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&step.input_indices[0]);
                format!("sinf({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&step.input_indices[0]);
                format!("sqrtf({a})")
            }
        };
        step_lines.push(format!("    float v{si} = {expr};"));
    }

    let last_step = steps.len() - 1;
    let init = reduce_init(kind);
    let accumulate = reduce_accumulate(kind, "acc", &format!("v{last_step}"));
    let combine = reduce_combine_expr(kind, "shared[lane]", "shared[lane + offset]");

    let mut index_lines = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        index_lines.push(format!(
            "    int idx_{pfx} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, {pfx}_s, {pfx}_ndim), {pfx}_size, 1);"
        ));
    }

    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  __shared__ float shared[{block_size}];
  int seg = blockIdx.x;
  int lane = threadIdx.x;
  if (seg >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(seg, out_sh, out_ndim, out_indices);
  float acc = {init};
  for (int k = lane; k < {axis_size}; k += blockDim.x) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < out_ndim + 1; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
{index_lines}
{step_lines}
    {accumulate}
  }}
  shared[lane] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {{
    if (lane < offset) {{
      shared[lane] = {combine};
    }}
    __syncthreads();
  }}
  if (lane == 0) out[seg] = shared[0];
}}
",
        params = params.join(",\n    "),
        arrays = body_arrays.join("\n"),
        index_lines = index_lines.join("\n"),
        step_lines = step_lines.join("\n"),
    )
}

/// Generate a fused reduction kernel: elementwise chain inlined into the reduction inner loop.
///
/// Instead of reading `a[src_idx]` in the reduction loop, this kernel applies
/// the elementwise chain's steps to compute the value from external inputs before
/// accumulating.
pub fn reduce_fused(
    kernel_name: &str,
    axis: usize,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
    reduce_kind: ReduceKind,
) -> String {
    use chelis_ir::dag::{FusedInput, FusedStepOp};

    // Build parameter list: external inputs (with strides), output (with shape)
    let mut params = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        params.push(format!("const float *{pfx}"));
        params.push(stride_params(&pfx));
        params.push(format!("int {pfx}_ndim"));
        params.push(format!("int {pfx}_size"));
    }
    params.push("float *out".into());
    params.push(shape_params("out"));
    params.push("int out_ndim".into());
    params.push("int out_size".into());
    params.push("int axis_size".into());

    // Build local array constructions for external input strides
    let mut body_arrays = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        body_arrays.push(build_array(&format!("{pfx}_s"), &pfx, "s"));
    }
    body_arrays.push(build_array("out_sh", "out", "sh"));

    // Resolve a FusedInput to a C expression
    let resolve = |input: &FusedInput| -> String {
        match input {
            FusedInput::External(i) => format!("ext{i}[idx_ext{i}]"),
            FusedInput::PreviousStep(j) => format!("v{j}"),
        }
    };

    // Build step computation lines
    let mut step_lines = Vec::new();
    for (si, step) in steps.iter().enumerate() {
        let expr = match step.op {
            FusedStepOp::Add => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("fmaxf({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("({a} < {b}) ? 1.0f : 0.0f")
            }
            FusedStepOp::Neg => {
                let a = resolve(&step.input_indices[0]);
                format!("-{a}")
            }
            FusedStepOp::Exp => {
                let a = resolve(&step.input_indices[0]);
                format!("expf({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&step.input_indices[0]);
                format!("logf({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&step.input_indices[0]);
                format!("sinf({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&step.input_indices[0]);
                format!("sqrtf({a})")
            }
        };
        step_lines.push(format!("      float v{si} = {expr};"));
    }

    let last_step = steps.len() - 1;
    let (init, accumulate) = match reduce_kind {
        ReduceKind::Sum => ("0.0f".to_string(), format!("acc += v{last_step};")),
        ReduceKind::Max => (
            "-3.402823466e+38F".to_string(),
            format!("acc = fmaxf(acc, v{last_step});"),
        ),
    };

    // Build index computation for each external input (inside the inner loop)
    let mut index_lines = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        index_lines.push(format!(
            "      int idx_{pfx} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, {pfx}_s, {pfx}_ndim), {pfx}_size, 1);"
        ));
    }

    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  float acc = {init};
  for (int k = 0; k < axis_size; k++) {{
    int full_indices[{MAX_DIM}];
    int out_d = 0;
    for (int d = 0; d < out_ndim + 1; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
{index_lines}
{step_lines}
    {accumulate}
  }}
  out[outer] = acc;
}}
",
        params = params.join(",\n    "),
        arrays = body_arrays.join("\n"),
        index_lines = index_lines.join("\n"),
        step_lines = step_lines.join("\n"),
    )
}

/// Kind of reduction for fused reduce kernels.
#[derive(Debug, Clone, Copy)]
pub enum ReduceKind {
    Sum,
    Max,
}

/// Generate kernel source for a fused elementwise chain.
///
/// Each step computes into a register `float v{step_idx}`, resolving inputs
/// from either external input arrays or previous step outputs.
pub fn fused_elementwise(
    kernel_name: &str,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
) -> String {
    use chelis_ir::dag::{FusedInput, FusedStepOp};

    // Build parameter list
    let mut params = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        params.push(format!("const float *{pfx}"));
        params.push(stride_params(&pfx));
        params.push(format!("int {pfx}_ndim"));
        params.push(format!("int {pfx}_size"));
    }
    params.push("float *out".into());
    params.push(shape_params("out"));
    params.push("int out_ndim".into());
    params.push("int out_size".into());

    // Build local array constructions
    let mut body_arrays = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        body_arrays.push(build_array(&format!("{pfx}_s"), &pfx, "s"));
    }
    body_arrays.push(build_array("out_sh", "out", "sh"));

    // Build index computation for each external input
    let mut index_lines = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        index_lines.push(format!(
            "  int idx_{pfx} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, {pfx}_s, {pfx}_ndim), {pfx}_size, 1);"
        ));
    }

    // Resolve a FusedInput to a C expression
    let resolve = |input: &FusedInput| -> String {
        match input {
            FusedInput::External(i) => format!("ext{i}[idx_ext{i}]"),
            FusedInput::PreviousStep(j) => format!("v{j}"),
        }
    };

    // Build step computation lines
    let mut step_lines = Vec::new();
    for (si, step) in steps.iter().enumerate() {
        let expr = match step.op {
            FusedStepOp::Add => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::MaxElem => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("fmaxf({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve(&step.input_indices[0]);
                let b = resolve(&step.input_indices[1]);
                format!("({a} < {b}) ? 1.0f : 0.0f")
            }
            FusedStepOp::Neg => {
                let a = resolve(&step.input_indices[0]);
                format!("-{a}")
            }
            FusedStepOp::Exp => {
                let a = resolve(&step.input_indices[0]);
                format!("expf({a})")
            }
            FusedStepOp::Log => {
                let a = resolve(&step.input_indices[0]);
                format!("logf({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve(&step.input_indices[0]);
                format!("sinf({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve(&step.input_indices[0]);
                format!("sqrtf({a})")
            }
        };
        step_lines.push(format!("  float v{si} = {expr};"));
    }

    let last_step = steps.len() - 1;

    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
{index_lines}
{step_lines}
  out[i] = v{last_step};
}}
",
        params = params.join(",\n    "),
        arrays = body_arrays.join("\n"),
        index_lines = index_lines.join("\n"),
        step_lines = step_lines.join("\n"),
    )
}

/// Generate kernel source for filling a tensor with a constant value.
pub fn fill(kernel_name: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(float *data, float value, int size) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  data[i] = value;
}}
"
    )
}

/// Generate kernel source for cast (currently f32->f32, identity copy).
pub fn cast(kernel_name: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim, int a_size,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = a[idx];
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_add_kernel_has_correct_structure() {
        let src = binary_elementwise("kernel_add", "+");
        assert!(src.contains("extern \"C\" __global__ void kernel_add("));
        assert!(src.contains("out[i] = a[idx_a] + b[idx_b];"));
        assert!(src.contains("chelis_flat_to_indices"));
        assert!(src.contains("chelis_indices_to_flat"));
    }

    #[test]
    fn binary_add_uses_int_params_not_pointers() {
        let src = binary_elementwise("kernel_add", "+");
        // Must have individual int params, not const int* pointers
        assert!(
            src.contains("int a_s0"),
            "stride params must be individual ints"
        );
        assert!(
            src.contains("int a_s7"),
            "must have all MAX_DIM stride params"
        );
        assert!(
            src.contains("int out_sh0"),
            "shape params must be individual ints"
        );
        // Must build local arrays from params
        assert!(src.contains("int a_s[] ="), "must build local stride array");
        assert!(
            src.contains("int out_sh[] ="),
            "must build local shape array"
        );
        // Must NOT have device pointer params for strides/shapes
        assert!(
            !src.contains("const int *a_strides"),
            "must NOT use device pointer for strides"
        );
    }

    #[test]
    fn unary_exp_kernel_correct() {
        let src = unary_func("kernel_exp", "expf");
        assert!(src.contains("extern \"C\" __global__ void kernel_exp("));
        assert!(src.contains("out[i] = expf(a[idx]);"));
    }

    #[test]
    fn cmplt_kernel_returns_float() {
        let src = cmplt("kernel_cmplt");
        assert!(src.contains("1.0f : 0.0f"));
    }

    #[test]
    fn reduce_sum_kernel_has_axis_loop() {
        let src = reduce_sum("kernel_sum_ax0", 0);
        assert!(src.contains("float acc = 0.0f;"));
        assert!(src.contains("for (int k = 0; k < axis_size; k++)"));
        assert!(src.contains("if (d == 0)"));
    }

    #[test]
    fn reduce_max_kernel_has_f32_min_sentinel() {
        let src = reduce_max("kernel_max_ax1", 1);
        assert!(src.contains("-3.402823466e+38F"));
        assert!(src.contains("fmaxf(acc,"));
    }

    #[test]
    fn fill_kernel_simple() {
        let src = fill("kernel_fill");
        assert!(src.contains("data[i] = value;"));
        // Fill has no stride/shape params (operates on raw buffer)
        assert!(!src.contains("a_s0"));
    }

    #[test]
    fn device_helpers_present_in_all_compute_kernels() {
        let add = binary_elementwise("k", "+");
        let neg = unary_prefix("k", "-");
        let sum = reduce_sum("k", 0);
        let fill_src = fill("k");
        for src in [&add, &neg, &sum, &fill_src] {
            assert!(src.contains("__device__ void chelis_flat_to_indices"));
            assert!(src.contains("__device__ int chelis_indices_to_flat"));
            assert!(src.contains("__device__ int chelis_gpu_failure = 0;"));
            assert!(src.contains("CHELIS_GUARD_INDEX"));
        }
    }
}
