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
    const float *a, {a_strides}, int a_ndim,
    const float *b, {b_strides}, int b_ndim,
    float *out, {out_shape}, int out_ndim, int size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = chelis_indices_to_flat(indices, a_s, a_ndim);
  int idx_b = chelis_indices_to_flat(indices, b_s, b_ndim);
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
    const float *a, {a_strides}, int a_ndim,
    const float *b, {b_strides}, int b_ndim,
    float *out, {out_shape}, int out_ndim, int size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = chelis_indices_to_flat(indices, a_s, a_ndim);
  int idx_b = chelis_indices_to_flat(indices, b_s, b_ndim);
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
    const float *a, {a_strides}, int a_ndim,
    const float *b, {b_strides}, int b_ndim,
    float *out, {out_shape}, int out_ndim, int size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = chelis_indices_to_flat(indices, a_s, a_ndim);
  int idx_b = chelis_indices_to_flat(indices, b_s, b_ndim);
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
    const float *a, {a_strides}, int a_ndim,
    float *out, {out_shape}, int out_ndim, int size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = chelis_indices_to_flat(indices, a_s, a_ndim);
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
    const float *a, {a_strides}, int a_ndim,
    float *out, {out_shape}, int out_ndim, int size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = chelis_indices_to_flat(indices, a_s, a_ndim);
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
pub fn reduce_sum(kernel_name: &str, axis: usize, axis_size: usize) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  float acc = 0.0f;
  for (int k = 0; k < {axis_size}; k++) {{
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
    int src_idx = chelis_indices_to_flat(full_indices, a_s, a_ndim);
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
pub fn reduce_max(kernel_name: &str, axis: usize, axis_size: usize) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const float *a, {a_strides}, int a_ndim,
    float *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  float acc = -3.402823466e+38F;
  for (int k = 0; k < {axis_size}; k++) {{
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
    int src_idx = chelis_indices_to_flat(full_indices, a_s, a_ndim);
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

/// Generate kernel source for filling a tensor with a constant value.
pub fn fill(kernel_name: &str) -> String {
    format!(
        "\
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
    const float *a, {a_strides}, int a_ndim,
    float *out, {out_shape}, int out_ndim, int size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = chelis_indices_to_flat(indices, a_s, a_ndim);
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
        let src = reduce_sum("kernel_sum_ax0", 0, 32);
        assert!(src.contains("float acc = 0.0f;"));
        assert!(src.contains("for (int k = 0; k < 32; k++)"));
        assert!(src.contains("if (d == 0)"));
    }

    #[test]
    fn reduce_max_kernel_has_f32_min_sentinel() {
        let src = reduce_max("kernel_max_ax1", 1, 10);
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
        let sum = reduce_sum("k", 0, 1);
        for src in [&add, &neg, &sum] {
            assert!(src.contains("__device__ void chelis_flat_to_indices"));
            assert!(src.contains("__device__ int chelis_indices_to_flat"));
        }
    }
}
