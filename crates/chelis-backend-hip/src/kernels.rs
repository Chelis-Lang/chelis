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
__device__ float chelis_uniform_sample_f32(unsigned long long seed, unsigned long long index, float low, float high) {
    unsigned long long x = seed ^ (index * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    double unit = (double)(x >> 11) / (double)(1ULL << 53);
    return low + (high - low) * (float)unit;
}
";

/// Maximum tensor dimensions (must match CHELIS_MAX_DIM in runtime).
pub const MAX_DIM: usize = 8;

/// Floating-point element kind for kernel emission. WS-A2 admits f64
/// alongside the original f32-only HIP path; integer/bool kernels are not
/// modeled here because the existing integer surface is limited to indices
/// (gather/scatter) which keep their bespoke `index_ty` parameter.
///
/// `c_type()` is the HIP scalar type name; `suffix()` is appended to
/// kernel function names so the precision is encoded into the symbol the
/// host launches; `init_min()` is the most-negative finite value used as
/// the seed for `max` reductions; `init_max()` mirrors it for `min`
/// reductions; `one()` is the multiplicative identity for `prod`
/// reductions; `func(name)` rewrites a single-precision libm name (e.g.
/// `expf`) to the f64 equivalent (`exp`) when the kind is `F64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemKind {
    F32,
    F64,
}

impl ElemKind {
    pub fn c_type(self) -> &'static str {
        match self {
            ElemKind::F32 => "float",
            ElemKind::F64 => "double",
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            ElemKind::F32 => "f32",
            ElemKind::F64 => "f64",
        }
    }

    /// Most-negative finite literal for `max` reductions.
    pub fn init_min(self) -> &'static str {
        match self {
            ElemKind::F32 => "-3.402823466e+38F",
            ElemKind::F64 => "-1.7976931348623157e+308",
        }
    }

    /// Most-positive finite literal for `min` reductions.
    pub fn init_max(self) -> &'static str {
        match self {
            ElemKind::F32 => "3.402823466e+38F",
            ElemKind::F64 => "1.7976931348623157e+308",
        }
    }

    /// Additive identity literal for `sum` reductions.
    pub fn zero(self) -> &'static str {
        match self {
            ElemKind::F32 => "0.0f",
            ElemKind::F64 => "0.0",
        }
    }

    /// Multiplicative identity literal for `prod` reductions.
    pub fn one(self) -> &'static str {
        match self {
            ElemKind::F32 => "1.0f",
            ElemKind::F64 => "1.0",
        }
    }

    /// Boolean-as-element constants for `cmplt` outputs in this precision.
    pub fn one_lit_bool(self) -> &'static str {
        match self {
            ElemKind::F32 => "1.0f",
            ElemKind::F64 => "1.0",
        }
    }

    pub fn zero_lit_bool(self) -> &'static str {
        match self {
            ElemKind::F32 => "0.0f",
            ElemKind::F64 => "0.0",
        }
    }

    /// Map an f32-suffixed libm function name (e.g. `expf`) to the
    /// matching name for this kind. f64 drops the trailing `f`; everything
    /// else is passed through. Used by `unary_func` and the fused
    /// elementwise generator.
    pub fn func(self, libm_f32_name: &str) -> String {
        match self {
            ElemKind::F32 => libm_f32_name.to_string(),
            ElemKind::F64 => match libm_f32_name {
                "expf" => "exp".into(),
                "logf" => "log".into(),
                "sinf" => "sin".into(),
                "sqrtf" => "sqrt".into(),
                "cosf" => "cos".into(),
                "tanf" => "tan".into(),
                "atanf" => "atan".into(),
                "fabsf" => "fabs".into(),
                "floorf" => "floor".into(),
                "ceilf" => "ceil".into(),
                "fmaxf" => "fmax".into(),
                "fminf" => "fmin".into(),
                other => other.to_string(),
            },
        }
    }
}

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

/// WS-A2 + WS-A4: dtype-parameterized binary elementwise op (add, mul).
/// `elem_c_ty` is the C++ type spelling (e.g. `float`, `double`,
/// `int8_t`, `int16_t`) used for both operand pointers and the result
/// pointer. Same-precision arithmetic per spec/04-type-system.md §5.4
/// (no implicit promotion); both inputs and the output share
/// `elem_c_ty`. The accompanying kernel name should already encode the
/// dtype suffix (e.g. `kernel_add_f64`, `kernel_add_i8`).
pub fn binary_elementwise_typed(kernel_name: &str, op: &str, elem_c_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, int a_ndim, int a_size,
    const {elem_c_ty} *b, {b_strides}, int b_ndim, int b_size,
    {elem_c_ty} *out, {out_shape}, int out_ndim, int out_size) {{
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

/// WS-A2 thin wrapper for the legacy `ElemKind`-based call sites and
/// for callers that already have an `ElemKind` in hand. Forwards to the
/// dtype-parameterized [`binary_elementwise_typed`] using the
/// `ElemKind`'s C-type spelling (`float` or `double`).
pub fn binary_elementwise(kernel_name: &str, op: &str, kind: ElemKind) -> String {
    binary_elementwise_typed(kernel_name, op, kind.c_type())
}

/// Generate kernel source for a binary function op (fmaxf for max_elem).
/// `func` is the f32-suffixed libm name (e.g. `fmaxf`); for f64 the
/// f-suffix is dropped per [`ElemKind::func`].
pub fn binary_func(kernel_name: &str, func: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let resolved = kind.func(func);
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    const {ty} *b, {b_strides}, int b_ndim, int b_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  int idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = {resolved}(a[idx_a], b[idx_b]);
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

/// Generate kernel source for cmplt. Returns the in-precision boolean
/// constants (`1.0f`/`0.0f` for f32; `1.0`/`0.0` for f64).
pub fn cmplt(kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let one = kind.one_lit_bool();
    let zero = kind.zero_lit_bool();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    const {ty} *b, {b_strides}, int b_ndim, int b_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  int idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = (a[idx_a] < b[idx_b]) ? {one} : {zero};
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
pub fn unary_prefix(kernel_name: &str, op: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size) {{
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

/// Generate kernel source for a unary function op (expf, logf, sinf,
/// sqrtf). `func` is the f32-suffixed libm name; for f64 the f-suffix is
/// dropped per [`ElemKind::func`].
pub fn unary_func(kernel_name: &str, func: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let resolved = kind.func(func);
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {resolved}(a[idx]);
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for uniform_like random fill. The PRNG itself
/// always runs in f32 — the f64 variant simply widens at the final store
/// because `chelis_uniform_sample_f32` is the only PRNG the runtime ships
/// today and the spec does not pin a higher-precision tensor random
/// surface.
pub fn uniform_like(kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    float low, float high, unsigned long long seed,
    {ty} *out, {out_shape}, int out_ndim, int out_size) {{
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  out[i] = ({ty})chelis_uniform_sample_f32(seed, (unsigned long long)i, low, high);
}}
",
        out_shape = shape_params("out"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate kernel source for sum reduction (naive: one thread per
/// output element). `operand_kind` is the input precision; `acc_kind` is
/// the running accumulator precision (also the output precision per
/// spec/04-type-system.md §5.7.1: the result of `reduce_sum` IS the
/// accumulator). For f32→f32 and f64→f64 the two coincide; for the cross
/// case (f32 operand with f64 accumulator) the loaded value is promoted on
/// read. The WS-A4 i8/i16 → i32 promoted integer path lives in
/// [`reduce_sum_promoted`] because the integer dtypes are not (yet)
/// admitted by `ElemKind`.
pub fn reduce_sum(
    kernel_name: &str,
    axis: usize,
    operand_kind: ElemKind,
    acc_kind: ElemKind,
) -> String {
    let op_ty = operand_kind.c_type();
    let acc_ty = acc_kind.c_type();
    let zero = acc_kind.zero();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {op_ty} *a, {a_strides}, int a_ndim, int a_size,
    {acc_ty} *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {acc_ty} acc = {zero};
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
    acc += ({acc_ty})a[src_idx];
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

/// WS-A4: integer reduce_sum with a wider accumulator dtype.
/// `src_c_ty` is the element type of the input tensor (e.g. `int8_t`,
/// `int16_t`); `acc_c_ty` is the running-sum AND output element type
/// (e.g. `int32_t` for the spec/04-type-system.md §5.7.1 i8/i16 → i32
/// promoted path). Each source element is widened to `acc_c_ty` before
/// being added so partial sums of e.g. 200 i8 ones produce 200, not the
/// wrap-around result of accumulating at the source width.
pub fn reduce_sum_promoted(
    kernel_name: &str,
    axis: usize,
    src_c_ty: &str,
    acc_c_ty: &str,
) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {src_c_ty} *a, {a_strides}, int a_ndim, int a_size,
    {acc_c_ty} *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {acc_c_ty} acc = 0;
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
    acc += ({acc_c_ty})a[src_idx];
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

/// Generate kernel source for max reduction (naive: one thread per
/// output element). MaxReduce result precision matches the operand
/// precision; the accumulator runs at the operand precision.
pub fn reduce_max(kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_min();
    let fmax = kind.func("fmaxf");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {init};
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
    acc = {fmax}(acc, a[src_idx]);
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

/// Generate kernel source for min reduction. Mirrors `reduce_max` but
/// seeds the accumulator with `+max` and uses `fmin`. WS-A2 lifts this
/// from the previous "C backend only" deferral.
pub fn reduce_min(kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_max();
    let fmin = kind.func("fminf");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {init};
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
    acc = {fmin}(acc, a[src_idx]);
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

/// Generate kernel source for product reduction. Identity is `1.0`.
/// The eval-side `prod` adjoint needs special handling for zero elements;
/// that's a host-side AD concern, not a kernel concern.
pub fn reduce_prod(kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let one = kind.one();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {one};
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
    acc *= a[src_idx];
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

/// Generate kernel source for argmax reduction. Output dtype is `i64`
/// (matching the spec/05 §2.3 argmax/argmin signature). Tie-break is
/// "first index wins", matching the C backend.
pub fn reduce_argmax(kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_min();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    long long *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} best = {init};
  long long best_idx = 0;
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
    {ty} v = a[src_idx];
    if (v > best) {{ best = v; best_idx = (long long)k; }}
  }}
  out[outer] = best_idx;
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Mirror of [`reduce_argmax`] for argmin.
pub fn reduce_argmin(kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_max();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    long long *out, {out_shape}, int out_ndim, int out_size, int axis_size) {{
{build_a_s}
{build_out_sh}
  int outer = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  int out_indices[{MAX_DIM}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} best = {init};
  long long best_idx = 0;
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
    {ty} v = a[src_idx];
    if (v < best) {{ best = v; best_idx = (long long)k; }}
  }}
  out[outer] = best_idx;
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Resolve a [`chelis_ir::dag::FusedInput`] to a C expression. Shared
/// between [`fused_elementwise`] and [`reduce_fused`].
fn resolve_fused_input(input: &chelis_ir::dag::FusedInput) -> String {
    use chelis_ir::dag::FusedInput;
    match input {
        FusedInput::External(i) => format!("ext{i}[idx_ext{i}]"),
        FusedInput::PreviousStep(j) => format!("v{j}"),
    }
}

/// Generate the `{ty} v{i} = {expr};` body lines for a FusedStep chain.
/// `kind` selects the libm function names (e.g. `expf` vs `exp`) and the
/// register type.
fn fused_step_lines(
    steps: &[chelis_ir::dag::FusedStep],
    kind: ElemKind,
    indent: &str,
) -> Vec<String> {
    use chelis_ir::dag::FusedStepOp;
    let ty = kind.c_type();
    let one = kind.one_lit_bool();
    let zero = kind.zero_lit_bool();
    let mut step_lines = Vec::new();
    for (si, step) in steps.iter().enumerate() {
        let expr = match step.op {
            FusedStepOp::Add => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Mul => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::MaxElem => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                let fmax = kind.func("fmaxf");
                format!("{fmax}({a}, {b})")
            }
            FusedStepOp::CmpLt => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("({a} < {b}) ? {one} : {zero}")
            }
            FusedStepOp::Neg => {
                let a = resolve_fused_input(&step.input_indices[0]);
                format!("-{a}")
            }
            FusedStepOp::Exp => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("expf");
                format!("{f}({a})")
            }
            FusedStepOp::Log => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("logf");
                format!("{f}({a})")
            }
            FusedStepOp::Sin => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("sinf");
                format!("{f}({a})")
            }
            FusedStepOp::Sqrt => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("sqrtf");
                format!("{f}({a})")
            }
            FusedStepOp::Cos => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("cosf");
                format!("{f}({a})")
            }
            FusedStepOp::Tan => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("tanf");
                format!("{f}({a})")
            }
            FusedStepOp::Atan => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("atanf");
                format!("{f}({a})")
            }
            FusedStepOp::Abs => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("fabsf");
                format!("{f}({a})")
            }
            FusedStepOp::Floor => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("floorf");
                format!("{f}({a})")
            }
            FusedStepOp::Ceil => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("ceilf");
                format!("{f}({a})")
            }
        };
        step_lines.push(format!("{indent}{ty} v{si} = {expr};"));
    }
    step_lines
}

pub fn reduce_fused(
    kernel_name: &str,
    axis: usize,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
    reduce_kind: ReduceKind,
    kind: ElemKind,
) -> String {
    let ty = kind.c_type();

    // Build parameter list: external inputs (with strides), output (with shape)
    let mut params = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        params.push(format!("const {ty} *{pfx}"));
        params.push(stride_params(&pfx));
        params.push(format!("int {pfx}_ndim"));
        params.push(format!("int {pfx}_size"));
    }
    params.push(format!("{ty} *out"));
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

    let step_lines = fused_step_lines(steps, kind, "      ");

    let last_step = steps.len() - 1;
    let (init, accumulate) = match reduce_kind {
        ReduceKind::Sum => (kind.zero().to_string(), format!("acc += v{last_step};")),
        ReduceKind::Max => {
            let fmax = kind.func("fmaxf");
            (
                kind.init_min().to_string(),
                format!("acc = {fmax}(acc, v{last_step});"),
            )
        }
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
  {ty} acc = {init};
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
/// Each step computes into a register `{ty} v{step_idx}`, resolving inputs
/// from either external input arrays or previous step outputs.
///
/// When `in_place_aliased_ext` is `Some(i)`, the fused output is aliased
/// onto external input `i`'s device buffer at runtime (Perf-F2(b)).
/// To keep that alias sound, neither `ext{i}` nor `out` may carry a
/// `__restrict__` qualifier — the two pointers reference the same
/// memory in the in-place fast path. All other externals do carry
/// `__restrict__` so the compiler can still hoist their loads.
///
/// When `in_place_aliased_ext` is `None`, no aliasing is possible and
/// the legacy non-`__restrict__` parameter list is preserved (the HIP
/// fused kernel has historically not used `__restrict__` for either
/// the non-aliased nor aliased case).
pub fn fused_elementwise(
    kernel_name: &str,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
    in_place_aliased_ext: Option<usize>,
    kind: ElemKind,
) -> String {
    let ty = kind.c_type();

    // Build parameter list. `__restrict__` is added only when the
    // kernel ships the in-place aliasing path, and only on the
    // pointers that are provably non-aliasing (every external except
    // the aliased one, but never on `out`).
    let mut params = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        let qual = if in_place_aliased_ext.is_some() && in_place_aliased_ext != Some(i) {
            format!("const {ty} *__restrict__ ")
        } else {
            format!("const {ty} *")
        };
        params.push(format!("{qual}{pfx}"));
        params.push(stride_params(&pfx));
        params.push(format!("int {pfx}_ndim"));
        params.push(format!("int {pfx}_size"));
    }
    params.push(format!("{ty} *out"));
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

    let step_lines = fused_step_lines(steps, kind, "  ");

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
/// The value is passed in the destination precision so the host-side
/// emitter does not need a per-precision launch shim beyond casting the
/// literal.
pub fn fill(kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}({ty} *data, {ty} value, int size) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  data[i] = value;
}}
"
    )
}

/// Generate kernel source for cast / Realize / Copy (in-precision identity).
/// Mixed-precision casts (e.g. f32→f64) emit the dedicated
/// [`cast_convert`] kernel.
pub fn cast(kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, int a_ndim, int a_size,
    {ty} *out, {out_shape}, int out_ndim, int out_size) {{
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

/// Generate kernel source for a true cross-precision cast (e.g.
/// `f32 → f64`). Differs from [`cast`] only in that the source and
/// destination types may disagree.
pub fn cast_convert(kernel_name: &str, src: ElemKind, dst: ElemKind) -> String {
    let src_ty = src.c_type();
    let dst_ty = dst.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {src_ty} *a, {a_strides}, int a_ndim, int a_size,
    {dst_ty} *out, {out_shape}, int out_ndim, int out_size) {{
{build_a_s}
{build_out_sh}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int indices[{MAX_DIM}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  int idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = ({dst_ty})a[idx];
}}
",
        a_strides = stride_params("a"),
        out_shape = shape_params("out"),
        build_a_s = build_array("a_s", "a", "s"),
        build_out_sh = build_array("out_sh", "out", "sh"),
    )
}

/// Generate sparse gather kernel for typed payload precision and typed integer indices.
pub fn gather(kernel_name: &str, index_ty: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *values,
    const {index_ty} *indices,
    {ty} *out,
    int before,
    int axis_size,
    int after,
    int index_count,
    int total) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % after;
  int tmp = i / after;
  int index_pos = tmp % index_count;
  int b = tmp / index_count;
  int g = (int)indices[index_pos];
  if (g < 0 || g >= axis_size || b >= before) {{
    CHELIS_GUARD_INDEX(g, axis_size, 2);
    return;
  }}
  int src = ((b * axis_size + g) * after) + d;
  out[i] = values[src];
}}
"
    )
}

/// Generate sparse scatter-add kernel for typed payload precision and typed integer indices.
/// HIP's `atomicAdd` is overloaded for `float` and `double` on supported devices.
pub fn scatter_add(kernel_name: &str, index_ty: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {index_ty} *indices,
    const {ty} *updates,
    {ty} *out,
    int before,
    int axis_size,
    int after,
    int index_count,
    int total) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int d = i % after;
  int tmp = i / after;
  int index_pos = tmp % index_count;
  int b = tmp / index_count;
  int g = (int)indices[index_pos];
  if (g < 0 || g >= axis_size || b >= before) {{
    CHELIS_GUARD_INDEX(g, axis_size, 3);
    return;
  }}
  int dst = ((b * axis_size + g) * after) + d;
  atomicAdd(&out[dst], updates[i]);
}}
"
    )
}

/// Generate sparse replace-scatter (last-write-wins) kernel for f32
/// payloads and typed integer indices.
///
/// Per `spec/05-risc-primitives.md` §3.5, the deterministic order is
/// updates-tensor row-major flat iteration. HIP atomics do not
/// guarantee ordered last-write semantics across concurrent threads,
/// so this kernel is executed by a **single thread** that walks
/// `i = 0..total` in ascending flat order and writes each update
/// non-atomically. The launch site uses grid=1, block=1. This
/// trades GPU throughput for the determinism the AD policy
/// depends on. Higher-throughput strategies (sort-then-scatter,
/// segmented scan) require a tie-breaker that picks the max flat
/// index per target cell; they are a future optimization but must
/// preserve this exact tie-breaking rule.
pub fn scatter_replace(kernel_name: &str, index_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {index_ty} *indices,
    const float *updates,
    float *out,
    int before,
    int axis_size,
    int after,
    int index_count,
    int total) {{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int i = 0; i < total; i++) {{
    int d = i % after;
    int tmp = i / after;
    int index_pos = tmp % index_count;
    int b = tmp / index_count;
    int g = (int)indices[index_pos];
    if (g < 0 || g >= axis_size || b >= before) {{
      CHELIS_GUARD_INDEX(g, axis_size, 4);
      return;
    }}
    int dst = ((b * axis_size + g) * after) + d;
    out[dst] = updates[i];
  }}
}}
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_add_kernel_has_correct_structure() {
        let src = binary_elementwise("kernel_add", "+", ElemKind::F32);
        assert!(src.contains("extern \"C\" __global__ void kernel_add("));
        assert!(src.contains("out[i] = a[idx_a] + b[idx_b];"));
        assert!(src.contains("chelis_flat_to_indices"));
        assert!(src.contains("chelis_indices_to_flat"));
    }

    #[test]
    fn binary_add_uses_int_params_not_pointers() {
        let src = binary_elementwise("kernel_add", "+", ElemKind::F32);
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
    fn binary_add_f64_uses_double_buffers() {
        let src = binary_elementwise("kernel_add_f64", "+", ElemKind::F64);
        assert!(src.contains("const double *a"));
        assert!(src.contains("const double *b"));
        assert!(src.contains("double *out"));
        assert!(!src.contains("const float *a"));
    }

    #[test]
    fn unary_exp_kernel_correct() {
        let src = unary_func("kernel_exp", "expf", ElemKind::F32);
        assert!(src.contains("extern \"C\" __global__ void kernel_exp("));
        assert!(src.contains("out[i] = expf(a[idx]);"));
    }

    #[test]
    fn unary_exp_f64_uses_double_libm() {
        let src = unary_func("kernel_exp_f64", "expf", ElemKind::F64);
        assert!(src.contains("out[i] = exp(a[idx]);"));
        assert!(!src.contains("expf("));
        assert!(src.contains("const double *a"));
        assert!(src.contains("double *out"));
    }

    #[test]
    fn cmplt_kernel_returns_float() {
        let src = cmplt("kernel_cmplt", ElemKind::F32);
        assert!(src.contains("1.0f : 0.0f"));
    }

    #[test]
    fn cmplt_f64_uses_unsuffixed_constants() {
        let src = cmplt("kernel_cmplt_f64", ElemKind::F64);
        assert!(src.contains("1.0 : 0.0"));
        assert!(src.contains("const double *a"));
    }

    #[test]
    fn reduce_sum_kernel_has_axis_loop() {
        let src = reduce_sum("kernel_sum_ax0_f32", 0, ElemKind::F32, ElemKind::F32);
        assert!(src.contains("float acc = 0.0f;"));
        assert!(src.contains("for (int k = 0; k < axis_size; k++)"));
        assert!(src.contains("if (d == 0)"));
    }

    #[test]
    fn reduce_sum_f64_uses_double_accumulator() {
        let src = reduce_sum("kernel_sum_ax0_f64", 0, ElemKind::F64, ElemKind::F64);
        assert!(src.contains("double acc = 0.0;"));
        assert!(src.contains("const double *a"));
        assert!(src.contains("double *out"));
    }

    #[test]
    fn reduce_max_kernel_has_f32_min_sentinel() {
        let src = reduce_max("kernel_max_ax1", 1, ElemKind::F32);
        assert!(src.contains("-3.402823466e+38F"));
        assert!(src.contains("fmaxf(acc,"));
    }

    #[test]
    fn reduce_max_f64_uses_double_min_sentinel_and_fmax() {
        let src = reduce_max("kernel_max_ax1_f64", 1, ElemKind::F64);
        assert!(src.contains("-1.7976931348623157e+308"));
        assert!(src.contains("fmax(acc,"));
        assert!(!src.contains("fmaxf(acc,"));
    }

    #[test]
    fn reduce_min_seeds_with_max_finite() {
        let src = reduce_min("kernel_min_ax0_f32", 0, ElemKind::F32);
        assert!(src.contains("3.402823466e+38F"));
        assert!(src.contains("fminf(acc,"));
    }

    #[test]
    fn reduce_prod_uses_one_identity() {
        let src = reduce_prod("kernel_prod_ax0_f64", 0, ElemKind::F64);
        assert!(src.contains("double acc = 1.0;"));
        assert!(src.contains("acc *= a[src_idx];"));
    }

    #[test]
    fn reduce_argmax_emits_long_long_output() {
        let src = reduce_argmax("kernel_argmax_ax0_f32", 0, ElemKind::F32);
        assert!(src.contains("long long *out"));
        assert!(src.contains("long long best_idx"));
        assert!(src.contains("if (v > best)"));
    }

    #[test]
    fn reduce_argmin_emits_long_long_output() {
        let src = reduce_argmin("kernel_argmin_ax0_f32", 0, ElemKind::F32);
        assert!(src.contains("long long *out"));
        assert!(src.contains("if (v < best)"));
    }

    #[test]
    fn fill_kernel_simple() {
        let src = fill("kernel_fill", ElemKind::F32);
        assert!(src.contains("data[i] = value;"));
        // Fill has no stride/shape params (operates on raw buffer)
        assert!(!src.contains("a_s0"));
    }

    #[test]
    fn fill_f64_uses_double_value() {
        let src = fill("kernel_fill_f64", ElemKind::F64);
        assert!(src.contains("double *data, double value"));
    }

    #[test]
    fn cast_convert_widens_f32_to_f64() {
        let src = cast_convert("kernel_cast_f32_to_f64", ElemKind::F32, ElemKind::F64);
        assert!(src.contains("const float *a"));
        assert!(src.contains("double *out"));
        assert!(src.contains("out[i] = (double)a[idx];"));
    }

    #[test]
    fn device_helpers_present_in_all_compute_kernels() {
        let add = binary_elementwise("k", "+", ElemKind::F32);
        let neg = unary_prefix("k", "-", ElemKind::F32);
        let sum = reduce_sum("k", 0, ElemKind::F32, ElemKind::F32);
        let fill_src = fill("k", ElemKind::F32);
        for src in [&add, &neg, &sum, &fill_src] {
            assert!(src.contains("__device__ void chelis_flat_to_indices"));
            assert!(src.contains("__device__ int chelis_indices_to_flat"));
            assert!(src.contains("__device__ int chelis_gpu_failure = 0;"));
            assert!(src.contains("CHELIS_GUARD_INDEX"));
        }
    }
}
