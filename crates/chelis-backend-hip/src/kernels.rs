//! HIP kernel source string templates for RISC operations.
//!
//! Each function returns a complete kernel source string ready for hiprtc
//! compilation. Device-side indexing helpers are prepended to every kernel.
//!
//! Shape and stride parameters use i64 metadata. Each program specializes
//! every template and launch argument list to the same maximum checked DAG
//! rank. Rank-zero programs carry one inert parameter slot; their semantic
//! rank remains zero and their null shape/stride arrays are never read.

/// Device-side helper functions included at the top of every kernel source.
pub const DEVICE_HELPERS: &str = "\
typedef signed char chelis_i8;
typedef short chelis_i16;
typedef int chelis_i32;
typedef long long chelis_i64;
typedef unsigned short chelis_u16;
typedef unsigned int chelis_u32;
typedef unsigned long long chelis_u64;
typedef chelis_i64 chelis_device_metadata;
__device__ unsigned int chelis_numeric_failure_flag = 0;
__device__ unsigned long long chelis_numeric_failure_index = ~0ULL;
__device__ void chelis_record_numeric_failure(unsigned long long index) {
    unsigned long long observed = chelis_numeric_failure_index;
    while (index < observed) {
        unsigned long long prior =
            atomicCAS(&chelis_numeric_failure_index, observed, index);
        if (prior == observed) break;
        observed = prior;
    }
    atomicExch(&chelis_numeric_failure_flag, 1u);
}
#if CHELIS_DEBUG_BOUNDS
__device__ int chelis_gpu_failure = 0;
__device__ void chelis_record_failure(int code) {
    if (code != 0) {
        atomicCAS(&chelis_gpu_failure, 0, code);
    }
}
__device__ chelis_device_metadata chelis_bounds_guard(chelis_device_metadata idx, chelis_device_metadata size, int code) {
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
__device__ void chelis_flat_to_indices(chelis_device_metadata flat, const chelis_device_metadata *shape, chelis_device_metadata ndim, chelis_device_metadata *out) {
    for (chelis_device_metadata d = ndim - 1; d >= 0; d--) {
        out[d] = flat % shape[d];
        flat /= shape[d];
    }
}
__device__ chelis_device_metadata chelis_indices_to_flat(const chelis_device_metadata *indices, const chelis_device_metadata *strides, chelis_device_metadata ndim) {
    chelis_device_metadata flat = 0;
    for (chelis_device_metadata d = 0; d < ndim; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}
__device__ chelis_device_metadata chelis_logical_offset(chelis_device_metadata linear, const chelis_device_metadata *shape, const chelis_device_metadata *strides, chelis_device_metadata rank) {
    chelis_device_metadata offset = 0;
    for (chelis_device_metadata axis = rank - 1; axis >= 0; --axis) {
        offset += (linear % shape[axis]) * strides[axis];
        linear /= shape[axis];
    }
    return offset;
}
";

/// Value-decoding helpers needed only by semantic f16/bf16 comparisons.
pub const REDUCED_FLOAT_COMPARISON_HELPERS: &str = "\
__device__ float chelis_f32_from_storage_bits(chelis_u32 bits) {
    union {
        chelis_u32 bits;
        float value;
    } decoded;
    decoded.bits = bits;
    return decoded.value;
}
__device__ chelis_u32 chelis_f32_to_storage_bits(float value) {
    union {
        chelis_u32 bits;
        float value;
    } encoded;
    encoded.value = value;
    return encoded.bits;
}
__device__ float chelis_bf16_to_f32(chelis_u16 value) {
    return chelis_f32_from_storage_bits(((chelis_u32)value) << 16);
}
__device__ float chelis_f16_to_f32(chelis_u16 value) {
    chelis_u32 sign = ((chelis_u32)value & 0x8000u) << 16;
    chelis_u32 exponent = ((chelis_u32)value >> 10) & 0x1fu;
    chelis_u32 fraction = (chelis_u32)value & 0x03ffu;
    chelis_u32 bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            chelis_u32 shifts = 0;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                shifts += 1;
            }
            fraction &= 0x03ffu;
            bits = sign | ((113u - shifts) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    return chelis_f32_from_storage_bits(bits);
}
__device__ chelis_u32 chelis_round_shift_right_even(chelis_u32 value, chelis_u32 shift) {
    chelis_u32 truncated = value >> shift;
    chelis_u32 remainder = value & ((1u << shift) - 1u);
    chelis_u32 halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (truncated & 1u) != 0)) {
        truncated += 1u;
    }
    return truncated;
}
__device__ chelis_u16 chelis_f32_to_bf16(float value) {
    chelis_u32 bits = chelis_f32_to_storage_bits(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (chelis_u16)0x7fc0u;
    }
    chelis_u32 rounding_bias = 0x00007fffu + ((bits >> 16) & 1u);
    return (chelis_u16)((bits + rounding_bias) >> 16);
}
__device__ chelis_u16 chelis_f32_to_f16(float value) {
    chelis_u32 bits = chelis_f32_to_storage_bits(value);
    chelis_u32 sign = bits & 0x80000000u;
    chelis_u32 exponent = bits & 0x7f800000u;
    chelis_u32 mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u) {
        return (chelis_u16)(mantissa == 0 ? ((sign >> 16) | 0x7c00u) : 0x7e00u);
    }
    chelis_u32 half_sign = sign >> 16;
    int unbiased_exponent = (int)(exponent >> 23) - 127;
    int half_exponent = unbiased_exponent + 15;
    if (half_exponent >= 0x1f) {
        return (chelis_u16)(half_sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        chelis_u32 shift = (chelis_u32)(14 - half_exponent);
        if (shift > 24u) {
            return (chelis_u16)half_sign;
        }
        mantissa |= 0x00800000u;
        chelis_u32 half_mantissa = chelis_round_shift_right_even(mantissa, shift);
        return (chelis_u16)(half_sign | half_mantissa);
    }
    chelis_u32 encoded_exponent = ((chelis_u32)half_exponent) << 10;
    chelis_u32 shift = 13u;
    chelis_u32 half_mantissa = chelis_round_shift_right_even(mantissa, shift);
    return (chelis_u16)(half_sign | (encoded_exponent + half_mantissa));
}
";

/// Device-side helpers needed only by uniform random kernels: the HIP port
/// of `chelis_types::dtype_semantics`'s `[05-RNG-1]` word and `[05-OP-8]`
/// samplers (chelis#2408). A sampler takes the draw key the lowering baked,
/// never a seed.
pub const NUMERIC_DEVICE_HELPERS: &str = "\
__device__ unsigned long long chelis_random_mix(unsigned long long value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}
__device__ double chelis_random_unit(unsigned long long key, unsigned long long index) {
    unsigned long long element = chelis_random_mix(index);
    unsigned long long word = chelis_random_mix(key ^ ((element << 41) | (element >> 23)));
    return (double)(word >> 11) / (double)(1ULL << 53);
}
__device__ float chelis_uniform_sample_f32(unsigned long long key, unsigned long long index, float low, float high) {
    return fmaf(high - low, (float)chelis_random_unit(key, index), low);
}
__device__ double chelis_uniform_sample_f64(unsigned long long key, unsigned long long index, double low, double high) {
    return fma(high - low, chelis_random_unit(key, index), low);
}
";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducedFloatKind {
    F16,
    Bf16,
}

impl ReducedFloatKind {
    fn decode(self) -> &'static str {
        match self {
            Self::F16 => "chelis_f16_to_f32",
            Self::Bf16 => "chelis_bf16_to_f32",
        }
    }

    fn encode(self) -> &'static str {
        match self {
            Self::F16 => "chelis_f32_to_f16",
            Self::Bf16 => "chelis_f32_to_bf16",
        }
    }
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
                "rintf" => "rint".into(),
                "fmaxf" => "fmax".into(),
                "fminf" => "fmin".into(),
                other => other.to_string(),
            },
        }
    }
}

/// Stride parameter names for one tensor: `{prefix}_s0, {prefix}_s1, ..., {prefix}_s7`.
fn stride_params(rank: usize, prefix: &str) -> String {
    (0..rank)
        .map(|i| format!("chelis_device_metadata {prefix}_s{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Shape parameter names: `{prefix}_sh0, {prefix}_sh1, ..., {prefix}_sh7`.
fn shape_params(rank: usize, prefix: &str) -> String {
    (0..rank)
        .map(|i| format!("chelis_device_metadata {prefix}_sh{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Emit code to build a local `chelis_device_metadata[]` array from individual params.
fn build_array(rank: usize, var_name: &str, prefix: &str, suffix: &str) -> String {
    let elems: Vec<String> = (0..rank).map(|i| format!("{prefix}_{suffix}{i}")).collect();
    format!(
        "  chelis_device_metadata {var_name}[] = {{ {} }};",
        elems.join(", ")
    )
}

/// Generic per-axis `chelis_device_metadata` parameter list `{prefix}_{suffix}0 .. {suffix}7`.
/// Used by `pad`/`shrink` for the per-axis low-padding / start-offset /
/// source-shape vectors that are not strides or output shapes.
fn int_params(rank: usize, prefix: &str, suffix: &str) -> String {
    (0..rank)
        .map(|i| format!("chelis_device_metadata {prefix}_{suffix}{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build a local `chelis_device_metadata[]` array from the [`int_params`] declarations.
fn build_int_array(rank: usize, var_name: &str, prefix: &str, suffix: &str) -> String {
    build_array(rank, var_name, prefix, suffix)
}

/// Dedicated [05-OP-29] Count kernel.
///
/// `bool_c_ty` is the backend's spelling of the exact one-byte `Bool8`
/// carrier chelis#1308 landed (`unsigned char`, see `dtype_c_type`), so the
/// kernel reads one byte per element and never a four-byte payload;
/// `out_c_ty` is the same authority's spelling of the i64 result, so this
/// template names no element type of its own (chelis#893). The
/// runtime's `chelis_tensor_end_write` already rejects a byte outside {0, 1}
/// at the host write boundary, so error code 1 is a backstop for a producer
/// that bypasses the runtime, never the primary check. Each output thread
/// enumerates its selected-axis leaves in original row-major order, then
/// evaluates the specified adjacent-pair tree with a fixed-depth explicit
/// stack. Error code 1 is a non-boolean payload, 2 is checked-i64 overflow,
/// 3 is a stack hardware limit, and 4 is an invalid storage index.
pub fn count(
    rank: usize,
    name: &str,
    axes: &[usize],
    input_rank: usize,
    bool_c_ty: &str,
    out_c_ty: &str,
) -> String {
    let output_rank = input_rank - axes.len();
    let mut selected = vec![0; rank];
    for &axis in axes {
        selected[axis] = 1;
    }
    let selected_literal = selected
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let a_strides = stride_params(rank, "a");
    let a_shapes = shape_params(rank, "a");
    let out_shapes = shape_params(rank, "out");
    let build_a_strides = build_array(rank, "a_strides", "a", "s");
    let build_a_shapes = build_array(rank, "input_shape", "a", "sh");
    let build_out_shapes = build_array(rank, "output_shape", "out", "sh");

    format!(
        r#"{DEVICE_HELPERS}
__device__ void chelis_count_record_error(int *count_error, int code) {{
  if (code != 0) atomicCAS(count_error, 0, code);
}}
extern "C" __global__ void {name}(
    const {bool_c_ty} *a, {a_strides}, {a_shapes},
    chelis_device_metadata input_ndim, chelis_device_metadata input_size,
    {out_c_ty} *out, {out_shapes},
    chelis_device_metadata output_ndim, chelis_device_metadata output_size,
    chelis_device_metadata count_n, int *count_error) {{
  chelis_device_metadata out_flat =
      (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (out_flat >= output_size) return;
  if (input_ndim != {input_rank} || output_ndim != {output_rank}) {{
    chelis_count_record_error(count_error, 4);
    return;
  }}
  {build_a_strides}
  {build_a_shapes}
  {build_out_shapes}
  const int __count_selected[{rank}] = {{ {selected_literal} }};
  chelis_device_metadata __count_output_indices[{rank}] = {{ 0 }};
  chelis_device_metadata __count_full_indices[{rank}] = {{ 0 }};
  chelis_flat_to_indices(out_flat, output_shape, output_ndim, __count_output_indices);
  chelis_device_metadata __count_output_axis = 0;
  for (chelis_device_metadata __count_axis = 0; __count_axis < input_ndim; ++__count_axis) {{
    if (!__count_selected[__count_axis]) {{
      __count_full_indices[__count_axis] = __count_output_indices[__count_output_axis++];
    }}
  }}

  long long __count_frame_start[64];
  long long __count_frame_len[64];
  unsigned char __count_frame_state[64];
  long long __count_frame_left[64];
  int __count_sp = 0;
  __count_frame_start[0] = 0;
  __count_frame_len[0] = count_n;
  __count_frame_state[0] = 0;
  long long __count_value = 0;
  bool __count_have_value = false;

  while (__count_sp >= 0) {{
    if (__count_have_value) {{
      if (__count_frame_state[__count_sp] == 1) {{
        __count_frame_left[__count_sp] = __count_value;
        __count_frame_state[__count_sp] = 2;
        long long __count_len = __count_frame_len[__count_sp];
        unsigned long long __count_split = 1;
        while ((__count_split << 1) < (unsigned long long)__count_len) __count_split <<= 1;
        if (__count_sp == 63) {{
          chelis_count_record_error(count_error, 3);
          return;
        }}
        long long __count_start = __count_frame_start[__count_sp];
        ++__count_sp;
        __count_frame_start[__count_sp] = __count_start + (long long)__count_split;
        __count_frame_len[__count_sp] = __count_len - (long long)__count_split;
        __count_frame_state[__count_sp] = 0;
        __count_have_value = false;
        continue;
      }}
      if (__count_frame_state[__count_sp] == 2) {{
        long long __count_right = __count_value;
        if (__count_frame_left[__count_sp] > INT64_MAX - __count_right) {{
          chelis_count_record_error(count_error, 2);
          return;
        }}
        __count_value = __count_frame_left[__count_sp] + __count_right;
        --__count_sp;
        continue;
      }}
      chelis_count_record_error(count_error, 3);
      return;
    }}

    long long __count_len = __count_frame_len[__count_sp];
    if (__count_len == 0) {{
      __count_value = 0;
      --__count_sp;
      __count_have_value = true;
      continue;
    }}
    if (__count_len == 1) {{
      long long __count_leaf = __count_frame_start[__count_sp];
      long long __count_rem = __count_leaf;
      for (chelis_device_metadata __count_axis = input_ndim - 1;
           __count_axis >= 0; --__count_axis) {{
        if (__count_selected[__count_axis]) {{
          chelis_device_metadata __count_extent = input_shape[__count_axis];
          if (__count_extent <= 0) {{
            chelis_count_record_error(count_error, 4);
            return;
          }}
          __count_full_indices[__count_axis] =
              (chelis_device_metadata)(__count_rem % __count_extent);
          __count_rem /= __count_extent;
        }}
      }}
      long long __count_offset = 0;
      for (chelis_device_metadata __count_axis = 0;
           __count_axis < input_ndim; ++__count_axis) {{
        __count_offset += (long long)__count_full_indices[__count_axis] * a_strides[__count_axis];
      }}
      if (__count_offset < 0 || __count_offset >= input_size) {{
        chelis_count_record_error(count_error, 4);
        return;
      }}
      {bool_c_ty} __count_bit = a[__count_offset];
      if (__count_bit != ({bool_c_ty})0 && __count_bit != ({bool_c_ty})1) {{
        chelis_count_record_error(count_error, 1);
        return;
      }}
      __count_value = (__count_bit == ({bool_c_ty})1) ? 1 : 0;
      --__count_sp;
      __count_have_value = true;
      continue;
    }}

    unsigned long long __count_split = 1;
    while ((__count_split << 1) < (unsigned long long)__count_len) __count_split <<= 1;
    __count_frame_state[__count_sp] = 1;
    if (__count_sp == 63) {{
      chelis_count_record_error(count_error, 3);
      return;
    }}
    long long __count_start = __count_frame_start[__count_sp];
    ++__count_sp;
    __count_frame_start[__count_sp] = __count_start;
    __count_frame_len[__count_sp] = (long long)__count_split;
    __count_frame_state[__count_sp] = 0;
  }}

  out[out_flat] = __count_value;
}}
"#
    )
}

/// The activation gate of a checking kernel (spec/10 section 3.2): a node
/// whose activation is false computes a value but checks nothing, so where
/// an output element's row is inactive, each operand read takes the value
/// `inactive` names for its slot, one every check of the operation accepts
/// (`chelis_ir::dag::DagNode::inactive_operand`, the values the evaluator
/// and the C lane substitute). The check and the computation stay as they
/// are; only what they read changes. A kernel built without a gate is
/// byte-identical to its ungated form.
///
/// The gated kernel takes trailing parameters after the output's: the
/// activation's bool bytes with their shape, strides and rank (it may be a
/// view, such as a call site's activation expanded to every row), its
/// element count, and the output elements per activation row. Each thread
/// computes one output element `i` and reads the activation once, before
/// any operand: a rank-0 activation is its one element, a per-row one
/// (under `vmap`) is the element of `i`'s row, `i / row`, and a `row` of
/// zero (an activation of higher rank than the node) is "some row is
/// active".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandGate {
    /// The value each operand slot reads where the element is inactive.
    pub inactive: Vec<i64>,
}

/// The device helper a gated kernel reads its activation through; see
/// [`OperandGate`].
pub const ACTIVATION_GATE_HELPER: &str = "\
__device__ bool chelis_activation_at(const unsigned char *act, const chelis_device_metadata *shape, const chelis_device_metadata *strides, chelis_device_metadata ndim, chelis_device_metadata count, chelis_device_metadata row, chelis_device_metadata i) {
    if (row > 0) return act[chelis_logical_offset(i / row, shape, strides, ndim)] != 0;
    for (chelis_device_metadata r = 0; r < count; ++r) {
        if (act[chelis_logical_offset(r, shape, strides, ndim)] != 0) return true;
    }
    return false;
}
";

impl OperandGate {
    /// The device helper a gated kernel declares after [`DEVICE_HELPERS`].
    fn helper(gate: Option<&Self>) -> &'static str {
        if gate.is_some() {
            ACTIVATION_GATE_HELPER
        } else {
            ""
        }
    }

    /// The trailing parameters, after the output's `out_size`, for a
    /// kernel specialized to `rank`.
    fn params(gate: Option<&Self>, rank: usize) -> String {
        match gate {
            None => String::new(),
            Some(_) => format!(
                ",\n    const unsigned char *chelis_act, {}, {}, chelis_device_metadata chelis_act_ndim, chelis_device_metadata chelis_act_count, chelis_device_metadata chelis_act_row",
                shape_params(rank, "chelis_act"),
                stride_params(rank, "chelis_act"),
            ),
        }
    }

    /// The thread's one activation read, placed after its bounds check.
    fn prologue(gate: Option<&Self>, rank: usize) -> String {
        match gate {
            None => String::new(),
            Some(_) => format!(
                "{}\n{}\n  const bool chelis_active = chelis_activation_at(chelis_act, chelis_act_sh, chelis_act_s, chelis_act_ndim, chelis_act_count, chelis_act_row, i);\n",
                build_array(rank, "chelis_act_sh", "chelis_act", "sh"),
                build_array(rank, "chelis_act_s", "chelis_act", "s"),
            ),
        }
    }

    /// Operand `slot`'s read `read` of a `c_ty` element, as the gate
    /// leaves it.
    fn read(gate: Option<&Self>, slot: usize, read: &str, c_ty: &str) -> String {
        match gate {
            None => read.to_string(),
            Some(gate) => {
                let inactive = match gate.inactive[slot] {
                    i64::MIN => "(-9223372036854775807LL - 1)".to_string(),
                    value => format!("{value}LL"),
                };
                format!("(chelis_active ? {read} : ({c_ty})({inactive}))")
            }
        }
    }
}

/// WS-A2 + WS-A4: dtype-parameterized binary elementwise op (add, mul).
/// `elem_c_ty` is the C++ type spelling (e.g. `float`, `double`,
/// `int8_t`, `int16_t`) used for both operand pointers and the result
/// pointer. Same-precision arithmetic per spec/04-type-system.md §5.4
/// (no implicit promotion); both inputs and the output share
/// `elem_c_ty`. The accompanying kernel name should already encode the
/// dtype suffix (e.g. `kernel_add_f64`, `kernel_add_i8`). A `gate` reads
/// the operands through an activation ([`OperandGate`]).
pub fn binary_elementwise_typed(
    rank: usize,
    kernel_name: &str,
    op: &str,
    elem_c_ty: &str,
    gate: Option<&OperandGate>,
) -> String {
    let a = OperandGate::read(gate, 0, "a[idx_a]", elem_c_ty);
    let b = OperandGate::read(gate, 1, "b[idx_b]", elem_c_ty);
    let expression = match (op, elem_c_ty) {
        ("-", "float") => {
            format!("isnan({a} - {b}) ? __int_as_float(0x7fc00000) : {a} - {b}")
        }
        ("-", "double") => {
            format!("isnan({a} - {b}) ? __longlong_as_double(0x7ff8000000000000LL) : {a} - {b}")
        }
        _ => format!("{a} {op} {b}"),
    };
    format!(
        "{DEVICE_HELPERS}{helper}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {elem_c_ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    {elem_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size{params}) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
{active}  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = {expression};
}}
",
        helper = OperandGate::helper(gate),
        params = OperandGate::params(gate, rank),
        active = OperandGate::prologue(gate, rank),
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// WS-A2 thin wrapper for the legacy `ElemKind`-based call sites and
/// for callers that already have an `ElemKind` in hand. Forwards to the
/// dtype-parameterized [`binary_elementwise_typed`] using the
/// `ElemKind`'s C-type spelling (`float` or `double`), without a gate.
pub fn binary_elementwise(rank: usize, kernel_name: &str, op: &str, kind: ElemKind) -> String {
    binary_elementwise_typed(rank, kernel_name, op, kind.c_type(), None)
}

/// Direct checked signed-integer subtraction. The bounds test avoids
/// evaluating an overflowing signed expression, and every failing lane
/// contributes its row-major flat index to the module-local numeric trap
/// record. The host reads that record after dispatch and raises [04-NUM-9]'s
/// exact branded trap. A `gate` reads the operands through an activation
/// ([`OperandGate`]): an inactive element subtracts zero from zero.
pub fn binary_checked_sub_integer(
    rank: usize,
    kernel_name: &str,
    elem_c_ty: &str,
    minimum: &str,
    maximum: &str,
    gate: Option<&OperandGate>,
) -> String {
    format!(
        "{DEVICE_HELPERS}{helper}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {elem_c_ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    {elem_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size{params}) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
{active}  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  {elem_c_ty} av = {a};
  {elem_c_ty} bv = {b};
  bool overflow =
      (bv > ({elem_c_ty})0 && av < ({elem_c_ty})({minimum} + bv))
      || (bv < ({elem_c_ty})0 && av > ({elem_c_ty})({maximum} + bv));
  if (overflow) {{
    chelis_record_numeric_failure((unsigned long long)i);
    out[i] = ({elem_c_ty})0;
    return;
  }}
  out[i] = ({elem_c_ty})(av - bv);
}}
",
        helper = OperandGate::helper(gate),
        params = OperandGate::params(gate, rank),
        active = OperandGate::prologue(gate, rank),
        a = OperandGate::read(gate, 0, "a[idx_a]", elem_c_ty),
        b = OperandGate::read(gate, 1, "b[idx_b]", elem_c_ty),
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// f16/bf16 subtraction computes at f32 and finalizes exactly once to the
/// operand storage width.
pub fn binary_sub_reduced(rank: usize, kernel_name: &str, kind: ReducedFloatKind) -> String {
    let decode = kind.decode();
    let encode = kind.encode();
    format!(
        "{DEVICE_HELPERS}{REDUCED_FLOAT_COMPARISON_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const chelis_u16 *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const chelis_u16 *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    chelis_u16 *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = {encode}({decode}(a[idx_a]) - {decode}(b[idx_b]));
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// chelis#178: floor-division kernel (round quotient toward −∞).
///
/// - `is_int == true` (integer dtype): native `/` plus a remainder-sign
///   correction, matching the C backend and evaluator.
/// - `is_int == false` (float dtype): `floorf(a / b)` (the device `floorf`
///   handles the f32/f64 promotion through the C type).
///
/// A `gate` reads the operands through an activation ([`OperandGate`]):
/// an inactive element divides zero by one.
pub fn binary_floor_div_typed(
    rank: usize,
    kernel_name: &str,
    elem_c_ty: &str,
    is_int: bool,
    gate: Option<&OperandGate>,
) -> String {
    let a = OperandGate::read(gate, 0, "a[idx_a]", elem_c_ty);
    let b = OperandGate::read(gate, 1, "b[idx_b]", elem_c_ty);
    let compute = if is_int {
        format!(
            "  {elem_c_ty} an = {a};\n  \
             {elem_c_ty} bn = {b};\n  \
             {elem_c_ty} q = an / bn;\n  \
             {elem_c_ty} r = an % bn;\n  \
             if (r != 0 && ((r < 0) != (bn < 0))) q -= 1;\n  \
             out[i] = q;"
        )
    } else {
        format!("  out[i] = floorf({a} / {b});")
    };
    format!(
        "{DEVICE_HELPERS}{helper}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {elem_c_ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    {elem_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size{params}) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
{active}  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
{compute}
}}
",
        helper = OperandGate::helper(gate),
        params = OperandGate::params(gate, rank),
        active = OperandGate::prologue(gate, rank),
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate an exact direct extrema-selection kernel. The chosen operand is
/// assigned unchanged so NaN payloads/signs and signed zero bits survive.
pub fn binary_extrema(rank: usize, kernel_name: &str, is_max: bool, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let comparison = if is_max { ">=" } else { "<=" };
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  {ty} av = a[idx_a];
  {ty} bv = b[idx_b];
  bool select_left = isnan(av) || (!isnan(bv) && av {comparison} bv);
  out[i] = select_left ? av : bv;
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Exact signed-integer direct extrema selection. Equality selects the lhs;
/// no arithmetic is performed.
pub fn binary_extrema_integer(
    rank: usize,
    kernel_name: &str,
    is_max: bool,
    elem_c_ty: &str,
) -> String {
    let comparison = if is_max { ">=" } else { "<=" };
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {elem_c_ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    {elem_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  {elem_c_ty} av = a[idx_a];
  {elem_c_ty} bv = b[idx_b];
  out[i] = av {comparison} bv ? av : bv;
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Exact f16/bf16 extrema. Comparison happens on decoded f32 values, while
/// the selected stored operand is copied unchanged.
pub fn binary_extrema_reduced(
    rank: usize,
    kernel_name: &str,
    is_max: bool,
    kind: ReducedFloatKind,
) -> String {
    let comparison = if is_max { ">=" } else { "<=" };
    let decode = kind.decode();
    format!(
        "{DEVICE_HELPERS}{REDUCED_FLOAT_COMPARISON_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const chelis_u16 *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const chelis_u16 *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    chelis_u16 *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  float av = {decode}(a[idx_a]);
  float bv = {decode}(b[idx_b]);
  bool select_left = isnan(av) || (!isnan(bv) && av {comparison} bv);
  out[i] = select_left ? a[idx_a] : b[idx_b];
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate the AD-only exact extrema cotangent kernel. Inputs are the two
/// forward operands and the incoming cotangent; output is the complete
/// cotangent for the selected operand and exact positive zero otherwise.
pub fn extrema_adjoint(
    rank: usize,
    kernel_name: &str,
    is_max: bool,
    select_left_operand: bool,
    kind: ElemKind,
) -> String {
    let ty = kind.c_type();
    let comparison = if is_max { ">=" } else { "<=" };
    let selected = if select_left_operand {
        "select_left"
    } else {
        "!select_left"
    };
    let zero = kind.zero_lit_bool();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    const {ty} *g, {g_strides}, chelis_device_metadata g_ndim, chelis_device_metadata g_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_g_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  chelis_device_metadata idx_g = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, g_s, g_ndim), g_size, 1);
  {ty} av = a[idx_a];
  {ty} bv = b[idx_b];
  bool select_left = isnan(av) || (!isnan(bv) && av {comparison} bv);
  out[i] = {selected} ? g[idx_g] : {zero};
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        g_strides = stride_params(rank, "g"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_g_s = build_array(rank, "g_s", "g", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Exact f16/bf16 extrema cotangent routing. The comparison decodes values;
/// the selected cotangent is copied as raw storage bits.
pub fn extrema_adjoint_reduced(
    rank: usize,
    kernel_name: &str,
    is_max: bool,
    select_left_operand: bool,
    kind: ReducedFloatKind,
) -> String {
    let comparison = if is_max { ">=" } else { "<=" };
    let selected = if select_left_operand {
        "select_left"
    } else {
        "!select_left"
    };
    let decode = kind.decode();
    format!(
        "{DEVICE_HELPERS}{REDUCED_FLOAT_COMPARISON_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const chelis_u16 *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const chelis_u16 *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    const chelis_u16 *g, {g_strides}, chelis_device_metadata g_ndim, chelis_device_metadata g_size,
    chelis_u16 *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_g_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  chelis_device_metadata idx_g = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, g_s, g_ndim), g_size, 1);
  float av = {decode}(a[idx_a]);
  float bv = {decode}(b[idx_b]);
  bool select_left = isnan(av) || (!isnan(bv) && av {comparison} bv);
  out[i] = {selected} ? g[idx_g] : (chelis_u16)0;
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        g_strides = stride_params(rank, "g"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_g_s = build_array(rank, "g_s", "g", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate the dedicated ReLU kernel. The selected input is assigned
/// unchanged, preserving NaN payloads/signs and negative zero bits.
pub fn relu(rank: usize, kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let zero = kind.zero_lit_bool();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  {ty} value = a[idx];
  out[i] = value < {zero} ? {zero} : value;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate the AD-only ReLU cotangent kernel. The incoming cotangent is
/// copied exactly only for strictly positive inputs; every rejected lane is
/// exact positive zero, including both zeros and NaNs.
pub fn relu_adjoint(rank: usize, kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let zero = kind.zero_lit_bool();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {ty} *g, {g_strides}, chelis_device_metadata g_ndim, chelis_device_metadata g_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_g_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_g = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, g_s, g_ndim), g_size, 1);
  out[i] = {zero} < a[idx_a] ? g[idx_g] : {zero};
}}
",
        a_strides = stride_params(rank, "a"),
        g_strides = stride_params(rank, "g"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_g_s = build_array(rank, "g_s", "g", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Raw IEEE-754 binary16/bfloat16 ReLU. HIP's narrow tensors use a tagged
/// 16-bit unsigned carrier outside matmul, so the predicate is expressed on the
/// sign/exponent/fraction fields and the selected stored bits are copied.
pub fn relu_reduced(
    rank: usize,
    kernel_name: &str,
    exponent_mask: u16,
    fraction_mask: u16,
) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const unsigned short *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    unsigned short *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  unsigned short value = a[idx];
  bool is_nan = (value & 0x{exponent_mask:04x}u) == 0x{exponent_mask:04x}u && (value & 0x{fraction_mask:04x}u) != 0;
  bool is_negative = (value & 0x8000u) != 0 && (value & 0x7fffu) != 0 && !is_nan;
  out[i] = is_negative ? (unsigned short)0 : value;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Raw IEEE-754 binary16/bfloat16 ReLU adjoint. `0 < x` is true exactly
/// for positive, nonzero, non-NaN encodings; selected cotangent bits survive.
pub fn relu_adjoint_reduced(
    rank: usize,
    kernel_name: &str,
    exponent_mask: u16,
    fraction_mask: u16,
) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const unsigned short *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const unsigned short *g, {g_strides}, chelis_device_metadata g_ndim, chelis_device_metadata g_size,
    unsigned short *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_g_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_g = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, g_s, g_ndim), g_size, 1);
  unsigned short value = a[idx_a];
  bool is_nan = (value & 0x{exponent_mask:04x}u) == 0x{exponent_mask:04x}u && (value & 0x{fraction_mask:04x}u) != 0;
  bool is_positive = (value & 0x8000u) == 0 && (value & 0x7fffu) != 0 && !is_nan;
  out[i] = is_positive ? g[idx_g] : (unsigned short)0;
}}
",
        a_strides = stride_params(rank, "a"),
        g_strides = stride_params(rank, "g"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_g_s = build_array(rank, "g_s", "g", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Direct comparison with Bool8 output. `lhs_expr` and `rhs_expr` are
/// storage-aware value expressions, so f16/bf16 compare semantically while
/// every other dtype compares its native stored-width value.
pub fn comparison(
    rank: usize,
    kernel_name: &str,
    op: &str,
    operand_c_ty: &str,
    value_helpers: &str,
    lhs_expr: &str,
    rhs_expr: &str,
) -> String {
    format!(
        "{DEVICE_HELPERS}{value_helpers}\
extern \"C\" __global__ void {kernel_name}(
    const {operand_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {operand_c_ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    unsigned char *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  chelis_device_metadata idx_b = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = (unsigned char)(({lhs_expr}) {op} ({rhs_expr}));
}}
",
        a_strides = stride_params(rank, "a"),
        b_strides = stride_params(rank, "b"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_b_s = build_array(rank, "b_s", "b", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Eager Bool8 binary logical operation.
pub fn logical_binary(rank: usize, kernel_name: &str, op: &str) -> String {
    comparison(
        rank,
        kernel_name,
        op,
        "unsigned char",
        "",
        "a[idx_a] != 0",
        "b[idx_b] != 0",
    )
}

/// Eager Bool8 logical negation.
pub fn logical_not(rank: usize, kernel_name: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const unsigned char *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    unsigned char *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_a = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = (unsigned char)(a[idx_a] == 0);
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Eager conditional selection over raw stored values. Float branch values
/// use integer carriers so NaN payloads and signed zero survive bit-for-bit.
/// Each element indexes only the branch its condition selects ([05-OP-53]).
pub fn where_stored(rank: usize, kernel_name: &str, branch_c_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const unsigned char *cond, {cond_strides}, chelis_device_metadata cond_ndim, chelis_device_metadata cond_size,
    const {branch_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    const {branch_c_ty} *b, {b_strides}, chelis_device_metadata b_ndim, chelis_device_metadata b_size,
    {branch_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_cond_s}
{build_a_s}
{build_b_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx_cond = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, cond_s, cond_ndim), cond_size, 1);
  chelis_device_metadata idx_a = cond[idx_cond] != 0 ? CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1) : 0;
  chelis_device_metadata idx_b = cond[idx_cond] != 0 ? 0 : CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, b_s, b_ndim), b_size, 1);
  out[i] = cond[idx_cond] != 0 ? a[idx_a] : b[idx_b];
}}
",
        // The generic ternary launcher names its three metadata groups a/b/g.
        cond_strides = stride_params(rank, "a"),
        a_strides = stride_params(rank, "b"),
        b_strides = stride_params(rank, "g"),
        out_shape = shape_params(rank, "out"),
        build_cond_s = build_array(rank, "cond_s", "a", "s"),
        build_a_s = build_array(rank, "a_s", "b", "s"),
        build_b_s = build_array(rank, "b_s", "g", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for a unary prefix op (neg: `-`).
pub fn unary_prefix(rank: usize, kernel_name: &str, op: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {op}a[idx];
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for IEEE elementwise reciprocal.
/// Emits `1.0f / a[idx]` (or `1.0 / a[idx]` for f64) — kept separate
/// from `unary_prefix` because the numerator is a typed constant, not
/// a prefix operator.
///
/// f32 / f64 only by `ElemKind` definition. Half-precision (`f16`,
/// `bf16`) coverage is tracked under the WS-A1 backlog (issue #174);
/// a future ElemKind extension that admits those dtypes must update
/// this `match` exhaustively.
pub fn unary_recip(rank: usize, kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let one = match kind {
        ElemKind::F32 => "1.0f",
        ElemKind::F64 => "1.0",
    };
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {one} / a[idx];
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for a unary function op (expf, logf, sinf,
/// sqrtf). `func` is the f32-suffixed libm name; for f64 the f-suffix is
/// dropped per [`ElemKind::func`].
pub fn unary_func(rank: usize, kernel_name: &str, func: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let resolved = kind.func(func);
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {resolved}(a[idx]);
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for `pad` (typed; full active dtype set).
///
/// One thread per output element. The output buffer is the materialized
/// contiguous slot, so thread `i` is the output flat index. Decompose it
/// into per-axis output indices over the output shape, subtract the
/// per-axis low padding to recover the source index, and copy the source
/// element if every source index is in `[0, src_shape[d])`. Otherwise
/// write the `fill` value. Mirrors the C backend's `emit_pad` semantics
/// and the evaluator's `pad` (spec/05-risc-primitives.md), but expressed
/// as a scatter-free per-output-element gather so concurrent threads
/// never race.
///
/// `elem_c_ty` is the C++ scalar spelling (`float`, `double`, `int8_t`,
/// …) chosen by the launch site via `dtype_c_type`. The kernel takes the
/// source strides + source shape + per-axis low offsets as runtime
/// arguments, so a single kernel per dtype serves every pad node of that
/// dtype regardless of rank or padding amounts.
pub fn pad_typed(rank: usize, kernel_name: &str, elem_c_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {pad_lo}, {src_sh},
    {elem_c_ty} fill,
    {elem_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_pad_lo}
{build_src_sh}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, out_indices);
  chelis_device_metadata src_indices[{rank}];
  chelis_device_metadata in_source = 1;
  for (chelis_device_metadata d = 0; d < out_ndim; d++) {{
    chelis_device_metadata s = out_indices[d] - pad_lo[d];
    src_indices[d] = s;
    if (s < 0 || s >= src_sh[d]) {{
      in_source = 0;
    }}
  }}
  if (in_source) {{
    chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(src_indices, a_s, a_ndim), a_size, 1);
    out[i] = a[idx];
  }} else {{
    out[i] = fill;
  }}
}}
",
        a_strides = stride_params(rank, "a"),
        pad_lo = int_params(rank, "pad", "lo"),
        src_sh = int_params(rank, "src", "sh"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_pad_lo = build_int_array(rank, "pad_lo", "pad", "lo"),
        build_src_sh = build_int_array(rank, "src_sh", "src", "sh"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for `shrink` (typed; full active dtype set).
///
/// One thread per output element over the materialized contiguous output
/// slot. The shrink output is always strictly inside the source, so every
/// output index maps to a valid source element: `src_index[d] =
/// out_index[d] + start[d]` (the `start` of each axis bound). Mirrors the
/// C backend's `emit_shrink` and the evaluator's `shrink`
/// (spec/05-risc-primitives.md).
pub fn shrink_typed(rank: usize, kernel_name: &str, elem_c_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {elem_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {shrink_start},
    {elem_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_a_s}
{build_start}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, out_indices);
  chelis_device_metadata src_indices[{rank}];
  for (chelis_device_metadata d = 0; d < out_ndim; d++) {{
    src_indices[d] = out_indices[d] + shrink_start[d];
  }}
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(src_indices, a_s, a_ndim), a_size, 1);
  out[i] = a[idx];
}}
",
        a_strides = stride_params(rank, "a"),
        shrink_start = int_params(rank, "shrink", "start"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_start = build_int_array(rank, "shrink_start", "shrink", "start"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for uniform_like random fill. [05-OP-8] binds
/// each output width to its own affine: f32 uses fmaf and f64 uses fma.
pub fn uniform_like(rank: usize, kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let sampler = match kind {
        ElemKind::F32 => "chelis_uniform_sample_f32",
        ElemKind::F64 => "chelis_uniform_sample_f64",
    };
    format!(
        "{DEVICE_HELPERS}{NUMERIC_DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {ty} low, {ty} high, unsigned long long key,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size) {{
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  out[i] = {sampler}(key, (unsigned long long)i, low, high);
}}
",
        out_shape = shape_params(rank, "out"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
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
    rank: usize,
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
    const {op_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {acc_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {acc_ty} acc = {zero};
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc += ({acc_ty})a[src_idx];
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
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
    rank: usize,
    kernel_name: &str,
    axis: usize,
    src_c_ty: &str,
    acc_c_ty: &str,
) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {src_c_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {acc_c_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {acc_c_ty} acc = 0;
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc += ({acc_c_ty})a[src_idx];
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for max reduction (naive: one thread per
/// output element). MaxReduce result precision matches the operand
/// precision; the accumulator runs at the operand precision.
pub fn reduce_max(rank: usize, kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_min();
    let fmax = kind.func("fmaxf");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {init};
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc = {fmax}(acc, a[src_idx]);
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for min reduction. Mirrors `reduce_max` but
/// seeds the accumulator with `+max` and uses `fmin`. WS-A2 lifts this
/// from the previous "C backend only" deferral.
pub fn reduce_min(rank: usize, kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_max();
    let fmin = kind.func("fminf");
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {init};
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc = {fmin}(acc, a[src_idx]);
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for product reduction. Identity is `1.0`.
/// The eval-side `prod` adjoint needs special handling for zero elements;
/// that's a host-side AD concern, not a kernel concern.
pub fn reduce_prod(rank: usize, kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let one = kind.one();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {one};
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    acc *= a[src_idx];
  }}
  out[outer] = acc;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for argmax reduction. Output dtype is `i64`
/// (matching the spec/05 §2.3 argmax/argmin signature). Tie-break is
/// "first index wins", matching the C backend.
pub fn reduce_argmax(rank: usize, kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_min();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    long long *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} best = {init};
  long long best_idx = 0;
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    {ty} v = a[src_idx];
    if (v > best) {{ best = v; best_idx = (long long)k; }}
  }}
  out[outer] = best_idx;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Mirror of [`reduce_argmax`] for argmin.
pub fn reduce_argmin(rank: usize, kernel_name: &str, axis: usize, kind: ElemKind) -> String {
    let ty = kind.c_type();
    let init = kind.init_max();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    long long *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size, chelis_device_metadata axis_size) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} best = {init};
  long long best_idx = 0;
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < a_ndim; d++) {{
      if (d == {axis}) {{
        full_indices[d] = k;
      }} else {{
        full_indices[d] = out_indices[out_d];
        out_d++;
      }}
    }}
    chelis_device_metadata src_idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, a_s, a_ndim), a_size, 1);
    {ty} v = a[src_idx];
    if (v < best) {{ best = v; best_idx = (long long)k; }}
  }}
  out[outer] = best_idx;
}}
",
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
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
    let mut step_lines = Vec::new();
    for (si, step) in steps.iter().enumerate() {
        let expr = match step.op {
            FusedStepOp::Add => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("{a} + {b}")
            }
            FusedStepOp::Sub => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                let canonical = match kind {
                    ElemKind::F32 => "__int_as_float(0x7fc00000)",
                    ElemKind::F64 => "__longlong_as_double(0x7ff8000000000000LL)",
                };
                let raw = format!("({a} - {b})");
                format!("isnan({raw}) ? {canonical} : {raw}")
            }
            FusedStepOp::Mul => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("{a} * {b}")
            }
            FusedStepOp::Div => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("{a} / {b}")
            }
            // chelis#178: the fused-elem path is float (`ElemKind`) only, so
            // only float `floor_div` reaches here — `floor(a / b)` via the
            // kind-resolved math fn. `trunc_div` is integer-only and cannot
            // fuse to this float path.
            FusedStepOp::FloorDiv => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                let floor = kind.func("floorf");
                format!("{floor}({a} / {b})")
            }
            FusedStepOp::TruncDiv => {
                unreachable!(
                    "trunc_div is integer-only (chelis#178); the HIP fused-elem path is \
                     float-only and cannot carry an integer trunc_div step"
                )
            }
            FusedStepOp::MaxElem => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("(isnan({a}) || (!isnan({b}) && ({a}) >= ({b})) ? ({a}) : ({b}))")
            }
            FusedStepOp::MinElem => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let b = resolve_fused_input(&step.input_indices[1]);
                format!("(isnan({a}) || (!isnan({b}) && ({a}) <= ({b})) ? ({a}) : ({b}))")
            }
            FusedStepOp::Neg => {
                let a = resolve_fused_input(&step.input_indices[0]);
                format!("-{a}")
            }
            FusedStepOp::Recip => {
                let a = resolve_fused_input(&step.input_indices[0]);
                format!("{one} / {a}")
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
            FusedStepOp::Round => {
                let a = resolve_fused_input(&step.input_indices[0]);
                let f = kind.func("rintf");
                format!("{f}({a})")
            }
        };
        step_lines.push(format!("{indent}{ty} v{si} = {expr};"));
    }
    step_lines
}

pub fn reduce_fused(
    rank: usize,
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
        params.push(stride_params(rank, &pfx));
        params.push(format!("chelis_device_metadata {pfx}_ndim"));
        params.push(format!("chelis_device_metadata {pfx}_size"));
    }
    params.push(format!("{ty} *out"));
    params.push(shape_params(rank, "out"));
    params.push("chelis_device_metadata out_ndim".into());
    params.push("chelis_device_metadata out_size".into());
    params.push("chelis_device_metadata axis_size".into());

    // Build local array constructions for external input strides
    let mut body_arrays = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        body_arrays.push(build_array(rank, &format!("{pfx}_s"), &pfx, "s"));
    }
    body_arrays.push(build_array(rank, "out_sh", "out", "sh"));

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
            "      chelis_device_metadata idx_{pfx} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(full_indices, {pfx}_s, {pfx}_ndim), {pfx}_size, 1);"
        ));
    }

    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  chelis_device_metadata outer = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (outer >= out_size) return;
  chelis_device_metadata out_indices[{rank}];
  chelis_flat_to_indices(outer, out_sh, out_ndim, out_indices);
  {ty} acc = {init};
  for (chelis_device_metadata k = 0; k < axis_size; k++) {{
    chelis_device_metadata full_indices[{rank}];
    chelis_device_metadata out_d = 0;
    for (chelis_device_metadata d = 0; d < out_ndim + 1; d++) {{
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

fn fused_reduced_step_lines(
    steps: &[chelis_ir::dag::FusedStep],
    kind: ReducedFloatKind,
) -> Vec<String> {
    use chelis_ir::dag::FusedStepOp;
    let decode = kind.decode();
    let encode = kind.encode();
    let mut lines = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        let left = resolve_fused_input(&step.input_indices[0]);
        let right = resolve_fused_input(&step.input_indices[1]);
        match step.op {
            FusedStepOp::Sub => {
                lines.push(format!(
                    "  chelis_u16 v{index} = {encode}({decode}({left}) - {decode}({right}));"
                ));
            }
            FusedStepOp::MaxElem | FusedStepOp::MinElem => {
                let comparison = if matches!(step.op, FusedStepOp::MaxElem) {
                    ">="
                } else {
                    "<="
                };
                lines.push(format!("  float v{index}_left = {decode}({left});"));
                lines.push(format!("  float v{index}_right = {decode}({right});"));
                lines.push(format!(
                    "  bool v{index}_select_left = isnan(v{index}_left) || (!isnan(v{index}_right) && v{index}_left {comparison} v{index}_right);"
                ));
                lines.push(format!(
                    "  chelis_u16 v{index} = v{index}_select_left ? {left} : {right};"
                ));
            }
            _ => unreachable!("the reduced fused HIP kernel admits only sub/max_elem/min_elem"),
        }
    }
    lines
}

/// Fused f16/bf16 direct arithmetic. Every step is finalized back to the
/// 16-bit stored representation before a later step consumes it, preserving
/// the per-operation finalization boundary required by [04-NUM-8].
pub fn fused_elementwise_reduced(
    rank: usize,
    kernel_name: &str,
    steps: &[chelis_ir::dag::FusedStep],
    n_external: usize,
    in_place_aliased_ext: Option<usize>,
    kind: ReducedFloatKind,
) -> String {
    let mut params = Vec::new();
    for i in 0..n_external {
        let prefix = format!("ext{i}");
        let qualifier = if in_place_aliased_ext.is_some() && in_place_aliased_ext != Some(i) {
            "const chelis_u16 *__restrict__ "
        } else {
            "const chelis_u16 *"
        };
        params.push(format!("{qualifier}{prefix}"));
        params.push(stride_params(rank, &prefix));
        params.push(format!("chelis_device_metadata {prefix}_ndim"));
        params.push(format!("chelis_device_metadata {prefix}_size"));
    }
    params.push("chelis_u16 *out".to_string());
    params.push(shape_params(rank, "out"));
    params.push("chelis_device_metadata out_ndim".to_string());
    params.push("chelis_device_metadata out_size".to_string());

    let mut arrays = Vec::new();
    let mut indices = Vec::new();
    for i in 0..n_external {
        let prefix = format!("ext{i}");
        arrays.push(build_array(rank, &format!("{prefix}_s"), &prefix, "s"));
        indices.push(format!(
            "  chelis_device_metadata idx_{prefix} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, {prefix}_s, {prefix}_ndim), {prefix}_size, 1);"
        ));
    }
    arrays.push(build_array(rank, "out_sh", "out", "sh"));
    let step_lines = fused_reduced_step_lines(steps, kind);
    let last_step = steps.len() - 1;

    format!(
        "{DEVICE_HELPERS}{REDUCED_FLOAT_COMPARISON_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
{indices}
{steps}
  out[i] = v{last_step};
}}
",
        params = params.join(",\n    "),
        arrays = arrays.join("\n"),
        indices = indices.join("\n"),
        steps = step_lines.join("\n"),
    )
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
    rank: usize,
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
        params.push(stride_params(rank, &pfx));
        params.push(format!("chelis_device_metadata {pfx}_ndim"));
        params.push(format!("chelis_device_metadata {pfx}_size"));
    }
    params.push(format!("{ty} *out"));
    params.push(shape_params(rank, "out"));
    params.push("chelis_device_metadata out_ndim".into());
    params.push("chelis_device_metadata out_size".into());

    // Build local array constructions
    let mut body_arrays = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        body_arrays.push(build_array(rank, &format!("{pfx}_s"), &pfx, "s"));
    }
    body_arrays.push(build_array(rank, "out_sh", "out", "sh"));

    // Build index computation for each external input
    let mut index_lines = Vec::new();
    for i in 0..n_external {
        let pfx = format!("ext{i}");
        index_lines.push(format!(
            "  chelis_device_metadata idx_{pfx} = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, {pfx}_s, {pfx}_ndim), {pfx}_size, 1);"
        ));
    }

    let step_lines = fused_step_lines(steps, kind, "  ");

    let last_step = steps.len() - 1;

    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    {params}) {{
{arrays}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  chelis_device_metadata indices[{rank}];
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
pub fn fill(_rank: usize, kernel_name: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}({ty} *data, {ty} value, chelis_device_metadata size) {{
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  data[i] = value;
}}
"
    )
}

/// Generate kernel source for cast / Realize / Copy (in-precision identity).
/// Mixed-precision casts (e.g. f32→f64) emit the dedicated
/// [`cast_convert`] kernel. A `gate` reads the source through an
/// activation ([`OperandGate`]).
pub fn cast(rank: usize, kernel_name: &str, kind: ElemKind, gate: Option<&OperandGate>) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}{helper}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size{params}) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
{active}  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = {a};
}}
",
        helper = OperandGate::helper(gate),
        params = OperandGate::params(gate, rank),
        active = OperandGate::prologue(gate, rank),
        a = OperandGate::read(gate, 0, "a[idx]", ty),
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate kernel source for a true cross-precision cast (e.g.
/// `f32 → f64`). Differs from [`cast`] only in that the source and
/// destination types may disagree. A `gate` reads the source through an
/// activation ([`OperandGate`]).
pub fn cast_convert(
    rank: usize,
    kernel_name: &str,
    src: ElemKind,
    dst: ElemKind,
    gate: Option<&OperandGate>,
) -> String {
    let src_ty = src.c_type();
    let dst_ty = dst.c_type();
    format!(
        "{DEVICE_HELPERS}{helper}\
extern \"C\" __global__ void {kernel_name}(
    const {src_ty} *a, {a_strides}, chelis_device_metadata a_ndim, chelis_device_metadata a_size,
    {dst_ty} *out, {out_shape}, chelis_device_metadata out_ndim, chelis_device_metadata out_size{params}) {{
{build_a_s}
{build_out_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
{active}  chelis_device_metadata indices[{rank}];
  chelis_flat_to_indices(i, out_sh, out_ndim, indices);
  chelis_device_metadata idx = CHELIS_GUARD_INDEX(chelis_indices_to_flat(indices, a_s, a_ndim), a_size, 1);
  out[i] = ({dst_ty}){a};
}}
",
        helper = OperandGate::helper(gate),
        params = OperandGate::params(gate, rank),
        active = OperandGate::prologue(gate, rank),
        a = OperandGate::read(gate, 0, "a[idx]", src_ty),
        a_strides = stride_params(rank, "a"),
        out_shape = shape_params(rank, "out"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Generate sparse gather kernel for typed payload precision and typed integer indices.
pub fn gather(rank: usize, kernel_name: &str, index_ty: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {ty} *values,
    const {index_ty} *indices,
    {ty} *out,
    chelis_device_metadata before,
    chelis_device_metadata axis_size,
    chelis_device_metadata after,
    chelis_device_metadata index_count,
    chelis_device_metadata total, {values_shape}, {values_strides}, chelis_device_metadata values_ndim, {idx_shape}, {idx_strides}, chelis_device_metadata idx_ndim) {{
{build_values_sh}
{build_values_s}
{build_idx_sh}
{build_idx_s}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  chelis_device_metadata d = i % after;
  chelis_device_metadata tmp = i / after;
  chelis_device_metadata index_pos = tmp % index_count;
  chelis_device_metadata b = tmp / index_count;
  chelis_device_metadata g = (chelis_device_metadata)indices[chelis_logical_offset(index_pos, idx_sh, idx_s, idx_ndim)];
  if (g < 0 || g >= axis_size || b >= before) {{
    CHELIS_GUARD_INDEX(g, axis_size, 2);
    return;
  }}
  chelis_device_metadata src = ((b * axis_size + g) * after) + d;
  out[i] = values[chelis_logical_offset(src, values_sh, values_s, values_ndim)];
}}
",
        values_shape = shape_params(rank, "values"),
        values_strides = stride_params(rank, "values"),
        build_values_sh = build_array(rank, "values_sh", "values", "sh"),
        build_values_s = build_array(rank, "values_s", "values", "s"),
        idx_shape = shape_params(rank, "idx"),
        idx_strides = stride_params(rank, "idx"),
        build_idx_sh = build_array(rank, "idx_sh", "idx", "sh"),
        build_idx_s = build_array(rank, "idx_s", "idx", "s"),
    )
}

/// Generate sparse scatter-add kernel for typed payload precision and typed integer indices.
/// HIP's `atomicAdd` is overloaded for `float` and `double` on supported devices.
pub fn scatter_add(rank: usize, kernel_name: &str, index_ty: &str, kind: ElemKind) -> String {
    let ty = kind.c_type();
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {index_ty} *indices,
    const {ty} *updates,
    {ty} *out,
    chelis_device_metadata before,
    chelis_device_metadata axis_size,
    chelis_device_metadata after,
    chelis_device_metadata index_count,
    chelis_device_metadata total, {idx_shape}, {idx_strides}, chelis_device_metadata idx_ndim, {updates_shape}, {updates_strides}, chelis_device_metadata updates_ndim) {{
{build_idx_sh}
{build_idx_s}
{build_updates_sh}
{build_updates_s}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  chelis_device_metadata d = i % after;
  chelis_device_metadata tmp = i / after;
  chelis_device_metadata index_pos = tmp % index_count;
  chelis_device_metadata b = tmp / index_count;
  chelis_device_metadata g = (chelis_device_metadata)indices[chelis_logical_offset(index_pos, idx_sh, idx_s, idx_ndim)];
  if (g < 0 || g >= axis_size || b >= before) {{
    CHELIS_GUARD_INDEX(g, axis_size, 3);
    return;
  }}
  chelis_device_metadata dst = ((b * axis_size + g) * after) + d;
  atomicAdd(&out[dst], updates[chelis_logical_offset(i, updates_sh, updates_s, updates_ndim)]);
}}
",
        idx_shape = shape_params(rank, "idx"),
        idx_strides = stride_params(rank, "idx"),
        build_idx_sh = build_array(rank, "idx_sh", "idx", "sh"),
        build_idx_s = build_array(rank, "idx_s", "idx", "s"),
        updates_shape = shape_params(rank, "updates"),
        updates_strides = stride_params(rank, "updates"),
        build_updates_sh = build_array(rank, "updates_sh", "updates", "sh"),
        build_updates_s = build_array(rank, "updates_s", "updates", "s"),
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
pub fn scatter_replace(rank: usize, kernel_name: &str, index_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {index_ty} *indices,
    const float *updates,
    float *out,
    chelis_device_metadata before,
    chelis_device_metadata axis_size,
    chelis_device_metadata after,
    chelis_device_metadata index_count,
    chelis_device_metadata total, {idx_shape}, {idx_strides}, chelis_device_metadata idx_ndim, {updates_shape}, {updates_strides}, chelis_device_metadata updates_ndim) {{
{build_idx_sh}
{build_idx_s}
{build_updates_sh}
{build_updates_s}
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (chelis_device_metadata i = 0; i < total; i++) {{
    chelis_device_metadata d = i % after;
    chelis_device_metadata tmp = i / after;
    chelis_device_metadata index_pos = tmp % index_count;
    chelis_device_metadata b = tmp / index_count;
    chelis_device_metadata g = (chelis_device_metadata)indices[chelis_logical_offset(index_pos, idx_sh, idx_s, idx_ndim)];
    if (g < 0 || g >= axis_size || b >= before) {{
      CHELIS_GUARD_INDEX(g, axis_size, 4);
      return;
    }}
    chelis_device_metadata dst = ((b * axis_size + g) * after) + d;
    out[dst] = updates[chelis_logical_offset(i, updates_sh, updates_s, updates_ndim)];
  }}
}}
",
        idx_shape = shape_params(rank, "idx"),
        idx_strides = stride_params(rank, "idx"),
        build_idx_sh = build_array(rank, "idx_sh", "idx", "sh"),
        build_idx_s = build_array(rank, "idx_s", "idx", "s"),
        updates_shape = shape_params(rank, "updates"),
        updates_strides = stride_params(rank, "updates"),
        build_updates_sh = build_array(rank, "updates_sh", "updates", "sh"),
        build_updates_s = build_array(rank, "updates_s", "updates", "s"),
    )
}

/// ONNX `ScatterElements` (spec §3.5.1). Element-wise: `indices` and
/// `updates` share a shape (`idx_sh`), `out` has the data shape
/// (`out_sh`), all sharing a rank. For each flat update index `i`, the
/// kernel decomposes `i` over `idx_sh`, replaces the `axis` coordinate
/// with `indices[i]`, and writes `updates[i]` at the corresponding
/// `out_sh` row-major offset. Single-thread serial (`<<<1,1>>>`) to
/// preserve deterministic last-write-wins at duplicate indices. The two
/// shapes are passed as `rank` scalar ints each (the launch site
/// reads `d_t->shape[d]`), matching the scalar-param convention used by
/// the elementwise kernels.
pub fn scatter_elements(rank: usize, kernel_name: &str, index_ty: &str) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const {index_ty} *indices,
    const float *updates,
    float *out,
    {idx_shape},
    {out_shape},
    chelis_device_metadata ndim,
    chelis_device_metadata axis,
    chelis_device_metadata axis_size,
    chelis_device_metadata total, {idx_strides}, {updates_strides}, {out_strides}) {{
{build_idx_s}
{build_updates_s}
{build_out_s}
{build_idx_sh}
{build_out_sh}
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  chelis_device_metadata coord[{rank}];
  for (chelis_device_metadata i = 0; i < total; i++) {{
    chelis_flat_to_indices(i, idx_sh, ndim, coord);
    chelis_device_metadata index_offset = chelis_indices_to_flat(coord, idx_s, ndim);
    chelis_device_metadata update_offset = chelis_indices_to_flat(coord, updates_s, ndim);
    chelis_device_metadata g = (chelis_device_metadata)indices[index_offset];
    if (g < 0 || g >= axis_size) {{
      CHELIS_GUARD_INDEX(g, axis_size, 4);
      return;
    }}
    coord[axis] = g;
    chelis_device_metadata dst = chelis_indices_to_flat(coord, out_s, ndim);
    out[dst] = updates[update_offset];
  }}
}}
",
        idx_strides = stride_params(rank, "idx"),
        updates_strides = stride_params(rank, "updates"),
        out_strides = stride_params(rank, "out"),
        build_idx_s = build_array(rank, "idx_s", "idx", "s"),
        build_updates_s = build_array(rank, "updates_s", "updates", "s"),
        build_out_s = build_array(rank, "out_s", "out", "s"),
        idx_shape = shape_params(rank, "idx"),
        out_shape = shape_params(rank, "out"),
        build_idx_sh = build_array(rank, "idx_sh", "idx", "sh"),
        build_out_sh = build_array(rank, "out_sh", "out", "sh"),
    )
}

/// Materialize input logical order without converting any dtype's stored bits.
pub fn reshape_copy(rank: usize, kernel_name: &str, width: usize) -> String {
    format!(
        "{DEVICE_HELPERS}\
extern \"C\" __global__ void {kernel_name}(
    const unsigned char *a, {a_strides}, {a_shape}, chelis_device_metadata a_ndim,
    unsigned char *out, chelis_device_metadata size) {{
{build_a_s}
{build_a_sh}
  chelis_device_metadata i = (chelis_device_metadata)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  chelis_device_metadata coordinates[{rank}];
  chelis_flat_to_indices(i, a_sh, a_ndim, coordinates);
  chelis_device_metadata source = chelis_indices_to_flat(coordinates, a_s, a_ndim);
  for (chelis_device_metadata byte = 0; byte < {width}; ++byte) {{
    out[i * {width} + byte] = a[source * {width} + byte];
  }}
}}
",
        a_strides = stride_params(rank, "a"),
        a_shape = shape_params(rank, "a"),
        build_a_s = build_array(rank, "a_s", "a", "s"),
        build_a_sh = build_array(rank, "a_sh", "a", "sh"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    use chelis_ir::ownership::{lower_dag_ownership, verify_ownership};
    use chelis_types::types::Prim;

    // chelis#2408: the HIP device helpers are a port of the C backend's
    // [05-RNG-1] word and [05-OP-8] samplers, which the CLI spec oracle holds
    // to the spec bit for bit. Respelled with C's declaration words, every
    // device helper must appear verbatim in the standalone C kernel prelude,
    // so a changed rotation, multiplier or operation order fails here.
    #[test]
    fn random_device_helpers_are_the_c_backend_port() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let template = dag.add_node(
            decl,
            RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let rank0 = |precision| TensorType {
            dims: vec![],
            precision,
        };
        let low = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 0.0),
            vec![],
            rank0(Prim::F32),
            None,
        );
        let high = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            rank0(Prim::F32),
            None,
        );
        let seed = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 7.0),
            vec![],
            rank0(Prim::Int64),
            None,
        );
        let key = dag.add_node(
            decl,
            RiscOp::KeyFromSeed,
            vec![seed],
            rank0(Prim::Key),
            None,
        );
        let draw = dag.add_node(
            decl,
            RiscOp::UniformLike,
            vec![template, low, high, key],
            ty,
            None,
        );
        dag.add_root(draw);
        let options = chelis_backend_c::CodegenOptions::default();
        let prepared = chelis_backend_c::prepare_dag_for_codegen(dag, options);
        let verified = verify_ownership(lower_dag_ownership(prepared).expect("ownership lowers"))
            .expect("uniform DAG verifies");
        let c = chelis_backend_c::codegen_with_options(verified, "draw", options)
            .expect("uniform DAG emits C")
            .c_source;
        let prelude = c
            .split("/* CHELIS_UNIFORM_HELPERS_BEGIN */\n")
            .nth(1)
            .and_then(|rest| rest.split("/* CHELIS_UNIFORM_HELPERS_END */").next())
            .expect("the C kernel carries the Random prelude");
        let respelled = NUMERIC_DEVICE_HELPERS
            .replace("__device__ ", "static inline ")
            .replace("unsigned long long", "uint64_t");
        let helpers = respelled
            .split_inclusive("\n}\n")
            .map(str::trim_end)
            .collect::<Vec<_>>();
        assert_eq!(helpers.len(), 4, "device helpers:\n{respelled}");
        for helper in helpers {
            assert!(
                prelude.contains(helper),
                "HIP helper differs from the C port:\n{helper}\nC prelude:\n{prelude}"
            );
        }
    }

    #[test]
    fn reduced_float_conversions_emit_round_to_nearest_ties_to_even() {
        assert!(
            REDUCED_FLOAT_COMPARISON_HELPERS.contains(concat!(
                "if ((bits & 0x7fffffffu) > 0x7f800000u) {",
                "\n        return (chelis_u16)0x7fc0u;"
            )),
            "bf16 arithmetic NaNs must finalize to the canonical positive quiet NaN"
        );
        assert!(
            REDUCED_FLOAT_COMPARISON_HELPERS.contains(concat!(
                "if (exponent == 0x7f800000u) {",
                "\n        return (chelis_u16)",
                "(mantissa == 0 ? ((sign >> 16) | 0x7c00u) : 0x7e00u);"
            )),
            "f16 arithmetic NaNs must finalize to the canonical positive quiet NaN while infinities keep their sign"
        );
        assert!(
            REDUCED_FLOAT_COMPARISON_HELPERS.contains("0x00007fffu + ((bits >> 16) & 1u)"),
            "bf16 rounding must bias exact ties by the retained LSB"
        );
        assert!(
            REDUCED_FLOAT_COMPARISON_HELPERS
                .contains("chelis_round_shift_right_even(mantissa, shift)"),
            "f16 normal and subnormal rounding must share the exact RNE shift helper"
        );
        assert!(
            REDUCED_FLOAT_COMPARISON_HELPERS
                .contains("remainder > halfway || (remainder == halfway && (truncated & 1u) != 0)"),
            "exact halfway cases must increment only when the retained LSB is odd"
        );
        assert!(
            !REDUCED_FLOAT_COMPARISON_HELPERS.contains("3u * round_bit - 1u"),
            "the ties-away mask must not return"
        );
    }

    #[test]
    fn reduced_float_halfway_vectors_distinguish_even_and_odd_retained_lsbs() {
        for (bits, expected) in [(0x3f80_8000, 0x3f80), (0x3f81_8000, 0x3f82)] {
            assert_eq!(
                half::bf16::from_f32(f32::from_bits(bits)).to_bits(),
                expected
            );
        }

        for (bits, expected) in [
            (0x3f80_1000, 0x3c00),
            (0x3f80_3000, 0x3c02),
            (0x3300_0000, 0x0000),
            (0x33c0_0000, 0x0002),
            (0x387f_e000, 0x0400),
        ] {
            assert_eq!(
                half::f16::from_f32(f32::from_bits(bits)).to_bits(),
                expected
            );
        }
    }

    #[test]
    fn binary_add_kernel_has_correct_structure() {
        let src = binary_elementwise(8, "kernel_add", "+", ElemKind::F32);
        assert!(src.contains("extern \"C\" __global__ void kernel_add("));
        assert!(src.contains("out[i] = a[idx_a] + b[idx_b];"));
        assert!(src.contains("chelis_flat_to_indices"));
        assert!(src.contains("chelis_indices_to_flat"));
    }

    #[test]
    fn binary_add_uses_int_params_not_pointers() {
        let src = binary_elementwise(8, "kernel_add", "+", ElemKind::F32);
        // Must have individual int params, not const int* pointers
        assert!(
            src.contains("chelis_device_metadata a_s0"),
            "stride params must be individual ints"
        );
        assert!(
            src.contains("chelis_device_metadata a_s7"),
            "must have all MAX_DIM stride params"
        );
        assert!(
            src.contains("chelis_device_metadata out_sh0"),
            "shape params must be individual ints"
        );
        // Must build local arrays from params
        assert!(
            src.contains("chelis_device_metadata a_s[] ="),
            "must build local stride array"
        );
        assert!(
            src.contains("chelis_device_metadata out_sh[] ="),
            "must build local shape array"
        );
        // Must NOT have device pointer params for strides/shapes
        assert!(
            !src.contains("const chelis_device_metadata *a_strides"),
            "must NOT use device pointer for strides"
        );
    }

    #[test]
    fn binary_add_f64_uses_double_buffers() {
        let src = binary_elementwise(8, "kernel_add_f64", "+", ElemKind::F64);
        assert!(src.contains("const double *a"));
        assert!(src.contains("const double *b"));
        assert!(src.contains("double *out"));
        assert!(!src.contains("const float *a"));
    }

    #[test]
    fn unary_exp_kernel_correct() {
        let src = unary_func(8, "kernel_exp", "expf", ElemKind::F32);
        assert!(src.contains("extern \"C\" __global__ void kernel_exp("));
        assert!(src.contains("out[i] = expf(a[idx]);"));
    }

    #[test]
    fn unary_exp_f64_uses_double_libm() {
        let src = unary_func(8, "kernel_exp_f64", "expf", ElemKind::F64);
        assert!(src.contains("out[i] = exp(a[idx]);"));
        assert!(!src.contains("expf("));
        assert!(src.contains("const double *a"));
        assert!(src.contains("double *out"));
    }

    #[test]
    fn comparison_kernels_return_canonical_bool8() {
        let src = comparison(8, "kernel_cmplt", "<", "float", "", "a[idx_a]", "b[idx_b]");
        assert!(src.contains("const float *a"));
        assert!(src.contains("unsigned char *out"));
        assert!(src.contains("out[i] = (unsigned char)("));
    }

    #[test]
    fn reduce_sum_kernel_has_axis_loop() {
        let src = reduce_sum(8, "kernel_sum_ax0_f32", 0, ElemKind::F32, ElemKind::F32);
        assert!(src.contains("float acc = 0.0f;"));
        assert!(src.contains("for (chelis_device_metadata k = 0; k < axis_size; k++)"));
        assert!(src.contains("if (d == 0)"));
    }

    #[test]
    fn reduce_sum_f64_uses_double_accumulator() {
        let src = reduce_sum(8, "kernel_sum_ax0_f64", 0, ElemKind::F64, ElemKind::F64);
        assert!(src.contains("double acc = 0.0;"));
        assert!(src.contains("const double *a"));
        assert!(src.contains("double *out"));
    }

    #[test]
    fn reduce_max_kernel_has_f32_min_sentinel() {
        let src = reduce_max(8, "kernel_max_ax1", 1, ElemKind::F32);
        assert!(src.contains("-3.402823466e+38F"));
        assert!(src.contains("fmaxf(acc,"));
    }

    #[test]
    fn reduce_max_f64_uses_double_min_sentinel_and_fmax() {
        let src = reduce_max(8, "kernel_max_ax1_f64", 1, ElemKind::F64);
        assert!(src.contains("-1.7976931348623157e+308"));
        assert!(src.contains("fmax(acc,"));
        assert!(!src.contains("fmaxf(acc,"));
    }

    #[test]
    fn reduce_min_seeds_with_max_finite() {
        let src = reduce_min(8, "kernel_min_ax0_f32", 0, ElemKind::F32);
        assert!(src.contains("3.402823466e+38F"));
        assert!(src.contains("fminf(acc,"));
    }

    #[test]
    fn reduce_prod_uses_one_identity() {
        let src = reduce_prod(8, "kernel_prod_ax0_f64", 0, ElemKind::F64);
        assert!(src.contains("double acc = 1.0;"));
        assert!(src.contains("acc *= a[src_idx];"));
    }

    #[test]
    fn reduce_argmax_emits_long_long_output() {
        let src = reduce_argmax(8, "kernel_argmax_ax0_f32", 0, ElemKind::F32);
        assert!(src.contains("long long *out"));
        assert!(src.contains("long long best_idx"));
        assert!(src.contains("if (v > best)"));
    }

    #[test]
    fn reduce_argmin_emits_long_long_output() {
        let src = reduce_argmin(8, "kernel_argmin_ax0_f32", 0, ElemKind::F32);
        assert!(src.contains("long long *out"));
        assert!(src.contains("if (v < best)"));
    }

    #[test]
    fn fill_kernel_simple() {
        let src = fill(8, "kernel_fill", ElemKind::F32);
        assert!(src.contains("data[i] = value;"));
        // Fill has no stride/shape params (operates on raw buffer)
        assert!(!src.contains("a_s0"));
    }

    #[test]
    fn fill_f64_uses_double_value() {
        let src = fill(8, "kernel_fill_f64", ElemKind::F64);
        assert!(src.contains("double *data, double value"));
    }

    #[test]
    fn cast_convert_widens_f32_to_f64() {
        let src = cast_convert(
            8,
            "kernel_cast_f32_to_f64",
            ElemKind::F32,
            ElemKind::F64,
            None,
        );
        assert!(src.contains("const float *a"));
        assert!(src.contains("double *out"));
        assert!(src.contains("out[i] = (double)a[idx];"));
    }

    #[test]
    fn device_helpers_present_in_all_compute_kernels() {
        let add = binary_elementwise(8, "k", "+", ElemKind::F32);
        let neg = unary_prefix(8, "k", "-", ElemKind::F32);
        let sum = reduce_sum(8, "k", 0, ElemKind::F32, ElemKind::F32);
        let fill_src = fill(8, "k", ElemKind::F32);
        for src in [&add, &neg, &sum, &fill_src] {
            assert!(
                !src.contains("#include <stdint.h>"),
                "HIPRTC supplies device integer typedefs; host stdint conflicts with hiprtc_runtime.h"
            );
            assert!(src.contains("__device__ void chelis_flat_to_indices"));
            assert!(src.contains("__device__ chelis_device_metadata chelis_indices_to_flat"));
            assert!(src.contains("__device__ int chelis_gpu_failure = 0;"));
            assert!(src.contains("CHELIS_GUARD_INDEX"));
        }
    }
}
