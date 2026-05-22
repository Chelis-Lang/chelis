//! WS-M1: per-dtype helpers for the Metal backend.
//!
//! Mirrors `chelis_backend_hip::memory::bytes_per_element` and the
//! `dtype_c_type` / `dtype_kernel_suffix` shape from
//! `chelis_backend_hip::emit`. Centralizes per-dtype mappings so the
//! kernel templates, the host emitter, and the runtime header agree on
//! a single source of truth for MSL spelling, kernel-name suffixing,
//! per-element width, and reduction-accumulator promotion (per spec
//! `spec/04-type-system.md` §1.1.3 + §5.7.1).
//!
//! Per `spec/04-type-system.md` §1.1.3 the active Metal dtype set is
//! `{f32, f16, bf16, int8, int16, int32, int64, bool}` (everything
//! except `f64`, which Apple Silicon GPUs lack ALU support for).

use chelis_types::types::Prim;

/// MSL spelling of the type for kernel parameters and bodies.
///
/// `f64` deliberately panics here as defense in depth; the IR validation
/// guard and the CLI gate (`reject_unsupported_metal_ops`) reject `f64`
/// upstream with the FP64-ALU diagnostic so this arm should be
/// unreachable from any well-formed build.
pub fn msl_type(prec: Prim) -> &'static str {
    match prec {
        Prim::F32 => "float",
        Prim::F16 => "half",
        Prim::Bf16 => "bfloat",
        Prim::Int8 => "char",
        Prim::Int16 => "short",
        Prim::Int32 => "int",
        Prim::Int64 => "long",
        Prim::Bool => "bool",
        Prim::F64 => panic!(
            "Metal backend rejects f64 (Apple Silicon GPUs lack FP64 ALUs); \
             reject_unsupported_metal_ops should have caught the rest"
        ),
        other => panic!(
            "Metal backend dtype not in the active per-backend matrix: {} \
             (see spec/04-type-system.md §1.1.3)",
            other.name()
        ),
    }
}

/// Per-element byte size for an active Metal dtype.
///
/// Mirrors `chelis_backend_hip::memory::bytes_per_element` and the
/// runtime allocator's `chelis_alloc` element width. Used in
/// `peak_device_bytes` and any host-side allocation arithmetic so no
/// caller hardcodes `sizeof(float)` (the RT-4-Fixups F2 lesson).
pub fn metal_elem_size(prec: Prim) -> usize {
    match prec {
        Prim::Int8 | Prim::Bool => 1,
        Prim::Int16 | Prim::F16 | Prim::Bf16 => 2,
        Prim::F32 | Prim::Int32 => 4,
        Prim::Int64 => 8,
        Prim::F64 => panic!(
            "Metal backend rejects f64 (Apple Silicon GPUs lack FP64 ALUs); \
             reject_unsupported_metal_ops should have caught the rest"
        ),
        other => panic!(
            "Metal backend dtype not in the active per-backend matrix: {} \
             (see spec/04-type-system.md §1.1.3)",
            other.name()
        ),
    }
}

/// Suffix used in generated MSL kernel function names so per-dtype
/// instances don't collide. `f32` keeps the bare name (legacy) so
/// existing kernel-name string match tests still resolve.
pub fn kernel_suffix(prec: Prim) -> &'static str {
    match prec {
        Prim::F32 => "",
        Prim::F16 => "_f16",
        Prim::Bf16 => "_bf16",
        Prim::Int8 => "_i8",
        Prim::Int16 => "_i16",
        Prim::Int32 => "_i32",
        Prim::Int64 => "_i64",
        Prim::Bool => "_bool",
        Prim::F64 => panic!(
            "Metal backend rejects f64 (Apple Silicon GPUs lack FP64 ALUs); \
             reject_unsupported_metal_ops should have caught the rest"
        ),
        other => panic!(
            "Metal backend dtype not in the active per-backend matrix: {} \
             (see spec/04-type-system.md §1.1.3)",
            other.name()
        ),
    }
}

/// Host-safe `sizeof(...)` expression for emitted Objective-C++ host code.
///
/// MSL types `half` and `bfloat` are only defined inside MSL kernel
/// translation units; they do not exist in plain C++ host code. Emitting
/// `sizeof(half)` or `sizeof(bfloat)` host-side fails to compile with
/// `clang++ -fobjc-arc` (the MPS f16 wrapper already correctly uses
/// `sizeof(uint16_t)` for exactly this reason). Every host-side byte-width
/// computation in the emitter must route through this helper instead of
/// `sizeof({msl_ty})` so the emitted .mm links cleanly against any Apple
/// toolchain.
///
/// `f64` deliberately panics here as defense in depth; the IR validation
/// guard, the CLI gate, and the codegen entry all reject `f64` upstream.
pub fn host_sizeof_expr(prec: Prim) -> &'static str {
    match prec {
        Prim::F32 => "sizeof(float)",
        Prim::F16 => "sizeof(uint16_t)",
        Prim::Bf16 => "sizeof(uint16_t)",
        Prim::Int8 => "sizeof(int8_t)",
        Prim::Int16 => "sizeof(int16_t)",
        Prim::Int32 => "sizeof(int32_t)",
        Prim::Int64 => "sizeof(int64_t)",
        Prim::Bool => "sizeof(bool)",
        Prim::F64 => panic!(
            "Metal backend rejects f64 (Apple Silicon GPUs lack FP64 ALUs); \
             reject_unsupported_metal_ops should have caught the rest"
        ),
        other => panic!(
            "Metal backend dtype not in the active per-backend matrix: {} \
             (see spec/04-type-system.md §1.1.3)",
            other.name()
        ),
    }
}

/// Shared `chelis_runtime` dtype enum tag for an active Metal dtype.
/// Used at the host writeback site to match the runtime's `chelis_alloc`
/// dispatch. Tag spellings match `crates/chelis-runtime/include/chelis_runtime.h`
/// (`CHELIS_I8`, `CHELIS_I16`, `CHELIS_I32`, `CHELIS_I64` — not `CHELIS_INT*`).
pub fn runtime_dtype_tag(prec: Prim) -> &'static str {
    match prec {
        Prim::F32 => "CHELIS_F32",
        Prim::F16 => "CHELIS_F16",
        Prim::Bf16 => "CHELIS_BF16",
        Prim::Int8 => "CHELIS_I8",
        Prim::Int16 => "CHELIS_I16",
        Prim::Int32 => "CHELIS_I32",
        Prim::Int64 => "CHELIS_I64",
        Prim::Bool => "CHELIS_BOOL",
        Prim::F64 => panic!(
            "Metal backend rejects f64 (Apple Silicon GPUs lack FP64 ALUs); \
             reject_unsupported_metal_ops should have caught the rest"
        ),
        other => panic!(
            "Metal backend dtype not in the active per-backend matrix: {} \
             (see spec/04-type-system.md §1.1.3)",
            other.name()
        ),
    }
}

/// Reduction-accumulator promotion per spec §5.7.1.
///
/// `reduce_sum` widens narrow floats to f32, narrow ints to i32, leaves
/// i64/f32 in place, and (per the spec table) keeps bool accumulators as
/// requested. `max`/`min` reductions never promote, so this helper is
/// only invoked on the sum path; the caller is responsible for routing
/// non-sum reductions through the operand precision directly.
pub fn sum_accumulator(prec: Prim) -> Prim {
    match prec {
        Prim::F16 | Prim::Bf16 => Prim::F32,
        Prim::Int8 | Prim::Int16 => Prim::Int32,
        other => other,
    }
}

/// Whether emitting a kernel for this dtype requires the
/// `#if __METAL_VERSION__ >= 320 ... #endif` guard. Today only `bf16`
/// needs the guard (Apple7+ / MSL 3.2+ requirement per
/// `spec/04-type-system.md` §1.1.3).
pub fn requires_msl_320_guard(prec: Prim) -> bool {
    matches!(prec, Prim::Bf16)
}

/// Host-side Const-fill block for an active Metal dtype.
///
/// Returns the inner Objective-C++ block (including the outer `{ ... }`) that
/// fills `n` elements at host-side buffer `buf` with `value`. Routes through
/// host-safe types for every active Metal dtype so the emitted `.mm` compiles
/// under `clang++ -fobjc-arc` (which is not a Metal kernel translation unit
/// and therefore has no `half` or `bfloat` type in scope).
///
/// For `F16` and `Bf16` the f64 `value` is converted to the IEEE-754 bit
/// pattern at codegen time (`half::f16::from_f64(...).to_bits()` /
/// `half::bf16::from_f64(...).to_bits()`) and written through a `uint16_t*`,
/// matching the MPS f16 wrapper convention already established in the Metal
/// runtime. Integer dtypes use the matching `intN_t` host type with an
/// explicit integral cast; `Bool` uses C++ `bool`; `F32` keeps the typed
/// `float*` form for byte-identical compatibility with the pre-WS-2 emission.
///
/// `F64` panics here as defense in depth; the IR validation pass and the
/// `reject_unsupported_metal_ops` CLI gate reject `f64` upstream with the
/// FP64-ALU diagnostic so this arm should be unreachable from any well-formed
/// build.
pub fn host_const_fill_body(prec: Prim, value: f64, buf: &str, n: usize) -> String {
    match prec {
        Prim::F32 => format!(
            "{{ float *p = (float*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = {value:?}f; }}",
        ),
        Prim::F16 => {
            let bits = half::f16::from_f64(value).to_bits();
            format!(
                "{{ uint16_t *p = (uint16_t*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = 0x{bits:04X}u; }}",
            )
        }
        Prim::Bf16 => {
            let bits = half::bf16::from_f64(value).to_bits();
            format!(
                "{{ uint16_t *p = (uint16_t*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = 0x{bits:04X}u; }}",
            )
        }
        Prim::Int8 => {
            let v = value as i64;
            format!(
                "{{ int8_t *p = (int8_t*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = (int8_t){v}; }}",
            )
        }
        Prim::Int16 => {
            let v = value as i64;
            format!(
                "{{ int16_t *p = (int16_t*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = (int16_t){v}; }}",
            )
        }
        Prim::Int32 => {
            let v = value as i64;
            format!(
                "{{ int32_t *p = (int32_t*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = (int32_t){v}; }}",
            )
        }
        Prim::Int64 => {
            let v = value as i64;
            format!(
                "{{ int64_t *p = (int64_t*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = (int64_t){v}LL; }}",
            )
        }
        Prim::Bool => {
            let v = value != 0.0;
            format!(
                "{{ bool *p = (bool*)[{buf} contents]; for (uint i = 0; i < {n}u; ++i) p[i] = {v}; }}",
            )
        }
        Prim::F64 => panic!(
            "Metal backend rejects f64 (Apple Silicon GPUs lack FP64 ALUs); \
             reject_unsupported_metal_ops should have caught the rest"
        ),
        other => panic!(
            "Metal backend dtype not in the active per-backend matrix: {} \
             (see spec/04-type-system.md §1.1.3)",
            other.name()
        ),
    }
}

/// Matmul-accumulator promotion per spec/04-type-system.md §5.7.1.
///
/// Mirrors [`sum_accumulator`] but explicitly named for the matmul
/// path so call sites read intent: `bf16/f16` operands accumulate in
/// `f32`, narrow ints (`int8/int16`) accumulate in `int32`, and other
/// dtypes accumulate in their own precision. The kernel template
/// downcasts the f32 accumulator back to operand precision at write-out
/// time and casts both tile operands to f32 at the multiply (the spec
/// requires the partial product to compute in accumulator precision,
/// not just the running sum).
pub fn matmul_accumulator(operand_prec: Prim) -> Prim {
    match operand_prec {
        Prim::F16 | Prim::Bf16 => Prim::F32,
        Prim::Int8 | Prim::Int16 => Prim::Int32,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_type_covers_active_metal_dtypes() {
        assert_eq!(msl_type(Prim::F32), "float");
        assert_eq!(msl_type(Prim::F16), "half");
        assert_eq!(msl_type(Prim::Bf16), "bfloat");
        assert_eq!(msl_type(Prim::Int8), "char");
        assert_eq!(msl_type(Prim::Int16), "short");
        assert_eq!(msl_type(Prim::Int32), "int");
        assert_eq!(msl_type(Prim::Int64), "long");
        assert_eq!(msl_type(Prim::Bool), "bool");
    }

    #[test]
    #[should_panic(expected = "Metal backend rejects f64")]
    fn msl_type_panics_on_f64() {
        let _ = msl_type(Prim::F64);
    }

    /// F6: pin the panic message format on the three other dtype
    /// helpers that defensively reject f64 (the spec §1.1.3 hardware
    /// constraint diagnostic must mention "f64" so a regression that
    /// changes the wording surfaces here, not at a downstream user).
    #[test]
    #[should_panic(expected = "Metal backend rejects f64")]
    fn metal_elem_size_panics_on_f64() {
        let _ = metal_elem_size(Prim::F64);
    }

    #[test]
    #[should_panic(expected = "Metal backend rejects f64")]
    fn runtime_dtype_tag_panics_on_f64() {
        let _ = runtime_dtype_tag(Prim::F64);
    }

    #[test]
    #[should_panic(expected = "Metal backend rejects f64")]
    fn kernel_suffix_panics_on_f64() {
        let _ = kernel_suffix(Prim::F64);
    }

    #[test]
    fn elem_sizes_match_runtime_widths() {
        assert_eq!(metal_elem_size(Prim::F32), 4);
        assert_eq!(metal_elem_size(Prim::F16), 2);
        assert_eq!(metal_elem_size(Prim::Bf16), 2);
        assert_eq!(metal_elem_size(Prim::Int8), 1);
        assert_eq!(metal_elem_size(Prim::Int16), 2);
        assert_eq!(metal_elem_size(Prim::Int32), 4);
        assert_eq!(metal_elem_size(Prim::Int64), 8);
        assert_eq!(metal_elem_size(Prim::Bool), 1);
    }

    #[test]
    fn sum_accumulator_promotes_per_spec() {
        assert_eq!(sum_accumulator(Prim::F16), Prim::F32);
        assert_eq!(sum_accumulator(Prim::Bf16), Prim::F32);
        assert_eq!(sum_accumulator(Prim::Int8), Prim::Int32);
        assert_eq!(sum_accumulator(Prim::Int16), Prim::Int32);
        assert_eq!(sum_accumulator(Prim::Int32), Prim::Int32);
        assert_eq!(sum_accumulator(Prim::Int64), Prim::Int64);
        assert_eq!(sum_accumulator(Prim::F32), Prim::F32);
    }

    #[test]
    fn matmul_accumulator_promotes_per_spec() {
        assert_eq!(matmul_accumulator(Prim::F16), Prim::F32);
        assert_eq!(matmul_accumulator(Prim::Bf16), Prim::F32);
        assert_eq!(matmul_accumulator(Prim::F32), Prim::F32);
        assert_eq!(matmul_accumulator(Prim::Int8), Prim::Int32);
        assert_eq!(matmul_accumulator(Prim::Int16), Prim::Int32);
        assert_eq!(matmul_accumulator(Prim::Int32), Prim::Int32);
        assert_eq!(matmul_accumulator(Prim::Int64), Prim::Int64);
    }

    #[test]
    fn host_sizeof_returns_host_safe_types_for_active_matrix() {
        assert_eq!(host_sizeof_expr(Prim::F32), "sizeof(float)");
        assert_eq!(host_sizeof_expr(Prim::F16), "sizeof(uint16_t)");
        assert_eq!(host_sizeof_expr(Prim::Bf16), "sizeof(uint16_t)");
        assert_eq!(host_sizeof_expr(Prim::Int8), "sizeof(int8_t)");
        assert_eq!(host_sizeof_expr(Prim::Int16), "sizeof(int16_t)");
        assert_eq!(host_sizeof_expr(Prim::Int32), "sizeof(int32_t)");
        assert_eq!(host_sizeof_expr(Prim::Int64), "sizeof(int64_t)");
        assert_eq!(host_sizeof_expr(Prim::Bool), "sizeof(bool)");
    }

    #[test]
    #[should_panic(expected = "Metal backend rejects f64")]
    fn host_sizeof_panics_on_f64() {
        let _ = host_sizeof_expr(Prim::F64);
    }

    #[test]
    fn bf16_requires_msl_320_guard() {
        assert!(requires_msl_320_guard(Prim::Bf16));
        assert!(!requires_msl_320_guard(Prim::F32));
        assert!(!requires_msl_320_guard(Prim::F16));
    }

    #[test]
    fn host_const_fill_body_uses_host_safe_types_for_active_matrix() {
        // F32 keeps the typed-pointer form so the byte-for-byte output is
        // identical to the pre-WS-2 emission for the default case.
        let f32_body = host_const_fill_body(Prim::F32, 2.5, "buf_0", 8);
        assert!(
            f32_body.contains("float *p = (float*)[buf_0 contents]"),
            "F32 body must use typed `float*`: {f32_body}"
        );
        assert!(
            f32_body.contains("p[i] = 2.5f"),
            "F32 body must use `f`-suffixed literal: {f32_body}"
        );

        // F16/Bf16 must route through `uint16_t*` with the bit-pattern
        // literal so the `.mm` compiles under `clang++ -fobjc-arc` (which
        // has no `half`/`bfloat` host type in scope).
        let f16_body = host_const_fill_body(Prim::F16, 2.5, "buf_1", 4);
        assert!(
            f16_body.contains("uint16_t *p = (uint16_t*)[buf_1 contents]"),
            "F16 body must use `uint16_t*`: {f16_body}"
        );
        assert!(
            f16_body.contains("0x4100u"),
            "F16(2.5) bit pattern is 0x4100: {f16_body}"
        );
        assert!(
            !f16_body.contains("half"),
            "F16 body must not mention `half` (kernel-only type): {f16_body}"
        );

        let bf16_body = host_const_fill_body(Prim::Bf16, 2.5, "buf_2", 4);
        assert!(
            bf16_body.contains("uint16_t *p = (uint16_t*)[buf_2 contents]"),
            "Bf16 body must use `uint16_t*`: {bf16_body}"
        );
        assert!(
            bf16_body.contains("0x4020u"),
            "Bf16(2.5) bit pattern is 0x4020: {bf16_body}"
        );
        assert!(
            !bf16_body.contains("bfloat"),
            "Bf16 body must not mention `bfloat` (kernel-only type): {bf16_body}"
        );

        // Integer dtypes carry the matching `intN_t` cast.
        let i8_body = host_const_fill_body(Prim::Int8, 5.0, "buf_3", 4);
        assert!(
            i8_body.contains("int8_t *p = (int8_t*)[buf_3 contents]"),
            "Int8 body shape: {i8_body}"
        );
        assert!(i8_body.contains("(int8_t)5"), "Int8 cast: {i8_body}");

        let i64_body = host_const_fill_body(Prim::Int64, 7.0, "buf_4", 4);
        assert!(
            i64_body.contains("int64_t *p = (int64_t*)[buf_4 contents]"),
            "Int64 body shape: {i64_body}"
        );
        assert!(i64_body.contains("(int64_t)7LL"), "Int64 cast: {i64_body}");

        // Bool uses the C++ literal `true`/`false`, not 0/1.
        let bool_true = host_const_fill_body(Prim::Bool, 1.0, "buf_5", 2);
        assert!(
            bool_true.contains("bool *p = (bool*)[buf_5 contents]"),
            "Bool body shape: {bool_true}"
        );
        assert!(bool_true.contains("p[i] = true"), "Bool true: {bool_true}");
        let bool_false = host_const_fill_body(Prim::Bool, 0.0, "buf_6", 2);
        assert!(
            bool_false.contains("p[i] = false"),
            "Bool false: {bool_false}"
        );
    }

    /// Pinned bit-pattern sweep per WS-2 plan. Catches a regression in
    /// `half::{f16,bf16}::from_f64` round-tripping or a typo in the helper's
    /// format string.
    #[test]
    fn host_const_fill_body_pinned_bit_patterns() {
        let cases = [
            (Prim::F16, 2.5_f64, 0x4100_u16),
            (Prim::Bf16, 2.5, 0x4020),
            (Prim::F16, 1.5, 0x3E00),
            (Prim::Bf16, 1.5, 0x3FC0),
            (Prim::F16, -1.0, 0xBC00),
            (Prim::Bf16, -1.0, 0xBF80),
            (Prim::F16, 0.0, 0x0000),
            (Prim::Bf16, 0.0, 0x0000),
        ];
        for (prec, value, expected_bits) in cases {
            let body = host_const_fill_body(prec, value, "buf", 4);
            let needle = format!("0x{expected_bits:04X}u");
            assert!(
                body.contains(&needle),
                "{prec:?}({value}) must emit `{needle}`; got: {body}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "Metal backend rejects f64")]
    fn host_const_fill_body_panics_on_f64() {
        let _ = host_const_fill_body(Prim::F64, 1.0, "buf", 1);
    }

    #[test]
    fn kernel_suffix_distinct_per_dtype() {
        let suffixes: Vec<&str> = [
            Prim::F32,
            Prim::F16,
            Prim::Bf16,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ]
        .iter()
        .copied()
        .map(kernel_suffix)
        .collect();
        let mut deduped = suffixes.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            suffixes.len(),
            deduped.len(),
            "kernel suffixes must be unique"
        );
    }
}
