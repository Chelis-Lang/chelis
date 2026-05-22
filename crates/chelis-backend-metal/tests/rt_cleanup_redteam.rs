//! RT-Cleanup adversarial tests for WS-2 (Metal emit_const f16/bf16
//! host-fill fix). Runs on Linux without a Metal toolchain since the
//! attack surface is the string shape of the emitted Objective-C++
//! source. The macOS compile-and-run gate lives in
//! `tests/gpu_correctness.rs::metal_const_f16_bf16_compiles_under_objc_arc_and_produces_exact_bits`.
//!
//! These tests extend the WS-2 sweep to:
//!   * a non-edge f16 value (0.123); sanity-check the bit-pattern
//!     computation path, not just round-numbered powers of two
//!   * a bf16 program where the host code MUST NOT contain `bfloat`
//!     (kernel-only) anywhere in the Const-fill block
//!   * a precision-pair sweep confirming `host_const_fill_body` and
//!     `host_sizeof_expr` agree on host-safe typing for every active
//!     Metal dtype

use chelis_backend_metal::codegen_metal;
use chelis_backend_metal::dtype;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn vec_prec(n: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: prec,
    }
}

fn build_const_root_dag(prec: Prim, value: f64) -> Dag {
    let mut dag = Dag::new();
    let c = dag.add_node(RiscOp::Const { value }, vec![], vec_prec(4, prec), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![c],
        vec_prec(4, prec),
        None,
    );
    dag.add_root(stored);
    dag
}

/// Non-edge f16 value: 0.123 is not a power of two and has a non-trivial
/// IEEE 754 binary16 encoding. The emitted host code must contain the
/// exact bit pattern produced by `half::f16::from_f64(0.123).to_bits()`
/// (which is 0x2FDF). Catches a regression that special-cased only
/// edge values (zero, +/-1, +/-inf) in the bit-pattern path.
#[test]
fn rt_metal_emit_const_f16_non_edge_value_uses_correct_bit_pattern() {
    let value = 0.123_f64;
    let expected_bits = half::f16::from_f64(value).to_bits();
    let dag = build_const_root_dag(Prim::F16, value);
    let result = codegen_metal(&dag, "const_f16_0_123");
    let src = &result.mm_source;
    let needle = format!("0x{expected_bits:04X}u");
    assert!(
        src.contains(&needle),
        "Metal f16({value}) emit must contain bit literal `{needle}` (computed via \
         `half::f16::from_f64(...).to_bits()`); got source:\n{src}"
    );
    assert!(
        src.contains("(uint16_t*)"),
        "host-side fill must cast to `(uint16_t*)`:\n{src}"
    );
    assert!(
        !src.contains("(half)"),
        "host-side fill must not use kernel-only `(half)` value cast:\n{src}"
    );
}

#[test]
fn rt_metal_emit_const_bf16_non_edge_value_uses_correct_bit_pattern() {
    let value = 0.123_f64;
    let expected_bits = half::bf16::from_f64(value).to_bits();
    let dag = build_const_root_dag(Prim::Bf16, value);
    let result = codegen_metal(&dag, "const_bf16_0_123");
    let src = &result.mm_source;
    let needle = format!("0x{expected_bits:04X}u");
    assert!(
        src.contains(&needle),
        "Metal bf16({value}) emit must contain bit literal `{needle}`; got source:\n{src}"
    );
    assert!(
        src.contains("(uint16_t*)"),
        "host-side fill must cast to `(uint16_t*)`:\n{src}"
    );
    assert!(
        !src.contains("(bfloat)"),
        "host-side fill must not use kernel-only `(bfloat)` value cast:\n{src}"
    );
}

/// The bf16 Const-fill block in host code must NEVER mention the
/// MSL-only `bfloat` type. Some other parts of the emitted source MAY
/// mention `bfloat` inside MSL raw-string kernels; that is correct.
/// This test scans the host-side Const block specifically.
#[test]
fn rt_metal_emit_const_bf16_host_block_has_no_bfloat_token() {
    let value = 2.5_f64;
    let dag = build_const_root_dag(Prim::Bf16, value);
    let result = codegen_metal(&dag, "const_bf16_host_no_bfloat");
    let src = &result.mm_source;
    let const_marker = format!("= Const {value}");
    let pos = src
        .find(&const_marker)
        .unwrap_or_else(|| panic!("missing `// node N = Const {value}` marker:\n{src}"));
    let after = &src[pos..];
    let next_node = after[const_marker.len()..]
        .find("// node ")
        .map(|off| const_marker.len() + off)
        .unwrap_or(after.len());
    let fill_block = &after[..next_node];
    assert!(
        !fill_block.contains("bfloat"),
        "bf16 Const host-fill block must not mention `bfloat` (MSL kernel-only type):\n{fill_block}"
    );
    assert!(
        !fill_block.contains("(half"),
        "bf16 Const host-fill block must not mention `(half` either:\n{fill_block}"
    );
}

/// Pairwise consistency: for every active Metal dtype, the byte-width
/// `host_sizeof_expr` returns and the per-element store width in
/// `host_const_fill_body` must agree on host-safe types. Catches a
/// drift where a future precision added one helper but not the other.
#[test]
fn rt_metal_host_sizeof_and_const_fill_agree_on_host_safe_types() {
    let cases = [
        (Prim::F32, "sizeof(float)", "float *p"),
        (Prim::F16, "sizeof(uint16_t)", "uint16_t *p"),
        (Prim::Bf16, "sizeof(uint16_t)", "uint16_t *p"),
        (Prim::Int8, "sizeof(int8_t)", "int8_t *p"),
        (Prim::Int16, "sizeof(int16_t)", "int16_t *p"),
        (Prim::Int32, "sizeof(int32_t)", "int32_t *p"),
        (Prim::Int64, "sizeof(int64_t)", "int64_t *p"),
        (Prim::Bool, "sizeof(bool)", "bool *p"),
    ];
    for (prec, expect_sizeof, expect_ptr) in cases {
        let sz = dtype::host_sizeof_expr(prec);
        assert_eq!(
            sz, expect_sizeof,
            "{prec:?}: host_sizeof_expr returned `{sz}`, expected `{expect_sizeof}`"
        );
        let body = dtype::host_const_fill_body(prec, 1.0, "buf_0", 4);
        assert!(
            body.contains(expect_ptr),
            "{prec:?}: host_const_fill_body must use `{expect_ptr}` host-safe pointer cast; got: {body}"
        );
    }
}

/// Const-rooted f16 DAG with all WS-2 pinned edge values. WS-2 already
/// covers the f16/bf16 cases in `ws2_emit_const_f16_bf16_use_uint16_bit_pattern_not_msl_kernel_types`;
/// this extra sweep adds two pathological values (max-finite, smallest
/// normal) that the WS-2 sweep did not touch.
#[test]
fn rt_metal_const_f16_bf16_pathological_values_emit_correct_bit_pattern() {
    // f16 max finite = 0x7BFF (= 65504.0).
    // bf16 max finite = 0x7F7F (~ 3.39e38).
    // f16 smallest normal = 0x0400 (~ 6.10e-5).
    // bf16 smallest normal = 0x0080 (~ 1.18e-38).
    let f16_max = half::f16::from_bits(0x7BFF).to_f64();
    let bf16_max = half::bf16::from_bits(0x7F7F).to_f64();
    let f16_min = half::f16::from_bits(0x0400).to_f64();
    let bf16_min = half::bf16::from_bits(0x0080).to_f64();
    let cases = [
        (Prim::F16, f16_max, 0x7BFF_u16),
        (Prim::Bf16, bf16_max, 0x7F7F_u16),
        (Prim::F16, f16_min, 0x0400_u16),
        (Prim::Bf16, bf16_min, 0x0080_u16),
    ];
    for (prec, value, expected_bits) in cases {
        let body = dtype::host_const_fill_body(prec, value, "buf", 4);
        let needle = format!("0x{expected_bits:04X}u");
        assert!(
            body.contains(&needle),
            "{prec:?}({value}) must emit `{needle}`; got: {body}"
        );
    }
}

/// COVERAGE GAP: `host_const_fill_body(F32, NaN, ...)` emits `p[i] = NaNf;`
/// which is invalid C++ (`NaNf` is not a valid literal; the C++ literal
/// for NaN is `NAN` or `__builtin_nanf("")`). Same for `INFINITY`
/// (`inff` is not valid C++). Today this only fires if a Const node
/// with a NaN/Inf value reaches the Metal codegen (IR validation does
/// not reject those values), so this is a latent coverage gap rather
/// than a today-BLOCKER for typical programs.
///
/// Asserting the spec-correct behavior would require the formatter
/// to emit `NAN` instead of `NaNf` and `INFINITY` instead of `inff`.
/// This test documents the divergence; if/when the fix lands, drop the
/// `#[ignore]`.
#[test]
#[ignore = "SPEC-DIVERGENCE: host_const_fill_body(F32, NaN/Inf, ...) emits invalid C++ literal NaNf/inff"]
fn rt_metal_emit_const_f32_nan_does_not_emit_invalid_c_literal_nanf() {
    let body = dtype::host_const_fill_body(Prim::F32, f64::NAN, "buf", 4);
    assert!(
        !body.contains("NaNf"),
        "F32 NaN must not emit the invalid C literal `NaNf`; got: {body}"
    );
}

#[test]
#[ignore = "SPEC-DIVERGENCE: host_const_fill_body(F32, NaN/Inf, ...) emits invalid C++ literal NaNf/inff"]
fn rt_metal_emit_const_f32_infinity_does_not_emit_invalid_c_literal_inff() {
    let body = dtype::host_const_fill_body(Prim::F32, f64::INFINITY, "buf", 4);
    assert!(
        !body.contains("inff"),
        "F32 +inf must not emit the invalid C literal `inff`; got: {body}"
    );
}
