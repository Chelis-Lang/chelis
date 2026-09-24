//! RT-1 adversarial coverage for WS-A0 dtype foundation.
//!
//! Attacks the dtype-set boundary (§1.1, §1.1.1, §1.1.2), the literal-default
//! rule (§5.3), the cast-target admissibility (§1.1.2 / §1.1.1), and the
//! diagnostic-shape contract advertised by the existing
//! `f8e4m3_rejection.rs` acceptance file.
//!
//! Fixtures preserve the original literal metadata admitted by the checker.

use chelis_types::check_ir_program;
use chelis_types::types::Prim;

fn parse_core_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture")
}

// ---------------------------------------------------------------
// A. Dtype-set boundary
// ---------------------------------------------------------------

/// §1.1 + §1.1.1/§1.1.2: `u8` should never resolve to a Prim variant.
#[test]
fn parse_name_rejects_unsigned_u8() {
    assert_eq!(
        Prim::parse_name("u8"),
        None,
        "spec §1.1.1: unsigned types are deferred (and `u8` is not even the \
         reserved spelling); parse_name must not resolve it"
    );
}

/// §1.1.2: `uint32` must not resolve.
#[test]
fn parse_name_rejects_uint32() {
    assert_eq!(Prim::parse_name("uint32"), None);
}

/// §1.1: `f128` is not in the set.
#[test]
fn parse_name_rejects_f128() {
    assert_eq!(Prim::parse_name("f128"), None);
}

/// §1.1: `fp32` is not the spelling.
#[test]
fn parse_name_rejects_fp32_misspelling() {
    assert_eq!(Prim::parse_name("fp32"), None);
}

// `cast(1, u8)` rejection at check time is pinned with the exact §1.1.1
// diagnostic by `unsigned_dtype_rejection.rs::
// cast_scalar_to_u8_rejected_with_spec_1_1_1_diagnostic`, which asserts
// a strict superset of the looser rejection-only check that previously
// lived here.

// ---------------------------------------------------------------
// D. Literal-default rule (§5.3)
// ---------------------------------------------------------------

/// §5.3: bare integer literal defaults to i32.
#[test]
fn bare_int_literal_defaults_to_int32() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/bare_int_literal_defaults_to_int32_71.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "bare `42` should type-check as i32 against an i32 declared return; \
         got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.3: bare integer literal must NOT default to i64.
/// A program that asserts `i64` against bare `42` must fail.
#[test]
fn bare_int_literal_rejected_against_int64_context() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/bare_int_literal_rejected_against_int64_context_86.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    // Per §5.3 contextual inference (§5.6) should NOT widen a bare
    // integer literal to i64 outside a tensor literal context. So
    // either: (a) the checker rejects mismatch, or (b) it silently
    // widens. We pin (a) and let a failure flag (b) as a finding.
    if res.is_ok() {
        panic!(
            "spec §5.3: bare integer literal `42` defaults to i32 and must NOT \
             silently widen to i64 in a non-tensor scalar context. The program \
             `def main() -> i64 = 42` should be rejected, not accepted."
        );
    }
}

/// §5.3 + §5.6 boundary: bare float literal defaults to f32 even inside an
/// i64-typed scalar binding. Should error.
#[test]
fn bare_float_literal_does_not_satisfy_int64() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/bare_float_literal_does_not_satisfy_int64_106.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_err(),
        "spec §5.3: bare float literal `1.0` is f32 by default, never i64"
    );
}

// The `def main() -> i32 = 2147483648` out-of-i32-range default case is
// pinned with the exact §5.3 range diagnostic by
// `int_literal_overflow.rs::
// literal_2_pow_31_rejected_with_spec_5_3_range_diagnostic`, which
// asserts a strict superset of the rejection-only check that previously
// lived here (same source, plus the exact out-of-range phrase, the i64
// suffix workaround, the cast workaround, and the §5.3 spec citation).

// ---------------------------------------------------------------
// H. Negative-parity audit on f8e4m3_rejection.rs
// ---------------------------------------------------------------

/// The existing acceptance file checks `cast(1.0, f8e4m3)` errors, but
/// does NOT check that the diagnostic appears under the `to_tensor`
/// indirect path. This pins it.
#[test]
fn cast_to_f8e4m3_via_to_tensor_pipe_still_rejected() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/cast_to_f8e4m3_via_to_tensor_pipe_still_rejected_132.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let rep = res.expect_err("indirect cast to f8e4m3 must still be rejected");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("f8e4m3 is deferred")),
        "diagnostic must still cite the spec phrase under indirect path; got: {messages:?}"
    );
}

/// Explicit f32 literal metadata is accepted against an f32 return.
#[test]
fn literal_suffix_f32_is_implemented_per_spec_5_5() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/literal_suffix_f32_is_implemented_per_spec_5_5_154.deep"
    );
    let exprs = parse_core_fixture(src);
    let res = check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "spec §5.5: `1.0f32` should bind at f32 against an f32 return; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.5: integer literal suffixes (`42i64`).
///
/// WS-B1 status: implemented; see `literal_suffix_f32_is_implemented_per_spec_5_5`.
#[test]
fn literal_suffix_i64_is_implemented_per_spec_5_5() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/literal_suffix_i64_is_implemented_per_spec_5_5_182.deep"
    );
    let exprs = parse_core_fixture(src);
    let res = check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "spec §5.5: `42i64` should bind at i64 against an i64 return"
    );
}

/// The acceptance file checks the DIRECT cast diagnostic. It does NOT
/// pin that subsequent operations on the `cast(_, f8e4m3)` result also
/// cleanly halt rather than cascade. Pin here.
#[test]
fn cast_to_f8e4m3_then_arithmetic_does_not_cascade_silently() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/cast_to_f8e4m3_then_arithmetic_does_not_cascade_silently_254.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let rep = res.expect_err("must error on the f8e4m3 cast");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("f8e4m3 is deferred")),
        "primary error must still be the f8e4m3 deferral; got: {messages:?}"
    );
}

// ---------------------------------------------------------------
// I. WS-B1: literal suffix grammar (spec §5.5)
// ---------------------------------------------------------------

/// §5.5: `1.0bf16` infers `bf16`, no widening.
#[test]
fn bf16_literal_suffix_infers_bf16() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/bf16_literal_suffix_infers_bf16_272.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "spec §5.5: `1.0bf16` should bind at bf16; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.5: `42i64` infers `i64`.
#[test]
fn i64_literal_suffix_infers_int64() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/i64_literal_suffix_infers_int64_285.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_ok(),
        "spec §5.5: `42i64` should bind at i64; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.5: a typed literal does NOT widen. `1.0f32` against an `f64`
/// return position is a TypeMismatch, not a silent promotion. The
/// concrete diagnostic shape is implementation-defined (the checker
/// today reports a signature mismatch); the load-bearing assertion is
/// that the program is rejected.
#[test]
fn f32_literal_does_not_widen_to_f64_context() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/f32_literal_does_not_widen_to_f64_context_302.deep"
    );
    let deep = parse_core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_err(),
        "spec §5.5: typed `1.0f32` must not silently widen to f64; checker accepted: {res:?}"
    );
}

/// An i64 return context does not rescue an overflowing unsuffixed i32 literal.
#[test]
fn overflowing_unsuffixed_literal_rejects_in_i64_context() {
    let src = include_str!(
        "fixtures/core/tests/dtype_foundation_adversarial/d1_diagnostic_mentions_i64_suffix_and_cast_384.deep"
    );
    let deep = parse_core_fixture(src);
    check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("literal must overflow the i32 default even in an i64 context");
}
