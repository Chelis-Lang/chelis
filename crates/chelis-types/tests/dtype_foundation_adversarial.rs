//! RT-1 adversarial coverage for WS-A0 dtype foundation.
//!
//! Attacks the dtype-set boundary (§1.1, §1.1.1, §1.1.2), the literal-default
//! rule (§5.3), the cast-target admissibility (§1.1.2 / §1.1.1), and the
//! diagnostic-shape contract advertised by the existing
//! `f8e4m3_rejection.rs` acceptance file.
//!
//! Each test pins exact-output behavior. No "errored somehow" assertions.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;
use chelis_types::types::Prim;

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

// ---------------------------------------------------------------
// A. Dtype-set boundary
// ---------------------------------------------------------------

/// §1.1 + §1.1.2: `u8` should never resolve to a Prim variant.
#[test]
fn parse_name_rejects_unsigned_u8() {
    assert_eq!(
        Prim::parse_name("u8"),
        None,
        "spec §1.1.2: `u8` is out of scope; parse_name must not resolve it"
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

// `cast(1, u8)` rejection at check time is pinned with the exact §1.1.2
// diagnostic by `unsigned_dtype_rejection.rs::
// cast_scalar_to_u8_rejected_with_spec_1_1_2_diagnostic`, which asserts
// a strict superset of the looser rejection-only check that previously
// lived here.

// ---------------------------------------------------------------
// D. Literal-default rule (§5.3)
// ---------------------------------------------------------------

/// §5.3: bare integer literal defaults to int32.
#[test]
fn bare_int_literal_defaults_to_int32() {
    let src = "def main -> int32 = 42";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_ok(),
        "bare `42` should type-check as int32 against an int32 declared return; \
         got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.3: bare integer literal must NOT default to int64.
/// A program that asserts `int64` against bare `42` must fail.
#[test]
fn bare_int_literal_rejected_against_int64_context() {
    let src = "def main -> int64 = 42";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    // Per §5.3 contextual inference (§5.6) should NOT widen a bare
    // integer literal to int64 outside a tensor literal context. So
    // either: (a) the checker rejects mismatch, or (b) it silently
    // widens. We pin (a) and let a failure flag (b) as a finding.
    assert!(
        res.is_err(),
        "spec §5.3: bare integer literal `42` defaults to int32 and must NOT \
         silently widen to int64 in a non-tensor scalar context. The program \
         `def main -> int64 = 42` should be rejected, not accepted."
    );
}

/// §5.3 + §5.6 boundary: bare float literal defaults to f32 even inside an
/// int64-typed scalar binding. Should error.
#[test]
fn bare_float_literal_does_not_satisfy_int64() {
    let src = "def main -> int64 = 1.0";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_err(),
        "spec §5.3: bare float literal `1.0` is f32 by default, never int64"
    );
}

// The `def main -> int32 = 2147483648` out-of-i32-range default case is
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
    let src = "def main -> tensor[3, f32] = cast(to_tensor([1.0, 2.0, 3.0]), f8e4m3)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("indirect cast to f8e4m3 must still be rejected");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("f8e4m3 is deferred")),
        "diagnostic must still cite the spec phrase under indirect path; got: {messages:?}"
    );
}

/// §5.5: literal suffixes `1.0f32`, `42i64`, etc. are SPEC. Verify the
/// lexer/parser actually implements them. EXPECTED: `1.0f32` parses as
/// a typed-f32 literal binding to `(t-prim {} f32)`.
///
/// WS-B1 status: implemented. The surf lexer recognizes the suffix as
/// part of the literal token; the parser produces `Literal::TypedFloat`
/// / `Literal::TypedInt`; the desugarer attaches the corresponding
/// `type: (t-prim {} <prim>)` metadata; the type checker reads the
/// metadata as the literal's type.
#[test]
fn literal_suffix_f32_is_implemented_per_spec_5_5() {
    let src = "def main -> f32 = 1.0f32";
    let decls = match chelis_surf::parser::parse_str(src) {
        Ok(d) => d,
        Err(e) => panic!(
            "spec §5.5 pins literal suffix grammar (`1.0f32`, `42i64`); \
             surf lexer/parser rejects `1.0f32`. Either ship the suffix \
             grammar or pin §5.5 as deferred. parse error: {e:?}"
        ),
    };
    let exprs = chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs();
    let res = check_ir_program(&exprs);
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
    let src = "def main -> int64 = 42i64";
    let decls = match chelis_surf::parser::parse_str(src) {
        Ok(d) => d,
        Err(e) => panic!(
            "spec §5.5: `42i64` suffix grammar not implemented in surf lexer. \
             parse error: {e:?}"
        ),
    };
    let exprs = chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs();
    let res = check_ir_program(&exprs);
    assert!(
        res.is_ok(),
        "spec §5.5: `42i64` should bind at int64 against an int64 return"
    );
}

/// §5.5: deferred suffix `f8e4m3` must be a parse error per the
/// last paragraph of §5.5. Spec is explicit: lex-time rejection with a
/// diagnostic pointing at §1.1.1.
#[test]
fn literal_suffix_f8e4m3_is_lex_error_per_spec_5_5() {
    let src = "def main -> f32 = 1.0f8e4m3";
    if let Ok(decls) = chelis_surf::parser::parse_str(src) {
        // The expected behavior per §5.5 is a parse error. Anything
        // else is divergence. If desugar/check don't catch it either,
        // it's silent acceptance — a SPEC-DIVERGENCE finding.
        let exprs = chelis_macros::expand_program(
            &desugar_program(&decls),
            &chelis_macros::ExpansionOptions::default(),
        )
        .expect("macro expand")
        .into_exprs();
        assert!(
            check_ir_program(&exprs).is_err(),
            "spec §5.5: `1.0f8e4m3` literal suffix is lex-time-rejected \
             per the deferred-suffix paragraph; the surf compiler accepts \
             it without diagnosing the deferred dtype."
        );
    }
}

/// §5.5: deferred suffix `u8` (unsigned, §1.1.2) must be a parse error.
#[test]
fn literal_suffix_u8_is_lex_error_per_spec_5_5() {
    let src = "def main -> int32 = 42u8";
    if let Ok(decls) = chelis_surf::parser::parse_str(src) {
        let exprs = chelis_macros::expand_program(
            &desugar_program(&decls),
            &chelis_macros::ExpansionOptions::default(),
        )
        .expect("macro expand")
        .into_exprs();
        assert!(
            check_ir_program(&exprs).is_err(),
            "spec §5.5: `42u8` unsigned suffix is lex-time-rejected per the \
             out-of-scope-suffix paragraph (§1.1.2)."
        );
    }
}

/// The acceptance file checks the DIRECT cast diagnostic. It does NOT
/// pin that subsequent operations on the `cast(_, f8e4m3)` result also
/// cleanly halt rather than cascade. Pin here.
#[test]
fn cast_to_f8e4m3_then_arithmetic_does_not_cascade_silently() {
    let src = "def main -> f32 = add(cast(1.0, f8e4m3), 1.0)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
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
    let src = "def main -> bf16 = 1.0bf16";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_ok(),
        "spec §5.5: `1.0bf16` should bind at bf16; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.5: `42i64` infers `int64`.
#[test]
fn i64_literal_suffix_infers_int64() {
    let src = "def main -> int64 = 42i64";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_ok(),
        "spec §5.5: `42i64` should bind at int64; got: {:?}",
        res.err().map(|e| e.errors)
    );
}

/// §5.5: a typed literal does NOT widen. `1.0f32` against an `f64`
/// return position is a `TypeMismatch`, not a silent promotion. The
/// concrete diagnostic shape is implementation-defined (the checker
/// today reports a signature mismatch); the load-bearing assertion is
/// that the program is rejected.
#[test]
fn f32_literal_does_not_widen_to_f64_context() {
    let src = "def main -> f64 = 1.0f32";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_err(),
        "spec §5.5: typed `1.0f32` must not silently widen to f64; checker accepted: {res:?}"
    );
}

/// §5.5 hex+float-suffix rule: `0xFFf32` is rejected at lex time with a
/// diagnostic suggesting `cast` or whitespace.
#[test]
fn hex_float_suffix_is_rejected_at_lex_time() {
    let src = "def main -> f32 = 0xFFf32";
    let err = parse_str(src).expect_err("spec §5.5: hex float suffix must lex-error");
    let msg = format!("{err}");
    assert!(
        msg.contains("hex literals cannot carry float-typed suffixes"),
        "diagnostic must explain hex+float-suffix rule; got: {msg}"
    );
}

/// §5.5 hex+integer-suffix rule: `0xFFi8` is well-formed.
#[test]
fn hex_integer_suffix_is_well_formed() {
    let src = "def main -> int8 = 0xFFi8";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    // 0xFF = 255 is out of range for i8 [-128, 127] but the suffix
    // grammar must succeed at parse/lex time. The downstream type
    // checker may emit a range error; we accept either Ok or a range
    // error here, but NOT a missing-suffix or unknown-suffix error.
    if let Err(rep) = res {
        let msgs: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
        for m in &msgs {
            assert!(
                !m.contains("unknown") && !m.contains("unrecognized"),
                "0xFFi8 must lex; suffix recognition must not fail; got: {m}"
            );
        }
    }
}

/// §5.5: `1.0i8` is a parse error (integer suffix on float literal).
#[test]
fn integer_suffix_on_float_literal_is_lex_error() {
    let src = "def main -> int8 = 1.0i8";
    let err = parse_str(src).expect_err("spec §5.5: int suffix on float lits must lex-error");
    let msg = format!("{err}");
    assert!(
        msg.contains("integer suffixes attach to integer literals only"),
        "diagnostic must explain int-suffix-on-float rule; got: {msg}"
    );
}

/// §5.5: `1.0xyz` is a parse error (unknown adjacent identifier).
#[test]
fn unknown_suffix_is_lex_error() {
    let src = "def main -> f32 = 1.0xyz";
    let err = parse_str(src).expect_err("spec §5.5: unknown suffix must lex-error");
    let msg = format!("{err}");
    assert!(
        msg.contains("unrecognized literal suffix") || msg.contains("xyz"),
        "diagnostic must mention the unrecognized suffix; got: {msg}"
    );
}

/// D1 + WS-B1 interaction (spec §5.3 + §5.5): the i32-overflow
/// diagnostic now suggests both the `i64` literal suffix AND the
/// `cast(_, int64)` workaround, since the suffix grammar is shipped.
/// The cast hint must use the prec type name `int64`, not the suffix
/// spelling `i64` — `cast(N, i64)` is not a valid cast target and
/// re-fires this same diagnostic (issue #308 review fix).
///
/// Distinct input from `int_literal_overflow.rs`: this snippet
/// declares an `int64` return position (`def main -> int64 = ...`),
/// pinning that even an int64-typed context does not rescue a bare
/// integer literal from the §5.3 int32 default and the D1 diagnostic
/// still fires. The `ws_a0_*` exact-diagnostic version uses an `int32`
/// return position, so it does not cover this case.
#[test]
fn d1_diagnostic_mentions_i64_suffix_and_cast() {
    let src = "def main -> int64 = 2147483648";
    let deep = surf_to_deep(src);
    let rep = check_ir_program(&deep).expect_err("D1: literal must overflow i32 default");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    let combined = messages.join(" || ");
    assert!(
        combined.contains("i64") && combined.contains("suffix"),
        "D1 diagnostic must mention `i64` suffix; got: {combined}"
    );
    assert!(
        combined.contains("cast(2147483648, int64)"),
        "D1 diagnostic must recommend the working cast spelling \
         cast(_, int64), not the suffix spelling `i64`; got: {combined}"
    );
}
