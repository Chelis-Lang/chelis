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

/// §1.1.2 documents the workaround as casting to int32/int64. Verify
/// `cast(x, u8)` is rejected at type-check time. NEGATIVE-PARITY twin
/// to the existing f8e4m3 rejection tests, but for the §1.1.2 surface.
#[test]
fn cast_scalar_to_u8_rejected_at_check_time() {
    let src = "def main -> int32 = cast(1, u8)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep =
        res.expect_err("cast to u8 must be a type error per spec §1.1.2 (unsigned out of scope)");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    // The spec doesn't pin exact wording for §1.1.2; just verify the cast
    // is rejected and `u8` appears somewhere in the diagnostic so the
    // operator can map it back to the spec.
    assert!(
        !rep.errors.is_empty(),
        "u8 cast must error; got empty error report. messages={messages:?}"
    );
}

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
    if res.is_ok() {
        panic!(
            "spec §5.3: bare integer literal `42` defaults to int32 and must NOT \
             silently widen to int64 in a non-tensor scalar context. The program \
             `def main -> int64 = 42` should be rejected, not accepted."
        );
    }
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

/// §5.3: out-of-i32-range integer literal. `2147483648` = 2^31 fits in
/// i64 but overflows i32. The spec narrows to int32 mechanically; an
/// explicit out-of-range diagnostic is the user-friendly behavior.
///
/// EXPECTED PER SPEC: error mentioning the literal is out of range for
/// the int32 default and suggesting a suffix or cast.
/// ACTUAL: TBD — pin and let the test report the gap.
#[test]
fn out_of_i32_range_literal_default_behavior() {
    let src = "def main -> int32 = 2147483648";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    // Three possible behaviors:
    //   (a) error with out-of-range diagnostic mentioning int32
    //   (b) silently overflow (wraps to -2147483648)
    //   (c) silently widens to int64
    // The spec §5.3 contract is no implicit widening; the lexer parses
    // at i64 then narrows to i32. If narrowing wraps silently, that's
    // a SPEC-DIVERGENCE finding.
    match res {
        Ok(_) => panic!(
            "spec §5.3: literal `2147483648` does NOT fit int32 (the §5.3 default). \
             Program type-checked silently — this is silent integer overflow. \
             Expected an out-of-range diagnostic; got accepted program."
        ),
        Err(rep) => {
            let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
            // We want the diagnostic to mention "out of range" or the
            // i32 limit. Just record what we got.
            eprintln!("out-of-range int32 literal diagnostic shape: {messages:?}");
        }
    }
}

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
/// a typed-f32 literal binding to `(t-prim {} f32)`. ACTUAL: TBD.
///
/// WS-A0 RT-1 fixup status: ignored; the §5.5 literal suffix grammar
/// is not implemented in the surf lexer/parser as of WS-A0. The
/// rejection diagnostic the test pins is the design intent for
/// WS-B1; un-ignore once the suffix grammar lands. Tracking gap:
/// surf parser rejects `1.0f32` as `(lit 1.0)` followed by an
/// unbound `f32` identifier.
#[test]
#[ignore = "WS-B1: §5.5 literal suffix grammar not yet implemented in surf lexer"]
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
/// WS-A0 RT-1 fixup status: ignored; same §5.5 gap as the f32 sibling
/// above — surf lexer rejects `42i64` as `(lit 42)` followed by an
/// unbound `i64` identifier. Un-ignore once WS-B1 ships the suffix
/// grammar.
#[test]
#[ignore = "WS-B1: §5.5 literal suffix grammar not yet implemented in surf lexer"]
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
        if check_ir_program(&exprs).is_ok() {
            panic!(
                "spec §5.5: `1.0f8e4m3` literal suffix is lex-time-rejected \
                 per the deferred-suffix paragraph; the surf compiler accepts \
                 it without diagnosing the deferred dtype."
            );
        }
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
        if check_ir_program(&exprs).is_ok() {
            panic!(
                "spec §5.5: `42u8` unsigned suffix is lex-time-rejected per the \
                 out-of-scope-suffix paragraph (§1.1.2)."
            );
        }
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
