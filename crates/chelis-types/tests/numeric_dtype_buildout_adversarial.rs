//! RT-2 adversarial coverage for Wave 2 numeric dtype build-out.
//!
//! Attacks the integration of WS-A1..A4 (backend dtype completeness),
//! WS-B1 (literal suffix grammar), and WS-B2 (contextual tensor-literal
//! inference) at the type-checker layer.
//!
//! Spec references:
//!   - spec/04-type-system.md §1.1, §1.1.1, §1.1.2, §5.3, §5.4, §5.5,
//!     §5.6, §5.7, §5.7.1, §5.7.2
//!   - spec/05-risc-primitives.md §2.3, §4.1
//!   - spec/02-surf-syntax.md §P10, §P10a, §P10b
//!
//! No "did not panic" assertions. Every test pins exact behavior.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn err_messages(rep: &chelis_types::infer::InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.6 + §5.3 — out-of-range literals in contextual
// tensor positions must be rejected per the WS-A0 D1 rule.
//
// Spec §5.3: "the lexer parses an unsuffixed integer or float literal
// token at i64/f64 precision so that out-of-range literals can be
// diagnosed before defaulting".
//
// In a contextual int8 position (§5.6), the literal `200` overflows
// int8 (max 127). The implementation must error.
// ----------------------------------------------------------------

#[test]
fn ws_b2_int8_context_accepts_out_of_range_literal_silent_overflow() {
    // Per spec §5.6 + §5.3, [1, 2, 200] in a tensor[3, int8] context
    // should emit a range-overflow diagnostic. If it doesn't, the
    // implementation silently truncates 200 → -56 (int8 wrap), which is
    // exactly the silent-data-loss class the WS-A0 D1 rule was added
    // to prevent.
    let src = "xs: tensor[3, int8] = [1, 2, 200]";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.6 + §5.3: literal 200 is out of int8 range when contextually \
         inferred as int8; must produce a diagnostic. Currently: silent acceptance \
         (truncation to -56). errs={errs:?}"
    );
    assert!(
        errs.iter()
            .any(|m| m.contains("range") || m.contains("out of") || m.contains("200")),
        "diagnostic must cite range/overflow/value 200; got: {errs:?}"
    );
}

#[test]
fn ws_b2_int16_context_accepts_out_of_range_literal_silent_overflow() {
    let src = "xs: tensor[3, int16] = [1, 2, 70000]";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.6 + §5.3: literal 70000 is out of int16 range; must be a \
         type error. Currently: silent acceptance. errs={errs:?}"
    );
    assert!(
        errs.iter()
            .any(|m| m.contains("range") || m.contains("out of") || m.contains("70000")),
        "diagnostic must cite range/overflow/value 70000; got: {errs:?}"
    );
}

#[test]
fn ws_b2_int8_context_negative_out_of_range_literal_silent_overflow() {
    // int8 range is [-128, 127]. -200 is out of range.
    let src = "xs: tensor[3, int8] = [1, 2, -200]";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.6 + §5.3: literal -200 is out of int8 range; must be a \
         type error. errs={errs:?}"
    );
}

#[test]
fn ws_b2_int8_context_at_boundary_accepted() {
    // int8 range is [-128, 127]. 127 is the boundary.
    let src = "xs: tensor[3, int8] = [125, 126, 127]";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(res.is_ok(), "int8 boundary 127 must type-check cleanly");
}

#[test]
fn ws_b2_int8_context_at_boundary_negative_accepted() {
    // Surf parses -128 as `(neg 128)`, where the inner literal 128
    // overflows int8 (max 127). Whether this is rejected or admitted
    // depends on whether the contextual rule applies inside `neg`.
    // Pin the actual behavior.
    let src = "xs: tensor[3, int8] = [-128, -127, -126]";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match &res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(rep),
    };
    // Either:
    //   (a) int8 boundary -128 admitted (the parser/desugarer folds the
    //       sign into the literal, or the contextual rule sees -128 as
    //       a single literal), or
    //   (b) int8 -128 errors because the literal-128 inner overflows.
    // Document which one ships. If (b), this is a footgun: user-facing
    // -128 is rejected at the int8 boundary because of a parser quirk.
    if res.is_err() {
        panic!(
            "boundary footgun: int8 boundary value -128 (i8::MIN) is rejected \
             because the parser produces (neg 128) and 128 overflows int8 max. \
             User-facing literal -128 should be admitted as i8::MIN; \
             errs={errs:?}"
        );
    }
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.7.2 — integer matmul must be a TYPE error.
//
// Spec: "The active matmul signature does not admit integer operand
// precisions (int8, int16, int32, int64)." This is a type-check
// rule, not a backend or IR-verify rule.
//
// The IR verify layer rejects integer matmul, but ideally this should
// surface as a type error with §5.7.2 citation, not as an IR-verify
// failure. Document where the rejection actually happens.
// ----------------------------------------------------------------

#[test]
fn ws_a4_matmul_int8_must_be_rejected_with_spec_5_7_2_diagnostic() {
    let src = r#"
        a: tensor[2, 3, int8] = [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
        b: tensor[3, 2, int8] = [[1i8, 2i8], [3i8, 4i8], [5i8, 6i8]]
        out: tensor[2, 2, int8] = matmul(&a, &b)
    "#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.2: matmul on int8 operands must be a type error. \
         Currently the type checker accepts it (rejection happens later \
         at IR-verify or codegen). errs={errs:?}"
    );
    assert!(
        errs.iter()
            .any(|m| m.contains("§5.7.2") || m.contains("integer") || m.contains("int8")),
        "matmul-int8 type-check rejection must cite §5.7.2 or mention integer; \
         got: {errs:?}"
    );
}

#[test]
fn ws_a4_matmul_int32_must_be_rejected_with_spec_5_7_2_diagnostic() {
    let src = r#"
        a: tensor[2, 3, int32] = [[1, 2, 3], [4, 5, 6]]
        b: tensor[3, 2, int32] = [[1, 2], [3, 4], [5, 6]]
        out: tensor[2, 2, int32] = matmul(&a, &b)
    "#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.2: matmul on int32 operands must be a type error; \
         got no errors"
    );
    assert!(
        errs.iter()
            .any(|m| m.contains("§5.7.2") || m.contains("integer") || m.contains("int32")),
        "matmul-int32 rejection must cite §5.7.2 or integer; got: {errs:?}"
    );
}

// ----------------------------------------------------------------
// SPEC-DIVERGENCE: spec §5.7.1 result-precision rule for reduce_sum.
//
// Spec §5.7.1: "The result precision of `reduce_sum` is the
// accumulator precision."
//
// Spec table column: "Result precision" reads "operand precision (`bf16`)"
// for the float lane and "int32" for int8/int16. The text and table
// AGREE for int8/int16 (both → int32) and for int32, int64, f32, f64
// (where operand == accumulator). They DISAGREE for bf16/f16 (table
// says operand = bf16; text says result = accumulator = f32).
// ----------------------------------------------------------------

#[test]
fn ws_b2_reduce_sum_int8_result_precision_must_be_int32_per_spec_5_7_1() {
    // Per spec §5.7.1 row "int8": Result precision is int32.
    // The implementation should type sum(tensor[N, int8]) as
    // tensor[..., int32], not tensor[..., int8].
    let src = "xs: tensor[3, int8] = [1i8, 2i8, 3i8]\nout: tensor[int32] = sum(&xs, 0)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        errs.is_empty(),
        "spec §5.7.1: reduce_sum on int8 must yield int32 result; \
         binding to tensor[int32] should type-check; got errors: {errs:?}"
    );
}

#[test]
fn ws_b2_reduce_sum_int8_result_must_not_be_int8_per_spec_5_7_1() {
    // The contrapositive: binding sum(int8 tensor) to tensor[int8]
    // must error per spec §5.7.1 (result precision = accumulator = int32).
    let src = "xs: tensor[3, int8] = [1i8, 2i8, 3i8]\nout: tensor[int8] = sum(&xs, 0)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.1: sum(int8) result is int32, not int8; binding to \
         tensor[int8] must error. Got: no errors (the type checker \
         types sum(int8) as int8, contradicting §5.7.1)."
    );
}

#[test]
fn ws_b2_reduce_sum_int16_result_must_not_be_int16_per_spec_5_7_1() {
    let src = "xs: tensor[3, int16] = [1i16, 2i16, 3i16]\nout: tensor[int16] = sum(&xs, 0)";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.1: sum(int16) result is int32, not int16; binding to \
         tensor[int16] must error."
    );
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.5 canonical decimal suffix rule — integer-typed suffixes only.
// ----------------------------------------------------------------

/// §5.5: canonical decimal `255i8` should bind at int8. Then `255i8 +
/// 255i16` is a precision mismatch per §5.4 and must error.
#[test]
fn ws_b1_decimal_suffix_mixed_precision_addition_rejected_per_spec_5_4() {
    let src = "out: int16 = 255i8 + 255i16";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_err(),
        "spec §5.4: arithmetic requires same precision; \
         255i8 + 255i16 must error"
    );
}

// ----------------------------------------------------------------
// SPEC-DIVERGENCE: literal suffix grammar — `1.0i8` is a parse error
// per §5.5 ("Integer-typed suffixes attach to integer literal tokens
// only; `1.0i8` is a parse error.").
// ----------------------------------------------------------------

#[test]
fn ws_b1_float_token_with_int_suffix_is_parse_error_per_spec_5_5() {
    let src = "out: int8 = 1.0i8";
    // Per §5.5, this MUST fail at lex/parse time.
    let parse_res = parse_str(src);
    assert!(
        parse_res.is_err(),
        "spec §5.5: `1.0i8` (float token + int suffix) must be a parse error; \
         got Ok parse"
    );
}

// ----------------------------------------------------------------
// NEGATIVE PARITY: spec §5.4 same-precision with WS-B1 suffixes.
// ----------------------------------------------------------------

#[test]
fn ws_b1_mixed_float_suffix_addition_rejected_per_spec_5_4() {
    let src = "out: f64 = 1.0f32 + 1.0f64";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_err(),
        "spec §5.4: f32 + f64 must error; got: {:?}",
        res.ok().map(|_| "no error")
    );
}

#[test]
fn ws_b1_mixed_int_suffix_addition_rejected_per_spec_5_4() {
    let src = "out: int32 = 1i8 + 1i32";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    assert!(
        res.is_err(),
        "spec §5.4: i8 + i32 must error; got: {:?}",
        res.ok().map(|_| "no error")
    );
}

// ----------------------------------------------------------------
// NEGATIVE PARITY: spec §1.1.1 — f8e4m3 in tensor literal contextual
// inference must error.
// ----------------------------------------------------------------

#[test]
fn ws_a0_f8e4m3_in_contextual_tensor_position_rejected() {
    let src = "xs: tensor[3, f8e4m3] = [1.0, 2.0, 3.0]";
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
    let rep = res.expect_err("spec §1.1.1: f8e4m3 tensor element type must be rejected");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("§1.1.1") || m.contains("f8e4m3") || m.contains("deferred")),
        "f8e4m3 tensor-element rejection must cite §1.1.1 or mention deferred; got: {messages:?}"
    );
}

// ----------------------------------------------------------------
// NEGATIVE PARITY: spec §1.1.2 — unsigned suffixes must lex error.
// Existing unsigned_dtype_rejection.rs covers `cast(x, u8)` but
// not the suffix path.
// ----------------------------------------------------------------

#[test]
fn ws_b1_unsigned_suffix_u8_lex_rejected_per_spec_1_1_2() {
    let src = "out: int8 = 42u8";
    let parse_res = parse_str(src);
    assert!(
        parse_res.is_err(),
        "spec §1.1.2: `42u8` suffix must be a lex error"
    );
}

#[test]
fn ws_b1_unsigned_suffix_u16_lex_rejected_per_spec_1_1_2() {
    let src = "out: int16 = 42u16";
    let parse_res = parse_str(src);
    assert!(
        parse_res.is_err(),
        "spec §1.1.2: `42u16` suffix must be a lex error"
    );
}

#[test]
fn ws_b1_f8e4m3_suffix_lex_rejected_per_spec_1_1_1() {
    let src = "out: f32 = 1.0f8e4m3";
    let parse_res = parse_str(src);
    assert!(
        parse_res.is_err(),
        "spec §1.1.1: `1.0f8e4m3` suffix must be a lex error"
    );
}
