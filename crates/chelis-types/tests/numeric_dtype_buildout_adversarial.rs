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
//!
//! No "did not panic" assertions. Every test pins exact behavior.

use chelis_types::check_ir_program;

fn core_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture must parse")
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
// In a contextual i8 position (§5.6), the literal `200` overflows
// i8 (max 127). The implementation must error.
// ----------------------------------------------------------------

#[test]
fn ws_b2_int8_context_accepts_out_of_range_literal_silent_overflow() {
    // Per spec §5.6 + §5.3, [1, 2, 200] in a tensor[3, i8] context
    // should emit a range-overflow diagnostic. If it doesn't, the
    // implementation silently truncates 200 → -56 (i8 wrap), which is
    // exactly the silent-data-loss class the WS-A0 D1 rule was added
    // to prevent.
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_001.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.6 + §5.3: literal 200 is out of i8 range when contextually \
         inferred as i8; must produce a diagnostic. Currently: silent acceptance \
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
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_002.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.6 + §5.3: literal 70000 is out of i16 range; must be a \
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
    // i8 range is [-128, 127]. -200 is out of range.
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_003.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.6 + §5.3: literal -200 is out of i8 range; must be a \
         type error. errs={errs:?}"
    );
}

#[test]
fn ws_b2_int8_context_at_boundary_accepted() {
    // i8 range is [-128, 127]. 127 is the boundary.
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_004.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(res.is_ok(), "i8 boundary 127 must type-check cleanly");
}

#[test]
fn ws_b2_int8_context_at_boundary_negative_accepted() {
    // Surf parses -128 as `(neg 128)`, where the inner literal 128
    // overflows i8 (max 127). Whether this is rejected or admitted
    // depends on whether the contextual rule applies inside `neg`.
    // Pin the actual behavior.
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_005.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match &res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(rep),
    };
    // Either:
    //   (a) i8 boundary -128 admitted (the parser/desugarer folds the
    //       sign into the literal, or the contextual rule sees -128 as
    //       a single literal), or
    //   (b) i8 -128 errors because the literal-128 inner overflows.
    // Document which one ships. If (b), this is a footgun: user-facing
    // -128 is rejected at the i8 boundary because of a parser quirk.
    if res.is_err() {
        panic!(
            "boundary footgun: i8 boundary value -128 (i8::MIN) is rejected \
             because the parser produces (neg 128) and 128 overflows i8 max. \
             User-facing literal -128 should be admitted as i8::MIN; \
             errs={errs:?}"
        );
    }
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.7.2 — integer matmul must be a TYPE error.
//
// Spec: "The active matmul signature does not admit integer operand
// precisions (i8, i16, i32, i64)." This is a type-check
// rule, not a backend or IR-verify rule.
//
// The IR verify layer rejects integer matmul, but ideally this should
// surface as a type error with §5.7.2 citation, not as an IR-verify
// failure. Document where the rejection actually happens.
// ----------------------------------------------------------------

#[test]
fn ws_a4_matmul_int8_must_be_rejected_with_spec_5_7_2_diagnostic() {
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_006.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.2: matmul on i8 operands must be a type error. \
         Currently the type checker accepts it (rejection happens later \
         at IR-verify or codegen). errs={errs:?}"
    );
    assert!(
        errs.iter()
            .any(|m| m.contains("§5.7.2") || m.contains("integer") || m.contains("i8")),
        "matmul-i8 type-check rejection must cite §5.7.2 or mention integer; \
         got: {errs:?}"
    );
}

#[test]
fn ws_a4_matmul_int32_must_be_rejected_with_spec_5_7_2_diagnostic() {
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_007.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.2: matmul on i32 operands must be a type error; \
         got no errors"
    );
    assert!(
        errs.iter()
            .any(|m| m.contains("§5.7.2") || m.contains("integer") || m.contains("i32")),
        "matmul-i32 rejection must cite §5.7.2 or integer; got: {errs:?}"
    );
}

// ----------------------------------------------------------------
// SPEC-DIVERGENCE: spec §5.7.1 result-precision rule for reduce_sum.
//
// Spec §5.7.1: "The result precision of `reduce_sum` is the
// accumulator precision."
//
// Spec table column: "Result precision" reads "operand precision (`bf16`)"
// for the float lane and "i32" for i8/i16. The text and table
// AGREE for i8/i16 (both → i32) and for i32, i64, f32, f64
// (where operand == accumulator). They DISAGREE for bf16/f16 (table
// says operand = bf16; text says result = accumulator = f32).
// ----------------------------------------------------------------

#[test]
fn ws_b2_reduce_sum_int8_result_precision_must_be_int32_per_spec_5_7_1() {
    // Per spec §5.7.1 row "i8": Result precision is i32.
    // The implementation should type sum(tensor[N, i8]) as
    // tensor[..., i32], not tensor[..., i8].
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_008.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        errs.is_empty(),
        "spec §5.7.1: reduce_sum on i8 must yield i32 result; \
         binding to tensor[i32] should type-check; got errors: {errs:?}"
    );
}

#[test]
fn ws_b2_reduce_sum_int8_result_must_not_be_int8_per_spec_5_7_1() {
    // The contrapositive: binding sum(i8 tensor) to tensor[i8]
    // must error per spec §5.7.1 (result precision = accumulator = i32).
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_009.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.1: sum(i8) result is i32, not i8; binding to \
         tensor[i8] must error. Got: no errors (the type checker \
         types sum(i8) as i8, contradicting §5.7.1)."
    );
}

#[test]
fn ws_b2_reduce_sum_int16_result_must_not_be_int16_per_spec_5_7_1() {
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_010.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let errs = match res {
        Ok(_) => Vec::<String>::new(),
        Err(rep) => err_messages(&rep),
    };
    assert!(
        !errs.is_empty(),
        "spec §5.7.1: sum(i16) result is i32, not i16; binding to \
         tensor[i16] must error."
    );
}

// ----------------------------------------------------------------
// CRITICAL: spec §5.5 canonical decimal suffix rule — integer-typed suffixes only.
// ----------------------------------------------------------------

/// §5.5: canonical decimal `255i8` should bind at i8. Then `255i8 +
/// 255i16` is a precision mismatch per §5.4 and must error.
#[test]
fn ws_b1_decimal_suffix_mixed_precision_addition_rejected_per_spec_5_4() {
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_011.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_err(),
        "spec §5.4: arithmetic requires same precision; \
         255i8 + 255i16 must error"
    );
}

// ----------------------------------------------------------------
// NEGATIVE PARITY: spec §5.4 same-precision with WS-B1 suffixes.
// ----------------------------------------------------------------

#[test]
fn ws_b1_mixed_float_suffix_addition_rejected_per_spec_5_4() {
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_012.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        res.is_err(),
        "spec §5.4: f32 + f64 must error; got: {:?}",
        res.ok().map(|_| "no error")
    );
}

#[test]
fn ws_b1_mixed_int_suffix_addition_rejected_per_spec_5_4() {
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_013.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
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
    let src = include_str!("fixtures/core/tests/numeric_dtype_buildout_adversarial/case_014.deep");
    let deep = core_fixture(src);
    let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
    let rep = res.expect_err("spec §1.1.1: f8e4m3 tensor element type must be rejected");
    let messages: Vec<&str> = rep.errors.iter().map(|e| e.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("§1.1.1") || m.contains("f8e4m3") || m.contains("deferred")),
        "f8e4m3 tensor-element rejection must cite §1.1.1 or mention deferred; got: {messages:?}"
    );
}
