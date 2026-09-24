//! RT-2 adversarial coverage for IR lowering of reduce_sum across the
//! integer-precision rows (i8/i16) where spec §5.7.1 requires
//! result-precision = accumulator-precision = wider type.
//!
//! The lowering path (`crates/chelis-ir/src/lower.rs`) reads the
//! type-checker's output precision and uses it as the IR `accumulator`
//! field. If the type checker types `sum(int8_tensor)` as
//! `tensor[..., i8]` instead of `tensor[..., i32]` per spec, the
//! lowered IR will have `accumulator: i8` and IR-verify will reject
//! it with the §5.7.1 narrowness diagnostic. That's the symptom of the
//! type-checker spec divergence pinned in
//! crates/chelis-types/tests/numeric_dtype_buildout_adversarial.rs.

use chelis_ir::lower::lower_program;
use chelis_types::check_ir_program;

fn lower_deep(src: &str) -> Result<chelis_ir::dag::Dag, String> {
    let exprs = chelis_deep::parser::parse_str(src).expect("parse Deep fixture");
    let checked = check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked).map_err(|r| {
        format!(
            "check: {:?}",
            r.errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
        )
    })?;
    let checked = chelis_effects::check_program(&checked).expect("fixture effects");
    let checked = chelis_types::check_linearity(&checked).expect("fixture linearity");
    Ok(chelis_ir::lowering_runtime::with_native_lowering_runtime(
        |runtime| lower_program(&checked, runtime),
    ))
}

/// CRITICAL: end-to-end smoking gun. A program that passes type-check
/// but the lowered IR fails verify because the type-checker types
/// sum(i8) as i8 (operand precision), then lowering propagates
/// accumulator=i8, then verify catches the §5.7.1 narrowness
/// violation. The user sees an internal-IR error for what should be a
/// type error or a successful i32 result.
#[test]
fn end_to_end_sum_int8_lowering_produces_verify_failure() {
    // The user-visible source is innocuous: sum a tensor of i8
    // suffixed literals.
    let src = include_str!(
        "fixtures/core/tests/reduce_sum_lowering_adversarial/end_to_end_sum_int8_lowering_produces_verify_failure.deep"
    );
    lower_deep(src).expect_err("sum(i8) bound to i8 must be rejected before IR lowering");
}

/// The i16 result binding must likewise be rejected before IR lowering.
#[test]
fn end_to_end_sum_int16_lowering_produces_verify_failure() {
    let src = include_str!(
        "fixtures/core/tests/reduce_sum_lowering_adversarial/end_to_end_sum_int16_lowering_produces_verify_failure.deep"
    );
    lower_deep(src).expect_err("sum(i16) bound to i16 must be rejected before IR lowering");
}
