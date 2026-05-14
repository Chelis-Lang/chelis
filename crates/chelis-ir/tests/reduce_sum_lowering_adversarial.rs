//! RT-2 adversarial coverage for IR lowering of reduce_sum across the
//! integer-precision rows (int8/int16) where spec §5.7.1 requires
//! result-precision = accumulator-precision = wider type.
//!
//! The lowering path (`crates/chelis-ir/src/lower.rs`) reads the
//! type-checker's output precision and uses it as the IR `accumulator`
//! field. If the type checker types `sum(int8_tensor)` as
//! `tensor[..., int8]` instead of `tensor[..., int32]` per spec, the
//! lowered IR will have `accumulator: int8` and IR-verify will reject
//! it with the §5.7.1 narrowness diagnostic. That's the symptom of the
//! type-checker spec divergence pinned in
//! crates/chelis-types/tests/numeric_dtype_buildout_adversarial.rs.

use chelis_ir::lower::lower_program;
use chelis_ir::verify;
use chelis_types::check_ir_program;

fn lower_surf(src: &str) -> Result<chelis_ir::dag::Dag, String> {
    let decls = chelis_surf::parser::parse_str(src).map_err(|e| format!("parse: {e:?}"))?;
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .map_err(|e| format!("expand: {e:?}"))?
    .into_exprs();
    let checked = check_ir_program(&exprs).map_err(|r| {
        format!(
            "check: {:?}",
            r.errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
        )
    })?;
    let checked = chelis_effects::check_program(&checked).map_err(|e| format!("effects: {e:?}"))?;
    let checked =
        chelis_types::check_linearity(&checked).map_err(|e| format!("linearity: {e:?}"))?;
    Ok(lower_program(&checked))
}

/// CRITICAL: end-to-end smoking gun. A program that passes type-check
/// but the lowered IR fails verify because the type-checker types
/// sum(int8) as int8 (operand precision), then lowering propagates
/// accumulator=int8, then verify catches the §5.7.1 narrowness
/// violation. The user sees an internal-IR error for what should be a
/// type error or a successful int32 result.
#[test]
fn end_to_end_sum_int8_lowering_produces_verify_failure() {
    // The user-visible source is innocuous: sum a tensor of int8
    // suffixed literals.
    let src = r#"
        xs: tensor[3, int8] = [1i8, 2i8, 3i8]
        out: tensor[int8] = sum(&xs, 0)
    "#;
    let result = lower_surf(src);
    match result {
        Ok(dag) => {
            let errs = verify::verify(&dag);
            assert!(
                !errs.is_empty(),
                "expected EITHER a type-check error or an IR-verify error from \
                 sum(int8); got neither (silent-data-loss class)"
            );
            assert!(
                errs.iter().any(|e| e.contains("§5.7.1")),
                "verify error must cite §5.7.1; got: {errs:?}"
            );
            // Symptom: the type checker accepted sum(int8) -> int8, then
            // lowering propagated accumulator=int8, then verify caught
            // the spec violation. The user-facing diagnostic should be
            // a type error per §5.7.1, not an IR-verify message.
            panic!(
                "spec divergence symptom: program lowers but IR verify \
                 rejects. The type checker should have rejected this OR \
                 should have widened the result to int32. errs={errs:?}"
            );
        }
        Err(msg) => {
            // The program failed earlier in the pipeline. Whether the
            // failure is a type-check error or an effects/linearity
            // error depends on the implementation; spec says it must
            // be a type error.
            assert!(
                msg.contains("§5.7.1") || msg.contains("int32") || msg.contains("accumulator"),
                "rejection must cite §5.7.1 or mention int32 widening; got: {msg}"
            );
        }
    }
}

/// Same shape for int16: sum(int16) must error with a §5.7.1 widening
/// hint, not an internal-IR diagnostic.
#[test]
fn end_to_end_sum_int16_lowering_produces_verify_failure() {
    let src = r#"
        xs: tensor[3, int16] = [1i16, 2i16, 3i16]
        out: tensor[int16] = sum(&xs, 0)
    "#;
    let result = lower_surf(src);
    match result {
        Ok(dag) => {
            let errs = verify::verify(&dag);
            assert!(
                !errs.is_empty(),
                "expected an error from sum(int16) -> int16 binding"
            );
            panic!(
                "spec divergence symptom: int16 sum lowers but IR verify \
                 rejects. errs={errs:?}"
            );
        }
        Err(msg) => {
            assert!(
                msg.contains("§5.7.1") || msg.contains("int32") || msg.contains("accumulator"),
                "rejection must cite §5.7.1 or mention int32 widening; got: {msg}"
            );
        }
    }
}
