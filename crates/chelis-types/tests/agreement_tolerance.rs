//! chelis#732 Phase 3 / chelis#687: executable contract for the one
//! byte-equal-or-table-bounded numerical comparator.

use chelis_types::agreement::{
    AgreementError, AgreementOp, AgreementOutcome, ArithmeticWidthStatus, OP_TOLERANCES,
    compare_exact_observations, compare_rendered_elements, render_spec_tolerance_table,
    tolerance_for,
};
use chelis_types::types::Prim;
use chelis_types::{ElementRef, format_element};

fn f32_text(bits: u32) -> String {
    format_element(Prim::F32, ElementRef::F32(f32::from_bits(bits)))
}

fn f64_text(bits: u64) -> String {
    format_element(Prim::F64, ElementRef::F64(f64::from_bits(bits)))
}

const WIDTHS_CONFORM: ArithmeticWidthStatus = ArithmeticWidthStatus::StoredAtArithmeticWidth;

#[test]
fn phase3_table_has_one_row_per_transcendental_and_no_blanket_fallback() {
    let rows: Vec<(AgreementOp, u64)> = OP_TOLERANCES
        .iter()
        .map(|row| (row.op, row.max_ulps))
        .collect();
    assert_eq!(
        rows,
        vec![
            (AgreementOp::Atan, 1),
            (AgreementOp::Cos, 1),
            (AgreementOp::Exp, 1),
            (AgreementOp::Log, 1),
            (AgreementOp::Sin, 1),
            (AgreementOp::Sqrt, 0),
            (AgreementOp::Tan, 1),
        ]
    );
    assert_eq!(tolerance_for(AgreementOp::Sqrt), 0);
    assert_eq!(tolerance_for(AgreementOp::Exp), 1);
    assert_eq!(tolerance_for(AgreementOp::Exact), 0);
}

#[test]
fn one_f32_ulp_is_allowed_only_for_a_named_toleranced_op() {
    let one = f32_text(1.0_f32.to_bits());
    let next = f32_text(1.0_f32.to_bits() + 1);

    assert_eq!(
        compare_rendered_elements(AgreementOp::Exp, Prim::F32, WIDTHS_CONFORM, &one, &next)
            .unwrap(),
        AgreementOutcome::WithinTolerance {
            distance_ulps: 1,
            max_ulps: 1,
        }
    );
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F32, WIDTHS_CONFORM, &one, &next),
        Err(AgreementError::UlpExceeded { max_ulps: 0, .. })
    ));
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Sqrt, Prim::F32, WIDTHS_CONFORM, &one, &next),
        Err(AgreementError::UlpExceeded { max_ulps: 0, .. })
    ));
}

#[test]
fn one_f64_ulp_passes_but_two_ulps_fail() {
    let one = f64_text(1.0_f64.to_bits());
    let next = f64_text(1.0_f64.to_bits() + 1);
    let next_next = f64_text(1.0_f64.to_bits() + 2);

    assert!(
        compare_rendered_elements(AgreementOp::Sin, Prim::F64, WIDTHS_CONFORM, &one, &next).is_ok()
    );
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Sin,
            Prim::F64,
            WIDTHS_CONFORM,
            &one,
            &next_next,
        ),
        Err(AgreementError::UlpExceeded {
            distance_ulps: 2,
            max_ulps: 1,
            ..
        })
    ));
}

#[test]
fn integer_mismatches_never_pass_through_float_parsing() {
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exp,
            Prim::Int64,
            WIDTHS_CONFORM,
            "9007199254740992",
            "9007199254740993",
        ),
        Err(AgreementError::ExactMismatch { .. })
    ));
}

#[test]
fn different_spellings_of_identical_bits_are_formatting_errors() {
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Exp, Prim::F32, WIDTHS_CONFORM, "1.0", "1.00",),
        Err(AgreementError::NonCanonical { .. }) | Err(AgreementError::FormattingMismatch { .. })
    ));
}

#[test]
fn non_finite_value_differences_are_never_tolerated() {
    assert!(
        compare_rendered_elements(AgreementOp::Exp, Prim::F32, WIDTHS_CONFORM, "NaN", "NaN",)
            .is_ok()
    );
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exp,
            Prim::F32,
            WIDTHS_CONFORM,
            "inf",
            "3.4028235e38",
        ),
        Err(AgreementError::NonFiniteMismatch { .. })
    ));
}

#[test]
fn signed_zero_differences_are_never_tolerated() {
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Sin, Prim::F32, WIDTHS_CONFORM, "-0.0", "0.0",),
        Err(AgreementError::SignedZeroMismatch { .. })
    ));
}

#[test]
fn adjacent_half_storage_values_can_witness_a_one_f32_ulp_computation_split() {
    let lower = half::f16::from_bits(0x3c00);
    let upper = half::f16::from_bits(0x3c01);
    let lower = format_element(Prim::F16, ElementRef::F16(lower));
    let upper = format_element(Prim::F16, ElementRef::F16(upper));
    let midpoint = f32::from_bits(1.0_f32.to_bits() + (1 << 12));
    let next = f32::from_bits(midpoint.to_bits() + 1);
    assert_eq!(half::f16::from_f32(midpoint).to_bits(), 0x3c00);
    assert_eq!(half::f16::from_f32(next).to_bits(), 0x3c01);
    let evidence = ArithmeticWidthStatus::ReducedFloatPreFinal {
        eval_bits: midpoint.to_bits(),
        compiled_bits: next.to_bits(),
    };

    assert!(
        compare_rendered_elements(AgreementOp::Exp, Prim::F16, evidence, &lower, &upper).is_ok()
    );
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Sqrt, Prim::F16, evidence, &lower, &upper),
        Err(AgreementError::UlpExceeded { max_ulps: 0, .. })
    ));
}

#[test]
fn reduced_float_mismatches_require_pre_final_f32_evidence() {
    let lower = format_element(Prim::F16, ElementRef::F16(half::f16::from_bits(0x3c00)));
    let upper = format_element(Prim::F16, ElementRef::F16(half::f16::from_bits(0x3c01)));

    assert!(matches!(
        compare_rendered_elements(AgreementOp::Exp, Prim::F16, WIDTHS_CONFORM, &lower, &upper),
        Err(AgreementError::MissingReducedFloatEvidence { .. })
    ));
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exp,
            Prim::F16,
            ArithmeticWidthStatus::ReducedFloatPreFinal {
                eval_bits: 1.0_f32.to_bits(),
                compiled_bits: 1.0_f32.to_bits() + 1,
            },
            &lower,
            &upper,
        ),
        Err(AgreementError::ReducedFloatEvidenceMismatch { .. })
    ));
}

#[test]
fn a_tolerance_cannot_hide_a_known_arithmetic_width_violation() {
    let one = f32_text(1.0_f32.to_bits());
    let next = f32_text(1.0_f32.to_bits() + 1);

    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exp,
            Prim::F32,
            ArithmeticWidthStatus::Nonconforming { issue: 897 },
            &one,
            &next,
        ),
        Err(AgreementError::ArithmeticWidthNonconforming { issue: 897, .. })
    ));
}

#[test]
fn exact_diagnostics_and_values_use_the_same_comparator_entrypoint() {
    assert_eq!(
        compare_exact_observations("unsupported diagnostic", "unsupported: x", "unsupported: x")
            .unwrap(),
        AgreementOutcome::ByteExact
    );
    assert!(matches!(
        compare_exact_observations("unsupported diagnostic", "unsupported: x", "unsupported: y"),
        Err(AgreementError::ExactMismatch { .. })
    ));
}

#[test]
fn numbered_spec_table_is_generated_from_the_machine_rows() {
    let spec = include_str!("../../../spec/05-risc-primitives.md");
    let begin = "<!-- BEGIN GENERATED OBSERVATION TOLERANCE TABLE -->";
    let end = "<!-- END GENERATED OBSERVATION TOLERANCE TABLE -->";
    let (_, after_begin) = spec.split_once(begin).expect("spec table begin marker");
    let (actual, _) = after_begin.split_once(end).expect("spec table end marker");
    assert_eq!(actual.trim(), render_spec_tolerance_table().trim());
}

#[test]
fn phase3_oracles_use_the_shared_comparator_without_f64_fallbacks() {
    let parity = include_str!("../../chelis-cli/tests/parity.rs");
    let eval_agreement = include_str!("../../chelis-e2e/tests/eval_agreement.rs");
    let rejected = include_str!("../../chelis-cli/tests/issue_687_rejected_cells_corpus.rs");

    assert!(parity.contains("compare_exact_observations"));
    assert!(eval_agreement.contains("compare_rendered_elements"));
    assert!(eval_agreement.contains("ArithmeticWidthStatus::Nonconforming { issue: 897 }"));
    assert!(eval_agreement.contains("chelis_format_shortest"));
    assert!(rejected.contains("compare_exact_observations"));

    for forbidden in [
        "fn eval_last(dag: &Dag) -> f64",
        "fn parse_c_output(output: &str) -> f64",
        "fn assert_close(a: f64, b: f64, tol: f64",
        "printf(\"%.6f\"",
        "printf(\"%.8f",
    ] {
        assert!(
            !eval_agreement.contains(forbidden),
            "Phase 3 forbids the old f64/tolerance oracle path: {forbidden}"
        );
    }
}
