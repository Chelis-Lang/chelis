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
fn the_table_grants_no_row_and_every_operation_is_exact() {
    // chelis#2967: [05-OP-46] makes every transcendental and `sqrt` correctly
    // rounded, so [05-OBS-3] grants no tolerance and the only identity is
    // `Exact`.
    assert!(OP_TOLERANCES.is_empty());
    assert_eq!(tolerance_for(AgreementOp::Exact), 0);
}

#[test]
fn one_f32_ulp_is_a_failed_comparison() {
    let one = f32_text(1.0_f32.to_bits());
    let next = f32_text(1.0_f32.to_bits() + 1);

    assert!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F32, WIDTHS_CONFORM, &one, &one)
            .is_ok()
    );
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F32, WIDTHS_CONFORM, &one, &next),
        Err(AgreementError::UlpExceeded {
            distance_ulps: 1,
            max_ulps: 0,
            ..
        })
    ));
}

#[test]
fn one_f64_ulp_is_a_failed_comparison() {
    let one = f64_text(1.0_f64.to_bits());
    let next = f64_text(1.0_f64.to_bits() + 1);

    assert!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F64, WIDTHS_CONFORM, &one, &one)
            .is_ok()
    );
    assert!(matches!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F64, WIDTHS_CONFORM, &one, &next),
        Err(AgreementError::UlpExceeded {
            distance_ulps: 1,
            max_ulps: 0,
            ..
        })
    ));
}

#[test]
fn integer_mismatches_never_pass_through_float_parsing() {
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exact,
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
        compare_rendered_elements(AgreementOp::Exact, Prim::F32, WIDTHS_CONFORM, "1.0", "1.00",),
        Err(AgreementError::NonCanonical { .. }) | Err(AgreementError::FormattingMismatch { .. })
    ));
}

#[test]
fn non_finite_value_differences_are_never_tolerated() {
    assert!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F32, WIDTHS_CONFORM, "NaN", "NaN",)
            .is_ok()
    );
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exact,
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
        compare_rendered_elements(AgreementOp::Exact, Prim::F32, WIDTHS_CONFORM, "-0.0", "0.0",),
        Err(AgreementError::SignedZeroMismatch { .. })
    ));
}

#[test]
fn adjacent_half_storage_values_from_a_one_f32_ulp_split_fail() {
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

    assert!(matches!(
        compare_rendered_elements(AgreementOp::Exact, Prim::F16, evidence, &lower, &upper),
        Err(AgreementError::UlpExceeded {
            distance_ulps: 1,
            max_ulps: 0,
            ..
        })
    ));
}

#[test]
fn reduced_float_mismatches_require_pre_final_f32_evidence() {
    let lower = format_element(Prim::F16, ElementRef::F16(half::f16::from_bits(0x3c00)));
    let upper = format_element(Prim::F16, ElementRef::F16(half::f16::from_bits(0x3c01)));

    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exact,
            Prim::F16,
            WIDTHS_CONFORM,
            &lower,
            &upper
        ),
        Err(AgreementError::MissingReducedFloatEvidence { .. })
    ));
    assert!(matches!(
        compare_rendered_elements(
            AgreementOp::Exact,
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
            AgreementOp::Exact,
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
    assert!(eval_agreement.contains("chelis_string_from_scalar"));
    assert!(!eval_agreement.contains("chelis_format_shortest"));
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

/// chelis#2967 oracle: the generated [05-OBS-3] block is the single source of
/// every cross-lane agreement tolerance. A sentence states one when it names
/// a numeric bound (`ABS_TOL`/`REL_TOL`, a nonzero ULP count, or a scientific
/// epsilon), a lane comparison, and a grant word, and does not negate it.
/// The heuristic reads prose, so a reworded grant can evade it; the check
/// exists to catch the spellings the project has used.
fn states_agreement_tolerance(sentence: &str) -> bool {
    let tokens: Vec<String> = sentence
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_'))
        .map(|token| {
            token
                .trim_matches(|c| c == '.' || c == '-')
                .to_ascii_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect();
    let has = |words: &[&str]| tokens.iter().any(|token| words.contains(&token.as_str()));
    let nonzero_number = |token: &str| {
        token.parse::<f64>().is_ok_and(|value| value > 0.0)
            && token.chars().all(|c| c.is_ascii_digit() || c == '.')
    };
    let ulp_bound = tokens.iter().enumerate().any(|(index, token)| {
        let hyphenated = ["-ulp", "-ulps"]
            .iter()
            .any(|suffix| token.strip_suffix(suffix).is_some_and(&nonzero_number));
        let spaced = nonzero_number(token)
            && tokens
                .get(index + 1)
                .is_some_and(|next| next == "ulp" || next == "ulps");
        hyphenated || spaced
    });
    let epsilon = tokens.iter().any(|token| {
        token
            .trim_start_matches('~')
            .split_once("e-")
            .is_some_and(|(mantissa, exponent)| {
                nonzero_number(mantissa)
                    && !exponent.is_empty()
                    && exponent.chars().all(|c| c.is_ascii_digit())
            })
    });
    let named_constant = sentence.contains("ABS_TOL") || sentence.contains("REL_TOL");
    let bound = named_constant || ulp_bound || epsilon;
    let lane = has(&[
        "lane",
        "lanes",
        "cross-lane",
        "evaluator",
        "eval",
        "gpu",
        "cpu",
        "backend",
        "backends",
        "metal",
        "hip",
        "agreement",
        "agree",
        "agrees",
        "parity",
    ]);
    let grant = named_constant
        || has(&[
            "within",
            "tolerance",
            "tolerances",
            "tol",
            "permit",
            "permits",
            "permitted",
            "bound",
            "bounds",
            "margin",
        ]);
    let negated = has(&["not", "no", "never", "rejected", "cannot"])
        || tokens.iter().any(|token| token.starts_with("forbid"));
    bound && lane && grant && !negated
}

/// Sentences of `text`: blank lines, list items, and table rows separate
/// units, and each unit splits at `. ` and `; `.
fn sentences(text: &str) -> Vec<String> {
    let mut units = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        let starts_item = trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with('|')
            || trimmed.split_once(". ").is_some_and(|(number, _)| {
                number.chars().all(|c| c.is_ascii_digit()) && !number.is_empty()
            });
        if trimmed.is_empty() || starts_item {
            units.push(std::mem::take(&mut current));
        }
        current.push(' ');
        current.push_str(trimmed);
    }
    units.push(current);
    units
        .iter()
        .flat_map(|unit| {
            unit.split(". ")
                .flat_map(|part| part.split("; "))
                .map(|part| part.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
        })
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

#[test]
fn tolerance_statement_detector_separates_grants_from_measurements() {
    for grant in [
        "All unary ops: GPU matches CPU (within 1e-5 for f32)",
        "Every op path matches the evaluator within 1e-4",
        "`[05-OBS-3]` permits a cross-lane value difference of 1 ULP for `exp`",
        "eval and C agree within 2 ULPs",
        "The Metal lane compares under ABS_TOL",
        "the Metal binary must match `chelis eval` within the dtype's tolerance (f32 ~1e-6 relative)",
    ] {
        assert!(states_agreement_tolerance(grant), "missed: {grant}");
    }
    for other in [
        "up to 32 ULP between `chelis eval` and `chelis build` for `tanh` at f32",
        "Blanket `1e-6` (f32) bounds are not a conforming cross-lane oracle",
        "Operations absent from the table have a zero-ULP bound",
        "`layer_norm` takes an explicit epsilon operand of 1e-5",
        "eval and C agree within 0 ULP",
    ] {
        assert!(!states_agreement_tolerance(other), "flagged: {other}");
    }
    assert_eq!(
        sentences("- GPU matches CPU (within 1e-5)\n- add is exact\n\nOne. Two; three"),
        vec![
            "- GPU matches CPU (within 1e-5)",
            "- add is exact",
            "One",
            "Two",
            "three"
        ]
    );
}

#[test]
fn no_numbered_chapter_or_design_doc_states_an_agreement_tolerance() {
    let spec = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec");
    let mut documents = Vec::new();
    for dir in ["", "registry", "design"] {
        for entry in std::fs::read_dir(spec.join(dir)).expect("spec directory") {
            let path = entry.expect("spec entry").path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let numbered = dir.is_empty()
                && name
                    .as_bytes()
                    .get(..2)
                    .is_some_and(|prefix| prefix.iter().all(u8::is_ascii_digit));
            if path.extension().is_some_and(|ext| ext == "md") && (numbered || !dir.is_empty()) {
                documents.push(path);
            }
        }
    }
    documents.sort();
    assert!(
        documents
            .iter()
            .any(|path| path.ends_with("05-risc-primitives.md")),
        "the scan must reach the [05-OBS-3] chapter"
    );
    let begin = "<!-- BEGIN GENERATED OBSERVATION TOLERANCE TABLE -->";
    let end = "<!-- END GENERATED OBSERVATION TOLERANCE TABLE -->";
    let mut violations = Vec::new();
    for path in &documents {
        let mut text = std::fs::read_to_string(path).expect("spec document");
        if let (Some(start), Some(stop)) = (text.find(begin), text.find(end)) {
            text.replace_range(start..stop, "");
        }
        for sentence in sentences(&text) {
            if states_agreement_tolerance(&sentence) {
                violations.push(format!("{}: {sentence}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "[05-OBS-3]'s generated block is the only place a cross-lane agreement tolerance may be \
         stated:\n{}",
        violations.join("\n")
    );
}
