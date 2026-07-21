//! Chelis-Lang/chelis#756 -- Surf-reachable false green: `cast` to an unknown
//! type name scored a perfect 1.0 with an empty error list, because the
//! deep-type conversion family (`deep_type_to_type_inner`,
//! `deep_type_to_type_with_params`) reduced every malformed or unknown type
//! expression to a bare `Type::Error` with no diagnostic and no error-vector
//! access. The same silence fed declared-signature parsing, so a whole family
//! of malformed type expressions was silently untyped.
//!
//! chelis#731 Phase 2 closes the Surf-reachable cast hole through the
//! centralized witnessed Deep type resolver: an unparseable target reports
//! once and returns a witnessed failure that callers must propagate. A `cast`
//! to an unrecognized type is now rejected at `check` instead of exempted from
//! checking. The repair must preserve both accepted Deep target spellings:
//! canonical `(t-prim {} name)` and the historically supported bare primitive
//! symbol.
//!
//! Both polarities: a cast to a recognized primitive is ACCEPTED; a cast to an
//! unknown type name is REJECTED, naming the offending target.
//!
//! Spec authority: spec/04-type-system.md §10 [04-TOT-2];
//! spec/design/checker_totality.md §C3.

use chelis_deep::Expr;
use chelis_deep::parser::parse_str_strict as parse_deep;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{FitnessReport, InferResult, check_ir_fitness, check_ir_program};

const ACTIVE_CAST_TARGETS: [&str; 9] = [
    "f32", "f64", "bf16", "f16", "bool", "int8", "int16", "int32", "int64",
];

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn reject_messages(source: &str, label: &str) -> Vec<String> {
    let deep = surf_to_deep(source);
    let rep: InferResult = check_ir_program(&deep)
        .err()
        .unwrap_or_else(|| panic!("{label}: expected a check rejection, but it passed"));
    assert!(
        !rep.errors.is_empty(),
        "{label}: a rejection must carry at least one diagnostic"
    );
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

fn accept(source: &str, label: &str) {
    let deep = surf_to_deep(source);
    if let Err(rep) = check_ir_program(&deep) {
        let msgs: Vec<String> = rep.errors.iter().map(|e| e.message.clone()).collect();
        panic!("{label}: expected a clean check, got errors: {msgs:?}");
    }
}

fn deep_cast_program(target: &str) -> String {
    format!(
        "(def {{}} x (lit {{type: (t-prim {{}} f32)}} 1.0))\n\
         (def {{}} y (cast {{}} (var {{}} x) {target}))\n"
    )
}

fn deep_fitness(source: &str) -> FitnessReport {
    let deep = parse_deep(source).expect("Deep cast fixture must parse");
    check_ir_fitness(&deep)
}

fn assert_deep_accepts(source: &str, label: &str) {
    let report = deep_fitness(source);
    assert!(
        report.errors.is_empty() && report.score == 1.0,
        "{label}: expected a perfect check, got {report:?}"
    );
}

// ── Positive polarity: a cast to a recognized primitive is typed ─────────────

#[test]
fn cast_to_recognized_primitive_is_accepted() {
    accept("def f() -> f32 = cast(1.0, f32)\n", "cast to f32");
}

#[test]
fn every_active_bare_deep_primitive_cast_target_is_accepted() {
    for target in ACTIVE_CAST_TARGETS {
        assert_deep_accepts(
            &deep_cast_program(target),
            &format!("bare Deep cast target `{target}`"),
        );
    }
}

#[test]
fn every_active_canonical_t_prim_cast_target_is_accepted() {
    for target in ACTIVE_CAST_TARGETS {
        assert_deep_accepts(
            &deep_cast_program(&format!("(t-prim {{}} {target})")),
            &format!("canonical Deep cast target `(t-prim {{}} {target})`"),
        );
    }
}

// ── Negative polarity: the chelis#756 Surf repro is rejected ─────────────────

#[test]
fn cast_to_unknown_type_name_is_rejected() {
    // The exact chelis#756 repro: `cast(1.0, madeup)`.
    let msgs = reject_messages("def f() -> f32 = cast(1.0, madeup)\n", "cast to madeup");
    assert!(
        msgs.iter().any(|m| m.contains("madeup")),
        "expected a diagnostic naming the unknown cast target `madeup`, got: {msgs:?}"
    );
}

#[test]
fn unknown_bare_deep_cast_target_is_rejected_loudly() {
    let report = deep_fitness(&deep_cast_program("madeup"));
    assert!(
        report.score < 1.0,
        "unknown bare target must reduce fitness below 1, got {report:?}"
    );
    assert!(
        report.errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::CastNonTensor)
                && error
                    .message
                    .contains("cast target `madeup` is not a recognized primitive type")
        }),
        "unknown bare target must produce the intended cast diagnostic, got {:?}",
        report.errors
    );
}

#[test]
fn canonical_formatter_output_for_bare_deep_cast_remains_checkable() {
    let source = deep_cast_program("f16");
    let parsed = parse_deep(&source).expect("bare-symbol Deep cast must parse");
    let formatted = print_canonical(&parsed);
    let reparsed = parse_deep(&formatted).expect("canonical formatter output must parse");
    let report = check_ir_fitness(&reparsed);
    assert!(
        report.errors.is_empty() && report.score == 1.0,
        "canonical formatter output must remain checkable, got {report:?}\n{formatted}"
    );
}
