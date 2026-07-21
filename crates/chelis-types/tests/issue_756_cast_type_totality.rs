//! Chelis-Lang/chelis#756 -- Surf-reachable false green: `cast` to an unknown
//! type name scored a perfect 1.0 with an empty error list, because the
//! deep-type conversion family (`deep_type_to_type_inner`,
//! `deep_type_to_type_with_params`) reduced every malformed or unknown type
//! expression to a bare `Type::Error` with no diagnostic and no error-vector
//! access. The same silence fed declared-signature parsing, so a whole family
//! of malformed type expressions was silently untyped.
//!
//! chelis#731 Phase 2 closes it: the converter family is threaded with the
//! error vector and reports the malformed / unknown type (spec/design/
//! checker_totality.md §C3 -- "forces these converters to either take the
//! error vector or return a witness-carrying error"), and `infer_cast`
//! surfaces the rejection at the use site. A `cast` to an unrecognized type is
//! now rejected at `check` instead of exempted from checking.
//!
//! Both polarities: a cast to a recognized primitive is ACCEPTED; a cast to an
//! unknown type name is REJECTED, naming the offending target.
//!
//! Spec authority: spec/04-type-system.md §10 [04-TOT-2];
//! spec/design/checker_totality.md §C3.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

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

// ── Positive polarity: a cast to a recognized primitive is typed ─────────────

#[test]
fn cast_to_recognized_primitive_is_accepted() {
    accept("def f() -> f32 = cast(1.0, f32)\n", "cast to f32");
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
