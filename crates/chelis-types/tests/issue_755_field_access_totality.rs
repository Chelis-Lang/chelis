//! Chelis-Lang/chelis#755 — Surf-reachable false green: field access on a
//! multi-variant / positional-field / non-record target silently typed as
//! `Type::Error` with NO diagnostic, so `chelis check` scored a perfect 1.0
//! on programs that could not run (they failed only at runtime). This is the
//! chelis#709/#710 silent-`Type::Error` mechanism reached from ordinary Surf.
//!
//! chelis#731 Phase 2 closes it: `infer_access`'s "conservative status quo
//! (silently untyped)" arms become explicit rejections with diagnostics (the
//! sanctioned direction from chelis#755), and the deferred-`Var` arm returns a
//! FRESH type variable (an explicit typed rule) instead of a silent
//! `Type::Error`. Every silent site is now either a `report(...)` or a
//! non-error type, so the §C4.1 totality invariant holds and the program is
//! rejected at `check` instead of exempted.
//!
//! Both polarities per the repo negative-test-parity rule: field access on a
//! single-record variant is ACCEPTED; on a multi-variant ADT and on a scalar
//! it is REJECTED with the right diagnostic.
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

// ── Positive polarity: field access on a single-record variant is typed ──────

#[test]
fn field_access_on_single_record_variant_is_accepted() {
    accept(
        "type Point = | Point { x: f32, y: f32 }\n\
         def get_x(p: Point) -> f32 = p.x\n",
        "single-record field access",
    );
}

// ── Negative polarity: the two chelis#755 Surf repros are rejected ───────────

#[test]
fn field_access_on_multi_variant_adt_is_rejected() {
    // The exact chelis#755 repro: `.radius` on a two-variant `Shape`.
    let msgs = reject_messages(
        "type Shape = | Circle(f32) | Square(f32)\n\
         def f(s: Shape) -> f32 = s.radius\n",
        "multi-variant field access",
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("single-record-variant") && m.contains("755")),
        "expected the chelis#755 multi-variant diagnostic, got: {msgs:?}"
    );
}

#[test]
fn field_access_on_scalar_is_rejected() {
    // The second chelis#755 repro: `.field` on a scalar `f32`.
    let msgs = reject_messages("def f(x: f32) -> f32 = x.field\n", "scalar field access");
    assert!(
        msgs.iter()
            .any(|m| m.contains("record value") && m.contains("755")),
        "expected the chelis#755 non-record diagnostic, got: {msgs:?}"
    );
}
