//! Chelis-Lang/chelis#755 -- Surf-reachable false green: field access on a
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

use chelis_types::{InferResult, check_ir_program};

fn parse_core(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("valid core fixture")
}

fn reject_messages(source: &str, label: &str) -> Vec<String> {
    let deep = parse_core(source);
    let rep: InferResult = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .err()
        .unwrap_or_else(|| panic!("{label}: expected a check rejection, but it passed"));
    assert!(
        !rep.errors.is_empty(),
        "{label}: a rejection must carry at least one diagnostic"
    );
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

fn accept(source: &str, label: &str) {
    let deep = parse_core(source);
    if let Err(rep) = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked) {
        let msgs: Vec<String> = rep.errors.iter().map(|e| e.message.clone()).collect();
        panic!("{label}: expected a clean check, got errors: {msgs:?}");
    }
}

// ── Positive polarity: field access on a single-record variant is typed ──────

#[test]
fn field_access_on_single_record_variant_is_accepted() {
    accept(
        include_str!(
            "fixtures/core/tests/issue_755_field_access_totality/field_access_on_single_record_variant_is_accepted.0.deep"
        ),
        "single-record field access",
    );
}

// ── Negative polarity: the two chelis#755 Surf repros are rejected ───────────

#[test]
fn field_access_on_multi_variant_adt_is_rejected() {
    // The exact chelis#755 repro: `.radius` on a two-variant `Shape`.
    let msgs = reject_messages(
        include_str!(
            "fixtures/core/tests/issue_755_field_access_totality/field_access_on_multi_variant_adt_is_rejected.0.deep"
        ),
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
    let msgs = reject_messages(
        include_str!(
            "fixtures/core/tests/issue_755_field_access_totality/field_access_on_scalar_is_rejected.0.deep"
        ),
        "scalar field access",
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("record value") && m.contains("755")),
        "expected the chelis#755 non-record diagnostic, got: {msgs:?}"
    );
}
