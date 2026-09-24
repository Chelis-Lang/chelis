//! Issue Chelis-Lang/chelis#773 — `infer_app` short-circuited the WHOLE
//! call to `Type::Error` the moment any argument inferred to `Type::Error`
//! (an already-reported error such as an unbound variable or a projection
//! off an error target). That collapse ran BEFORE the per-slot arg/param
//! unification, so every SIBLING argument in the same call skipped type
//! checking: one reported error in one slot masked genuine mismatches in
//! the others.
//!
//! Fix: only the *callee* being `Error` short-circuits. Error-typed
//! *arguments* flow into the per-slot `unify`, which treats `Type::Error`
//! as permissive (`unify.rs`: `(Error, _) | (_, Error) => Ok(())`), so
//! siblings still check and the call returns the callee's resolved return
//! type instead of `Error`.
//!
//! Deliberate behavior change pinned here: a call carrying one
//! reported-error argument can now ALSO surface a genuine sibling mismatch
//! (1 -> 2 diagnostics for that shape). The cascade controls confirm the
//! change does not flood correct siblings and does not re-poison the #772
//! tuple-projection fix.
//!
//! Each fixture binds the exercised call to an unused name and returns a
//! clean literal, so the call's result never sits in the def's tail
//! position. This isolates the #773 sibling-checking effect from the
//! pre-existing, unrelated "body doesn't match declared signature" cascade
//! that ANY call returning `Type::Error` triggers in tail position (a
//! genuinely-mismatched call — with or without #773 — returns `Error` from
//! `infer_app`'s failed arg/param `unify`). Measured baselines with these
//! shapes: (A) 1 diagnostic pre-fix and post-fix; (B) 1 pre-fix -> 2
//! post-fix (the de-mask); (C) 1 pre-fix and post-fix.
//!
//! Spec authority: spec/04-type-system.md (application typing; `Error`
//! permissive unification).

use chelis_deep::Expr;
use chelis_deep::parser::parse_str;
use chelis_types::{InferResult, check_ir_program};

fn core_fixture(source: &str) -> Vec<Expr> {
    parse_str(source).expect("core fixture")
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

/// Reject-path helper: `check_ir_program` must `Err`; returns its
/// diagnostic messages for exact-count / exact-content assertions.
fn reject_messages(source: &str, label: &str) -> Vec<String> {
    let deep = core_fixture(source);
    let rep = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .err()
        .unwrap_or_else(|| panic!("{label}: expected a check rejection, but it passed"));
    messages(&rep)
}

// ─── (A) unbound arg + CORRECT sibling: no flood ─────────────────────────

#[test]
fn unbound_arg_with_correct_sibling_reports_only_the_unbound() {
    // `take_two(missing_p, vv)` — slot `a` is the unbound name (already an
    // error), slot `b` is a correctly-typed `V`. The de-poison must NOT
    // manufacture a spurious mismatch on the correct sibling: exactly the
    // one unbound-variable diagnostic, identical to pre-fix behavior.
    let msgs = reject_messages(
        include_str!(
            "fixtures/core/tests/issue_773_arg_error_sibling_checking/unbound_arg_with_correct_sibling_reports_only_the_unbound.deep"
        ),
        "A: unbound + correct sibling",
    );
    assert_eq!(
        msgs.len(),
        1,
        "A: expected exactly one diagnostic (the unbound var), got {msgs:?}",
    );
    assert!(
        msgs[0].contains("unbound variable") && msgs[0].contains("missing_p"),
        "A: the single diagnostic must be the unbound-variable error; got {msgs:?}",
    );
}

// ─── (B) unbound arg + WRONG sibling: the intended de-mask ────────────────

#[test]
fn unbound_arg_no_longer_masks_a_wrong_sibling() {
    // `take_two(missing_p, gg)` — slot `a` is the unbound name, slot `b`
    // gets a `P` value where a `V` is required. Pre-fix the arg-Error
    // short-circuit collapsed the call and the `P`/`V` mismatch was NEVER
    // reported (1 diagnostic). Post-fix the sibling is checked: exactly two
    // diagnostics — the unbound var AND the nominal mismatch.
    let msgs = reject_messages(
        include_str!(
            "fixtures/core/tests/issue_773_arg_error_sibling_checking/unbound_arg_no_longer_masks_a_wrong_sibling.deep"
        ),
        "B: unbound + wrong sibling",
    );
    assert_eq!(
        msgs.len(),
        2,
        "B: expected exactly two diagnostics (unbound var + sibling mismatch), got {msgs:?}",
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("unbound variable") && m.contains("missing_p")),
        "B: one diagnostic must be the unbound-variable error; got {msgs:?}",
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("mismatch") && m.contains('P') && m.contains('V')),
        "B: the de-masked diagnostic must be the P/V sibling mismatch; got {msgs:?}",
    );
}

// ─── (C) error-tuple target then projection: no re-poison of #772 ─────────

#[test]
fn error_tuple_projection_stays_single_diagnostic() {
    // A tuple whose element is an unbound name is an error-target; the Surf
    // `.0` projection off it (the #772 tuple-get path) feeds a downstream
    // call. The de-poison must not turn the projected `Error` into extra
    // sibling diagnostics: exactly the one unbound-variable error.
    let msgs = reject_messages(
        include_str!(
            "fixtures/core/tests/issue_773_arg_error_sibling_checking/error_tuple_projection_stays_single_diagnostic.deep"
        ),
        "C: error-tuple projection",
    );
    assert_eq!(
        msgs.len(),
        1,
        "C: expected exactly one diagnostic (the unbound var), got {msgs:?}",
    );
    assert!(
        msgs[0].contains("unbound variable") && msgs[0].contains("missing_val"),
        "C: the single diagnostic must be the unbound-variable error; got {msgs:?}",
    );
}

// ─── (D) separate-sig SDPA still checks clean (annotation-pass guard) ──────

#[test]
fn separate_sig_sdpa_checks_clean() {
    let deep = core_fixture(include_str!(
        "fixtures/core/tests/issue_773_arg_error_sibling_checking/separate_sig_sdpa_checks_clean.deep"
    ));
    check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect("separate-sig SDPA must check clean");
}
