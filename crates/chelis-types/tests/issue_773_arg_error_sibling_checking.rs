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
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

/// Reject-path helper: `check_ir_program` must `Err`; returns its
/// diagnostic messages for exact-count / exact-content assertions.
fn reject_messages(source: &str, label: &str) -> Vec<String> {
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep)
        .err()
        .unwrap_or_else(|| panic!("{label}: expected a check rejection, but it passed"));
    messages(&rep)
}

/// Two distinct same-shaped nominal records plus a two-arg function whose
/// slots have distinct nominal types, so a sibling in the wrong slot is a
/// genuine (not incidental) mismatch.
const PRELUDE: &str = "\
type P =
  | P { v: tensor[2, f32] }
type V =
  | V { v: tensor[2, f32] }
def take_two(a: P, b: V) -> f32 = cast(0.0, f32)
";

fn src(body: &str) -> String {
    format!("{PRELUDE}{body}")
}

// ─── (A) unbound arg + CORRECT sibling: no flood ─────────────────────────

#[test]
fn unbound_arg_with_correct_sibling_reports_only_the_unbound() {
    // `take_two(missing_p, vv)` — slot `a` is the unbound name (already an
    // error), slot `b` is a correctly-typed `V`. The de-poison must NOT
    // manufacture a spurious mismatch on the correct sibling: exactly the
    // one unbound-variable diagnostic, identical to pre-fix behavior.
    let msgs = reject_messages(
        &src("\
def driver() -> f32 = {
  vv = V { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  sink = take_two(missing_p, vv)
  cast(0.0, f32)
}
"),
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
        &src("\
def driver() -> f32 = {
  pp = P { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  sink = take_two(missing_p, pp)
  cast(0.0, f32)
}
"),
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
        &src("\
def use_p(x: P) -> f32 = cast(0.0, f32)
def driver() -> f32 = {
  pair = (missing_val, cast(1.0, f32))
  sink = use_p(pair.0)
  cast(0.0, f32)
}
"),
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
    // End-to-end companion to the structural regression lock
    // `issue_319_separate_sig_body_annotation::
    //  issue_319_separate_sig_body_carries_no_bare_tvar_app_type`
    // (that test pins "no body `app` is a bare `t-var`" node-by-node; this
    // one pins that the same separate-sig program type-checks cleanly with
    // the arg-Error short-circuit removed — the annotation-pass var-ID fix
    // that de-poisoning required). Do not duplicate the structural
    // traversal here; keep this as the coarse end-to-end guard.
    let deep = surf_to_deep(
        "\
sig sdpa[s, d, p: Float]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]
def sdpa(q, k, v, scale) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  weights = softmax(mul(scores, scale), -1)
  matmul(weights, v)
}
",
    );
    assert!(
        check_ir_program(&deep).is_ok(),
        "D: separate-sig SDPA must check clean; got {:?}",
        check_ir_program(&surf_to_deep(
            "\
sig sdpa[s, d, p: Float]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]
def sdpa(q, k, v, scale) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  weights = softmax(mul(scores, scale), -1)
  matmul(weights, v)
}
",
        ))
        .err()
        .map(|r| messages(&r)),
    );
}
