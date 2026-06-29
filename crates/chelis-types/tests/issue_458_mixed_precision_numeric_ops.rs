//! chelis#458 regression lock — mixed-precision numeric ops reject
//! CONSISTENTLY whether the operand is a literal or a bound variable.
//!
//! Background: on chelis 0.9.0 a numeric op over a mixed `(f32, i32)`
//! pair *const-folded when both operands were literals* (`div(1.0, 4)`
//! → 0.25) but *errored when one operand was a bound variable*
//! (`div(1.0, n)` with `n : i32`). That check↔eval inconsistency was the
//! soundness defect #458 reported.
//!
//! Spec disposition (DECLINE implicit promotion):
//!   - spec/04-type-system.md §5.1: "All operands of an arithmetic
//!     operation must have the same precision. Mixed precision is a type
//!     error." (fix: `cast(...)`)
//!   - §5.2: "Cast is always explicit. The compiler never inserts
//!     implicit casts."
//!   - §5.6: the CLOSED set of contextual element-type positions does NOT
//!     include a numeric-op operand, so an i32 operand is never silently
//!     widened to meet an f32 sibling.
//!   - §5.4: `div` is float-only.
//!
//! Therefore the spec-correct behavior is that BOTH the literal pair and
//! the bound-variable pair are a type error, and the user inserts an
//! explicit `cast(n, f32)` (which is exactly how Octant unblocks its
//! `sample_mean` / `sample_variance` / `mse_loss` averaging fixtures).
//!
//! These tests pin that both paths reject IDENTICALLY — the literal pair
//! is rejected at type-check before any const-fold, so the old 0.25 fold
//! can never reappear. No "did not panic" assertions: every test pins the
//! error kind and the operand precisions in the diagnostic.

use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

/// Parse → desugar → macro-expand → type-check, returning the
/// `(kind, message)` pairs of any diagnostics. An empty vector means the
/// program type-checked cleanly.
fn check_diagnostics(source: &str) -> Vec<(CheckErrorKind, String)> {
    let decls = parse_str(source).expect("surf parse");
    let exprs = expand_program(&desugar_program(&decls), &ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs();
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(rep) => rep
            .errors
            .into_iter()
            .map(|e| (e.kind, e.message))
            .collect(),
    }
}

fn is_precision_mismatch(diags: &[(CheckErrorKind, String)]) -> bool {
    diags
        .iter()
        .any(|(kind, _)| matches!(kind, CheckErrorKind::PrecisionMismatch))
}

/// Bound-variable path: `div(1.0, n)` with `n : i32` is a type error.
/// This was already true on 0.9.0; we pin it so the consistency fix
/// can't regress it in the *other* direction (silently promoting the
/// bound var).
#[test]
fn issue_458_div_f32_literal_over_i32_bound_var_is_precision_mismatch() {
    let diags = check_diagnostics("def recip_n(n: int32) -> f32 = div(1.0, n)");
    assert!(
        !diags.is_empty(),
        "spec §5.1/§5.2: div(f32, i32-bound-var) must be a type error \
         (no implicit i32→f32 promotion); got no diagnostics"
    );
    assert!(
        is_precision_mismatch(&diags),
        "spec §5.1: mixed (f32, int32) numeric op must be a PrecisionMismatch; got: {diags:?}"
    );
    assert!(
        diags
            .iter()
            .any(|(_, m)| m.contains("f32") && m.contains("int32")),
        "diagnostic must name both operand precisions (f32 and int32); got: {diags:?}"
    );
}

/// Literal path: `div(1.0, 4)` — the integer literal `4` is `int32`
/// (§5.3), so this is the SAME mixed `(f32, int32)` pair. It must be
/// rejected at type-check, NOT const-folded to 0.25. This is the precise
/// #458 regression: the literal and bound-variable paths must produce the
/// same diagnostic.
#[test]
fn issue_458_div_f32_literal_over_i32_literal_rejects_not_folds_to_quarter() {
    let diags = check_diagnostics("out = div(1.0, 4)");
    assert!(
        !diags.is_empty(),
        "spec §5.1: div(1.0, 4) is a mixed (f32, int32) pair and must be a type \
         error, NOT a silent 0.25 const-fold (the chelis#458 defect); got no diagnostics"
    );
    assert!(
        is_precision_mismatch(&diags),
        "spec §5.1: literal mixed (f32, int32) div must be a PrecisionMismatch; got: {diags:?}"
    );
    assert!(
        diags
            .iter()
            .any(|(_, m)| m.contains("f32") && m.contains("int32")),
        "diagnostic must name both operand precisions (f32 and int32); got: {diags:?}"
    );
}

/// Consistency lock (the soundness kernel): the literal pair and the
/// bound-variable pair must produce the SAME diagnostic message. A
/// divergence here is exactly the check↔eval gap #458 reported.
#[test]
fn issue_458_literal_and_bound_var_div_reject_identically() {
    let literal = check_diagnostics("out = div(1.0, 4)");
    let bound = check_diagnostics("def recip_n(n: int32) -> f32 = div(1.0, n)");
    let literal_msgs: Vec<&String> = literal.iter().map(|(_, m)| m).collect();
    let bound_msgs: Vec<&String> = bound.iter().map(|(_, m)| m).collect();
    assert_eq!(
        literal_msgs, bound_msgs,
        "chelis#458: the literal pair div(1.0, 4) and the bound-var pair \
         div(1.0, n:i32) must reject with the same diagnostic; a divergence \
         is the check-vs-eval inconsistency the issue reported"
    );
}

/// The other numeric ops in the same family reject the mixed pair too.
/// `add` is the simplest representative (it is not float-only like `div`,
/// so the only reason it rejects is the §5.1 precision rule, not §5.4).
#[test]
fn issue_458_add_f32_literal_over_i32_bound_var_is_precision_mismatch() {
    let diags = check_diagnostics("def bump(n: int32) -> f32 = add(1.0, n)");
    assert!(
        is_precision_mismatch(&diags),
        "spec §5.1: add(f32, int32) must be a PrecisionMismatch (no implicit \
         promotion); got: {diags:?}"
    );
}

/// Positive: the spec-sanctioned fix — an explicit `cast(n, f32)` —
/// type-checks cleanly. This is exactly the rewrite Octant applies to
/// unblock its `\frac{1}{n}` averaging fixtures, confirming the
/// implicit-promotion ask in #458 is correctly DECLINED rather than
/// blocking those programs outright.
#[test]
fn issue_458_explicit_cast_n_to_f32_typechecks() {
    let diags = check_diagnostics("def recip_n(n: int32) -> f32 = div(1.0, cast(n, f32))");
    assert!(
        diags.is_empty(),
        "the explicit-cast fix div(1.0, cast(n, f32)) must type-check cleanly \
         (the §5.2 escape hatch for the declined implicit promotion); got: {diags:?}"
    );
}
