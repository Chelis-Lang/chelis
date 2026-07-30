//! chelis#916: `reshape` with an unsuffixed shape list reported a backwards
//! precision mismatch — `precision mismatch: expected int32, got int64` —
//! naming the type the user already wrote as the one they should have used.
//!
//! Root cause: `unify` labels its *first* argument "expected"
//! (`chelis-types/src/unify.rs:449-456`), and `infer_reshape_app` passed the
//! actual type first at all three of its shape-argument sites, so the two
//! were rendered backwards.
//!
//! Fix: rather than swap the argument order — the order is inconsistent
//! across `infer.rs` as a whole, and settling that convention is a separate
//! change with a much wider message blast radius — this follows the
//! `infer_shrink_app` precedent and substitutes a purpose-built message that
//! names the remediation (an `i64` literal suffix or an explicit `cast`).
//!
//! The diagnostic kind moves from `PrecisionMismatch` to `TypeMismatch` to
//! match `infer_shrink_app`, since the substituted message no longer talks
//! about precision at all.

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

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

// ── The reported defect ─────────────────────────────────────────────

/// chelis#916's exact reproducer: an unsuffixed shape list.
#[test]
fn unsuffixed_reshape_shape_list_names_the_int64_remediation() {
    let src = "def fixed_dims(x) -> tensor[2, 2, f32] = reshape(x, [2, 2])\n";
    let deep = surf_to_deep(src);
    let rep = check_ir_program(&deep).expect_err("an int32 shape list must be rejected");
    let msgs = messages(&rep);

    let hit = msgs
        .iter()
        .find(|m| m.contains("reshape expects an int64 shape list"))
        .unwrap_or_else(|| panic!("expected the reshape shape-list diagnostic, got {msgs:?}"));

    // It must name the remediation, which is the whole point of the change.
    assert!(
        hit.contains("i64") || hit.contains("cast"),
        "message must show how to fix it; got: {hit}"
    );
    assert!(
        hit.contains("default to int32"),
        "message must explain why an unsuffixed literal is int32; got: {hit}"
    );

    // The inverted phrasing must be gone. This is the regression that
    // chelis#916 is about: a user who wrote `[2, 2]` was told to write int32.
    assert!(
        !msgs
            .iter()
            .any(|m| m.contains("precision mismatch: expected int32, got int64")),
        "the inverted precision-mismatch message must not survive; got {msgs:?}"
    );
}

/// The spec's own example (`spec/04-type-system.md` §4.7.5) — a mixed list
/// whose first element is a bare `shape(x, 0)`.
///
/// This is the case that proves §4.7.5's stated *mechanism* wrong, not just
/// its transcribed message. The spec says "the first list element fixes the
/// 'expected' precision, and later entries that disagree trip the mismatch",
/// which would make this a same-precision list (`shape(x, 0)` is int32, the
/// literal `4` is int32) and therefore clean. It is not: the error comes from
/// the *outer* `unify(&shape_ty, &List[Int64])`, so an internally consistent
/// int32 list still fails — exactly as `[2, 2]` does.
///
/// Without this test only `[2, 2]` is covered, and a fix that moved just the
/// homogeneous-literal case would look complete while leaving the spec's own
/// example on the old message.
#[test]
fn spec_example_mixed_shape_list_names_the_int64_remediation() {
    let src = "def rows(x) -> tensor[2, 4, f32] = reshape(x, [shape(x, 0), 4])\n";
    let deep = surf_to_deep(src);
    let rep = check_ir_program(&deep).expect_err("a bare shape(x, 0) list must be rejected");
    let msgs = messages(&rep);

    assert!(
        msgs.iter()
            .any(|m| m.contains("reshape expects an int64 shape list")),
        "the spec's own §4.7.5 example must reach the new diagnostic, got {msgs:?}"
    );
    assert!(
        !msgs
            .iter()
            .any(|m| m.contains("precision mismatch: expected int32, got int64")),
        "the inverted precision-mismatch message must not survive here either; got {msgs:?}"
    );
}

/// The remediation stays in the structured `suggestions` field rather than
/// migrating into prose (chelis#886's direction). The pre-fix path attached
/// `["Insert explicit cast"]` via `From<TypeError>`; emptying it would be a
/// silent wire regression that no message assertion catches.
#[test]
fn the_remediation_stays_in_the_structured_suggestions_field() {
    let src = "def fixed_dims(x) -> tensor[2, 2, f32] = reshape(x, [2, 2])\n";
    let deep = surf_to_deep(src);
    let rep = check_ir_program(&deep).expect_err("an int32 shape list must be rejected");

    let err = rep
        .errors
        .iter()
        .find(|e| e.message.contains("reshape expects an int64 shape list"))
        .expect("the reshape shape-list diagnostic");

    assert!(
        !err.suggestions.is_empty(),
        "suggestions must not be empty: it is a typed field on the compiler-api \
         wire and the pre-fix path populated it"
    );
    assert!(
        err.suggestions.iter().any(|s| s.contains("i64")),
        "the suffix fix must be offered structurally; got {:?}",
        err.suggestions
    );
    assert!(
        err.suggestions.iter().any(|s| s.contains("cast")),
        "the cast fix must survive from the pre-fix path; got {:?}",
        err.suggestions
    );
}

// ── Controls: the forms that already worked must keep working ───────

/// An `i64`-suffixed shape list is the documented fix and must check clean.
#[test]
fn i64_suffixed_shape_list_checks_clean() {
    let src = "def fixed_dims(x) -> tensor[2, 2, f32] = reshape(x, [2i64, 2i64])\n";
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("an i64-suffixed shape list must check clean");
}

/// The `cast(_, int64)` spelling used by the example corpus
/// (`examples/illustrative/runtime_shape_semantics.ch`) must keep working.
#[test]
fn cast_int64_shape_list_checks_clean() {
    let src =
        "def fixed_dims(x) -> tensor[2, 2, f32] = reshape(x, [cast(2, int64), cast(2, int64)])\n";
    let deep = surf_to_deep(src);
    check_ir_program(&deep).expect("a cast-to-int64 shape list must check clean");
}

/// `reshape` without a shape argument is a different arm of
/// `infer_reshape_app` and must be untouched by the shape-slot change.
#[test]
fn reshape_without_shape_argument_is_unaffected() {
    let src = "def flat(x) -> tensor[4, f32] = reshape(x)\n";
    let deep = surf_to_deep(src);
    // Whether this checks clean is not the point — the point is that it does
    // not acquire the shape-list diagnostic, which only applies when a shape
    // argument is present.
    if let Err(rep) = check_ir_program(&deep) {
        assert!(
            !messages(&rep)
                .iter()
                .any(|m| m.contains("reshape expects an int64 shape list")),
            "the shape-slot diagnostic must not fire without a shape argument"
        );
    }
}

/// `shrink`'s neighbouring bounds diagnostic is the precedent this fix
/// follows; it must not be perturbed.
#[test]
fn shrink_bounds_diagnostic_is_unchanged() {
    let src = "def s(x) -> tensor[1, f32] = shrink(x, [[0i64, 1i64]])\n";
    let deep = surf_to_deep(src);
    if let Err(rep) = check_ir_program(&deep) {
        assert!(
            !messages(&rep)
                .iter()
                .any(|m| m.contains("reshape expects an int64 shape list")),
            "the reshape diagnostic must not leak into the shrink path"
        );
    }
}
