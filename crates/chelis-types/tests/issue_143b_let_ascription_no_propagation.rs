//! Probe for chelis#143 sub-issue (B): let-binding ascription is not
//! propagated to a generic-builtin RHS during inference.
//!
//! Background
//! ----------
//! Writing `q: tensor[N, f32] = to_tensor(...)` SHOULD unify
//! `to_tensor`'s output type with the declared `tensor[N, f32]`,
//! binding `q`'s type to `tensor[Lit(N), f32]`. It doesn't, today.
//!
//! Path comparison:
//!
//!  * **Parameter ascription** (works). `def f(q: &tensor[1, 8, f32]) -> ...`
//!    is handled by `infer_def_body_with_sig` in
//!    `crates/chelis-types/src/infer.rs:10542-10641`, which seeds bare
//!    parameters with declared sig types (lines ~10607-10612) before
//!    body inference. The body's `q` then carries `&tensor[Lit(1), Lit(8), f32]`,
//!    and a downstream sig-call's shared dim var unifies correctly.
//!
//!  * **Let-binding ascription** (broken). The desugarer at
//!    `crates/chelis-surf/src/desugar.rs:664-682` emits the ascription
//!    type as a `defsig` next to the `def`, so the ascription survives.
//!    But `infer_let` at `crates/chelis-types/src/infer.rs:10708-10760`
//!    infers the RHS via `infer_expr` with no ascription input, and
//!    binds the generalized result without unifying against any
//!    declared type. The RHS's fresh `Type::Var(output)` stays
//!    unconstrained.
//!
//! Why this matters for chelis#143: the natural workaround for
//! sub-issue (A) — "ascribe the let-binding so the dims are pinned" —
//! also doesn't work. Both paths fail to give the sig var a concrete
//! `Lit` to bind against.
//!
//! What this test asserts
//! ----------------------
//! The probe (`#[ignore]`) is the desired post-fix behavior: writing
//! `q: &tensor[3, f32] = to_tensor(...)` followed by a call to a
//! shared-dim sig with a `tensor[5, f32]` second arg should trip
//! `DimensionMismatch`. Today it does not.
//!
//! The two passing tests are counter-probes: the parameter-ascription
//! path (already documented in
//! `issue_143a_to_tensor_shape_erasure.rs`) and a `to_tensor` literal
//! whose dims are determined by a CALL-SITE annotation in a known
//! contextual-typing slot — these isolate the sub-bug to "let-binding
//! ascription path doesn't propagate," not "the type system as a whole
//! can't handle ascription."
//!
//! Tracking: chelis#159 (sub-issue (B) of chelis#143)
//! Diagnosis: docs/investigations/issue_143b_let_ascription_no_propagation_diagnosis.md

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn typecheck_surf(source: &str) -> Vec<CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(prog_errors) => prog_errors.errors,
    }
}

fn has_dimension_mismatch(errors: &[CheckError]) -> bool {
    errors
        .iter()
        .any(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch))
}

fn errors_summary(errors: &[CheckError]) -> String {
    if errors.is_empty() {
        "(no errors)".to_string()
    } else {
        errors
            .iter()
            .map(|e| format!("[{:?}] {}", e.kind, e.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[test]
fn let_binding_ascription_propagates_to_to_tensor_rhs() {
    // Desired post-fix behavior: the ascription `a: tensor[3, f32]`
    // should unify with the RHS `to_tensor([1.0, 2.0, 3.0], f32)`, binding
    // `a` to `tensor[Lit(3), f32]`. The follow-on `pair_id(&a, &b)`
    // then unifies sig's `n` against `Lit(3)`, then `Lit(3)` vs
    // `Lit(5)` from `b` → DimensionMismatch.
    //
    // Today: `infer_let` ignores the ascription, both `a` and `b`
    // carry `tensor[Wildcard, f32]` (or a free Type::Var that resolves
    // to one), and the sig var `n` stays free.
    let errors = typecheck_surf(
        r#"
sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller() -> tensor[3, f32] =
  {
    a: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0], f32)
    b: tensor[5, f32] = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], f32)
    pair_id(&a, &b)
  }
"#,
    );
    assert!(
        has_dimension_mismatch(&errors),
        "expected DimensionMismatch - ascriptions should propagate to to_tensor RHS, \
         allowing sig `n` to bind to 3 from `a` and conflict with 5 from `b`; \
         got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn parameter_ascription_path_works_today() {
    // Counter-probe: the parameter-ascription path is fully functional.
    // This rules out "the type system can't handle ascription at all"
    // and isolates the bug above to the let-binding path specifically.
    let errors = typecheck_surf(
        r#"
sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller(a: &tensor[3, f32], b: &tensor[5, f32]) -> tensor[3, f32] =
  pair_id(a, b)
"#,
    );
    assert!(
        has_dimension_mismatch(&errors),
        "parameter-ascription path should trip DimensionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_binding_ascription_with_matched_shapes_does_not_spuriously_fail() {
    // Sanity: even with the bug present, a let-binding ascription
    // whose RHS dims would match the sig must not produce a spurious
    // DimensionMismatch. Pin this so a future fix doesn't over-correct
    // and introduce false positives for the matched-shape case.
    let errors = typecheck_surf(
        r#"
sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller() -> tensor[3, f32] =
  {
    a: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0], f32)
    b: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0], f32)
    pair_id(&a, &b)
  }
"#,
    );
    assert!(
        !has_dimension_mismatch(&errors),
        "matched-shape let-binding ascriptions must not spuriously trip \
         DimensionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}

// ── Direct diagnostic-template tests (review comment 1) ─────────────
//
// The three tests above validate the patch indirectly via downstream
// shared-sig-dim unification. The tests below exercise the patch's
// most direct property: a let-binding whose declared type literally
// disagrees with the inferred RHS type produces a clean diagnostic
// with the right `CheckErrorKind` AND the chelis#159 substring
// template.

fn errors_contain_kind_and_message(
    errors: &[CheckError],
    kind: &CheckErrorKind,
    msg_substring: &str,
) -> bool {
    errors.iter().any(|e| {
        std::mem::discriminant(&e.kind) == std::mem::discriminant(kind)
            && e.message.contains(msg_substring)
    })
}

#[test]
fn let_ascription_dim_mismatch_reports_dimension_mismatch_with_template() {
    // Direct dim mismatch: declared `tensor[3, f32]`, RHS has concrete
    // dim 2 (via a parameter ascription). Must report
    // DimensionMismatch with the chelis#159 substring template.
    let errors = typecheck_surf(
        r#"
def caller(t: &tensor[2, f32]) -> &tensor[2, f32] =
  {
    x: &tensor[3, f32] = t
    x
  }
"#,
    );
    assert!(
        errors_contain_kind_and_message(
            &errors,
            &CheckErrorKind::DimensionMismatch,
            "let-binding `x` ascription does not match RHS",
        ),
        "expected DimensionMismatch with the chelis#159 substring template, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_ascription_precision_mismatch_reports_precision_mismatch_with_template() {
    // Direct precision mismatch: declared `f64`, RHS is an explicitly
    // f32-cast scalar. Must report PrecisionMismatch with the
    // chelis#159 substring template.
    let errors = typecheck_surf(
        r#"
def caller() -> f64 =
  {
    x: f64 = cast(1.0, f32)
    x
  }
"#,
    );
    assert!(
        errors_contain_kind_and_message(
            &errors,
            &CheckErrorKind::PrecisionMismatch,
            "let-binding `x` ascription does not match RHS",
        ),
        "expected PrecisionMismatch with the chelis#159 substring template, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_ascription_type_mismatch_reports_type_mismatch_with_template() {
    // Direct top-level type mismatch: declared `tensor[3, f32]`, RHS
    // is a scalar `f32`. Must report TypeMismatch with the
    // chelis#159 substring template (a unify-level type-shape
    // failure — Tensor vs Prim — surfaces under TypeMismatch, not
    // DimensionMismatch or PrecisionMismatch).
    let errors = typecheck_surf(
        r#"
def caller() -> tensor[3, f32] =
  {
    x: tensor[3, f32] = cast(1.0, f32)
    to_tensor([1.0, 2.0, 3.0], f32)
  }
"#,
    );
    assert!(
        errors_contain_kind_and_message(
            &errors,
            &CheckErrorKind::TypeMismatch,
            "let-binding `x` ascription does not match RHS",
        ),
        "expected TypeMismatch with the chelis#159 substring template, got errors:\n{}",
        errors_summary(&errors)
    );
}
