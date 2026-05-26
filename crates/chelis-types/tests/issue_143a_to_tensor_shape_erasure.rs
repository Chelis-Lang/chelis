//! Probe for chelis#143 sub-issue (A): `to_tensor` erases list-literal shape.
//!
//! Background
//! ----------
//! `to_tensor` is registered as `generic_unop` in
//! `crates/chelis-types/src/builtins.rs:873`, which gives it the scheme
//! `∀α β. α → β` — no dim variables, no shape constraints. When the user
//! writes `to_tensor([1.0, 2.0, 3.0])`, the parser preserves the list
//! length in `Expr::List(Vec<Expr>, Span)`
//! (`crates/chelis-surf/src/ast.rs:125`), and the desugarer has
//! `items.len()` available at
//! `crates/chelis-surf/src/desugar.rs:1489-1500`
//! (`desugar_list_as_tensor_literal`), but the resulting `Cons` chain is
//! handed to a bare `to_tensor` application whose output type is a
//! fresh `Type::Var` with no dim info. The sig-typed `&tensor[n, f32]`
//! it later flows into cannot bind `n` to any concrete `Lit`, so the
//! cross-arg dim contract a shared-dim sig promises is silently dropped.
//!
//! This is the actual root cause of the chelis#143 repro
//! (`scaled_dot_product_attention` with tensors produced by
//! `pad_sequences_to`); the original commit 1681b52 added a no-op match
//! arm to `unify_dim` that did not address this path.
//!
//! What this test asserts
//! ----------------------
//! Regression lock for the post-fix behavior delivered by PR #227:
//! passing two `to_tensor` results with different statically-derivable
//! shapes into a sig with a shared dim var trips `DimensionMismatch`.
//! The `static_to_tensor_shape` walker in `chelis-types::infer`
//! emits per-axis `Dim::Lit(n)` from the list-literal Cons chain so
//! the sig var `n` binds to `Lit(3)` on the first arg and rejects
//! `Lit(5)` on the second.
//!
//! The counter-probe demonstrates that the sig-dim unification path
//! itself works correctly when the args carry concrete `Lit` dims —
//! complementary coverage isolating the original bug surface to
//! tensor-builder shape erasure, not `unify_dim`.
//!
//! Tracking: chelis#158 (sub-issue (A) of chelis#143)
//! Diagnosis: docs/investigations/issue_143a_to_tensor_shape_erasure_diagnosis.md

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn typecheck_surf(source: &str) -> Vec<CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
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
fn to_tensor_mismatched_list_lengths_should_trip_dim_mismatch() {
    // Desired post-fix behavior:
    //   - `to_tensor([f32; 3])` infers `tensor[Lit(3), f32]`
    //   - `to_tensor([f32; 5])` infers `tensor[Lit(5), f32]`
    //   - the shared-dim sig binds `n := 3` on the first arg, then
    //     `Lit(3)` vs `Lit(5)` on the second arg trips DimensionMismatch.
    //
    // Today: both `to_tensor` outputs are fresh `Type::Var`s with no
    // dim info; the sig var `n` stays free; no DimensionMismatch is
    // reported (the failure surfaces only at runtime, as documented in
    // chelis#143).
    let errors = typecheck_surf(
        r#"
sig pair_id: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller() -> tensor[3, f32] =
  {
    a = to_tensor([1.0, 2.0, 3.0])
    b = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    pair_id(&a, &b)
  }
"#,
    );
    assert!(
        has_dimension_mismatch(&errors),
        "expected DimensionMismatch (sig `n` should bind to 3 from `a`, then \
         reject 5 from `b`), got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn parameter_ascription_with_shared_sig_dim_does_trip_dim_mismatch() {
    // Counter-probe: when the caller's args carry concrete `Lit` dims
    // (via direct parameter ascription, not `to_tensor`), the sig's
    // shared dim var unification works correctly. This isolates the
    // bug above to tensor-builder shape erasure, not unify_dim.
    let errors = typecheck_surf(
        r#"
sig pair_id: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller(a: &tensor[3, f32], b: &tensor[5, f32]) -> tensor[3, f32] =
  pair_id(a, b)
"#,
    );
    assert!(
        has_dimension_mismatch(&errors),
        "expected DimensionMismatch (sig `n` binds to 3 from `a`, conflicts with 5 from `b`), \
         got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn matched_to_tensor_lengths_does_not_trip_dim_mismatch_today() {
    // Sanity: with matched list lengths, today's buggy behavior still
    // does not produce a spurious DimensionMismatch. (This will remain
    // true post-fix; it's an invariant lock against over-corrections.)
    let errors = typecheck_surf(
        r#"
sig pair_id: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(x, y) = x
def caller() -> tensor[3, f32] =
  {
    a = to_tensor([1.0, 2.0, 3.0])
    b = to_tensor([1.0, 2.0, 3.0])
    pair_id(&a, &b)
  }
"#,
    );
    assert!(
        !has_dimension_mismatch(&errors),
        "matched dims must not produce DimensionMismatch, got errors:\n{}",
        errors_summary(&errors)
    );
}
