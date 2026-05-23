//! Probes for chelis#206: `reshape(x, [cast(shape(x, axis), int64), ...])`
//! is syntactically accepted but the type checker does not propagate the
//! corresponding symbolic dim of `x` into the reshape result type.
//!
//! Background
//! ----------
//! From the issue's reproducer, a function declared as
//!
//! ```text
//! sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
//! def flatten_batch(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
//! ```
//!
//! reports a `TypeMismatch` because the body is inferred as
//! `tensor[Wildcard, f32]`. The neighbor pattern
//! `expand(b, 0, shape(x, cast(0, int32)))` type-checks cleanly because
//! `expand`'s output dim is forced by the declared signature -- but the
//! issue is that `reshape` collapses the entire output rank to a
//! `vec![Wildcard]` (the input precision case in `infer_reshape_app`
//! recovers only the precision, not the dims).
//!
//! Expected fix: when an element of `reshape`'s shape list is the
//! syntactic form `cast(shape(x, lit_axis), int64)` and `x` is the same
//! input tensor being reshaped AND `lit_axis` resolves to a known axis
//! of `x`, the typer must propagate `x`'s dim at that axis into the
//! corresponding output dim. Plain literal dims like `cast(4, int64)`
//! must continue to produce `Dim::Lit(4)`. Any other shape-list element
//! shape (arbitrary expression that the recognizer doesn't know how to
//! interpret) must fall back to `Dim::Wildcard` -- no regression in
//! diagnostic quality.

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

fn has_type_mismatch(errors: &[CheckError]) -> bool {
    errors
        .iter()
        .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch))
}

/// The issue's exact reproducer (`flatten_batch`) MUST type-check. Before
/// the fix this is the failing case and emits a `TypeMismatch`.
#[test]
fn issue_206_flatten_batch_typechecks() {
    let errors = typecheck_surf(
        r#"
module Probe.Runtime
export (flatten_batch)

sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
"#,
    );
    assert!(
        errors.is_empty(),
        "flatten_batch from issue 206 should type-check; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// The neighbor case (`bias_broadcast`) MUST still type-check -- this
/// pins the no-regression for the existing working pattern.
#[test]
fn issue_206_bias_broadcast_typechecks() {
    let errors = typecheck_surf(
        r#"
module Probe.Runtime
export (bias_broadcast)

sig bias_broadcast: &tensor[n, 4, f32] -> &tensor[4, f32] -> tensor[n, 4, f32]
def bias_broadcast(x, b) = expand(b, 0, shape(x, cast(0, int32)))
"#,
    );
    assert!(
        errors.is_empty(),
        "bias_broadcast should keep working after the fix; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// The issue's full module exactly as written -- both functions in one
/// module must pass `chelis check`.
#[test]
fn issue_206_full_module_typechecks() {
    let errors = typecheck_surf(
        r#"
module Probe.Runtime
export (bias_broadcast, flatten_batch)

sig bias_broadcast: &tensor[n, 4, f32] -> &tensor[4, f32] -> tensor[n, 4, f32]
def bias_broadcast(x, b) = expand(b, 0, shape(x, cast(0, int32)))

sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
"#,
    );
    assert!(
        errors.is_empty(),
        "full issue 206 module should type-check; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Two symbolic dims sourced from different axes of the same input. Both
/// axis-0 and axis-1 dims should propagate.
#[test]
fn issue_206_two_symbolic_dims_propagate() {
    let errors = typecheck_surf(
        r#"
sig flatten_two: &tensor[n, m, f32] -> tensor[n, m, f32]
def flatten_two(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(shape(x, cast(1, int32)), int64)])
"#,
    );
    assert!(
        errors.is_empty(),
        "two symbolic dims from axis-0 and axis-1 of the same input \
         should both propagate; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Mixed: symbolic dim from `shape(x, 0)` plus a plain literal. The
/// symbolic dim propagates; the literal stays a `Dim::Lit`.
#[test]
fn issue_206_mixed_symbolic_and_literal_dim() {
    let errors = typecheck_surf(
        r#"
sig with_literal: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def with_literal(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
"#,
    );
    assert!(
        errors.is_empty(),
        "mixed symbolic+literal dim list should preserve the symbolic dim; \
         got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Regression: all-literal dim list still type-checks. This was the
/// RT-A1W1 CRIT case (chelis#35) and should not regress.
#[test]
fn issue_206_all_literal_dims_still_work() {
    let errors = typecheck_surf(
        r#"
sig fixed: &tensor[n, 4, f32] -> tensor[4, 4, f32]
def fixed(x) = reshape(x, [cast(4, int64), cast(4, int64)])
"#,
    );
    assert!(
        errors.is_empty(),
        "all-literal dim list should keep type-checking; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Negative: `cast(shape(y, ...), int64)` where `y` is a DIFFERENT
/// tensor (not the input being reshaped) — the rule only propagates a
/// dim when the shape source IS the same input being reshaped. With a
/// foreign tensor we fall back to `Dim::Wildcard` and the declared sig
/// (which expects `tensor[n, 4, f32]`) does NOT match `Wildcard`, so we
/// expect a `TypeMismatch`.
#[test]
fn issue_206_shape_of_other_tensor_does_not_propagate() {
    let errors = typecheck_surf(
        r#"
sig cross_dim: &tensor[n, 4, f32] -> &tensor[m, 4, f32] -> tensor[n, 4, f32]
def cross_dim(x, y) = reshape(x, [cast(shape(y, cast(0, int32)), int64), cast(4, int64)])
"#,
    );
    assert!(
        has_type_mismatch(&errors),
        "reshape's symbolic-dim recognizer must require the shape source \
         to be the same tensor being reshaped; cross-tensor shape calls \
         should not propagate y's dim into x's reshape result. Expected \
         a TypeMismatch; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Negative: arbitrary expression in the dim list (e.g. `cast(add(...),
/// int64)`) is not a recognized symbolic-dim source — falls back to
/// `Dim::Wildcard`. With a declared sig that expects a concrete `n`,
/// this produces a `TypeMismatch`.
#[test]
fn issue_206_unrecognized_dim_expression_falls_back_to_wildcard() {
    let errors = typecheck_surf(
        r#"
sig add_one_dim: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def add_one_dim(x) = reshape(x, [cast(add(shape(x, cast(0, int32)), 1), int64), cast(4, int64)])
"#,
    );
    assert!(
        has_type_mismatch(&errors),
        "an arithmetic-wrapped dim expression must not pretend to \
         preserve a symbolic dim; expected a TypeMismatch fallback to \
         Wildcard; got errors:\n{}",
        errors_summary(&errors)
    );
}
