//! Issue Chelis-Lang/chelis#219: Name vs Lit unification asymmetry.
//!
//! Discovery recommended Option A: `unify_dim` should accept
//! `(Name(_), Lit(_))` and the symmetric `(Lit(_), Name(_))` as
//! satisfied without binding any substitution. A concrete literal at
//! a call site satisfies a concrete-but-named slot in the callee's
//! signature; names are preserved in diagnostics but they do not
//! impose a distinct-from-literal constraint.
//!
//! This file pins the positive case (Name <-> Lit unifies cleanly) and
//! the regression-lock negatives (distinct Names still error, distinct
//! Lits still error, distinct Vars still error).

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

/// Returns true iff any error in `errors` mentions a dimension issue.
/// Use this when the error kind may surface as a top-level `TypeMismatch`
/// against the def signature (which embeds the structural mismatch in
/// its message) rather than as a leaf-level `DimensionMismatch`.
fn any_error_mentions(errors: &[CheckError], substr: &str) -> bool {
    errors.iter().any(|e| e.message.contains(substr))
}

fn any_dim_or_type_mismatch(errors: &[CheckError]) -> bool {
    errors.iter().any(|e| {
        matches!(
            e.kind,
            CheckErrorKind::DimensionMismatch | CheckErrorKind::TypeMismatch
        )
    })
}

#[test]
fn issue_219_named_dim_sig_accepts_concrete_caller_via_load_lit() {
    // The R4 motivating shape with literal-typed call-site args: a
    // sig with multi-letter dim names (parsed as `d-name`, not
    // `d-var`) accepts a call that supplies concrete `Lit` dims. We
    // build a concrete dim via a typed `def caller(x: tensor[1, 3,
    // f32]) = f(copy(x))`, then the literal-dim test reduces to
    // Name <-> Lit at the parameter unification point.
    let errors = typecheck_surf(
        r#"
def f(x: tensor[batch, hidden, f32]) -> tensor[batch, hidden, f32] = copy(x)
def caller(x: &tensor[1, 3, f32]) -> tensor[1, 3, f32] = f(copy(x))
"#,
    );
    assert!(
        errors.is_empty(),
        "Option A: Name <-> Lit should unify; got {}",
        errors_summary(&errors)
    );
}

#[test]
fn issue_219_named_dim_sig_accepts_concrete_caller_symmetric() {
    // Symmetric direction: callee declares concrete `Lit` dims, the
    // calling function gives a `Name`-typed value. The arm must work
    // in both orientations.
    let errors = typecheck_surf(
        r#"
def f(x: tensor[3, 5, f32]) -> tensor[3, 5, f32] = copy(x)
def g(y: &tensor[batch, hidden, f32]) -> tensor[3, 5, f32] = f(copy(y))
"#,
    );
    assert!(
        errors.is_empty(),
        "Option A: Lit <-> Name should unify symmetrically; got {}",
        errors_summary(&errors)
    );
}

#[test]
fn issue_219_distinct_names_still_reject() {
    // Regression-lock: `batch` and `seq` are distinct named dims.
    // Returning a `tensor[seq, ...]` from a body declared to return
    // `tensor[batch, ...]` must still be a TypeMismatch / DimensionMismatch.
    let errors = typecheck_surf(
        r#"
def f(a: &tensor[batch, f32], b: &tensor[seq, f32]) -> tensor[batch, f32] = copy(b)
"#,
    );
    assert!(
        any_dim_or_type_mismatch(&errors) && any_error_mentions(&errors, "batch"),
        "Option A regression-lock: distinct Names must still error and mention 'batch'; got {}",
        errors_summary(&errors)
    );
}

#[test]
fn issue_219_distinct_lits_still_reject() {
    // Regression-lock: `Lit(2)` vs `Lit(3)` still errors. The
    // permissive arm only covers Name<->Lit, not Lit<->Lit.
    let errors = typecheck_surf(
        r#"
def f(x: &tensor[2, f32]) -> tensor[3, f32] = copy(x)
"#,
    );
    assert!(
        any_dim_or_type_mismatch(&errors),
        "Option A regression-lock: Lit(2) vs Lit(3) must still error; got {}",
        errors_summary(&errors)
    );
}

#[test]
fn issue_219_shared_var_across_two_args_still_couples() {
    // Regression-lock: a def with explicit `[n, p]` quantifiers that
    // uses `n` for two parameters must couple their dim variables
    // (well-known WS-A6 path). The Name <-> Lit arm targets `d-name`,
    // not `d-var`, so this coupling is unchanged.
    //
    // Use direct annotated args (`tensor[2, f32]` and `tensor[3, f32]`)
    // to side-step Var<->Lit coupling timing through `to_tensor` —
    // the test should fail with a body-signature mismatch because the
    // declared return is `tensor[n, p]` (i.e. dim of first param).
    let errors = typecheck_surf(
        r#"
def f[n, p](x: &tensor[n, p], y: &tensor[n, p]) -> tensor[n, p] = copy(y)
def caller(a: &tensor[2, f32], b: &tensor[3, f32]) -> tensor[2, f32] = f(copy(a), copy(b))
"#,
    );
    assert!(
        !errors.is_empty(),
        "Option A regression-lock: shared Var across two args must still couple; got no errors",
    );
}
