//! Probe for chelis#159 review comment 4: enclosing dim-var capture
//! behavior in let-binding ascriptions.
//!
//! Background
//! ----------
//! The chelis#159 prototype patch in `infer_let` calls
//! `deep_type_to_resolved_type(ty_expr, vg, adt_reg, &mut UnordMap::new())`
//! with a fresh `UnordMap` per binding. The reviewer asked whether this
//! means dim names in the ascription correctly share with the enclosing
//! function signature's dim vars, or get fresh vars.
//!
//! Current explicit-binder contract
//! --------------------------------
//! [04-INF-6] now makes every listed declaration binder rigid throughout the
//! body. An ordinary annotation spelling a listed dimension name therefore
//! reuses that declaration-owned `DimVar`; an unlisted name remains outside
//! the declaration binder scope. The third probe below was intentionally
//! inverted when that normative contract replaced the historical label-only
//! behavior.
//!
//! The probes below pin the observed behavior so the design is
//! discoverable and so future refactors don't silently change it.

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
fn let_ascription_with_sig_dim_name_typechecks_when_dim_matches() {
    // Sig declares two params sharing dim `n`. Body uses two
    // let-bindings ascribed `tensor[n, f32]`, each backed by one of
    // the sig params. The sig contract guarantees the params share
    // the same instantiated dim var; the let-ascriptions are
    // consistent with that. No error expected.
    let errors = typecheck_surf(
        r#"
sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(a: &tensor[n, f32], b: &tensor[n, f32]) -> tensor[n, f32] =
  {
    x: &tensor[n, f32] = a
    y: &tensor[n, f32] = b
    pair_id(x, y)
  }
"#,
    );
    assert!(
        !has_dimension_mismatch(&errors),
        "expected no DimensionMismatch when let-ascriptions \
         consistently name the sig's shared dim, got errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn let_ascription_with_concrete_dim_mismatch_against_rhs_errors() {
    // Most direct probe of the patch's behavior: declared concrete
    // dim disagrees with the RHS's concrete dim. unify_dim must
    // produce DimensionMismatch via the patch's surfaced diagnostic.
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
        has_dimension_mismatch(&errors),
        "expected DimensionMismatch - declared `tensor[3, f32]` should \
         not accept a `&tensor[2, f32]` RHS, got errors:\n{}",
        errors_summary(&errors)
    );
    assert!(
        errors.iter().any(|e| e
            .message
            .contains("let-binding `x` ascription does not match RHS")),
        "expected the chelis#159 diagnostic template `let-binding `x` ascription does not \
         match RHS` in errors:\n{}",
        errors_summary(&errors)
    );
}

#[test]
fn declared_dim_name_captures_the_declaration_identity() {
    // `n` and `m` are distinct authored binders. Reusing `n` on the
    // ascription backed by the `m` parameter identifies those declarations,
    // which [04-INF-6] rejects.
    let errors = typecheck_surf(
        r#"
def caller[n, m](a: &tensor[n, f32], b: &tensor[m, f32]) -> &tensor[n, f32] =
  {
    x: &tensor[n, f32] = a
    y: &tensor[n, f32] = b
    x
  }
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("`n`")
                && error.message.contains("`m`")
                && error.message.contains("[04-INF-6]")
        }),
        "an ordinary body annotation must reuse the declaration-owned dim \
         identity and reject collapsing `n` with `m`; got errors:\n{}",
        errors_summary(&errors)
    );
}
