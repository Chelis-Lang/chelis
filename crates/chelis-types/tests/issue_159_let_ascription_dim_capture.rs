//! Probe for chelis#159 review comment 4: enclosing dim-var capture
//! behavior in let-binding ascriptions.
//!
//! Background
//! ----------
//! The chelis#159 prototype patch in `infer_let` calls
//! `deep_type_to_resolved_type(ty_expr, vg, adt_reg, &mut HashMap::new())`
//! with a fresh `HashMap` per binding. The reviewer asked whether this
//! means dim names in the ascription correctly share with the enclosing
//! function signature's dim vars, or get fresh vars.
//!
//! What the codebase actually does
//! -------------------------------
//! The `&mut HashMap` passed to `deep_type_to_resolved_type` is a
//! `HashMap<String, TypeVar>` (`infer.rs:5404-5412`). It maps named
//! type variables (`'a`, `T`, etc.) to internal `TypeVar` IDs. It is
//! NOT a dim-variable map.
//!
//! Dim names in Chelis types parse to `Dim::Name(String)` — string
//! labels, not capturing variables (`infer.rs:12007-12013`,
//! `crates/chelis-types/src/types.rs`). At unification time
//! (`unify_dim` in `unify.rs`):
//!
//! - `Dim::Name(n1) <-> Dim::Name(n2)` succeeds iff `n1 == n2` (line 442).
//! - `Dim::Name(n) <-> Dim::Var(v)` binds `v := Name(n)` (line 446-447).
//!
//! So the user's `n` in a let-ascription is a string label. It does
//! NOT capture the enclosing sig's `n` directly; it unifies with
//! whatever the RHS expression's dim happens to be, by name equality
//! for `Name <-> Name` and by binding for `Var <-> Name`.
//!
//! In practice this means: when the body's RHS type already carries
//! the sig's instantiated dim var (because the sig param was seeded
//! into the body's env), the user's `Name("n")` binds that var to
//! `Name("n")` and any further reference to the var resolves to the
//! same name. Cross-position sig contracts (`&tensor[n, f32] -> &tensor[n, f32]`)
//! are enforced via the sig's instantiation step, not via let-ascription
//! "capture."
//!
//! The probes below pin the observed behavior so the design is
//! discoverable and so future refactors don't silently change it.

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
fn let_ascription_with_sig_dim_name_typechecks_when_dim_matches() {
    // Sig declares two params sharing dim `n`. Body uses two
    // let-bindings ascribed `tensor[n, f32]`, each backed by one of
    // the sig params. The sig contract guarantees the params share
    // the same instantiated dim var; the let-ascriptions are
    // consistent with that. No error expected.
    let errors = typecheck_surf(
        r"
sig pair_id: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def pair_id(a: &tensor[n, f32], b: &tensor[n, f32]) -> tensor[n, f32] =
  {
    x: &tensor[n, f32] = a
    y: &tensor[n, f32] = b
    pair_id(x, y)
  }
",
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
        r"
def caller(t: &tensor[2, f32]) -> &tensor[2, f32] =
  {
    x: &tensor[3, f32] = t
    x
  }
",
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
fn dim_name_is_a_label_not_a_capture() {
    // OBSERVATIONAL: in Chelis today, dim names like `n` are
    // `Dim::Name(String)` labels. A let-ascription `x: tensor[n, f32]`
    // does NOT "capture" the enclosing sig's `n` per-binding; instead,
    // the user's `Name("n")` unifies by string equality with whatever
    // dim the RHS expression carries, and binds any free dim var to
    // `Name("n")`.
    //
    // For a sig like `&tensor[n, f32] -> &tensor[m, f32] -> ...` (two
    // DISTINCT dim names), writing both let-ascriptions as `n` does
    // NOT produce a cross-position contract violation: it just binds
    // both fresh sig vars (d_n, d_m) to `Name("n")` independently.
    //
    // This is the existing behavior, not a bug introduced by the
    // chelis#159 patch. It is pinned here so a future refactor that
    // tries to "make let-ascription dim names capture" doesn't silently
    // change the semantics without trip-wiring this test.
    let errors = typecheck_surf(
        r"
def caller(a: &tensor[n, f32], b: &tensor[m, f32]) -> &tensor[n, f32] =
  {
    x: &tensor[n, f32] = a
    y: &tensor[n, f32] = b
    x
  }
",
    );
    assert!(
        !has_dimension_mismatch(&errors),
        "today, naming both let-ascriptions `n` against sig params \
         with distinct dim names `n` and `m` does not produce a \
         DimensionMismatch (dim names are labels, not captures). If \
         this test starts failing, the design has shifted - update \
         the diagnosis doc accordingly. Got errors:\n{}",
        errors_summary(&errors)
    );
}
