//! Issue #5 regression diagnosis: when both `gt(tensor, scalar)` and
//! `gt(scalar, tensor)` appear in the same module, type inference fails for
//! both. Each form alone works (when written without a declared `def -> T`
//! signature). This test pin-points the failure surface so we can iterate on
//! the fix in this crate's fast inner loop.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_phase0e_program;

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

#[test]
fn issue5_both_forms_in_same_module_typecheck_clean() {
    let src = r#"
def above -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0]), 1.5)
def below -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0]))
"#;
    let deep = surf_to_deep(src);
    let res = check_phase0e_program(&deep);
    match res {
        Ok(_) => {}
        Err(rep) => {
            for err in &rep.errors {
                eprintln!("ERR: {:?}: {}", err.kind, err.message);
            }
            panic!("expected clean check, got {} error(s)", rep.errors.len());
        }
    }
}

#[test]
fn issue5_scalar_first_alone_typechecks_clean() {
    let src = r#"
def below -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0]))
"#;
    let deep = surf_to_deep(src);
    let res = check_phase0e_program(&deep);
    match res {
        Ok(_) => {}
        Err(rep) => {
            for err in &rep.errors {
                eprintln!("ERR: {:?}: {}", err.kind, err.message);
            }
            panic!("expected clean check, got {} error(s)", rep.errors.len());
        }
    }
}

#[test]
fn issue5_tensor_first_alone_typechecks_clean() {
    let src = r#"
def above -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0]), 1.5)
"#;
    let deep = surf_to_deep(src);
    let res = check_phase0e_program(&deep);
    match res {
        Ok(_) => {}
        Err(rep) => {
            for err in &rep.errors {
                eprintln!("ERR: {:?}: {}", err.kind, err.message);
            }
            panic!("expected clean check, got {} error(s)", rep.errors.len());
        }
    }
}
