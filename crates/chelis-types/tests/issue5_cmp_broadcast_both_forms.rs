//! Issue #5 regression diagnosis: when both `gt(tensor, scalar)` and
//! `gt(scalar, tensor)` appear in the same module, type inference fails for
//! both. Each form alone works (when written without a declared `def -> T`
//! signature). This test pin-points the failure surface so we can iterate on
//! the fix in this crate's fast inner loop.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;

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
def above() -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0]), 1.5)
def below() -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0]))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
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
def below() -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0]))
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
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
def above() -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0]), 1.5)
"#;
    let deep = surf_to_deep(src);
    let res = check_ir_program(&deep);
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

// A comparison relates its operands' shapes and its result's dtype under
// [05-OP-36], so it routes its result through unification rather than
// constructing it out of band.
//
// The five rows that proved this by having a consumer SELECT among an
// operand's candidate shapes are gone with the candidate model: under
// `spec/04-type-system.md` section 4.7.2 `expand` and `insert` each have one
// result shape, so no operand is ever pending and no consumer selects
// (chelis#1277 S2b). chelis#1265's own subject went with them; the design doc
// records it as superseded and to be re-read against section 4.7.2.
//
// The row that survives is the one whose subject was never selection: a
// comparison must publish a concrete result rather than an inference
// variable. It is a disposition lock under the single result shape.

use chelis_deep::printer::print_canonical;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check_surf(source: &str) -> Result<String, Vec<CheckError>> {
    let deep = surf_to_deep(source);
    match check_typed_program(&deep) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

#[test]
fn an_unconsumed_comparison_publishes_no_unresolved_variable() {
    let rendered = check_surf(
        "t = eq(expand(to_tensor([0.5f32]), 0, 3i64), expand(to_tensor([0.5f32]), 0, 3i64))\n",
    )
    .expect("an unconsumed comparison over two expand results still checks");
    assert!(
        !rendered.contains("(t-var"),
        "a comparison result must settle to a concrete tensor rather than \
         escaping as an inference variable:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} bool))"),
        "the result mirrors the operand's shape at bool:\n{rendered}"
    );
}
