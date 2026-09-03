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

// chelis#1265 and Slice C c3: a comparison relates its operands' shapes and
// its result's dtype under [05-OP-36], so it must route its result through
// unification instead of constructing it out of band. While a positional
// `expand` operand's choice is open the result carries that same open choice
// and supplies no evidence back to the operand; a shape independently
// supplied to the result fixes the operand, because [05-OP-36] makes all
// three shapes one equation.
//
// Every row below is a regression test, not a disposition lock. On
// a5137d2c3 the checker fabricated `tensor[D, bool]` from whichever operand
// already had a shape and never touched the call's result variable, so a
// declared result shape bound a free variable and selected nothing.

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

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn comparison_result_shape_selects_the_replacement_candidate() {
    let rendered = check_surf(
        r#"
def f(a: tensor[2, f32]) -> tensor[3, bool] = {
  e = expand(a, 0, 3i64)
  eq(e, e)
}
"#,
    )
    .expect("the replacement candidate satisfies the declared result shape");
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} f32))"),
        "the declared rank-1 bool result must select the operand's \
         same-rank replacement candidate:\n{rendered}"
    );
}

#[test]
fn comparison_result_shape_selects_the_insertion_candidate() {
    let rendered = check_surf(
        r#"
def f(a: tensor[2, f32]) -> tensor[3, 2, bool] = {
  e = expand(a, 0, 3i64)
  eq(e, e)
}
"#,
    )
    .expect("the insertion candidate satisfies the declared result shape");
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))"),
        "the declared rank-2 bool result must select the operand's insertion \
         candidate; stamping the operand at the rank-1 freeze default beside a \
         rank-2 result publishes an internally inconsistent program:\n{rendered}"
    );
    assert!(
        !rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} f32))"),
        "the operand must not keep the rank-1 freeze default while the \
         result is declared at rank 2:\n{rendered}"
    );
}

#[test]
fn comparison_rejects_a_result_shape_no_candidate_satisfies() {
    let errors = check_surf(
        r#"
def f(a: tensor[2, f32]) -> tensor[9, 9, bool] = {
  e = expand(a, 0, 3i64)
  eq(e, e)
}
"#,
    )
    .expect_err("neither expand candidate has shape 9 by 9");
    assert!(
        !errors.is_empty(),
        "a declared comparison result shape that no operand candidate \
         satisfies must be rejected"
    );
    assert!(
        errors.iter().any(|error| {
            format!("{:?}", error.kind).contains("Dimension")
                || format!("{:?}", error.kind).contains("TypeMismatch")
        }),
        "the rejection must name the shape disagreement:\n{}",
        summary(&errors)
    );
}

#[test]
fn comparison_over_a_pending_operand_publishes_no_unresolved_variable() {
    let rendered = check_surf(
        "t = eq(expand(to_tensor([0.5f32]), 0, 3i64), expand(to_tensor([0.5f32]), 0, 3i64))\n",
    )
    .expect("an unconsumed comparison over pending operands still checks");
    assert!(
        !rendered.contains("(t-var"),
        "a comparison result whose operands only freeze at the program \
         freeze point must still settle to a concrete tensor rather than \
         escaping as an inference variable:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} bool))"),
        "the result mirrors the operand's frozen shape at bool:\n{rendered}"
    );
}

#[test]
fn two_pending_operands_of_one_comparison_are_solved_as_one_equation() {
    let rendered = check_surf(
        r#"
def f(a: tensor[2, f32]) -> tensor[3, 2, bool] = {
  x = expand(a, 0, 3i64)
  y = expand(a, 0, 3i64)
  eq(x, y)
}
"#,
    )
    .expect("both operands admit the insertion candidate");
    let insertion = "(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))";
    assert_eq!(
        rendered.matches(insertion).count(),
        2,
        "the comparison's own signature unifies its two operands, so both \
         pending results must settle on one shape:\n{rendered}"
    );
}

/// The coupling is on the dimensions, not on the forms.
///
/// [05-OP-36] constrains the dimensions a comparison's operands produce, not
/// the forms that produce them, so two operands solved as one equation may
/// select different forms. Here the left operand's candidates are rank 1 and
/// rank 2 and the right operand's are rank 2 and rank 3, and exactly one pair
/// lands at the same dimensions: insertion on the left, replacement on the
/// right. A checker that required the same form would reject this program, and
/// one that required nothing would accept a pair that cannot execute.
///
/// This row is a disposition lock, not a regression test: it passes on
/// `a5137d2c3` too, because the selection was already right. It is here
/// because the rule is easy to implement wrongly, not because it was broken.
/// Note that the program still fails in `chelis eval`, for an unrelated reason
/// this change does not touch: lowering's `fallback_expand_type` has no
/// replacement branch, so the right operand lowers as an insertion. chelis#597
/// and Slice B own that.
#[test]
fn two_pending_operands_may_select_different_forms_to_reach_one_shape() {
    let rendered = check_surf(
        "x1 = to_tensor([1.0f32, 2.0f32])\n\
         y2 = to_tensor([[1.0f32, 2.0f32]])\n\
         a = expand(x1, 0, 3i64)\n\
         b = expand(y2, 0, 3i64)\n\
         out = cmplt(a, b)\n",
    )
    .expect("one pair of forms satisfies the shared dimension equation");
    let shared = "(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))";
    assert_eq!(
        rendered.matches(shared).count(),
        4,
        "the rank-1 operand takes its insertion form and the rank-2 operand \
         takes its replacement form, and both land at tensor[3, 2]; each is \
         stamped on its own binding and on its expand application:\n{rendered}"
    );
    assert!(
        !rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} f32))"),
        "the left operand must not take its replacement form:\n{rendered}"
    );
    assert!(
        !rendered.contains("(d-lit {} 3) (d-lit {} 1) (d-lit {} 2)"),
        "the right operand must not take its insertion form:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} bool))"),
        "the comparison result carries the shared dimensions at bool:\n{rendered}"
    );
}

#[test]
fn issue1265_expand_then_add_then_comparison_selects_the_declared_shape() {
    let rendered = check_surf(
        r#"
def h5(a: tensor[2, f32]) -> tensor[3, 2, bool] = {
  e = expand(a, 0, 3i64)
  s = add(e, e)
  eq(s, s)
}
"#,
    )
    .expect("the dtype-preserving consumer carries the candidate to the comparison");
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))"),
        "chelis#1265: an intervening `add` carries the candidate unselected, \
         and the comparison must then select it:\n{rendered}"
    );
}
