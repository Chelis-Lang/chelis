//! Issue #3000: a comparison over borrowed tensor operands returns a bool
//! tensor of the operands' shape.
//!
//! `spec/05-risc-primitives.md` [05-OP-36] admits two tensors of one active
//! numeric dtype and identical dimensions, and [05-OP-53] makes `where` take
//! a bool condition of the branches' shape. Whether the author spells the
//! operands `x` or `&x` does not change either rule, so `where(lt(&x, &y), x,
//! y)` checks exactly as `where(lt(x, y), x, y)` does.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

const OPS: [&str; 6] = ["lt", "lte", "gt", "gte", "eq", "neq"];
const FLOATS: [&str; 4] = ["f16", "bf16", "f32", "f64"];

fn assert_checks(source: &str) {
    if let Err(report) = check_ir_program(&surf_to_deep(source)) {
        panic!("must check:\n{source}\ngot {:?}", report.errors);
    }
}

#[test]
fn borrowed_comparison_selects_inside_a_def() {
    for op in OPS {
        for dtype in FLOATS {
            assert_checks(&format!(
                "def pick(x: tensor[2, {dtype}], y: tensor[2, {dtype}]) -> tensor[2, {dtype}] = where({op}(&x, &y), x, y)\n"
            ));
        }
    }
}

#[test]
fn borrowed_comparison_selects_at_top_level() {
    for op in OPS {
        for dtype in FLOATS {
            assert_checks(&format!(
                "module Probe.Case\nx = to_tensor([1.0{dtype}, 3.0{dtype}])\ny = to_tensor([2.0{dtype}, 0.5{dtype}])\nr = print(where({op}(&x, &y), x, y))\n"
            ));
        }
    }
}

#[test]
fn borrowed_comparison_result_is_a_bool_tensor_of_the_operand_shape() {
    for op in OPS {
        assert_checks(&format!(
            "def mask(x: &tensor[2, 3, f32], y: &tensor[2, 3, f32]) -> tensor[2, 3, bool] = {op}(x, y)\n"
        ));
        assert_checks(&format!(
            "def mask(x: tensor[2, 3, f32], y: tensor[2, 3, f32]) -> tensor[2, 3, bool] = {op}(&x, &y)\n"
        ));
    }
}

#[test]
fn borrowed_comparison_result_is_not_the_operand_dtype() {
    for op in OPS {
        let source = format!(
            "def mask(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = {op}(&x, &y)\n"
        );
        let report = check_ir_program(&surf_to_deep(&source))
            .expect_err("a comparison returns bool, never the operand dtype");
        assert!(
            report.errors.iter().any(|error| matches!(
                error.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::PrecisionMismatch
            )),
            "{op}: got {:?}",
            report.errors
        );
    }
}

#[test]
fn borrowed_comparison_rejects_mismatched_shapes_and_dtypes() {
    for op in OPS {
        let shapes = format!(
            "def pick(x: tensor[2, f32], y: tensor[3, f32]) -> tensor[2, bool] = {op}(&x, &y)\n"
        );
        check_ir_program(&surf_to_deep(&shapes))
            .expect_err("[05-OP-36] requires identical dimensions");
        let dtypes = format!(
            "def pick(x: tensor[2, f32], y: tensor[2, f64]) -> tensor[2, bool] = {op}(&x, &y)\n"
        );
        check_ir_program(&surf_to_deep(&dtypes)).expect_err("[05-OP-36] requires one dtype");
    }
}

#[test]
fn borrowed_comparison_condition_must_match_the_branch_shape() {
    let source = "def pick(x: tensor[2, f32], y: tensor[2, f32], a: tensor[3, f32], b: tensor[3, f32]) -> tensor[3, f32] = where(lt(&x, &y), a, b)\n";
    check_ir_program(&surf_to_deep(source))
        .expect_err("[05-OP-53]: the condition's shape equals the branch shape");
}
