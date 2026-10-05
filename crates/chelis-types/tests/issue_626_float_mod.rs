//! Issue #626: `mod` admits two float operands of one dtype.
//!
//! `spec/05-risc-primitives.md` [05-OP-64] gives `mod` the signed integers
//! and the floats, on two scalars or two same-shaped tensors of one dtype.
//! Float `mod` is C `fmod`. Mixed dtypes, a float beside an integer, bool,
//! and differentiation are refused.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn errors_of(source: &str) -> Vec<String> {
    match check_ir_program(&surf_to_deep(source)) {
        Ok(_) => vec![],
        Err(report) => report.errors.iter().map(|e| e.message.clone()).collect(),
    }
}

const FLOATS: [&str; 4] = ["f16", "bf16", "f32", "f64"];

#[test]
fn float_scalars_and_tensors_return_the_operand_type() {
    for dtype in FLOATS {
        let scalar = format!("def f(x: {dtype}, y: {dtype}) -> {dtype} = mod(x, y)\n");
        let errors = errors_of(&scalar);
        assert!(errors.is_empty(), "scalar {dtype}: {errors:?}");
        let tensor = format!(
            "def f(x: tensor[2, 3, {dtype}], y: tensor[2, 3, {dtype}]) -> tensor[2, 3, {dtype}] = mod(&x, &y)\n"
        );
        let errors = errors_of(&tensor);
        assert!(errors.is_empty(), "tensor {dtype}: {errors:?}");
    }
    let bound = "def g[p: Float](x: tensor[3, p]) -> tensor[3, p] = mod(x, x)\n";
    let errors = errors_of(bound);
    assert!(errors.is_empty(), "Float-bound binder: {errors:?}");
}

#[test]
fn float_mod_result_is_not_another_dtype() {
    let source = "def f(x: f32, y: f32) -> f64 = mod(x, y)\n";
    assert!(!errors_of(source).is_empty());
}

#[test]
fn mixed_float_and_integer_operands_are_refused_by_name() {
    for (lhs, rhs) in [
        ("f32", "f64"),
        ("f32", "i32"),
        ("i64", "f64"),
        ("f16", "bf16"),
    ] {
        let scalar = format!("def f(x: {lhs}, y: {rhs}) -> {lhs} = mod(x, y)\n");
        let errors = errors_of(&scalar);
        assert!(
            errors.iter().any(|m| m.contains(lhs) && m.contains(rhs)),
            "{lhs} beside {rhs}: {errors:?}"
        );
        let tensor = format!(
            "def f(x: tensor[2, {lhs}], y: tensor[2, {rhs}]) -> tensor[2, {lhs}] = mod(x, y)\n"
        );
        assert!(!errors_of(&tensor).is_empty(), "tensor {lhs} beside {rhs}");
    }
}

#[test]
fn bool_and_float_bitwise_stay_refused() {
    assert!(!errors_of("def f(x: bool, y: bool) -> bool = mod(x, y)\n").is_empty());
    for op in ["bitand", "bitor", "bitxor", "shl", "shr"] {
        let source = format!("def f(x: f32, y: f32) -> f32 = {op}(x, y)\n");
        assert!(!errors_of(&source).is_empty(), "{op} stays integer-only");
    }
    let trunc = "def f(x: f32, y: f32) -> f32 = trunc_div(x, y)\n";
    assert!(!errors_of(trunc).is_empty(), "trunc_div stays integer-only");
}
