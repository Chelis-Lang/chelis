//! Issue #2076: `mod`, `bitand`, `bitor`, `bitxor`, `shl` and `shr` admit two
//! same-shaped, same-dtype signed-integer tensors.
//!
//! `spec/05-risc-primitives.md` [05-OP-64] gives `mod` "two same-shaped,
//! same-dtype tensors", and [05-OP-47] gives the bitwise and shift operations
//! "same-shaped tensors" of one signed-integer dtype. Each returns the input
//! dtype and shape. A float, bool or mixed dtype, a shape mismatch, and a
//! scalar beside a tensor are type errors.

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

const OPS: [&str; 6] = ["mod", "bitand", "bitor", "bitxor", "shl", "shr"];
const INTS: [&str; 4] = ["i8", "i16", "i32", "i64"];

fn errors_of(source: &str) -> Vec<String> {
    match check_ir_program(&surf_to_deep(source)) {
        Ok(_) => vec![],
        Err(report) => report.errors.iter().map(|e| e.message.clone()).collect(),
    }
}

#[test]
fn integer_tensors_return_the_operand_type() {
    for op in OPS {
        for dtype in INTS {
            let source = format!(
                "def f(x: tensor[2, 3, {dtype}], y: tensor[2, 3, {dtype}]) -> tensor[2, 3, {dtype}] = {op}(x, y)\n"
            );
            let errors = errors_of(&source);
            assert!(errors.is_empty(), "{op} over {dtype}: {errors:?}");
        }
    }
}

#[test]
fn int_bound_binder_is_admitted() {
    for op in OPS {
        let source = format!("def g[p: Int](x: tensor[3, p]) -> tensor[3, p] = {op}(x, x)\n");
        let errors = errors_of(&source);
        assert!(
            errors.is_empty(),
            "{op} over an Int-bound binder: {errors:?}"
        );
    }
}

#[test]
fn integer_tensor_result_is_not_another_dtype() {
    for op in OPS {
        let source =
            format!("def f(x: tensor[2, i32], y: tensor[2, i32]) -> tensor[2, i64] = {op}(x, y)\n");
        assert!(
            !errors_of(&source).is_empty(),
            "{op}: the result is i32, not i64"
        );
    }
}

#[test]
fn mixed_integer_dtypes_are_refused_by_name() {
    for op in OPS {
        let source =
            format!("def f(x: tensor[2, i32], y: tensor[2, i64]) -> tensor[2, i32] = {op}(x, y)\n");
        let errors = errors_of(&source);
        assert!(
            errors
                .iter()
                .any(|m| m.contains("i32") && m.contains("i64")),
            "{op} must name both dtypes: {errors:?}"
        );
    }
}

#[test]
fn mismatched_shapes_are_refused() {
    for op in OPS {
        let source =
            format!("def f(x: tensor[2, i32], y: tensor[3, i32]) -> tensor[2, i32] = {op}(x, y)\n");
        let errors = errors_of(&source);
        assert!(
            !errors.is_empty(),
            "{op}: a 2-tensor beside a 3-tensor must not check"
        );
        assert!(
            !errors.iter().any(|m| m.contains("matching integer")),
            "{op}: a shape mismatch is not a dtype mismatch: {errors:?}"
        );
    }
}

#[test]
fn float_and_bool_tensors_are_refused() {
    for op in ["bitand", "bitor", "bitxor", "shl", "shr"] {
        for dtype in ["f32", "f64", "bool"] {
            let source = format!(
                "def f(x: tensor[2, {dtype}], y: tensor[2, {dtype}]) -> tensor[2, {dtype}] = {op}(x, y)\n"
            );
            assert!(
                !errors_of(&source).is_empty(),
                "{op} over {dtype} must not check"
            );
        }
    }
    let source = "def f(x: tensor[2, bool], y: tensor[2, bool]) -> tensor[2, bool] = mod(x, y)\n";
    assert!(
        !errors_of(source).is_empty(),
        "mod over bool must not check"
    );
}

#[test]
fn a_scalar_beside_a_tensor_is_refused() {
    for op in OPS {
        let source = format!("def f(x: tensor[2, i32]) -> tensor[2, i32] = {op}(x, 1i32)\n");
        let errors = errors_of(&source);
        assert!(!errors.is_empty(), "{op}: no implicit broadcasting");
    }
}

#[test]
fn integer_scalars_are_unchanged() {
    for op in OPS {
        let source = format!("def f(x: i32, y: i32) -> i32 = {op}(x, y)\n");
        let errors = errors_of(&source);
        assert!(errors.is_empty(), "{op} over scalars: {errors:?}");
    }
}

/// A lambda operand that binds after the call is decided once it binds: an
/// agreeing tensor checks and a disagreeing one is refused.
#[test]
fn a_late_bound_tensor_operand_is_decided_when_it_binds() {
    let valid = "def f(x: tensor[2, i32], y: tensor[2, i32]) -> tensor[2, i32] = {\n  g = fn (t) -> mod(t, y)\n  g(x)\n}\n";
    let errors = errors_of(valid);
    assert!(errors.is_empty(), "{errors:?}");
    let invalid = "def f(x: tensor[2, i64], y: tensor[2, i32]) -> tensor[2, i32] = {\n  g = fn (t) -> mod(t, y)\n  g(x)\n}\n";
    let errors = errors_of(invalid);
    assert!(
        errors
            .iter()
            .any(|m| m.contains("i64") && m.contains("i32")),
        "{errors:?}"
    );
}
