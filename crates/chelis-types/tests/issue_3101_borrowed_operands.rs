//! Issue #3101: the read-only operand of `cast` admits a borrowed tensor.
//!
//! `spec/05-risc-primitives.md` section 1.3.1 types read-only tensor
//! primitive parameters as `&tensor[...]`, and [05-OP-63] makes a tensor
//! `value` one: an owned argument auto-borrows and an explicit borrow `&x` is
//! admitted unchanged. The borrow changes nothing else, so the result and
//! every refusal are the owned call's.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{check_ir_program, check_linearity, check_typed_program};

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

#[test]
fn borrowed_cast_source_returns_the_target_dtype_at_the_source_shape() {
    for (source, target) in [
        ("f32", "f64"),
        ("f64", "f16"),
        ("i32", "f32"),
        ("bool", "i64"),
    ] {
        let program = format!(
            "def f(x: tensor[2, 3, {source}]) -> tensor[2, 3, {target}] = cast(&x, {target})\n"
        );
        if let Err(report) = check_ir_program(&surf_to_deep(&program)) {
            panic!(
                "cast(&x, {target}) from {source} must check, got {:?}",
                report.errors
            );
        }
    }
}

#[test]
fn borrowed_cast_source_stays_readable_after_the_cast() {
    let program =
        "def f(x: tensor[3, f32]) -> (tensor[3, f64], tensor[3, f32]) = (cast(&x, f64), exp(x))\n";
    let errors = linearity_errors_of(program);
    assert!(
        errors.is_empty(),
        "a borrow does not consume x, got {errors:?}"
    );
}

#[test]
fn borrowed_cast_result_is_not_the_source_dtype() {
    let program = "def f(x: tensor[2, f32]) -> tensor[2, f32] = cast(&x, f64)\n";
    let report = check_ir_program(&surf_to_deep(program))
        .expect_err("the result carries the target dtype, never the source's");
    assert!(
        !report.errors.iter().any(|error| error
            .message
            .contains("expected tensor or numeric/bool scalar")),
        "the refusal must be the result mismatch, not a source-kind error: {:?}",
        report.errors
    );
}

#[test]
fn borrowed_cast_keeps_the_owned_refusals() {
    let unsupported = "def f(x: tensor[2, f32]) -> tensor[2, string] = cast(&x, string)\n";
    check_ir_program(&surf_to_deep(unsupported)).expect_err("string is no cast target");
    let shape = "def f(x: tensor[2, f32]) -> tensor[3, f64] = cast(&x, f64)\n";
    check_ir_program(&surf_to_deep(shape)).expect_err("a cast preserves the source shape");
}

#[test]
fn borrowed_scalar_is_still_not_a_cast_source() {
    let program = "def f(x: f32) -> f64 = cast(&x, f64)\n";
    let report =
        check_ir_program(&surf_to_deep(program)).expect_err("section 1.3.1 borrows tensors only");
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.message.contains("borrow requires tensor")),
        "the refusal names the borrow, got {:?}",
        report.errors
    );
}

const INT_OPS: [&str; 6] = ["mod", "bitand", "bitor", "bitxor", "shl", "shr"];

fn errors_of(source: &str) -> Vec<String> {
    match check_ir_program(&surf_to_deep(source)) {
        Ok(_) => vec![],
        Err(report) => report.errors.iter().map(|e| e.message.clone()).collect(),
    }
}

/// [05-OP-47] and [05-OP-64]: both tensor operands are read-only
/// `&tensor[D,p]` parameters, so an explicit borrow types as the owned call.
#[test]
fn borrowed_integer_tensor_operands_return_the_operand_type() {
    for op in INT_OPS {
        for dtype in ["i8", "i16", "i32", "i64"] {
            let source = format!(
                "def f(x: tensor[2, 3, {dtype}], y: tensor[2, 3, {dtype}]) -> tensor[2, 3, {dtype}] = {op}(&x, &y)\n"
            );
            let errors = errors_of(&source);
            assert!(errors.is_empty(), "{op}(&x, &y) over {dtype}: {errors:?}");
        }
    }
}

#[test]
fn borrowed_integer_operands_stay_readable() {
    for op in INT_OPS {
        let source = format!(
            "def f(x: tensor[2, i32], y: tensor[2, i32]) -> (tensor[2, i32], tensor[2, i32], tensor[2, i32]) = ({op}(&x, &y), neg(x), neg(y))\n"
        );
        let errors = linearity_errors_of(&source);
        assert!(
            errors.is_empty(),
            "{op}: a borrow consumes nothing: {errors:?}"
        );
        let owned = format!(
            "def f(x: tensor[2, i32], y: tensor[2, i32]) -> (tensor[2, i32], tensor[2, i32]) = ({op}(x, y), neg(x))\n"
        );
        let errors = linearity_errors_of(&owned);
        assert!(
            errors.is_empty(),
            "{op}: an owned operand auto-borrows: {errors:?}"
        );
    }
}

/// A genuine mismatch through a borrow names its real cause, never a
/// dtype mismatch between identical dtypes.
#[test]
fn borrowed_integer_mismatches_name_the_real_cause() {
    for op in INT_OPS {
        let dtypes = format!(
            "def f(x: tensor[2, i32], y: tensor[2, i64]) -> tensor[2, i32] = {op}(&x, &y)\n"
        );
        let errors = errors_of(&dtypes);
        assert!(
            errors
                .iter()
                .any(|m| m.contains("i32") && m.contains("i64")),
            "{op}: {errors:?}"
        );
        let shapes = format!(
            "def f(x: tensor[2, i32], y: tensor[3, i32]) -> tensor[2, i32] = {op}(&x, &y)\n"
        );
        let errors = errors_of(&shapes);
        assert!(!errors.is_empty(), "{op}: shapes differ");
        assert!(
            !errors
                .iter()
                .any(|m| m.contains("matching integer") || m.contains("one dtype")),
            "{op}: a shape mismatch is not a dtype mismatch: {errors:?}"
        );
        let result = format!(
            "def f(x: tensor[2, i32], y: tensor[2, i32]) -> tensor[2, i64] = {op}(&x, &y)\n"
        );
        assert!(!errors_of(&result).is_empty(), "{op}: the result is i32");
        let floats = format!(
            "def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = {op}(&x, &y)\n"
        );
        // [05-OP-64] admits float `mod` (chelis#626); the bitwise and shift
        // operations stay integer-only.
        assert_eq!(errors_of(&floats).is_empty(), op == "mod", "{op}: floats");
    }
}

#[test]
fn borrowed_cast_trunc_source_is_admitted() {
    let source = "def f(x: tensor[3, f32]) -> (tensor[3, i32], tensor[3, f32]) = (cast_trunc(&x, i32), exp(x))\n";
    let errors = linearity_errors_of(source);
    assert!(errors.is_empty(), "{errors:?}");
    let wrong = "def f(x: tensor[3, i32]) -> tensor[3, i64] = cast_trunc(&x, i64)\n";
    assert!(
        errors_of(wrong)
            .iter()
            .any(|m| m.contains("is not a float dtype")),
        "cast_trunc keeps its float-source rule through a borrow"
    );
}

/// The type checker's and then linearity's messages, as
/// reports them.
fn linearity_errors_of(source: &str) -> Vec<String> {
    let checked = match check_typed_program(&surf_to_deep(source)) {
        Ok(checked) => checked,
        Err(report) => return report.errors.iter().map(|e| e.message.clone()).collect(),
    };
    match check_linearity(&checked) {
        Ok(_) => vec![],
        Err(errors) => errors.iter().map(|e| e.message.clone()).collect(),
    }
}

/// Every read-only tensor operation and the operand dtype it admits.
const READ_ONLY_OPS: [(&str, &str); 10] = [
    ("cast(x, f64)", "f32"),
    ("cast_trunc(x, i32)", "f32"),
    ("cast_saturate(x, i16)", "f32"),
    ("cast_wrap(x, i8)", "i32"),
    ("mod(x, x)", "i32"),
    ("bitand(x, x)", "i32"),
    ("bitor(x, x)", "i32"),
    ("bitxor(x, x)", "i32"),
    ("shl(x, x)", "i32"),
    ("shr(x, x)", "i32"),
];

/// [05-OP-63], [05-OP-6], [05-OP-23], [05-OP-24], [05-OP-47] and
/// [05-OP-64]: an owned argument auto-borrows, so the source stays readable
/// after the operation, at a def's body and at top level.
#[test]
fn an_owned_read_only_operand_auto_borrows() {
    for (call, dtype) in READ_ONLY_OPS {
        let body = format!(
            "def f(x: tensor[3, {dtype}]) -> tensor[3, {dtype}] = {{\n  a = {call}\n  add(&x, &x)\n}}\n"
        );
        let errors = linearity_errors_of(&body);
        assert!(errors.is_empty(), "`{call}` consumed x: {errors:?}");
        let top =
            format!("x = to_tensor([1{dtype}, 2{dtype}, 3{dtype}])\na = {call}\nb = add(&x, &x)\n");
        let errors = linearity_errors_of(&top);
        assert!(
            errors.is_empty(),
            "top-level `{call}` consumed x: {errors:?}"
        );
    }
}

/// The negative twin: a position that takes its operand by value (here a
/// tuple element of the result) still consumes it after the read-only call,
/// so a later borrow is refused.
#[test]
fn a_consuming_operation_still_consumes_beside_a_read_only_one() {
    for (call, dtype) in READ_ONLY_OPS {
        let body = format!(
            "def f(x: tensor[3, {dtype}]) -> (tensor[3, {dtype}], tensor[3, {dtype}]) = {{\n  a = {call}\n  (x, add(&x, &x))\n}}\n"
        );
        let errors = linearity_errors_of(&body);
        assert!(
            errors.iter().any(|m| m.contains("already consumed")),
            "returning x after `{call}` must consume it: {errors:?}"
        );
    }
}
