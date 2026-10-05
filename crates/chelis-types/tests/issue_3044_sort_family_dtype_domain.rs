//! Issue #3044: `sort`, `cumsum`, `trace` and `clamp` admit only the
//! arithmetic dtypes.
//!
//! `spec/05-risc-primitives.md` [05-OP-53] gives the four operations "the
//! arithmetic dtype domains ... specified for their corresponding tensor
//! operations in [05-OP-33]", which admits signed-integer and float tensors
//! and makes bool a type error. The checker refuses a bool operand with a
//! precision diagnostic, so neither evaluation lane ever receives one.

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

/// `sum_result(p, default(p))` (spec/04 section 5.7.1), the result dtype of
/// `cumsum` and `trace`: i8 and i16 sum at i32, and every other admitted
/// dtype at itself.
fn sum_result(dtype: &str) -> &str {
    match dtype {
        "i8" | "i16" => "i32",
        other => other,
    }
}

/// One call per operation over a rank-two operand of dtype `dtype`.
fn call(op: &str, dtype: &str) -> String {
    let operand = format!("x: tensor[2, 2, {dtype}]");
    let summed = sum_result(dtype);
    match op {
        "sort" => format!(
            "def f({operand}) -> (tensor[2, 2, {dtype}], tensor[2, 2, i64]) = sort(x, 0i32)\n"
        ),
        "cumsum" => format!("def f({operand}) -> tensor[2, 2, {summed}] = cumsum(x, 0i32)\n"),
        "trace" => format!("def f({operand}) -> tensor[{summed}] = trace(x, 0i32, 1i32)\n"),
        "clamp" => format!(
            "def f({operand}, lo: tensor[2, 2, {dtype}], hi: tensor[2, 2, {dtype}]) -> tensor[2, 2, {dtype}] = clamp(x, lo, hi)\n"
        ),
        _ => unreachable!("not a [05-OP-53] arithmetic operation: {op}"),
    }
}

const OPS: [&str; 4] = ["sort", "cumsum", "trace", "clamp"];

#[test]
fn bool_operand_is_a_precision_error_for_every_arithmetic_operation() {
    for op in OPS {
        let report = check_ir_program(&surf_to_deep(&call(op, "bool")))
            .expect_err(&format!("{op} must refuse a bool tensor"));
        assert!(
            report.errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                    && error.message.contains(&format!("`{op}`"))
                    && error.message.contains("bool")
            }),
            "{op} on bool must carry a precision diagnostic naming the operation and bool, got {:?}",
            report.errors
        );
    }
}

#[test]
fn bool_operand_is_refused_at_top_level_too() {
    let deep = surf_to_deep(
        "module Demo.Main\nb_sorted = sort(to_tensor([true, false, true, false]), 0i32)\n",
    );
    let report = check_ir_program(&deep).expect_err("the #3044 witness must not check");
    assert!(
        report.errors.iter().any(
            |error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                && error.message.contains("`sort`")
        ),
        "got {:?}",
        report.errors
    );
}

#[test]
fn signed_integer_and_float_operands_are_admitted() {
    for op in OPS {
        for dtype in ["i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"] {
            if let Err(report) = check_ir_program(&surf_to_deep(&call(op, dtype))) {
                panic!("{op} must admit {dtype}, got {:?}", report.errors);
            }
        }
    }
}

/// A bounded binder satisfies the operand family. `cumsum`'s result is
/// `sum_result(p, default(p))`, which is `p` across `Float` but not across
/// `Numeric` (chelis#3009), so its generic form is bounded to `Float`.
#[test]
fn numeric_bound_binder_is_admitted_and_unbounded_binder_is_refused() {
    for bounded in [
        "def f[p: Numeric](x: tensor[3, p]) -> (tensor[3, p], tensor[3, i64]) = sort(x, 0i32)\n",
        "def f[p: Float](x: tensor[3, p]) -> tensor[3, p] = cumsum(x, 0i32)\n",
    ] {
        if let Err(report) = check_ir_program(&surf_to_deep(bounded)) {
            panic!(
                "a bounded binder satisfies the operation: {bounded}, got {:?}",
                report.errors
            );
        }
    }
    let unbounded = "def f[p](x: tensor[3, p]) -> tensor[3, p] = cumsum(x, 0i32)\n";
    let report = check_ir_program(&surf_to_deep(unbounded))
        .expect_err("an unbounded binder admits bool, so cumsum must refuse it");
    assert!(
        report
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
        "got {:?}",
        report.errors
    );
}
