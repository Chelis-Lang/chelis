//! chelis#729 Phase 2 reduction-width and trap matrix.
//!
//! The kernel split is incomplete if a reduction can still widen through
//! `f64`/`i64`: [04-NUM-8] fixes the accumulator arithmetic width and
//! [04-NUM-12] makes intermediate traps observable at the lane's documented
//! order. These end-to-end host-eval rows complement the structural kernel
//! boundary tests with values that a widened implementation gets wrong.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn eval(program: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("phase2_reduction.ch");
    write_file(&path, program);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("utf8 path")])
        .output()
        .expect("chelis eval should run");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).trim().to_string(),
        String::from_utf8_lossy(&output.stderr).trim().to_string(),
    )
}

fn eval_list(expr: &str) -> String {
    let program = format!("module Phase2.Reduction\nout = print(to_list({expr}))\n");
    let (ok, stdout, stderr) = eval(&program);
    assert!(ok, "eval failed for `{expr}`: {stderr}");
    stdout.lines().next().unwrap_or("").trim().to_string()
}

fn eval_value(expr: &str) -> String {
    let program = format!("module Phase2.Reduction\nout = print({expr})\n");
    let (ok, stdout, stderr) = eval(&program);
    assert!(ok, "eval failed for `{expr}`: {stderr}");
    stdout.lines().next().unwrap_or("").trim().to_string()
}

fn assert_numeric_trap(expr: &str, expected: &str) {
    let program = format!("module Phase2.Reduction\nout = print(to_list({expr}))\n");
    let (ok, stdout, stderr) = eval(&program);
    assert!(
        !ok,
        "`{expr}` unexpectedly succeeded with stdout `{stdout}`"
    );
    let diagnostic = expected;
    assert!(
        stderr.lines().any(|line| line.trim() == diagnostic),
        "`{expr}` must raise exact trap `{expected}`, got stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "a user numeric trap must not panic: {stderr}"
    );
}

fn assert_numeric_value_trap(expr: &str, expected: &str) {
    let program = format!("module Phase2.Reduction\nout = print({expr})\n");
    let (ok, stdout, stderr) = eval(&program);
    assert!(
        !ok,
        "`{expr}` unexpectedly succeeded with stdout `{stdout}`"
    );
    let diagnostic = expected;
    assert!(
        stderr.lines().any(|line| line.trim() == diagnostic),
        "`{expr}` must raise exact trap `{expected}`, got stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "numeric trap panicked: {stderr}"
    );
}

#[test]
fn window_sum_uses_declared_float_arithmetic_width() {
    assert_eq!(
        eval_list(
            "reduce_window_sum(\
             to_tensor([16777216.0f32, 1.0f32, -16777216.0f32]), [3i64], [1i64])"
        ),
        "[0.0]",
        "the sequential f32 accumulator rounds after the first add"
    );

    // Negative-parity control: a nearby exactly representable sum must stay
    // successful rather than acquiring a spurious precision trap.
    assert_eq!(
        eval_list("reduce_window_sum(to_tensor([1.0f32, 2.0f32, 3.0f32]), [3i64], [1i64])"),
        "[6.0]"
    );
}

#[test]
fn window_sum_preserves_exact_int64_above_two_pow_53() {
    assert_eq!(
        eval_list(
            "reduce_window_sum(\
             to_tensor([9007199254740992i64, 1i64]), [2i64], [1i64])"
        ),
        "[9007199254740993]"
    );
    assert_eq!(
        eval_list("reduce_window_sum(to_tensor([40i64, 2i64]), [2i64], [1i64])"),
        "[42]"
    );
}

#[test]
fn window_sum_traps_at_each_integer_operand_width() {
    for (dtype, values, control, expected_control) in [
        ("i8", "100i8, 100i8, -100i8", "40i8, 40i8, -40i8", "[40]"),
        (
            "i16",
            "30000i16, 30000i16, -30000i16",
            "1000i16, 2000i16, -1000i16",
            "[2000]",
        ),
        (
            "i32",
            "2147483647i32, 1i32, -1i32",
            "1000000i32, 2000000i32, -1000000i32",
            "[2000000]",
        ),
        (
            "i64",
            "9223372036854775807i64, 1i64, -1i64",
            "9007199254740992i64, 1i64, -1i64",
            "[9007199254740992]",
        ),
    ] {
        let expr = format!("reduce_window_sum(to_tensor([{values}]), [3i64], [1i64])");
        assert_numeric_trap(
            &expr,
            &format!("numeric trap: overflow in reduce_window_sum at {dtype}"),
        );

        let control_expr = format!("reduce_window_sum(to_tensor([{control}]), [3i64], [1i64])");
        assert_eq!(
            eval_list(&control_expr),
            expected_control,
            "{dtype} control"
        );
    }
}

#[test]
fn ordinary_sum_observes_stride4_intermediate_overflow() {
    assert_numeric_value_trap(
        "sum(to_tensor([2147483647i32, 1i32, -1i32]), 0)",
        "numeric trap: overflow in sum at i32",
    );
    assert_eq!(
        eval_value("sum(to_tensor([2147483646i32, 1i32, -1i32]), 0)"),
        "2147483646"
    );
}

#[test]
fn arg_reductions_compare_exact_int64_values() {
    assert_eq!(
        eval_value(
            "argmax_reduce(\
             to_tensor([9007199254740992i64, 9007199254740993i64]), 0)"
        ),
        "1"
    );
    assert_eq!(
        eval_value(
            "argmin_reduce(\
             to_tensor([9007199254740993i64, 9007199254740992i64]), 0)"
        ),
        "1"
    );

    assert_eq!(
        eval_value("argmax_reduce(to_tensor([4i64, 9i64, 2i64]), 0)"),
        "1"
    );
    assert_eq!(
        eval_value("argmin_reduce(to_tensor([4i64, 9i64, 2i64]), 0)"),
        "2"
    );
}
