//! A tensor builtin must keep its tensor execution path when an operand is
//! computed by a non-recursive helper or a nested host value expression.
//! [05-OP-30] defines the reduction, and spec/04 §4.7.2 admits an inline
//! `i64` extent expression. The accepted programs must agree between eval
//! and an executable C build. The invalid twin must still be rejected.

use assert_cmd::Command;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

const HELPER_SUM: &str = "module Demo.Main
def running(x: tensor[3, f32]) -> tensor[3, f32] = cumsum(x, 0i32)
def total(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(running(x), 0i32))
out = [1.0f32, 2.0f32, 3.0f32] |> to_tensor |> total
";

const RECURSIVE_HELPER_SUM: &str = "module Demo.Main
def double(x: tensor[4, f32]) -> tensor[4, f32] = add(x, x)
def walk(n: i64, x: tensor[4, f32]) -> tensor[4, f32] = if eq(n, 0i64) then x else walk(sub(n, 1i64), double(x))
def main() -> f32 = tensor_to_scalar(sum(walk(4i64, to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])), 0i32))
";

const INLINE_INSERT_SIZE: &str = "module Demo.Main
def h(xs: List[i64], t: tensor[2, i64]) = insert(t, 0i32, index(xs, 1i64))
out = h([3i64, 4i64], to_tensor([1i64, 2i64]))
";

const INLINE_TENSOR_EXTENT: &str = "module Demo.Main
def from_values(dims: tensor[1, i64]) -> tensor[*, f32] = insert(scalar_to_tensor(1.0f32), 0i32, index(to_list(dims), 0i64))
v = from_values(to_tensor([2i64]))
";

// The two-width School pinned probe, with its imported depthwise helper
// embedded so the compiler regression remains executable in this repository.
const SCHOOL_TWO_WIDTH: &str = include_str!("fixtures/issue_2939/depthwise_two_width_sum.ch");

const INVALID_REDUCTION: &str = "module Demo.Main
def running(x: tensor[3, bool]) -> tensor[3, bool] = x
out = sum(running(to_tensor([true, false, true])), 0i32)
";

const NEGATIVE_INLINE_INSERT_SIZE: &str = "module Demo.Main
def h(xs: List[i64], t: tensor[2, i64]) = insert(t, 0i32, index(xs, 0i64))
out = h([-3i64], to_tensor([1i64, 2i64]))
";

fn eval(source: &str, name: &str) -> std::process::Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn nested_tensor_operands_agree_with_executable_c() {
    if !common::gcc_available() {
        return;
    }
    for (name, source, expected) in [
        ("helper_sum", HELPER_SUM, "out = 10.0\n"),
        (
            "recursive_helper_sum",
            RECURSIVE_HELPER_SUM,
            "main = 160.0\n",
        ),
        (
            "inline_insert_size",
            INLINE_INSERT_SIZE,
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])\n",
        ),
        (
            "inline_tensor_extent",
            INLINE_TENSOR_EXTENT,
            "v = tensor(shape=[2], data=[1.0, 1.0])\n",
        ),
        ("depthwise_two_width_sum", SCHOOL_TWO_WIDTH, "main = 81.0\n"),
    ] {
        let evaluated = eval(source, name);
        assert!(
            evaluated.status.success(),
            "{name}: eval: {}",
            String::from_utf8_lossy(&evaluated.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&evaluated.stdout), expected);
        assert_eq!(common::build_and_run(source, name), expected, "{name}");
    }
}

#[test]
fn a_bool_helper_result_still_cannot_be_summed() {
    let evaluated = eval(INVALID_REDUCTION, "invalid_reduction");
    assert!(!evaluated.status.success());
    assert!(
        String::from_utf8_lossy(&evaluated.stderr).contains("sum"),
        "{}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
}

#[test]
fn an_inline_insert_size_preserves_the_runtime_domain_trap() {
    if !common::gcc_available() {
        return;
    }
    let expected = "insert target extent at axis 0 is negative: -3\n\
numeric trap: domain in insert at i64\n";
    let evaluated = eval(NEGATIVE_INLINE_INSERT_SIZE, "negative_insert_size");
    assert!(!evaluated.status.success());
    assert_eq!(String::from_utf8_lossy(&evaluated.stderr), expected);

    let dir = tempdir().unwrap();
    let path = dir.path().join("negative_insert_size.ch");
    let out = dir.path().join("out");
    std::fs::write(&path, NEGATIVE_INLINE_INSERT_SIZE).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    let ran = StdCommand::new(out.join("negative_insert_size"))
        .output()
        .unwrap();
    assert!(!ran.status.success());
    assert_eq!(String::from_utf8_lossy(&ran.stderr), expected);
}
