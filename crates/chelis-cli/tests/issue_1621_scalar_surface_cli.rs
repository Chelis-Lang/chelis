//! #1621: invalid generic mixing fails at the CLI boundary; explicit shapes
//! retain bounded specialization and exact numerical values on eval and C.
mod common;

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn run(source: &str, args: &[&str]) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("input.ch"), source).expect("source");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(args)
        .output()
        .expect("run")
}

fn output(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn invalid_generic_programs_cannot_check_evaluate_or_build() {
    for op in [
        "cmplt",
        "eq",
        "neq",
        "lt",
        "gt",
        "lte",
        "gte",
        "add",
        "sub",
        "mul",
        "div",
        "floor_div",
        "trunc_div",
        "max_elem",
        "min_elem",
    ] {
        let bound = if ["floor_div", "trunc_div"].contains(&op) {
            "Int"
        } else {
            "Float"
        };
        for (lhs, rhs) in [("xs", "cast(1, p)"), ("cast(1, p)", "xs")] {
            let source = format!("def f[p: {bound}](xs: tensor[3, p]) = {op}({lhs}, {rhs})\n");
            for args in [
                &["check", "input.ch"][..],
                &["eval", "--file", "input.ch"],
                &["build", "input.ch", "--target", "c", "--output", "out"],
            ] {
                let out = run(&source, args);
                assert!(
                    !out.status.success(),
                    "{source}\n{args:?}\n{}",
                    output(&out)
                );
                assert!(
                    output(&out).contains("scalar beside a tensor"),
                    "{}",
                    output(&out)
                );
                assert!(
                    output(&out).contains("scalar_to_tensor"),
                    "{}",
                    output(&out)
                );
            }
        }
    }
}

#[test]
fn explicit_generic_shapes_execute_at_each_instantiation() {
    for (name, source, expected) in [
        (
            "generic_f32",
            "def f[p: Float](xs: tensor[3, p]) -> tensor[3, bool] = gt(xs, insert(scalar_to_tensor(cast(1.5, p)), 0i32, shape(xs, 0i32)))\ndef main() -> tensor[3, bool] = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
            "data=[false, true, true]",
        ),
        (
            "generic_f64",
            "def f[p: Float](xs: tensor[1, p]) -> tensor[1, p] = mul(xs, insert(scalar_to_tensor(cast(0.1, p)), 0i32, shape(xs, 0i32)))\ndef main() -> tensor[1, f64] = f(to_tensor([3.0f64]))\n",
            "0.30000000000000004",
        ),
        (
            "generic_int64",
            "def f[p: Int](xs: tensor[1, p]) -> tensor[1, p] = add(xs, insert(scalar_to_tensor(cast(1, p)), 0i32, shape(xs, 0i32)))\ndef main() -> tensor[1, i64] = f(to_tensor([9007199254740993i64]))\n",
            "9007199254740994",
        ),
    ] {
        let out = run(source, &["eval", "--file", "input.ch"]);
        assert!(
            out.status.success() && output(&out).contains(expected),
            "{name}: {}",
            output(&out)
        );
        let native = common::build_and_run(source, name);
        assert!(native.contains(expected), "{name}: {native}");
    }
}

#[test]
fn the_migration_example_executes_on_eval_and_c() {
    let source = include_str!("../../../examples/generic_explicit_shape.ch");
    let out = run(source, &["eval", "--file", "input.ch"]);
    assert!(out.status.success(), "{}", output(&out));
    assert!(
        output(&out).contains("data=[false, true, true]"),
        "{}",
        output(&out)
    );
    let native = common::build_and_run(source, "generic_explicit_shape");
    assert!(native.contains("data=[false, true, true]"), "{native}");
}
