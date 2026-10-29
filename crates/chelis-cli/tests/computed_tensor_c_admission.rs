//! Checked tensor consumers retain their dtype, geometry and guards across
//! computed operands: spec/04 sections 4.7.2-4.7.6 and [05-OP-11]/[05-OP-30].

use assert_cmd::Command;
use serde_json::Value;
use std::process::{Command as StdCommand, Output};
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn cli(source: &str, name: &str, arguments: &[&str]) -> Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    common::write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(arguments)
        .arg(&path)
        .output()
        .unwrap()
}

fn checked_eval(source: &str, name: &str) -> Value {
    let checked = cli(source, name, &["check"]);
    assert!(checked.status.success(), "{name}: {checked:?}");
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1);
    assert_eq!(report["errors"], serde_json::json!([]));
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert_eq!(report["untyped_nodes"], 0);
    let evaluated = cli(source, name, &["eval", "--json", "--file"]);
    assert!(evaluated.status.success(), "{name}: {evaluated:?}");
    serde_json::from_slice(&evaluated.stdout).unwrap()
}

fn root<'a>(report: &'a Value, name: &str) -> &'a Value {
    &report["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == name)
        .unwrap()["value"]
}

fn assert_native_matches_eval(source: &str, name: &str) -> String {
    assert!(common::gcc_available(), "this oracle executes generated C");
    let evaluated = cli(source, name, &["eval", "--file"]);
    assert!(evaluated.status.success(), "{name}: {evaluated:?}");
    let expected = String::from_utf8(evaluated.stdout).unwrap();
    let native = common::build_and_run(source, name);
    assert_eq!(native, expected, "{name}: complete lane observations");
    native
}

#[test]
fn inline_and_bound_range_tensors_keep_expand_admission() {
    for (name, body) in [
        (
            "range_inline",
            "expand(reshape(to_tensor(range(0i64, 3i64), i64), [1i64, 3i64]), 0i32, 2i64)",
        ),
        (
            "range_bound",
            "{\n base = reshape(to_tensor(range(0i64, 3i64), i64), [1i64, 3i64])\n expand(base, 0i32, 2i64)\n}",
        ),
        (
            "range_literal_control",
            "expand(reshape(to_tensor([0i64, 1i64, 2i64]), [1i64, 3i64]), 0i32, 2i64)",
        ),
    ] {
        let source = format!(
            "def rows() -> tensor[2, 3, i64] = {body}\nout = tensor_to_scalar(sum(sum(rows(), 1i32), 0i32))\n"
        );
        let report = checked_eval(&source, name);
        assert_eq!(root(&report, "out")["value"]["value"], 6);
        assert_eq!(
            root(&report, "rows")["value"]["shape"],
            serde_json::json!([2, 3])
        );
        assert_eq!(
            root(&report, "rows")["value"]["data"]["values"],
            serde_json::json!([0, 1, 2, 0, 1, 2])
        );
        assert!(assert_native_matches_eval(&source, name).contains("out = 6\n"));
    }
}

fn mean_program(
    dtype: &str,
    symbolic: bool,
    reshaped: bool,
    tensor: bool,
    guarded: bool,
) -> String {
    let binders = if symbolic { "[n]" } else { "" };
    let extent = if symbolic { "n" } else { "3" };
    let operand = if reshaped {
        "reshape(x, [numel(x)])"
    } else {
        "x"
    };
    let mean = format!("mean({operand}, 0i32)");
    let (result_type, body, empty) = if tensor {
        (
            format!("tensor[{dtype}]"),
            mean,
            format!("scalar_to_tensor(0.0{dtype})"),
        )
    } else {
        (
            dtype.to_string(),
            format!("tensor_to_scalar({mean})"),
            format!("0.0{dtype}"),
        )
    };
    let body = if guarded {
        format!("if gt(numel(x), 0i64) then {body} else {empty}")
    } else {
        body
    };
    format!(
        "def average{binders}(x: tensor[{extent}, {dtype}]) -> {result_type} = {body}\nout = average(to_tensor([1.0{dtype}, 0.0{dtype}, 0.0{dtype}]))\n"
    )
}

fn mean_matrix(dtype: &str, expected_bits: &str) {
    for symbolic in [false, true] {
        for reshaped in [false, true] {
            for tensor in [false, true] {
                for guarded in [false, true] {
                    let name = format!("mean_{dtype}_{symbolic}_{reshaped}_{tensor}_{guarded}");
                    let source = mean_program(dtype, symbolic, reshaped, tensor, guarded);
                    let report = checked_eval(&source, &name);
                    let value = root(&report, "out");
                    if tensor {
                        assert_eq!(value["type"], "tensor");
                        assert_eq!(value["value"]["shape"], serde_json::json!([]));
                        assert_eq!(value["value"]["data"]["dtype"], dtype);
                        assert_eq!(
                            value["value"]["data"]["bits"],
                            serde_json::json!([expected_bits])
                        );
                    } else {
                        assert_eq!(value["type"], "scalar");
                        assert_eq!(value["value"]["dtype"], dtype);
                        assert_eq!(value["value"]["bits"], expected_bits);
                    }
                    assert_native_matches_eval(&source, &name);
                }
            }
        }
    }
}

#[test]
fn f16_mean_keeps_computed_tensor_admission() {
    mean_matrix("f16", "3555");
}
#[test]
fn f32_mean_keeps_computed_tensor_admission() {
    mean_matrix("f32", "3eaaaaab");
}
#[test]
fn f64_mean_keeps_computed_tensor_admission() {
    mean_matrix("f64", "3fd5555555555555");
}

#[test]
fn helper_and_tuple_inputs_keep_distinct_runtime_extents() {
    for dtype in ["f16", "f32", "f64"] {
        let source = format!(
            "def flatten_input[n](x: tensor[n, {dtype}]) -> tensor[*, {dtype}] = reshape(x, [numel(x)])\ndef average[n](pair: (tensor[n, {dtype}], bool)) -> {dtype} = if pair.1 then tensor_to_scalar(mean(flatten_input(pair.0), 0i32)) else 0.0{dtype}\none_third = average((to_tensor([1.0{dtype}, 0.0{dtype}, 0.0{dtype}]), true))\none_quarter = average((to_tensor([1.0{dtype}, 0.0{dtype}, 0.0{dtype}, 0.0{dtype}]), true))\n"
        );
        checked_eval(&source, "helper_tuple");
        let native = assert_native_matches_eval(&source, &format!("helper_tuple_{dtype}"));
        assert!(native.contains("one_quarter = 0.25\n"));
    }
}

#[test]
fn inline_and_bound_linspace_matmul_keep_the_same_values() {
    let (_dir, reef, app) = common::make_app("computed-matmul-admission");
    for bound in [false, true] {
        let lhs = "insert(linspace(0.0f32, 1.0f32, k), 0i32, m)";
        let binding = if bound {
            format!("  a = {lhs}\n")
        } else {
            String::new()
        };
        let lhs = if bound { "a" } else { lhs };
        let source = format!(
            "module Demo.Main\nimport Std.Tensor.Construct (linspace)\ndef run(m: i64, k: i64, n: i64) -> tensor[f32] = {{\n w = insert(linspace(0.0f32, 1.0f32, n), 0i32, k)\n{binding} y = matmul({lhs}, w)\n sum(sum(y, 1i32), 0i32)\n}}\ntotal = run(2i64, 3i64, 4i64)\n"
        );
        common::write_file(&app.join("src/main.ch"), &source);
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .current_dir(&app)
            .args(["eval", "--file", "src/main.ch"])
            .output()
            .unwrap();
        assert!(eval.status.success(), "{eval:?}");
        let native = common::build_and_run_app(&reef, &app, "main");
        assert_eq!(native, String::from_utf8(eval.stdout).unwrap());
        assert!(native.contains("total = 6.0\n"), "{native}");
    }
}

#[test]
fn invalid_axis_and_dtype_still_reject_before_emission() {
    for source in [
        "def average[n](x: tensor[n, f32]) -> f32 = tensor_to_scalar(mean(reshape(x, [numel(x)]), 1i32))\n",
        "def average[n](x: tensor[n, bool]) -> bool = tensor_to_scalar(mean(reshape(x, [numel(x)]), 0i32))\n",
        "def rows() -> tensor[2, 3, i64] = expand(reshape(to_tensor(range(0i64, 3i64), i64), [1i64, 3i64]), 2i32, 2i64)\n",
        "def average[n](x: tensor[n, f32]) -> f32 ! {} = {\n _ = print(\"forbidden\")\n tensor_to_scalar(mean(reshape(x, [numel(x)]), 0i32))\n}\n",
        "def average[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> f32 = tensor_to_scalar(mean(add(x, y), 0i32))\n",
        "def average[n](x: tensor[n, f32]) -> f64 = tensor_to_scalar(mean(reshape(x, [numel(x)]), 0i32))\n",
        "def average[n](k: key, x: tensor[n, f32]) -> f32 = {\n first = dropout(k, x, 0.5f32)\n tensor_to_scalar(mean(dropout(k, reshape(first, [numel(first)]), 0.5f32), 0i32))\n}\n",
    ] {
        let checked = cli(source, "invalid_admission", &["check"]);
        assert!(!checked.status.success(), "{checked:?}");
        let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(!report["errors"].as_array().unwrap().is_empty());
        assert!(report["score"].as_f64().unwrap() < 1.0);
    }
}

#[test]
fn operand_and_extent_effects_execute_once_in_source_order() {
    let source = "def operand(marker: i64) -> tensor[3, f32] ! { IO } = {\n _ = print(\"operand\")\n to_tensor([1.0f32, 0.0f32, 0.0f32])\n}\ndef extent(marker: i64) -> i64 ! { IO } = {\n _ = print(\"extent\")\n 3i64\n}\ndef average(marker: i64) -> f32 ! { IO } = tensor_to_scalar(mean(reshape(operand(marker), [extent(marker)]), 0i32))\nout = average(0i64)\n";
    let native = assert_native_matches_eval(source, "effect_order");
    assert_eq!(native.matches("operand\n").count(), 1, "{native}");
    assert_eq!(native.matches("extent\n").count(), 1, "{native}");
    assert!(native.starts_with("operand\nextent\n"), "{native}");
}

#[test]
fn computed_expand_retains_unit_axis_checks_and_empty_results() {
    let positive = "def rows(size: i64) -> tensor[*, 3, i64] = expand(reshape(to_tensor(range(0i64, 3i64), i64), [1i64, 3i64]), 0i32, size)\nout = rows(0i64)\n";
    let report = checked_eval(positive, "expand_empty");
    assert_eq!(
        root(&report, "out")["value"]["shape"],
        serde_json::json!([0, 3])
    );
    assert_native_matches_eval(positive, "expand_empty");
    let negative = "def rows(size: i64) -> tensor[2, 3, i64] = expand(reshape(to_tensor(range(0i64, mul(size, 3i64)), i64), [size, 3i64]), 0i32, 2i64)\nout = rows(2i64)\n";
    for output in [
        cli(negative, "expand_nonunit", &["eval", "--file"]),
        build_and_execute(negative, "expand_nonunit"),
    ] {
        assert!(!output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("numeric trap: domain in expand at i64"),
            "{output:?}"
        );
    }
}

fn build_and_execute(source: &str, name: &str) -> Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    let out = dir.path().join("native");
    common::write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build"])
        .arg(&path)
        .arg("-o")
        .arg(&out)
        .assert()
        .success();
    StdCommand::new(out.join(name)).output().unwrap()
}

#[test]
fn a_computed_reshape_keeps_its_product_guard_before_mean() {
    let source = "def average[n](x: tensor[n, f32], extra: i64) -> f32 = tensor_to_scalar(mean(reshape(x, [add(numel(x), extra)]), 0i32))\nout = average(to_tensor([1.0f32, 0.0f32, 0.0f32]), 1i64)\n";
    let evaluated = cli(source, "bad_product", &["eval", "--file"]);
    let native = build_and_execute(source, "bad_product");
    for output in [evaluated, native] {
        assert!(!output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("numeric trap: domain in reshape at i64"),
            "{output:?}"
        );
    }
}

#[test]
fn mean_keeps_empty_axis_traps_and_untaken_guarded_branches() {
    for dtype in ["f16", "f32", "f64"] {
        for guarded in [false, true] {
            let reduction = "tensor_to_scalar(mean(reshape(x, [numel(x)]), 0i32))";
            let body = if guarded {
                format!("if gt(numel(x), 0i64) then {reduction} else 0.0{dtype}")
            } else {
                reduction.to_string()
            };
            let source = format!(
                "def average[n](x: tensor[n, {dtype}]) -> {dtype} = {body}\ndef empty(size: i64) -> tensor[*, {dtype}] = insert(scalar_to_tensor(0.0{dtype}), 0i32, size)\nout = average(empty(0i64))\n"
            );
            let name = format!("empty_mean_{dtype}_{guarded}");
            let evaluated = cli(&source, &name, &["eval", "--file"]);
            let native = build_and_execute(&source, &name);
            if guarded {
                assert!(evaluated.status.success(), "{evaluated:?}");
                assert!(native.status.success(), "{native:?}");
                assert_eq!(native.stdout, evaluated.stdout);
            } else {
                for output in [evaluated, native] {
                    assert!(!output.status.success(), "{output:?}");
                    assert!(
                        String::from_utf8_lossy(&output.stderr)
                            .contains(&format!("numeric trap: domain in mean at {dtype}")),
                        "{output:?}"
                    );
                }
            }
        }
    }
}
