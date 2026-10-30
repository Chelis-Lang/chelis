//! spec/04 §4.7.2-4.7.4 and [05-OP-11]/[05-OP-65]: precision
//! specialization preserves the rank of runtime-sized inserted axes.

use assert_cmd::Command;
use serde_json::Value;
use std::process::{Command as StdCommand, Output};
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

#[derive(Clone, Copy, Debug)]
enum Genericity {
    Concrete,
    Precision,
    Dimensions,
    Both,
}

impl Genericity {
    fn precision(self, dtype: &str) -> &str {
        match self {
            Self::Precision | Self::Both => "p",
            Self::Concrete | Self::Dimensions => dtype,
        }
    }

    fn dimensions(self) -> bool {
        matches!(self, Self::Dimensions | Self::Both)
    }

    fn binders(self, matrix: bool) -> &'static str {
        match (self, matrix) {
            (Self::Concrete, _) => "",
            (Self::Precision, false) => "[p: {f32, f64, f16}]",
            (Self::Precision, true) => "[p: Float]",
            (Self::Dimensions, false) => "[d0]",
            (Self::Dimensions, true) => "[d0, d1]",
            (Self::Both, false) => "[d0, p: {f32, f64, f16}]",
            (Self::Both, true) => "[d0, d1, p: Float]",
        }
    }
}

fn cli(source: &str, name: &str, arguments: &[&str]) -> Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    common::write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(arguments)
        .arg(path)
        .output()
        .unwrap()
}

fn assert_checked(source: &str, name: &str) {
    let checked = cli(source, name, &["check"]);
    assert!(checked.status.success(), "{name}: {checked:?}");
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1, "{name}: {report}");
    assert_eq!(report["errors"], serde_json::json!([]));
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert_eq!(report["untyped_nodes"], 0);
}

fn assert_lanes(source: &str, name: &str, expected: &str) {
    assert!(
        common::gcc_available(),
        "this regression executes generated C"
    );
    assert_checked(source, name);
    let evaluated = cli(source, name, &["eval", "--file"]);
    assert!(evaluated.status.success(), "{name}: {evaluated:?}");
    assert_eq!(String::from_utf8(evaluated.stdout).unwrap(), expected);
    assert_eq!(common::build_and_run(source, name), expected);
}

fn scale_program(dtype: &str, mode: Genericity, extent: usize) -> (String, String) {
    let precision = mode.precision(dtype);
    let dim = if mode.dimensions() {
        "d0".to_string()
    } else {
        extent.to_string()
    };
    let size = if matches!(mode, Genericity::Precision) {
        format!("{extent}i64")
    } else {
        "shape(a, 0i32)".to_string()
    };
    let values = (1..=extent)
        .map(|i| format!("{}.0", 2 * i))
        .collect::<Vec<_>>()
        .join(", ");
    let expected = (1..=extent)
        .map(|i| format!("{i}.0"))
        .collect::<Vec<_>>()
        .join(", ");
    (
        format!(
            "module Witness\nexport (case_0)\n\
             def scale{}(a: tensor[{dim}, {precision}], s: f64) -> tensor[{dim}, {precision}] = {{\n\
             \x20 value0 = cast(s, {precision})\n\
             \x20 mul(a, insert(reshape(to_tensor([value0]), []), 0i32, {size}))\n}}\n\
             def case_0(a: tensor[{extent}, {dtype}]) -> tensor[{extent}, {dtype}] = scale(a, 0.5f64)\n\
             out = case_0(to_tensor([{values}], {dtype}))\n",
            mode.binders(false),
        ),
        format!("out = tensor(shape=[{extent}], data=[{expected}])\n"),
    )
}

fn mean_program(dtype: &str, mode: Genericity, rows: usize, cols: usize) -> (String, String) {
    let precision = mode.precision(dtype);
    let dims = if mode.dimensions() {
        "d0, d1".to_string()
    } else {
        format!("{rows}, {cols}")
    };
    let (size0, size1) = if matches!(mode, Genericity::Precision) {
        (format!("{rows}i64"), format!("{cols}i64"))
    } else {
        ("shape(x, 0i32)".to_string(), "shape(x, 1i32)".to_string())
    };
    let count = rows * cols;
    let values = (1..=count)
        .map(|i| format!("{i}.0"))
        .collect::<Vec<_>>()
        .join(", ");
    // These small integer inputs have an exactly represented half-integer
    // mean and result in each tested dtype, independently of either lane.
    let average = (count + 1) as f64 / 2.0;
    let expected = (1..=count)
        .map(|i| format!("{:.1}", i as f64 + average))
        .collect::<Vec<_>>()
        .join(", ");
    (
        format!(
            "module Repro\nexport (case_0)\n\
             def centre{}(x: tensor[{dims}, {precision}]) -> tensor[{dims}, {precision}] = {{\n\
             \x20 avg = tensor_to_scalar(mean(reshape(x, [numel(x)]), 0i32))\n\
             \x20 add(x, insert(insert(reshape(to_tensor([avg]), []), 0i32, {size0}), 1i32, {size1}))\n}}\n\
             def case_0(x: tensor[{rows}, {cols}, {dtype}]) -> tensor[{rows}, {cols}, {dtype}] = centre(x)\n\
             out = case_0(reshape(to_tensor([{values}], {dtype}), [{rows}i64, {cols}i64]))\n",
            mode.binders(true),
        ),
        format!("out = tensor(shape=[{rows}, {cols}], data=[{expected}])\n"),
    )
}

#[test]
fn inserted_scalar_preserves_rank_under_dimension_and_precision_specialization() {
    for dtype in ["f16", "f32", "f64"] {
        for mode in [
            Genericity::Concrete,
            Genericity::Precision,
            Genericity::Dimensions,
            Genericity::Both,
        ] {
            let name = format!("scale_{dtype}_{mode:?}").to_lowercase();
            let (source, expected) = scale_program(dtype, mode, 3);
            assert_lanes(&source, &name, &expected);
        }
    }
    for extent in [1, 5] {
        let (source, expected) = scale_program("f16", Genericity::Both, extent);
        assert_lanes(&source, &format!("scale_f16_{extent}"), &expected);
    }
}

#[test]
fn one_helper_retains_independent_precision_and_dimension_instantiations() {
    let (mut source, mut expected) = scale_program("f16", Genericity::Both, 3);
    source.push_str(
        "out32 = scale(to_tensor([8.0f32]), 0.5f64)\n\
         out64 = scale(to_tensor([6.0f64, 10.0f64]), 0.5f64)\n",
    );
    expected.push_str(
        "out32 = tensor(shape=[1], data=[4.0])\n\
         out64 = tensor(shape=[2], data=[3.0, 5.0])\n",
    );
    assert_lanes(&source, "scale_independent_calls", &expected);
}

#[test]
fn all_axis_mean_addback_preserves_both_inserted_axes() {
    for dtype in ["f16", "f32", "f64"] {
        for mode in [
            Genericity::Concrete,
            Genericity::Precision,
            Genericity::Dimensions,
            Genericity::Both,
        ] {
            let name = format!("centre_{dtype}_{mode:?}").to_lowercase();
            let (source, expected) = mean_program(dtype, mode, 2, 3);
            assert_lanes(&source, &name, &expected);
        }
    }
    for (rows, cols) in [(1, 1), (3, 2)] {
        let (source, expected) = mean_program("f16", Genericity::Both, rows, cols);
        assert_lanes(&source, &format!("centre_f16_{rows}_{cols}"), &expected);
    }
}

#[test]
fn bounded_precision_and_positive_extent_guards_retain_the_addback_composition() {
    for (rows, cols) in [(2, 3), (0, 3)] {
        let (source, expected) = mean_program("f16", Genericity::Both, rows, cols);
        let source = source
            .replace("p: Float", "p: {f32, f64, f16}")
            .replacen(
                "= {\n",
                "= if gt(shape(x, 0i32), 0i64) then if gt(shape(x, 1i32), 0i64) then {\n",
                1,
            )
            .replacen("\n}\ndef case_0", "\n} else x else x\ndef case_0", 1);
        assert_lanes(&source, &format!("centre_guarded_{rows}_{cols}"), &expected);
    }
}

#[test]
fn inserted_scalar_retains_static_shape_and_precision_rejections() {
    let (scale, _) = scale_program("f16", Genericity::Both, 3);
    for (name, source) in [
        ("insert_negative", scale.replace("shape(a, 0i32)", "-1i64")),
        ("insert_axis", scale.replace("[]), 0i32,", "[]), 1i32,")),
        (
            "insert_precision",
            scale.replace("cast(s, p)", "cast(s, f64)"),
        ),
    ] {
        let checked = cli(&source, name, &["check"]);
        let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(!checked.status.success(), "{name}: {checked:?}");
        assert!(report["score"].as_f64().unwrap() < 1.0);
        assert!(!report["errors"].as_array().unwrap().is_empty());
        for arguments in [["eval", "--file"].as_slice(), ["build"].as_slice()] {
            let output = cli(&source, name, arguments);
            assert!(!output.status.success(), "{name}: {output:?}");
        }
    }
}

#[test]
fn inserted_scalar_retains_the_reshape_element_count_guard() {
    let (source, _) = scale_program("f16", Genericity::Both, 3);
    let source = source.replace("to_tensor([value0])", "to_tensor([value0, value0])");
    assert_checked(&source, "reshape_product");
    for output in [
        cli(&source, "reshape_product", &["eval", "--file"]),
        execute_native(&source, "reshape_product"),
    ] {
        assert!(!output.status.success(), "{output:?}");
        assert!(output.stdout.is_empty(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("numeric trap: domain in reshape"),
            "{output:?}"
        );
    }
}

fn execute_native(source: &str, name: &str) -> Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{name}.ch"));
    let out = dir.path().join("native");
    common::write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .arg("-o")
        .arg(&out)
        .assert()
        .success();
    StdCommand::new(out.join(name)).output().unwrap()
}

#[test]
fn inserted_scalar_checks_runtime_sizes_and_result_claims() {
    let source = "def repeat[p: {f32, f64, f16}](s: p, size: i64) -> tensor[3, p] = insert(reshape(to_tensor([s]), []), 0i32, size)\nout = repeat(0.5f16, SIZEi64)\n";
    assert_lanes(
        &source.replace("SIZE", "3"),
        "insert_claim_equal",
        "out = tensor(shape=[3], data=[0.5, 0.5, 0.5])\n",
    );
    for size in ["-1", "2"] {
        let source = source.replace("SIZE", size);
        let name = format!("insert_claim_{}", size.replace('-', "negative"));
        assert_checked(&source, &name);
        for output in [
            cli(&source, &name, &["eval", "--file"]),
            execute_native(&source, &name),
        ] {
            assert!(!output.status.success(), "{name}: {output:?}");
            assert!(output.stdout.is_empty(), "{name}: {output:?}");
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("numeric trap: domain in insert"),
                "{name}: {output:?}"
            );
        }
    }
}
