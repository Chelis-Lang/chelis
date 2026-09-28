//! spec/04 §4.7 and spec/06 §2.1: captures retain actual tensor axes and claims.
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;
use common::{build_and_run, parse_tensor_data};

fn write_file(path: &std::path::Path, source: &str) {
    common::write_file(
        path,
        &chelis_surf::format::format_source(source).expect("canonical Surf"),
    );
}

fn assert_gradient(source: &str, expected: &[f64], stem: &str) {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for (lane, text) in [
        ("eval", String::from_utf8(output.stdout).unwrap()),
        ("C", build_and_run(source, stem)),
    ] {
        assert!(
            text.contains(&format!("out = tensor(shape=[{}],", expected.len())),
            "{lane}: {text}"
        );
        assert_eq!(parse_tensor_data(&text, "out"), expected, "{lane}");
    }
}

#[test]
fn literal_tensor_capture_retains_its_axes() {
    assert_gradient(
        r#"
def loss(a: tensor[2, f32], s: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(add(a, s), 0i32))
def consume(g: tensor[2, f32]) -> tensor[2, f32] = g
out = {
  s = to_tensor([0.5f32, 0.5f32])
  g = grad(fn (x: tensor[2, f32]) -> loss(x, s))(to_tensor([1.0f32, 2.0f32]))
  consume(g)
}
"#,
        &[1.0, 1.0],
        "literal_capture",
    );
}

#[test]
fn symbolic_tensor_capture_retains_its_axes() {
    assert_gradient(
        r#"
def loss[n](a: tensor[n, f32], s: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(mul(a, s), 0i32))
def gradient[n](a: tensor[n, f32], s: tensor[n, f32]) -> tensor[n, f32] = {
  g = grad(fn (x: tensor[n, f32]) -> loss(x, s))(a)
  g
}
out = gradient(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32]))
"#,
        &[3.0, 5.0],
        "symbolic_capture",
    );
}

#[test]
fn direct_tensor_arguments_remain_the_control() {
    for slot in [0, 1] {
        let source = format!(
            "def loss(a: tensor[2, f32], s: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(add(a, s), 0i32))\nout = (grad(loss)(to_tensor([1.0f32, 2.0f32]), to_tensor([0.5f32, 0.5f32]))).{slot}\n"
        );
        assert_gradient(&source, &[1.0, 1.0], "direct_capture");
    }
}

#[test]
fn tensor_capture_shape_disagreement_keeps_the_entry_guard() {
    let source = r#"
def loss[n](a: tensor[n, f32], s: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(add(a, s), 0i32))
def gradient(a: tensor[2, f32], s: tensor[*, f32]) -> tensor[2, f32] = {
  g = grad(fn (x: tensor[2, f32]) -> loss(x, s))(a)
  g
}
out = gradient(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32, 7.0f32]))
"#;
    let dir = tempdir().unwrap();
    let path = dir.path().join("capture_guard.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    let out = dir.path().join("c");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(&out)
        .assert()
        .success();
    assert!(common::link_generated(&out, "capture_guard.c", "capture_guard").success());
    let native = std::process::Command::new(out.join("capture_guard"))
        .output()
        .unwrap();
    for (lane, result) in [("eval", eval), ("C", native)] {
        assert!(
            !result.status.success(),
            "{lane} accepted mismatching capture"
        );
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            text.contains("extent `n`: a axis 0 = 2, s axis 0 = 3"),
            "{lane}: {text}"
        );
    }
}

#[test]
fn captured_tensor_gradient_runs_in_the_native_test_runner() {
    let dir = tempdir().unwrap();
    common::write_file(
        &dir.path().join("reef.toml"),
        &format!(
            "[package]\nname = \"capture-extents\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Capture\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    );
    write_file(
        &dir.path().join("src/main.ch"),
        "module Capture.Main\ndef main() -> unit = ()\n",
    );
    let path = dir.path().join("tests/capture.ch");
    write_file(
        &path,
        r#"
module Capture.Tests

def loss(a: tensor[2, f32], s: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(add(a, s), 0i32))
def test_tensor_capture() -> unit ! { Test } = {
  a = to_tensor([1.0f32, 2.0f32])
  s = to_tensor([0.5f32, 0.5f32])
  g = grad(fn (x: tensor[2, f32]) -> loss(x, s))(a)
  test_assert_close_tensor(g, to_tensor([1.0f32, 1.0f32]), 0.0f32, "captured tensor gradient")
}
def test_direct_tensor_arguments() -> unit ! { Test } = {
  g = grad(loss)(to_tensor([1.0f32, 2.0f32]), to_tensor([0.5f32, 0.5f32]))
  _ = test_assert_close_tensor(g.0, to_tensor([1.0f32, 1.0f32]), 0.0f32, "direct first argument")
  test_assert_close_tensor(g.1, to_tensor([1.0f32, 1.0f32]), 0.0f32, "direct second argument")
}
"#,
    );
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["test", path.to_str().unwrap(), "--json", "--jobs", "1"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let records: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.last().unwrap()["summary"]["passed"], 2);
    assert_eq!(records.last().unwrap()["summary"]["failed"], 0);
}

#[test]
fn statically_wrong_capture_shape_is_rejected_by_check() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wrong_capture.ch");
    write_file(
        &path,
        r#"
def loss(a: tensor[2, f32], s: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(add(a, s), 0i32))
out = {
  s = to_tensor([3.0f32, 5.0f32, 7.0f32])
  grad(fn (x: tensor[2, f32]) -> loss(x, s))(to_tensor([1.0f32, 2.0f32]))
}
"#,
    );
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .unwrap();
    let checked: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(checked["score"].as_f64().unwrap() < 1.0, "{checked}");
    assert!(
        !checked["errors"].as_array().unwrap().is_empty(),
        "{checked}"
    );
}
