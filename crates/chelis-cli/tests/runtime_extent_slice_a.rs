//! Public CLI acceptance rows for runtime-extent Slice A (chelis#1277).

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn check(source: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("case.ch");
    fs::write(&path, source).expect("fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("check");
    serde_json::from_slice(&output.stdout).expect("check json")
}

fn errors(report: &Value) -> Vec<&str> {
    report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .filter_map(|error| error["message"].as_str())
        .collect()
}

fn compile_and_run_c(build_dir: &Path, stem: &str) -> std::process::Output {
    let binary = build_dir.join(format!("{stem}-bin"));
    let compile = StdCommand::new("cc")
        .args([
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            build_dir.join(format!("{stem}.c")).to_str().unwrap(),
            build_dir.join("libchelis_runtime.a").to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-o",
            binary.to_str().unwrap(),
        ])
        .output()
        .expect("compile emitted C");
    assert!(
        compile.status.success(),
        "C compilation failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    StdCommand::new(binary).output().expect("run emitted C")
}

#[test]
fn zero_extent_is_check_clean_and_evaluates_to_empty_tensor() {
    let source = "out = expand(scalar_to_tensor(1.0f32), 0, 0i64)\n";
    let report = check(source);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("zero.ch");
    fs::write(&path, source).expect("fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("eval");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("shape=[0]") && stdout.contains("data=[]"),
        "{stdout}"
    );

    let build_dir = dir.path().join("zero-out");
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            build_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let compiled = compile_and_run_c(&build_dir, "zero");
    assert!(
        compiled.status.success(),
        "compiled zero extent failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    assert!(
        compiled_stdout.contains("shape=[0]") && compiled_stdout.contains("data=[]"),
        "{compiled_stdout}"
    );
}

#[test]
fn negative_extent_remains_a_static_type_error() {
    let report = check("out = expand(scalar_to_tensor(1.0f32), 0, -1i64)\n");
    let joined = errors(&report).join("\n");
    assert!(
        joined.contains("expand") && joined.contains("-1"),
        "{report}"
    );
}

#[test]
fn shape_sourced_expand_rejects_wrong_rank_ascription() {
    let source = "def bad(g: &tensor[c, f32], x: &tensor[a, c, h, w, f32]) -> tensor[c, h, w, f32] = {\n\
        \x20 step1: tensor[c, h, w, f32] = expand(g, 1, shape(x, cast(2, int32)))\n\
        \x20 step1\n\
        }\n";
    let report = check(source);
    let joined = errors(&report).join("\n");
    assert!(
        joined.contains("expand") && joined.contains("rank"),
        "wrong-rank ascription must reject at check: {report}"
    );
}

#[test]
fn bare_dimension_binder_executes_and_builds_without_symbolic_dim_ice() {
    let source = "def f(b: tensor[f32], c: tensor[k, f32]) -> tensor[k, f32] = expand(b, 0, k)\n\
        out = f(scalar_to_tensor(2.0f32), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    let report = check(source);
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("binder.ch");
    fs::write(&path, source).expect("fixture");
    let eval = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("eval");
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let stdout = String::from_utf8_lossy(&eval.stdout);
    assert!(
        stdout.contains("shape=[3]") && stdout.contains("[2.0, 2.0, 2.0]"),
        "{stdout}"
    );

    let out_dir = dir.path().join("out");
    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&build.stderr).contains("internal compiler error"),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let compiled = compile_and_run_c(&out_dir, "binder");
    assert!(
        compiled.status.success(),
        "compiled binder extent failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    assert!(
        compiled_stdout.contains("shape=[3]") && compiled_stdout.contains("[2.0, 2.0, 2.0]"),
        "{compiled_stdout}"
    );
}

#[test]
fn stale_extent_guidance_is_removed_but_axis_guidance_stays_int32() {
    let extent = check("def bad(b: tensor[f32], k: int64) -> tensor[k, f32] = expand(b, 0, k)\n");
    let extent_errors = errors(&extent).join("\n");
    assert!(!extent_errors.contains("Form-3"), "{extent_errors}");
    assert!(!extent_errors.contains("cast(N, int32)"), "{extent_errors}");
    assert!(extent_errors.contains("int64"), "{extent_errors}");

    let axis = check(
        "def bad(x: tensor[2, f32], axis: int32) -> tensor[2, 2, f32] = expand(x, axis, 2i64)\n",
    );
    let axis_errors = errors(&axis).join("\n");
    assert!(axis_errors.contains("int32"), "{axis_errors}");
}
