//! Caller actuals precede formal bindings (spec/03 §4.4); no new alias admission.
use assert_cmd::Command;
use serde_json::json;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

const DIRECT: &str = include_str!("../../../examples/caller_actual_scope.ch");
const CONCAT: &str = "module Formal_Alias\n\
def second[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([y, y], 1i32)\n\
def run[s](x: tensor[s, 2, f32], y: tensor[s, *, f32]) -> tensor[s, *, f32] = softmax(second(y, x), -1)\n\
output = run(to_tensor([[0.0, 1.0], [2.0, 0.0]], f32), to_tensor([[2.0], [-1.0]], f32))\n";

fn assert_eval(source: &str, shape: &[usize], bits: &[&str]) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("actuals.ch");
    fs::write(&path, chelis_surf::format::format_source(source).unwrap()).unwrap();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1);
    assert_eq!(report["errors"], json!([]));
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .args(["eval", "--json", "--timeout", "10", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["roots"].as_array().unwrap().len(), 1);
    assert_eq!(
        report["roots"][0]["value"]["value"],
        json!({"shape": shape, "data": {"dtype": "f32", "bits": bits}})
    );
}

#[test]
fn pure_direct_actual_and_formal_alpha_control_have_exact_native_values() {
    assert!(common::gcc_available(), "native execution is required");
    for source in [
        DIRECT.to_owned(),
        DIRECT.replace(
            "first(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = sub(x, y)",
            "first(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = sub(a, b)",
        ),
    ] {
        assert_eval(&source, &[], &["41c80000"]);
        assert_eq!(common::build_and_run(&source, "actuals"), "out = 25.0\n");
    }
}

#[test]
fn original_concat_softmax_formal_alpha_controls_have_all_values() {
    assert!(common::gcc_available(), "native execution is required");
    for source in [
        CONCAT.to_owned(),
        CONCAT.replace(
            "second[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([y, y]",
            "second[s](a: tensor[s, *, f32], b: tensor[s, *, f32]) = concat([b, b]",
        ),
    ] {
        assert_eval(
            &source,
            &[2, 4],
            &[
                "3e09b2b1", "3ebb26a8", "3e09b2b1", "3ebb26a8", "3ee17bea", "3d7420a9", "3ee17bea",
                "3d7420a9",
            ],
        );
        assert_eq!(
            common::build_and_run(&source, "concat_actuals"),
            "output = tensor(shape=[2, 4], data=[0.13447072, 0.3655293, 0.13447072, 0.3655293, 0.4403985, 0.05960146, 0.4403985, 0.05960146])\n"
        );
    }
}

#[test]
fn invalid_axis_still_rejects_before_evaluation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("invalid.ch");
    fs::write(
        &path,
        chelis_surf::format::format_source(&CONCAT.replace("second(y, x), -1", "second(y, x), 2"))
            .unwrap(),
    )
    .unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--json", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("axis"));
}

#[test]
fn named_alias_keeps_its_existing_native_admission_boundary() {
    let source = "def aligned_left[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = mul(x, gain)\n\
alias = { f = aligned_left\n sum(f(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32])), fixed) }\n";
    assert_eval(source, &[], &["40a00000"]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("alias.ch");
    fs::write(&path, chelis_surf::format::format_source(source).unwrap()).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["build"])
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("unresolved host inference variable"));
}

#[test]
fn existing_mixed_callable_parameters_keep_their_native_values() {
    let source = "def increment(v: f32) -> f32 = add(v, 1.0f32)\n\
def twice(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\n\
def invoke(f: (tensor[2, f32]) -> tensor[2, f32], g: (f32) -> f32, x: tensor[2, f32]) -> tensor[2, f32] = f(add(x, to_tensor([g(1.0f32), g(2.0f32)])))\n\
out = invoke(twice, increment, to_tensor([1.0f32, 2.0f32]))\n";
    assert_eval(source, &[2], &["40c00000", "41200000"]);
    assert!(common::gcc_available(), "native execution is required");
    assert_eq!(
        common::build_and_run(source, "mixed"),
        "out = tensor(shape=[2], data=[6.0, 10.0])\n"
    );
}
