use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

#[test]
fn eval_distinguishes_rank_zero_tensors_from_host_scalars() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "scalar_to_tensor(cast(3, int64))"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tensor(shape=[]"));
}

#[test]
fn check_rejects_negative_shape_axis_when_rank_is_known() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("negative_shape_axis.ch");
    write_file(&path, "bad = shape(scalar_to_tensor(cast(3, int64)), -1)\n");

    let json = run_json_check(&path);
    assert!(
        !json["errors"].as_array().unwrap().is_empty(),
        "expected check errors for negative shape axis, got {json}"
    );
    assert!(
        json["score"].as_f64().unwrap() < 1.0,
        "expected score below 1.0 for negative shape axis, got {json}"
    );
}
