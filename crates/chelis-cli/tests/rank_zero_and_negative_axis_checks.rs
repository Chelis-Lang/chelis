use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. The `check_rejects_negative_shape_axis`
// test expects a non-empty errors array, so this helper does not
// assert on the process exit status.
fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

/// chelis#732 P1 migration ([05-OBS-4], the chelis#775 decision): a
/// rank-0 tensor renders as its bare element at every exit, so
/// `scalar_to_tensor` output is observationally the scalar. The
/// rank-0-vs-scalar distinction is a TYPE-level fact (visible to
/// `chelis check` and `shape`), not an observation-channel one - the
/// pre-migration assertion that eval "distinguishes" rank-0 tensors by
/// printing the `tensor(shape=[])` wrapper locked exactly the leak the
/// atom removes.
#[test]
fn eval_renders_rank_zero_tensors_bare() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "scalar_to_tensor(cast(3, i64))"])
        .assert()
        .success()
        .stdout(predicate::str::contains("3"))
        .stdout(predicate::str::contains("tensor(shape=[]").not());
}

#[test]
fn check_rejects_negative_shape_axis_when_rank_is_known() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("negative_shape_axis.ch");
    write_file(&path, "bad = shape(scalar_to_tensor(cast(3, i64)), -1)\n");

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
