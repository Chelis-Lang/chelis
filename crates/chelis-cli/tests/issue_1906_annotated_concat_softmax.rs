//! Executable annotation boundary; the evaluator repair does not claim C support.
use assert_cmd::Command;
use std::fs;

const SOURCE: &str = include_str!("../../../examples/annotated_concat_softmax.ch");

#[test]
fn annotated_example_checks_and_evaluates_all_values() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("probabilities.ch");
    fs::write(&file, SOURCE).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(temp.path())
        .arg("check")
        .arg(&file)
        .assert()
        .success();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(temp.path())
        .args(["eval", "--json", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let tensor = &result["roots"][0]["value"]["value"];
    assert_eq!(tensor["shape"], serde_json::json!([2, 4]));
    assert_eq!(tensor["data"]["dtype"], "f32");
    assert_eq!(
        tensor["data"]["bits"],
        serde_json::json!(vec!["3e800000"; 8])
    );
}

#[test]
fn invalid_axis_rejects_before_evaluation() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("bad_axis.ch");
    fs::write(&file, SOURCE.replace(", -1)", ", 2)")).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(temp.path())
        .args(["eval", "--json", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("axis"));
}
