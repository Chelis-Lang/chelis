//! Spec/05 section 3.4: the CLI must expose the ordinary binding rule.
use assert_cmd::Command;
use serde_json::Value;

#[test]
fn undeclared_normalize_fails_with_an_unbound_variable_diagnostic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("undeclared.ch");
    std::fs::write(
        &path,
        "def f(x: tensor[4, f32]) -> tensor[4, f32] = normalize(x)\n",
    )
    .unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "UnboundVariable"
                && e["message"].as_str().unwrap().contains("normalize")),
        "{report}"
    );
}

#[test]
fn authored_normalization_example_checks_and_evaluates() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/explicit_normalization.ch");
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1.0, "{report}");
    assert!(report["errors"].as_array().unwrap().is_empty());
    let evaluated = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert_eq!(
        String::from_utf8(evaluated.stdout).unwrap().trim(),
        "result = 2.0"
    );
}
