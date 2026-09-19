//! User-facing selector diagnostics for chelis#1955 / chelis#1473.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn run(command: &str, source: &str) -> std::process::Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("selector.ch");
    fs::write(&path, source).expect("write fixture");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([command, path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis")
}

#[test]
fn deep_rejects_unknown_selector_without_emitting_a_guessed_index() {
    let output = run(
        "deep",
        "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=typo)\n",
    );
    assert!(!output.status.success(), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(
        stderr.contains("unknown `grad` parameter `typo`"),
        "{stderr}"
    );
    assert!(!stderr.contains("(lit {type:"), "{stderr}");
}

#[test]
fn check_reports_selector_rejection_in_json() {
    let output = run(
        "check",
        "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=typo)\n",
    );
    assert!(!output.status.success(), "{output:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check JSON report");
    let errors = report["errors"].as_array().expect("errors array");
    assert_eq!(errors.len(), 1, "{report}");
    assert_eq!(errors[0]["kind"], "TypeMismatch", "{report}");
    assert!(
        errors[0]["message"]
            .as_str()
            .is_some_and(|message| message.contains("unknown `grad` parameter `typo`")),
        "{report}"
    );
}

#[test]
fn deep_emits_the_second_parameter_index_for_an_alias() {
    let output = run(
        "deep",
        r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
alias = pair
out = grad(alias, wrt=w)(2.0f32, 3.0f32)
"#,
    );
    assert!(output.status.success(), "{output:?}");
    let deep = String::from_utf8(output.stdout).expect("UTF-8 Deep");
    assert!(
        deep.contains("(grad {span:")
            && deep.contains("wrt: (var {} w)")
            && deep.contains("(var {span:")
            && deep.contains("alias)")
            && deep.contains("(lit {type: (t-prim {} i32)} 1)"),
        "{deep}"
    );
}
