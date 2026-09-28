//! User-facing selector diagnostics for chelis#1955 / chelis#1473.

use assert_cmd::Command;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

fn run_file(command: &str, extension: &str, source: &str) -> std::process::Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("selector.{extension}"));
    fs::write(&path, source).expect("write fixture");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([command, path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis")
}

fn run(command: &str, source: &str) -> std::process::Output {
    run_file(command, "ch", source)
}

fn illustrative_example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/illustrative/grad_selector_provenance.ch")
        .canonicalize()
        .expect("illustrative selector example")
}

#[test]
fn illustrative_grad_selector_provenance_checks_and_evaluates() {
    let path = illustrative_example();
    let checked = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("check illustrative selector example");
    assert!(checked.status.success(), "{checked:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&checked.stdout).expect("check JSON report");
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");

    let evaluated = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["eval", "--file", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("evaluate illustrative selector example");
    assert!(evaluated.status.success(), "{evaluated:?}");
    assert_eq!(
        evaluated.stdout,
        b"constructor_direct = 6.0\nconstructor_grad = 2.0\nrecord_direct = 6.0\nrecord_grad = 2.0\n",
    );
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

#[test]
fn surf_rejects_a_malformed_operative_selector_instead_of_dropping_it() {
    let output = run_file(
        "surf",
        "dp",
        "(def {} selected \
           (grad {wrt: (var {} w)} \
             (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))) \
             (var {} selector)))",
    );
    assert!(!output.status.success(), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(
        stderr.contains("grad") && stderr.contains("integer"),
        "{stderr}"
    );
    assert!(!stderr.contains("grad("), "{stderr}");
}
