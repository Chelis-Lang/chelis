//! [04-PAT-1] reaches `chelis check` and rejects an invalid flexible pattern
//! before `chelis eval` can return the dead arm's answer.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn run(command: &str, source: &str) -> std::process::Output {
    let dir = tempdir().expect("temporary fixture directory");
    let path = dir.path().join("pattern.ch");
    fs::write(&path, source).expect("write Surf fixture");
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    match command {
        "check" => {
            cmd.args(["check", path.to_str().expect("UTF-8 path")]);
        }
        "eval" => {
            cmd.args(["eval", "--file", path.to_str().expect("UTF-8 path")]);
        }
        "build" => {
            let output = dir.path().join("built");
            cmd.args([
                "build",
                "--emit-c",
                path.to_str().expect("UTF-8 path"),
                "--target",
                "c",
                "-o",
                output.to_str().expect("UTF-8 path"),
            ]);
        }
        _ => panic!("unsupported test command"),
    }
    cmd.output().expect("run chelis")
}

#[test]
fn flexible_dead_arm_is_rejected_before_evaluation() {
    let source = "def h(x: f32) -> bool = {\n  k = fn (y) -> match y with {\n    | 0 => true\n    | _ => false\n  }\n  k(x)\n}\nout = h(0.0f32)\n";
    let check = run("check", source);
    let json: serde_json::Value = serde_json::from_slice(&check.stdout).expect("check JSON");
    assert!(json["score"].as_f64().expect("score") < 1.0, "{json}");
    let errors = json["errors"].as_array().expect("errors array");
    let [error] = errors.as_slice() else {
        panic!("expected one pattern error: {json}");
    };
    assert_eq!(error["kind"].as_str(), Some("TypeMismatch"));
    assert!(
        error["message"]
            .as_str()
            .expect("message")
            .contains("[04-PAT-1]")
    );

    let eval = run("eval", source);
    assert!(!eval.status.success());
    assert!(!String::from_utf8_lossy(&eval.stdout).contains("out ="));
    assert!(String::from_utf8_lossy(&eval.stderr).contains("[04-PAT-1]"));

    let build = run("build", source);
    assert!(!build.status.success());
    assert!(String::from_utf8_lossy(&build.stderr).contains("[04-PAT-1]"));
}

#[test]
fn valid_flexible_literal_pattern_still_runs() {
    let source = "def h(x: i64) -> bool = {\n  k = fn (y) -> match y with {\n    | 0 => true\n    | _ => false\n  }\n  k(x)\n}\nout = h(0i64)\n";
    let check = run("check", source);
    let json: serde_json::Value = serde_json::from_slice(&check.stdout).expect("check JSON");
    assert_eq!(json["score"].as_f64(), Some(1.0), "{json}");
    assert_eq!(json["errors"].as_array().map(Vec::len), Some(0));

    let eval = run("eval", source);
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert!(String::from_utf8_lossy(&eval.stdout).contains("out = true"));
}

#[test]
fn float_pattern_diagnostic_keeps_exponent_notation() {
    let source = "def f(x: i32) -> i32 =\n  match x with {\n    | 3.4e38 => 1\n    | _ => 0\n  }\n";
    let check = run("check", source);
    let json: serde_json::Value = serde_json::from_slice(&check.stdout).expect("check JSON");
    let errors = json["errors"].as_array().expect("errors array");
    let [error] = errors.as_slice() else {
        panic!("expected one pattern error: {json}");
    };
    let message = error["message"].as_str().expect("message");
    assert!(
        message.contains("floating-point literal pattern `3.4e38`"),
        "{message}"
    );
}
