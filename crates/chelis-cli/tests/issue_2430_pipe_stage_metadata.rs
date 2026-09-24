//! A pipe stage keeps selector metadata on the selected transform.
//! `x |> grad(sq, wrt=v)` is the application `(grad(sq, wrt=v))(x)`.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

const DEFINITION: &str = "def sq(v: f32) -> f32 = mul(v, v)\n";

fn source(body: &str) -> String {
    format!("{DEFINITION}def run(x: f32) -> f32 = {body}\nout = run(3.0f32)\n")
}

#[test]
fn a_pipe_with_a_grad_selector_checks_evaluates_and_builds_like_direct_application() {
    let dir = tempdir().expect("temporary source directory");
    let mut direct_eval = None;
    for (name, body) in [
        ("direct", "(grad(sq, wrt=v))(x)"),
        ("pipe", "x |> grad(sq, wrt=v)"),
    ] {
        let path = dir.path().join(format!("{name}.ch"));
        fs::write(&path, source(body)).expect("write source");

        let check = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .expect("check program");
        assert!(
            check.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&check.stderr)
        );
        let report: Value = serde_json::from_slice(&check.stdout).expect("check JSON");
        assert_eq!(report["score"], 1.0, "{name}: {report}");
        assert_eq!(
            report["errors"].as_array().map(Vec::len),
            Some(0),
            "{name}: {report}"
        );

        let eval = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("evaluate program");
        assert!(
            eval.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&eval.stderr)
        );
        let stdout = String::from_utf8(eval.stdout).expect("eval output");
        assert!(stdout.contains("out = 6.0"), "{name}: {stdout}");
        if let Some(ref expected) = direct_eval {
            assert_eq!(&stdout, expected, "pipe and direct application disagree");
        } else {
            direct_eval = Some(stdout);
        }

        let output = dir.path().join(format!("{name}-c"));
        let build = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .expect("build program");
        assert!(
            build.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&build.stderr)
        );
        assert!(output.join(format!("{name}.c")).is_file());
    }
}

#[test]
fn an_invalid_grad_selector_in_a_pipe_is_rejected() {
    let dir = tempdir().expect("temporary source directory");
    let path = dir.path().join("bad_selector.ch");
    fs::write(&path, source("x |> grad(sq, wrt=missing)")).expect("write source");
    let check = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("check program");
    assert!(!check.status.success(), "invalid selector was accepted");
    let report: Value = serde_json::from_slice(&check.stdout).expect("check JSON");
    assert_ne!(report["score"], 1.0, "{report}");
    assert!(
        report["errors"]
            .as_array()
            .is_some_and(|errors| !errors.is_empty()),
        "{report}"
    );
}

#[test]
fn a_binding_ascription_on_a_pipe_is_checked_like_direct_application() {
    let dir = tempdir().expect("temporary source directory");
    for (name, value) in [("direct", "twice(x)"), ("pipe", "x |> twice")] {
        let path = dir.path().join(format!("{name}-ascription.ch"));
        fs::write(
            &path,
            format!(
                "def twice(v: f32) -> f32 = add(v, v)\n\
                 def run(x: f32) -> f32 = {{\n\
                 \x20 t: i32 = {value}\n\
                 \x20 cast(t, f32)\n\
                 }}\n\
                 out = run(1.5f32)\n"
            ),
        )
        .expect("write source");
        let check = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .expect("check program");
        assert!(
            !check.status.success(),
            "{name}: accepted a wrong binding type"
        );
        let report: Value = serde_json::from_slice(&check.stdout).expect("check JSON");
        assert_ne!(report["score"], 1.0, "{name}: {report}");
        assert!(
            report["errors"]
                .as_array()
                .is_some_and(|errors| !errors.is_empty()),
            "{name}: {report}"
        );
    }
}
