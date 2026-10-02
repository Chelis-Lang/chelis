//! Macro expansion preserves caller ascriptions and hygienic typed binders
//! across the public checker and evaluator commands.

use assert_cmd::Command;
use serde_json::Value;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn check(source: &str) -> (bool, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro.ch");
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("check runs");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "check emits JSON: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), report)
}

fn eval(source: &str) -> (bool, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro.ch");
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("eval runs");
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

fn assert_rejected(source: &str) {
    let (success, report) = check(source);
    assert!(!success, "ill-typed binding passed check: {report:#}");
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "ill-typed binding received perfect fitness: {report:#}"
    );
    assert!(
        report["errors"]
            .as_array()
            .is_some_and(|errors| errors.iter().any(|error| {
                error["kind"] == "PrecisionMismatch"
                    && error["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("let-binding `t` ascription"))
            })),
        "expected the authored binding's precision error: {report:#}"
    );
    let (success, output) = eval(source);
    assert!(!success, "ill-typed binding evaluated: {output}");
    assert!(output.contains("PrecisionMismatch"), "{output}");
}

fn assert_accepted(source: &str, expected_out: &str) {
    let (success, report) = check(source);
    assert!(success, "well-typed binding failed check: {report:#}");
    assert_eq!(report["score"], 1.0, "{report:#}");
    assert_eq!(report["errors"].as_array().map(Vec::len), Some(0));
    let (success, output) = eval(source);
    assert!(success, "well-typed binding failed eval: {output}");
    assert!(output.contains(expected_out), "{output}");
}

#[test]
fn macro_and_direct_binding_ascriptions_have_the_same_rejection() {
    for rhs in ["twice(x)", "add(x, x)"] {
        assert_rejected(&format!(
            "macro twice(v) = add(v, v)\n\
             def run(x: f32) -> f32 = {{\n\
               t: i32 = {rhs}\n\
               cast(t, f32)\n\
             }}\n\
             out = run(1.5f32)\n"
        ));
    }
    assert_rejected(
        "macro one() = 1.0f32\n\
         def run() -> f32 = {\n\
           t: i32 = one()\n\
           cast(t, f32)\n\
         }\n\
         out = run()\n",
    );
}

#[test]
fn agreeing_macro_and_direct_binding_ascriptions_execute() {
    for rhs in ["twice(x)", "add(x, x)"] {
        assert_accepted(
            &format!(
                "macro twice(v) = add(v, v)\n\
                 def run(x: f32) -> f32 = {{\n\
                   t: f32 = {rhs}\n\
                   t\n\
                 }}\n\
                 out = run(1.5f32)\n"
            ),
            "out = 3.0",
        );
    }
}

#[test]
fn template_ascription_survives_argument_substitution() {
    let source = "macro as_i32(v) = v : i32\n\
                  def run(x: f32) -> f32 = cast(as_i32(x), f32)\n\
                  out = run(1.5f32)\n";
    let (success, report) = check(source);
    assert!(!success, "template ascription disappeared: {report:#}");
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "PrecisionMismatch")),
        "template ascription must produce a precision error: {report:#}"
    );
    let (success, output) = eval(source);
    assert!(!success && output.contains("PrecisionMismatch"), "{output}");

    assert_accepted(
        "macro as_i32(v) = v : i32\n\
         def run(x: i32) -> i32 = as_i32(x)\n\
         out = run(3)\n",
        "out = 3",
    );
}

#[test]
fn vocabulary_named_typed_macro_parameter_does_not_capture_caller_argument() {
    assert_accepted(
        "macro bump(v) = (fn (record: f32) -> add(record, v))(1.0f32)\n\
         def run(record: f32) -> f32 = bump(record)\n\
         out = run(10.0f32)\n",
        "out = 11.0",
    );
    assert_accepted(
        "macro bump(v) = (fn (y: f32) -> add(y, v))(1.0f32)\n\
         def run(record: f32) -> f32 = bump(record)\n\
         out = run(10.0f32)\n",
        "out = 11.0",
    );
}

#[test]
fn typed_macro_binding_executes_identically_in_eval_and_c() {
    let source = "macro add_one(v) = (fn (record: f32) -> add(record, v))(1.0f32)\n\
                  def run(record: f32) -> f32 = {\n\
                    result: f32 = add_one(record)\n\
                    result\n\
                  }\n\
                  out = run(10.0f32)\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro.ch");
    write_file(&path, source);
    let checked = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .expect("check runs");
    assert!(checked.status.success(), "{checked:?}");
    let report: Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert_eq!(report["score"], 1.0, "{report:#}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report:#}");

    let evaluated = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .expect("eval runs");
    assert!(evaluated.status.success(), "{evaluated:?}");
    assert_eq!(evaluated.stdout, b"out = 11.0\n");

    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg("--emit-c")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(dir.path())
        .output()
        .expect("C build runs");
    assert!(built.status.success(), "{built:?}");
    assert!(common::link_generated(dir.path(), "macro.c", "macro").success());
    let native = StdCommand::new(dir.path().join("macro"))
        .output()
        .expect("compiled program runs");
    assert!(native.status.success(), "{native:?}");
    assert_eq!(native.stdout, evaluated.stdout);
}
