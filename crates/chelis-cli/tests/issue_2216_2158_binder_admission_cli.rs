//! chelis#2216 and chelis#2158 through the CLI: a builtin's operand-kind or
//! dtype-family rule is decided at check time when the operand's type is a
//! rigid authored binder.
//!
//! Before the fix each issue program below scored 1 at `chelis check`. The
//! #2216 program then trapped in `eval` with an internal payload description
//! and built C that clang rejects; the #2158 programs panicked in `eval` and in
//! C emission at arms whose comments say the checker rejects them. Each is now
//! a check-time diagnostic in every lane, and the `[p: Float]` control from
//! #2158 still checks clean and runs to `[7, -8]` in both execution lanes.
//!
//! `crates/chelis-types/tests/issue_2216_rigid_binder_admission.rs` owns the
//! checker-level matrix; this file covers the issue programs end to end.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn run(command: &str, source: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("binder.ch");
    fs::write(&path, source).expect("source");
    let mut cli = Command::cargo_bin("chelis").expect("chelis");
    cli.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(command);
    if command == "eval" {
        cli.arg("--file");
    }
    cli.arg(&path);
    if command == "build" {
        cli.args(["--target", "c", "--output"])
            .arg(dir.path().join("out"));
    }
    cli.output().expect(command)
}

/// `check`'s JSON report for `source`.
fn check_report(source: &str) -> serde_json::Value {
    let output = run("check", source);
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout[stdout.find('{').expect("check JSON")..]).expect("check JSON")
}

const NUMEL_ON_A_FLOAT_BINDER: &str = "module Bounded.Main\nexport (main)\n\
     def bad[p: Float](k: p) -> p = cast(numel(k), p)\n\
     def main() -> tensor[1, f64] = to_tensor([bad(1.0f64)])\n";

fn trunc_program(bound: &str, input: &str) -> String {
    format!(
        "module P.Main\nexport (main)\n\
         def tn[p: {bound}](x: tensor[2, p]) -> tensor[2, i64] = cast_trunc(x, i64)\n\
         def main() -> tensor[2, i64] = tn(to_tensor([{input}]))\n"
    )
}

/// `check` reports a diagnostic containing `fragment` with a score below 1,
/// and `eval` and `build` refuse the program with that diagnostic instead of
/// trapping, panicking, or emitting C.
fn rejected_at_check_in_every_lane(source: &str, fragment: &str) {
    let report = check_report(source);
    let messages: Vec<&str> = report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .filter_map(|error| error["message"].as_str())
        .collect();
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "check must not score the program 1: {report}"
    );
    assert!(
        messages.iter().any(|message| message.contains(fragment)),
        "check must report {fragment:?}, got {messages:?}"
    );
    for command in ["eval", "build"] {
        let output = run(command, source);
        let diagnostic = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.status.success(), "{command} accepted:\n{source}");
        assert!(
            diagnostic.contains(fragment) && !diagnostic.contains("panicked"),
            "{command} must stop at the check-time diagnostic, got:\n{diagnostic}"
        );
    }
}

/// REGRESSION TEST for chelis#2216.
#[test]
fn numel_on_a_float_binder_scalar_is_rejected_at_check() {
    rejected_at_check_in_every_lane(
        NUMEL_ON_A_FLOAT_BINDER,
        "numel expects tensor input, got f32",
    );
}

/// REGRESSION TEST for chelis#2158, both bounds the issue measured.
#[test]
fn cast_trunc_from_an_int_or_numeric_binder_is_rejected_at_check() {
    for (bound, input) in [("Numeric", "7i32, 8i32"), ("Int", "7i32, 8i32")] {
        rejected_at_check_in_every_lane(
            &trunc_program(bound, input),
            "its source dtype is the declared type parameter `p`",
        );
    }
}

/// NEGATIVE PARITY: #2158's `[p: Float]` control checks at score 1 and both
/// execution lanes truncate toward zero.
#[test]
fn cast_trunc_from_a_float_binder_checks_clean_and_runs() {
    let source = trunc_program("Float", "7.5f32, -8.5f32");
    let report = check_report(&source);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(
        report["errors"].as_array().is_some_and(Vec::is_empty),
        "{report}"
    );
    let interpreted = run("eval", &source);
    assert!(
        interpreted.status.success(),
        "eval: {}",
        String::from_utf8_lossy(&interpreted.stderr)
    );
    let interpreted = String::from_utf8(interpreted.stdout).expect("UTF-8");
    assert_eq!(common::parse_tensor_data(&interpreted, "main"), [7.0, -8.0]);
    let native = common::build_and_run(&source, "issue_2158_float_control");
    assert_eq!(native.trim(), interpreted.trim(), "eval vs C");
}
