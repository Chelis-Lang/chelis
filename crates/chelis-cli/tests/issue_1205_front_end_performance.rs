//! Executable parity guard for chelis#1205's nested-expression corpus.
//!
//! The structural work counters live beside the effect and host lowerers.
//! This public-surface companion proves that the optimization preserves the
//! evaluator result, generated-C result, and deterministic C artifact for the
//! representative nested and flat programs.
//!
//! These compile-and-run controls are intentionally ignored in the default
//! workspace suite and are run serially by the authoritative command:
//! `.venv/bin/python scripts/compiler_front_end_performance.py`. Success is
//! all five tests passing followed by the runner's documented PASS marker.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

fn issue_1205_source(operations: usize, flat: bool) -> String {
    let module = if flat { "Flat" } else { "Nested" };
    let mut lines = vec![
        format!("module FrontEndPerformance.{module}N{operations}"),
        "def bc(c: f32) -> tensor[8, f32] = \
         reshape(expand(to_tensor([c]), 0, 8i64), [8i64])"
            .to_string(),
    ];
    if flat {
        lines.push(
            "def st(s: tensor[8, f32], i: int64) -> tensor[8, f32] = \
             if gte(i, 5i64) then s else {"
                .to_string(),
        );
        let mut previous = "s".to_string();
        for index in 0..operations {
            lines.push(format!(
                "  t{index} = mul(add({previous}, bc(cast(1.0, f32))), \
                 bc(cast(0.5, f32)))"
            ));
            previous = format!("t{index}");
        }
        lines.extend([format!("  st({previous}, add(i, 1i64))"), "}".to_string()]);
    } else {
        let mut body = "s".to_string();
        for _ in 0..operations {
            body = format!("mul(add({body}, bc(cast(1.0, f32))), bc(cast(0.5, f32)))");
        }
        lines.push(format!(
            "def st(s: tensor[8, f32], i: int64) -> tensor[8, f32] = \
             if gte(i, 5i64) then s else st({body}, add(i, 1i64))"
        ));
    }
    lines.push("r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)".to_string());
    lines.join("\n") + "\n"
}

fn eval(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    common::write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 eval output")
}

fn canonical_source(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    common::write_file(&path, source);
    let formatted = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["fmt", path.to_str().unwrap()])
        .output()
        .expect("chelis fmt runs");
    assert!(
        formatted.status.success(),
        "formatting failed: {}",
        String::from_utf8_lossy(&formatted.stderr)
    );
    let canonical = String::from_utf8(formatted.stdout).expect("utf-8 formatted source");
    common::write_file(&path, &canonical);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["fmt", "--check", path.to_str().unwrap()])
        .assert()
        .success();
    canonical
}

fn check_report(source: &str, name: &str, allow_style_violations: bool) -> (bool, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    common::write_file(&path, source);
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.args(["check", path.to_str().unwrap()]);
    if allow_style_violations {
        command.arg("--allow-style-violations");
    }
    let output = command.output().expect("chelis check runs");
    let stdout = String::from_utf8(output.stdout).expect("utf-8 check report");
    let report = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("check stdout must be JSON: {error}\n{stdout}"));
    (output.status.success(), report)
}

fn build_c_source(source: &str, name: &str) -> (TempDir, Vec<u8>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join("out");
    common::write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let emitted = fs::read(out_dir.join(format!("{name}.c"))).expect("generated C source");
    (dir, emitted)
}

#[test]
#[ignore = "authoritative manual/CI runner: scripts/compiler_front_end_performance.py"]
fn issue_1205_corpus_is_canonical_and_checks_clean() {
    let nested = canonical_source(&issue_1205_source(20, false), "nested_canonical");
    let flat = canonical_source(&issue_1205_source(20, true), "flat_canonical");

    for (name, source) in [("nested_check", &nested), ("flat_check", &flat)] {
        let (success, report) = check_report(source, name, false);
        assert!(success, "{name} must check successfully: {report}");
        assert_eq!(report["score"], 1, "{name} must earn a perfect score");
        assert!(
            report["errors"]
                .as_array()
                .is_some_and(|errors| errors.is_empty()),
            "{name} must have an empty error list: {report}"
        );
    }
}

#[test]
#[ignore = "authoritative manual/CI runner: scripts/compiler_front_end_performance.py"]
fn issue_1205_nested_and_flat_eval_agree() {
    let nested = canonical_source(&issue_1205_source(20, false), "nested_eval_source");
    let flat = canonical_source(&issue_1205_source(20, true), "flat_eval_source");

    let nested_eval = eval(&nested, "nested_eval");
    let flat_eval = eval(&flat, "flat_eval");
    assert_eq!(nested_eval, flat_eval, "nested and flat eval must agree");
}

#[test]
#[ignore = "authoritative manual/CI runner: scripts/compiler_front_end_performance.py"]
fn issue_1205_nested_and_flat_compiled_c_match_eval() {
    if !common::gcc_available() {
        eprintln!("skipping generated-C execution: host C compiler unavailable");
        return;
    }
    let nested = canonical_source(&issue_1205_source(20, false), "nested_c_source");
    let flat = canonical_source(&issue_1205_source(20, true), "flat_c_source");
    let nested_eval = eval(&nested, "nested_c_eval");
    let flat_eval = eval(&flat, "flat_c_eval");
    let nested_c = common::build_and_run(&nested, "nested_issue_1205");
    let flat_c = common::build_and_run(&flat, "flat_issue_1205");
    assert_eq!(nested_c, nested_eval, "nested C must agree with eval");
    assert_eq!(flat_c, flat_eval, "flat C must agree with eval");
}

#[test]
#[ignore = "authoritative manual/CI runner: scripts/compiler_front_end_performance.py"]
fn issue_1205_repeated_builds_emit_deterministic_nonempty_c() {
    let nested = canonical_source(&issue_1205_source(20, false), "nested_deterministic_source");
    let (_first_dir, first) = build_c_source(&nested, "deterministic_issue_1205");
    let (_second_dir, second) = build_c_source(&nested, "deterministic_issue_1205");
    assert!(!first.is_empty(), "build must emit nonempty C");
    assert_eq!(first, second, "repeated builds must emit byte-identical C");
}

#[test]
#[ignore = "authoritative manual/CI runner: scripts/compiler_front_end_performance.py"]
fn issue_1205_malformed_control_is_rejected() {
    let (success, report) = check_report(
        "module FrontEndPerformance.Malformed\ndef broken(\n",
        "malformed_issue_1205",
        true,
    );
    assert!(!success, "malformed source must fail: {report}");
    assert!(
        report["errors"]
            .as_array()
            .is_some_and(|errors| !errors.is_empty()),
        "malformed source must report at least one error: {report}"
    );
}
