use std::io::Write;
use std::process::{Command, Output};

fn validate_surf(source: &str) -> Output {
    let mut file = tempfile::Builder::new()
        .suffix(".ch")
        .tempfile()
        .expect("create source file");
    file.write_all(source.as_bytes()).expect("write source");
    Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(["validate", "--surf", "--allow-style-violations"])
        .arg(file.path())
        .output()
        .expect("run validator")
}

#[test]
fn deeply_nested_valid_comment_does_not_abort_validator() {
    let source = format!("{}{}\nx = 1i64\n", "{-".repeat(10_000), "-}".repeat(10_000));
    let output = validate_surf(&source);
    assert!(
        output.status.success(),
        "valid comments must be accepted; status={:?}, stdout={}, stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn deeply_nested_unterminated_comment_reports_parse_error() {
    let source = format!("{}\nx = 1i64\n", "{-".repeat(10_000));
    let output = validate_surf(&source);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "status={:?}; {stderr}",
        output.status
    );
    assert!(stderr.contains("unterminated block comment"), "{stderr}");
}

#[test]
fn deeply_nested_comment_before_invalid_surf_reports_a_parse_error() {
    let source = format!(
        "module Probe.Deep\n{}{}\nx = 1i64 y\n",
        "{-".repeat(5_000),
        "-}".repeat(5_000)
    );
    let output = validate_surf(&source);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(stderr.contains("compiler parse failed"), "{stderr}");
}

#[test]
fn extreme_parentheses_keep_cli_accept_and_reject_verdicts() {
    let valid = format!("x = {}1i64{}\n", "(".repeat(40_000), ")".repeat(40_000));
    let accepted = validate_surf(&valid);
    assert!(accepted.status.success(), "{accepted:?}");

    let invalid = format!("{} y\n", valid.trim_end());
    let rejected = validate_surf(&invalid);
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert_eq!(rejected.status.code(), Some(1), "{rejected:?}");
    assert!(stderr.contains("compiler parse failed"), "{stderr}");
}

#[test]
fn extreme_type_and_pattern_parentheses_reach_cli_verdicts() {
    let open = "(".repeat(40_000);
    let close = ")".repeat(40_000);
    for valid in [
        format!("def identity(x: {open}i64{close}) -> i64 = x\n"),
        format!("x = match 1i64 with {{ | {open}_{close} => 1i64 }}\n"),
    ] {
        let accepted = validate_surf(&valid);
        assert!(accepted.status.success(), "{accepted:?}");

        let invalid = format!("{} y\n", valid.trim_end());
        let rejected = validate_surf(&invalid);
        let stderr = String::from_utf8_lossy(&rejected.stderr);
        assert_eq!(rejected.status.code(), Some(1), "{rejected:?}");
        assert!(stderr.contains("compiler parse failed"), "{stderr}");
    }
}
