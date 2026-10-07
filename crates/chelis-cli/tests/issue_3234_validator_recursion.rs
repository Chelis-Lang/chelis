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
