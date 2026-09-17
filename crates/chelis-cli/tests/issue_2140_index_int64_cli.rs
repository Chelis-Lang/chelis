//! chelis#2140 through the serialized `chelis check` surface.

use assert_cmd::Command;
use std::fs;
use std::process::{ExitStatus, Output};
use tempfile::tempdir;

struct CheckResult {
    status: ExitStatus,
    report: serde_json::Value,
}

fn check(source: &str) -> CheckResult {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("index_int64.ch");
    fs::write(&path, source).expect("write fixture");
    let Output {
        status,
        stdout,
        stderr,
    } = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let report = serde_json::from_slice(&stdout).unwrap_or_else(|error| {
        panic!(
            "check JSON: {error}; status={status:?}; stdout={}; stderr={}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        )
    });
    CheckResult { status, report }
}

fn assert_accepted(label: &str, source: &str) {
    let CheckResult { status, report } = check(source);
    assert!(status.success(), "{label}: {status:?}: {report}");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{label}: {report}");
    assert!(
        report["errors"].as_array().unwrap().is_empty(),
        "{label}: {report}"
    );
}

fn assert_index_type_rejected(label: &str, source: &str, index_type: &str) {
    let CheckResult { status, report } = check(source);
    assert!(!status.success(), "{label}: {status:?}: {report}");
    assert!(report["score"].as_f64().unwrap() < 1.0, "{label}: {report}");
    let expected = format!("index expects i64 index, got {index_type}");
    assert!(
        report["errors"].as_array().unwrap().iter().any(|error| {
            error["kind"] == "TypeMismatch"
                && error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(&expected))
        }),
        "{label}: expected operation-owned diagnostic {expected:?}: {report}"
    );
}

#[test]
fn direct_index_accepts_exact_i64_in_cli() {
    assert_accepted("exact i64 index", "out: i64 = index([1i64], 0i64)\n");
}

#[test]
fn direct_index_rejects_every_other_integer_width_in_cli() {
    // Mutation-equivalent negative control for the former broad predicate.
    for (index_type, literal) in [("i8", "0i8"), ("i16", "0i16"), ("i32", "0i32")] {
        let source = format!("out: i64 = index([1i64], {literal})\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}

#[test]
fn direct_index_rejects_non_integer_primitives_in_cli() {
    for index_type in ["f16", "bf16", "f32", "f64", "bool", "string"] {
        let source = format!("def pick(xs: List[i64], i: {index_type}) -> i64 = index(xs, i)\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}
