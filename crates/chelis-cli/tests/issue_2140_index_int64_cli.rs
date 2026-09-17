//! chelis#2140 through the serialized `chelis check` surface.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
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

fn non_i64_literal_index_arguments(source: &str) -> Vec<(usize, String)> {
    let bytes = source.as_bytes();
    let mut matches = Vec::new();
    let mut cursor = 0;
    while cursor + "index".len() <= bytes.len() {
        let Some(offset) = source[cursor..].find("index") else {
            break;
        };
        let start = cursor + offset;
        cursor = start + "index".len();
        if start > 0
            && matches!(
                bytes[start - 1],
                b'.' | b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'
            )
        {
            continue;
        }
        let mut open = cursor;
        while open < bytes.len() && bytes[open].is_ascii_whitespace() {
            open += 1;
        }
        if bytes.get(open) != Some(&b'(') {
            continue;
        }

        let mut nested = Vec::new();
        let mut quote = None;
        let mut escaped = false;
        let mut second_start = None;
        let mut end = open + 1;
        while end < bytes.len() {
            let byte = bytes[end];
            if let Some(delimiter) = quote {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == delimiter {
                    quote = None;
                }
            } else {
                match byte {
                    b'\'' | b'"' => quote = Some(byte),
                    b'(' | b'[' | b'{' => nested.push(byte),
                    b')' if nested.is_empty() => break,
                    b')' | b']' | b'}' => {
                        nested.pop();
                    }
                    b',' if nested.is_empty() && second_start.is_none() => {
                        second_start = Some(end + 1);
                    }
                    _ => {}
                }
            }
            end += 1;
        }
        let Some(second_start) = second_start else {
            continue;
        };
        let argument = source[second_start..end].trim();
        let exact_i64 = argument
            .strip_suffix("i64")
            .is_some_and(|number| number.parse::<i128>().is_ok());
        let recognized_non_i64 = argument.parse::<i128>().is_ok()
            || argument.parse::<f64>().is_ok()
            || [
                "i8", "i16", "i32", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64",
            ]
            .iter()
            .any(|suffix| {
                argument
                    .strip_suffix(suffix)
                    .is_some_and(|number| number.parse::<f64>().is_ok())
            });
        if recognized_non_i64 && !exact_i64 {
            matches.push((
                source[..start]
                    .bytes()
                    .filter(|byte| *byte == b'\n')
                    .count()
                    + 1,
                argument.to_string(),
            ));
        }
    }
    matches
}

#[test]
fn direct_index_literal_fixture_oracle_fails_closed() {
    let direct = "index";
    let source = format!(
        "\
bare = {direct}([1i64], 0)
wrong_width = {direct}([1i64], 1i32)
wrong_family = {direct}([1i64], 2.0f32)
exact = {direct}([1i64], 3i64)
nested = {direct}({direct}([[1i64]], 0i64), 4)
method = values.{direct}(5)
"
    );
    assert_eq!(
        non_i64_literal_index_arguments(&source),
        vec![
            (1, "0".to_string()),
            (2, "1i32".to_string()),
            (3, "2.0f32".to_string()),
            (5, "4".to_string()),
        ]
    );
}

#[test]
fn tracked_direct_index_numeric_literal_fixtures_use_exact_i64() {
    // This is the executable-corpus migration oracle. Intentional non-i64
    // rejection controls remain generated through `{literal}` in the focused
    // matrix below; they are executed, not silently exempted by path.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tracked = std::process::Command::new("git")
        .current_dir(&root)
        .args([
            "ls-files",
            "-z",
            "--",
            "crates",
            "examples",
            "packages/chelis-std",
            "scripts",
            "tests",
        ])
        .output()
        .expect("list tracked source carriers");
    assert!(
        tracked.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&tracked.stderr)
    );

    let mut violations = Vec::new();
    for relative in tracked.stdout.split(|byte| *byte == b'\0') {
        if relative.is_empty() {
            continue;
        }
        let relative = String::from_utf8(relative.to_vec()).expect("UTF-8 tracked path");
        if !matches!(
            Path::new(&relative)
                .extension()
                .and_then(|value| value.to_str()),
            Some("ch" | "py" | "rs")
        ) {
            continue;
        }
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        for (line, argument) in non_i64_literal_index_arguments(&source) {
            violations.push(format!("{relative}:{line}: index argument `{argument}`"));
        }
    }
    violations.sort();
    assert!(
        violations.is_empty(),
        "direct index literal audit requires exact i64:\n{}",
        violations.join("\n")
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
fn direct_index_rejects_active_non_integer_primitives_in_cli() {
    for index_type in ["f16", "bf16", "f32", "f64", "bool", "string"] {
        let source = format!("def pick(xs: List[i64], i: {index_type}) -> i64 = index(xs, i)\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}
