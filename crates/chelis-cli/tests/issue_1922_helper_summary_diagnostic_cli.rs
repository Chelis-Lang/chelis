//! The source-reachable fatal summary diagnostic must be an ordinary CLI
//! failure, with its original location, rather than silent process exit 101.
//!
//! The fixture's fatal was its `insert` sized by `seq` until chelis#469 made
//! that an ordinary runtime extent. The fatal public eval now reaches is the
//! static-DAG `concat` rejection in `join_columns`; which fatal it is does not
//! matter to this lock, only that it reaches the user through the diagnostic
//! channel with its source span.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

const SOURCE: &str = include_str!("../../../tests/support/helper_summary_fatal.ch");

#[test]
fn fatal_summary_diagnostic_is_visible_in_text_and_json_eval() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sink.ch");
    let formatted = chelis_surf::format::format_source(SOURCE).expect("format fixture");
    let operand = "[x, y]";
    let start = formatted
        .find(operand)
        .expect("join_columns' concat operand remains");
    let source_id = format!("surf:{start}..{}", start + operand.len());
    fs::write(&path, formatted).expect("fixture");
    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .arg("check")
        .arg(&path)
        .output()
        .expect("check fixture with the default style gate");
    assert!(checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert_eq!(report["errors"], serde_json::json!([]));
    for json in [false, true] {
        let mut command = Command::cargo_bin("chelis").expect("binary");
        command
            .current_dir(dir.path())
            .args(["eval", "--file"])
            .arg(&path);
        if json {
            command.arg("--json");
        }
        let output = command.output().expect("eval");
        let stderr = String::from_utf8(output.stderr).expect("stderr");
        assert_eq!(output.status.code(), Some(1), "{stderr}");
        assert!(output.stdout.is_empty(), "no successful or partial result");
        assert!(stderr.starts_with("error:"), "{stderr}");
        assert!(
            stderr.contains(
                "tensor concat cannot be represented by the static tensor DAG; use its host \
                 execution path (chelis#1906)"
            ),
            "{stderr}"
        );
        assert!(
            stderr.contains(&format!("at source span `{source_id}`")),
            "{stderr}"
        );
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert!(!stderr.contains("Box<dyn Any>"), "{stderr}");
    }
}
