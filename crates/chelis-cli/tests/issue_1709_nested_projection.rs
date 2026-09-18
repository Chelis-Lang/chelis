//! chelis#1709: canonical formatting must not turn two adjacent numeric tuple
//! projections into a float token. These tests exercise the real CLI with the
//! style gate enabled.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const CANONICAL: &str = concat!(
    "def main() -> i64 = {\n",
    "  pairs = ((2i64, 3i64), (4i64, 5i64))\n",
    "  (pairs.0).0\n",
    "}\n",
);

fn run(path: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .args(args)
        .arg(path)
        .output()
        .expect("run chelis")
}

#[test]
fn nested_projection_survives_fmt_check_and_exact_eval() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("nested.ch");
    fs::write(&path, CANONICAL).expect("write fixture");

    let rendered = run(&path, &["fmt"]);
    assert!(
        rendered.status.success(),
        "fmt failed:\n{}",
        String::from_utf8_lossy(&rendered.stderr),
    );
    assert_eq!(
        rendered.stdout,
        CANONICAL.as_bytes(),
        "stdout formatting must preserve the required grouping",
    );

    let first = run(&path, &["fmt", "--inplace"]);
    assert!(
        first.status.success(),
        "fmt --inplace failed:\n{}",
        String::from_utf8_lossy(&first.stderr),
    );
    assert_eq!(
        fs::read_to_string(&path).expect("read formatted fixture"),
        CANONICAL,
        "fmt --inplace must not erase the grouping around the inner projection",
    );

    let second = run(&path, &["fmt", "--inplace"]);
    assert!(
        second.status.success(),
        "the second formatter pass must succeed"
    );
    assert_eq!(
        fs::read_to_string(&path).expect("read fixed-point fixture"),
        CANONICAL,
        "formatter output must be a fixed point",
    );
    assert!(
        run(&path, &["fmt", "--check"]).status.success(),
        "formatter output must pass fmt --check",
    );

    let checked = run(&path, &["check", "--show-inferred"]);
    assert!(
        checked.status.success(),
        "style-gated check failed:\n{}\n{}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr),
    );
    let report: Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert_eq!(report["score"], 1.0);
    assert_eq!(report["errors"], serde_json::json!([]));
    let main = report["inferred_signatures"]
        .as_array()
        .expect("inferred signatures")
        .iter()
        .find(|entry| entry["function"] == "main")
        .expect("main signature");
    assert_eq!(main["display_signature"], "() -> i64");

    let evaluated = run(&path, &["eval", "--file"]);
    assert!(
        evaluated.status.success(),
        "style-gated eval failed:\n{}\n{}",
        String::from_utf8_lossy(&evaluated.stdout),
        String::from_utf8_lossy(&evaluated.stderr),
    );
    assert_eq!(evaluated.stdout, b"main = 2\n");
    assert!(evaluated.stderr.is_empty());
}

#[test]
fn ungrouped_adjacent_numeric_projections_remain_rejected_without_rewrite() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("ambiguous.ch");
    let source = "value = pairs.0.0\n";
    fs::write(&path, source).expect("write fixture");

    let output = run(&path, &["fmt", "--inplace"]);
    assert!(
        !output.status.success(),
        "the lexer-ambiguous spelling must remain rejected",
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Float(0.0)"),
        "the rejection must remain the tuple-index lexical ambiguity:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        fs::read_to_string(&path).expect("read rejected fixture"),
        source,
        "failed fmt --inplace must not mutate the input",
    );
}
