//! Acceptance oracle for chelis#1076 / chelis#672: compiler-provided names
//! never silently replace the lexically selected user callable.
//!
//! Run:
//! `cargo nextest run -p chelis-cli --test issue_1076_672_name_precedence --no-fail-fast`

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, write_file};

const BLOCK_LOCAL_BUILTIN: &str = "\
def apply3(x: f64) -> f64 = {
  round_to = fn (v: f64, p: int64) -> mul(v, 1000.0f64)
  round_to(x, cast(0, int64))
}
out = apply3(1.55f64)
";

const PRELUDE_COLLISION: &str = "\
def cross_entropy(logits: tensor[2, 3, f32], labels: tensor[2, 3, f32]) -> tensor[f32] =
  scalar_to_tensor(cast(999.0, f32))
def use_it(logits: tensor[2, 3, f32], labels: tensor[2, 3, f32]) -> tensor[f32] =
  cross_entropy(logits, labels)
";

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

#[test]
fn block_local_builtin_name_wins_in_check_eval_and_compiled_c() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("block_local_builtin.ch");
    write_file(&source, BLOCK_LOCAL_BUILTIN);

    let check = chelis()
        .args(["check", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let report: Value = serde_json::from_slice(&check).expect("check JSON");
    assert_eq!(report["score"], 1.0, "check report: {report}");
    assert_eq!(
        report["errors"],
        Value::Array(Vec::new()),
        "check report: {report}"
    );

    chelis()
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .stdout("out = 1550.0\n");

    let compiled = build_and_run(BLOCK_LOCAL_BUILTIN, "block_local_builtin");
    assert_eq!(compiled, "out = 1550.0\n");
}

#[test]
fn ordinary_def_cannot_collide_with_standard_prelude_macro() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("prelude_collision.ch");
    write_file(&source, PRELUDE_COLLISION);

    let output = chelis()
        .args(["check", source.to_str().unwrap()])
        .output()
        .expect("run check");
    assert_eq!(output.status.code(), Some(2));
    let report: Value = serde_json::from_slice(&output.stdout).expect("check JSON");
    let errors = report["errors"].as_array().expect("errors array");
    assert_eq!(errors.len(), 1, "report: {report}");
    assert_eq!(errors[0]["kind"], "macro_error", "report: {report}");
    let message = errors[0]["message"].as_str().expect("message");
    assert!(
        message.contains("`def cross_entropy`")
            && message.contains("standard prelude macro")
            && message.contains("spec/02-surf-syntax.md §P5b"),
        "report: {report}"
    );
}
