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

const PARAM_BUILTIN: &str = "\
def apply(round_to: (f64 -> int64 -> f64), x: f64) -> f64 =
  round_to(x, cast(0, int64))
def scale(v: f64, p: int64) -> f64 = mul(v, 1000.0f64)
out = apply(scale, 1.55f64)
";

const PIPE_LOCAL_BUILTIN: &str = "\
def apply_pipe(x: f64) -> f64 = {
  relu = fn (v: f64) -> add(v, 100.0f64)
  x |> relu
}
out = apply_pipe(-2.0f64)
";

const NAMED_AXIS_SHAPED_LOCAL: &str = "\
def apply_sum(x: f64) -> f64 = {
  axis = cast(2, int32)
  sum = fn (value: f64, offset: int32) -> add(value, cast(offset, f64))
  sum(x, axis)
}
out = apply_sum(3.0f64)
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

fn assert_check_clean(source: &std::path::Path) {
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
}

fn assert_eval_and_c(source: &str, stem: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    assert_check_clean(&path);
    chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(expected.to_owned());
    assert_eq!(build_and_run(source, stem), expected);
}

#[test]
fn block_local_builtin_name_wins_in_check_eval_and_compiled_c() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("block_local_builtin.ch");
    write_file(&source, BLOCK_LOCAL_BUILTIN);

    assert_check_clean(&source);

    chelis()
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .stdout("out = 1550.0\n");

    let compiled = build_and_run(BLOCK_LOCAL_BUILTIN, "block_local_builtin");
    assert_eq!(compiled, "out = 1550.0\n");
}

#[test]
fn builtin_named_parameter_wins_in_check_eval_and_compiled_c() {
    assert_eval_and_c(PARAM_BUILTIN, "param_builtin", "out = 1550.0\n");
}

#[test]
fn builtin_named_pipe_stage_wins_in_check_eval_and_compiled_c() {
    assert_eval_and_c(PIPE_LOCAL_BUILTIN, "pipe_local_builtin", "out = 98.0\n");
}

#[test]
fn builtin_named_local_bypasses_named_axis_interception() {
    assert_eval_and_c(
        NAMED_AXIS_SHAPED_LOCAL,
        "named_axis_shaped_local",
        "out = 5.0\n",
    );
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
    assert_eq!(output.status.code(), Some(1));
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(
        message.contains("`def cross_entropy`")
            && message.contains("standard prelude macro")
            && message.contains("spec/02-surf-syntax.md §P5b"),
        "stderr: {message}"
    );

    for command in ["eval", "build"] {
        let mut invocation = chelis();
        if command == "eval" {
            invocation.args(["eval", "--file", source.to_str().unwrap()]);
        } else {
            invocation.args(["build", source.to_str().unwrap()]);
        }
        let output = invocation.output().expect("run lane");
        assert!(!output.status.success(), "{command} must reject");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("standard prelude macro"),
            "{command} must report the collision: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    chelis()
        .args(["validate", "--surf", source.to_str().unwrap()])
        .assert()
        .success();
}
