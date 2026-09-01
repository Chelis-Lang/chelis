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

const SHAPE_SENSITIVE_LOCAL: &str = "\
def call_local(x: f64) -> f64 = {
  conv2d = fn (a: f64, b: f64, c: f64, d: f64) -> add(add(a, b), add(c, d))
  conv2d(x, 2.0f64, 3.0f64, 4.0f64)
}
out = call_local(1.0f64)
";

const APPLIED_UPPERCASE_CONSTRUCTOR: &str = "\
type Box = | N(f64)
def wrap(N: (f64 -> f64), x: f64) -> Box = N(x)
def add_hundred(x: f64) -> f64 = add(x, 100.0f64)
out = wrap(add_hundred, 1.0f64)
";

const BARE_UPPERCASE_BINDING: &str = "\
type Box = | N(f64)
def keep(N: f64) -> f64 = N
out = keep(7.0f64)
";

const LOCAL_CONSTRUCTOR_COLLIDES_WITH_PRELUDE: &str = "\
type Wrapper =
  | Empty
  | Some(f64)
def wrap(x: f64) -> Wrapper = Some(x)
out = wrap(1.0f64)
";

const UNKNOWN_APPLIED_UPPERCASE: &str = "\
def apply_n(N: (f64 -> f64), x: f64) -> f64 = N(x)
def add_hundred(x: f64) -> f64 = add(x, 100.0f64)
out = apply_n(add_hundred, 1.0f64)
";

const INVALID_BUILTIN_CONV2D: &str = "\
def bad_conv(
  x: tensor[1, 1, 3, 3, f32],
  k: tensor[1, 1, 1, 1, f32]
) -> tensor[1, 1, 3, 3, f32] = conv2d(x, k, 1.0f64, 0)
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
fn builtin_named_local_bypasses_post_inference_shape_validation() {
    assert_eval_and_c(
        SHAPE_SENSITIVE_LOCAL,
        "shape_sensitive_local",
        "out = 10.0\n",
    );
}

#[test]
fn applied_uppercase_head_remains_a_constructor_despite_a_value_binder() {
    assert_eval_and_c(
        APPLIED_UPPERCASE_CONSTRUCTOR,
        "applied_uppercase_constructor",
        "out = N(1.0)\n",
    );
}

#[test]
fn bare_uppercase_name_remains_an_ordinary_value_binding() {
    assert_eval_and_c(
        BARE_UPPERCASE_BINDING,
        "bare_uppercase_binding",
        "out = 7.0\n",
    );
}

#[test]
fn local_constructor_identity_wins_over_same_named_prelude_constructor() {
    assert_eval_and_c(
        LOCAL_CONSTRUCTOR_COLLIDES_WITH_PRELUDE,
        "local_constructor_collides_with_prelude",
        "out = Some(1.0)\n",
    );
}

#[test]
fn applied_uppercase_head_without_a_declared_constructor_rejects() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("unknown_applied_uppercase.ch");
    write_file(&source, UNKNOWN_APPLIED_UPPERCASE);

    let output = chelis()
        .args(["check", source.to_str().unwrap()])
        .output()
        .expect("run check");
    assert_eq!(output.status.code(), Some(2));
    let report: Value = serde_json::from_slice(&output.stdout).expect("check JSON");
    assert!(
        report["errors"].as_array().is_some_and(|errors| {
            errors.iter().any(|error| {
                error["kind"] == "UnknownConstructor"
                    && error["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("unknown constructor: N"))
            })
        }),
        "check report: {report}"
    );
}

#[test]
fn real_shape_sensitive_builtin_keeps_its_validation() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("invalid_builtin_conv2d.ch");
    write_file(&source, INVALID_BUILTIN_CONV2D);

    let output = chelis()
        .args(["check", source.to_str().unwrap()])
        .output()
        .expect("run check");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("IR builtin `conv2d` requires a literal integer stride"),
        "stdout: {}",
        String::from_utf8_lossy(&output.stdout)
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
