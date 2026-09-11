//! chelis#1512 at the CLI surface: a builtin route whose operand is still a
//! type variable when the route runs must reject the same programs the
//! resolved route rejects, and `chelis eval` must refuse such a program
//! BEFORE it evaluates anything.
//!
//! The in-crate matrix (`crates/chelis-types/tests/unresolved_operand_matrix.rs`)
//! owns the per-route cells. This file owns the two things only the CLI can
//! show: that `chelis check`'s machine-facing JSON carries the rejection in
//! its `errors` array with a non-perfect score, and that the refusal happens
//! at check time rather than at a runtime crash.
//!
//! Every assertion here is a REGRESSION TEST unless its comment says
//! otherwise: the late-bound programs below were accepted at score 1.0 with
//! an empty error list on `e813415d0`.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;

/// `chelis check` exits 2 when the errors array is non-empty (chelis#207).
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

fn write_program(prefix: &str, source: &str) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    tmp.write_all(source.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");
    tmp
}

fn check_json(source: &str, prefix: &str) -> (Option<i32>, Value) {
    let tmp = write_program(prefix, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let json = serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("chelis check stdout must be JSON: {err}\n{stdout}"));
    (output.status.code(), json)
}

fn messages(json: &Value) -> Vec<String> {
    json.get("errors")
        .and_then(Value::as_array)
        .expect("errors array")
        .iter()
        .filter_map(|e| e.get("message").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// The route's own diagnostic reaches the machine-facing JSON, and the score
/// is no longer perfect. A clean `chelis check` is what every downstream
/// consumer reads, so an acceptance hole here is invisible everywhere.
fn assert_check_rejects(source: &str, prefix: &str, diagnostic: &str) {
    let (code, json) = check_json(source, prefix);
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "{prefix}: a rejected program must exit {CHECK_ERRORS_EXIT_CODE}; json={json}"
    );
    let found = messages(&json);
    assert!(
        found.iter().any(|m| m.contains(diagnostic)),
        "{prefix}: expected {diagnostic:?} in the errors array, got {found:?}"
    );
    let score = json.get("score").and_then(Value::as_f64);
    assert!(
        score.is_none_or(|score| score < 1.0),
        "{prefix}: a program with errors must not report a perfect score, got {score:?}"
    );
}

/// CONTROL beside every rejection: the same call written correctly still
/// checks clean at a perfect score. A repair that rejects correct programs
/// fails here.
fn assert_check_accepts(source: &str, prefix: &str) {
    let (code, json) = check_json(source, prefix);
    assert_eq!(
        code,
        Some(0),
        "{prefix}: a correct program must exit 0; json={json}"
    );
    assert!(
        messages(&json).is_empty(),
        "{prefix}: a correct program must report no errors, got {:?}",
        messages(&json)
    );
}

const PERMUTE_BAD: &str = "module Probe\n\
def main() -> tensor[2, 3, f32] = {\n  \
  x = to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]])\n  \
  g = fn (t) -> permute(t, 1i32, 0i32, 2i32)\n  \
  g(x)\n}\n";

const PERMUTE_GOOD: &str = "module Probe\n\
def main() -> tensor[2, 3, f32] = {\n  \
  x = to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]])\n  \
  g = fn (t) -> permute(t, 1i32, 0i32)\n  \
  g(x)\n}\n";

const SHRINK_BAD: &str = "module Probe\n\
def main() -> tensor[1, 2, f32] = {\n  \
  x = to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]])\n  \
  g = fn (t) -> shrink(t, [[0i64, 1i64]])\n  \
  g(x)\n}\n";

const LEN_BAD: &str = "module Probe\n\
def f(x: List[int32]) -> int64 = {\n  \
  g = fn (t) -> len(to_tensor(t))\n  \
  g(x)\n}\n";

const LEN_GOOD: &str = "module Probe\n\
def f(x: List[int32]) -> int64 = {\n  \
  g = fn (t) -> len(t)\n  \
  g(x)\n}\n";

#[test]
fn check_rejects_a_window_route_over_a_late_bound_operand() {
    assert_check_rejects(
        PERMUTE_BAD,
        "issue1512-permute-bad-",
        "permute expects 2 axis indices for rank 2 tensor, got 3",
    );
    assert_check_accepts(PERMUTE_GOOD, "issue1512-permute-good-");
    assert_check_rejects(
        SHRINK_BAD,
        "issue1512-shrink-bad-",
        "shrink expects 2 bounds pairs for rank 2 tensor, got 1",
    );
}

#[test]
fn check_rejects_a_collection_route_over_a_late_bound_operand() {
    assert_check_rejects(
        LEN_BAD,
        "issue1512-len-bad-",
        "len expects List or Dict input",
    );
    assert_check_accepts(LEN_GOOD, "issue1512-len-good-");
}

/// `chelis eval` must refuse BEFORE evaluating. The distinction matters: an
/// accepted-then-crashing program and a refused program are different
/// products, and chelis#1512's witnesses reached execution.
#[test]
fn eval_refuses_a_late_bound_operand_before_evaluating() {
    let tmp = write_program("issue1512-eval-bad-", PERMUTE_BAD);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "eval must refuse the program; stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("permute expects 2 axis indices for rank 2 tensor, got 3"),
        "the refusal must carry the route's own diagnostic, got:\n{stderr}"
    );

    // CONTROL: the correct twin evaluates.
    let good = write_program("issue1512-eval-good-", PERMUTE_GOOD);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", good.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "the correct twin must evaluate; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}
