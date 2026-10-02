//! Issue #1355 through the CLI: `chelis check` must not score 1.0 on a
//! `diagonal` whose declared return extent the runtime cannot produce, and the
//! accepted program's runtime shape must equal its declared shape.
//!
//! `[05-OP-33]` (`spec/05-risc-primitives.md`) fixes the extent exactly:
//! `diagonal` "replaces the retained first axis extent with the smaller
//! selected extent". `infer_diagonal_result_type` previously widened every
//! unequal literal pair to a wildcard, which unified with any declared extent,
//! so `-> tensor[4, f32]` on a `tensor[3, 4, f32]` operand scored a clean 1.0
//! while the runtime returned `tensor[3]`.
//!
//! Every fixture uses NON-SQUARE extents: a square operand cannot distinguish
//! the old behaviour from the correct one. The fixtures are canonical Surf, so
//! the style gate runs on them rather than being disabled.

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// The `tensor[3, 4, f32]` operand shared by every fixture, whose diagonal
/// over axes (0, 1) is the three cells `[1.0, 6.0, 11.0]`.
const OPERAND: &str =
    "m = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])\n";

fn program(declared: &str) -> String {
    format!(
        "def f(x: tensor[3, 4, f32]) -> tensor[{declared}, f32] = diagonal(x, 0, 1)\n\
         {OPERAND}out = f(m)\n"
    )
}

fn write_program(source: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("diagonal.ch");
    std::fs::write(&path, source).expect("write program");
    (dir, path)
}

fn check_json(path: &Path) -> serde_json::Value {
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    serde_json::from_slice(&out.stdout).expect("chelis check must emit JSON")
}

fn eval_file(path: &Path) -> (bool, String, String) {
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// REGRESSION TEST. The issue program scored a perfect 1.0 with an empty error
/// list before the fix. It must now score strictly below 1.0 and carry exactly
/// one `DimensionMismatch` naming the inferred `tensor[3, f32]`.
#[test]
fn wrong_declared_extent_does_not_score_one() {
    let (_dir, path) = write_program(&program("4"));
    let json = check_json(&path);

    let score = json["score"].as_f64().expect("numeric score");
    assert!(
        score < 1.0,
        "a diagonal declared tensor[4, f32] on a tensor[3, 4, f32] operand \
         must not score 1.0, got {score}"
    );

    let errors = json["errors"].as_array().expect("errors array");
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error, got {errors:?}");
    };
    assert_eq!(error["kind"].as_str(), Some("DimensionMismatch"));
    assert_eq!(
        error["expected"], "(tensor[3, 4, f32]) -> tensor[4, f32]",
        "the declared extent is 4"
    );
    assert_eq!(
        error["got"], "(tensor[3, 4, f32]) -> tensor[3, f32]",
        "the inferred diagonal extent is 3"
    );
}

/// REGRESSION TEST. `chelis eval --file` refuses the same program before
/// running it, so no lane executes a shape the declaration denies. Before the
/// fix it ran and printed a `tensor[3]` result under a `tensor[4]` declaration.
#[test]
fn wrong_declared_extent_is_rejected_before_evaluation() {
    let (_dir, path) = write_program(&program("4"));
    let (ok, stdout, stderr) = eval_file(&path);

    assert!(!ok, "eval must fail; stdout was {stdout}");
    assert!(
        stderr.contains("DimensionMismatch") && stderr.contains("tensor[3, f32]"),
        "eval must reject with the extent mismatch, got {stderr}"
    );
    assert!(
        !stdout.contains("out ="),
        "eval must not print a result for a rejected program, got {stdout}"
    );
}

/// DISPOSITION LOCK plus the shape-agreement evidence. The correct declaration
/// still scores 1.0 (green before the fix too, so it proves only that the
/// narrowing is not over-applied), and the evaluated result carries `shape=[3]`
/// -- the same extent the declaration now states. That agreement is the point
/// of the fix: [05-OP-33]'s smaller selected extent is what the runtime
/// produces.
#[test]
fn correct_declared_extent_scores_one_and_matches_the_runtime_shape() {
    let (_dir, path) = write_program(&program("3"));

    let json = check_json(&path);
    assert_eq!(
        json["score"].as_f64(),
        Some(1.0),
        "the correct declaration must still score 1.0, got {json}"
    );
    assert_eq!(
        json["errors"].as_array().map(Vec::len),
        Some(0),
        "a score-1 program must carry an empty error list, got {json}"
    );

    let (ok, stdout, stderr) = eval_file(&path);
    assert!(ok, "eval must succeed; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
        "the runtime shape must equal the declared tensor[3, f32], got {stdout}"
    );
}

// ---------------------------------------------------------------------------
// chelis#1739 through the CLI: the literal selected axis bounds the declared
// result extent even when the other selected axis is symbolic.
//
// `min(n, 4) <= 4` for every runtime `n`, so `-> tensor[9, f32]` on a
// `tensor[n, 4, f32]` operand is unreachable and must not score 1.0. At the
// bound stays accepted and must still run, which is the control that the
// rejection is an upper bound and not an equality.
// ---------------------------------------------------------------------------

/// A three-row operand, whose diagonal against a four-wide axis is
/// `[1.0, 6.0, 11.0]`.
const THREE_BY_FOUR: &str =
    "m = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])\n";

/// A five-row operand, whose diagonal against a four-wide axis reaches the
/// bound exactly: `[1.0, 6.0, 11.0, 16.0]`.
const FIVE_BY_FOUR: &str = "m = to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], \
                            [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], \
                            [17.0, 18.0, 19.0, 20.0]])\n";

fn symbolic_program(declared: &str, operand: &str) -> String {
    format!(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[{declared}, f32] = diagonal(x, 0, 1)\n\
         {operand}out = f(m)\n"
    )
}

/// REGRESSION TEST. The chelis#1739 program scored a perfect 1.0 with an empty
/// error list before the fix, because the mixed (symbolic, literal) pair left
/// the result extent a wildcard. It must now score strictly below 1.0 and carry
/// exactly one `DimensionMismatch` naming the bound and the declared extent.
#[test]
fn a_symbolic_pair_declared_wider_than_its_literal_axis_does_not_score_one() {
    let source = symbolic_program("9", THREE_BY_FOUR);
    let (_dir, path) = write_program(&source);
    let json = check_json(&path);

    let score = json["score"].as_f64().expect("numeric score");
    assert!(
        score < 1.0,
        "a diagonal declared tensor[9, f32] on a tensor[n, 4, f32] operand \
         must not score 1.0, got {score}"
    );

    let errors = json["errors"].as_array().expect("errors array");
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error, got {errors:?}");
    };
    assert_eq!(error["kind"].as_str(), Some("DimensionMismatch"));
    assert_eq!(error["expected"], "extent at most 4");
    assert_eq!(error["got"], "declared extent 9");
    assert_eq!(
        error["span"]["offset"].as_u64(),
        source.find("diagonal(").map(|offset| offset as u64),
        "{error:?}"
    );
    assert!(error["message"].as_str().unwrap().contains("axis 1"), "{error:?}");
}

/// REGRESSION TEST. `chelis eval --file` refuses the same program before
/// running it. Before the fix it ran and printed a `shape=[3]` result under a
/// `tensor[9, f32]` declaration.
#[test]
fn a_symbolic_pair_declared_wider_than_its_literal_axis_is_rejected_before_evaluation() {
    let (_dir, path) = write_program(&symbolic_program("9", THREE_BY_FOUR));
    let (ok, stdout, stderr) = eval_file(&path);

    assert!(!ok, "eval must fail; stdout was {stdout}");
    assert!(
        stderr.contains("DimensionMismatch") && stderr.contains("at most 4"),
        "eval must reject with the upper-bound mismatch, got {stderr}"
    );
    assert!(
        !stdout.contains("out ="),
        "eval must not print a result for a rejected program, got {stdout}"
    );
}

/// DISPOSITION LOCK plus the shape-agreement evidence. A declaration AT the
/// bound is satisfiable and must still score 1.0 and run, with the runtime
/// shape equal to the declared one. Green before the fix too, so it proves
/// only that the bound rejects strictly greater rather than not-equal.
#[test]
fn a_symbolic_pair_declared_at_its_literal_axis_scores_one_and_runs() {
    let (_dir, path) = write_program(&symbolic_program("4", FIVE_BY_FOUR));

    let json = check_json(&path);
    assert_eq!(
        json["score"].as_f64(),
        Some(1.0),
        "a declaration at the bound must still score 1.0, got {json}"
    );
    assert_eq!(
        json["errors"].as_array().map(Vec::len),
        Some(0),
        "a score-1 program must carry an empty error list, got {json}"
    );

    let (ok, stdout, stderr) = eval_file(&path);
    assert!(ok, "eval must succeed; stderr was {stderr}");
    assert!(
        stdout.contains("out = tensor(shape=[4], data=[1.0, 6.0, 11.0, 16.0])"),
        "the runtime shape must equal the declared tensor[4, f32], got {stdout}"
    );
}
