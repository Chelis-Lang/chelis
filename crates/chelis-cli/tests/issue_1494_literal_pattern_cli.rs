//! Issue #1494 through the CLI: a `match` arm whose literal pattern cannot
//! denote the scrutinee's primitive must not score 1.0, and must be refused
//! before any lane runs.
//!
//! `spec/04-type-system.md` [04-PAT-1] states the rule. Before the fix
//! `chelis check` scored the issue program a clean 1.0 with an empty error
//! list and `chelis eval --file` ran it and printed `r = 2.5`: the `1.5` arm
//! can never match an `i32` scrutinee, so the program carried a silently
//! dead arm and the score said nothing was wrong.
//!
//! The fixtures are canonical Surf, so the style gate runs on them rather than
//! being disabled. What is claimed here is the two spellings named in the
//! tests: the float-versus-integer family violation and its matching-family
//! positive control.

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// The issue program, parameterised by the literal pattern on its first arm.
/// `1.5` is the float-versus-`i32` violation; `1` is the well-formed control.
fn program(pattern: &str) -> String {
    // Canonical Surf: `chelis fmt` removes the blank lines between top-level
    // declarations, and these fixtures run WITH the style gate enabled so the
    // corpus stays formatter-clean.
    format!(
        "module ScrutineeSigned\n\
         def g(n: i32) -> i32 = add(1, n)\n\
         r: f32 = match g(2) with {{\n\
         \x20 | {pattern} => 1.5\n\
         \x20 | _ => 2.5\n\
         }}\n"
    )
}

fn write_program(source: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pattern.ch");
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
/// list. It must now score strictly below 1.0 and carry exactly one
/// `TypeMismatch` citing [04-PAT-1].
#[test]
fn dead_literal_pattern_arm_does_not_score_one() {
    let (_dir, path) = write_program(&program("1.5"));
    let json = check_json(&path);

    let score = json["score"].as_f64().expect("numeric score");
    assert!(
        score < 1.0,
        "an f32 literal pattern against an i32 scrutinee must not score 1.0, got {score}"
    );

    let errors = json["errors"].as_array().expect("errors array");
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error, got {errors:?}");
    };
    assert_eq!(error["kind"].as_str(), Some("TypeMismatch"));
    let message = error["message"].as_str().expect("message string");
    assert!(
        message.contains("[04-PAT-1]")
            && message.contains("floating-point literal pattern `1.5`")
            && message.contains("`i32`"),
        "the diagnostic must cite the atom and name both sides, got {message}"
    );
}

/// REGRESSION TEST. `chelis eval --file` refuses the same program before
/// running it. Before the fix it ran and printed `r = 2.5`, a value produced by
/// falling through a dead arm nothing had reported.
#[test]
fn dead_literal_pattern_arm_is_rejected_before_evaluation() {
    let (_dir, path) = write_program(&program("1.5"));
    let (ok, stdout, stderr) = eval_file(&path);

    assert!(!ok, "eval must fail; stdout was {stdout}");
    assert!(
        stderr.contains("TypeMismatch") && stderr.contains("[04-PAT-1]"),
        "eval must reject with the [04-PAT-1] mismatch, got {stderr}"
    );
    assert!(
        !stdout.contains("r ="),
        "eval must not print a result for a rejected program, got {stdout}"
    );
}

/// DISPOSITION LOCK. The same program with a matching literal family still
/// scores 1.0 and still evaluates. Green before the fix and green after, so it
/// proves only that the rejection did not creep into a well-formed match. The
/// `1` arm does not match `g(2) == 3`, so the fall-through result is the same
/// `2.5` the issue program printed; the difference is that here the arm is
/// merely not taken rather than impossible.
#[test]
fn matching_literal_pattern_family_still_scores_one_and_evaluates() {
    let (_dir, path) = write_program(&program("1"));

    let json = check_json(&path);
    assert_eq!(
        json["score"].as_f64(),
        Some(1.0),
        "a matching literal family must still score 1.0, got {json}"
    );
    assert_eq!(
        json["errors"].as_array().map(Vec::len),
        Some(0),
        "a score-1 program must carry an empty error list, got {json}"
    );

    let (ok, stdout, stderr) = eval_file(&path);
    assert!(ok, "eval must succeed; stderr was {stderr}");
    assert!(
        stdout.contains("r = 2.5"),
        "the well-formed program must still evaluate, got {stdout}"
    );
}
