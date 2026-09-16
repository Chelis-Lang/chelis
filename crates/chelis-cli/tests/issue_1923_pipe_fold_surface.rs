//! What the pipe fold changes about the public CLI surface, and what it does
//! not.
//!
//! `chelis_deep::pipe::fold_pipes` runs over the CHECKER's input
//! (chelis#1923). Three printed surfaces sit either side of that boundary and
//! only the middle one moves:
//!
//! * `chelis deep` prints the desugarer's output, which is upstream of the
//!   checker. It still prints `pipe`, and `spec/02-surf-syntax.md` section
//!   0.1's three laws are stated over exactly this tree, so nothing about
//!   them moves either.
//! * `chelis deep --annotate` prints the checker's output. It now prints the
//!   application the pipe denotes. That is a public-surface change and it is
//!   honest: the annotated tree is post-inference, and the types it carries
//!   are the types of that application.
//! * `chelis surf` resugars Deep back to Surf. It still prints `|>`.
//!
//! The pipe resugaring laws themselves are locked in
//! `issue_1242_pipe_resugaring.rs`, which this change leaves untouched.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::{TempDir, tempdir};

/// A pipe chain with both stage shapes: a bare-name stage and a call stage
/// the desugarer wraps in a synthesized unary lambda.
const PIPED: &str = "def f(x: tensor[3, f32]) -> tensor[f32] = x |> sum(cast(0, i32))\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

fn fixture(dir: &TempDir, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, source).expect("fixture");
    path
}

fn run(args: &[&str]) -> String {
    let out = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("chelis invocation");
    assert!(
        out.status.success(),
        "command {args:?} failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn deep_of(path: &Path, annotate: bool) -> String {
    if annotate {
        run(&["deep", "--annotate", path.to_str().unwrap()])
    } else {
        run(&["deep", path.to_str().unwrap()])
    }
}

/// The desugarer's output is upstream of the fold and keeps its `pipe`.
///
/// This is the user-visible face of the property that the fold returns a new
/// tree rather than rewriting its caller's program; the Deep-level assertion
/// is in `crates/chelis-types/tests/issue_1923_pipe_fold_placement.rs`.
///
/// EVIDENTIARY STATUS: disposition lock. Unchanged on `08e46ebe6`.
#[test]
fn the_unannotated_deep_still_carries_the_pipe() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "piped.ch", PIPED);
    let printed = deep_of(&path, false);
    assert!(
        printed.contains("(pipe "),
        "the desugarer's output is what section 0.1's laws are stated over: {printed}"
    );
}

/// The annotated tree is the checker's output, so it prints the application.
///
/// EVIDENTIARY STATUS: regression test on the new behaviour, which is a
/// deliberate public-surface change. On `08e46ebe6` this printed `(pipe ` in
/// four places and no `app` for the stage.
#[test]
fn the_annotated_deep_prints_the_application_the_pipe_denotes() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "piped_annotated.ch", PIPED);
    let printed = deep_of(&path, true);
    assert!(
        !printed.contains("(pipe "),
        "no pipe survives into the checked program: {printed}"
    );
    assert!(
        printed.contains("(var {span: \"surf:47..50\"} sum)"),
        "the stage callee is in callee position of an app: {printed}"
    );
}

/// Resugaring still prints the pipe, from the same Deep the first row read.
///
/// EVIDENTIARY STATUS: disposition lock. Unchanged on `08e46ebe6`.
#[test]
fn resugaring_the_deep_still_prints_the_pipe() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "piped_resugar.ch", PIPED);
    let deep_path = fixture(&dir, "piped_resugar.dp", &deep_of(&path, false));
    let printed = run(&["surf", deep_path.to_str().unwrap()]);
    assert!(
        printed.contains("x |> sum(cast(0, i32))"),
        "the pipe comes back: {printed}"
    );
}

fn checked_eval(source: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "lambda_stage.ch", source);
    let checked = run(&["check", path.to_str().unwrap()]);
    let checked: serde_json::Value = serde_json::from_str(&checked).expect("check JSON");
    assert_eq!(checked["score"], 1.0, "{checked}");
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval")
}

// [02] section 0.1 inserts the input as an argument. The callee's binders
// must retain their ordinary lexical meaning, including sequential lets.
#[test]
fn lambda_pipe_stages_preserve_lexical_bindings() {
    for (stage, call, expected) in [
        ("fn (x) -> fn (y) -> x", "result(9i64)", 3),
        ("fn (x) -> {\n y = 9i64\n x\n }", "result", 3),
        ("fn (x) -> {\n x = 9i64\n x\n }", "result", 9),
        ("fn (x: i64) -> add(x, x)", "result", 6),
    ] {
        for input in [format!("y |> {stage}"), format!("({stage})(y)")] {
            let source = format!(
                "def f(y: i64) -> i64 = {{\n result = {input}\n {call}\n}}\nout = f(3i64)\n"
            );
            let out = checked_eval(&source);
            assert!(out.status.success(), "{source}\n{out:?}");
            let value: serde_json::Value = serde_json::from_slice(&out.stdout).expect("eval JSON");
            assert_eq!(
                value["roots"][0]["value"]["value"]["value"], expected,
                "{source}"
            );
        }
    }
}

// Calling a lambda evaluates its argument before its body, even if the
// body uses that parameter conditionally, later, or inside another lambda.
#[test]
fn lambda_pipe_stages_evaluate_the_input_before_the_body() {
    for (stage, tail, succeeds) in [
        ("fn (x) -> if false then x else 7i64", "result", true),
        ("fn (x) -> fn (y) -> add(x, y)", "7i64", true),
        (
            "fn (x) -> add(trunc_div(-9223372036854775808i64, -1i64), x)",
            "result",
            false,
        ),
    ] {
        for input in [
            format!("trunc_div(1i64, z) |> {stage}"),
            format!("({stage})(trunc_div(1i64, z))"),
        ] {
            for z in [0, 1] {
                let source = format!(
                    "def f(z: i64) -> i64 = {{\n result = {input}\n {tail}\n}}\nout = f({z}i64)\n"
                );
                let out = checked_eval(&source);
                if z == 1 && succeeds {
                    assert!(out.status.success(), "{source}\n{out:?}");
                    let value: serde_json::Value =
                        serde_json::from_slice(&out.stdout).expect("eval JSON");
                    assert_eq!(value["roots"][0]["value"]["value"]["value"], 7, "{source}");
                } else {
                    assert!(!out.status.success(), "{source}\n{out:?}");
                    let expected = if z == 0 {
                        "division by zero"
                    } else {
                        "overflow"
                    };
                    assert!(
                        String::from_utf8_lossy(&out.stderr).contains(expected),
                        "{source}\n{out:?}"
                    );
                }
            }
        }
    }
}
