//! Surf pipes normalize before literal typing; Deep and decompiler output use calls.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::{TempDir, tempdir};

/// A first-argument call stage normalized directly to application.
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

/// Desugaring emits applications before the checker runs.
#[test]
fn the_unannotated_deep_already_contains_applications() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "piped.ch", PIPED);
    let printed = deep_of(&path, false);
    assert!(
        !printed.contains("(pipe "),
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

/// Resugaring emits the ordinary call represented by Deep.
#[test]
fn resugaring_the_deep_prints_calls() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "piped_resugar.ch", PIPED);
    let deep_path = fixture(&dir, "piped_resugar.dp", &deep_of(&path, false));
    let printed = run(&["surf", deep_path.to_str().unwrap()]);
    assert!(
        printed.contains("sum(x, cast(0, i32))"),
        "the pipe decompiles to a call: {printed}"
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
        for input in [format!("y |> ({stage})"), format!("({stage})(y)")] {
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

// [04] §8.3: returned closures keep lexical captures. [05-HOST-1]: a
// computed callee still evaluates an unused actual before entering its body.
#[test]
fn computed_closures_keep_captures_and_eager_arguments() {
    for body in [
        "{\n later = y |> (fn (x: i64) -> fn (y: i64) -> x)\n alias = later\n alias(ACTUAL)\n}",
        "((fn (x: i64) -> fn (y: i64) -> x)(y))(ACTUAL)",
        "{\n pair = ((fn (x: i64) -> y), 7i64)\n selected = pair.0\n selected(ACTUAL)\n}",
    ] {
        for (actual, succeeds) in [("9i64", true), ("trunc_div(9i64, z)", false)] {
            let body = body.replace("ACTUAL", actual);
            let source = format!("def f(y: i64, z: i64) -> i64 = {body}\nout = f(3i64, 0i64)\n");
            let out = checked_eval(&source);
            assert_eq!(out.status.success(), succeeds, "{source}\n{out:?}");
            if succeeds {
                let value: serde_json::Value =
                    serde_json::from_slice(&out.stdout).expect("eval JSON");
                assert_eq!(value["roots"][0]["value"]["value"]["value"], 3, "{source}");
            } else {
                assert!(
                    String::from_utf8_lossy(&out.stderr).contains("division by zero"),
                    "{source}\n{out:?}"
                );
            }
        }
    }
    for (actual, succeeds) in [("9i64", true), ("trunc_div(9i64, 0i64)", false)] {
        let source = format!(
            "pair = ((fn (x: i64) -> 3i64), 7i64)\nselected = pair.0\nout = selected({actual})\n"
        );
        let out = checked_eval(&source);
        assert_eq!(out.status.success(), succeeds, "{source}\n{out:?}");
        if succeeds {
            let value: serde_json::Value = serde_json::from_slice(&out.stdout).expect("eval JSON");
            let last = value["roots"]
                .as_array()
                .expect("roots")
                .last()
                .expect("out");
            assert_eq!(last["value"]["value"]["value"], 3, "{value}");
        } else {
            assert!(
                String::from_utf8_lossy(&out.stderr).contains("division by zero"),
                "{source}\n{out:?}"
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
            format!("trunc_div(1i64, z) |> ({stage})"),
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
