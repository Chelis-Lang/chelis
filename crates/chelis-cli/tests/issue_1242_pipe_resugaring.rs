//! spec/02 §0.1: pipe resugaring preserves the program or reports failure.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn unsafe_call_first_stages_emit_no_substitute_program() {
    let dir = tempdir().unwrap();
    for (name, param, body) in [
        ("lone", "p", "(app {} (var {} g) (var {} p))"),
        ("callee", "p", "(app {} (var {} p) (var {} p) (var {} y))"),
        ("repeated", "p", "(app {} (var {} f) (var {} p) (var {} p))"),
        (
            "nested",
            "p",
            "(app {} (var {} f) (var {} p) (app {} (var {} g) (var {} p)))",
        ),
        (
            "operator",
            "mul",
            "(app {} (var {} f) (var {} mul) (app {} (var {} mul) (var {} a) (var {} b)))",
        ),
    ] {
        let path = dir.path().join(format!("{name}.dp"));
        fs::write(
            &path,
            format!(
                "(def {{}} result (pipe {{}} (var {{}} x) \
                 (fn {{surf_pipe_stage: \"call-first\"}} (params {{}} {param}) {body})))"
            ),
        )
        .unwrap();
        Command::cargo_bin("chelis")
            .unwrap()
            .arg("surf")
            .arg(&path)
            .assert()
            .failure()
            .stdout("")
            .stderr(predicate::str::contains("Deep format 0.20"));
    }
}

#[test]
fn safe_call_first_programs_reparse_and_preserve_execution() {
    let dir = tempdir().unwrap();
    for (name, program, expected) in [
        (
            "ordinary",
            "def combine(x: i32, y: i32) -> i32 = x + y\nresult = 3 |> combine(4)\n",
            "result = 7\n",
        ),
        (
            "nested",
            "result = 3 |> mul(4 |> add(2))\n",
            "result = 18\n",
        ),
        (
            "special",
            "result = to_tensor([3], i32) |> copy |> realize\n",
            "result = tensor(shape=[1], data=[3])\n",
        ),
    ] {
        let source = dir.path().join(format!("{name}.ch"));
        let deep_path = dir.path().join(format!("{name}.dp"));
        let surf_path = dir.path().join(format!("{name}_roundtrip.ch"));
        fs::write(&source, program).unwrap();
        let deep = Command::cargo_bin("chelis")
            .unwrap()
            .arg("deep")
            .arg(&source)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        fs::write(&deep_path, deep).unwrap();
        let rendered = Command::cargo_bin("chelis")
            .unwrap()
            .arg("surf")
            .arg(&deep_path)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        fs::write(&surf_path, rendered).unwrap();
        Command::cargo_bin("chelis")
            .unwrap()
            .args(["fmt", "--check"])
            .arg(&surf_path)
            .assert()
            .success();
        for path in [&source, &deep_path, &surf_path] {
            Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--file"])
                .arg(path)
                .assert()
                .success()
                .stdout(expected);
        }
    }
}
