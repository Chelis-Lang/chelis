//! Release inventory: a refusal states which part of the program it covers.
#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn rejection(source: &str) -> String {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scope.ch");
    fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--target", "c", "--emit-c", "--output"])
        .arg(dir.path().join("out"))
        .arg(path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("unsupported:"), "{stderr}");
    assert!(!dir.path().join("out/scope.c").exists());
    stderr
}

#[test]
fn whole_program_gates_state_their_scope() {
    let scan = rejection(
        "def dead() -> tensor[3, f32] = tensor_scan(0.0f32, fn (previous: f32, i: i64) -> add(previous, 1.0f32), 3i64)\nout = print(7i32)\n",
    );
    assert!(
        scan.contains("rejection scope: whole checked program"),
        "{scan}"
    );
    assert!(
        scan.contains("tensor_scan") && scan.contains("[05-HOST-1]"),
        "{scan}"
    );
    let clock = rejection("reading = clock_wall_read()\n");
    assert!(
        clock.contains("rejection scope: whole lowered host program"),
        "{clock}"
    );
    assert!(
        clock.contains("clock_wall_read") && clock.contains("[05-HOST-2]"),
        "{clock}"
    );
}

#[test]
fn live_assertion_refuses_but_unreachable_assertion_does_not() {
    let live = rejection("out = test_assert(true, \"live\")\n");
    assert!(
        live.contains("rejection scope: emitted live host code"),
        "{live}"
    );
    assert!(live.contains("test_assert"), "{live}");
    let source = "def dead() -> unit = test_assert(false, \"unreachable\")\nout = print(7i32)\n";
    assert_eq!(common::build_and_run(source, "scope"), "7\nout = ()\n");
}
