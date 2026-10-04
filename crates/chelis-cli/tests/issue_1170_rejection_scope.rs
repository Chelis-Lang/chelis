//! Release inventory: executable witnesses distinguish each refusal scope.
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
fn whole_program_gates_reject_even_unreachable_calls() {
    let scan = rejection(
        "def dead() -> tensor[3, f32] = tensor_scan(0.0f32, fn (previous: f32, i: i64) -> add(previous, 1.0f32), 3i64)\nout = print(7i32)\n",
    );
    assert!(scan.contains("host emission (codegen:c)"), "{scan}");
    assert!(
        scan.contains("tensor_scan") && scan.contains("[05-HOST-1]"),
        "{scan}"
    );
}

#[test]
fn live_and_unreachable_assertions_both_compile() {
    assert_eq!(
        common::build_and_run("out = test_assert(true, \"live\")\n", "scope"),
        "out = ()\n"
    );
    let source = "def dead() -> unit = test_assert(false, \"unreachable\")\nout = print(7i32)\n";
    assert_eq!(common::build_and_run(source, "scope"), "7\nout = ()\n");
}
