//! Host lowering must handle Deep nesting that the checker accepts.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn nested_deep_program(depth: usize) -> String {
    let body = format!(
        "{}(lit {{type: (t-prim {{}} i32)}} 1){}",
        "(app {} (var {} id) ".repeat(depth),
        ")".repeat(depth)
    );
    format!(
        "(defsig {{}} id (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} i32)))\n\
         (def {{}} id (fn {{}} (params {{}} x) (var {{}} x)))\n\
         (defsig {{}} main (t-fn {{}} (t-prim {{}} i32)))\n\
         (def {{}} main (fn {{}} (params {{}}) {body}))\n"
    )
}

fn checked_chain(depth: usize) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("nested.dp");
    fs::write(&file, nested_deep_program(depth)).expect("write Deep program");

    let check = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", file.to_str().expect("utf8 path")])
        .output()
        .expect("run check");
    assert!(
        check.status.success(),
        "check: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    (dir, file)
}

fn build(file: &std::path::Path, output: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            file.to_str().expect("utf8 path"),
            "--target",
            "c",
            "--emit-c",
            "--output",
            output.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("run build")
}

#[test]
fn a_shallow_checked_chain_builds_c() {
    let (dir, file) = checked_chain(1_000);
    let output = dir.path().join("nested.c");
    let result = build(&file, &output);
    assert!(
        result.status.success(),
        "build: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists(), "build emitted C");
}

#[test]
fn a_deep_checked_chain_completes_or_reports_a_lowering_budget() {
    let (dir, file) = checked_chain(4_000);
    let output = dir.path().join("nested.c");
    let surf = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["surf", file.to_str().expect("utf8 path")])
        .output()
        .expect("run surf");
    assert!(
        surf.status.success(),
        "surf: {}",
        String::from_utf8_lossy(&surf.stderr)
    );
    assert!(!surf.stdout.is_empty());

    let fmt = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["fmt", "--check", file.to_str().expect("utf8 path")])
        .output()
        .expect("run fmt");
    assert!(fmt.status.code().is_some(), "fmt died on a signal");
    assert!(
        String::from_utf8_lossy(&fmt.stderr).contains("not canonically formatted"),
        "fmt: {}",
        String::from_utf8_lossy(&fmt.stderr)
    );

    let cost = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["cost", file.to_str().expect("utf8 path"), "--json"])
        .output()
        .expect("run cost");
    assert!(
        cost.status.success(),
        "cost: {}",
        String::from_utf8_lossy(&cost.stderr)
    );
    serde_json::from_slice::<serde_json::Value>(&cost.stdout).expect("cost JSON");

    let result = build(&file, &output);
    assert!(result.status.code().is_some(), "build died on a signal");
    if result.status.success() {
        assert!(output.exists(), "build emitted C");
        return;
    }
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("host lowering stack budget exhausted"),
        "{stderr}"
    );
    assert!(
        stderr.contains(" at byte "),
        "diagnostic lacks the source span: {stderr}"
    );
}
