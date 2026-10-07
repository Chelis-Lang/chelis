//! Deep and Surf inputs that checking accepts must not abort other CLI lanes.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;
use tempfile::tempdir;

fn nested_deep_program(dir: &Path, depth: usize) -> PathBuf {
    let body = format!(
        "{}(lit {{type: (t-prim {{}} i32)}} 1){}",
        "(app {} (var {} id) ".repeat(depth),
        ")".repeat(depth)
    );
    let source = format!(
        "(defsig {{}} id (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} i32)))\n\
         (def {{}} id (fn {{}} (params {{}} x) (var {{}} x)))\n\
         (defsig {{}} main (t-fn {{}} (t-prim {{}} i32)))\n\
         (def {{}} main (fn {{}} (params {{}}) {body}))\n"
    );
    let path = dir.join(format!("app_{depth}.dp"));
    fs::write(&path, source).expect("write Deep program");
    path
}

fn run(args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("run chelis")
}

fn assert_exited(output: &Output, label: &str) {
    assert!(
        output.status.code().is_some(),
        "{label} died on a signal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn deep_application_chain_reaches_eval_cost_and_build() {
    let dir = tempdir().expect("tempdir");
    let path = nested_deep_program(dir.path(), 1_000);
    let file = path.to_str().expect("utf8 path");
    let check = run(&["check", file]);
    assert!(
        check.status.success(),
        "check: {}",
        String::from_utf8_lossy(&check.stdout)
    );

    let eval = run(&["eval", "--file", file]);
    assert_exited(&eval, "eval");
    assert!(
        eval.status.success(),
        "eval: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert!(String::from_utf8_lossy(&eval.stdout).contains('1'));

    let cost = run(&["cost", file, "--json"]);
    assert_exited(&cost, "cost");
    assert!(
        cost.status.success(),
        "cost: {}",
        String::from_utf8_lossy(&cost.stderr)
    );
    serde_json::from_slice::<serde_json::Value>(&cost.stdout).expect("cost JSON");

    let output = dir.path().join("app.c");
    let build = run(&[
        "build",
        file,
        "--target",
        "c",
        "--emit-c",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_exited(&build, "build");
    assert!(
        build.status.success(),
        "build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(output.exists(), "build emitted C");
}

#[test]
fn deep_parser_limit_is_diagnostic_across_cli_lanes() {
    let dir = tempdir().expect("tempdir");
    let path = nested_deep_program(dir.path(), 400_000);
    let file = path.to_str().expect("utf8 path");
    for args in [
        vec!["eval", "--file", file],
        vec!["cost", file, "--json"],
        vec!["build", file, "--target", "c", "--emit-c"],
    ] {
        let output = run(&args);
        assert_exited(&output, args[0]);
        assert!(
            !output.status.success(),
            "{} accepted extreme nesting",
            args[0]
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("Deep input nests deeper than the parser supports at byte "),
            "{} did not report the located parser error: {}",
            args[0],
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn nested_surf_calls_build_without_a_stack_abort() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested.ch");
    let expression = (0..40).fold("1i64".to_string(), |inner, _| {
        format!("f(len([{inner}, 2i64]))")
    });
    fs::write(
        &path,
        format!("def f(n: i64) -> i64 = n\na = {expression}\n"),
    )
    .expect("write Surf program");
    let file = path.to_str().unwrap();
    let check = run(&["check", file]);
    assert!(
        check.status.success(),
        "check: {}",
        String::from_utf8_lossy(&check.stdout)
    );
    let output = dir.path().join("nested.c");
    let build = run(&[
        "build",
        file,
        "--target",
        "c",
        "--emit-c",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_exited(&build, "nested Surf build");
    assert!(
        build.status.success(),
        "build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
}
