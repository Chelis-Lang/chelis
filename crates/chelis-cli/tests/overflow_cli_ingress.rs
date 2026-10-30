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

// spec/03 §4.4 and [05-OP-54]: knowing a literal List's length does not
// permit skipping its element expressions, their effects, or their traps.
#[test]
fn nested_list_length_preserves_eager_element_work() {
    for traps in [false, true] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("eager.ch");
        let seed = if traps { 0 } else { 1 };
        let expression = (0..8).fold(format!("leaf({seed}i64)"), |inner, _| {
            format!("f(len([{inner}, later(2i64)]))")
        });
        fs::write(
            &path,
            format!(
                "def f(n: i64) -> i64 = n\n\
                 def leaf(n: i64) -> i64 ! {{ IO }} = {{\n\
                   _ = print(\"leaf\")\n trunc_div(1i64, n)\n}}\n\
                 def later(n: i64) -> i64 ! {{ IO }} = {{\n\
                   _ = print(\"later\")\n n\n}}\n\
                 a = {expression}\n"
            ),
        )
        .expect("write eager List program");
        let file = path.to_str().unwrap();
        let check = run(&["check", file]);
        assert!(check.status.success(), "{check:?}");
        let report: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
        assert_eq!(report["score"], 1);
        assert_eq!(report["errors"], serde_json::json!([]));
        let evaluated = run(&["eval", "--file", file]);
        let out = dir.path().join("native");
        let build = run(&["build", file, "--target", "c", "-o", out.to_str().unwrap()]);
        assert!(build.status.success(), "{build:?}");
        let native = std::process::Command::new(out.join("eager"))
            .output()
            .expect("execute generated C");
        assert_eq!(evaluated.status.success(), !traps, "{evaluated:?}");
        assert_eq!(native.status.success(), !traps, "{native:?}");
        assert_eq!(evaluated.stdout, native.stdout, "complete observable work");
        let stdout = String::from_utf8_lossy(&native.stdout);
        assert_eq!(stdout.matches("leaf\n").count(), 1, "{stdout}");
        assert_eq!(
            stdout.matches("later\n").count(),
            if traps { 0 } else { 8 },
            "{stdout}"
        );
        if traps {
            for output in [evaluated, native] {
                assert_eq!(
                    String::from_utf8_lossy(&output.stderr),
                    "numeric trap: division by zero in trunc_div at i64\n",
                    "{output:?}"
                );
            }
        } else {
            assert!(stdout.ends_with("a = 2\n"), "{stdout}");
        }
    }
}
