//! The CLI rejects patterns that cannot match their scrutinee before execution.

use assert_cmd::Command;
use std::fs;

fn run(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("CLI runs")
}

#[test]
fn mismatched_patterns_fail_check_eval_and_build() {
    for (name, source) in [
        (
            "nominal",
            "type T = | A(i32)\ntype U = | C(i32)\n\
             def f(u: U) -> i32 = match u with { | A(a) => a | C(c) => c }\n\
             out = f(C(5i32))\n",
        ),
        (
            "scalar_tuple",
            "def f(n: i32) -> i32 = match n with { | (a, b) => a | _ => 0 }\n\
             out = f(5i32)\n",
        ),
        (
            "tuple_arity",
            "def f(t: (i32, i32)) -> i32 = match t with { | (a, b, c) => a | _ => 0 }\n\
             out = f((5i32, 6i32))\n",
        ),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("{name}.ch"));
        fs::write(&path, source).expect("write fixture");
        let file = path.to_str().expect("UTF-8 path");

        let checked = run(&["check", file]);
        let report = String::from_utf8_lossy(&checked.stdout);
        assert!(
            report.contains("TypeMismatch") || report.contains("ArityMismatch"),
            "{name} check must diagnose the pattern: {report}"
        );

        let evaluated = run(&["eval", "--file", file]);
        assert!(
            !evaluated.status.success(),
            "{name} eval must refuse the invalid program: {evaluated:?}"
        );

        let output = dir.path().join("generated");
        let built = run(&[
            "build",
            file,
            "--target",
            "c",
            "--output",
            output.to_str().expect("UTF-8 path"),
        ]);
        assert!(
            !built.status.success(),
            "{name} build must refuse the invalid program: {built:?}"
        );
    }
}

#[test]
fn valid_tuple_pattern_checks_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("valid.ch");
    fs::write(
        &path,
        "def f(t: (i32, i32)) -> i32 = match t with { | (a, b) => add(a, b) }\n\
         out = f((2i32, 3i32))\n",
    )
    .expect("write fixture");
    let checked = run(&["check", path.to_str().expect("UTF-8 path")]);
    let report = String::from_utf8_lossy(&checked.stdout);
    assert!(report.contains("\"errors\": []"), "valid pattern: {report}");
}
