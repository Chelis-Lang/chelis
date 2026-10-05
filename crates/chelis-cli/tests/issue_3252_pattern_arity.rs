//! The CLI rejects a malformed constructor pattern before either execution lane runs.

mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use std::path::Path;

fn run(root: &Path, reef_home: Option<&Path>, args: &[&str]) -> std::process::Output {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root);
    if let Some(home) = reef_home {
        command.env("CHELIS_REEF_HOME", home);
    }
    command.args(args).output().expect("CLI runs")
}

fn assert_rejected_in_each_command(
    root: &Path,
    reef_home: Option<&Path>,
    source: &Path,
    ctor: &str,
) {
    let source = source.to_str().expect("UTF-8 path");
    let checked = run(root, reef_home, &["check", source]);
    let report = String::from_utf8_lossy(&checked.stdout);
    assert!(
        report.contains("ArityMismatch") && report.contains(ctor),
        "check must diagnose the constructor: {report}\nstderr: {}",
        String::from_utf8_lossy(&checked.stderr)
    );

    let evaluated = run(root, reef_home, &["eval", "--file", source]);
    assert!(!evaluated.status.success(), "eval must reject the pattern");
    assert!(
        String::from_utf8_lossy(&evaluated.stderr).contains("constructor pattern")
            && String::from_utf8_lossy(&evaluated.stderr).contains(ctor),
        "eval diagnostic: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );

    let output = root.join("generated");
    let built = run(
        root,
        reef_home,
        &[
            "build",
            source,
            "--target",
            "c",
            "--output",
            output.to_str().unwrap(),
        ],
    );
    assert!(!built.status.success(), "C build must reject the pattern");
    assert!(
        String::from_utf8_lossy(&built.stderr).contains("constructor pattern")
            && String::from_utf8_lossy(&built.stderr).contains(ctor),
        "build diagnostic: {}",
        String::from_utf8_lossy(&built.stderr)
    );
}

#[test]
fn local_constructor_under_wildcard_rejects_in_check_eval_and_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("local.ch");
    write_file(
        &source,
        "type T = | A(i32, i32) | B\n\
         def f(t: T) -> i32 = match t with { | A(x) => x | _ => 0 }\n\
         out = f(A(7i32, 8i32))\n",
    );
    assert_rejected_in_each_command(dir.path(), None, &source, "A");
}

#[test]
fn imported_constructor_under_wildcard_rejects_in_check_eval_and_build() {
    let (_dir, reef_home, app) = make_app("issue-3252-imported");
    let source = app.join("src/main.ch");
    write_file(
        &source,
        "module Demo.Main\n\
         import Std.Io.Json (Json, JsonFloat)\n\
         def f(value: Json) -> string = match value with {\n\
           | JsonFloat(n) => to_string(n)\n\
           | _ => \"other\"\n\
         }\n\
         out = f(JsonFloat(1.0f64, \"1.0\"))\n",
    );
    assert_rejected_in_each_command(&app, Some(&reef_home), &source, "JsonFloat");
}
