//! A macro call with the wrong argument count, or a macro definition that
//! repeats a parameter, is rejected by `check`, `eval`, and `build` before
//! any lane runs (spec/02-surf-syntax.md [02-MACRO-1], [02-MACRO-2]).

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

/// `check` reports the diagnostic in its JSON errors list with an imperfect
/// score; `eval` and C `build` fail with it before producing any value,
/// effect, or artifact.
fn assert_rejected_in_each_command(
    root: &Path,
    reef_home: Option<&Path>,
    source: &Path,
    diagnostic: &str,
) {
    let source = source.to_str().expect("UTF-8 path");
    let checked = run(root, reef_home, &["check", source]);
    let stdout = String::from_utf8_lossy(&checked.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("check must emit JSON ({error}): {stdout}"));
    let errors = report["errors"].as_array().expect("errors list");
    assert!(
        !checked.status.success()
            && report["score"].as_f64().expect("score") < 1.0
            && errors.iter().any(|error| error["message"]
                .as_str()
                .is_some_and(|message| message.contains(diagnostic))),
        "check must report `{diagnostic}`: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&checked.stderr)
    );

    let evaluated = run(root, reef_home, &["eval", "--file", source]);
    let eval_stdout = String::from_utf8_lossy(&evaluated.stdout);
    assert!(
        !evaluated.status.success()
            && String::from_utf8_lossy(&evaluated.stderr).contains(diagnostic)
            && !eval_stdout.contains("out =")
            && !eval_stdout.contains("boom"),
        "eval must fail with `{diagnostic}` before running anything: stdout {eval_stdout}\n\
         stderr: {}",
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
            output.to_str().expect("UTF-8 path"),
        ],
    );
    assert!(
        !built.status.success() && String::from_utf8_lossy(&built.stderr).contains(diagnostic),
        "C build must fail with `{diagnostic}`: {}",
        String::from_utf8_lossy(&built.stderr)
    );
}

fn assert_file_rejected(name: &str, program: &str, diagnostic: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join(name);
    write_file(&source, program);
    assert_rejected_in_each_command(dir.path(), None, &source, diagnostic);
}

/// Without the rule, the missing `y` read `f`'s own parameter and `out`
/// evaluated to 9.
#[test]
fn short_user_macro_call_is_rejected() {
    assert_file_rejected(
        "short.ch",
        "macro second(x, y) = y\n\
         def f(y: i32) -> i32 = second(1i32)\n\
         out = f(9i32)\n",
        "macro `second` expects 2 argument(s), but this call supplies 1",
    );
}

/// Without the rule, `residual`'s missing `f` called the caller's `f`, and
/// `out` evaluated to 11.
#[test]
fn short_prelude_macro_call_is_rejected() {
    assert_file_rejected(
        "prelude.ch",
        "def f(x: i32) -> i32 = mul(x, 10i32)\n\
         out = residual(1i32)\n",
        "macro `residual` expects 2 argument(s), but this call supplies 1",
    );
}

/// Without the rule, each surplus argument was dropped unchecked: an
/// undefined name, an effect that never ran, and a second consuming use of
/// an affine key, which a function call rejects under [04-LIN-9].
#[test]
fn long_user_macro_calls_are_rejected() {
    for (name, program) in [
        (
            "undefined.ch",
            "macro keep(x) = x\n\
             out = keep(7i32, never_declared)\n",
        ),
        (
            "effect.ch",
            "macro keep(x) = x\n\
             out = keep(7i32, print(\"boom\"))\n",
        ),
        (
            "key.ch",
            "macro keep(x) = x\n\
             def f(k: key) -> key = keep(k, k)\n\
             out = f(key_from_seed(42i64))\n",
        ),
        (
            "tensor.ch",
            "macro keep(x) = x\n\
             def f(t: tensor[2, f32]) -> i32 = keep(7i32, t)\n",
        ),
    ] {
        assert_file_rejected(
            name,
            program,
            "macro `keep` expects 1 argument(s), but this call supplies 2",
        );
    }
}

/// Without the rule, the later `x` won and `out` evaluated to 2.
#[test]
fn repeated_macro_parameter_is_rejected() {
    assert_file_rejected(
        "duplicate.ch",
        "macro dup(x, x) = x\n\
         out = dup(1i32, 2i32)\n",
        "macro `dup` declares parameter `x` more than once",
    );
}

/// Inside a package the linker qualifies the macro's name before expansion,
/// so the diagnostic names `..__Demo__Main__keep`; the rule is unchanged.
#[test]
fn long_macro_call_in_a_package_module_is_rejected() {
    let (_dir, reef_home, app) = make_app("issue-3239-package");
    let source = app.join("src/main.ch");
    write_file(
        &source,
        "module Demo.Main\n\
         macro keep(x) = x\n\
         out = keep(7i32, never_declared)\n",
    );
    assert_rejected_in_each_command(
        &app,
        Some(&reef_home),
        &source,
        "Demo__Main__keep` expects 1 argument(s), but this call supplies 2",
    );
}

#[test]
fn exact_arity_macro_calls_check_clean_and_evaluate() {
    for (name, program, value) in [
        (
            "user.ch",
            "macro second(x, y) = y\n\
             def f(y: i32) -> i32 = second(1i32, 9i32)\n\
             out = f(5i32)\n",
            "out = 9",
        ),
        (
            "prelude.ch",
            "def g(x: i32) -> i32 = mul(x, 10i32)\n\
             out = residual(1i32, g)\n",
            "out = 11",
        ),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join(name);
        write_file(&source, program);
        let source = source.to_str().expect("UTF-8 path");

        let checked = run(dir.path(), None, &["check", source]);
        let report: serde_json::Value =
            serde_json::from_slice(&checked.stdout).expect("check emits JSON");
        assert!(
            checked.status.success()
                && report["score"].as_f64() == Some(1.0)
                && report["errors"].as_array().is_some_and(Vec::is_empty),
            "an exact-arity call must check clean: {report}"
        );

        let evaluated = run(dir.path(), None, &["eval", "--file", source]);
        assert!(
            evaluated.status.success(),
            "eval must succeed: {}",
            String::from_utf8_lossy(&evaluated.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&evaluated.stdout).trim(),
            value,
            "{name} must evaluate to the argument the macro selects"
        );
    }
}
