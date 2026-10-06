//! A macro definition whose body never references one of its parameters is
//! rejected by `check`, `eval`, and `build` before any lane runs, so no call
//! argument is discarded without name, type, effect, or linearity checking
//! (spec/02-surf-syntax.md [02-MACRO-4]).

mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, make_app, write_file};
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
            && eval_stdout.trim().is_empty(),
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

const UNUSED_B: &str = "macro `first` never references its parameter `b`";

/// Without the rule, each argument for `b` was dropped unchecked: `check`
/// scored 1 with no errors, `eval` and the compiled C executable printed
/// `out = 1`, and neither `boom` nor `2` was ever printed. The same arguments
/// passed to a `def` are an unbound name, a type error, a type mismatch, and
/// an effect that runs.
#[test]
fn an_argument_cannot_hide_in_an_unused_parameter() {
    for (name, discarded) in [
        ("unbound.ch", "never_declared"),
        ("ill_typed.ch", "add(1i32, true)"),
        ("print.ch", "print(\"boom\")"),
        ("debug.ch", "debug(2i32)"),
    ] {
        assert_file_rejected(
            name,
            &format!("macro first(a, b) = a\nout = first(1i32, {discarded})\n"),
            UNUSED_B,
        );
    }
}

/// Without the rule, the second consuming use of `k`, which a function call
/// rejects under [04-LIN-9], was dropped before linearity checking.
#[test]
fn a_reused_key_cannot_hide_in_an_unused_parameter() {
    assert_file_rejected(
        "key.ch",
        "macro first(a, b) = a\n\
         def f(k: key) -> key = first(k, k)\n\
         out = f(key_from_seed(1i64))\n",
        UNUSED_B,
    );
}

/// Without the rule, the wrong-arity call to `keep` inside the dropped
/// argument was never expanded, so [02-MACRO-2] never saw it.
#[test]
fn a_wrong_arity_call_cannot_hide_in_an_unused_parameter() {
    assert_file_rejected(
        "hidden_arity.ch",
        "macro first(a, b) = a\n\
         macro keep(x) = x\n\
         out = first(1i32, keep(1i32, 2i32))\n",
        UNUSED_B,
    );
}

/// With every parameter referenced, the argument is expanded where the body
/// places it, and its wrong-arity call is rejected.
#[test]
fn a_wrong_arity_call_in_a_used_argument_is_rejected() {
    assert_file_rejected(
        "used_arity.ch",
        "macro both(a, b) = add(a, b)\n\
         macro keep(x) = x\n\
         out = both(1i32, keep(1i32, 2i32))\n",
        "macro `keep` expects 1 argument(s), but this call supplies 2",
    );
}

#[test]
fn an_uncalled_definition_with_an_unused_parameter_is_rejected() {
    assert_file_rejected(
        "uncalled.ch",
        "macro first(a, b) = a\nout = 1i32\n",
        UNUSED_B,
    );
}

/// Inside a package the linker qualifies the macro's name before expansion.
#[test]
fn an_unused_parameter_in_a_package_module_is_rejected() {
    let (_dir, reef_home, app) = make_app("issue-3266-package");
    let source = app.join("src/main.ch");
    write_file(
        &source,
        "module Demo.Main\n\
         macro first(a, b) = a\n\
         out = first(1i32, never_declared)\n",
    );
    assert_rejected_in_each_command(
        &app,
        Some(&reef_home),
        &source,
        "Demo__Main__first` never references its parameter `b`",
    );
}

/// `check` reports a perfect score with no errors, and `eval` succeeds; the
/// evaluator's output.
fn checked_clean_and_evaluated(name: &str, program: &str) -> String {
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
        "{name} must check clean: {report}"
    );

    let evaluated = run(dir.path(), None, &["eval", "--file", source]);
    assert!(
        evaluated.status.success(),
        "{name} must evaluate: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    String::from_utf8(evaluated.stdout).expect("UTF-8 stdout")
}

const BOTH: &str = "macro both(a, b) = add(a, b)\nout = both(1i32, 2i32)\n";

#[test]
fn macros_that_reference_every_parameter_check_clean_and_evaluate() {
    for (name, program, value) in [
        ("both.ch", BOTH, "out = 3"),
        (
            "twice.ch",
            "macro double(x) = add(x, x)\nout = double(3i32)\n",
            "out = 6",
        ),
        (
            "lambda.ch",
            "macro bump(v) = (fn (y: i32) -> add(y, v))(1i32)\nout = bump(4i32)\n",
            "out = 5",
        ),
        (
            "prelude.ch",
            "def g(x: i32) -> i32 = mul(x, 10i32)\nout = residual(1i32, g)\n",
            "out = 11",
        ),
    ] {
        assert_eq!(
            checked_clean_and_evaluated(name, program).trim(),
            value,
            "{name} must evaluate to the value the macro expands to"
        );
    }
}

/// A macro parameter shadows an imported name of the same spelling inside
/// the body, as it does without the import. Without that, the entry-module
/// linker bound the body reference to the import: `apply(neg1, 5i32)` called
/// `Std.Scalar.abs` and dropped `neg1`, and the body was then reported as
/// never referencing its parameter.
#[test]
fn a_macro_parameter_shadows_an_imported_name() {
    for (name, program, value) in [
        (
            "shadow_abs.ch",
            "import Std.Scalar (abs)\n\
             macro apply(abs, x) = abs(x)\n\
             def neg1(x: i32) -> i32 = neg(x)\n\
             out = apply(neg1, 5i32)\n",
            "out = -5",
        ),
        (
            "control_abs.ch",
            "macro apply(abs, x) = abs(x)\n\
             def neg1(x: i32) -> i32 = neg(x)\n\
             out = apply(neg1, 5i32)\n",
            "out = -5",
        ),
        (
            "shadow_join.ch",
            "import Std.Text (join)\n\
             macro twice(join) = add(join, join)\n\
             out = twice(3i32)\n",
            "out = 6",
        ),
        (
            "control_join.ch",
            "macro twice(join) = add(join, join)\nout = twice(3i32)\n",
            "out = 6",
        ),
    ] {
        assert_eq!(
            checked_clean_and_evaluated(name, program).trim(),
            value,
            "{name} must substitute the argument for the parameter, not the import"
        );
    }
}

#[test]
fn a_macro_that_references_every_parameter_runs_in_compiled_c() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler");
        return;
    }
    assert_eq!(build_and_run(BOTH, "both").trim(), "out = 3");
}
