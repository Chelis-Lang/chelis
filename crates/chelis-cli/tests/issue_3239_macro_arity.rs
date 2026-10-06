//! A macro call with the wrong argument count or a named argument, or a macro
//! definition that repeats a parameter, is rejected by `check`, `eval`, and
//! `build` before any lane runs (spec/02-surf-syntax.md [02-MACRO-1],
//! [02-MACRO-2], [02-MACRO-3]).

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

/// `16777216 + 1 + 1` sums to `16777216` in `f32` and to `16777218` in an
/// `f64` accumulator.
const SUM_OPERAND: &str = "floats: tensor[3, f32] = to_tensor([16777216.0f32, 1.0f32, 1.0f32])\n";
const MATMUL_OPERANDS: &str = "a: tensor[1, 3, f32] = to_tensor([[16777216.0f32, 1.0f32, 1.0f32]])\n\
     b: tensor[3, 1, f32] = to_tensor([[1.0f32], [1.0f32], [1.0f32]])\n";

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

/// Without the rule, the missing `y` read `f`'s own parameter.
#[test]
fn short_user_macro_call_is_rejected() {
    assert_file_rejected(
        "short.ch",
        "macro minus(x, y) = sub(x, y)\n\
         def f(y: i32) -> i32 = minus(1i32)\n\
         out = f(9i32)\n",
        "macro `minus` expects 2 argument(s), but this call supplies 1",
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

/// Without the rule, the named argument was dropped with the replaced call:
/// `total(floats, accumulator=f64)` evaluated to 16777216.0, and the
/// compiled C executable of the `matmul` variant printed the same value.
#[test]
fn named_argument_on_a_macro_call_is_rejected() {
    for (name, program, callee) in [
        (
            "sum.ch",
            format!(
                "macro total(x) = sum(x, 0i32)\n{SUM_OPERAND}out = total(floats, accumulator=f64)\n"
            ),
            "total",
        ),
        (
            "matmul.ch",
            format!(
                "macro mm(a, b) = matmul(a, b)\n{MATMUL_OPERANDS}out = mm(a, b, accumulator=f64)\n"
            ),
            "mm",
        ),
        (
            "prelude.ch",
            "out = residual(1i32, fn (a: i32) -> a, accumulator=f64)\n".to_string(),
            "residual",
        ),
        (
            "pipe.ch",
            "macro keep(x) = x\nout = 7i32 |> keep(accumulator=f64)\n".to_string(),
            "keep",
        ),
    ] {
        assert_file_rejected(
            name,
            &program,
            &format!(
                "macro `{callee}` takes only positional arguments, but this call passes the \
                 named argument `accumulator=`"
            ),
        );
    }
}

/// A named argument written inside a macro body belongs to the built-in call
/// there, so the `f64` accumulator survives expansion.
#[test]
fn named_argument_inside_a_macro_body_is_kept() {
    let sum = format!(
        "macro total(x) = sum(x, 0i32, accumulator=f64)\n{SUM_OPERAND}out = total(floats)\n"
    );
    let matmul = format!(
        "macro mm(a, b) = matmul(a, b, accumulator=f64)\n{MATMUL_OPERANDS}out = mm(a, b)\n"
    );
    for (name, program, value) in [
        ("sum_body.ch", &sum, "out = 16777218.0"),
        (
            "matmul_body.ch",
            &matmul,
            "out = tensor(shape=[1, 1], data=[16777218.0])",
        ),
    ] {
        let stdout = checked_clean_and_evaluated(name, program);
        assert!(
            stdout.lines().any(|line| line == value),
            "{name} must evaluate with the f64 accumulator: {stdout}"
        );
    }
}

#[test]
fn named_argument_inside_a_macro_body_is_kept_in_compiled_c() {
    if !gcc_available() {
        eprintln!("skipping: c compiler not available");
        return;
    }
    let stdout = build_and_run(
        &format!(
            "macro mm(a, b) = matmul(a, b, accumulator=f64)\n{MATMUL_OPERANDS}out = mm(a, b)\n"
        ),
        "matmul_body",
    );
    assert!(
        stdout
            .lines()
            .any(|line| line == "out = tensor(shape=[1, 1], data=[16777218.0])"),
        "the compiled executable must accumulate in f64: {stdout}"
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
            "macro minus(x, y) = sub(x, y)\n\
             def f(y: i32) -> i32 = minus(9i32, y)\n\
             out = f(1i32)\n",
            "out = 8",
        ),
        (
            "prelude.ch",
            "def g(x: i32) -> i32 = mul(x, 10i32)\n\
             out = residual(1i32, g)\n",
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
