//! An ordinary top-level `def` or `sig` named like a loaded standard prelude
//! macro is rejected by `check`, `eval`, and `build` alike, in a reef package
//! module as in a standalone file, and the diagnostic names the declaration
//! as written (spec/02-surf-syntax.md §P5b, chelis#3270).

mod common;

use assert_cmd::Command;
use common::{build_and_run_app, gcc_available, make_app, write_file};
use std::path::Path;

const PRELUDE_MACROS: [&str; 3] = ["residual", "linear_layer", "cross_entropy"];

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

fn collision(declaration: &str, name: &str) -> String {
    format!(
        "`{declaration} {name}` collides with the standard prelude macro `{name}`: ordinary \
         top-level `def`/`sig` declarations may not reuse a loaded standard-prelude macro name \
         (spec/02-surf-syntax.md §P5b)"
    )
}

/// `check` reports exactly the collision with a zero score and a failing
/// exit; `eval` and C `build` fail with it before producing any value or
/// artifact. No command names the declaration by a linker-internal name.
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
    let messages = report["errors"]
        .as_array()
        .expect("errors list")
        .iter()
        .map(|error| error["message"].as_str().expect("message").to_string())
        .collect::<Vec<_>>();
    assert!(
        checked.status.code() == Some(2)
            && report["score"].as_f64() == Some(0.0)
            && messages.len() == 1
            && messages[0].contains(diagnostic),
        "check must report only `{diagnostic}`: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&checked.stderr)
    );

    let evaluated = run(root, reef_home, &["eval", "--file", source]);
    let eval_stdout = String::from_utf8_lossy(&evaluated.stdout);
    let eval_stderr = String::from_utf8_lossy(&evaluated.stderr);
    assert!(
        !evaluated.status.success() && eval_stderr.contains(diagnostic) && eval_stdout.is_empty(),
        "eval must fail with `{diagnostic}` before running anything: stdout {eval_stdout}\n\
         stderr: {eval_stderr}"
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
    let build_stderr = String::from_utf8_lossy(&built.stderr);
    assert!(
        !built.status.success() && build_stderr.contains(diagnostic) && !output.exists(),
        "C build must fail with `{diagnostic}` and write nothing: {build_stderr}"
    );

    for text in [&*stdout, &*eval_stderr, &*build_stderr] {
        assert!(
            !text.contains("pkg__"),
            "the diagnostic must name the authored declaration, not a linked name: {text}"
        );
    }
}

/// A `Demo` package app whose `Demo.Main` module is `body`.
fn package_main(
    dir_name: &str,
    body: &str,
) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let (dir, reef_home, app) = make_app(dir_name);
    write_file(
        &app.join("src/main.ch"),
        &format!("module Demo.Main\n{body}"),
    );
    (dir, reef_home, app)
}

/// Before the fix, `check` scored the package 1 with no errors, and `build`
/// compiled it to a binary that printed `out = 1`, calling the `def` the
/// macro should have shadowed; only `eval` rejected it.
#[test]
fn package_def_named_like_each_prelude_macro_is_rejected_by_every_command() {
    for name in PRELUDE_MACROS {
        let (_dir, reef_home, app) = package_main(
            "prelude_def",
            &format!("def {name}(x: i32) -> i32 = x\nout = {name}(1i32)\n"),
        );
        assert_rejected_in_each_command(
            &app,
            Some(&reef_home),
            &app.join("src/main.ch"),
            &collision("def", name),
        );
    }
}

/// A `sig` naming a prelude macro, paired with its definition, is rejected
/// under the `def` it annotates, as it is in a standalone file.
#[test]
fn package_sig_named_like_a_prelude_macro_is_rejected_by_every_command() {
    let (_dir, reef_home, app) = package_main(
        "prelude_sig",
        "sig cross_entropy: i32 -> i32\ncross_entropy = fn (x: i32) -> x\nout = cross_entropy(1i32)\n",
    );
    assert_rejected_in_each_command(
        &app,
        Some(&reef_home),
        &app.join("src/main.ch"),
        &collision("def", "cross_entropy"),
    );
}

/// The rule holds in every module of the package, not only the one a
/// command is pointed at: an imported colliding definition is rejected too.
#[test]
fn imported_package_def_named_like_a_prelude_macro_is_rejected_by_every_command() {
    let (_dir, reef_home, app) = package_main(
        "prelude_import",
        "import Demo.Layers (linear_layer)\nout = linear_layer(1i32)\n",
    );
    write_file(
        &app.join("src/layers.ch"),
        "module Demo.Layers\nexport (linear_layer)\ndef linear_layer(x: i32) -> i32 = x\n",
    );
    assert_rejected_in_each_command(
        &app,
        Some(&reef_home),
        &app.join("src/main.ch"),
        &collision("def", "linear_layer"),
    );
}

#[test]
fn standalone_def_named_like_a_prelude_macro_stays_rejected_by_every_command() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("main.ch");
    write_file(
        &source,
        "module Demo.Main\ndef residual(x: i32) -> i32 = x\nout = residual(1i32)\n",
    );
    assert_rejected_in_each_command(dir.path(), None, &source, &collision("def", "residual"));
}

/// `check` scores the package app 1 with no errors, `eval` prints `expected`,
/// and the C build's executable prints it too.
fn assert_package_accepted(app: &Path, reef_home: &Path, expected: &str) {
    let source = app.join("src/main.ch");
    let source = source.to_str().expect("UTF-8 path");
    let checked = run(app, Some(reef_home), &["check", source]);
    let report: serde_json::Value =
        serde_json::from_slice(&checked.stdout).expect("check emits JSON");
    assert!(
        checked.status.success()
            && report["score"].as_f64() == Some(1.0)
            && report["errors"].as_array().is_some_and(Vec::is_empty),
        "the package must check clean: {report}"
    );

    let evaluated = run(app, Some(reef_home), &["eval", "--file", source]);
    assert!(
        evaluated.status.success(),
        "the package must evaluate: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&evaluated.stdout).trim(), expected);

    if gcc_available() {
        assert_eq!(build_and_run_app(reef_home, app, "main").trim(), expected);
    }
}

#[test]
fn package_def_with_a_distinct_name_is_accepted() {
    let (_dir, reef_home, app) = package_main(
        "prelude_distinct",
        "def residual_step(x: i32) -> i32 = x\nout = residual_step(1i32)\n",
    );
    assert_package_accepted(&app, &reef_home, "out = 1");
}

/// A user macro takes precedence over the standard prelude macro of the same
/// name, so `residual(5, 2)` is `sub(5, 2)`, not the prelude's `add(x, f(x))`.
#[test]
fn package_user_macro_keeps_precedence_over_the_prelude_macro() {
    let (_dir, reef_home, app) = package_main(
        "prelude_override",
        "macro residual(x, y) = sub(x, y)\nout = residual(5i32, 2i32)\n",
    );
    assert_package_accepted(&app, &reef_home, "out = 3");
}
