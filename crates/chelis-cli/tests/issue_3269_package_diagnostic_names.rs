//! chelis#3269: a diagnostic about a reef package names its declarations as
//! the author wrote them.
//!
//! The package linker rewrites every top-level declaration of a package
//! module to a private `pkg__<package>__<Module>__<name>` spelling before the
//! front end runs. A checker message that interpolates a declaration name
//! used to print that spelling (`def 'pkg__demo__app__Demo__Main__out' ...`)
//! in `check`, `eval` and `build`, while the same source as a standalone file
//! printed `def 'out'`. Apart from location, a diagnostic reads the same in a
//! package module and in a standalone file, and an authored name that itself
//! contains `__` is shown whole rather than cut at its last `__`.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use std::process::Output;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

const TYPE_ERROR_SOURCE: &str = "module Demo.Main\nout: i32 = true\n";
const TYPE_ERROR_MESSAGE: &str =
    "def 'out' body doesn't match declared signature: expected `i32`, got `bool`";
const DUPLICATE_SOURCE: &str = "module Demo.Main\nout: i32 = 1i32\nout: i32 = 2i32\n";

fn chelis(dir: &Path, reef_home: &Path, args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(dir)
        .args(args)
        .output()
        .expect("run chelis")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The `errors` array of a `chelis check` report, after asserting the
/// report's exit status (2: errors present).
fn check_errors(output: &Output) -> Vec<Value> {
    let stdout = text(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(2),
        "check must reject; stdout={stdout}\nstderr={}",
        text(&output.stderr)
    );
    let report: Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("check report is JSON: {error}\n{stdout}"));
    report["errors"].as_array().expect("errors array").clone()
}

/// The human-readable fields of each diagnostic, in report order. Location
/// fields are excluded: a package module and a standalone file are allowed
/// to differ only there.
fn readable(errors: &[Value]) -> Vec<Value> {
    errors
        .iter()
        .map(|error| {
            serde_json::json!({
                "kind": error["kind"],
                "message": error["message"],
                "expected": error["expected"],
                "got": error["got"],
                "suggestions": error["suggestions"],
            })
        })
        .collect()
}

fn assert_no_linker_spelling(surface: &str, output: &str) {
    assert!(
        !output.contains("pkg__") && !output.contains("Pkg__"),
        "{surface} must not show a linker-private name:\n{output}"
    );
}

/// A package module `src/main.ch` holding `source`, inside package
/// `demo-app` with `module_prefix = "Demo"`.
fn package_with_main(source: &str) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let (dir, reef_home, app) = make_app("demo_app");
    write_file(&app.join("src/main.ch"), source);
    (dir, reef_home, app)
}

/// The same `source` as a standalone file outside every package.
fn standalone_errors(source: &str, reef_home: &Path) -> Vec<Value> {
    let dir = tempfile::tempdir().expect("tempdir");
    write_file(&dir.path().join("main.ch"), source);
    check_errors(&chelis(dir.path(), reef_home, &["check", "main.ch"]))
}

#[test]
fn package_type_error_names_the_authored_declaration_in_check_eval_and_build() {
    let (_dir, reef_home, app) = package_with_main(TYPE_ERROR_SOURCE);

    let check = chelis(&app, &reef_home, &["check", "src/main.ch"]);
    let errors = check_errors(&check);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0]["message"], TYPE_ERROR_MESSAGE, "{errors:?}");
    assert_no_linker_spelling("check stdout", &text(&check.stdout));
    assert_no_linker_spelling("check stderr", &text(&check.stderr));

    let eval = chelis(&app, &reef_home, &["eval", "--file", "src/main.ch"]);
    let eval_stderr = text(&eval.stderr);
    assert!(!eval.status.success(), "eval must reject: {eval_stderr}");
    assert!(
        eval_stderr.contains(&format!("error: {TYPE_ERROR_MESSAGE}")),
        "eval names the authored declaration:\n{eval_stderr}"
    );
    assert_no_linker_spelling("eval stderr", &eval_stderr);

    let out = app.join("type_error-out");
    let build = chelis(
        &app,
        &reef_home,
        &[
            "build",
            "src/main.ch",
            "--target",
            "c",
            "--output",
            out.to_str().expect("utf-8 path"),
        ],
    );
    let build_stderr = text(&build.stderr);
    assert!(!build.status.success(), "build must reject: {build_stderr}");
    assert!(
        build_stderr.contains(&format!("TypeMismatch: {TYPE_ERROR_MESSAGE}")),
        "build names the authored declaration:\n{build_stderr}"
    );
    assert_no_linker_spelling("build stderr", &build_stderr);

    // Control: the byte-identical source as a standalone file reads the same.
    assert_eq!(
        readable(&standalone_errors(TYPE_ERROR_SOURCE, &reef_home)),
        readable(&errors),
        "a package module and a standalone file report the same diagnostic"
    );
}

#[test]
fn package_duplicate_definition_names_the_authored_declaration() {
    let (_dir, reef_home, app) = package_with_main(DUPLICATE_SOURCE);

    let check = chelis(&app, &reef_home, &["check", "src/main.ch"]);
    let errors = check_errors(&check);
    assert_no_linker_spelling("check stdout", &text(&check.stdout));
    let duplicate = errors
        .iter()
        .find(|error| error["kind"] == "DuplicateDefinition")
        .unwrap_or_else(|| panic!("a duplicate-definition diagnostic: {errors:?}"));
    assert_eq!(
        duplicate["message"],
        "duplicate definition: `out` is defined more than once"
    );
    let suggestions = duplicate["suggestions"].as_array().expect("suggestions");
    assert!(
        !suggestions.is_empty()
            && suggestions
                .iter()
                .all(|suggestion| suggestion.as_str().is_some_and(|s| s.contains("`out`"))),
        "repair hints name the authored declaration too: {suggestions:?}"
    );

    // Control: the standalone file reports the same diagnostics.
    assert_eq!(
        readable(&standalone_errors(DUPLICATE_SOURCE, &reef_home)),
        readable(&errors)
    );
}

#[test]
fn authored_names_containing_double_underscores_are_shown_whole() {
    // `My__Shape` and `my__helper` are authored spellings. Cutting a linked
    // name at its last `__` would print `Shape` and `helper`.
    let (_dir, reef_home, app) = package_with_main(
        "module Demo.Main\n\
         import Demo.Shapes (make__circle)\n\
         def my__helper(x: i32) -> i32 = true\n\
         out: i32 = make__circle(1.0f32)\n",
    );
    write_file(
        &app.join("src/shapes.ch"),
        "module Demo.Shapes\n\
         export (make__circle, My__Shape)\n\
         type My__Shape =\n  | Circle(f32)\n\
         def make__circle(r: f32) -> My__Shape = Circle(r)\n",
    );

    let check = chelis(&app, &reef_home, &["check", "src/main.ch"]);
    let errors = check_errors(&check);
    assert_no_linker_spelling("check stdout", &text(&check.stdout));
    let messages = errors
        .iter()
        .map(|error| error["message"].as_str().expect("message"))
        .collect::<Vec<_>>();
    assert_eq!(
        messages,
        [
            "def 'my__helper' body doesn't match declared signature: \
             expected `(i32) -> i32`, got `(i32) -> bool`",
            "def 'out' body doesn't match declared signature: \
             expected `i32`, got `My__Shape`",
        ],
        "{errors:?}"
    );
    // The type rendering fields carry the authored type name as well.
    assert_eq!(errors[1]["got"], "My__Shape", "{errors:?}");

    let eval = chelis(&app, &reef_home, &["eval", "--file", "src/main.ch"]);
    let eval_stderr = text(&eval.stderr);
    assert!(
        eval_stderr.contains("def 'my__helper'") && eval_stderr.contains("got `My__Shape`"),
        "eval shows the authored names whole:\n{eval_stderr}"
    );
    assert_no_linker_spelling("eval stderr", &eval_stderr);
}
