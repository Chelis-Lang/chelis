//! chelis#3269: a diagnostic about a reef package names its declarations as
//! the author wrote them.
//!
//! The package linker rewrites every top-level declaration of a package
//! module to a private `pkg__<package>__<Module>__<name>` spelling before the
//! front end runs. A checker message that interpolates a declaration name
//! used to print that spelling (`def 'pkg__demo__app__Demo__Main__out' ...`)
//! in `check`, `eval` and `build`, while the same source as a standalone file
//! printed `def 'out'`. A declaration of the entry module is named bare,
//! so apart from location a diagnostic reads the same in a package module and
//! in a standalone file; a declaration of any other module is named qualified
//! by its module path (and by package when two packages share that path), so
//! no two declarations are named alike. An authored name that itself contains
//! `__` is shown whole rather than cut at its last `__`.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use std::process::Output;

#[path = "common/mod.rs"]
mod common;

use common::{COMPILER_VERSION, make_app, write_file};

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
             expected `i32`, got `Demo.Shapes.My__Shape`",
        ],
        "{errors:?}"
    );
    // The type rendering fields carry the authored type name as well.
    assert_eq!(errors[1]["got"], "Demo.Shapes.My__Shape", "{errors:?}");

    let eval = chelis(&app, &reef_home, &["eval", "--file", "src/main.ch"]);
    let eval_stderr = text(&eval.stderr);
    assert!(
        eval_stderr.contains("def 'my__helper'")
            && eval_stderr.contains("got `Demo.Shapes.My__Shape`"),
        "eval shows the authored names whole:\n{eval_stderr}"
    );
    assert_no_linker_spelling("eval stderr", &eval_stderr);
}

/// The `message` of every diagnostic of a `chelis check` of `src/main.ch`.
fn check_messages(app: &Path, reef_home: &Path) -> Vec<String> {
    let check = chelis(app, reef_home, &["check", "src/main.ch"]);
    assert_no_linker_spelling("check stdout", &text(&check.stdout));
    check_errors(&check)
        .iter()
        .map(|error| error["message"].as_str().expect("message").to_string())
        .collect()
}

/// Rewrite the package manifest of `app`, named `name`, to depend on the
/// package at `../lib` as `lib_name`.
fn depend_on_sibling(app: &Path, name: &str, lib_name: &str) {
    write_file(
        &app.join("reef.toml"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Demo\"\n\n\
             [dependencies]\nchelis-std = {{ version = \"0.4.0\" }}\n\
             {lib_name} = {{ path = \"../lib\" }}\n"
        ),
    );
}

/// A path-dependency package `lib_name` beside `app`, with `module_prefix`.
fn sibling_library(app: &Path, lib_name: &str, module_prefix: &str, modules: &[(&str, &str)]) {
    let lib = app.parent().expect("app has a parent").join("lib");
    write_file(
        &lib.join("reef.toml"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"{lib_name}\"\nversion = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"{module_prefix}\"\n\n\
             [dependencies]\nchelis-std = {{ version = \"0.4.0\" }}\n"
        ),
    );
    for (file, source) in modules {
        write_file(&lib.join("src").join(file), source);
    }
}

/// Red-team F1: equal names in two modules render differently. Only the
/// entry module's declarations are bare; another module's are qualified by
/// its module path, so `expected `T`, got `T`` cannot arise.
#[test]
fn equal_names_in_two_modules_are_told_apart_by_module() {
    let (_dir, reef_home, app) = package_with_main(
        "module Demo.Main\n\
         import Demo.A (T)\n\
         import Demo.B (make)\n\
         out: T = make(1i32)\n",
    );
    write_file(
        &app.join("src/a.ch"),
        "module Demo.A\nexport (T)\ntype T =\n  | MkA(i32)\n",
    );
    write_file(
        &app.join("src/b.ch"),
        "module Demo.B\nexport (make)\ntype T =\n  | MkB(i32)\n\
         def make(x: i32) -> T = MkB(x)\n",
    );
    let message = "def 'out' body doesn't match declared signature: \
                   expected `Demo.A.T`, got `Demo.B.T`";
    assert_eq!(check_messages(&app, &reef_home), [message]);
    let eval = chelis(&app, &reef_home, &["eval", "--file", "src/main.ch"]);
    assert!(
        text(&eval.stderr).contains(message),
        "eval: {}",
        text(&eval.stderr)
    );
}

/// Red-team F1 across packages: a dependency's declarations are qualified
/// by its module path, and an error inside the dependency names its module.
#[test]
fn a_dependency_declaration_is_named_by_its_module() {
    let (_dir, reef_home, app) = make_app("dep_app");
    depend_on_sibling(&app, "dep-app", "shapes-lib");
    sibling_library(
        &app,
        "shapes-lib",
        "Lib",
        &[(
            "shapes.ch",
            "module Lib.Shapes\nexport (make, T)\ntype T =\n  | MkL(i32)\n\
             def make(x: i32) -> T = MkL(x)\n\
             def twice(x: i32) -> bool = x\n",
        )],
    );
    write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\nimport Lib.Shapes (make)\ntype T =\n  | MkM(i32)\n\
         out: T = make(1i32)\n",
    );
    assert_eq!(
        check_messages(&app, &reef_home),
        [
            "def 'Lib.Shapes.twice' body doesn't match declared signature: \
             expected `(i32) -> bool`, got `(i32) -> i32`",
            "def 'out' body doesn't match declared signature: \
             expected `T`, got `Lib.Shapes.T`",
        ]
    );
}

/// Module paths can repeat across the packages of one graph: the linker
/// resolves an import to the importing package's own module first, and no
/// rule makes `module_prefix` unique. Two such declarations are qualified by
/// package as well.
#[test]
fn equal_module_paths_in_two_packages_are_told_apart_by_package() {
    let (_dir, reef_home, app) = make_app("col_app");
    depend_on_sibling(&app, "col-app", "other-lib");
    sibling_library(
        &app,
        "other-lib",
        "Demo",
        &[
            (
                "util.ch",
                "module Demo.Util\nexport (make, T)\ntype T =\n  | MkL(i32)\n\
                 def make(x: i32) -> T = MkL(x)\n",
            ),
            (
                "api.ch",
                "module Demo.Api\nimport Demo.Util (make)\nexport (make_t)\n\
                 def make_t(x: i32) -> Demo.Util.T = make(x)\n",
            ),
        ],
    );
    write_file(
        &app.join("src/util.ch"),
        "module Demo.Util\nexport (T, MkU)\ntype T =\n  | MkU(i32)\n",
    );
    write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\nimport Demo.Util (T)\nimport Demo.Api (make_t)\n\
         out: T = make_t(1i32)\n",
    );
    assert_eq!(
        check_messages(&app, &reef_home),
        ["def 'out' body doesn't match declared signature: \
          expected `col-app/Demo.Util.T`, got `other-lib/Demo.Util.T`"]
    );
}

/// Red-team F1 for a standalone file: its own `Date` is bare, the
/// standard library's is qualified.
#[test]
fn a_standalone_type_and_a_library_type_of_one_name_are_told_apart() {
    let reef_home = common::SHARED_REEF.reef_home.clone();
    let errors = standalone_errors(
        "import Std.Datetime (date_from_epoch_day)\n\
         type Date =\n  | MyDate(i32)\n\
         out: Date = date_from_epoch_day(1i64)\n",
        &reef_home,
    );
    assert_eq!(
        errors[0]["message"],
        "def 'out' body doesn't match declared signature: \
         expected `Date`, got `Std.Datetime.Date`",
        "{errors:?}"
    );
}

/// Red-team F2: an error in another module names that module, so it is not
/// read as one about the entry's declaration of the same name.
#[test]
fn an_error_in_another_module_names_its_module() {
    let (_dir, reef_home, app) = package_with_main(
        "module Demo.Main\n\
         import Demo.Util (go)\n\
         def helper(x: i32) -> i32 = go(x)\n\
         out = helper(1i32)\n",
    );
    write_file(
        &app.join("src/util.ch"),
        "module Demo.Util\nexport (go)\n\
         def helper(x: i32) -> bool = x\n\
         def go(x: i32) -> i32 = x\n",
    );
    let message = "def 'Demo.Util.helper' body doesn't match declared signature: \
                   expected `(i32) -> bool`, got `(i32) -> i32`";
    assert_eq!(check_messages(&app, &reef_home), [message]);
    let eval = chelis(&app, &reef_home, &["eval", "--file", "src/main.ch"]);
    assert!(
        text(&eval.stderr).contains(message),
        "eval: {}",
        text(&eval.stderr)
    );
}

/// Red-team F4: `unresolved_names` spells a linked name as the diagnostic
/// that reports it does.
#[test]
fn unresolved_names_are_spelled_as_their_diagnostic() {
    let (_dir, reef_home, app) =
        package_with_main("module Demo.Main\nimport Demo.Util (out)\nres = out\n");
    write_file(
        &app.join("src/util.ch"),
        "module Demo.Util\nexport (out, bad)\n\
         def bad(x: i32) -> bool = x\n\
         out = 1i32\n",
    );
    let check = chelis(&app, &reef_home, &["check", "src/main.ch"]);
    let stdout = text(&check.stdout);
    assert_no_linker_spelling("check stdout", &stdout);
    let report: Value = serde_json::from_str(&stdout).expect("check report is JSON");
    assert_eq!(
        report["unresolved_names"],
        serde_json::json!(["Demo.Util.out"]),
        "{stdout}"
    );
    assert!(
        report["errors"]
            .as_array()
            .expect("errors")
            .iter()
            .any(|error| error["message"] == "unbound variable: Demo.Util.out"),
        "{stdout}"
    );
}

/// Red-team F5: the linker encodes `My__Shape` in `Demo.Main` and `Shape`
/// in `Demo.Main.My` alike. No one authored name is that linked name, so
/// the diagnostic keeps it rather than claim the author wrote one twice.
#[test]
fn a_linked_name_of_two_declarations_is_not_attributed_to_either() {
    let (_dir, reef_home, app) =
        package_with_main("module Demo.Main\ntype My__Shape =\n  | A(i32)\nout = 1i32\n");
    write_file(
        &app.join("src/main/my.ch"),
        "module Demo.Main.My\ntype Shape =\n  | B(i32)\n",
    );
    let check = chelis(&app, &reef_home, &["check", "src/main.ch"]);
    let errors = check_errors(&check);
    let message = errors[0]["message"].as_str().expect("message");
    assert_eq!(
        message,
        "duplicate type definition: `Pkg__demo__app__Demo__Main__My__Shape` \
         was already declared as a deftype",
        "{errors:?}"
    );
}

/// Red-team F6: a failure while evaluating carries the program's own text,
/// which is never rewritten, even when it spells a linked name.
#[test]
fn an_evaluation_failure_keeps_the_program_text() {
    let reef_home = common::SHARED_REEF.reef_home.clone();
    let dir = tempfile::tempdir().expect("tempdir");
    write_file(
        &dir.path().join("main.ch"),
        "import Std.Datetime (date_from_epoch_day)\n\
         out: i32 = fail(\"pkg__chelis__std__Std__Datetime__date_from_epoch_day broke\")\n",
    );
    let eval = chelis(dir.path(), &reef_home, &["eval", "--file", "main.ch"]);
    let stderr = text(&eval.stderr);
    assert!(!eval.status.success(), "{stderr}");
    assert!(
        stderr.contains("error: pkg__chelis__std__Std__Datetime__date_from_epoch_day broke"),
        "the program's own text is kept:\n{stderr}"
    );
}
