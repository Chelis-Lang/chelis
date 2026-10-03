//! One name-resolution rule in every lane (spec/02 §P2).
//!
//! chelis#2885: a module that both imports a name into unqualified scope and
//! declares it is rejected by all three commands with one diagnostic naming
//! the import and the local declaration. The controls show that an import
//! alone, a local declaration alone, an import of a different name, a
//! qualified-only import, and a parameter reusing an imported name all still
//! compile and agree across the three lanes.

mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Output};

/// A `chelis` command run from `cwd` against the shared test reef.
fn chelis(reef_home: &Path, cwd: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .current_dir(cwd)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home);
    command
}

fn eval(reef_home: &Path, cwd: &Path, file: &Path) -> Output {
    chelis(reef_home, cwd)
        .arg("eval")
        .arg("--file")
        .arg(file)
        .output()
        .expect("run chelis eval")
}

fn check(reef_home: &Path, cwd: &Path, file: &Path) -> Output {
    chelis(reef_home, cwd)
        .arg("check")
        .arg(file)
        .output()
        .expect("run chelis check")
}

fn build(reef_home: &Path, cwd: &Path, file: &Path, out_dir: &Path) -> Output {
    chelis(reef_home, cwd)
        .arg("build")
        .arg(file)
        .args(["--target", "c", "--output"])
        .arg(out_dir)
        .output()
        .expect("run chelis build")
}

fn utf8(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("utf-8 output")
}

/// The one `error: ` line an `eval` or `build` rejection prints.
fn reported_error(lane: &str, output: &Output) -> String {
    assert!(
        !output.status.success(),
        "{lane} must reject the program\nstdout: {}\nstderr: {}",
        utf8(&output.stdout),
        utf8(&output.stderr)
    );
    let stderr = utf8(&output.stderr);
    let errors = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("error: "))
        .collect::<Vec<_>>();
    assert_eq!(errors.len(), 1, "{lane} must report one error:\n{stderr}");
    errors[0].to_string()
}

/// The one diagnostic a `check` rejection reports in its JSON document.
fn checked_error(output: &Output) -> String {
    assert_eq!(
        output.status.code(),
        Some(2),
        "check must reject the program\nstdout: {}\nstderr: {}",
        utf8(&output.stdout),
        utf8(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check prints one JSON document");
    let errors = report["errors"].as_array().expect("errors array");
    assert_eq!(errors.len(), 1, "check must report one error: {report}");
    errors[0]["message"]
        .as_str()
        .expect("error message")
        .to_string()
}

/// `eval`, `check`, and `build` all reject `file`, run from `cwd`, with the
/// same diagnostic, and that diagnostic is `expected`.
fn assert_rejected_identically(reef_home: &Path, cwd: &Path, file: &Path, expected: &str) {
    let out = tempfile::tempdir().expect("build output tempdir");
    let out_dir = out.path().join("out");
    let eval = reported_error("eval", &eval(reef_home, cwd, file));
    let check = checked_error(&check(reef_home, cwd, file));
    let build = reported_error("build", &build(reef_home, cwd, file, &out_dir));
    assert_eq!(eval, expected, "eval diagnostic");
    assert_eq!(check, expected, "check diagnostic");
    assert_eq!(build, expected, "build diagnostic");
    assert!(
        !out_dir.exists(),
        "a rejected build must publish no output under {}",
        out_dir.display()
    );
}

/// `eval` and the executable `build` produces both print `expected` for
/// `file`, run from `cwd`, and `check` reports no error.
fn assert_runs_identically(reef_home: &Path, cwd: &Path, file: &Path, expected: &str) {
    let evaluated = eval(reef_home, cwd, file);
    assert!(
        evaluated.status.success(),
        "eval must accept {}\nstderr: {}",
        file.display(),
        utf8(&evaluated.stderr)
    );
    assert_eq!(utf8(&evaluated.stdout), expected, "eval output");

    let checked = check(reef_home, cwd, file);
    assert!(
        checked.status.success(),
        "check must accept {}\nstdout: {}\nstderr: {}",
        file.display(),
        utf8(&checked.stdout),
        utf8(&checked.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&checked.stdout).expect("check prints one JSON document");
    assert_eq!(report["errors"], serde_json::json!([]), "check errors");

    let out = tempfile::tempdir().expect("build output tempdir");
    let out_dir = out.path().join("out");
    let built = build(reef_home, cwd, file, &out_dir);
    assert!(
        built.status.success(),
        "build must accept {}\nstderr: {}",
        file.display(),
        utf8(&built.stderr)
    );
    let stem = file.file_stem().and_then(|stem| stem.to_str()).unwrap();
    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("the built executable runs");
    assert!(
        run.status.success(),
        "the built executable failed: {}\nstderr: {}",
        run.status,
        utf8(&run.stderr)
    );
    assert_eq!(utf8(&run.stdout), expected, "compiled output");
}

const SELECTIVE_COLLISION: &str = "`max` is both imported and declared locally: \
     `import Std.Scalar (max)` brings it into unqualified scope, and `def max` declares it. \
     Rename the local declaration, or stop importing `max` unqualified and refer to the imported \
     one by its qualified name (`Std.Scalar.max`)";

const LOCAL_MAX: &str = "def max(a: i64, b: i64) -> i64 = a\n";

/// A package module importing `import_line` beside `body`, returning the
/// fixture and the module's path.
fn package_module(
    dir_name: &str,
    import_line: &str,
    body: &str,
) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let (dir, reef_home, package) = make_app(dir_name);
    let file = package.join("src/main.ch");
    write_file(
        &file,
        &format!("module Demo.Main\n{import_line}{body}def main() -> i64 = max(1i64, 2i64)\n"),
    );
    (dir, reef_home, file)
}

/// chelis#2885: a package module that selectively imports `max` and declares
/// its own `max` is rejected by every lane with one diagnostic. Before the
/// rule `eval` used the import and returned 2 while `build` used the local
/// declaration and returned 1.
#[test]
fn package_module_import_and_local_declaration_collision_is_rejected_by_every_lane() {
    let (dir, reef_home, file) = package_module(
        "collision-selective",
        "import Std.Scalar (max)\n",
        LOCAL_MAX,
    );
    assert_rejected_identically(&reef_home, dir.path(), &file, SELECTIVE_COLLISION);
}

/// chelis#2885: `import M (..)` brings `max` into unqualified scope as surely
/// as naming it does, so it collides the same way.
#[test]
fn package_module_wildcard_import_collision_is_rejected_by_every_lane() {
    let (dir, reef_home, file) =
        package_module("collision-wildcard", "import Std.Scalar (..)\n", LOCAL_MAX);
    assert_rejected_identically(
        &reef_home,
        dir.path(),
        &file,
        &SELECTIVE_COLLISION.replace("import Std.Scalar (max)", "import Std.Scalar (..)"),
    );
}

/// The negative-parity controls: with no collision, every lane accepts the
/// module and they agree on its value.
#[test]
fn package_module_without_an_import_collision_runs_identically_in_every_lane() {
    for (dir_name, import_line, body, expected) in [
        // The import alone.
        (
            "control-import",
            "import Std.Scalar (max)\n",
            "",
            "main = 2\n",
        ),
        // The local declaration alone.
        ("control-local", "", LOCAL_MAX, "main = 1\n"),
        // A local `max` beside an import of a different name.
        (
            "control-other-name",
            "import Std.Scalar (min)\n",
            "def max(a: i64, b: i64) -> i64 = min(a, b)\n",
            "main = 1\n",
        ),
        // A local `max` beside a qualified-only import of the module that
        // exports `max`: `Std.Scalar.max` stays reachable by its qualified
        // name.
        (
            "control-qualified",
            "import Std.Scalar\n",
            "def max(a: i64, b: i64) -> i64 = Std.Scalar.max(a, b) + 10i64\n",
            "main = 12\n",
        ),
        // A parameter reusing the imported name shadows it lexically.
        (
            "control-parameter",
            "import Std.Scalar (max)\n",
            "def pick(max: i64) -> i64 = max\n",
            "main = 2\n",
        ),
    ] {
        let (dir, reef_home, file) = package_module(dir_name, import_line, body);
        assert_runs_identically(&reef_home, dir.path(), &file, expected);
    }
}
