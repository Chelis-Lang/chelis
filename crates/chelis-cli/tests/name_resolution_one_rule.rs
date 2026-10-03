//! One name-resolution rule in every lane (spec/02 §P2).
//!
//! chelis#2918: a file belongs to the reef package found by walking up from
//! the file itself, so `eval`, `check`, `build`, and `chelis test` resolve its
//! imports the same way whatever directory they run in. A file inside a
//! package but outside its source roots, with or without a `module`
//! declaration, is an entry of that package and imports its modules; a file
//! outside every package cannot, even when the command runs inside one.
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

/// A package whose `Demo.Special` module exports `answer`. `package` is its
/// directory and `outside` the directory above it, which no package contains.
struct LooseFixture {
    _dir: tempfile::TempDir,
    reef_home: PathBuf,
    outside: PathBuf,
    package: PathBuf,
}

fn loose_fixture(dir_name: &str) -> LooseFixture {
    let (dir, reef_home, package) = make_app(dir_name);
    write_file(
        &package.join("src/special.ch"),
        "module Demo.Special\ndef answer() -> i32 = 7\n",
    );
    LooseFixture {
        outside: dir.path().to_path_buf(),
        _dir: dir,
        reef_home,
        package,
    }
}

const PACKAGE_IMPORT: &str = "import Demo.Special (answer)\nbench = answer()\n";

/// chelis#2918: a file with no `module` declaration inside a package imports
/// the package's modules in every lane, and the directory the command runs in
/// changes nothing.
#[test]
fn loose_file_inside_a_package_imports_its_modules_in_every_lane_from_any_directory() {
    let fixture = loose_fixture("loose-inside");
    let file = fixture.package.join("scratch.ch");
    write_file(&file, PACKAGE_IMPORT);
    assert_runs_identically(&fixture.reef_home, &fixture.outside, &file, "bench = 7\n");

    let from_inside = eval(&fixture.reef_home, &fixture.package, &file);
    assert!(from_inside.status.success(), "eval from inside the package");
    assert_eq!(utf8(&from_inside.stdout), "bench = 7\n");
}

/// chelis#2918: the same file outside every package belongs to no package, so
/// its package import is rejected by every lane with one diagnostic, even
/// when the command runs inside the package.
#[test]
fn loose_file_outside_every_package_is_rejected_identically_from_inside_a_package() {
    let fixture = loose_fixture("loose-outside");
    let file = fixture.outside.join("outside.ch");
    write_file(&file, PACKAGE_IMPORT);
    assert_rejected_identically(
        &fixture.reef_home,
        &fixture.package,
        &file,
        "unresolved import `Demo.Special`: a program outside a reef package imports only from \
         the compiler-bundled `chelis-std` (`Std.*`), so importing `Demo.Special` needs a \
         `reef.toml` package manifest",
    );
}

/// chelis#2885 and chelis#2918 together: a file with no `module` declaration
/// inside a package is an entry of that package and follows the same rule.
#[test]
fn loose_entry_import_and_local_declaration_collision_is_rejected_by_every_lane() {
    let fixture = loose_fixture("collision-loose");
    let file = fixture.package.join("scratch.ch");
    write_file(
        &file,
        &format!("import Std.Scalar (max)\n{LOCAL_MAX}def main() -> i64 = max(1i64, 2i64)\n"),
    );
    assert_rejected_identically(
        &fixture.reef_home,
        &fixture.outside,
        &file,
        SELECTIVE_COLLISION,
    );
}

const PASSING_TEST: &str = "def test_truth() -> unit = test_assert(true, \"truth\")\n";

fn chelis_test(reef_home: &Path, cwd: &Path, target: &Path) -> Output {
    chelis(reef_home, cwd)
        .arg("test")
        .arg(target)
        .output()
        .expect("run chelis test")
}

/// chelis#2918: `chelis test PATH` finds the package by walking up from the
/// target, never from the current directory. A test file inside a package
/// runs in it from anywhere; the same file outside every package has no
/// package to run in, even when the command runs inside one.
#[test]
fn chelis_test_finds_the_package_from_the_target_not_the_current_directory() {
    let fixture = loose_fixture("test-target");
    let inside = fixture.package.join("tests/truth.ch");
    write_file(&inside, PASSING_TEST);
    let ran = chelis_test(&fixture.reef_home, &fixture.outside, &inside);
    assert!(
        ran.status.success(),
        "a test file inside the package runs from outside it\nstdout: {}\nstderr: {}",
        utf8(&ran.stdout),
        utf8(&ran.stderr)
    );
    assert!(utf8(&ran.stdout).contains("1 passed, 0 failed"));

    let outside = fixture.outside.join("truth.ch");
    write_file(&outside, PASSING_TEST);
    let rejected = chelis_test(&fixture.reef_home, &fixture.package, &outside);
    assert!(
        !rejected.status.success(),
        "a test file outside every package must not run in the current directory's package\n\
         stdout: {}",
        utf8(&rejected.stdout)
    );
    assert!(
        utf8(&rejected.stderr).contains("no reef.toml found"),
        "the rejection names the missing manifest:\n{}",
        utf8(&rejected.stderr)
    );
}

/// chelis#2918: a file with a `module` declaration that lies inside a package
/// but outside its source roots, such as a test under `tests/`, is an entry of
/// the package too. `eval`, `check`, `build`, and `chelis test` all link it
/// against the package from any directory; its declaration adds no module to
/// the package. Before, `eval` and `chelis test` linked it while `check` and
/// `build` rejected it as "not a source file under any declared root".
#[test]
fn module_file_outside_the_source_roots_is_a_package_entry_in_every_lane() {
    let fixture = loose_fixture("module-entry");
    let file = fixture.package.join("tests/probe.ch");
    write_file(&file, &format!("module Demo.Tests.Probe\n{PACKAGE_IMPORT}"));
    assert_runs_identically(&fixture.reef_home, &fixture.outside, &file, "bench = 7\n");

    let suite = fixture.package.join("tests/answer.ch");
    write_file(
        &suite,
        "module Demo.Tests.Answer\n\
         import Demo.Special (answer)\n\
         def test_answer() -> unit = test_assert(eq(answer(), 7), \"answer\")\n",
    );
    let ran = chelis_test(&fixture.reef_home, &fixture.outside, &suite);
    assert!(
        ran.status.success(),
        "chelis test links the module file against the package\nstdout: {}\nstderr: {}",
        utf8(&ran.stdout),
        utf8(&ran.stderr)
    );
    assert!(utf8(&ran.stdout).contains("1 passed, 0 failed"));
}
