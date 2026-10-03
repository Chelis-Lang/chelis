//! chelis#2881: a single-file program outside every reef package links its
//! imports against the compiler-bundled chelis-std, the way a package module
//! does, in every command. An import it cannot resolve is rejected and the
//! diagnostic names the module; an import outside chelis-std says a package
//! manifest is needed. Package-mode resolution is unchanged.

mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

const SCALAR_MAX: &str = "import Std.Scalar (max)
def main() -> i64 = max(1i64, 2i64)
";

const SORT_SELECTIVE: &str = "import Std.Sort (sort)
def main() -> tensor[3, i64] = {
  pair = sort(to_tensor([3.0f32, 1.0f32, 2.0f32]), 0i32)
  pair.1
}
";

const SORT_QUALIFIED: &str = "import Std.Sort
def main() -> tensor[3, i64] = {
  pair = Std.Sort.sort(to_tensor([3.0f32, 1.0f32, 2.0f32]), 0i32)
  pair.1
}
";

/// `Std.Sort.sort` declares an `i32` axis; the two-argument builtin `sort`
/// rejects an `i64` axis with its own message. The diagnostic therefore
/// names which of the two the call bound to.
const SORT_I64_AXIS: &str = "def main() -> tensor[3, i64] = {
  pair = sort(to_tensor([3.0f32, 1.0f32, 2.0f32]), 0i64)
  pair.1
}
";

const SHADOWED_MAX: &str = "import Std.Scalar (max)
def max(a: i64, b: i64) -> i64 = a
def main() -> i64 = max(1i64, 2i64)
";

const MODULE_WRAPPED: &str = "module Probe.Single
import Std.Scalar (max)
export (main)
def main() -> i64 = max(1i64, 2i64)
";

const NONEXISTENT_STD: &str = "import Std.Nonexistent (nothing)
def main() -> i64 = 1i64
";

const UNEXPORTED_NAME: &str = "import Std.Scalar (nothing)
def main() -> i64 = 1i64
";

const NON_STD: &str = "import Foo.Bar (baz)
def main() -> i64 = 1i64
";

const IMPORT_FREE: &str = "def main() -> i64 = 1i64
";

const BUILTIN_SORT: &str = "def main() -> tensor[3, i64] = {
  pair = sort(to_tensor([3.0f32, 1.0f32, 2.0f32]), 0i32)
  pair.1
}
";

const SCALAR_PROPERTY: &str = "import Std.Scalar (max)
@property max_is_upper_bound forall(a: i64, b: i64):
  max(a, b) >= a
";

const NONEXISTENT_PROPERTY: &str = "import Std.Nonexistent (max)
@property max_is_upper_bound forall(a: i64, b: i64):
  max(a, b) >= a
";

const NONEXISTENT_DIAGNOSTIC: &str = "unresolved import `Std.Nonexistent`";
const MANIFEST_DIAGNOSTIC: &str = "needs a `reef.toml` package manifest";
const COLLISION_DIAGNOSTIC: &str = "`max` is both imported and declared locally: \
     `import Std.Scalar (max)` brings it into unqualified scope, and `def max` declares it. \
     Rename the local declaration, or stop importing `max` unqualified and refer to the imported \
     one by its qualified name (`Std.Scalar.max`)";

/// A directory outside every reef package holding one program file.
struct SingleFile {
    dir: TempDir,
    path: PathBuf,
}

fn single_file(name: &str, source: &str) -> SingleFile {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    SingleFile { dir, path }
}

impl SingleFile {
    fn chelis(&self) -> Command {
        let mut command = Command::cargo_bin("chelis").expect("binary");
        command
            .current_dir(self.dir.path())
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", self.dir.path().join("reef-home"));
        command
    }

    fn path(&self) -> &str {
        self.path.to_str().expect("utf-8 path")
    }

    fn eval(&self) -> assert_cmd::assert::Assert {
        self.chelis().args(["eval", "--file", self.path()]).assert()
    }

    fn check(&self) -> assert_cmd::assert::Assert {
        self.chelis().args(["check", self.path()]).assert()
    }

    fn build(&self) -> (assert_cmd::assert::Assert, PathBuf) {
        let out_dir = self.dir.path().join("out");
        let assert = self
            .chelis()
            .args(["build", self.path(), "--target", "c", "--output"])
            .arg(&out_dir)
            .assert();
        (assert, out_dir)
    }

    /// Build natively, run the executable `chelis build` publishes, and
    /// return stdout.
    fn build_and_run(&self) -> String {
        let (assert, out_dir) = self.build();
        assert.success();
        let stem = self.path.file_stem().and_then(|s| s.to_str()).unwrap();
        let run = StdCommand::new(out_dir.join(stem))
            .output()
            .expect("compiled binary should run");
        assert!(
            run.status.success(),
            "binary failed: {}\nstderr: {}",
            run.status,
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout).expect("utf-8 stdout")
    }
}

#[test]
fn single_file_scalar_import_resolves_in_eval_check_build_and_cost() {
    let program = single_file("scalar_max", SCALAR_MAX);
    program.eval().success().stdout("main = 2\n");
    program.check().success().stdout(contains("\"errors\": []"));
    assert_eq!(program.build_and_run(), "main = 2\n");
    program
        .chelis()
        .args(["cost", program.path()])
        .assert()
        .success()
        .stdout(contains("main:"));
}

#[test]
fn single_file_selective_sort_import_resolves_in_eval_check_and_build() {
    let program = single_file("sort_selective", SORT_SELECTIVE);
    program
        .eval()
        .success()
        .stdout("main = tensor(shape=[3], data=[1, 2, 0])\n");
    program.check().success().stdout(contains("\"errors\": []"));
    let (assert, out_dir) = program.build();
    assert.success();
    assert!(out_dir.join("sort_selective.c").is_file());
}

#[test]
fn single_file_selective_sort_import_binds_the_module_function_not_the_builtin() {
    let imported = single_file(
        "sort_imported",
        &format!("import Std.Sort (sort)\n{SORT_I64_AXIS}"),
    );
    imported.eval().failure().stderr(
        contains("PrecisionMismatch: precision mismatch: expected i32, got i64")
            .and(contains("sort expects i32 axis").not()),
    );

    let builtin = single_file("sort_builtin", SORT_I64_AXIS);
    builtin
        .eval()
        .failure()
        .stderr(contains("sort expects i32 axis, got i64"));
}

#[test]
fn single_file_import_free_program_keeps_the_builtin() {
    single_file("import_free", IMPORT_FREE)
        .eval()
        .success()
        .stdout("main = 1\n");
    single_file("builtin_sort", BUILTIN_SORT)
        .eval()
        .success()
        .stdout("main = tensor(shape=[3], data=[1, 2, 0])\n");
}

#[test]
fn single_file_qualified_import_resolves_and_its_absence_does_not() {
    single_file("sort_qualified", SORT_QUALIFIED)
        .eval()
        .success()
        .stdout("main = tensor(shape=[3], data=[1, 2, 0])\n");

    let unimported = SORT_QUALIFIED.trim_start_matches("import Std.Sort\n");
    single_file("sort_unimported", unimported)
        .eval()
        .failure()
        .stderr(contains("unknown constructor: Std"));
}

/// A declaration of an imported name is rejected, as it is in a package
/// module (spec/02 §P2, chelis#2885), by every command and with one
/// diagnostic naming the import and the local declaration.
#[test]
fn single_file_declaration_colliding_with_an_import_is_rejected() {
    let program = single_file("shadowed_max", SHADOWED_MAX);
    program
        .check()
        .code(2)
        .stdout(contains(COLLISION_DIAGNOSTIC));
    program
        .eval()
        .failure()
        .stderr(contains(COLLISION_DIAGNOSTIC))
        .stdout(contains("main =").not());
    let (assert, out_dir) = program.build();
    assert.failure().stderr(contains(COLLISION_DIAGNOSTIC));
    assert!(!out_dir.exists(), "a rejected build must publish no output");
}

#[test]
fn single_file_module_declaration_resolves_its_imports() {
    let program = single_file("module_wrapped", MODULE_WRAPPED);
    program.eval().success().stdout("main = 2\n");
    program.check().success().stdout(contains("\"errors\": []"));
}

#[test]
fn single_file_nonexistent_std_module_is_rejected_naming_it() {
    let program = single_file("nonexistent_std", NONEXISTENT_STD);
    program
        .check()
        .code(2)
        .stdout(contains(NONEXISTENT_DIAGNOSTIC).and(contains(MANIFEST_DIAGNOSTIC).not()));
    program
        .eval()
        .failure()
        .stderr(contains(NONEXISTENT_DIAGNOSTIC))
        .stdout(contains("main =").not());
    let (assert, _) = program.build();
    assert.failure().stderr(contains(NONEXISTENT_DIAGNOSTIC));
}

#[test]
fn single_file_unexported_name_is_rejected() {
    let program = single_file("unexported_name", UNEXPORTED_NAME);
    program
        .check()
        .code(2)
        .stdout(contains("module `Std.Scalar` does not export `nothing`"));
    program
        .eval()
        .failure()
        .stderr(contains("module `Std.Scalar` does not export `nothing`"));
}

#[test]
fn single_file_non_std_import_needs_a_package_manifest() {
    let program = single_file("non_std", NON_STD);
    program
        .check()
        .code(2)
        .stdout(contains("unresolved import `Foo.Bar`").and(contains(MANIFEST_DIAGNOSTIC)));
    program
        .eval()
        .failure()
        .stderr(contains("unresolved import `Foo.Bar`").and(contains(MANIFEST_DIAGNOSTIC)));
    let (assert, _) = program.build();
    assert
        .failure()
        .stderr(contains("unresolved import `Foo.Bar`").and(contains(MANIFEST_DIAGNOSTIC)));
}

#[test]
fn single_file_prove_resolves_imports_and_rejects_unresolvable_ones() {
    let program = single_file("scalar_property", SCALAR_PROPERTY);
    program
        .chelis()
        .args(["prove", program.path()])
        .assert()
        .success()
        .stdout(contains("1 passed, 0 failed, 0 unsupported, 0 errors"));

    let program = single_file("nonexistent_property", NONEXISTENT_PROPERTY);
    program
        .chelis()
        .args(["prove", program.path()])
        .assert()
        .failure()
        .stderr(contains(NONEXISTENT_DIAGNOSTIC));
}

fn package_chelis(reef_home: &Path, app: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .current_dir(app)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home);
    command
}

#[test]
fn package_module_std_import_resolves_as_before() {
    let (_dir, reef_home, app) = make_app("single_file_imports_pkg");
    write_file(
        &app.join("src/main.ch"),
        &format!("module Demo.Main\n{SCALAR_MAX}"),
    );
    package_chelis(&reef_home, &app)
        .args(["eval", "--file", "src/main.ch"])
        .assert()
        .success()
        .stdout("main = 2\n");
}

#[test]
fn package_module_nonexistent_std_import_keeps_the_package_diagnostic() {
    let (_dir, reef_home, app) = make_app("single_file_imports_pkg_bad");
    write_file(
        &app.join("src/main.ch"),
        &format!("module Demo.Main\n{NONEXISTENT_STD}"),
    );
    package_chelis(&reef_home, &app)
        .args(["check", "src/main.ch"])
        .assert()
        .code(2)
        .stdout(
            contains(NONEXISTENT_DIAGNOSTIC)
                .and(contains("compiler-bundled").not())
                .and(contains(MANIFEST_DIAGNOSTIC).not()),
        );
}
