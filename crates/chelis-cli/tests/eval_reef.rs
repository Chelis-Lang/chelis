use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

#[test]
fn eval_file_uses_current_reef_package_for_ad_hoc_imports() {
    let dir = tempdir().expect("tempdir");
    let app_pkg = dir.path().join("demo");
    let external = dir.path().join("snippet.ch");

    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "demo"
version = "0.1.0"
compiler = "={}"
module_prefix = "Demo"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &app_pkg.join("src/special.ch"),
        r#"module Demo.Special

def answer() -> i32 = 7
"#,
    );
    write_file(
        &external,
        r#"import Demo.Special (answer)

bench = answer()
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&app_pkg)
        .args(["eval", "--file", external.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

/// A file with no `module` declaration is evaluated in the package that
/// contains the current working directory, not the one around the file: from
/// a directory outside every package, such a file inside a package belongs to
/// no package, so its imports resolve against the bundled chelis-std alone.
#[test]
fn eval_file_without_a_module_ignores_the_package_around_it_from_outside_every_package() {
    let dir = tempdir().expect("tempdir");
    let app_pkg = dir.path().join("demo");
    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "demo"
version = "0.1.0"
compiler = "={}"
module_prefix = "Demo"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &app_pkg.join("src/special.ch"),
        r#"module Demo.Special
def answer() -> i32 = 7
"#,
    );
    let package_import = app_pkg.join("scratch.ch");
    write_file(
        &package_import,
        r#"import Demo.Special (answer)
bench = answer()
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["eval", "--file", package_import.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unresolved import `Demo.Special`").and(
                predicate::str::contains("needs a `reef.toml` package manifest"),
            ),
        );
    let std_import = app_pkg.join("scalar.ch");
    write_file(
        &std_import,
        r#"import Std.Scalar (abs)
bench = abs(-3i64)
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["eval", "--file", std_import.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("bench = 3"));
}
