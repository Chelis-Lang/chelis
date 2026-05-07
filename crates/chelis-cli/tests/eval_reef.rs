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

def answer() -> i64 = 7
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
