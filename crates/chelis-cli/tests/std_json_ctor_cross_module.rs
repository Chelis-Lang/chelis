//! Acceptance: a downstream reef package imports a chelis-std ADT
//! constructor by name and pattern-matches on it across module
//! boundaries (chelis#311, follow-up to chelis#157 / PR #307).
//!
//! chelis#157 made the reef linker mangle, rewrite, and *export* ADT
//! constructor names as module symbols, so a downstream program can
//! `import Std.Io.Json (JsonInt)` and pattern-match on `JsonInt`. But
//! the capability is only usable for std constructors once the prebuilt
//! chelis-std bundle is regenerated to carry the exported, mangled
//! constructor symbols in its shell (`.chb`). The bundle is served from
//! the embedded bytes in `crates/chelis-std-bundle` whenever the
//! consumer depends on the bundled version, so this test exercises the
//! shell that ships inside the chelis binary, not a publish-time copy.
//!
//! Fail-old / pass-new contract: against the pre-#311 bundle this test
//! fails at `chelis check` because the std shell does not export the
//! `JsonInt` constructor (`does not export 'JsonInt'`). Against the
//! regenerated bundle it type-checks clean (score 1) and evaluates,
//! proving the constructor symbol crosses the module boundary through
//! the bundle.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn example_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .canonicalize()
        .expect("example path should exist")
}

fn package_std() -> PathBuf {
    example_path("../../packages/chelis-std")
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dst dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

/// `reef.toml` for a downstream app that depends on the bundled
/// chelis-std. The compiler pin auto-syncs with the workspace version.
fn app_reef_toml(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#,
        ver = chelis_compiler_api::COMPILER_VERSION,
    )
}

/// Downstream program that imports the `Json` type and its `JsonInt`
/// constructor by name from `Std.Io.Json`, parses a JSON integer, and
/// pattern-matches the result on `JsonInt(n)` to extract the value.
/// The match arm naming `JsonInt` is the cross-module-constructor use
/// that the regenerated bundle must support.
const CROSS_MODULE_CTOR_PROGRAM: &str = r#"module Demo.Main

import Std.Io.Json (JsonValue, JsonInt, parse_json_value)

def extract_int(value: JsonValue) -> int64 = {
  match value with {
    | JsonInt(n) => n
    | _ => cast(0, int64)
  }
}

answer = extract_int(parse_json_value("42"))
view = print(answer)
"#;

/// `chelis check` on a downstream program that imports and matches the
/// std `JsonInt` constructor must type-check clean (score 1). This is
/// the type-check half of the #311 acceptance: it can only pass once
/// the bundle shell exports the mangled `JsonInt` constructor.
#[test]
#[ignore = "manual gate: chelis-std package acceptance exceeds the default inner-loop budget"]
fn reef_std_json_ctor_cross_module_checks() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("json-ctor-app");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("json-ctor-app"));
    write_file(&app_pkg.join("src/main.ch"), CROSS_MODULE_CTOR_PROGRAM);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
}

/// `chelis eval` on the same program must run and emit the extracted
/// integer (`42`), proving the cross-module-matched constructor lowers
/// and evaluates, not just type-checks.
#[test]
#[ignore = "manual gate: chelis-std package acceptance exceeds the default inner-loop budget"]
fn reef_std_json_ctor_cross_module_evaluates() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("json-ctor-eval-app");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        &app_reef_toml("json-ctor-eval-app"),
    );
    write_file(&app_pkg.join("src/main.ch"), CROSS_MODULE_CTOR_PROGRAM);

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(
        String::from_utf8_lossy(&eval_stdout).contains("42"),
        "eval must extract 42 from the cross-module JsonInt match; got: {}",
        String::from_utf8_lossy(&eval_stdout)
    );
}
