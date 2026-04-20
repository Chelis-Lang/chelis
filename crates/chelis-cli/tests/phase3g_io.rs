use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
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

fn phase3g_text_pipeline_example() -> PathBuf {
    example_path("../../examples/illustrative/phase3g_text_pipeline")
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

fn surf_string_literal(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn cpu_toolchain_for_source(
    out_dir: &Path,
    source: &str,
) -> chelis_backend_c::toolchain::NativeToolchain {
    let needs_blas = fs::read_to_string(out_dir.join(source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    )
}

fn gcc_link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let toolchain = cpu_toolchain_for_source(out_dir, source);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("gcc should run")
}

#[test]
fn reef_std_io_module_checks_and_builds() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("io-app");
    let out_dir = dir.path().join("out");
    let data_path = dir.path().join("dataset.txt");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(&data_path, "  alpha  \n\n beta \n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "io-app"
version = "0.1.0"
compiler = "=0.1.13"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    let dataset = surf_string_literal(data_path.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.IO (read_trimmed_lines, read_head_bytes, exists, mmap_size)

lines = read_trimmed_lines({dataset})
head = read_head_bytes({dataset}, cast(4, int64))
file_is_present = exists({dataset})
size = mmap_size({dataset})
lines_view = print(lines)
head_view = print(head)
exists_view = print(file_is_present)
size_view = print(size)
"#
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(out_dir.join("main.c").exists(), "expected generated main.c");
    assert!(out_dir.join("main.h").exists(), "expected generated main.h");
}

#[test]
fn reef_std_io_module_rejects_missing_export() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("io-app-bad");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "io-app-bad"
version = "0.1.0"
compiler = "=0.1.13"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.IO (missing_symbol)

x = missing_symbol("foo")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing_symbol"))
        .stderr(predicate::str::contains("does not export"));
}

#[test]
fn phase3g_text_pipeline_acceptance_oracle() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("phase3g-text-pipeline");
    let out_dir = dir.path().join("out");
    copy_dir_recursive(&package_std(), &std_pkg);
    copy_dir_recursive(&phase3g_text_pipeline_example(), &app_pkg);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
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

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = gcc_link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("main"))
        .current_dir(&app_pkg)
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn reef_std_json_module_fails_loudly_on_malformed_input() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("json-bad");
    let bad_json = dir.path().join("bad.json");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(&bad_json, "{\"oops\": [1, 2, }");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "json-bad"
version = "0.1.0"
compiler = "=0.1.13"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    let bad = surf_string_literal(bad_json.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.IO.Json (load_json)

cfg = load_json({bad})
view = print(cfg)
"#
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("load_json failed for"))
        .stderr(predicate::str::contains("bad.json"));
}

#[test]
fn reef_std_json_try_module_reports_none_on_malformed_input() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("json-try-bad");
    let bad_json = dir.path().join("bad.json");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(&bad_json, "{\"oops\": [1, 2, }");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "json-try-bad"
version = "0.1.0"
compiler = "=0.1.13"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    let bad = surf_string_literal(bad_json.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.IO.Json (try_load_json)

ok = match try_load_json({bad}) with {{
  | Some(_) => true
  | None => false
}}
view = print(ok)
"#
        ),
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
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
        String::from_utf8_lossy(&eval_stdout).contains("ok = false")
            || String::from_utf8_lossy(&eval_stdout).contains("false")
    );
}

#[test]
fn reef_std_csv_module_fails_loudly_on_unclosed_quote_rows() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("csv-bad");
    let bad_csv = dir.path().join("bad.csv");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(&bad_csv, "text,label\n\"broken,1\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "csv-bad"
version = "0.1.0"
compiler = "=0.1.13"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    let bad = surf_string_literal(bad_csv.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.IO.Csv (read_csv)

rows = read_csv({bad})
view = print(rows)
"#
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("read_csv failed for"))
        .stderr(predicate::str::contains("bad.csv"));
}

#[test]
fn reef_std_csv_try_module_reports_none_on_unclosed_quote_rows() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("csv-try-bad");
    let bad_csv = dir.path().join("bad.csv");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(&bad_csv, "text,label\n\"broken,1\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "csv-try-bad"
version = "0.1.0"
compiler = "=0.1.13"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    let bad = surf_string_literal(bad_csv.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.IO.Csv (try_read_csv)

ok = match try_read_csv({bad}) with {{
  | Some(_) => true
  | None => false
}}
view = print(ok)
"#
        ),
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
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
        String::from_utf8_lossy(&eval_stdout).contains("ok = false")
            || String::from_utf8_lossy(&eval_stdout).contains("false")
    );
}
