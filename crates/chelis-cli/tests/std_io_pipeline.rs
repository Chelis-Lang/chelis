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

fn io_pipeline_example() -> PathBuf {
    example_path("../../examples/illustrative/io_pipeline")
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

/// Build the standard `reef.toml` body for a phase-3g IO test app
/// depending on `chelis-std = "0.1.0"`. The compiler pin auto-syncs with
/// the workspace version via `chelis_compiler_api::COMPILER_VERSION`.
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
    cmd.arg(out_dir.join("libchelis_runtime.a"));
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("gcc should run")
}

#[test]
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("io-app"));
    let dataset = surf_string_literal(data_path.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io (read_trimmed_lines, read_head_bytes, exists, mmap_size)

lines = read_trimmed_lines({dataset})
head = read_head_bytes({dataset}, cast(4, i64))
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_io_module_rejects_missing_export() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("io-app-bad");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("io-app-bad"));
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io (missing_symbol)

x = missing_symbol("foo")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing_symbol"))
        .stderr(predicate::str::contains("does not export"));
}

#[test]
#[ignore = "manual gate: CSV and JSON example acceptance builds, links, and runs generated C"]
fn io_pipeline_acceptance_oracle() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("io-pipeline");
    let out_dir = dir.path().join("out");
    copy_dir_recursive(&package_std(), &std_pkg);
    copy_dir_recursive(&io_pipeline_example(), &app_pkg);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("json-bad"));
    let bad = surf_string_literal(bad_json.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io.Json (load_json)

cfg = load_json({bad})
view = print(cfg)
"#
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("json-try-bad"));
    let bad = surf_string_literal(bad_json.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io.Json (try_load_json)

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
        String::from_utf8_lossy(&eval_stdout).contains("ok = false")
            || String::from_utf8_lossy(&eval_stdout).contains("false")
    );
}

#[test]
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("csv-bad"));
    let bad = surf_string_literal(bad_csv.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io.Csv (read_csv)

rows = read_csv({bad})
view = print(rows)
"#
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("csv-try-bad"));
    let bad = surf_string_literal(bad_csv.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io.Csv (try_read_csv)

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
        String::from_utf8_lossy(&eval_stdout).contains("ok = false")
            || String::from_utf8_lossy(&eval_stdout).contains("false")
    );
}

#[test]
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_parquet_module_rejects_missing_export() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("parquet-bad");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(&app_pkg.join("reef.toml"), &app_reef_toml("parquet-bad"));
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Parquet (nonexistent_parquet_fn)

x = nonexistent_parquet_fn("foo")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("nonexistent_parquet_fn"))
        .stderr(predicate::str::contains("does not export"));
}

/// PR #1213 review finding: the malformed-CSV None contract was only
/// exercised on the eval lane (`chelis test` / `chelis eval`), so a
/// C-only defect in the compiled `try_read_csv` path (sentinel handling,
/// fold validation, the None branch) could pass every existing test.
/// This builds, links, and RUNS the generated C against a valid control
/// file plus the three malformed shapes (short row, wide row,
/// unterminated quote) and asserts the compiled verdicts byte-match the
/// eval lane's.
#[test]
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_csv_compiled_lane_matches_eval_on_malformed_rows() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("csv-compiled-verdicts");
    let out_dir = dir.path().join("out");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    let valid_csv = dir.path().join("valid.csv");
    let short_csv = dir.path().join("short.csv");
    let wide_csv = dir.path().join("wide.csv");
    let quote_csv = dir.path().join("quote.csv");
    write_file(&valid_csv, "a,b\n1,2\n3,4\n");
    write_file(&short_csv, "a,b\n1\n2,3\n");
    write_file(&wide_csv, "a,b\n1,2\n3,4,5\n");
    write_file(&quote_csv, "k\nok\n\"dangling\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        &app_reef_toml("csv-compiled-verdicts"),
    );
    let valid = surf_string_literal(valid_csv.to_str().unwrap());
    let short = surf_string_literal(short_csv.to_str().unwrap());
    let wide = surf_string_literal(wide_csv.to_str().unwrap());
    let quote = surf_string_literal(quote_csv.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io.Csv (try_read_csv)

ok_valid = match try_read_csv({valid}) with {{
  | Some(rows) => string_concat("valid:SOME:", to_string(len(rows)))
  | None => "valid:NONE"
}}
ok_short = match try_read_csv({short}) with {{
  | Some(_) => "short:SOME"
  | None => "short:NONE"
}}
ok_wide = match try_read_csv({wide}) with {{
  | Some(_) => "wide:SOME"
  | None => "wide:NONE"
}}
ok_quote = match try_read_csv({quote}) with {{
  | Some(_) => "quote:SOME"
  | None => "quote:NONE"
}}
v1 = print(ok_valid)
v2 = print(ok_short)
v3 = print(ok_wide)
v4 = print(ok_quote)
"#
        ),
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    let text = String::from_utf8_lossy(&run_output.stdout);
    assert!(text.contains("valid:SOME:2"), "valid control: {text}");
    assert!(text.contains("short:NONE"), "short row: {text}");
    assert!(text.contains("wide:NONE"), "wide row: {text}");
    assert!(text.contains("quote:NONE"), "unterminated quote: {text}");
}

/// PR #1213 review finding: a malformed FIRST data row followed by a
/// line long enough to be expensive must return None WITHOUT parsing
/// the later line (the recursive pre-#1213 parse_rows short-circuited;
/// a shape that parses every line before judging validity does not).
/// `parse_line_chars` recurses per character on every lane (chelis#1225),
/// so an unfixed parse of the 4 MiB line overflows the C stack here,
/// while the corpus twin of this test
/// (`test_short_first_row_before_long_line_returns_none`) covers the
/// same shape at 4 KiB on the eval lane, whose per-character evaluator
/// frames are ~3 orders of magnitude larger. A 4 MiB line in a VALID
/// row position would still crash the compiled lane — that is
/// chelis#1225's parser wall, not a row-control-flow defect.
#[test]
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_csv_compiled_lane_short_circuits_before_long_line() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("csv-compiled-longline");
    let out_dir = dir.path().join("out");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    let long_csv = dir.path().join("short_then_long.csv");
    let mut contents = String::from("a,b\n1\n");
    contents.push_str(&"x".repeat(4 * 1024 * 1024));
    contents.push('\n');
    write_file(&long_csv, &contents);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        &app_reef_toml("csv-compiled-longline"),
    );
    let long = surf_string_literal(long_csv.to_str().unwrap());
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            r#"module Demo.Main

import Std.Io.Csv (try_read_csv)

ok_longline = match try_read_csv({long}) with {{
  | Some(_) => "longline:SOME"
  | None => "longline:NONE"
}}
v1 = print(ok_longline)
"#
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        "compiled binary must not crash on the short-row + 4 MiB-line file, got status {}",
        run_output.status
    );
    let text = String::from_utf8_lossy(&run_output.stdout);
    assert!(text.contains("longline:NONE"), "long-line verdict: {text}");
}

/// End-to-end Parquet IO package acceptance: imports both
/// `read_parquet` and `write_parquet`, type-checks a `def` for each
/// (so `chelis check` clean-with-score-1 pins the type-check path for
/// both symbols), then builds through `chelis build --target c`. This
/// subsumes the narrower read-only and write-only resolves/type-checks
/// cases, which were strict subsets of this test's check assertion.
#[test]
#[ignore = "manual gate: standard IO package acceptance suite exceeds the default inner-loop budget"]
fn reef_std_parquet_module_builds_cleanly() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("parquet-build-app");
    let out_dir = dir.path().join("out");
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
        &app_reef_toml("parquet-build-app"),
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Parquet (read_parquet, write_parquet)

def load_rows(path: string) -> List[Dict[string, string]] = read_parquet(path)
def save_rows(path: string, rows: List[Dict[string, string]]) -> unit = write_parquet(path, rows)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
