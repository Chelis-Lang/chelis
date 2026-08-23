//! Phase 3t.4 — smoke tests for the `chelis test` CLI subcommand.
//!
//! These are sanity tests proving the pipe works: discovery, per-test
//! isolation, `--filter`, `--json`, and exit-code shape for the runner-error
//! paths. Exhaustive coverage (timeouts, typecheck failures, mixed
//! modules, full pseudo-nautilus fixture) is 3t.5's job.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

/// Create a bare reef package with a `tests/` directory and return
/// `(tempdir, package_root)`. No chelis-std dep — tests call
/// `test_assert_*` as direct runtime builtins.
fn make_reef_package(dir_name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(dir_name);
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Smoke"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    // Reef requires at least one source file to resolve a module graph.
    write_file(
        &pkg.join("src/main.ch"),
        "module Smoke.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    (dir, pkg)
}

/// The headline of the attributed report `chelis test` prints when
/// `--batch-mode auto` gives up on a batch it had already started (chelis#1261).
const BATCH_FALLBACK_NOTE: &str = "suite batching was abandoned";

/// A reef package with two library modules that export the same name `same`.
/// This is chelis#1261's shape: an explicit `import M (same)` in one test file
/// must survive a sibling test file that declares its own `same`, and must not
/// be silently rebound to a second module's `same` either.
fn make_shared_name_reef_package(dir_name: &str) -> (tempfile::TempDir, PathBuf) {
    let (dir, pkg) = make_reef_package(dir_name);
    write_file(
        &pkg.join("src/helpers.ch"),
        "module Smoke.Helpers\n\
         export (same)\n\
         def same(a: int64, b: int64) -> bool = eq(a, b)\n",
    );
    write_file(
        &pkg.join("src/other.ch"),
        "module Smoke.Other\n\
         export (same)\n\
         def same(a: int64, b: int64) -> bool = eq(a, b)\n",
    );
    (dir, pkg)
}

/// chelis#1261's two test files: `a_import.ch` imports the package's `same`
/// over `int64`, `b_local.ch` declares an unrelated local `same` over lists.
fn write_import_collision_files(pkg: &Path) {
    write_file(
        &pkg.join("tests/a_import.ch"),
        "module Smoke.Tests.UsesImport\n\
         import Smoke.Helpers (same)\n\
         def test_uses_import() -> unit = \
         test_assert(same(cast(1, int64), cast(1, int64)), \"1 == 1\")\n",
    );
    write_file(
        &pkg.join("tests/b_local.ch"),
        "module Smoke.Tests.LocalSame\n\
         def same(xs: List[int64], ys: List[int64]) -> bool = eq(len(xs), len(ys))\n\
         def test_local_same() -> unit = \
         test_assert(same([cast(1, int64)], [cast(2, int64)]), \"same length\")\n",
    );
}

/// Run the suite with the batch worker forced to abort, and return stderr.
///
/// The abort kills whatever files the parent handed the batch worker, and the
/// fallback note names exactly those files. That makes this the membership
/// oracle for the eligibility guard: a file the guard demoted never reaches
/// the worker, so it cannot appear in the note.
fn forced_batch_abort_stderr(pkg: &Path) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_FORCE_BATCH_ABORT", "1")
        .current_dir(pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        stderr.contains(BATCH_FALLBACK_NOTE),
        "the forced batch abort was not reported, so this run proves nothing \
         about batch membership:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&output.stdout)
    );
    stderr
}

#[test]
fn chelis_test_passing_file_exits_zero() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-pass");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.Pass

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_ok"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

#[test]
fn chelis_test_failing_file_exits_one_with_message() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-fail");
    write_file(
        &pkg.join("tests/fail.ch"),
        r#"module Smoke.Tests.Fail

def test_bad() -> unit = test_assert(false, "boom")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("test_bad"))
        .stdout(predicate::str::contains("FAIL"))
        .stdout(predicate::str::contains("assert failed: boom"))
        .stdout(predicate::str::contains("0 passed, 1 failed"));
}

#[test]
fn chelis_test_pipe_stage_host_runtime_fallback_does_not_panic() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-pipe-stage");
    write_file(
        &pkg.join("tests/pipe.ch"),
        r#"module Smoke.Tests.Pipe

type Sign =
  | Neg
  | Pos

def choose(s: Sign) -> f32 = match s with {
  | Neg => cast(0.0, f32)
  | Pos => cast(1.0, f32)
}

def pipe_choose(s: Sign) -> f32 = s |> choose

def test_pipe_choose() -> unit = test_assert(pipe_choose(Pos) == cast(1.0, f32), "pipe choose")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/pipe.ch"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_pipe_choose"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("worker exited").not())
        .stdout(predicate::str::contains("panic").not())
        .stderr(predicate::str::contains("panic").not());
}

#[test]
fn chelis_test_filter_narrows_selection() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-filter");
    write_file(
        &pkg.join("tests/many.ch"),
        r#"module Smoke.Tests.Many

def test_alpha() -> unit = test_assert(true, "a")
def test_beta() -> unit = test_assert(false, "b failed")
def test_gamma() -> unit = test_assert(true, "c")
"#,
    );

    // Filter picks a single test → suite passes.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--filter", "alpha", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_alpha"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("1 passed, 0 failed"))
        // Filter must drop test_beta and test_gamma entirely.
        .stdout(predicate::str::contains("test_beta").not())
        .stdout(predicate::str::contains("test_gamma").not());

    // Filter that matches nothing → no counts, runner exits 0 with "0 passed, 0 failed".
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--filter", "nonexistent_xyz", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("0 passed, 0 failed"));
}

#[test]
fn chelis_test_json_emits_ndjson_records() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-json");
    write_file(
        &pkg.join("tests/mixed.ch"),
        r#"module Smoke.Tests.Mixed

def test_one() -> unit = test_assert(true, "one")
def test_two() -> unit = test_assert(false, "two broken")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--json", "tests/"])
        .assert()
        .failure()
        .code(1)
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        3,
        "expected 2 test rows + 1 summary row, got {lines:?}"
    );
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("line not valid JSON: {line:?} ({e})"));
    }
    // First row: passing test with no `message` field.
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("first record parses");
    assert_eq!(first["test"], "test_one");
    assert_eq!(first["status"], "pass");
    assert!(first.get("message").is_none());
    // Second row: failing test carries its error string.
    let second: serde_json::Value = serde_json::from_str(lines[1]).expect("second record parses");
    assert_eq!(second["test"], "test_two");
    assert_eq!(second["status"], "fail");
    assert_eq!(second["message"], "assert failed: two broken");
    // Final row: summary.
    let summary: serde_json::Value = serde_json::from_str(lines[2]).expect("summary parses");
    assert_eq!(summary["summary"]["passed"], 1);
    assert_eq!(summary["summary"]["failed"], 1);
}

#[test]
fn chelis_test_jobs_emits_json_in_discovery_order() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-jobs");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.First

def test_first() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.Second

def test_second() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--json", "--jobs", "2", "tests/"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(lines[0]["file"], "tests/a_first.ch");
    assert_eq!(lines[0]["test"], "test_first");
    assert_eq!(lines[1]["file"], "tests/b_second.ch");
    assert_eq!(lines[1]["test"], "test_second");
    assert_eq!(lines[2]["summary"]["passed"], 2);
    assert_eq!(lines[2]["summary"]["failed"], 0);
}

#[test]
fn chelis_test_auto_batch_preserves_two_file_ndjson_order() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-order");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.BatchFirst

def test_first() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.BatchSecond

def test_second() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "--json", "tests/"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(lines[0]["file"], "tests/a_first.ch");
    assert_eq!(lines[0]["test"], "test_first");
    assert_eq!(lines[0]["status"], "pass");
    assert_eq!(lines[1]["file"], "tests/b_second.ch");
    assert_eq!(lines[1]["test"], "test_second");
    assert_eq!(lines[1]["status"], "pass");
    assert_eq!(lines[2]["summary"]["passed"], 2);
    assert_eq!(lines[2]["summary"]["failed"], 0);
}

#[test]
fn chelis_test_batch_mode_file_keeps_per_file_output_shape() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-file-mode");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.BatchFileMode

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "file", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tests/pass.ch"))
        .stdout(predicate::str::contains("test_ok"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

#[test]
fn chelis_test_auto_batch_compile_error_falls_back_to_file_rows() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-compile-fallback");
    write_file(
        &pkg.join("tests/a_broken.ch"),
        r#"module Smoke.Tests.BatchBroken

import Missing.Module (ghost)

def test_broken() -> unit = test_assert(true, "unreachable")
"#,
    );
    write_file(
        &pkg.join("tests/b_ok.ch"),
        r#"module Smoke.Tests.BatchOk

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected test failure exit; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("tests/a_broken.ch")
            && stdout.contains("<file>")
            && stdout.contains("FAIL"),
        "broken file-level row missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("tests/b_ok.ch") && stdout.contains("test_ok") && stdout.contains("PASS"),
        "healthy sibling PASS missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    // chelis#1261: the batch worker's own diagnostic used to be the only
    // signal that the batched path had been dropped, and nothing said which
    // files it belonged to. The runner now attributes the abandonment.
    assert!(
        stderr.contains(BATCH_FALLBACK_NOTE),
        "abandoned batch was not reported:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("tests/a_broken.ch") && stderr.contains("tests/b_ok.ch"),
        "fallback note did not name the abandoned files:\nstderr={stderr}"
    );
    assert!(
        stderr.contains("reason: batch worker exited with status"),
        "fallback note did not give a reason:\nstderr={stderr}"
    );
}

#[test]
fn chelis_test_auto_batch_name_collision_uses_file_isolation() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-collision");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.CollisionFirst

def test_same() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.CollisionSecond

def test_same() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "--json", "tests/"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let rows: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(rows[0]["file"], "tests/a_first.ch");
    assert_eq!(rows[0]["test"], "test_same");
    assert_eq!(rows[0]["status"], "pass");
    assert_eq!(rows[1]["file"], "tests/b_second.ch");
    assert_eq!(rows[1]["test"], "test_same");
    assert_eq!(rows[1]["status"], "pass");
    assert_eq!(rows[2]["summary"]["passed"], 2);
}

#[test]
fn chelis_test_auto_batch_skips_module_init_files() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-module-init");
    write_file(
        &pkg.join("tests/init.ch"),
        r#"module Smoke.Tests.BatchInit

_init_failure = test_assert(false, "module init failed in fallback")

def test_one() -> unit = test_assert(true, "would pass")
"#,
    );
    write_file(
        &pkg.join("tests/ok.ch"),
        r#"module Smoke.Tests.BatchInitOk

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("module-init"),
        "module-init row missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("module init failed in fallback"),
        "module-init error missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("test_ok") && stdout.contains("PASS"),
        "batchable sibling did not pass:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn chelis_test_auto_batch_worker_abort_falls_back_to_file_workers() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-abort-fallback");
    write_file(
        &pkg.join("tests/a_first.ch"),
        r#"module Smoke.Tests.BatchAbortFirst

def test_first() -> unit = test_assert(true, "first")
"#,
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        r#"module Smoke.Tests.BatchAbortSecond

def test_second() -> unit = test_assert(true, "second")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_FORCE_BATCH_ABORT", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "batch abort should fall back to per-file workers; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("test_first") && stdout.contains("test_second"),
        "fallback did not run both files:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("2 passed, 0 failed"),
        "fallback summary missing:\nstdout={stdout}\nstderr={stderr}"
    );
    // chelis#1261: a green summary is honest here (every test really ran),
    // but it must not be the whole report. The abandoned batch is named.
    assert!(
        stderr.contains(BATCH_FALLBACK_NOTE)
            && stderr.contains("tests/a_first.ch")
            && stderr.contains("tests/b_second.ch"),
        "green fallback did not report the abandoned batch:\nstderr={stderr}"
    );
}

/// The chelis#1261 reproducer: `tests/b_local.ch` declares a local `same` over
/// lists while `tests/a_import.ch` explicitly imports the package's `same` over
/// `int64`. Under `--batch-mode auto` the two files used to be merged into one
/// compilation unit, where B's declaration captured A's import, so the batch
/// failed to compile and the whole suite silently degraded to per-file.
#[test]
fn chelis_test_auto_batch_sibling_def_does_not_capture_an_explicit_import() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-import-capture");
    write_import_collision_files(&pkg);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("test_uses_import")
            && stdout.contains("test_local_same")
            && stdout.contains("2 passed, 0 failed"),
        "both files must run and pass:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("error:"),
        "chelis#1261: the merged unit still miscompiles:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains(BATCH_FALLBACK_NOTE),
        "eligibility demotion is the sanctioned path, not an abandoned batch:\nstderr={stderr}"
    );
}

/// The reversed file order of the reproducer. Discovery is alphabetical, so
/// `a_local.ch` is admitted to the batch scope first and the *importer* is the
/// one demoted. Import-versus-declaration is the same collision either way
/// round, and only the import-then-declaration order was otherwise exercised.
#[test]
fn chelis_test_auto_batch_local_def_before_sibling_import_demotes_the_importer() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-decl-then-import");
    write_file(
        &pkg.join("tests/a_local.ch"),
        "module Smoke.Tests.LocalFirst\n\
         def same(xs: List[int64], ys: List[int64]) -> bool = eq(len(xs), len(ys))\n\
         def test_local_same() -> unit = \
         test_assert(same([cast(1, int64)], [cast(2, int64)]), \"same length\")\n",
    );
    write_file(
        &pkg.join("tests/b_import.ch"),
        "module Smoke.Tests.ImportSecond\n\
         import Smoke.Helpers (same)\n\
         def test_uses_import() -> unit = \
         test_assert(same(cast(1, int64), cast(1, int64)), \"1 == 1\")\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("error:"),
        "the merged unit still miscompiles in the reversed order:\nstderr={stderr}"
    );
    assert!(
        stdout.contains("2 passed, 0 failed"),
        "stdout={stdout}\nstderr={stderr}"
    );

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_local.ch"),
        "the declaring file should still have been batched:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("tests/b_import.ch"),
        "the importer whose name a sibling declares was not demoted:\nstderr={stderr}"
    );
}

/// ADT variant constructors share one namespace in the merged unit, so two
/// types with different names but a shared variant collide. Nothing else
/// exercises the `Decl::TypeDef` variant arm.
#[test]
fn chelis_test_auto_batch_shared_adt_variant_uses_file_isolation() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-shared-variant");
    write_file(
        &pkg.join("tests/a_first.ch"),
        "module Smoke.Tests.VariantFirst\n\
         type FirstFlag =\n  | Shared\n  | OnlyFirst\n\
         def test_first() -> unit = test_assert(true, \"first\")\n",
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        "module Smoke.Tests.VariantSecond\n\
         type SecondFlag =\n  | Shared\n  | OnlySecond\n\
         def test_second() -> unit = test_assert(true, \"second\")\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 passed, 0 failed"));

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_first.ch"),
        "the first file should still have been batched:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("tests/b_second.ch"),
        "a shared ADT variant constructor did not demote the second file:\nstderr={stderr}"
    );
}

/// A `@property` binds a top-level name too, so two files declaring the same
/// property name collide. Nothing else exercises the `Decl::Property` arm.
#[test]
fn chelis_test_auto_batch_shared_property_name_uses_file_isolation() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-shared-property");
    write_file(
        &pkg.join("tests/a_first.ch"),
        "module Smoke.Tests.PropertyFirst\n\
         @property shared_bound forall(x: int32) where x > 0:\n  x > 0\n\
         def test_first() -> unit = test_assert(true, \"first\")\n",
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        "module Smoke.Tests.PropertySecond\n\
         @property shared_bound forall(y: int32) where y > 1:\n  y > 0\n\
         def test_second() -> unit = test_assert(true, \"second\")\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 passed, 0 failed"));

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_first.ch"),
        "the first file should still have been batched:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("tests/b_second.ch"),
        "a shared property name did not demote the second file:\nstderr={stderr}"
    );
}

/// The plain-text summary line carries the degradation too. A CI job that
/// captures only stdout would otherwise read a fallback run as identical to a
/// clean one, which is chelis#1261's complaint one channel over.
#[test]
fn chelis_test_auto_batch_fallback_marks_the_plain_summary_line() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-plain-marker");
    write_file(
        &pkg.join("tests/a_first.ch"),
        "module Smoke.Tests.MarkerFirst\n\
         def test_first() -> unit = test_assert(true, \"first\")\n",
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        "module Smoke.Tests.MarkerSecond\n\
         def test_second() -> unit = test_assert(true, \"second\")\n",
    );

    let degraded = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_FORCE_BATCH_ABORT", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&degraded.stdout);
    assert_eq!(degraded.status.code(), Some(0), "stdout={stdout}");
    assert!(
        stdout.contains("2 passed, 0 failed (batch abandoned: ran per-file)"),
        "the degraded summary line is indistinguishable from a clean one:\nstdout={stdout}"
    );

    // Negative parity: a clean run's summary line is byte-identical to before.
    let clean = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let clean_stdout = String::from_utf8_lossy(&clean.stdout);
    assert!(
        clean_stdout.ends_with("\n2 passed, 0 failed\n"),
        "a clean run's summary line changed:\nstdout={clean_stdout}"
    );
}

/// `CHELIS_TEST_EXPLAIN_BATCHING` is the operator knob for the silent half:
/// demotion stays quiet by default, and names its reason when asked.
#[test]
fn chelis_test_auto_batch_explains_demotion_only_when_asked() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-explain");
    write_import_collision_files(&pkg);

    let quiet = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let quiet_stderr = String::from_utf8_lossy(&quiet.stderr);
    assert!(
        quiet_stderr.is_empty(),
        "demotion must stay silent by default:\nstderr={quiet_stderr}"
    );

    let explained = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_EXPLAIN_BATCHING", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stderr = String::from_utf8_lossy(&explained.stderr);
    assert_eq!(explained.status.code(), Some(0), "stderr={stderr}");
    assert!(
        stderr.contains("tests/b_local.ch is not in the suite batch"),
        "the demoted file was not named:\nstderr={stderr}"
    );
    assert!(
        stderr.contains("`same`")
            && stderr.contains("Smoke.Helpers")
            && stderr.contains("tests/a_import.ch"),
        "the explanation did not name the colliding name, its module, and the \
         file it collides with:\nstderr={stderr}"
    );
}

/// Membership oracle: `CHELIS_TEST_FORCE_BATCH_ABORT` kills whatever the batch
/// worker was given, and the fallback note names exactly those files. A file
/// that the collision guard demoted is therefore absent from the note.
#[test]
fn chelis_test_auto_batch_import_collision_demotes_only_the_colliding_file() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-import-demote");
    write_import_collision_files(&pkg);

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains(BATCH_FALLBACK_NOTE) && stderr.contains("tests/a_import.ch"),
        "the importing file should still have been batched:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("tests/b_local.ch"),
        "the file whose `def same` captures the sibling import was not demoted:\nstderr={stderr}"
    );
}

/// Negative parity for the guard: importing the SAME name from the SAME module
/// is agreement, not collision. Demoting it would push every suite that shares
/// one helper import onto the per-file path and delete the batch optimization.
#[test]
fn chelis_test_auto_batch_shared_import_of_one_module_stays_batched() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-shared-import");
    write_file(
        &pkg.join("tests/a_one.ch"),
        "module Smoke.Tests.One\n\
         import Smoke.Helpers (same)\n\
         def test_one() -> unit = test_assert(same(cast(1, int64), cast(1, int64)), \"one\")\n",
    );
    write_file(
        &pkg.join("tests/b_two.ch"),
        "module Smoke.Tests.Two\n\
         import Smoke.Helpers (same)\n\
         def test_two() -> unit = test_assert(same(cast(2, int64), cast(2, int64)), \"two\")\n",
    );

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_one.ch") && stderr.contains("tests/b_two.ch"),
        "a shared import must not demote either file out of the batch:\nstderr={stderr}"
    );
}

/// The same name imported from two DIFFERENT modules is a real collision: the
/// merged unit has one top-level scope and cannot hold both bindings.
#[test]
fn chelis_test_auto_batch_same_name_from_two_modules_uses_file_isolation() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-two-modules");
    write_file(
        &pkg.join("tests/a_helpers.ch"),
        "module Smoke.Tests.FromHelpers\n\
         import Smoke.Helpers (same)\n\
         def test_from_helpers() -> unit = \
         test_assert(same(cast(1, int64), cast(1, int64)), \"helpers\")\n",
    );
    write_file(
        &pkg.join("tests/b_other.ch"),
        "module Smoke.Tests.FromOther\n\
         import Smoke.Other (same)\n\
         def test_from_other() -> unit = \
         test_assert(same(cast(2, int64), cast(2, int64)), \"other\")\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 passed, 0 failed"));

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_helpers.ch"),
        "the first importer should still have been batched:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("tests/b_other.ch"),
        "an import of the same name from another module was not demoted:\nstderr={stderr}"
    );
}

/// A wildcard import brings in a name set this runner cannot enumerate without
/// resolving the package graph, so it cannot prove no sibling declaration
/// captures one of those names. It takes the per-file path.
#[test]
fn chelis_test_auto_batch_wildcard_import_uses_file_isolation() {
    let (_dir, pkg) = make_shared_name_reef_package("phase3t-smoke-batch-wildcard-import");
    write_file(
        &pkg.join("tests/a_plain.ch"),
        "module Smoke.Tests.PlainSibling\n\
         def test_plain() -> unit = test_assert(true, \"plain\")\n",
    );
    write_file(
        &pkg.join("tests/b_wildcard.ch"),
        "module Smoke.Tests.Wildcard\n\
         import Smoke.Helpers (..)\n\
         def test_wildcard() -> unit = \
         test_assert(same(cast(3, int64), cast(3, int64)), \"wildcard\")\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 passed, 0 failed"));

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_plain.ch"),
        "the sibling without a wildcard should still be batched:\nstderr={stderr}"
    );
    assert!(
        !stderr.contains("tests/b_wildcard.ch"),
        "a wildcard-importing file was not demoted:\nstderr={stderr}"
    );
}

/// The parent's eligibility classifier and the batch worker's own guard have to
/// agree on what a collision is. They did not: the worker counted a `sig` and
/// its matching `def` as one name declared twice, hard-errored, and the suite
/// fell back per-file with no report. Both now admit files through one rule.
#[test]
fn chelis_test_auto_batch_sig_beside_its_def_stays_batched() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-sig-and-def");
    write_file(
        &pkg.join("tests/a_sig.ch"),
        "module Smoke.Tests.SigAndDef\n\
         sig helper: int64 -> bool\n\
         def helper(x: int64) -> bool = eq(x, x)\n\
         def test_helper() -> unit = test_assert(helper(cast(1, int64)), \"helper\")\n",
    );
    write_file(
        &pkg.join("tests/b_plain.ch"),
        "module Smoke.Tests.SigSibling\n\
         def test_plain() -> unit = test_assert(true, \"plain\")\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        !stderr.contains(BATCH_FALLBACK_NOTE),
        "the worker guard still rejects a manifest the classifier accepted:\nstderr={stderr}"
    );

    let stderr = forced_batch_abort_stderr(&pkg);
    assert!(
        stderr.contains("tests/a_sig.ch") && stderr.contains("tests/b_plain.ch"),
        "a `sig` beside its `def` must not cost the file its batch slot:\nstderr={stderr}"
    );
}

/// Negative parity for the report: a batch that completes says nothing, on
/// either channel. The note is a degradation signal, not a banner.
#[test]
fn chelis_test_auto_batch_healthy_run_reports_no_fallback() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-healthy-quiet");
    write_file(
        &pkg.join("tests/a_first.ch"),
        "module Smoke.Tests.HealthyFirst\n\
         def test_first() -> unit = test_assert(true, \"first\")\n",
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        "module Smoke.Tests.HealthySecond\n\
         def test_second() -> unit = test_assert(true, \"second\")\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "--json", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        !stderr.contains(BATCH_FALLBACK_NOTE),
        "a healthy batch must stay quiet on stderr:\nstderr={stderr}"
    );
    assert!(
        !stdout.contains("batch_fallback"),
        "a healthy batch must emit no fallback record and no summary flag:\nstdout={stdout}"
    );
    assert!(
        stdout.contains("{\"summary\":{\"passed\":2,\"failed\":0}}"),
        "the existing summary bytes must be unchanged:\nstdout={stdout}"
    );
}

/// The `--json` shape of an abandoned batch: one additive `batch_fallback`
/// record beside the rows, plus a `batch_fallback` flag on the summary so a
/// consumer that reads only the final record still sees the degradation.
#[test]
fn chelis_test_auto_batch_fallback_json_record_and_summary_flag() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-batch-fallback-json");
    write_file(
        &pkg.join("tests/a_first.ch"),
        "module Smoke.Tests.JsonFallbackFirst\n\
         def test_first() -> unit = test_assert(true, \"first\")\n",
    );
    write_file(
        &pkg.join("tests/b_second.ch"),
        "module Smoke.Tests.JsonFallbackSecond\n\
         def test_second() -> unit = test_assert(true, \"second\")\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_FORCE_BATCH_ABORT", "1")
        .current_dir(&pkg)
        .args(["test", "--batch-mode", "auto", "--json", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "stdout={stdout}");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();

    let fallback = lines
        .iter()
        .find(|line| line.get("batch_fallback").is_some_and(|v| v.is_object()))
        .unwrap_or_else(|| panic!("no batch_fallback record in {lines:#?}"))["batch_fallback"]
        .clone();
    assert_eq!(fallback["status"], "worker-failed");
    assert!(
        fallback["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty()),
        "fallback record carried no reason: {fallback}"
    );
    assert_eq!(
        fallback["files"],
        serde_json::json!(["tests/a_first.ch", "tests/b_second.ch"])
    );

    // Additive: the rows and the summary keep their existing shape.
    assert_eq!(lines[1]["file"], "tests/a_first.ch");
    assert_eq!(lines[2]["file"], "tests/b_second.ch");
    let summary = &lines.last().expect("summary")["summary"];
    assert_eq!(summary["passed"], 2);
    assert_eq!(summary["failed"], 0);
    assert_eq!(summary["batch_fallback"], true);
}

#[test]
fn chelis_test_rejects_zero_jobs() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-zero-jobs");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.Pass

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--jobs", "0", "tests/"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("jobs"));
}

#[test]
fn chelis_test_missing_tests_dir_exits_two() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-nodir");
    // Remove the tests dir so the default path does not resolve.
    fs::remove_dir_all(pkg.join("tests")).expect("remove tests dir");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("does not exist"));
}

#[test]
fn chelis_test_non_reef_context_exits_two() {
    // Fresh tempdir with NO reef.toml.
    let dir = tempdir().expect("tempdir");
    let tests = dir.path().join("tests");
    fs::create_dir_all(&tests).expect("mkdir tests");
    write_file(
        &tests.join("empty.ch"),
        "def test_nothing() -> unit = test_assert(true, \"noop\")\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["test", "tests/"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("reef"));
}

// === RT3 regression tests ===

#[test]
fn chelis_test_empty_tests_dir_exits_zero_with_zero_zero_summary() {
    // RT3 H5: empty `tests/` must exit 0 with "0 passed, 0 failed", not
    // exit 2. Matches `cargo test` and `pytest` ergonomics — a fresh
    // package with no tests yet is a legitimate state, not a runner error.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-empty");
    // tests/ exists but has no .ch files. Ensure it is truly empty.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .code(0)
        .stdout(predicate::str::contains("0 passed, 0 failed"));
}

#[test]
fn chelis_test_hidden_dotfile_is_skipped() {
    // RT3 M2: editor temp files and other dot-prefixed files must not be
    // enumerated as tests — avoids accidentally running an editor's backup
    // or a hidden fixture.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-dotfile");
    write_file(
        &pkg.join("tests/.secret.ch"),
        r#"module Smoke.Tests.Hidden

def test_should_not_run() -> unit = test_assert(false, "hidden should be skipped")
"#,
    );
    // A real test file alongside the dotfile so we still have something to run.
    write_file(
        &pkg.join("tests/visible.ch"),
        r#"module Smoke.Tests.Visible

def test_visible() -> unit = test_assert(true, "ok")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_visible"))
        .stdout(predicate::str::contains("test_should_not_run").not())
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

#[test]
fn chelis_test_non_unit_returning_def_is_not_enumerated() {
    // RT3 H4: `def test_x() -> bool = true` is a nullary function but is not
    // a test because its result is non-unit. The
    // enumerator must skip it so the real `def test_real() -> unit`
    // alongside it runs and passes.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-non-unit");
    write_file(
        &pkg.join("tests/mixed.ch"),
        r#"module Smoke.Tests.Mixed

def test_x() -> bool = true

def test_real() -> unit = test_assert(true, "real test runs")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The real test must PASS. The value-binding `test_x` must not appear
    // as its own row — enumerator must have skipped it.
    assert!(
        stdout.contains("test_real"),
        "test_real missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("PASS"),
        "PASS missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    // `test_x` must not appear as an enumerated test row — a cascade row
    // (e.g., `test_x ... FAIL (module-init failed)`) would imply it was
    // enumerated. Check that the only lines mentioning `test_x` (if any)
    // are not test-result rows.
    for line in stdout.lines() {
        if line.contains("test_x") {
            assert!(
                !line.contains("PASS") && !line.contains("FAIL"),
                "test_x appeared as a test row; got:\nstdout={stdout}\nstderr={stderr}"
            );
        }
    }
    assert!(
        stdout.contains("1 passed, 0 failed"),
        "summary missing from:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(output.status.success(), "expected exit 0");
}

#[test]
fn chelis_test_duplicate_test_name_is_reported_as_file_level_error() {
    // RT3 H3: two `def test_foo()` in the same file is Chelis-level
    // shadowing and will execute the second body silently. Test runner
    // must surface this as a fatal enumeration error attributed to the
    // file, not pretend both ran.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-dup");
    write_file(
        &pkg.join("tests/dup.ch"),
        r#"module Smoke.Tests.Dup

def test_foo() -> unit = test_assert(true, "first")

def test_foo() -> unit = test_assert(false, "second")
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("duplicate test definition"));
}

#[test]
fn chelis_test_module_level_bad_binding_cascades_briefly() {
    // RT3 H2 variant: a failing module-level binding (a `let _bad = ...`
    // that throws during eval) surfaces as a single module-init row with
    // the full error detail, and per-test rows only carry the brief
    // "module-init failed" cascade marker — the full error text is NOT
    // repeated per test. This exercises the module-init pre-check path.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-mod-init-cascade");
    write_file(
        &pkg.join("tests/bad.ch"),
        r#"module Smoke.Tests.Bad

_init_failure = test_assert(false, "module init broken")

def test_one() -> unit = test_assert(true, "would have passed 1")

def test_two() -> unit = test_assert(true, "would have passed 2")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The full module-init error detail must appear exactly once across
    // all rows — not repeated per test (that was the cascade bug).
    let full_err_count = stdout.matches("module init broken").count();
    assert_eq!(
        full_err_count, 1,
        "full module-init error should appear once, not per-test; got:\n{stdout}"
    );
    // Per-test rows carry the brief marker.
    let cascade_count = stdout.matches("module-init failed").count();
    assert_eq!(
        cascade_count, 2,
        "two tests should each carry the brief cascade marker; got:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn chelis_test_module_init_failure_surfaces_as_module_init_row_and_cascades() {
    // RT3 H1: a top-level `let _ = assert_*(...)` that fails must surface
    // as its own `module-init` row and cascade every discovered test to
    // FAIL with a "module-init failed" message. Matches the spec's
    // worked output example.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-moduleinit");
    write_file(
        &pkg.join("tests/init.ch"),
        r#"module Smoke.Tests.Init

_sanity = test_assert(false, "module init fails here")

def test_one() -> unit = test_assert(true, "would have passed one")

def test_two() -> unit = test_assert(true, "would have passed two")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("module-init"),
        "module-init row missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("test_one") && stdout.contains("test_two"),
        "cascade-failed tests missing from:\n{stdout}"
    );
    // All three rows are FAIL (module-init plus two cascades).
    let fail_count = stdout.matches("FAIL").count();
    assert!(
        fail_count >= 3,
        "expected module-init + 2 cascades = 3+ FAIL rows; got {fail_count} in:\n{stdout}"
    );
    assert!(
        stdout.contains("module-init failed"),
        "cascade message missing from:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

// === Phase 3t.5 extras: richer failure-mode coverage ===

#[test]
fn chelis_test_infinite_recursion_times_out_and_suite_continues() {
    // 3t.5 coverage: an infinite loop in one test should surface as a FAIL
    // carrying the timeout message, not hang the suite. A sibling passing
    // test must still run and pass so the runner is proven to proceed past
    // the timed-out worker. The timeout is the `--timeout` override so we do
    // not have to wait the default 30s.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-timeout");
    write_file(
        &pkg.join("tests/loopy.ch"),
        r#"module Smoke.Tests.Loopy

-- A fold over a multi-million-element range is a stack-safe way to trip
-- the per-test timeout without blowing the worker stack (native
-- recursion with no base case overflows the 32 MB test-worker stack
-- before the timeout deadline fires). 10_000_000 iterations consistently
-- exceeds the --timeout 2 budget used below.
def test_infinite() -> unit = test_assert(eq(fold(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), range(cast(0, int64), cast(10000000, int64))), cast(0, int64)), "never")

def test_quick() -> unit = test_assert(true, "quick")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "--timeout", "2", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The looping test must appear as FAIL with a timeout-flavored message.
    assert!(
        stdout.contains("test_infinite"),
        "test_infinite missing from:\n{stdout}"
    );
    assert!(
        stdout.contains("timeout"),
        "timeout message missing from:\n{stdout}"
    );
    // The sibling test must have PASSed — the runner must not bail after one
    // timed-out test.
    assert!(
        stdout.contains("test_quick"),
        "test_quick missing from:\n{stdout}"
    );
    let pass_count = stdout.matches("PASS").count();
    assert!(
        pass_count >= 1,
        "expected at least one PASS (test_quick); got {pass_count} in:\n{stdout}"
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn chelis_test_missing_import_is_reported_as_file_level_error() {
    // 3t.5 coverage: a test file that imports a module that does not exist
    // in the reef graph must report a file-level error row and continue to
    // the next file. Specifically: the compile error must surface, and the
    // runner must exit 1 (test failure) — not exit 2 (runner error), because
    // the file was enumerated successfully; it is one test file's content
    // that is broken, not the runner state.
    let (_dir, pkg) = make_reef_package("phase3t-smoke-missing-import");
    write_file(
        &pkg.join("tests/ghost.ch"),
        r#"module Smoke.Tests.Ghost

import Nonexistent.Module (imaginary_helper)

def test_uses_ghost() -> unit = test_assert(true, "would never compile")
"#,
    );
    // Sibling file that is fine — runner must keep going past the broken
    // file, exercise this one, and still exit 1 because of the first.
    write_file(
        &pkg.join("tests/ok.ch"),
        r#"module Smoke.Tests.Ok

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The broken file must produce a FAIL row that references either the
    // compile error or the unresolved module name.
    assert!(
        stdout.contains("ghost.ch"),
        "ghost.ch missing from output:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("FAIL"),
        "FAIL row missing from:\nstdout={stdout}"
    );
    // The healthy sibling file must still report its PASS row.
    assert!(
        stdout.contains("test_ok"),
        "sibling test_ok missing from:\nstdout={stdout}"
    );
    assert!(
        stdout.contains("PASS"),
        "PASS row missing from:\nstdout={stdout}"
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1 (failing tests), got {:?}\nstdout={stdout}\nstderr={stderr}",
        output.status.code()
    );
}

/// Bucket 6c: `chelis test path/to/file.ch` must succeed when invoked from
/// any cwd, by walking up the target file's ancestry to find `reef.toml`.
/// The pre-fix behavior required the cwd to be inside the reef package,
/// which forced workflows like `cd packages/foo && chelis test tests/x.ch`.
#[test]
fn chelis_test_resolves_reef_root_from_target_file_path() {
    let (_dir, pkg) = make_reef_package("phase3t-smoke-cwd-anywhere");
    write_file(
        &pkg.join("tests/pass.ch"),
        r#"module Smoke.Tests.Pass

def test_from_anywhere() -> unit = test_assert(true, "ok")
"#,
    );
    let test_file = pkg.join("tests/pass.ch");

    // Use the OS temp dir as the cwd. The temp-dir boundary in
    // `find_package_root_from_dir` ensures we don't accidentally pick up a
    // stray reef.toml from /tmp; we want the resolution to come from the
    // *target file's* directory walk, not from the cwd.
    let foreign_cwd = std::env::temp_dir();
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&foreign_cwd)
        .args(["test", test_file.to_str().expect("utf-8 path")])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_from_anywhere"))
        .stdout(predicate::str::contains("PASS"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}

/// Bucket 6c negative parity: `chelis test some_file.ch` from a foreign
/// cwd, where the target also has no reef.toml in its ancestry, must
/// produce the "no reef.toml found" error rather than silently succeeding
/// or panicking.
#[test]
fn chelis_test_errors_clearly_when_no_reef_anywhere() {
    let dir = tempdir().expect("tempdir");
    let test_file = dir.path().join("orphan.ch");
    write_file(
        &test_file,
        "module Orphan\n\ndef test_x() -> unit = test_assert(true, \"x\")\n",
    );
    let foreign_cwd = std::env::temp_dir();
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&foreign_cwd)
        .args(["test", test_file.to_str().expect("utf-8 path")])
        .output()
        .expect("run");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "expected failure when no reef found; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("reef.toml") || stdout.contains("reef.toml"),
        "expected reef.toml mention in error; stdout={stdout}\nstderr={stderr}"
    );
}

/// RT-1 F2-bypass (RFC v6): a reef package test file may NOT use a
/// hand-authored reef-internal mangled name (`pkg__<pkg>__<Module>__<Name>`)
/// that self-keys to a victim module and forges the real opaque type.
/// The `chelis test` entry path re-asserts the linked flag without
/// re-mangling user test/entry decl names, so the forge def self-keys to
/// `Demo.Types` and constructs the opaque type as if in-module. eval /
/// check / build / validate all reject the same forge; this was test-only.
fn make_opaque_reef_package(dir_name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(dir_name);
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            "[package]\nname = \"{dir_name}\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"Smoke\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &pkg.join("src/types.ch"),
        "module Smoke.Types\n\
         export (probability, prob_value)\n\
         @opaque\n\
         type Probability =\n  | Probability { value: f32 }\n\
         def probability(x: f32) -> Probability = Probability { value: x }\n\
         def prob_value(p: Probability) -> f32 = p.value\n",
    );
    (dir, pkg)
}

#[test]
fn chelis_test_rejects_reef_mangled_name_forge() {
    // The package name is `forgepkg`, so a def in module Smoke.Types
    // links to `pkg__forgepkg__Smoke__Types__<name>`. A test file that
    // hand-authors `def pkg__forgepkg__Smoke__Types__forge` self-keys to
    // the victim module `forgepkg.Smoke.Types`, so `Probability { ... }`
    // would be in-module. `chelis test` must REJECT this forge.
    let (_dir, pkg) = make_opaque_reef_package("forgepkg");
    write_file(
        &pkg.join("tests/forge.ch"),
        "module Smoke.Tests.Forge\n\
         import Smoke.Types (prob_value)\n\
         def pkg__forgepkg__Smoke__Types__forge(x: f32) -> f32 = {\n\
         \x20 p = Probability { value: x }\n\
         \x20 prob_value(p)\n\
         }\n\
         def test_forge() -> unit = \
         test_assert(pkg__forgepkg__Smoke__Types__forge(0.5) >= 0.0, \"forge\")\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains("1 passed, 0 failed"),
        "the forge must NOT pass `chelis test`; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("reserved internal-name format")
            || stdout.contains("ReservedLinkerName")
            || stdout.contains("outside its defining module"),
        "the forge must be rejected with a module-identity / reserved-name error; \
         stdout={stdout}\nstderr={stderr}"
    );
}

#[test]
fn chelis_test_legit_opaque_package_test_still_passes() {
    // No-regression: a LEGITIMATE test file (exported producers, normal
    // helper names) in the same @opaque package must still pass.
    let (_dir, pkg) = make_opaque_reef_package("legitpkg");
    write_file(
        &pkg.join("tests/use.ch"),
        "module Smoke.Tests.Use\n\
         import Smoke.Types (probability, prob_value)\n\
         def round_trip(x: f32) -> f32 = prob_value(probability(x))\n\
         def test_round_trip() -> unit = test_assert(round_trip(0.5) >= 0.0, \"ok\")\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test_round_trip"))
        .stdout(predicate::str::contains("1 passed, 0 failed"));
}
