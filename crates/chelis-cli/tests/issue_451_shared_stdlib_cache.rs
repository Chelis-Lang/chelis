//! Acceptance oracle for chelis#451: the `chelis test` and
//! `chelis eval --file` paths reuse the cross-process chelis-std typecheck
//! cache so the chelis-std library is not re-compiled per fresh process.
//!
//! Before this change, `compile_reef_context` (the package compile every
//! `chelis test` parent and `chelis eval --file` invocation runs) built the
//! WHOLE library — chelis-std included — monolithically on every fresh
//! process. A sharded test matrix (one `chelis test <file>` process per
//! matrix leg) or any package-source edit re-paid the chelis-std typecheck
//! each time. The fix routes `compile_reef_context` through the same
//! `StdLibContext` cache `chelis check` / `chelis build` already use.
//!
//! A stale or wrong cache hit on the TEST path is the worst possible
//! outcome — it would silently report wrong pass/fail. This file pins the
//! non-negotiable correctness properties, mirroring
//! `stdlib_typecheck_cache_oracle.rs` for the check/build paths:
//!
//! 1. `test_cold_vs_warm_byte_identical` /
//!    `eval_cold_vs_warm_byte_identical` — a cold-cache run and a
//!    warm-cache run produce byte-identical stdout. A cache hit must not
//!    perturb the result.
//! 2. `test_monolithic_vs_layered_byte_identical` /
//!    `eval_monolithic_vs_layered_byte_identical` — the monolithic path
//!    (`CHELIS_STDLIB_CACHE_DISABLE=1`) and the layered `_with_context`
//!    path produce byte-identical stdout. Independent of cache hit/miss:
//!    proves the monolithic->layered re-route in `compile_reef_context` is
//!    itself correctness-preserving, for both a passing and a FAILING
//!    test.
//! 3. `test_source_edit_warm_stdlib_still_correct` — editing the package
//!    source (so the whole-package `CompiledContext` disk cache MISSES)
//!    while chelis-std is unchanged (so the `StdLibContext` sub-cache HITS)
//!    still produces the correct result. This is the exact sharded-matrix /
//!    dev-loop scenario the fix targets, and the one the pre-existing
//!    whole-package cache could not cover.
//!
//! `CHELIS_STDLIB_CACHE_DISABLE=1` is the test seam for property 2 and is
//! never set in production CI.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Recursively copy a directory tree. Mirrors the helper in
/// `stdlib_typecheck_cache_oracle.rs`; duplicated rather than shared
/// because integration-test files are separate compilation units.
fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dst dir");
    for entry in fs::read_dir(src).expect("read_dir src") {
        let entry = entry.expect("dir entry");
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to);
        } else {
            fs::copy(&from, &to).expect("copy file");
        }
    }
}

/// Copy the `pseudo_nautilus` fixture (a reef package that imports the
/// bundled chelis-std and ships both `src/` and `tests/`) into `scratch`
/// and return the staged package root. Staging is required because
/// `chelis test` writes `reef.lock` next to the resolved `reef.toml`;
/// running against the real fixture path leaks the lockfile into the
/// source tree.
fn stage_fixture(scratch: &Path) -> PathBuf {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pseudo_nautilus")
        .canonicalize()
        .expect("pseudo_nautilus fixture must exist");
    let dst = scratch.join("pseudo_nautilus");
    copy_dir_recursive(&src, &dst);
    dst.canonicalize().expect("staged package root must exist")
}

/// Fresh, empty cache directory so "cold" is genuinely cold and tests do
/// not share cache state across the process-global `CHELIS_REEF_HOME`.
fn fresh_cache_home() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let home = dir.path().join("reef-home");
    fs::create_dir_all(&home).expect("mkdir reef-home");
    (dir, home)
}

/// Run `chelis test <pkg>/tests --json` (or another subcommand form) with
/// the cache isolated to `cache_home`. Returns raw stdout bytes.
/// `extra_env` carries the optional `CHELIS_STDLIB_CACHE_DISABLE` seam.
fn run_test_json(pkg_root: &Path, cache_home: &Path, extra_env: &[(&str, &str)]) -> Vec<u8> {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.current_dir(pkg_root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    // `--jobs 1` keeps worker scheduling deterministic so the NDJSON row
    // order is stable for the byte-identical comparison.
    cmd.arg("test")
        .arg("tests")
        .arg("--json")
        .arg("--jobs")
        .arg("1");
    cmd.assert().get_output().stdout.clone()
}

/// Run `chelis eval --file <file> --json` (the Phase-K eval-in-context
/// path that also builds a `CompiledContext`). Returns raw stdout bytes.
fn run_eval_json(file: &Path, cache_home: &Path, extra_env: &[(&str, &str)]) -> Vec<u8> {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.arg("eval").arg("--file").arg(file).arg("--json");
    cmd.assert().get_output().stdout.clone()
}

// ---------------------------------------------------------------------
// Property 1: cold-cache vs warm-cache byte-identical (`chelis test`).
// ---------------------------------------------------------------------

#[test]
fn test_cold_vs_warm_byte_identical() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());
    // Cold: empty cache, this run computes + writes the chelis-std artifact.
    let cold = run_test_json(&pkg, &cache_home, &[]);
    // Warm: same cache, the chelis-std sub-context now exists — must hit.
    let warm = run_test_json(&pkg, &cache_home, &[]);
    assert!(!cold.is_empty(), "cold `chelis test` produced no output");
    assert_eq!(
        cold, warm,
        "cold-cache vs warm-cache `chelis test` output diverged: a chelis-std \
         cache hit must never change pass/fail or diagnostics"
    );
}

// ---------------------------------------------------------------------
// Property 1b: cold-cache vs warm-cache byte-identical (`chelis eval`).
// ---------------------------------------------------------------------

#[test]
fn eval_cold_vs_warm_byte_identical() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());
    let entry = pkg.join("src/special.ch");
    let cold = run_eval_json(&entry, &cache_home, &[]);
    let warm = run_eval_json(&entry, &cache_home, &[]);
    assert!(!cold.is_empty(), "cold `chelis eval` produced no output");
    assert_eq!(
        cold, warm,
        "cold-cache vs warm-cache `chelis eval --file` output diverged: a \
         chelis-std cache hit must never perturb output"
    );
}

// ---------------------------------------------------------------------
// Property 2: monolithic vs layered byte-identical (`chelis test`),
// for a passing suite. Independent of cache hit/miss: proves the
// monolithic->layered re-route in `compile_reef_context` is itself
// correctness-preserving.
// ---------------------------------------------------------------------

#[test]
fn test_monolithic_vs_layered_byte_identical() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());
    // Monolithic: cache disabled, `compile_reef_context` runs the whole-
    // library build over chelis-std + the package in one pass.
    let monolithic = run_test_json(&pkg, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    // Layered: cache enabled, `compile_reef_context` reuses the cached
    // chelis-std sub-context and checks only the package decls.
    let layered = run_test_json(&pkg, &cache_home, &[]);
    assert!(
        !monolithic.is_empty(),
        "monolithic `chelis test` produced no output"
    );
    assert_eq!(
        monolithic, layered,
        "monolithic vs layered `chelis test` output diverged: a divergence here \
         is a compiler-correctness bug in the `_with_context` re-route, NOT a \
         cache bug"
    );
}

// ---------------------------------------------------------------------
// Property 2b: monolithic vs layered byte-identical for a FAILING test.
// The warm==cold contract must hold for failure diagnostics, not just
// the green path — a cache that perturbed a failure message would be a
// silent miscompilation of the test result.
// ---------------------------------------------------------------------

#[test]
fn test_failing_monolithic_vs_layered_byte_identical() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());
    // Add a test that fails deterministically: erf(0) is ~0, asserting it
    // equals 999 with a tiny tolerance fails with a stable message.
    let tests_dir = pkg.join("tests");
    fs::write(
        tests_dir.join("intentional_fail.ch"),
        "module PseudoNautilus.Tests.IntentionalFail\n\
         import PseudoNautilus.Special (erf_approx)\n\
         import Std.Test (assert_close)\n\
         def test_intentional_fail() -> () ! { Test } = \
         assert_close(erf_approx(cast(0.0, f32)), cast(999.0, f32), cast(0.00001, f32), \"intentional\")\n",
    )
    .expect("write failing test");

    let monolithic = run_test_json(&pkg, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let layered = run_test_json(&pkg, &cache_home, &[]);
    // The failing test must be present in the output (guards against the
    // suite silently not running it).
    assert!(
        String::from_utf8_lossy(&layered).contains("test_intentional_fail"),
        "the intentionally-failing test must appear in the report"
    );
    assert!(
        String::from_utf8_lossy(&layered).contains("\"fail\""),
        "the report must contain a failing row"
    );
    assert_eq!(
        monolithic, layered,
        "monolithic vs layered `chelis test` output diverged on a FAILING test: \
         the cache must reproduce failure diagnostics byte-for-byte"
    );
}

// ---------------------------------------------------------------------
// Property 2c: monolithic vs layered byte-identical (`chelis eval`).
// ---------------------------------------------------------------------

#[test]
fn eval_monolithic_vs_layered_byte_identical() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());
    let entry = pkg.join("src/special.ch");
    let monolithic = run_eval_json(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let layered = run_eval_json(&entry, &cache_home, &[]);
    assert!(
        !monolithic.is_empty(),
        "monolithic `chelis eval` produced no output"
    );
    assert_eq!(
        monolithic, layered,
        "monolithic vs layered `chelis eval --file` output diverged: \
         compiler-correctness divergence in the `_with_context` re-route"
    );
}

// ---------------------------------------------------------------------
// Property 3: the sharded-matrix / dev-loop scenario. Edit the package
// source so the whole-package `CompiledContext` disk cache MISSES, while
// chelis-std is unchanged so the `StdLibContext` sub-cache HITS. The
// result must stay correct AND byte-identical to the monolithic path —
// this is the case the pre-existing whole-package cache could not cover
// and the exact reason chelis#451 exists.
// ---------------------------------------------------------------------

#[test]
fn test_source_edit_warm_stdlib_still_correct() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());

    // First run warms the chelis-std sub-context (and writes a whole-package
    // CompiledContext for the original source).
    let first = run_test_json(&pkg, &cache_home, &[]);
    assert!(!first.is_empty(), "first `chelis test` produced no output");

    // Edit the package source: append a comment to a source file. This
    // changes the whole-package content hash (CompiledContext MISS) but not
    // chelis-std (StdLibContext HIT). The test set is unchanged, so the
    // pass/fail summary must be identical to a fresh monolithic run.
    let special = pkg.join("src/special.ch");
    let mut body = fs::read_to_string(&special).expect("read special.ch");
    body.push_str("\n-- chelis#451 oracle: source edit to bust the package cache\n");
    fs::write(&special, &body).expect("rewrite special.ch");

    // Layered run on the edited source: CompiledContext miss, StdLib hit.
    let layered_after_edit = run_test_json(&pkg, &cache_home, &[]);
    // Monolithic run on the same edited source, fresh cache home: no caches
    // at all. This is the ground-truth result.
    let (mono_guard, mono_home) = fresh_cache_home();
    let _ = mono_guard;
    let monolithic = run_test_json(&pkg, &mono_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);

    assert_eq!(
        layered_after_edit, monolithic,
        "after a package-source edit (CompiledContext miss, StdLibContext hit), \
         the layered result must match the monolithic ground truth byte-for-byte"
    );
}

// ---------------------------------------------------------------------
// Property 4: compile-ERROR parity. `build_library_triple_layered`
// returns `None` (monolithic fallback) when the package decls do not
// type-check clean, and documents that the monolithic path then produces
// the byte-identical error report. Drive an actual type error through the
// layered path and confirm the diagnostic matches the monolithic path,
// exercising the `Err(_) => None` fall-through (the green-path tests above
// never reach it).
// ---------------------------------------------------------------------

/// Run `chelis check <file>` with the cache isolated to `cache_home`.
/// `check` emits its JSON fitness report (including the `errors` array) on
/// stdout and exits nonzero on a type error, so capture stdout regardless
/// of exit status. (`check` takes no `--json` flag — the report is always
/// JSON.)
fn run_check(file: &Path, cache_home: &Path, extra_env: &[(&str, &str)]) -> Vec<u8> {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.arg("check").arg(file);
    cmd.assert().get_output().stdout.clone()
}

#[test]
fn check_compile_error_monolithic_vs_layered_byte_identical() {
    let (guard, cache_home) = fresh_cache_home();
    let pkg = stage_fixture(guard.path());
    // Introduce a deterministic TYPE error in a package source file: call a
    // chelis-std assert with a wrong-arity / wrong-type argument so the
    // package decls fail to type-check `_with_context`. `erf_approx` takes
    // one `f32`; passing a string literal is a hard type error.
    let bad = pkg.join("src/bad.ch");
    fs::write(
        &bad,
        "module PseudoNautilus.Bad\n\
         import PseudoNautilus.Special (erf_approx)\n\
         def broken() -> f32 = erf_approx(\"not a number\")\n",
    )
    .expect("write type-broken source");

    // Layered: cache enabled. The package decls fail to compose against the
    // cached chelis-std sub-context, so `build_library_triple_layered`
    // returns None and `compile_reef_context` falls back to the monolithic
    // build, which produces the diagnostic.
    let layered = run_check(&bad, &cache_home, &[]);
    // Monolithic: cache disabled, the whole-library build surfaces the same
    // diagnostic directly.
    let monolithic = run_check(&bad, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);

    let layered_str = String::from_utf8_lossy(&layered);
    assert!(
        !layered.is_empty(),
        "layered `chelis check` produced no output"
    );
    // Guard against the type error silently not being reported: a clean
    // report would have an empty `errors` array. Pin that a PrecisionMismatch
    // error surfaced; the exact byte shape is asserted below.
    assert!(
        layered_str.contains("PrecisionMismatch"),
        "the injected type error must surface in the check report; got: {layered_str}"
    );
    assert_eq!(
        layered, monolithic,
        "monolithic vs layered `chelis check` error report diverged: the \
         layered `Err(_) => None` fall-through must reproduce the monolithic \
         diagnostic byte-for-byte"
    );
}
