//! Red-team: `CompiledContext` disk-cache key collides across packages
//! that share name + version + source content but live at different
//! paths.
//!
//! Background. PR #127 wired a cross-process compiled-context disk
//! cache (`chelis_compiler_api::load_or_compile_for_package`). The
//! cache key is `(root_package_name, root_package_version,
//! source_hash)` where `source_hash` is a digest of the source file
//! *contents* only -- the package's location on disk is not part of
//! the key, and `CompiledContext::load_if_fresh` does not re-bind the
//! decoded `reef_state().package_root` to the live `package_dir`.
//!
//! Consequence. Two reef packages with the same `name`/`version` in
//! `reef.toml` and byte-identical sources, built from two different
//! directories, produce the same cache key. The second build gets a
//! cache HIT whose `CompiledContext.reef_state().package_root` still
//! points at the FIRST package's (by then often deleted) directory.
//! `chelis test`'s Phase H worker has a path-equality guard
//! (`package_root ... does not match worker cwd ...; refusing to run`)
//! that then correctly REFUSES to run -- so `chelis test` fails with a
//! mismatched-library-context error on a perfectly valid package, just
//! because a content-identical package was tested earlier.
//!
//! This is exactly why the `chelis test` integration suites (the
//! `test_command_smoke`, `subprocess_isolation`, and `std_test_module`
//! files) fail non-deterministically once the user's
//! `~/.cache/chelis/compiled/`
//! has been warmed by an earlier run: the `make_minimal_reef_with_test`
//! helpers write byte-identical sources every time.
//!
//! Each test below points `XDG_CACHE_HOME` at its own private tempdir
//! so the repro is hermetic (it does not read or pollute the real user
//! cache) and deterministic (the collision is forced within the
//! private cache, not dependent on prior global state).
//!
//! Status. These two tests REPRODUCE a live bug: they FAIL against the
//! current tree and PASS once the cache key (or
//! `CompiledContext::load_if_fresh`) accounts for the package
//! directory. They are `#[ignore]`-d so they do not red the per-PR
//! gate while the bug is open; the manual repro command is:
//!
//! ```text
//! cargo test -p chelis-cli \
//!   --test redteam_compiled_context_cache_collision -- --ignored
//! ```
//!
//! When the bug is fixed, remove the `#[ignore]` so the partition stays
//! locked. Owning issue: compiled-context disk cache key omits the
//! package path (PR #127 follow-up).

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Write a minimal reef package with one test file under `parent`,
/// using a FIXED package name + version + byte-identical sources so two
/// such packages built in different directories collide on the
/// content-only cache key.
fn write_identical_reef(parent: &Path) -> PathBuf {
    let pkg = parent.join("collide-pkg");
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    fs::write(
        pkg.join("reef.toml"),
        format!(
            "[package]\n\
             name = \"collide-pkg\"\n\
             version = \"0.1.0\"\n\
             compiler = \"={ver}\"\n\
             module_prefix = \"Collide\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    )
    .expect("write reef.toml");
    fs::write(
        pkg.join("src/main.ch"),
        "module Collide.Main\n\n\
         def noop() -> () = test_assert(true, \"noop\")\n",
    )
    .expect("write main.ch");
    fs::write(
        pkg.join("tests/foo.ch"),
        "module Collide.Tests.Foo\n\n\
         def test_one() -> () = test_assert(true, \"first\")\n",
    )
    .expect("write tests/foo.ch");
    pkg
}

/// Run `chelis test tests/` in `pkg`, with `XDG_CACHE_HOME` pointed at
/// `cache_home` so every invocation shares one private compiled-context
/// cache. Returns `(exit_code, stdout, stderr)`.
fn run_chelis_test(pkg: &Path, cache_home: &Path) -> (Option<i32>, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("XDG_CACHE_HOME", cache_home)
        // Make sure a developer-set CHELIS_REEF_HOME does not steer the
        // cache somewhere else and break the hermetic isolation.
        .env_remove("CHELIS_REEF_HOME")
        .args(["test", "tests/"])
        .output()
        .expect("run chelis test");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Core repro. Build + test two content-identical packages from two
/// different directories, sharing one private compiled-context cache.
/// The first run warms the cache; the second run must STILL pass. It
/// currently fails with a `does not match worker cwd` error because the
/// cache hit hands back a `CompiledContext` bound to the first
/// package's directory.
#[test]
#[ignore = "reproduces an open bug: compiled-context cache key omits the package path (PR #127 follow-up)"]
fn second_identical_package_in_fresh_dir_is_not_poisoned_by_cache_hit() {
    let cache_dir = TempDir::new().expect("cache tempdir");

    let first_home = TempDir::new().expect("first pkg tempdir");
    let first_pkg = write_identical_reef(first_home.path());
    let (first_code, first_stdout, first_stderr) = run_chelis_test(&first_pkg, cache_dir.path());
    assert_eq!(
        first_code,
        Some(0),
        "first `chelis test` run should pass (cold cache)\n\
         stdout=\n{first_stdout}\nstderr=\n{first_stderr}"
    );

    // Second package: different directory, byte-identical sources, same
    // package name/version -> same content-only cache key.
    let second_home = TempDir::new().expect("second pkg tempdir");
    let second_pkg = write_identical_reef(second_home.path());
    assert_ne!(
        first_pkg, second_pkg,
        "the two packages must live at different paths for the repro"
    );
    let (second_code, second_stdout, second_stderr) =
        run_chelis_test(&second_pkg, cache_dir.path());

    assert_eq!(
        second_code,
        Some(0),
        "second `chelis test` run on a content-identical package in a \
         DIFFERENT directory must also pass. It currently fails because \
         the compiled-context cache key omits the package path, so the \
         second run gets a cache hit whose `package_root` points at the \
         first (now-stale) package directory.\n\
         stdout=\n{second_stdout}\nstderr=\n{second_stderr}"
    );
    assert!(
        !second_stderr.contains("does not match worker cwd")
            && !second_stdout.contains("does not match worker cwd"),
        "second run hit the mismatched-library-context guard, i.e. the \
         cache returned a CompiledContext bound to the first package's \
         directory.\nstdout=\n{second_stdout}\nstderr=\n{second_stderr}"
    );
    assert!(
        second_stdout.contains("test_one") && second_stdout.contains("PASS"),
        "second run should report test_one PASS like the first.\n\
         stdout=\n{second_stdout}"
    );
}

/// Narrower restatement: re-running `chelis test` on a package whose
/// directory was recreated (same content, new path) must be idempotent.
/// This is the single-package shape of the same bug -- a developer who
/// `rm -rf`s and recreates a scratch package, or a test harness that
/// uses a fresh `tempdir()` each run, hits it.
#[test]
#[ignore = "reproduces an open bug: compiled-context cache key omits the package path (PR #127 follow-up)"]
fn rerun_after_recreating_package_dir_stays_green() {
    let cache_dir = TempDir::new().expect("cache tempdir");

    // Run #1 in dir A.
    let home_a = TempDir::new().expect("tempdir A");
    let pkg_a = write_identical_reef(home_a.path());
    let (code_a, out_a, err_a) = run_chelis_test(&pkg_a, cache_dir.path());
    assert_eq!(
        code_a,
        Some(0),
        "run #1 should pass\nstdout=\n{out_a}\nstderr=\n{err_a}"
    );
    drop(home_a); // the original package directory goes away

    // Run #2 in a fresh dir B with identical content.
    let home_b = TempDir::new().expect("tempdir B");
    let pkg_b = write_identical_reef(home_b.path());
    let (code_b, out_b, err_b) = run_chelis_test(&pkg_b, cache_dir.path());
    assert_eq!(
        code_b,
        Some(0),
        "run #2 on a recreated, content-identical package must pass; a \
         cache hit must not bind tests to a deleted directory.\n\
         stdout=\n{out_b}\nstderr=\n{err_b}"
    );
}
