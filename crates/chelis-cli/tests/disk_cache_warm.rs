//! Phase K disk-cache wire-up — integration probes for the
//! `load_or_compile_for_package` plumbing inside `cmd_eval` (and eventually
//! `cmd_check` / `cmd_test`).
//!
//! What the CLI tests need to lock in:
//!
//! 1. Two consecutive `chelis eval --file` invocations against the same
//!    reef package, with `CHELIS_REEF_HOME` pointed at an empty tempdir,
//!    produce byte-identical stdout. The first run misses the cache and
//!    saves; the second hits and skips the library compile.
//! 2. After the second run, the cache directory contains exactly one
//!    `.ctx` file matching the `<pkg>-<version>-<hash16>.ctx` shape.
//! 3. Editing a source file invalidates the cache key (different
//!    `source_hash` → different filename), and the new run still
//!    succeeds. The pre-edit cache file remains on disk (Phase K
//!    intentionally does not garbage-collect; that's a follow-up).
//! 4. A `--filter __no_match__`-style invocation (none here, since this
//!    test focuses on `cmd_eval`) is irrelevant to the disk cache wire-
//!    up; we keep the test surface scoped to `cmd_eval` to minimize
//!    flake risk.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// Build a reef package with a path-dep `mylib` exporting pure-int
/// helpers. Mirrors the Phase H fixture so the cache test exercises the
/// same dispatch as `eval_in_reef_context.rs`.
fn path_dep_package() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"

[dependencies]
mylib = {{ path = "./mylib" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder() -> i32 = cast(0, i32)\n",
    );

    write_file(
        &root.join("mylib/reef.toml"),
        &format!(
            r#"[package]
name = "mylib"
version = "0.1.0"
compiler = "={}"
module_prefix = "Mylib"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (add, double, square)\n\n\
         def add(x: i32, y: i32) -> i32 = x + y\n\
         def double(x: i32) -> i32 = x + x\n\
         def square(x: i32) -> i32 = x * x\n",
    );

    write_file(
        &root.join("reef.lock"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"

[[dependencies]]
name = "mylib"
version = "0.1.0"
compiler = "={}"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./mylib"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    (dir, root)
}

/// List the `.ctx` cache files under `<reef_home>/.cache/compiled/`.
/// Returns an empty vec if the directory doesn't exist yet.
fn list_cache_files(reef_home: &Path) -> Vec<PathBuf> {
    let cache_dir = reef_home.join(".cache").join("compiled");
    let Ok(entries) = fs::read_dir(&cache_dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ctx"))
        .collect();
    out.sort();
    out
}

#[test]
fn cmd_eval_warm_cache_hit_byte_identical_to_cold() {
    let (_pkg_dir, root) = path_dep_package();
    let entry_path = root.join("src/evalwarm.ch");
    let snippet = "module App.EvalWarm\n\ndef warm_value() -> i32 = 99\n";
    write_file(&entry_path, snippet);

    let reef_home = tempdir().expect("reef_home tempdir");
    assert!(
        list_cache_files(reef_home.path()).is_empty(),
        "fresh tempdir must not contain any cache files"
    );

    // Cold: first run pays the full compile and writes the cache.
    let cold = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home.path())
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval (cold)");
    assert!(
        cold.status.success(),
        "cold exit: {:?} stderr={}",
        cold.status,
        String::from_utf8_lossy(&cold.stderr)
    );
    let cold_stdout = String::from_utf8(cold.stdout).expect("utf8 cold");

    // After cold: exactly one cache file should exist.
    let cache_after_cold = list_cache_files(reef_home.path());
    assert_eq!(
        cache_after_cold.len(),
        1,
        "cold run must create exactly one cache file; got {:?}",
        cache_after_cold
    );

    // Warm: second run must hit the cache and produce identical stdout.
    let warm = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home.path())
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval (warm)");
    assert!(
        warm.status.success(),
        "warm exit: {:?} stderr={}",
        warm.status,
        String::from_utf8_lossy(&warm.stderr)
    );
    let warm_stdout = String::from_utf8(warm.stdout).expect("utf8 warm");

    assert_eq!(
        cold_stdout, warm_stdout,
        "warm cache must produce byte-identical stdout to cold run"
    );

    // Cache file count must not grow on warm hit.
    let cache_after_warm = list_cache_files(reef_home.path());
    assert_eq!(
        cache_after_warm.len(),
        1,
        "warm hit must NOT create a new cache file; got {:?}",
        cache_after_warm
    );
    assert_eq!(
        cache_after_cold[0], cache_after_warm[0],
        "warm hit must reuse the exact same cache file the cold run wrote"
    );
}

#[test]
fn cmd_eval_source_edit_invalidates_cache_and_re_saves() {
    // Edit a path-dep source file between two runs. The disk cache key
    // is keyed off `source_hash` so the second run must miss → recompile,
    // and produce a NEW cache file (different hash prefix in the
    // filename). Negative parity to the warm-hit test above.
    let (_pkg_dir, root) = path_dep_package();
    let entry_path = root.join("src/evaledit.ch");
    let snippet =
        "module App.EvalEdit\nimport Mylib.Math (square)\n\ndef edit_value() -> i32 = square(3)\n";
    write_file(&entry_path, snippet);

    let reef_home = tempdir().expect("reef_home tempdir");

    // Cold #1.
    let cold1 = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home.path())
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval (cold #1)");
    assert!(cold1.status.success(), "cold #1 must succeed");
    let files_after_cold1 = list_cache_files(reef_home.path());
    assert_eq!(files_after_cold1.len(), 1);

    // Edit a path-dep source file — the hash MUST change.
    let math_path = root.join("mylib/src/math.ch");
    let mut math_src = fs::read_to_string(&math_path).expect("read math.ch");
    math_src.push_str("\n-- a comment that changes file content\n");
    fs::write(&math_path, &math_src).expect("rewrite math.ch");

    // Cold #2: same eval, but the underlying library changed → new hash → new cache file.
    let cold2 = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home.path())
        .args(["eval", "--file", entry_path.to_str().unwrap()])
        .output()
        .expect("run chelis eval (cold #2)");
    assert!(
        cold2.status.success(),
        "cold #2 exit: {:?} stderr={}",
        cold2.status,
        String::from_utf8_lossy(&cold2.stderr)
    );

    let files_after_cold2 = list_cache_files(reef_home.path());
    assert_eq!(
        files_after_cold2.len(),
        2,
        "post-edit run must save a NEW cache file, leaving 2 total; got {:?}",
        files_after_cold2
    );
    // The pre-edit file should still be present (Phase K does not GC).
    assert!(
        files_after_cold2.contains(&files_after_cold1[0]),
        "pre-edit cache file should still exist; before={:?} after={:?}",
        files_after_cold1,
        files_after_cold2
    );
}

#[test]
fn cmd_test_warm_cache_creates_and_reuses_compiled_context() {
    // Mirror of `cmd_eval_warm_cache_hit_byte_identical_to_cold` but for
    // `chelis test`. The parent's `compile_reef_context` call inside
    // `cmd_test` now goes through `load_or_compile_for_package`, so a
    // re-run with unchanged sources reuses the on-disk artifact instead
    // of paying the full library compile.
    //
    // We need at least one runnable test: `chelis test` rejects a zero-test
    // selection BEFORE the parent's compile_reef_context call, so an empty
    // selection wouldn't exercise the disk-cache wire-up at all.
    // A trivial runnable `def test_*` is enough.
    let (_pkg_dir, root) = path_dep_package();
    fs::create_dir_all(root.join("tests")).expect("mkdir tests");
    write_file(
        &root.join("tests/smoke.ch"),
        "module App.SmokeTest\n\ndef test_trivial() -> unit = test_assert(true, \"trivial\")\n",
    );

    let reef_home = tempdir().expect("reef_home tempdir");
    assert!(
        list_cache_files(reef_home.path()).is_empty(),
        "fresh tempdir must not contain any cache files"
    );

    // Cold: full compile and save.
    let cold = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home.path())
        .current_dir(&root)
        .args(["test", "tests/"])
        .output()
        .expect("run chelis test (cold)");
    assert!(
        cold.status.success(),
        "cold chelis test must succeed on populated tests dir; exit={:?} stderr={}",
        cold.status,
        String::from_utf8_lossy(&cold.stderr)
    );
    let cache_after_cold = list_cache_files(reef_home.path());
    assert_eq!(
        cache_after_cold.len(),
        1,
        "cold chelis test must write exactly one cache file; got {:?}",
        cache_after_cold
    );

    // Warm: cache hit must NOT create a new cache file.
    let warm = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home.path())
        .current_dir(&root)
        .args(["test", "tests/"])
        .output()
        .expect("run chelis test (warm)");
    assert!(
        warm.status.success(),
        "warm chelis test must succeed; exit={:?} stderr={}",
        warm.status,
        String::from_utf8_lossy(&warm.stderr)
    );
    let cache_after_warm = list_cache_files(reef_home.path());
    assert_eq!(
        cache_after_warm.len(),
        1,
        "warm hit must NOT create a new cache file; got {:?}",
        cache_after_warm
    );
    assert_eq!(
        cache_after_cold[0], cache_after_warm[0],
        "warm hit must reuse the exact same cache file the cold run wrote"
    );
}
