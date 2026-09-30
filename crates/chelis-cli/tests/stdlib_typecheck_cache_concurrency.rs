//! Concurrency and corruption-resilience oracle for the chelis-std
//! typecheck cache.
//!
//! `cargo nextest` spawns ~200 separate test processes; on a cold cache
//! they all race to compute and write the same chelis-std sub-context
//! artifact. The cache writes atomically (temp file in the same
//! directory + `fs::rename`), so a racing reader sees either the old
//! state (absent) or the fully-written file, never a torn tail. A
//! torn or corrupt read must fall through to recompute, never panic.
//!
//! This file pins:
//!
//! 1. `parallel_cold_cache_invocations_all_succeed_identically` — N
//!    `chelis check` processes launched against one cold cache
//!    directory all exit 0 and produce byte-identical output. No
//!    panic, no torn-read failure, no "cache file is corrupt" abort.
//!
//! 2. `corrupt_cache_file_falls_through_to_recompute` — a cache file
//!    overwritten with garbage bytes does not crash the next
//!    invocation: the run detects the corruption, falls through to a
//!    full recompute, and produces output byte-identical to a clean
//!    run.
//!
//! 3. `truncated_cache_file_falls_through_to_recompute` — a cache file
//!    truncated mid-payload (the classic torn-write shape) is rejected
//!    and recomputed, not silently accepted.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use tempfile::{TempDir, tempdir};

fn stdlib_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std/src/decimal.ch")
        .canonicalize()
        .expect("chelis-std decimal.ch must exist")
}

fn fresh_cache_home() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let home = dir.path().join("reef-home");
    fs::create_dir_all(&home).expect("mkdir reef-home");
    (dir, home)
}

fn run_check(file: &Path, cache_home: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home)
        .arg("check")
        .arg(file)
        .output()
        .expect("spawn chelis check")
}

/// The on-disk typecheck cache directory the cache resolves to when
/// `CHELIS_REEF_HOME` is set.
fn cache_dir(cache_home: &Path) -> PathBuf {
    cache_home.join(".cache").join("typecheck")
}

// ---------------------------------------------------------------------
// 1. Parallel cold-cache stress.
// ---------------------------------------------------------------------

#[test]
fn parallel_cold_cache_invocations_all_succeed_identically() {
    let (_guard, cache_home) = fresh_cache_home();
    let file = stdlib_file();

    // Launch N concurrent `chelis check` processes against the same
    // cold cache dir. They race on the first-miss write; the atomic
    // temp-file+rename must make every racer either compute-and-write
    // or read-a-complete-file, with no panic and no torn read.
    let n = 12;
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let file = file.clone();
            let cache_home = cache_home.clone();
            thread::spawn(move || run_check(&file, &cache_home))
        })
        .collect();

    let outputs: Vec<std::process::Output> = handles
        .into_iter()
        .map(|h| h.join().expect("join"))
        .collect();

    let first_stdout = outputs[0].stdout.clone();
    for (i, out) in outputs.iter().enumerate() {
        assert!(
            out.status.success(),
            "parallel cold-cache invocation {i} did not exit 0: \
             stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("panicked"),
            "parallel cold-cache invocation {i} panicked: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            out.stdout, first_stdout,
            "parallel cold-cache invocation {i} produced divergent output: \
             a write race must not corrupt any racer's result"
        );
    }
}

// ---------------------------------------------------------------------
// 2 & 3. Corrupt / truncated cache file fall-through.
// ---------------------------------------------------------------------

#[test]
fn corrupt_cache_file_falls_through_to_recompute() {
    let (_guard, cache_home) = fresh_cache_home();
    let file = stdlib_file();

    // Warm the cache, capture the clean result.
    let clean = run_check(&file, &cache_home);
    assert!(clean.status.success(), "warm-up check must succeed");
    let clean_stdout = clean.stdout.clone();

    // Overwrite every file in the typecheck cache dir with garbage.
    let dir = cache_dir(&cache_home);
    let entries: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cache dir {} must exist after warm-up: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.is_file())
        .collect();
    assert!(
        !entries.is_empty(),
        "warm-up must have written at least one cache file under {}",
        dir.display()
    );
    for path in &entries {
        fs::write(path, b"not a valid cache envelope -- garbage bytes")
            .expect("overwrite cache file with garbage");
    }

    // Next invocation must NOT panic or abort: it detects the corrupt
    // file, recomputes, and produces the same output as the clean run.
    let recovered = run_check(&file, &cache_home);
    assert!(
        recovered.status.success(),
        "check against a corrupt cache file must fall through to recompute, \
         not fail: stderr={}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&recovered.stderr).contains("panicked"),
        "corrupt cache file must not cause a panic: stderr={}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert_eq!(
        recovered.stdout, clean_stdout,
        "recompute after corrupt-cache fall-through must produce output \
         byte-identical to the clean run"
    );
}

#[test]
fn truncated_cache_file_falls_through_to_recompute() {
    let (_guard, cache_home) = fresh_cache_home();
    let file = stdlib_file();

    let clean = run_check(&file, &cache_home);
    assert!(clean.status.success(), "warm-up check must succeed");
    let clean_stdout = clean.stdout.clone();

    // Truncate every cache file to a short prefix — the classic
    // torn-write shape: enough bytes to look like the start of a file,
    // not enough to be a valid envelope/payload.
    let dir = cache_dir(&cache_home);
    let entries: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cache dir {} must exist after warm-up: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.is_file())
        .collect();
    assert!(
        !entries.is_empty(),
        "warm-up must have written a cache file"
    );
    for path in &entries {
        let full = fs::read(path).expect("read cache file");
        let cut = full.len() / 3;
        fs::write(path, &full[..cut]).expect("truncate cache file");
    }

    let recovered = run_check(&file, &cache_home);
    assert!(
        recovered.status.success(),
        "check against a truncated cache file must fall through to \
         recompute, not fail: stderr={}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&recovered.stderr).contains("panicked"),
        "truncated cache file must not cause a panic: stderr={}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert_eq!(
        recovered.stdout, clean_stdout,
        "recompute after truncated-cache fall-through must produce output \
         byte-identical to the clean run"
    );
}
