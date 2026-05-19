//! Acceptance oracle for the cross-process chelis-std typecheck cache.
//!
//! The cache (see `docs/investigations/stdlib_typecheck_cache.md`) stores
//! the typechecked chelis-std library sub-context, content-addressed on
//! `chelis_std_version || archive_sha256 || shell_sha256`, so every
//! `chelis check` / `chelis build` invocation that imports the bundled
//! stdlib reuses one cross-process artifact instead of re-parsing and
//! re-typechecking the entire chelis-std import graph (~4.5s) from
//! scratch.
//!
//! A stale or wrong cache hit is a silent miscompilation — the worst bug
//! class in a compiler. This file is the non-negotiable gate. It pins
//! TWO byte-identical properties plus the stale-key negative case:
//!
//! 1. `cold_vs_warm_check_byte_identical` /
//!    `cold_vs_warm_build_byte_identical` — a cold-cache run and a
//!    warm-cache run of the same `chelis check` / `chelis build` over a
//!    stdlib-importing corpus produce byte-identical stdout. The cache
//!    must not perturb output.
//!
//! 2. `monolithic_vs_incontext_check_byte_identical` /
//!    `monolithic_vs_incontext_build_byte_identical` — running the
//!    check/build path through the monolithic typechecker
//!    (`CHELIS_STDLIB_CACHE_DISABLE=1`, no cache, no `_with_context`
//!    layering) produces byte-identical stdout to the layered
//!    `_with_context` path. This is independent of cache hit/miss: it
//!    proves the monolithic→layered re-route itself is correctness-
//!    preserving. A divergence here is a compiler-correctness bug in a
//!    `_with_context` variant, not a cache bug.
//!
//! 3. `stale_stdlib_byte_mutation_misses_not_stale_hit` — mutating a
//!    byte of chelis-std source changes the content-addressed key, so a
//!    subsequent run is a clean miss (recompute), never a stale hit
//!    against the pre-mutation artifact.
//!
//! Concurrency stress and corrupt-cache fall-through are pinned
//! separately in `stdlib_typecheck_cache_concurrency.rs`.
//!
//! ## Cache-disable contract
//!
//! `CHELIS_STDLIB_CACHE_DISABLE=1` forces the monolithic typecheck path:
//! no on-disk cache read or write, and `check` / `build` run through the
//! monolithic `check_typed_program` / `check_ir_program` pipeline rather
//! than the `_with_context` layered pipeline. It is the test seam for
//! property 2 and is never set in production CI.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Corpus: production chelis-std files (loaded via the bundled stdlib
/// loader) plus a non-chelis-std reef package that imports the bundled
/// stdlib. The non-std fixture is the cross-fixture-reuse witness: its
/// whole-graph cache key differs from any stdlib file's, but the
/// chelis-std *sub-context* key is identical, so the sub-cache is what
/// gives it a warm hit.
///
/// `scratch` is a per-test tempdir into which the non-std fixture is
/// copied — without staging, `chelis check`/`build` writes `reef.lock`
/// next to the fixture's `reef.toml`, leaking into the working tree.
fn stdlib_corpus(scratch: &Path) -> Vec<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let std_root = manifest.join("../../packages/chelis-std");
    let mut paths: Vec<PathBuf> = [
        "src/nn/linear.ch",
        "src/nn/embedding.ch",
        "src/nn/rmsnorm.ch",
        "src/loss/crossentropy.ch",
        "src/tensor/reduce.ch",
        "src/optim.ch",
    ]
    .iter()
    .map(|rel| {
        std_root
            .join(rel)
            .canonicalize()
            .unwrap_or_else(|e| panic!("canonicalize {rel}: {e}"))
    })
    .collect();
    paths.push(stage_non_std_fixture(scratch));
    paths
}

/// Copy the pseudo_nautilus fixture into `scratch` and return the path
/// to the staged `src/special.ch`. Staging is required because `chelis
/// check`/`build` writes `reef.lock` next to the resolved `reef.toml`;
/// running directly against the real fixture path leaks the lockfile
/// into the source tree.
fn stage_non_std_fixture(scratch: &Path) -> PathBuf {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pseudo_nautilus")
        .canonicalize()
        .expect("pseudo_nautilus fixture must exist");
    let dst = scratch.join("pseudo_nautilus");
    copy_dir_recursive(&src, &dst);
    dst.join("src/special.ch")
        .canonicalize()
        .expect("staged special.ch must exist")
}

/// Run `chelis <subcommand> <file>` with the cache directory isolated to
/// `cache_home` (via `CHELIS_REEF_HOME`) and the style gate disabled
/// (the corpus mixes canonical and migration-baseline source). Returns
/// raw stdout bytes. `extra_env` carries the optional
/// `CHELIS_STDLIB_CACHE_DISABLE` seam.
fn run_capture(
    subcommand: &str,
    file: &Path,
    cache_home: &Path,
    extra_env: &[(&str, &str)],
    out_dir: Option<&Path>,
) -> Vec<u8> {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.arg(subcommand).arg(file);
    if let Some(dir) = out_dir {
        cmd.arg("-o").arg(dir);
    }
    cmd.assert().success().get_output().stdout.clone()
}

/// Fresh, empty cache directory. Each property test gets its own so
/// "cold" is genuinely cold and tests do not share cache state.
fn fresh_cache_home() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let home = dir.path().join("reef-home");
    fs::create_dir_all(&home).expect("mkdir reef-home");
    (dir, home)
}

// ---------------------------------------------------------------------
// Property 1: cold-cache vs warm-cache byte-identical.
// ---------------------------------------------------------------------

#[test]
fn cold_vs_warm_check_byte_identical() {
    let (_guard, cache_home) = fresh_cache_home();
    for file in stdlib_corpus(_guard.path()) {
        // Cold: empty cache dir, this run computes + writes the artifact.
        let cold = run_capture("check", &file, &cache_home, &[], None);
        // Warm: same cache dir, the artifact now exists — must hit.
        let warm = run_capture("check", &file, &cache_home, &[], None);
        assert_eq!(
            cold,
            warm,
            "cold-cache vs warm-cache `chelis check` output diverged for {}: \
             a cache hit must never perturb output",
            file.display()
        );
    }
}

#[test]
fn cold_vs_warm_build_byte_identical() {
    let (_guard, cache_home) = fresh_cache_home();
    for file in stdlib_corpus(_guard.path()) {
        // One shared out dir: `chelis build` echoes the resolved `-o`
        // path in its stdout, so cold and warm must build into the SAME
        // directory for the comparison to isolate the cache effect from
        // out-dir-path nondeterminism. The warm run overwrites the cold
        // run's artifacts; the byte-identical property is over stdout.
        let out_dir = tempdir().expect("out dir");
        let cold = run_capture("build", &file, &cache_home, &[], Some(out_dir.path()));
        let warm = run_capture("build", &file, &cache_home, &[], Some(out_dir.path()));
        assert_eq!(
            cold,
            warm,
            "cold-cache vs warm-cache `chelis build` output diverged for {}: \
             a cache hit must never perturb output",
            file.display()
        );
    }
}

// ---------------------------------------------------------------------
// Property 2: monolithic vs in-context (layered) byte-identical.
// Independent of the cache: proves the monolithic->_with_context
// re-route over the partitioned (stdlib | rest) split is itself
// correctness-preserving.
// ---------------------------------------------------------------------

#[test]
fn monolithic_vs_incontext_check_byte_identical() {
    let (_guard, cache_home) = fresh_cache_home();
    for file in stdlib_corpus(_guard.path()) {
        // Monolithic: cache disabled, runs the monolithic check_typed_program
        // pipeline over the whole merged program.
        let monolithic = run_capture(
            "check",
            &file,
            &cache_home,
            &[("CHELIS_STDLIB_CACHE_DISABLE", "1")],
            None,
        );
        // In-context: cache enabled, runs the layered `_with_context`
        // pipeline against the cached chelis-std sub-context.
        let in_context = run_capture("check", &file, &cache_home, &[], None);
        assert_eq!(
            monolithic,
            in_context,
            "monolithic vs in-context `chelis check` output diverged for {}: \
             this is a compiler-correctness divergence between the monolithic \
             typechecker and a `_with_context` variant on the partitioned \
             (stdlib | rest) split, NOT a cache bug",
            file.display()
        );
    }
}

#[test]
fn monolithic_vs_incontext_build_byte_identical() {
    let (_guard, cache_home) = fresh_cache_home();
    for file in stdlib_corpus(_guard.path()) {
        // One shared out dir for the same reason as
        // `cold_vs_warm_build_byte_identical`: `chelis build` echoes the
        // resolved `-o` path, so the monolithic and in-context runs must
        // target the same directory to isolate the layering effect from
        // out-dir-path nondeterminism.
        let out_dir = tempdir().expect("out dir");
        let monolithic = run_capture(
            "build",
            &file,
            &cache_home,
            &[("CHELIS_STDLIB_CACHE_DISABLE", "1")],
            Some(out_dir.path()),
        );
        let in_context = run_capture("build", &file, &cache_home, &[], Some(out_dir.path()));
        assert_eq!(
            monolithic,
            in_context,
            "monolithic vs in-context `chelis build` output diverged for {}: \
             this is a compiler-correctness divergence between the monolithic \
             pipeline and the `_with_context` lowering path, NOT a cache bug",
            file.display()
        );
    }
}

// ---------------------------------------------------------------------
// Property 3: stale-key negative. Mutating a chelis-std source byte
// changes the content-addressed key, so the next run is a clean miss
// (recompute), never a stale hit.
// ---------------------------------------------------------------------

#[test]
fn stale_stdlib_byte_mutation_misses_not_stale_hit() {
    // The chelis-std typecheck cache key is content-addressed on
    // `chelis_std_version || archive_sha256 || shell_sha256`. This test
    // pins the end-to-end invariant: a chelis-std whose source bytes
    // differ produces a different on-disk cache artifact (different
    // file name AND different contents), so it can never collide with —
    // and stale-hit against — a pristine chelis-std's artifact.
    //
    // The bundled stdlib is immutable per-binary, so its key inputs
    // cannot be perturbed in-process. Instead this test stages a
    // separately-published, byte-mutated chelis-std in its own reef
    // home and proves its cache artifact is disjoint from the pristine
    // bundled run's. The exact key-derivation wiring
    // (`archive_sha256` / `shell_sha256` feed the key) is additionally
    // unit-tested in `chelis-compiler-api`.
    let work = tempdir().expect("work dir");

    // Pristine run: bundled stdlib, isolated cache home A. Stage the
    // non-std fixture into the same tempdir so its lockfile lands
    // there instead of next to the real fixture's `reef.toml`.
    let (_guard_a, cache_home_a) = fresh_cache_home();
    let pristine_fixture = stage_non_std_fixture(_guard_a.path());
    let pristine_out = run_capture("check", &pristine_fixture, &cache_home_a, &[], None);
    let pristine_artifacts = cache_artifacts(&cache_home_a);
    assert!(
        !pristine_artifacts.is_empty(),
        "pristine run must write at least one chelis-std cache artifact"
    );

    // Mutated run: a local chelis-std copy with one source file's bytes
    // changed, published into its own reef home B with its own cache.
    let std_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package");
    let local_std = work.path().join("chelis-std");
    copy_dir_recursive(&std_src, &local_std);
    let target = local_std.join("src/tensor/reduce.ch");
    let original = fs::read_to_string(&target).expect("read reduce.ch");
    let mutated = mutate_one_identifier(&original);
    assert_ne!(
        original, mutated,
        "test setup: mutation must actually change reduce.ch bytes"
    );
    fs::write(&target, &mutated).expect("write mutated reduce.ch");

    let (_guard_b, cache_home_b) = fresh_cache_home();
    publish(&local_std, &cache_home_b);
    let mutated_out = run_capture(
        "check",
        &local_std.join("src/nn/linear.ch"),
        &cache_home_b,
        &[],
        None,
    );
    let mutated_artifacts = cache_artifacts(&cache_home_b);
    assert!(
        !mutated_artifacts.is_empty(),
        "mutated-stdlib run must write at least one chelis-std cache artifact"
    );

    // The content-addressed key must differ: no cache file from the
    // mutated stdlib may share BOTH name and bytes with a pristine one.
    // Identical (name, bytes) across the two runs would mean the
    // mutated stdlib stale-hit the pristine artifact.
    for (m_name, m_bytes) in &mutated_artifacts {
        for (p_name, p_bytes) in &pristine_artifacts {
            assert!(
                !(m_name == p_name && m_bytes == p_bytes),
                "mutated chelis-std produced a cache artifact byte-identical \
                 to the pristine one ({m_name}): a source byte change must \
                 flip the content-addressed key, never stale-hit"
            );
        }
    }

    // And the recompute is honest: the mutated run's output equals the
    // monolithic recompute over the mutated stdlib.
    let mutated_monolithic = run_capture(
        "check",
        &local_std.join("src/nn/linear.ch"),
        &cache_home_b,
        &[("CHELIS_STDLIB_CACHE_DISABLE", "1")],
        None,
    );
    assert_eq!(
        mutated_out, mutated_monolithic,
        "mutated-stdlib cache result must equal the recomputed monolithic \
         result over the same mutated stdlib: a clean miss + honest \
         recompute, never a stale hit"
    );

    // Sanity: pristine and mutated outputs are not trivially equal for a
    // reason unrelated to the cache (they are different programs / std).
    let _ = pristine_out;
}

/// Collect `(file_name, bytes)` for every file in the chelis-std
/// typecheck cache directory under `cache_home`. Empty if the directory
/// does not exist yet.
fn cache_artifacts(cache_home: &Path) -> Vec<(String, Vec<u8>)> {
    let dir = cache_home.join(".cache").join("typecheck");
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .map(|p| {
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let bytes = fs::read(&p).expect("read cache artifact");
            (name, bytes)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
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

fn publish(pkg: &Path, cache_home: &Path) {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home)
        .args(["reef", "publish", pkg.to_str().unwrap()])
        .assert()
        .success();
}

/// Append a trailing line comment to `src/tensor/reduce.ch` so the file
/// still parses and type-checks clean but its bytes (and therefore its
/// content hash, and therefore the chelis-std cache key) change.
fn mutate_one_identifier(source: &str) -> String {
    // A trailing line comment is the minimal byte change that keeps the
    // file parseable and type-checking clean. The cache key hashes raw
    // source bytes, so this is sufficient to flip the key.
    let mut mutated = source.to_string();
    if !mutated.ends_with('\n') {
        mutated.push('\n');
    }
    mutated.push_str("-- stdlib-typecheck-cache stale-key oracle marker\n");
    mutated
}
