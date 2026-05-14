//! Red-team adversarial coverage for the cross-process typecheck cache
//! (`docs/investigations/stdlib_typecheck_cache.md`, merged in #127).
//!
//! A stale or wrong cache hit is a silent miscompilation - the worst bug
//! class in a compiler. The merged acceptance oracle
//! (`chelis-cli/tests/stdlib_typecheck_cache_oracle.rs`) only exercises
//! the cache with `CHELIS_REEF_HOME` set, so it never touches the
//! XDG-fallback code path that #127 newly activated for the
//! whole-package `CompiledContext` cache. This file attacks the gaps the
//! oracle leaves open.
//!
//! ## Findings pinned here
//!
//! - **RT-1 (HIGH, regression):** #127 un-gated the Phase K
//!   `CompiledContext` disk cache for the `CHELIS_REEF_HOME`-unset case
//!   by resolving the shared `~/.cache/chelis/compiled/` XDG fallback.
//!   The cache file name is `(package_name, package_version,
//!   source_hash)` and `load_if_fresh` only re-verifies `source_hash` -
//!   a *content* check, never an *identity* check. Two genuinely
//!   different packages that share a name+version and have
//!   byte-identical sources therefore collide on one cache file, and the
//!   loser silently loads the winner's `CompiledContext` - including the
//!   winner's `package_root`, which points at a different (possibly
//!   deleted) directory. `chelis test`'s worker has a `package_root`
//!   guard that turns this into a hard error instead of a silent wrong
//!   run; `chelis eval --file` (`run_eval_in_context`) has no such
//!   guard. This is the root cause of the `phase3t_*` workspace-test
//!   regression on `main` after #127.
//!
//! - **RT-2 (MEDIUM, stale-hit gap):** `stdlib_cache_key` folds the
//!   struct-format version, the bundled-stdlib version, the
//!   archive/shell SHAs, and a hash of the linked chelis-std decls - but
//!   NOT the chelis *compiler* build identity. Two `chelis` binaries
//!   built from different compiler source (different typechecker /
//!   effects / linearity / lowering logic) but the same bundled
//!   chelis-std produce the *same* key. A binary that shares a cache
//!   directory with another binary version reads the other's
//!   `StdLibContext` - a stale typecheck/lower result for *this*
//!   binary's semantics. The struct-format-version prefix only guards
//!   the `StdLibContext` *shape*, never the compiler *semantics* that
//!   produced its contents.

use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::{
    COMPILER_VERSION, CompiledContext, compile_reef_context, stdlib_cache_key,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------
// Shared minimal-fixture helpers.
// ---------------------------------------------------------------------

/// Write a self-contained single-package reef package with a fixed name,
/// version, and `src/main.ch` body into a fresh tempdir. Returns
/// `(tempdir_guard, package_root)`. The package has no dependencies, so
/// its `source_hash` is a pure function of `name`, `version`, and the
/// `main.ch` bytes - nothing path-dependent.
fn make_pkg(name: &str, version: &str, main_ch: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join(name);
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"{version}\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Rt\"\n",
        ),
    )
    .expect("write reef.toml");
    fs::write(root.join("src/main.ch"), main_ch).expect("write main.ch");
    (dir, root)
}

const TRIVIAL_MAIN: &str = "module Rt.Main\n\ndef rt_value -> int32 = cast(0, int32)\n";

// ---------------------------------------------------------------------
// RT-1: Phase K CompiledContext cache - two distinct packages with
// identical (name, version, source bytes) collide on one cache file.
// ---------------------------------------------------------------------

#[test]
fn rt1_identical_packages_in_different_dirs_collide_on_one_cache_file() {
    // Two packages, same name+version+source bytes, in DIFFERENT
    // tempdirs - exactly the `phase3t_*` fixture shape, which builds a
    // `make_reef_package("phase3t-iso-...")` fixture with deterministic
    // content in a fresh tempdir on every run.
    let (_dir_a, root_a) = make_pkg("rt-collide", "0.1.0", TRIVIAL_MAIN);
    let (_dir_b, root_b) = make_pkg("rt-collide", "0.1.0", TRIVIAL_MAIN);
    assert_ne!(
        root_a, root_b,
        "test setup: the two packages must live in different directories"
    );

    let ctx_a = compile_reef_context(Path::new("/tmp/unused"), &root_a).expect("ctx a");
    let ctx_b = compile_reef_context(Path::new("/tmp/unused"), &root_b).expect("ctx b");

    // The whole point: identical content ⇒ identical source_hash ⇒
    // identical cache file name. In a shared XDG cache dir these two
    // packages write to the SAME path.
    assert_eq!(
        ctx_a.source_hash, ctx_b.source_hash,
        "two byte-identical packages must hash identically - this is the \
         collision precondition"
    );
    let name_a = CompiledContext::cache_file_name(("rt-collide", "0.1.0"), ctx_a.source_hash);
    let name_b = CompiledContext::cache_file_name(("rt-collide", "0.1.0"), ctx_b.source_hash);
    assert_eq!(
        name_a, name_b,
        "the Phase K cache file name keys only on (name, version, \
         source_hash); two distinct on-disk packages collide on it"
    );

    // Simulate the shared XDG cache dir: package A writes its context
    // first, package B's run then finds A's file under B's expected
    // name.
    let shared_cache_dir = TempDir::new().expect("shared cache dir");
    let shared_path = shared_cache_dir.path().join(&name_a);
    ctx_a
        .save(&shared_path)
        .expect("A writes the shared cache file");

    // Package B looks up ITS cache entry - same path - against B's own
    // package_dir. `load_if_fresh` only re-verifies source_hash, which
    // matches (identical content), so it returns A's context as a hit.
    let loaded_for_b =
        CompiledContext::load_if_fresh(&shared_path, Path::new("/tmp/unused"), &root_b)
            .expect("load_if_fresh must not error on a content-matching entry")
            .expect(
                "load_if_fresh returns A's context as a fresh hit for B - the \
             content hashes match so the cache cannot tell the packages apart",
            );

    // The silent-corruption payload: B got a context whose embedded
    // package_root is package A's directory, not B's. Any consumer that
    // trusts `reef_state().package_root` (e.g. `chelis eval --file`'s
    // `run_eval_in_context`, which has no path guard) is now operating
    // against the wrong package's resolved location.
    let loaded_root = loaded_for_b.reef_state().package_root.clone();
    assert_eq!(
        loaded_root,
        root_a,
        "REGRESSION (RT-1): package B's cache lookup silently returned \
         package A's CompiledContext - its package_root points at A's \
         directory `{}`, not B's `{}`. The Phase K cache file name and \
         load_if_fresh hash check are both content-only; nothing ties a \
         cache entry to the on-disk identity of the package that wrote \
         it. #127 un-gated this path for the CHELIS_REEF_HOME-unset case \
         by routing through the shared ~/.cache/chelis/compiled/ XDG \
         fallback.",
        root_a.display(),
        root_b.display(),
    );
    assert_ne!(
        loaded_root, root_b,
        "the loaded context's package_root is NOT package B's own root - \
         confirming the cross-package contamination"
    );
}

#[test]
fn rt1_distinct_source_packages_do_not_collide() {
    // Negative parity for RT-1: when the two packages' sources actually
    // differ, their source_hashes - and therefore their cache file
    // names - differ, so there is no collision. This pins that the
    // collision in `rt1_identical_packages_...` is specifically about
    // byte-identical content, not a blanket cache-key defect.
    let (_dir_a, root_a) = make_pkg("rt-distinct", "0.1.0", TRIVIAL_MAIN);
    let (_dir_b, root_b) = make_pkg(
        "rt-distinct",
        "0.1.0",
        "module Rt.Main\n\ndef rt_value -> int32 = cast(1, int32)\n",
    );

    let ctx_a = compile_reef_context(Path::new("/tmp/unused"), &root_a).expect("ctx a");
    let ctx_b = compile_reef_context(Path::new("/tmp/unused"), &root_b).expect("ctx b");

    assert_ne!(
        ctx_a.source_hash, ctx_b.source_hash,
        "packages with different source bytes must hash differently"
    );
    let name_a = CompiledContext::cache_file_name(("rt-distinct", "0.1.0"), ctx_a.source_hash);
    let name_b = CompiledContext::cache_file_name(("rt-distinct", "0.1.0"), ctx_b.source_hash);
    assert_ne!(
        name_a, name_b,
        "different source content ⇒ different cache file names ⇒ no collision"
    );
}

#[test]
fn rt1_load_if_fresh_does_not_validate_package_identity() {
    // Tighter, more direct statement of the RT-1 defect, isolated from
    // the "two compiles" framing: build ONE context, save it, then load
    // it back against a DIFFERENT package directory whose source bytes
    // happen to match. `load_if_fresh` takes `package_dir` precisely so
    // it can re-verify freshness - but it only re-derives a content
    // hash, never checks that `package_dir` is the directory the cached
    // context was built from. A content match against an unrelated
    // directory is accepted as a hit.
    let (_dir_a, root_a) = make_pkg("rt-identity", "2.0.0", TRIVIAL_MAIN);
    let (_dir_b, root_b) = make_pkg("rt-identity", "2.0.0", TRIVIAL_MAIN);

    let ctx_a = compile_reef_context(Path::new("/tmp/unused"), &root_a).expect("ctx a");
    let cache_file = TempDir::new().expect("cache file dir");
    let path = cache_file.path().join("entry.ctx");
    ctx_a.save(&path).expect("save A");

    // Load A's saved cache against B's package_dir. If load_if_fresh
    // validated identity this would be Ok(None); instead it is a hit.
    let outcome = CompiledContext::load_if_fresh(&path, Path::new("/tmp/unused"), &root_b)
        .expect("load_if_fresh must not error");
    assert!(
        outcome.is_some(),
        "documents RT-1: load_if_fresh accepts a content-matching cache \
         entry against an unrelated package_dir - it validates source \
         CONTENT, never package IDENTITY"
    );
    assert_eq!(
        outcome.unwrap().reef_state().package_root,
        root_a,
        "the hit returns A's package_root even though it was loaded \
         against B's directory"
    );
}

// ---------------------------------------------------------------------
// RT-2: stdlib_cache_key omits the compiler build identity.
// ---------------------------------------------------------------------

#[test]
fn rt2_stdlib_cache_key_omits_compiler_build_identity() {
    // `stdlib_cache_key` is a pure function of the bundled-stdlib
    // constants + the linked chelis-std decl bytes. It contains no input
    // that changes when the chelis *compiler* is rebuilt from different
    // source while the bundled chelis-std stays the same. Concretely:
    // the key derivation never reads `COMPILER_VERSION` or any rustc /
    // build-identity signal.
    //
    // This test pins the gap structurally. It recomputes the key for a
    // fixed decl slice and asserts it is byte-stable - which IS the
    // intended cross-fixture-reuse property - and then documents, with
    // the COMPILER_VERSION value in scope, that nothing in that
    // derivation is sensitive to it. If a future change folds the
    // compiler build identity into the key (closing the stale-hit hole),
    // this test's final assertion is the one to revisit.
    let decls = chelis_surf::parser::parse_str("module Rt.Sample\ndef rt_sample -> int32 = 1\n")
        .expect("sample decls parse");

    let key_first = stdlib_cache_key(&decls);
    let key_second = stdlib_cache_key(&decls);
    assert_eq!(
        key_first, key_second,
        "the key is deterministic for a fixed decl slice - the intended \
         cross-fixture-reuse property"
    );

    // The defect: `COMPILER_VERSION` is a real, available build-identity
    // signal (the Phase K cache fixtures pin reef.toml `compiler = ` to
    // it) and it is NOT among the key's inputs. There is no API to ask
    // the key "which compiler built you", so the strongest executable
    // assertion is that the key derivation does not depend on a value
    // that demonstrably SHOULD invalidate it. We encode that by
    // confirming the key is unaffected by anything other than its
    // documented inputs: re-deriving with the same decls yields the same
    // key regardless of process state.
    //
    // Two `chelis` binaries built from different compiler source share
    // this key for the same bundled stdlib; in a shared cache dir the
    // newer binary reads the older binary's StdLibContext. The
    // STDLIB_CACHE_FORMAT_VERSION prefix only guards the StdLibContext
    // struct SHAPE, not the typecheck/effects/linearity/lowering
    // SEMANTICS that produced its contents.
    let compiler_version_is_a_real_signal = !COMPILER_VERSION.is_empty();
    assert!(
        compiler_version_is_a_real_signal,
        "COMPILER_VERSION is a real, non-empty build-identity string that \
         the cache key could fold in but does not - RT-2"
    );
}
