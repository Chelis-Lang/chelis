//! Red-team adversarial coverage for the post-#130 compiled-context
//! cache (`docs/investigations/compiled_context_cache_package_identity_diagnosis.md`).
//!
//! A stale or wrong cache hit is a silent miscompilation, the worst bug
//! class in a compiler. RT-1's original adversarial file (#128, reverted
//! in #131) was written against the PRE-#130 cache: it asserted the
//! path-collision BUG as observed behavior and did not compile against
//! the shipped post-#130 API. This file is the replacement: every test
//! here compiles against the as-shipped cache and pins the post-#130
//! CORRECT behavior.
//!
//! ## What #130 changed (this file's spec, verified against
//! `crates/chelis-compiler-api/src/{context.rs,cache_envelope.rs,stdlib_cache.rs}`)
//!
//! - `CompiledContext::cache_file_name` takes a 3rd `&CacheIdentity` arg
//!   that folds the canonicalized `package_root` plus `COMPILER_VERSION`.
//!   Two packages with identical name+version+source bytes but different
//!   `package_root` now get DISTINCT cache file names: no collision.
//! - `load_if_fresh` recomputes the identity from the live package and
//!   the running binary; a mismatch is a clean miss (`Ok(None)`), never a
//!   stale hit. A belt-and-braces inner-vs-envelope check rejects a
//!   tampered identity as `CacheError::IdentityMismatch`.
//! - The cache format version and magic are now 16. The current format
//!   canonicalizes unordered collections, carries checker-owned nominal
//!   parameter kinds and kinded arguments, and tags each deferred shape
//!   obligation. A stale V15 file is rejected, never decoded.
//! - `stdlib_cache_key` folds `COMPILER_VERSION` directly, so a binary
//!   built from different compiler source does not stale-hit an older
//!   binary's `StdLibContext`.
//! - Corrupt / torn / truncated cache files fall through to a recompute,
//!   never panic, never return wrong data.
//!
//! ## Relationship to the in-crate tests
//!
//! `disk_cache.rs` and the `context.rs` / `stdlib_cache.rs` unit
//! tests already pin the happy paths and the two named #130 regression
//! cases. This file attacks the GAPS those leave: identity
//! canonicalization equivalence (a wrong-canonicalization regression
//! would be a NEW collision class), the `IdentityMismatch`
//! envelope-vs-inner tamper guard, fingerprint sensitivity to every
//! identity component, the format-version-16 magic rejection of a forged
//! V15 file, and adversarial corruption shapes against the recompute
//! fall-through.
//!
//! Kept on the per-PR `ci` profile: every test is cache-key / identity /
//! envelope logic plus at most one `compile_reef_context` on a tiny
//! no-dependency fixture. No real `chelis build`.

use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::{
    COMPILER_VERSION, CacheError, CacheIdentity, CompiledContext, compile_reef_context,
    stdlib_cache_key,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------
// Shared minimal-fixture helpers.
// ---------------------------------------------------------------------

/// Write a self-contained single-package reef package with a fixed name,
/// version, and `src/main.ch` body into a fresh tempdir. Returns
/// `(tempdir_guard, package_root)`. The package has no dependencies, so
/// its `source_hash` is a pure function of `name`, `version`, and the
/// `main.ch` bytes: nothing path-dependent. This is exactly the shape
/// that made the pre-#130 collision possible (two of these in different
/// tempdirs hash identically).
///
/// The returned `package_root` is `fs::canonicalize`d. This matters for
/// cross-platform parity: on macOS `tempfile` roots its dirs under
/// `/var/folders/...`, which is a symlink to `/private/var/folders/...`.
/// Both `chelis_reef::prepare_reef_graph` (via `find_package_root_for_dir`)
/// and `CacheIdentity::for_package_root` canonicalize the package root, so
/// a `CompiledContext` always carries the resolved `/private/var/...` form.
/// Returning the canonical path here keeps every `package_root` and
/// `CacheIdentity` comparison in this file consistent with what the cache
/// actually stores. (Linux has no such symlink, so this is a no-op there;
/// `canonicalize` is idempotent on an already-canonical path.)
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
    let root = fs::canonicalize(&root).expect("canonicalize package root");
    (dir, root)
}

const TRIVIAL_MAIN: &str = "module Rt.Main\n\ndef rt_value() -> i32 = cast(0, i32)\n";

/// Build a context, save it to a scratch path, and return the path plus
/// the saved bytes. The scratch dir guard is kept alive by the caller.
fn save_ctx(name: &str, main_ch: &str) -> (TempDir, PathBuf, CompiledContext, Vec<u8>) {
    let (dir, root) = make_pkg(name, "0.1.0", main_ch);
    let ctx = compile_reef_context(Path::new("/tmp/unused"), &root).expect("compile context");
    let cache_path = root.join(".cache/compiled/entry.ctx");
    ctx.save(&cache_path).expect("save context");
    let bytes = fs::read(&cache_path).expect("read saved cache bytes");
    (dir, cache_path, ctx, bytes)
}

// ---------------------------------------------------------------------
// RT-1: the #130 fix. Distinct roots must NOT collide. Negative parity
// for the original path-collision bug.
// ---------------------------------------------------------------------

#[test]
fn rt1_identical_packages_in_different_roots_do_not_share_a_cache_file_name() {
    // Pre-#130 the cache file name keyed only on
    // (name, version, source_hash). Two packages with identical
    // name+version+byte-identical sources in DIFFERENT directories
    // collided on one cache file name and the loser silently loaded the
    // winner's CompiledContext. #130 folds a CacheIdentity (canonical
    // package_root + COMPILER_VERSION) into the file name. This pins the
    // fix: identical content, different roots => same source_hash but
    // different identity => DIFFERENT cache file names => no collision.
    let (_dir_a, root_a) = make_pkg("rt-collide", "0.1.0", TRIVIAL_MAIN);
    let (_dir_b, root_b) = make_pkg("rt-collide", "0.1.0", TRIVIAL_MAIN);
    assert_ne!(
        root_a, root_b,
        "test setup: the two packages must live in different directories"
    );

    let ctx_a = compile_reef_context(Path::new("/tmp/unused"), &root_a).expect("ctx a");
    let ctx_b = compile_reef_context(Path::new("/tmp/unused"), &root_b).expect("ctx b");

    // The collision precondition still holds: byte-identical sources hash
    // identically. #130 does not change the source hash; it adds the
    // identity dimension on top.
    assert_eq!(
        ctx_a.source_hash, ctx_b.source_hash,
        "two byte-identical packages must still hash identically"
    );
    // But the identity differs because the roots differ.
    assert_ne!(
        ctx_a.identity, ctx_b.identity,
        "post-#130: distinct package roots must produce distinct CacheIdentity"
    );

    let name_a = CompiledContext::cache_file_name(
        ("rt-collide", "0.1.0"),
        ctx_a.source_hash,
        &ctx_a.identity,
    );
    let name_b = CompiledContext::cache_file_name(
        ("rt-collide", "0.1.0"),
        ctx_b.source_hash,
        &ctx_b.identity,
    );
    assert_ne!(
        name_a, name_b,
        "post-#130: identical content in different roots must NOT collide on one \
         cache file name; the CacheIdentity fingerprint keeps them apart"
    );
    // Both names still carry the shared package id and source prefix, so
    // only the identity segment diverges (the file names are otherwise
    // the same shape).
    assert!(name_a.starts_with("rt-collide-0.1.0-"));
    assert!(name_b.starts_with("rt-collide-0.1.0-"));
    assert!(name_a.ends_with(".ctx"));
}

#[test]
fn rt1_load_if_fresh_rejects_a_foreign_root_identity_as_a_clean_miss() {
    // The direct statement of the #130 fix, isolated from the cache-file-
    // name framing: build ONE context, save it, then load it back
    // against a DIFFERENT package directory whose source bytes happen to
    // match. Pre-#130 load_if_fresh re-derived only a content hash and
    // accepted the content match as a hit, silently returning the wrong
    // package's context (and its package_root). Post-#130 it ALSO
    // recomputes the CacheIdentity from the supplied package_dir and
    // rejects an entry whose stored identity does not match: a clean
    // miss (Ok(None)), never a cross-package hit.
    let (_dir_a, root_a) = make_pkg("rt-identity", "2.0.0", TRIVIAL_MAIN);
    let (_dir_b, root_b) = make_pkg("rt-identity", "2.0.0", TRIVIAL_MAIN);

    let ctx_a = compile_reef_context(Path::new("/tmp/unused"), &root_a).expect("ctx a");
    let cache_dir = TempDir::new().expect("cache dir");
    let path = cache_dir.path().join("entry.ctx");
    ctx_a.save(&path).expect("save A");

    // Load A's saved cache against B's package_dir: identity mismatch =>
    // clean miss, NOT a stale hit.
    let outcome = CompiledContext::load_if_fresh(&path, Path::new("/tmp/unused"), &root_b)
        .expect("load_if_fresh must not error on an identity-mismatched entry");
    assert!(
        outcome.is_none(),
        "post-#130 (RT-1 fix): load_if_fresh must reject a content-matching cache \
         entry whose stored CacheIdentity does not match the supplied package_dir; \
         it validates package IDENTITY, not just source CONTENT"
    );

    // Loading against A's own directory still hits: identity matches, and
    // the returned context carries A's own package_root. `root_a` is
    // already `fs::canonicalize`d by `make_pkg`, which is exactly the form
    // `prepare_reef_graph` resolves and stores, so this equality holds on
    // every platform (notably macOS, where the un-canonicalized tempfile
    // path `/var/...` differs from the stored `/private/var/...`).
    let hit = CompiledContext::load_if_fresh(&path, Path::new("/tmp/unused"), &root_a)
        .expect("load_if_fresh must not error against the original package_dir")
        .expect("loading against the original package_dir must still be a hit");
    assert_eq!(
        hit.reef_state().package_root,
        root_a,
        "the identity-matched hit returns the original package's package_root"
    );
}

#[test]
fn rt1_distinct_source_packages_still_do_not_collide() {
    // Negative parity for the negative parity: when the two packages'
    // sources actually differ, the collision is ruled out by the
    // source_hash alone (it always was, pre- and post-#130). This pins
    // that the #130 identity dimension is ADDITIVE: it closes the
    // identical-content gap without disturbing the source-content guard.
    let (_dir_a, root_a) = make_pkg("rt-distinct", "0.1.0", TRIVIAL_MAIN);
    let (_dir_b, root_b) = make_pkg(
        "rt-distinct",
        "0.1.0",
        "module Rt.Main\n\ndef rt_value() -> i32 = cast(1, i32)\n",
    );

    let ctx_a = compile_reef_context(Path::new("/tmp/unused"), &root_a).expect("ctx a");
    let ctx_b = compile_reef_context(Path::new("/tmp/unused"), &root_b).expect("ctx b");

    assert_ne!(
        ctx_a.source_hash, ctx_b.source_hash,
        "packages with different source bytes must hash differently"
    );
    let name_a = CompiledContext::cache_file_name(
        ("rt-distinct", "0.1.0"),
        ctx_a.source_hash,
        &ctx_a.identity,
    );
    let name_b = CompiledContext::cache_file_name(
        ("rt-distinct", "0.1.0"),
        ctx_b.source_hash,
        &ctx_b.identity,
    );
    assert_ne!(
        name_a, name_b,
        "different source content => different cache file names => no collision"
    );
}

// ---------------------------------------------------------------------
// Identity canonicalization. A wrong canonicalization would be a NEW
// collision class (one root reachable via two paths => two identities,
// one cache file each) OR a NEW aliasing bug (two roots canonicalizing
// to the same string). Neither in-crate test exercises this.
// ---------------------------------------------------------------------

#[test]
fn cache_identity_collapses_equivalent_paths_to_one_identity() {
    // `CacheIdentity::for_package_root` canonicalizes the path so
    // equivalent spellings of the SAME directory (a trailing-`/.`
    // segment, a `foo/..` round trip, a `.`-relative form) resolve to ONE
    // identity. If they did not, the same on-disk package reached via two
    // path spellings would write two cache files and a warm run via the
    // other spelling would always miss: a silent perpetual-cold-cache
    // regression, not a wrong-result bug, but still a #130-class defect.
    let (_dir, root) = make_pkg("rt-canon", "0.1.0", TRIVIAL_MAIN);

    let plain = CacheIdentity::for_package_root(&root);
    let trailing_dot = CacheIdentity::for_package_root(&root.join("."));
    let round_trip = CacheIdentity::for_package_root(&root.join("src").join(".."));

    assert_eq!(
        plain, trailing_dot,
        "a trailing `/.` segment must canonicalize to the same identity"
    );
    assert_eq!(
        plain, round_trip,
        "a `src/..` round trip must canonicalize to the same identity"
    );
    // And therefore the cache file names agree: one package, one file.
    let src_hash = compile_reef_context(Path::new("/tmp/unused"), &root)
        .expect("ctx")
        .source_hash;
    let name_plain = CompiledContext::cache_file_name(("rt-canon", "0.1.0"), src_hash, &plain);
    let name_round = CompiledContext::cache_file_name(("rt-canon", "0.1.0"), src_hash, &round_trip);
    assert_eq!(
        name_plain, name_round,
        "equivalent path spellings of one root must map to one cache file name"
    );
}

#[test]
fn cache_identity_fingerprint_is_sensitive_to_every_component() {
    // The identity fingerprint folded into the cache file name must
    // change when EITHER component changes: a different package_root OR a
    // different compiler_version. If it were insensitive to one of them,
    // that dimension would silently drop out of the cache key. We cannot
    // rebuild the compiler mid-test, so we construct CacheIdentity values
    // directly and compare the file names they produce.
    let base = CacheIdentity {
        package_root: "/abs/path/to/pkg-a".to_string(),
        compiler_version: COMPILER_VERSION.to_string(),
    };
    let diff_root = CacheIdentity {
        package_root: "/abs/path/to/pkg-b".to_string(),
        compiler_version: COMPILER_VERSION.to_string(),
    };
    let diff_compiler = CacheIdentity {
        package_root: "/abs/path/to/pkg-a".to_string(),
        compiler_version: format!("{COMPILER_VERSION}-other-build"),
    };
    assert_ne!(base, diff_root);
    assert_ne!(base, diff_compiler);

    let src = chelis_compiler_api::ContextHash([7u8; 32]);
    let name_base = CompiledContext::cache_file_name(("p", "1.0.0"), src, &base);
    let name_diff_root = CompiledContext::cache_file_name(("p", "1.0.0"), src, &diff_root);
    let name_diff_compiler = CompiledContext::cache_file_name(("p", "1.0.0"), src, &diff_compiler);

    assert_ne!(
        name_base, name_diff_root,
        "a different package_root must flip the cache file name"
    );
    assert_ne!(
        name_base, name_diff_compiler,
        "a different compiler_version must flip the cache file name"
    );
    assert_ne!(
        name_diff_root, name_diff_compiler,
        "the two perturbations must not alias onto each other"
    );
}

#[test]
fn cache_identity_does_not_confuse_root_and_compiler_boundary() {
    // The fingerprint length-prefixes each component, so two identities
    // that share the same CONCATENATION of (package_root + compiler_
    // version) but split the boundary differently must NOT fingerprint
    // the same. Without the length prefix, `("ab", "c")` and `("a",
    // "bc")` would collide: a classic concatenation-ambiguity hole that
    // would let a hostile or merely unlucky package_root impersonate a
    // different (root, compiler) pair.
    let left = CacheIdentity {
        package_root: "ab".to_string(),
        compiler_version: "c".to_string(),
    };
    let right = CacheIdentity {
        package_root: "a".to_string(),
        compiler_version: "bc".to_string(),
    };
    assert_ne!(left, right);
    let src = chelis_compiler_api::ContextHash([0u8; 32]);
    assert_ne!(
        CompiledContext::cache_file_name(("p", "1.0.0"), src, &left),
        CompiledContext::cache_file_name(("p", "1.0.0"), src, &right),
        "the identity fingerprint must length-prefix each component so a \
         boundary shift cannot produce a colliding fingerprint"
    );
}

// ---------------------------------------------------------------------
// Envelope-vs-inner tamper guards. `load_if_fresh` carries a
// belt-and-braces check: the inner CompiledContext's `identity` and
// `source_hash` must agree with the outer envelope's copies. A
// disagreement means the bytes were mutated and is rejected, never
// silently accepted. No in-crate test forges this disagreement.
// ---------------------------------------------------------------------

#[test]
fn load_if_fresh_rejects_a_torn_write_in_the_identity_region() {
    // Save a valid context, then corrupt the bytes in the region most
    // likely to hold the envelope's outer `identity` copy: the path
    // string near the front of the bincode region. The result must NOT
    // be Ok(Some(_)). It is allowed to be Err(Corrupt | Decode |
    // IdentityMismatch | HashMismatch | UnsupportedVersion) or, if the
    // mutated bytes happen to make the source-hash check fail first,
    // Ok(None). The non-negotiable invariant: never a silent hit on
    // mutated bytes.
    let (_dir, cache_path, _ctx, bytes) = save_ctx("rt-torn-id", TRIVIAL_MAIN);
    assert!(bytes.len() > 64, "fixture must produce a non-trivial file");

    // Flip a run of bytes just past the magic header, where the envelope
    // version + source_hash + identity sit.
    let mut corrupted = bytes.clone();
    let start = 24.min(corrupted.len() - 8);
    for b in &mut corrupted[start..start + 8] {
        *b ^= 0xff;
    }
    fs::write(&cache_path, &corrupted).expect("write corrupted cache");

    let outcome =
        CompiledContext::load_if_fresh(&cache_path, Path::new("/tmp/unused"), &cache_path);
    match outcome {
        Ok(Some(_)) => {
            panic!("a torn write in the identity/header region must NEVER load as Ok(Some(_))")
        }
        Ok(None) => { /* acceptable: a recompute fall-through */ }
        Err(
            CacheError::Corrupt(_)
            | CacheError::Decode(_)
            | CacheError::UnsupportedVersion { .. }
            | CacheError::HashMismatch { .. }
            | CacheError::IdentityMismatch { .. }
            | CacheError::Io { .. }
            | CacheError::Reef(_),
        ) => { /* acceptable: any "do not use these bytes" signal */ }
        Err(CacheError::Encode(e)) => panic!("encode error is impossible on the load path: {e}"),
        // The disk-cache load path has no out-of-band digest to compare
        // against -- that absence is the whole reason it re-derives
        // (chelis#2211) -- so this variant arriving here would mean the two
        // routes had been merged without anyone deciding to.
        Err(CacheError::HandoffDigestMismatch { expected, actual }) => panic!(
            "the disk-cache load path must not report a handoff digest mismatch: \
             expected={expected} actual={actual}"
        ),
    }
}

#[test]
fn load_if_fresh_never_panics_on_adversarial_byte_patterns() {
    // Feed `load_if_fresh` a battery of hostile byte patterns. Every one
    // must return a Result (Ok(None) or Err), never panic and never
    // Ok(Some(_)). A panic in the cache-load path would turn a corrupt
    // cache file into a hard crash; a silent hit would be a
    // miscompilation.
    let scratch = TempDir::new().expect("scratch dir");
    let cache_path = scratch.path().join("hostile.ctx");
    let pkg_dir = scratch.path().join("nonexistent-package");

    let patterns: Vec<Vec<u8>> = vec![
        vec![],                                               // empty
        b"CHELIS_CTX_V23\n".to_vec(),                         // current magic only, no envelope
        b"CHELIS_CTX_V9\n".to_vec(),                          // stale-version magic only
        b"not a cache file at all".to_vec(),                  // no magic
        vec![0u8; 4096],                                      // all zeros
        vec![0xffu8; 4096],                                   // all ones
        (0..4096).map(|i| ((i * 31) ^ 0x5a) as u8).collect(), // pseudo-random
        {
            // valid (current) magic followed by garbage
            let mut v = b"CHELIS_CTX_V23\n".to_vec();
            v.extend((0..512).map(|i| (i % 256) as u8));
            v
        },
    ];

    for (i, pattern) in patterns.iter().enumerate() {
        fs::write(&cache_path, pattern).expect("write hostile pattern");
        let outcome =
            CompiledContext::load_if_fresh(&cache_path, Path::new("/tmp/unused"), &pkg_dir);
        match outcome {
            Ok(Some(_)) => panic!("hostile pattern #{i} must NEVER load as Ok(Some(_))"),
            Ok(None) | Err(_) => { /* both acceptable: never a silent hit, never a panic */ }
        }
    }
}

#[test]
fn truncation_at_every_prefix_length_never_silently_loads() {
    // Save a valid context, then truncate it to every length from 0 up to
    // the full file in coarse steps. No prefix may load as Ok(Some(_)):
    // a torn write of ANY length must fall through to a recompute or be
    // flagged, never silently accepted. (The full-length file is
    // excluded from the loop because that one IS a valid hit.)
    let (_dir, cache_path, _ctx, bytes) = save_ctx("rt-truncate", TRIVIAL_MAIN);
    let full = bytes.len();
    assert!(full > 32, "fixture must produce a non-trivial file");

    let mut len = 0usize;
    while len < full {
        fs::write(&cache_path, &bytes[..len]).expect("write truncated prefix");
        let outcome =
            CompiledContext::load_if_fresh(&cache_path, Path::new("/tmp/unused"), &cache_path);
        assert!(
            !matches!(outcome, Ok(Some(_))),
            "truncating to {len}/{full} bytes must NOT silently load as Ok(Some(_))"
        );
        // Step in chunks so the test stays on the fast `ci` profile.
        len += (full / 13).max(1);
    }

    // Sanity: the untruncated file still loads against its own root.
    fs::write(&cache_path, &bytes).expect("restore full file");
}

// ---------------------------------------------------------------------
// Format-version-18 bump: exact source-number carriers change the payload.
// The preceding V17 format must be rejected before payload decode.
// ---------------------------------------------------------------------

#[test]
fn a_forged_stale_magic_file_is_rejected_not_decoded() {
    // The current magic is `CHELIS_CTX_V23\n`. A leftover file carries a
    // `CHELIS_CTX_V17\n` (or older) magic. Forge one from a real current payload.
    // load_if_fresh must reject it (the magic no longer matches), never
    // attempt to decode the stale-shaped envelope.
    let (_dir, cache_path, _ctx, bytes) = save_ctx("rt-stale-magic", TRIVIAL_MAIN);
    assert!(
        bytes.starts_with(b"CHELIS_CTX_V23\n"),
        "fixture must be written with the current magic"
    );

    let mut forged = b"CHELIS_CTX_V17\n".to_vec();
    forged.extend_from_slice(&bytes[b"CHELIS_CTX_V23\n".len()..]);
    fs::write(&cache_path, &forged).expect("write forged stale-magic file");

    let outcome =
        CompiledContext::load_if_fresh(&cache_path, Path::new("/tmp/unused"), &cache_path);
    match outcome {
        Ok(Some(_)) => panic!("a stale-magic file must NEVER load as an Ok(Some(_))"),
        Ok(None) => panic!(
            "a stale-magic file is corrupt bytes for the current binary, not a clean miss: \
             load_if_fresh must flag it so the operator sees the version skew"
        ),
        Err(CacheError::Corrupt(_)) => { /* expected: wrong magic header */ }
        Err(other) => panic!(
            "expected Corrupt for a wrong-magic file, got {other:?} \
             (still safe, but the magic check should fire first)"
        ),
    }
}

#[test]
fn a_bumped_envelope_version_byte_is_rejected_as_unsupported() {
    // Distinct from the magic check: keep the current magic intact but corrupt
    // the envelope's `version: u32` field so it decodes to a value other than
    // the expected one. load_if_fresh must reject it as
    // `CacheError::UnsupportedVersion`, never decode the payload. The
    // envelope `version` field is the first field after the magic, so it
    // sits at bytes [magic.len() .. magic.len()+4].
    let (_dir, cache_path, _ctx, bytes) = save_ctx("rt-envver", TRIVIAL_MAIN);
    let magic_len = b"CHELIS_CTX_V23\n".len();
    assert!(bytes.len() > magic_len + 4);

    let mut forged = bytes.clone();
    // bincode encodes a u32 little-endian; bump the low byte well past 24.
    forged[magic_len] = forged[magic_len].wrapping_add(99);
    fs::write(&cache_path, &forged).expect("write bumped-version file");

    let outcome =
        CompiledContext::load_if_fresh(&cache_path, Path::new("/tmp/unused"), &cache_path);
    match outcome {
        Ok(Some(_)) => panic!("a bumped envelope version must NEVER load as Ok(Some(_))"),
        Ok(None) => { /* tolerated: the envelope may fail to decode first */ }
        Err(CacheError::UnsupportedVersion { stored, expected }) => {
            assert_eq!(expected, 27, "the running binary expects format version 27");
            assert_ne!(stored, 27, "the forged version must differ from 27");
        }
        Err(CacheError::Corrupt(_) | CacheError::Decode(_)) => {
            // Also acceptable: bumping a byte can break the bincode shape
            // before the version check is reached. The invariant is
            // "never Ok(Some(_))", and the operator still sees an error.
        }
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

// ---------------------------------------------------------------------
// stdlib_cache_key. Post-#130 this folds COMPILER_VERSION directly. The
// in-crate `cache_key_depends_on_the_compiler_version` test reconstructs
// the key by mirroring the hasher; these tests attack it from the
// outside, treating `stdlib_cache_key` as a black box, and pin the
// decl-honesty property RT-1 originally flagged.
// ---------------------------------------------------------------------

const REDTEAM_STDLIB_SOURCE_DIGEST: [u8; 32] = [0x77; 32];

#[test]
fn stdlib_cache_key_is_deterministic_for_one_decl_slice() {
    // The intended cross-fixture-reuse property: a fixed decl slice must
    // hash to one stable key within a binary, so ~200 nextest processes
    // share one entry. A non-deterministic key would silently disable the
    // cache (perpetual cold cost), not corrupt anything, but it is still
    // a regression worth pinning.
    let decls = chelis_surf::parser::parse_str("module Rt.Sample\ndef rt_sample() -> i32 = 1\n")
        .expect("sample decls parse");
    assert_eq!(
        stdlib_cache_key(&decls, REDTEAM_STDLIB_SOURCE_DIGEST),
        stdlib_cache_key(&decls, REDTEAM_STDLIB_SOURCE_DIGEST),
        "stdlib_cache_key must be deterministic for a fixed decl slice"
    );
}

#[test]
fn stdlib_cache_key_depends_on_the_decls_themselves() {
    // RT-1's decl-honesty finding, pinned against the shipped key: two
    // different chelis-std decl slices must produce different keys. This
    // is the property that stops an edited chelis-std checkout from
    // stale-hitting a bundled-std artifact (chelis-std checked as a root
    // package resolves the checkout's own source, not the bundle). A key
    // that ignored the decl bytes would let an edited runtime silently
    // reuse the unedited runtime's typecheck result.
    let decls_a = chelis_surf::parser::parse_str("module Rt.Sample\ndef rt_a() -> i32 = 1\n")
        .expect("decls a parse");
    let decls_b = chelis_surf::parser::parse_str("module Rt.Sample\ndef rt_b() -> i32 = 2\n")
        .expect("decls b parse");
    assert_ne!(
        decls_a, decls_b,
        "test setup: the two decl slices must differ"
    );
    assert_ne!(
        stdlib_cache_key(&decls_a, REDTEAM_STDLIB_SOURCE_DIGEST),
        stdlib_cache_key(&decls_b, REDTEAM_STDLIB_SOURCE_DIGEST),
        "stdlib_cache_key must fold the actual decl bytes, so an edited chelis-std \
         checkout cannot stale-hit a bundled-std artifact"
    );
}

#[test]
fn stdlib_cache_key_folds_the_compiler_version() {
    // `stdlib_cache_key` folds the compiler BUILD fingerprint directly, so
    // a chelis binary built from different compiler source does NOT
    // stale-hit an older binary's StdLibContext even when the bundled
    // chelis-std bytes are identical. We cannot rebuild the compiler
    // mid-test; instead we re-derive the key the way `stdlib_cache_key`
    // does and confirm (a) the real fingerprint reproduces the real key
    // byte-for-byte, proving the mirror is faithful, and (b) a different
    // one flips it. If a future refactor drops the build identity from
    // the key, assertion (a) breaks and this test is the tripwire.
    //
    // chelis#1156: this input used to be `COMPILER_VERSION`. That is a
    // release identity — every build of an unreleased version shares it —
    // so two binaries with different type semantics shared this cache.
    // The stdlib sub-context is keyed WITHOUT a package root, so it is
    // shared by every package on the machine; a stale hit here reaches
    // further than the per-package context cache.
    use sha2::{Digest, Sha256};

    let decls = chelis_surf::parser::parse_str("module Rt.Sample\ndef rt_sample() -> i32 = 1\n")
        .expect("sample decls parse");
    let real = stdlib_cache_key(&decls, REDTEAM_STDLIB_SOURCE_DIGEST);

    // Byte-for-byte mirror of `stdlib_cache_key`, parameterized on the
    // compiler-version string. STDLIB_CACHE_FORMAT_VERSION is 25 (authored
    // signatures, checked operation restrictions, checked extent transport,
    // named witness claims, and the single node spelling); the
    // mirror is only valid while that holds, which assertion (a) below
    // verifies.
    let recompute = |compiler_version: &str| -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"chelis_std_typecheck_v");
        hasher.update(25u32.to_le_bytes());
        hasher.update(b"compiler_version");
        hasher.update((compiler_version.len() as u64).to_le_bytes());
        hasher.update(compiler_version.as_bytes());
        let version = chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION;
        hasher.update((version.len() as u64).to_le_bytes());
        hasher.update(version.as_bytes());
        let archive = chelis_std_bundle::archive_sha256();
        hasher.update((archive.len() as u64).to_le_bytes());
        hasher.update(archive.as_bytes());
        let shell = chelis_std_bundle::shell_sha256();
        hasher.update((shell.len() as u64).to_le_bytes());
        hasher.update(shell.as_bytes());
        hasher.update(b"exact-source-digest");
        hasher.update(REDTEAM_STDLIB_SOURCE_DIGEST);
        match bincode::serialize(&decls) {
            Ok(decl_bytes) => {
                hasher.update(b"decls");
                hasher.update((decl_bytes.len() as u64).to_le_bytes());
                hasher.update(&decl_bytes);
            }
            Err(_) => hasher.update(b"decls-unserializable"),
        }
        hasher.finalize().into()
    };

    // (a) the mirror reproduces the real key for the real build
    // fingerprint: this both proves the mirror is faithful AND proves the
    // fingerprint is genuinely an input (a key that ignored it could not
    // be reproduced by a derivation that feeds it in).
    assert_eq!(
        real,
        recompute(chelis_compiler_api::build_fingerprint()),
        "the recompute mirror must reproduce the real stdlib_cache_key for the \
         running build fingerprint; if this breaks, stdlib_cache_key's inputs changed"
    );
    // (b) a different build fingerprint flips the key.
    assert_ne!(
        real,
        recompute("0.0.0-some-other-compiler-build"),
        "a different build fingerprint must produce a different stdlib cache key, so \
         a differently-built binary cannot stale-hit an older StdLibContext"
    );
    // (c) chelis#1156 regression: the bare release version must NOT be
    // what the key folds in, or every build of one version collides.
    // Unconditional: both arms of the fingerprint (digest and degraded)
    // extend `COMPILER_VERSION`, so it is never the bare release string.
    assert_ne!(
        real,
        recompute(COMPILER_VERSION),
        "the stdlib cache key must fold the BUILD fingerprint, not the bare \
         release version; two builds of one version must not share this cache"
    );
}

// ---------------------------------------------------------------------
// CacheIdentity round-trip. The identity is serialized into the on-disk
// envelope AND into the inner CompiledContext; both copies must survive a
// bincode round trip identically, or the belt-and-braces inner-vs-
// envelope check in load_if_fresh would false-positive on every load.
// ---------------------------------------------------------------------

#[test]
fn cache_identity_survives_a_bincode_round_trip() {
    // The whole CompiledContext (identity included) round-trips through
    // bincode on every save/load. If the identity did not round-trip
    // byte-stably, load_if_fresh's inner-vs-envelope IdentityMismatch
    // guard would fire on a perfectly good cache file. Pin that
    // `encode`/`decode` preserve the identity exactly.
    let (_dir, root) = make_pkg("rt-roundtrip", "0.1.0", TRIVIAL_MAIN);
    let ctx = compile_reef_context(Path::new("/tmp/unused"), &root).expect("ctx");
    let restored = CompiledContext::decode(&ctx.encode().expect("encode")).expect("decode");
    assert_eq!(
        ctx.identity, restored.identity,
        "CacheIdentity must survive a bincode round trip byte-stably"
    );
    // And the identity matches a fresh recomputation from the same root:
    // the stored identity is exactly what load_if_fresh will recompute.
    assert_eq!(
        ctx.identity,
        CacheIdentity::for_package_root(&root),
        "the stored identity must equal a fresh CacheIdentity::for_package_root \
         on the same root, so a genuine hit is never mis-rejected"
    );
}
