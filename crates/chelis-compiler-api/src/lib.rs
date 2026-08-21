#![forbid(unsafe_code)]

pub(crate) mod cache_envelope;
pub mod compiler;
pub mod context;
pub mod decode;
pub mod fragment;
pub mod layered;
pub mod library_cache;
pub mod pipeline;
pub mod prune;
pub(crate) mod runtime;
pub mod schema;
pub mod stdlib_cache;
pub mod target_capability;

#[cfg(test)]
mod source_arch;

/// Re-export of reef package-root discovery so callers (e.g. the Python
/// bindings' `compile_and_load` auto-discovery) can resolve the enclosing
/// reef project without depending on `chelis-reef` directly. See issue #816.
pub use chelis_reef::{find_package_root_for_dir, find_package_root_for_input};
/// Cooperative cancellation for long-running evaluation (chelis#914).
/// Install a token on the thread that will run the eval, hand a clone to
/// whoever may want to stop it, and the eval unwinds with
/// [`EVAL_CANCELLED_MSG`] within one node visit of the request.
pub use chelis_types::{
    CancelToken, CancelTokenGuard, EVAL_CANCELLED_MSG, current_cancel_token, install_cancel_token,
    is_cancellation,
};
pub use chelis_types::{LinkedProgramGuard, install_linked_program_guard};
pub use compiler::{
    EVAL_CANCELLED_KIND, PreparedEvalInContext, check_in_context, compile_for_execution_in_context,
    eval_in_context, eval_in_context_with_bindings, eval_many_in_context, prepare_eval_in_context,
    surf_source_has_import,
};
pub use compiler::{add_function, replace_function_body};
pub use context::{
    CacheError, CacheIdentity, CompiledContext, ContextHash, ContextLoadPath, compile_reef_context,
    load_or_compile_for_package, load_or_compile_with_local_registry_fallback,
};
/// Experimental decode chokepoint for opaque-type invariant revalidation
/// (RFC `opaque_invariants_rfc.md` D-DECODE). No production codec consumes
/// it in V1 -- see `decode` module docs.
pub use decode::{DecodeError, decode_adt_value, try_decode_adt_value};
pub use fragment::{
    DeepErrorPath, EditValidationError, ReplacementError, ReplacementReport, ValidatedModule,
    check_body_replacement, check_whole_module_edit,
};
pub use layered::{LayeredCheck, check_layered, check_layered_for_build, stdlib_structural_stats};
pub use library_cache::{
    LibraryContext, build_library_context, library_cache_key, load_or_build_library_context,
};
/// The host-runtime value type returned by the decode chokepoint.
/// Experimental: surfaced for the decode contract point; its shape is not
/// yet a stable public commitment (V1 has no production decode caller).
pub use runtime::RuntimeValue;
pub use stdlib_cache::{
    StdLibContext, build_stdlib_context, cache_disabled, load_or_build_stdlib_context,
    stdlib_cache_key, typecheck_cache_dir,
};

/// Pinned compiler version for fixture `reef.toml` files in tests and for
/// the `package.compiler` tripwire on real `.toml` files. This auto-syncs
/// with `workspace.package.version` because every workspace crate uses
/// `version.workspace = true`, so `CARGO_PKG_VERSION` here equals the
/// workspace version at build time.
///
/// Single source of truth for the workspace compiler pin string used by
/// integration tests across `chelis-cli` and `chelis-compiler-api`.
pub const COMPILER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Per-BUILD identity of the running compiler, for cache keying.
///
/// [`COMPILER_VERSION`] identifies a *release*, not a *build*: every
/// binary compiled from any commit that carries the same
/// `workspace.package.version` reports the same string. Type-checking
/// semantics, however, change between such builds - a released `0.18.2`
/// and a `0.18.2`-versioned `main` that has since landed a dtype change
/// disagree about what a program means. Keying a compiled-context cache
/// on the version alone lets those two binaries read each other's
/// entries, so a program is checked under the *other* build's semantics:
/// either a spurious rejection of valid code, or, worse, silent
/// acceptance of code this build would reject.
///
/// The discriminator is the SHA-256 of the running executable's own
/// bytes. Content *is* build identity: two binaries with identical bytes
/// cannot disagree about anything, so sharing a cache entry between them
/// is correct rather than merely tolerable; and any difference that
/// could change type semantics - different compiler source, a different
/// `rustc`, different codegen flags or features - necessarily changes
/// those bytes and so changes the key. There is no false merge and no
/// false split.
///
/// # Why not filesystem metadata
///
/// An earlier revision of this function used the executable's byte
/// length plus its mtime. That infers build identity from metadata that
/// several shipping distribution paths deliberately normalize, so the
/// discriminator silently degrades to byte length alone:
///
/// - Nix canonicalizes every store file's mtime to exactly 1 second past
///   the epoch. Measured on nix 2.35.1: 3000 of 3000 sampled
///   `/nix/store/*/bin/*` executables reported `mtime_ns == 1_000_000_000`,
///   and a probe derivation that explicitly `touch`ed its output to a
///   2026 timestamp still landed in the store with mtime 1. `flake.nix`
///   exports `packages.chelis` (and `default = chelis`), so this is a
///   first-class channel for this compiler, not a hypothetical one.
/// - `cp -p`, `rsync -t`, and tar/OCI layers with preserved timestamps
///   reproduce the same normalization on any platform.
///
/// That reduction is not theoretical. On the machine where this was
/// measured, 394 distinct `(byte length, mtime)` pairs were each shared
/// by two or more store executables whose SHA-256 digests differ,
/// including two different `cargo` binaries of 31,232,680 bytes apiece.
/// Under the old scheme those would have been one cache identity, which
/// is the silent-acceptance half of chelis#1156 reopened.
///
/// # Why not a `build.rs` stamp
///
/// A git SHA plus a dirty flag is cheap and metadata-immune, but it
/// merges builds that differ: every dirty-tree rebuild from one commit
/// reports one fingerprint, which is precisely the compiler developer's
/// inner loop and the place type semantics change most often. It is also
/// stale-prone, because cargo reruns a package's `build.rs` for changes
/// to *that package's* files: editing `chelis-types` relinks this crate
/// without rerunning its build script, leaving a stamp that claims two
/// semantically different binaries are the same build. Both failures are
/// in the unsound direction. Hashing the finished artifact has neither
/// problem, and subsumes the `rustc`-version concern for free.
///
/// # Cost
///
/// One SHA-256 pass over the executable, memoized by `OnceLock` to once
/// per process and reached lazily: [`stdlib_cache::cache_disabled`] and
/// the layered-build entry both short-circuit before any cache key is
/// computed, so a run with the cache off never pays it. Measured at
/// roughly 60ms for a 30MB release `chelis`, against a cache whose
/// purpose is to avoid seconds of stdlib type-checking.
///
/// # Degraded path
///
/// When the executable cannot be read - `current_exe`, `open`, or the
/// read itself fails - the fingerprint fails toward cache MISSES, never
/// toward sharing: a per-process discriminator (pid plus a random nonce,
/// pinned by the same `OnceLock`) takes the digest's place, so a
/// degraded binary simply never shares compiled contexts across
/// processes. Collapsing to bare [`COMPILER_VERSION`] instead would let
/// two uninspectable builds share entries again, the exact defect this
/// fingerprint exists to close. Because that mode turns every invocation
/// into a guaranteed full cache miss, and so presents as unexplained
/// recompile-every-run slowness, it also emits a one-time stderr
/// warning naming itself (chelis#1156 review F4).
pub fn build_fingerprint() -> &'static str {
    static FINGERPRINT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        let inspected = std::env::current_exe().ok().and_then(|exe| hash_file(&exe));
        if inspected.is_none() {
            // Diagnosable, not silent: this path is safe (it can only
            // cause misses) but it disables the compiled-context cache
            // outright, and without a breadcrumb that reads as the
            // compiler having become mysteriously slow.
            eprintln!(
                "chelis: warning: could not read the running executable to identify \
                 this compiler build, so compiled-context caches cannot be shared \
                 between invocations and every run will rebuild them (chelis#1156)."
            );
        }
        fingerprint_string(inspected)
    })
}

/// SHA-256 of a file's full contents, or `None` if it cannot be read.
///
/// Chunked rather than `io::copy` into the hasher so this does not depend
/// on `sha2`'s `std` feature staying enabled by feature unification.
fn hash_file(path: &std::path::Path) -> Option<[u8; 32]> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
    Some(hasher.finalize().into())
}

/// The fingerprint text for one inspection result. Split from
/// [`build_fingerprint`] so the degraded arm is testable: the production
/// path cannot be made to fail its read on demand.
fn fingerprint_string(inspected: Option<[u8; 32]>) -> String {
    match inspected {
        Some(digest) => {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut hex = String::with_capacity(digest.len() * 2);
            for b in digest {
                hex.push(HEX[(b >> 4) as usize] as char);
                hex.push(HEX[(b & 0xf) as usize] as char);
            }
            format!("{COMPILER_VERSION}+{hex}")
        }
        None => {
            // Fail toward misses, not sharing (see `build_fingerprint`).
            // The nonce comes from `RandomState`, whose per-instance keys
            // are process-random, so two degraded processes disagree even
            // under pid reuse.
            use std::hash::{BuildHasher, Hasher};
            let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
            hasher.write_u32(std::process::id());
            let nonce = hasher.finish();
            format!(
                "{COMPILER_VERSION}+degraded.{pid:x}.{nonce:x}",
                pid = std::process::id()
            )
        }
    }
}

#[cfg(test)]
mod build_fingerprint_tests {
    /// chelis#1156: the fingerprint must be stable within one process
    /// (otherwise every invocation is a cache miss and the compiled-
    /// context cache stops working at all).
    #[test]
    fn build_fingerprint_is_stable_across_calls() {
        assert_eq!(super::build_fingerprint(), super::build_fingerprint());
    }

    /// It must remain a superset of the release identity: a version bump
    /// alone still changes the key.
    #[test]
    fn build_fingerprint_carries_the_compiler_version() {
        assert!(
            super::build_fingerprint().starts_with(super::COMPILER_VERSION),
            "fingerprint {} must extend the compiler version {}",
            super::build_fingerprint(),
            super::COMPILER_VERSION
        );
    }

    /// chelis#1156 review: when the executable cannot be inspected the
    /// fingerprint must fail toward cache MISSES, never toward sharing.
    /// Collapsing to the bare release version would let two
    /// uninspectable builds share compiled contexts again, in the one
    /// code path where nothing would ever report that it happened.
    #[test]
    fn degraded_fingerprint_does_not_collapse_to_the_release_version() {
        let degraded = super::fingerprint_string(None);
        assert_ne!(degraded, super::COMPILER_VERSION);
        assert!(
            degraded.starts_with(&format!("{}+degraded.", super::COMPILER_VERSION)),
            "degraded fingerprint must be marked as such, got {degraded}"
        );
    }

    /// The production fingerprint really is the digest of the running
    /// executable, not merely *some* string derived from it. Recomputed
    /// independently here, so a refactor that quietly reverted the
    /// derivation to metadata (or to a constant) breaks this test rather
    /// than passing silently.
    #[test]
    fn build_fingerprint_is_the_digest_of_the_running_executable() {
        let exe = std::env::current_exe().expect("test binary must be locatable");
        let digest = super::hash_file(&exe).expect("test binary must be readable");
        assert_eq!(
            super::build_fingerprint(),
            super::fingerprint_string(Some(digest))
        );
    }

    /// chelis#1156 (PR #1161 review, F1): the discriminator must come
    /// from the executable's CONTENT, never from filesystem metadata.
    ///
    /// Nix canonicalizes every store file's mtime to exactly 1 second
    /// past the epoch, and `cp -p` / `rsync -t` / tar and OCI layers
    /// preserve timestamps across genuinely different builds. A
    /// metadata-derived fingerprint therefore collapses to byte length
    /// alone on those paths. This pins the property directly: identical
    /// bytes at two different paths with two different mtimes (one of
    /// them the literal Nix value) must produce one fingerprint.
    #[test]
    fn fingerprint_ignores_path_and_mtime() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::write(&a, b"identical build output").expect("write a");
        std::fs::write(&b, b"identical build output").expect("write b");
        set_mtime(&a, 1);
        set_mtime(&b, 1_755_000_000);

        let (da, db) = (super::hash_file(&a), super::hash_file(&b));
        assert_eq!(da, db, "identical content must fingerprint identically");
        assert_eq!(
            super::fingerprint_string(da),
            super::fingerprint_string(db),
            "path and mtime must not reach the fingerprint"
        );
    }

    /// The converse, and the actual chelis#1156 failure shape: two
    /// DIFFERENT builds that share a byte length and an mtime must not
    /// share a fingerprint. This is the exact collision the old
    /// `{len:x}.{mtime:x}` scheme could not see. It is not hypothetical:
    /// sampling `/nix/store` on the machine where this was written found
    /// 394 `(length, mtime)` pairs each shared by two or more
    /// content-differing executables, every one of them at mtime 1, the
    /// value both files carry here.
    #[test]
    fn fingerprint_separates_builds_sharing_a_length_and_mtime() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::write(&a, b"build one, semantics A").expect("write a");
        std::fs::write(&b, b"build two, semantics B").expect("write b");
        assert_eq!(
            std::fs::metadata(&a).expect("meta a").len(),
            std::fs::metadata(&b).expect("meta b").len(),
            "fixture must hold byte length constant, or it tests nothing"
        );
        set_mtime(&a, 1);
        set_mtime(&b, 1);

        assert_ne!(
            super::fingerprint_string(super::hash_file(&a)),
            super::fingerprint_string(super::hash_file(&b)),
            "two different builds must not share a fingerprint just because \
             a store normalized their timestamps to a common value"
        );
    }

    /// Set a file's mtime to `secs` past the epoch. `File::set_modified`
    /// keeps this dependency-free; `1` is the value Nix stamps on every
    /// store file.
    fn set_mtime(path: &std::path::Path, secs: u64) {
        let f = std::fs::File::options()
            .write(true)
            .open(path)
            .expect("open for set_modified");
        f.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .expect("set_modified");
        assert_eq!(
            std::fs::metadata(path)
                .expect("meta")
                .modified()
                .expect("mtime")
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .expect("post-epoch")
                .as_secs(),
            secs,
            "the fixture must actually control mtime, or these tests are vacuous"
        );
    }

    /// The degraded discriminator must not be a constant: two draws must
    /// differ, so a degraded binary never shares compiled contexts across
    /// processes. Distinctness across two draws in ONE process is a
    /// stronger property than the cross-process one it stands in for; the
    /// production path pins one draw per process through its `OnceLock`,
    /// which `build_fingerprint_is_stable_across_calls` above locks.
    #[test]
    fn degraded_fingerprints_are_distinct_per_draw() {
        assert_ne!(
            super::fingerprint_string(None),
            super::fingerprint_string(None),
            "the degraded discriminator must not be a constant"
        );
    }
}
