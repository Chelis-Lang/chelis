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
    prepare_rewritten_entry_batch_in_context, surf_source_has_import,
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
/// The discriminator comes from [`chelis_image_id::running_image`], which
/// answers "which build of which object is executing this code?" without
/// consulting filesystem metadata. It is the linker's own content id
/// (Mach-O `LC_UUID`, ELF `NT_GNU_BUILD_ID`) where the image carries one,
/// and a SHA-256 of the whole image otherwise; the scheme tag and the
/// image length are folded in alongside it so the two derivations can
/// never be confused and a size change cannot pass unnoticed.
///
/// Note that a linker id covers the MAPPED image, not the whole file, so
/// it deliberately ignores debug-map and symbol-table churn that cannot
/// change what the compiler computes. That is the correct granularity
/// here: this cache stores type-checking results, and two images with
/// identical code and data produce identical results. See
/// [`chelis_image_id::ImageId`] for the measured evidence in both
/// directions.
///
/// It also asks the loader which object it is running inside rather than
/// which process is hosting it. `chelis-python` is a `cdylib`, where
/// `std::env::current_exe()` names the Python interpreter: keying on that
/// would merge two different binding builds under one interpreter and
/// split one build across two virtualenvs.
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
/// by two or more store executables whose SHA-256 digests differ. It also
/// reproduced on this compiler directly: two `chelis` builds differing by
/// one added function came out at *identical* byte length, because the
/// added symbol landed in padding.
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
/// in the unsound direction. Deriving from the finished image has
/// neither problem, and subsumes the `rustc`-version concern for free.
///
/// # Cost
///
/// Memoized by `OnceLock` to once per process. On a linker id it is a
/// header read, microseconds regardless of image size. On the digest
/// fallback it is one SHA-256 pass over the image, which for a 138MB
/// debug `chelis` costs roughly 250ms; that is why the linker id is
/// preferred rather than treated as an optimization.
///
/// This is NOT gated on [`stdlib_cache::cache_disabled`]. That seam gates
/// the stdlib sub-context, but [`context::CacheIdentity::for_package_root`]
/// derives an identity unconditionally, so a run with the cache disabled
/// still resolves the fingerprint once.
///
/// # Degraded path
///
/// When the running image cannot be identified at all, the fingerprint
/// fails toward cache MISSES, never toward sharing: a per-process
/// discriminator (pid plus a random nonce, pinned by the same `OnceLock`)
/// takes its place, so a degraded binary simply never shares compiled
/// contexts across processes. Collapsing to bare [`COMPILER_VERSION`]
/// instead would let two unidentifiable builds share entries again, the
/// exact defect this fingerprint exists to close. Because that mode turns
/// every invocation into a guaranteed full cache miss, and so presents as
/// unexplained recompile-every-run slowness, it also emits a one-time
/// stderr warning naming itself.
pub fn build_fingerprint() -> &'static str {
    static FINGERPRINT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        let image = chelis_image_id::running_image();
        if image.is_none() {
            // Diagnosable, not silent: this path is safe (it can only
            // cause misses) but it disables the compiled-context cache
            // outright, and without a breadcrumb that reads as the
            // compiler having become mysteriously slow.
            //
            // Deliberately not `eprintln!`: that panics if stderr is
            // closed or full, and panicking inside a `OnceLock`
            // initializer would defeat the graceful degradation this arm
            // exists to provide.
            use std::io::Write as _;
            let _ = writeln!(
                std::io::stderr(),
                "chelis: warning: could not identify the running compiler image, so \
                 compiled-context caches cannot be shared between invocations and \
                 every run will rebuild them (chelis#1156)."
            );
        }
        fingerprint_string(image.as_ref())
    })
}

/// The fingerprint text for one identification result. Split from
/// [`build_fingerprint`] so the degraded arm is testable: the production
/// path cannot be made to fail identification on demand.
fn fingerprint_string(image: Option<&chelis_image_id::RunningImage>) -> String {
    match image {
        Some(image) => format!(
            "{COMPILER_VERSION}+{scheme}.{id}.{len:x}",
            scheme = image.id.scheme(),
            id = image.id.hex(),
            len = image.len
        ),
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

    /// The fingerprint is derived from the identified running image.
    ///
    /// This is a mirror: it recomputes through the same
    /// `chelis_image_id::running_image`, so it CANNOT catch a change to
    /// how that function derives an id. What it does catch is this crate
    /// wiring the fingerprint to something else entirely, silently
    /// falling into the degraded arm, or returning a constant. The
    /// derivation itself is pinned by `chelis-image-id`'s own tests,
    /// which compare content against metadata directly.
    #[test]
    fn build_fingerprint_is_derived_from_the_running_image() {
        let image = chelis_image_id::running_image().expect("test binary must be identifiable");
        assert_eq!(
            super::build_fingerprint(),
            super::fingerprint_string(Some(&image))
        );
        assert!(
            !super::build_fingerprint().contains("degraded"),
            "the production path must not be taking the degraded arm"
        );
    }

    /// The fast path must actually be applying. If a refactor loses the
    /// linker id, everything stays correct but every process starts
    /// paying a full SHA-256 over the image (about 250ms for a debug
    /// `chelis`), which is a silent performance cliff rather than a
    /// visible failure.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn build_fingerprint_uses_the_linker_build_id_fast_path() {
        let fp = super::build_fingerprint();
        assert!(
            fp.contains("+bid."),
            "expected the linker-build-id scheme in {fp}; falling back to the \
             whole-image digest is correct but costs a full hash per process"
        );
    }

    /// chelis#1156 review: when the running image cannot be identified
    /// the fingerprint must fail toward cache MISSES, never toward
    /// sharing. Collapsing to the bare release version would let two
    /// unidentifiable builds share compiled contexts again, in the one
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

    /// Two images that differ only in their id must not share a
    /// fingerprint, and the scheme tag must keep a linker id and a digest
    /// apart even when their hex happens to coincide.
    #[test]
    fn distinct_images_produce_distinct_fingerprints() {
        use chelis_image_id::{ImageId, RunningImage};
        let base = RunningImage {
            path: std::path::PathBuf::from("/nowhere"),
            len: 1024,
            id: ImageId::LinkerBuildId(vec![0xaa; 16]),
        };
        let other_id = RunningImage {
            id: ImageId::LinkerBuildId(vec![0xbb; 16]),
            ..base.clone()
        };
        let other_len = RunningImage {
            len: 2048,
            ..base.clone()
        };
        // Same hex bytes, different scheme: the tag is what separates them.
        let as_digest = RunningImage {
            id: ImageId::ContentDigest([0xaa; 32]),
            ..base.clone()
        };
        let fp = |i: &RunningImage| super::fingerprint_string(Some(i));
        assert_ne!(fp(&base), fp(&other_id), "a different id must flip the key");
        assert_ne!(
            fp(&base),
            fp(&other_len),
            "a different length must flip the key"
        );
        assert_ne!(
            fp(&base),
            fp(&as_digest),
            "a linker id and a digest must never collide"
        );
        // The path is diagnostics only and must NOT reach the key, or a
        // relocated identical binary would orphan its own cache.
        let moved = RunningImage {
            path: std::path::PathBuf::from("/somewhere/else"),
            ..base.clone()
        };
        assert_eq!(
            fp(&base),
            fp(&moved),
            "the image path must not reach the key"
        );
    }
}
