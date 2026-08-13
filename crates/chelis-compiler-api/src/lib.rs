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
/// semantics, however, change between such builds — a released `0.18.2`
/// and a `0.18.2`-versioned `main` that has since landed a dtype change
/// disagree about what a program means. Keying a compiled-context cache
/// on the version alone lets those two binaries read each other's
/// entries, so a program is checked under the *other* build's semantics:
/// either a spurious rejection of valid code, or — worse — silent
/// acceptance of code this build would reject.
///
/// This fingerprint adds a discriminator derived from the running
/// executable (byte length + modification time), which changes on every
/// rebuild while staying stable across invocations of one binary. When
/// the executable cannot be inspected — `current_exe`, its metadata, or
/// its mtime read fails — the fingerprint fails toward cache MISSES,
/// never toward sharing: a per-process discriminator (pid plus a random
/// nonce, pinned by the same `OnceLock`) takes the inspection's place,
/// so a degraded binary simply never shares compiled contexts across
/// processes. That is a performance cost in a rare mode, not a
/// correctness cost; collapsing to bare [`COMPILER_VERSION`] instead
/// would let two uninspectable builds share entries again — the exact
/// defect this fingerprint exists to close. A same-binary copy that
/// lands at a different path or timestamp likewise only costs a cache
/// miss (recompile), never a stale hit.
pub fn build_fingerprint() -> &'static str {
    static FINGERPRINT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        let inspected = std::env::current_exe().ok().and_then(|exe| {
            let meta = std::fs::metadata(&exe).ok()?;
            let mtime_nanos = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())?;
            Some((meta.len(), mtime_nanos))
        });
        fingerprint_string(inspected)
    })
}

/// The fingerprint text for one inspection result. Split from
/// [`build_fingerprint`] so the degraded arm is testable: the production
/// path cannot be made to fail its metadata read on demand.
fn fingerprint_string(inspected: Option<(u64, u128)>) -> String {
    match inspected {
        Some((len, mtime_nanos)) => format!("{COMPILER_VERSION}+{len:x}.{mtime_nanos:x}"),
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
