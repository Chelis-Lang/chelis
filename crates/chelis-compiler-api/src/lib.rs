pub(crate) mod cache_envelope;
pub mod compiler;
pub mod context;
pub mod decode;
pub mod fragment;
pub mod layered;
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
