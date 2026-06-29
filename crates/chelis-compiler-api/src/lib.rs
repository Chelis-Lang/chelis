pub(crate) mod cache_envelope;
pub mod compiler;
pub mod context;
pub mod decode;
pub mod fragment;
pub mod layered;
pub mod prune;
pub(crate) mod runtime;
pub mod schema;
pub mod stdlib_cache;

pub use chelis_types::{LinkedProgramGuard, install_linked_program_guard};
pub use compiler::{
    PreparedEvalInContext, check_in_context, eval_in_context, eval_many_in_context,
    prepare_eval_in_context,
};
pub use compiler::{add_function, replace_function_body};
pub use context::{
    CacheError, CacheIdentity, CompiledContext, ContextHash, compile_reef_context,
    load_or_compile_for_package,
};
/// Experimental decode chokepoint for opaque-type invariant revalidation
/// (RFC `opaque_invariants_rfc.md` D-DECODE). No production codec consumes
/// it in V1 -- see `decode` module docs.
pub use decode::{DecodeError, decode_adt_value, try_decode_adt_value};
pub use fragment::{
    DeepErrorPath, EditValidationError, EditValidationReport, ReplacementError, ReplacementReport,
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
