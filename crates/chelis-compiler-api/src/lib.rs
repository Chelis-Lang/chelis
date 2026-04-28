pub mod compiler;
pub mod context;
pub(crate) mod runtime;
pub mod schema;

pub use compiler::{
    PreparedEvalInContext, check_in_context, eval_in_context, eval_many_in_context,
    prepare_eval_in_context,
};
pub use context::{
    CacheError, CompiledContext, ContextHash, compile_reef_context, load_or_compile_for_package,
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
