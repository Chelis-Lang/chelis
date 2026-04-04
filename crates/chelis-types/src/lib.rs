//! Type checker for the Chelis language.

pub mod env;
pub mod types;
pub mod unify;

mod builtins;
pub use builtins::builtin_env;
