//! Type checker for the Chelis language.

pub mod adt;
pub mod env;
pub mod errors;
pub mod infer;
pub mod types;
pub mod unify;

mod builtins;
pub use builtins::builtin_env;
pub use infer::{InferResult, infer_program};
