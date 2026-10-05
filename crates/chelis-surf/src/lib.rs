//! Surface syntax parser and desugaring for the Chelis language.

pub mod ast;
pub mod decompile;
pub mod desugar;
mod dtype_name;
pub mod format;
pub mod lexer;
pub mod module_identity;
pub mod parser;
#[cfg(feature = "pre-020-pipe-migration")]
pub mod pipe_migration;
mod pipe_sugar;
pub mod resugar;
pub mod token;
