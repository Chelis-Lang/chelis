//! Surface syntax parser and desugaring for the Chelis language.
//!
//! This crate uses `alloc` without the standard library.
//! Parser, desugar, and source-format APIs need an explicit stack service.
//! Native callers can use the host adapter:
//!
//! ```
//! use chelis_syntax_host::NativeStack;
//!
//! let stack = NativeStack;
//! let declarations = chelis_surf::parser::parse_str("def identity(x) = x", &stack)?;
//! let deep = chelis_surf::desugar::desugar_program(&declarations, &stack)?;
//! let source = chelis_surf::decompile::try_decompile_program(&deep, &stack)?;
//! assert_eq!(source, "def identity(x) = x\n");
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```
#![no_std]

#[macro_use]
extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod ast;
pub mod decompile;
pub mod desugar;
mod dtype_name;
pub mod format;
pub mod lexer;
mod literal_sites;
pub mod parser;
#[cfg(feature = "pre-020-pipe-migration")]
pub mod pipe_migration;
mod pipe_sugar;
pub mod resugar;
pub mod token;
