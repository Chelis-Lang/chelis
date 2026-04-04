//! Deep (s-expression) parser for the Chelis language.
//!
//! This crate provides the lexer, parser, AST types, and canonical printer
//! for Chelis Deep syntax (`.dp` files). Deep is the canonical s-expression
//! form that the compiler operates on internally.

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod printer;
pub mod span;

pub use ast::{Atom, Expr, List, MetaExpr};
pub use span::Span;
