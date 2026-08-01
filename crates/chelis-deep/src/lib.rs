//! Deep (s-expression) parser for the Chelis language.
//!
//! This crate provides the lexer, parser, AST types, and canonical printer
//! for Chelis Deep syntax (`.dp` files). Deep is the canonical s-expression
//! form that the compiler operates on internally.

pub mod ast;
pub mod authoring;
pub mod effect_kind;
pub mod lexer;
pub mod node;
pub mod parser;
pub mod path;
pub mod printer;
pub mod raw;
pub mod role;
pub mod span;
pub mod stamp_to_typed;
pub mod tag;
pub mod validate;

pub use ast::{Atom, Expr, List, MetaExpr, MetaMap, UnknownFormData};
pub use effect_kind::decode_effect_kind;
pub use lexer::LiteralSuffix;
pub use parser::{StampOrParseError, parse_and_stamp, parse_raw_str};
pub use raw::{RawAtom, RawExpr};
pub use stamp_to_typed::{StampError, StampErrorKind, stamp_to_typed};
pub use path::{
    DeepPath, InsertFunctionError, PathError, PathSegment, ResolveError, ResolvedFunction,
    function_body, function_defsig, insert_function_decls, module_excluding_function_def,
    module_has_defsig_for, resolve_function, splice_function_body, spliced_function_def,
};
pub use span::Span;
pub use tag::DeepTag;
