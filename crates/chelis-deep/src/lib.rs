//! Deep (s-expression) parser for the Chelis language.
//!
//! This crate provides the lexer, parser, AST types, and canonical printer
//! for Chelis Deep syntax (`.dp` files). Deep is the canonical s-expression
//! form that the compiler operates on internally.

pub mod ast;
pub mod authoring;
pub mod dtype_bounds;
pub mod effect_kind;
pub mod lexer;
pub mod literal_source;
pub mod node;
pub mod parser;
pub mod path;
mod pattern;
pub mod printer;
pub mod raw;
pub mod role;
pub mod span;
pub mod stamp_to_typed;
pub mod tag;
pub mod validate;

pub use ast::{Atom, CastMode, Expr, List, MetaExpr, MetaMap, UnknownFormData, cast_mode_of};
pub use dtype_bounds::{
    DTYPE_BOUNDS_KEY, DtypeBoundsError, DtypeFamily, decode_dtype_bounds, encode_dtype_bounds,
};
pub use effect_kind::decode_effect_kind;
pub use lexer::LiteralSuffix;
pub use literal_source::{
    BinderLiteralUse, LiteralFamilyFit, LiteralSource, classify_literal_source,
    exact_type_variable_name, visit_binder_literal_uses,
};
pub use parser::{
    StampOrParseError, parse_and_stamp, parse_and_stamp_file, parse_and_stamp_runtime_exprs,
    parse_and_stamp_tagged, parse_and_stamp_type, parse_raw_str,
};
pub use path::{
    DeepPath, InsertFunctionError, PathError, PathSegment, ResolveError, ResolvedFunction,
    function_body, function_defsig, insert_function_decls, module_excluding_function_def,
    module_has_defsig_for, resolve_function, splice_function_body, spliced_function_def,
};
pub use pattern::pattern_binder_names;
pub use raw::{RawAtom, RawExpr};
pub use span::Span;
pub use stamp_to_typed::{
    FormClass, FormIdentity, StampError, StampErrorKind, stamp_as_tagged, stamp_deep_file,
    stamp_runtime_exprs, stamp_to_typed,
};
pub use tag::DeepTag;
