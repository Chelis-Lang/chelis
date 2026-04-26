use serde::{Deserialize, Serialize};

use crate::span::Span;

/// A Deep expression — the core AST node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// An atomic value (symbol, number, string, keyword, bool).
    Atom(Atom, Span),
    /// A parenthesized list `(tag {} children...)`.
    List(List, Span),
    /// An inline metadata map `{key: value, ...}` or `{}`.
    Map(MetaMap, Span),
    /// A metadata-annotated expression `^{k1 v1 ...} expr` (legacy, kept for compat).
    MetaExpr(MetaExpr, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Atom(_, s) | Expr::List(_, s) | Expr::Map(_, s) | Expr::MetaExpr(_, s) => *s,
        }
    }
}

/// An atomic (leaf) value in the AST.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Atom {
    Symbol(String),
    Int(i64),
    Float(f64),
    Str(String),
    /// Keyword without the leading `:`, e.g. `":axis"` → `"axis"`.
    Keyword(String),
    Bool(bool),
}

/// A parenthesized list of expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct List {
    /// All elements of the list. In canonical 3-tuple form:
    /// elements[0] is the tag, elements[1] is a Map (metadata),
    /// elements[2..] are children.
    pub elements: Vec<Expr>,
}

/// Inline metadata map: `{key: value, ...}` or `{}`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MetaMap {
    /// Key-value pairs. Keys are bare identifiers.
    pub entries: Vec<(String, Expr)>,
}

/// Legacy metadata: `^{k1 v1 ...} expr` (prefix form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaExpr {
    /// Key-value pairs. Keys are keyword strings (without `:`).
    pub entries: Vec<(String, Expr)>,
    /// The expression this metadata is attached to.
    pub expr: Box<Expr>,
}
