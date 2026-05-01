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

    /// Return the external-source span identifier carried in this node's
    /// metadata map under the `span` key, if any.
    ///
    /// Per `spec/03-deep-syntax.md` §1.1.1 and
    /// `spec/design/chelis_span_survival.md`, every Deep AST node is a
    /// 3-tuple `(tag {meta} children...)` with the metadata map at element
    /// index 1. The `span` key holds an opaque string identifier issued by
    /// an external producer (e.g., Octant's LaTeX-to-Deep translator).
    ///
    /// Returns `Some(id)` when:
    /// - the node is an `Expr::List` with at least two elements,
    /// - element 1 is an `Expr::Map`,
    /// - that map contains a `span` entry whose value is a string literal
    ///   (`Expr::Atom(Atom::Str(_), _)`).
    ///
    /// The empty string is a valid (though unusual) span ID and is returned
    /// as `Some("")`. Lock this convention in tests; do not silently coerce
    /// `Some("")` to `None`.
    ///
    /// Returns `None` for atoms, bare maps, legacy `MetaExpr` nodes, lists
    /// without a metadata map at index 1, lists whose metadata map has no
    /// `span` key, or `span` values that are not string literals (those are
    /// shape errors callers handle separately, not a missing span).
    pub fn span_id(&self) -> Option<&str> {
        let list = match self {
            Expr::List(list, _) => list,
            _ => return None,
        };
        let meta = match list.elements.get(1)? {
            Expr::Map(m, _) => m,
            _ => return None,
        };
        for (key, value) in &meta.entries {
            if key == "span"
                && let Expr::Atom(Atom::Str(s), _) = value
            {
                return Some(s.as_str());
            }
        }
        None
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
