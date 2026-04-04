use crate::span::Span;

/// A Deep expression — the core AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// An atomic value (symbol, number, string, keyword, bool).
    Atom(Atom, Span),
    /// A parenthesized list `(tag child1 child2 ...)`.
    List(List, Span),
    /// A metadata-annotated expression `^{k1 v1 ...} expr`.
    MetaExpr(MetaExpr, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Atom(_, s) | Expr::List(_, s) | Expr::MetaExpr(_, s) => *s,
        }
    }
}

/// An atomic (leaf) value in the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Atom {
    Symbol(String),
    Int(i64),
    Float(f64),
    Str(String),
    /// Keyword without the leading `:`, e.g. `":axis"` → `"axis"`.
    Keyword(String),
    Bool(bool),
}

/// A parenthesized list with a tag.
#[derive(Debug, Clone, PartialEq)]
pub struct List {
    /// The tag (first symbol in the list).
    pub tag: String,
    /// The remaining child expressions.
    pub children: Vec<Expr>,
}

/// Metadata map plus the annotated expression.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaExpr {
    /// Key-value pairs. Keys are keyword strings (without `:`).
    pub entries: Vec<(String, Expr)>,
    /// The expression this metadata is attached to.
    pub expr: Box<Expr>,
}
