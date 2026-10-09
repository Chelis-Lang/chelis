//! Raw AST — the parser's output before role-directed stamping.
//!
//! `RawExpr` admits arbitrary list heads, has no role typing, and no
//! construction gate. The stamp pass (`stamp_to_typed`) converts
//! `Vec<RawExpr>` → `Result<Vec<Expr>, StampError>` via a top-down
//! role-directed walk. This is technique C (staging) in minimal form.

use crate::span::Span;

/// A raw atom as produced by the lexer/parser.
///
/// No `Keyword` (map keys are String; bare `:kw` is a parse error).
/// No `Tag` (decode happens at stamp time, not parse time).
#[derive(Debug, Clone, PartialEq)]
pub enum RawAtom {
    Symbol(String),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// A raw expression — the parser's output before stamping.
///
/// Lists have arbitrary heads (no vocabulary gate). The stamp pass
/// inspects each list's head in context of its parent's role expectation.
#[derive(Debug, Clone, PartialEq)]
pub enum RawExpr {
    /// Lexically parsed producer data; stamping it as program syntax is invalid.
    ExtensionData(crate::ExtensionData),
    Atom(RawAtom, Span),
    /// A parenthesized list with arbitrary contents.
    List(Vec<RawExpr>, Span),
    /// An inline metadata map `{key: value, ...}`.
    Map(Vec<(String, RawExpr)>, Span),
    /// Legacy prefix metadata `^{k1 v1 ...} expr`.
    MetaExpr {
        entries: Vec<(String, RawExpr)>,
        expr: Box<RawExpr>,
        span: Span,
    },
}

impl RawExpr {
    pub fn span(&self) -> Span {
        match self {
            RawExpr::ExtensionData(data) => data.span(),
            RawExpr::Atom(_, s) | RawExpr::List(_, s) | RawExpr::Map(_, s) => *s,
            RawExpr::MetaExpr { span, .. } => *span,
        }
    }
}

/// Release raw source arguments held by typed annotation storage without
/// recursing through a deeply nested list or legacy metadata wrapper.
pub(crate) fn drain_owned_source_arguments(mut pending: Vec<RawExpr>) {
    while let Some(expression) = pending.pop() {
        match expression {
            RawExpr::ExtensionData(_) | RawExpr::Atom(..) => {}
            RawExpr::List(children, _) => pending.extend(children.into_iter().rev()),
            RawExpr::Map(entries, _) => {
                pending.extend(entries.into_iter().rev().map(|(_, value)| value));
            }
            RawExpr::MetaExpr { entries, expr, .. } => {
                pending.push(*expr);
                pending.extend(entries.into_iter().rev().map(|(_, value)| value));
            }
        }
    }
}
