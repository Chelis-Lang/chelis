//! Test-only constructors for adversarial checker-boundary mutations.

use chelis_deep::ast::{Atom, Expr, List, MetaExpr, MetaMap};
use chelis_deep::{RawAtom, RawExpr};

/// Parse Deep syntax without running the stamped `Node` constructor.
///
/// Production parsing must reject malformed arity and role shapes before the
/// checker sees them. A few totality tests also need to prove the checker
/// fails closed when a programmatic/legacy caller bypasses that boundary.
/// This test-only converter recreates that mutation explicitly and then
/// decodes known list heads with the deprecated legacy tag stamper.
///
/// This replaces an earlier seam that parsed a valid program and then widened
/// one node through `Node`'s mutable child API. That API no longer exists:
/// the successor carrier revalidates every candidate before commit, so a
/// wrong-arity `Node` is unconstructible. The legacy `List` carrier is the
/// remaining way to hand the checker a malformed tree.
pub(crate) fn parse_unchecked_legacy(source: &str) -> Vec<Expr> {
    let raw = chelis_deep::parse_raw_str(source).expect("adversarial Deep syntax must lex/parse");
    let mut exprs: Vec<_> = raw.into_iter().map(raw_to_legacy).collect();
    chelis_deep::parser::stamp_tags(&mut exprs);
    exprs
}

fn raw_to_legacy(raw: RawExpr) -> Expr {
    match raw {
        RawExpr::Atom(atom, span) => Expr::Atom(
            match atom {
                RawAtom::Symbol(value) => Atom::Name(value),
                RawAtom::Int(value) => Atom::Int(value),
                RawAtom::Float(value) => Atom::Float(value),
                RawAtom::Str(value) => Atom::Str(value),
                RawAtom::Bool(value) => Atom::Bool(value),
            },
            span,
        ),
        RawExpr::List(elements, span) => Expr::List(
            List {
                elements: elements.into_iter().map(raw_to_legacy).collect(),
            },
            span,
        ),
        RawExpr::Map(entries, span) => Expr::Map(
            MetaMap {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| (key, raw_to_legacy(value)))
                    .collect(),
            },
            span,
        ),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => Expr::MetaExpr(
            MetaExpr {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| (key, raw_to_legacy(value)))
                    .collect(),
                expr: Box::new(raw_to_legacy(*expr)),
            },
            span,
        ),
    }
}
