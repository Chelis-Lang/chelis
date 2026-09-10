//! Raw source DTO projection. Numeric admission is separate from source
//! stamping/type checking; transporting syntax never establishes executability.

use crate::schema::numbers::{SourceFloat, SourceInteger};
use crate::schema::{
    Span, WireDeepAtom, WireDeepExpr, WireDeepExprKind, WireLiteral, WireMetaEntry,
};
use chelis_deep::Expr as DeepExpr;
use chelis_surf::ast::Literal;

pub(crate) type SourceWireResult<T> = Result<T, String>;

fn span(span: chelis_deep::Span) -> Span {
    span.into()
}

pub(crate) fn wire_literal(lit: &Literal) -> SourceWireResult<WireLiteral> {
    Ok(match lit {
        Literal::Int(value) => WireLiteral::Int {
            value: SourceInteger::new(*value),
        },
        Literal::Float(value) => WireLiteral::Float {
            value: SourceFloat::new(*value)?,
        },
        // Typed-suffix literals (spec §5.5): preserve the suffix across
        // the wire boundary so the receiving side sees the same type.
        Literal::TypedInt(value, suffix) => WireLiteral::TypedInt {
            value: SourceInteger::new(*value),
            suffix: suffix.as_str().to_string(),
        },
        Literal::TypedFloat(value, suffix) => WireLiteral::TypedFloat {
            value: SourceFloat::new(*value)?,
            suffix: suffix.as_str().to_string(),
        },
        Literal::Str(value) => WireLiteral::Str {
            value: value.clone(),
        },
        Literal::Bool(value) => WireLiteral::Bool { value: *value },
    })
}

pub(crate) fn wire_deep_expr(expr: &DeepExpr) -> SourceWireResult<WireDeepExpr> {
    fn encode(expr: chelis_deep::raw::RawExpr) -> SourceWireResult<WireDeepExpr> {
        use chelis_deep::raw::{RawAtom, RawExpr};
        let source_span = Some(span(expr.span()));
        let kind = match expr {
            RawExpr::ExtensionData(data) => WireDeepExprKind::ExtensionData {
                syntax: data.syntax().into(),
            },
            RawExpr::Atom(atom, _) => WireDeepExprKind::Atom {
                atom: match atom {
                    RawAtom::Symbol(value) => WireDeepAtom::Symbol { value },
                    RawAtom::Int(value) => WireDeepAtom::Int {
                        value: SourceInteger::new(value),
                    },
                    RawAtom::Float(value) => WireDeepAtom::Float {
                        value: SourceFloat::new(value)?,
                    },
                    RawAtom::Str(value) => WireDeepAtom::Str { value },
                    RawAtom::Bool(value) => WireDeepAtom::Bool { value },
                },
            },
            RawExpr::List(elements, _) => WireDeepExprKind::List {
                elements: elements
                    .into_iter()
                    .map(encode)
                    .collect::<SourceWireResult<_>>()?,
            },
            RawExpr::Map(entries, _) => WireDeepExprKind::Map {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| {
                        Ok(WireMetaEntry {
                            key,
                            value: encode(value)?,
                        })
                    })
                    .collect::<SourceWireResult<_>>()?,
            },
            RawExpr::MetaExpr { entries, expr, .. } => WireDeepExprKind::MetaExpr {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| {
                        Ok(WireMetaEntry {
                            key,
                            value: encode(value)?,
                        })
                    })
                    .collect::<SourceWireResult<_>>()?,
                expr: Box::new(encode(*expr)?),
            },
        };
        Ok(WireDeepExpr {
            kind,
            span: source_span,
        })
    }
    encode(expr.to_raw())
}
