//! Rejections for retired signed-integer primitive names.

use crate::{RawAtom, RawExpr, parser};

fn migrated_integer_name(name: &str) -> Option<&'static str> {
    match name {
        "int8" => Some("i8"),
        "int16" => Some("i16"),
        "int32" => Some("i32"),
        "int64" => Some("i64"),
        _ => None,
    }
}

/// Reject retired v0.18 integer names at every current stamped-ingress edge.
///
/// Raw parsing preserves syntax without assigning current primitive semantics.
/// Stamped entry points call this before constructing the validated tree.
pub(crate) fn reject_retired_integer_names(
    expressions: &[RawExpr],
) -> Result<(), parser::ParseError> {
    for expression in expressions {
        reject_retired_integer_name(expression)?;
    }
    Ok(())
}

fn reject_retired_integer_name(expression: &RawExpr) -> Result<(), parser::ParseError> {
    match expression {
        RawExpr::List(items, _) => {
            if matches!(
                items.first(),
                Some(RawExpr::Atom(RawAtom::Symbol(head), _))
                    if matches!(head.as_str(), "t-prim" | "t-var")
            ) && let Some(RawExpr::Atom(RawAtom::Symbol(name), span)) = items.get(2)
                && let Some(canonical) = migrated_integer_name(name)
            {
                return Err(parser::ParseError::Expected {
                    expected: format!("canonical Deep integer dtype `{canonical}`"),
                    found: format!("retired v0.18 Deep integer dtype `{name}`"),
                    offset: span.offset,
                });
            }
            for item in items {
                reject_retired_integer_name(item)?;
            }
        }
        RawExpr::Map(entries, _) => {
            for (_, value) in entries {
                reject_retired_integer_name(value)?;
            }
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            for (_, value) in entries {
                reject_retired_integer_name(value)?;
            }
            reject_retired_integer_name(expr)?;
        }
        RawExpr::ExtensionData(_) | RawExpr::Atom(..) => {}
    }
    Ok(())
}
