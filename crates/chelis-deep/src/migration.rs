//! Explicit migrations for retired Deep source versions.

use crate::{RawAtom, RawExpr, parser, printer, stamp_to_typed};

/// Rewrite v0.18 signed-integer `t-prim` names to canonical v0.19 `i*`
/// spellings, then validate and print the canonical stamped Deep tree.
pub fn migrate_source_v018(source: &str) -> Result<String, String> {
    let mut expressions = parser::parse_raw_str(source).map_err(|error| error.to_string())?;
    expressions.iter_mut().for_each(migrate_expr);
    let stamped =
        stamp_to_typed::stamp_deep_file(expressions).map_err(|error| error.to_string())?;
    Ok(printer::print_canonical(&stamped))
}

fn migrate_expr(expr: &mut RawExpr) {
    match expr {
        RawExpr::List(items, _) => {
            let t_prim = matches!(
                items.first(),
                Some(RawExpr::Atom(RawAtom::Symbol(head), _)) if head == "t-prim"
            );
            if t_prim
                && let Some(RawExpr::Atom(RawAtom::Symbol(name), _)) = items.get_mut(2)
                && let Some(canonical) = migrated_integer_name(name)
            {
                *name = canonical.to_string();
            }
            items.iter_mut().for_each(migrate_expr);
        }
        RawExpr::Map(entries, _) => {
            entries
                .iter_mut()
                .for_each(|(_, value)| migrate_expr(value));
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            entries
                .iter_mut()
                .for_each(|(_, value)| migrate_expr(value));
            migrate_expr(expr);
        }
        RawExpr::ExtensionData(_) | RawExpr::Atom(..) => {}
    }
}

fn migrated_integer_name(name: &str) -> Option<&'static str> {
    match name {
        "int8" => Some("i8"),
        "int16" => Some("i16"),
        "int32" => Some("i32"),
        "int64" => Some("i64"),
        _ => None,
    }
}
