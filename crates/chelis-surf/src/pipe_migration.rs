//! Explicit migration from the previous pipe grammar. This is the only reader
//! of retired raw `pipe` forms; it never admits them to normal Deep ingress.
use crate::{ast::*, pipe_sugar};
use chelis_deep::{RawAtom, RawExpr};
use std::collections::BTreeMap;

pub struct PipeMigration {
    pub source: String,
    pub baseline: Vec<chelis_deep::Expr>,
}

fn head(expr: &RawExpr) -> Option<&str> {
    let RawExpr::List(items, _) = expr else {
        return None;
    };
    name(items.first()?)
}
fn name(expr: &RawExpr) -> Option<&str> {
    if let RawExpr::Atom(RawAtom::Symbol(name), _) = expr {
        Some(name)
    } else {
        None
    }
}
fn kids(expr: &RawExpr) -> Option<&[RawExpr]> {
    let RawExpr::List(items, _) = expr else {
        return None;
    };
    items.get(2..)
}
fn variable(expr: &RawExpr) -> Option<&str> {
    (head(expr) == Some("var"))
        .then(|| kids(expr)?.first().and_then(name))
        .flatten()
}
fn mentions(expr: &RawExpr, target: &str) -> bool {
    match expr {
        RawExpr::Atom(RawAtom::Symbol(n), _) => n == target,
        RawExpr::List(items, _) => items.iter().any(|item| mentions(item, target)),
        RawExpr::Map(entries, _) => entries.iter().any(|(_, value)| mentions(value, target)),
        RawExpr::MetaExpr { entries, expr, .. } => {
            mentions(expr, target) || entries.iter().any(|(_, value)| mentions(value, target))
        }
        RawExpr::Atom(..) | RawExpr::ExtensionData(_) => false,
    }
}

fn map_entries(expr: &RawExpr) -> Option<&[(String, RawExpr)]> {
    let RawExpr::List(items, _) = expr else {
        return None;
    };
    let RawExpr::Map(entries, _) = items.get(1)? else {
        return None;
    };
    Some(entries)
}

fn fold_stage(mut stage: RawExpr, carried: RawExpr) -> Result<RawExpr, String> {
    if head(&stage) == Some("fn")
        && let Some([params, body]) = kids(&stage)
        && head(params) == Some("params")
        && let Some([parameter]) = kids(params)
        && let Some(parameter) = name(parameter)
        && let Some(arguments) = kids(body)
    {
        let inserted = match head(body) {
            Some("app") if arguments.first().and_then(variable).is_some() => Some(1),
            Some("cast" | "copy" | "realize") => Some(0),
            _ => None,
        };
        if let Some(inserted) = inserted
            && arguments.get(inserted).and_then(variable) == Some(parameter)
            && !arguments
                .iter()
                .enumerate()
                .any(|(i, arg)| i != inserted && mentions(arg, parameter))
        {
            let RawExpr::List(mut items, span) = body.clone() else {
                unreachable!()
            };
            items[inserted + 2] = carried;
            return Ok(RawExpr::List(items, span));
        }
    }
    let span = stage.span();
    let span_metadata = map_entries(&stage)
        .into_iter()
        .flatten()
        .filter(|(key, _)| key == "span")
        .cloned()
        .collect();
    if let RawExpr::List(items, _) = &mut stage
        && let Some(RawExpr::Map(entries, _)) = items.get_mut(1)
    {
        entries.retain(|(key, _)| key != "surf_pipe_stage");
    }
    Ok(RawExpr::List(
        vec![
            RawExpr::Atom(RawAtom::Symbol("app".into()), span),
            RawExpr::Map(span_metadata, span),
            stage,
            carried,
        ],
        span,
    ))
}

fn fold(expr: &mut RawExpr) -> Result<(), String> {
    match expr {
        RawExpr::List(items, _) => {
            for item in items.iter_mut() {
                fold(item)?;
            }
        }
        RawExpr::Map(entries, _) => {
            entries.retain(|(key, _)| key != "surf_pipe_stage");
            for (_, value) in entries {
                fold(value)?;
            }
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            entries.retain(|(key, _)| key != "surf_pipe_stage");
            for (_, value) in entries {
                fold(value)?;
            }
            fold(expr)?;
        }
        RawExpr::Atom(..) | RawExpr::ExtensionData(_) => {}
    }
    if head(expr) != Some("pipe") {
        return Ok(());
    }
    let RawExpr::List(items, span) = expr else {
        unreachable!()
    };
    let mut items = std::mem::take(items);
    if items.len() < 4 {
        return Err("malformed previous Deep pipe".into());
    }
    let metadata = items.remove(1);
    items.remove(0);
    let mut carried = items.remove(0);
    for stage in items {
        carried = fold_stage(stage, carried)?;
    }
    if let RawExpr::Map(entries, _) = metadata {
        let annotations: Vec<_> = entries
            .into_iter()
            .filter(|(key, _)| matches!(key.as_str(), "type" | "surf_binding_type"))
            .collect();
        if !annotations.is_empty() {
            let conflict = annotations.iter().any(|(key, _)| {
                map_entries(&carried)
                    .is_none_or(|entries| entries.iter().any(|(existing, _)| existing == key))
            });
            if conflict {
                carried = RawExpr::List(
                    vec![
                        RawExpr::Atom(RawAtom::Symbol("block".into()), *span),
                        RawExpr::Map(annotations, *span),
                        carried,
                    ],
                    *span,
                );
            } else if let RawExpr::List(items, _) = &mut carried
                && let Some(RawExpr::Map(entries, _)) = items.get_mut(1)
            {
                entries.extend(annotations);
            }
        }
    }
    *expr = carried;
    Ok(())
}

fn literal_dtypes(expr: &RawExpr, result: &mut BTreeMap<(usize, usize), String>) {
    if head(expr) == Some("lit")
        && let Some(entries) = map_entries(expr)
        && let Some((_, RawExpr::Atom(RawAtom::Str(span), _))) =
            entries.iter().find(|(key, _)| key == "span")
        && let Some((_, ty)) = entries.iter().find(|(key, _)| key == "type")
        && head(ty) == Some("t-prim")
        && let Some(dtype) = kids(ty).and_then(|items| items.first()).and_then(name)
        && let Some((start, end)) = span
            .strip_prefix("surf:")
            .and_then(|span| span.split_once(".."))
        && let (Ok(start), Ok(end)) = (start.parse(), end.parse())
    {
        result.insert((start, end), dtype.into());
    }
    match expr {
        RawExpr::List(items, _) => {
            for item in items {
                literal_dtypes(item, result);
            }
        }
        RawExpr::Map(entries, _) => {
            for (_, value) in entries {
                literal_dtypes(value, result);
            }
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            for (_, value) in entries {
                literal_dtypes(value, result);
            }
            literal_dtypes(expr, result);
        }
        RawExpr::Atom(..) | RawExpr::ExtensionData(_) => {}
    }
}

/// Prepare grouped, explicitly typed source using the previous compiler's Deep.
/// The caller MUST compare the expanded migrated Deep with `baseline` before
/// publishing or replacing any file. A missing old dtype never defaults.
pub fn prepare(source: &str, previous_deep: &str) -> Result<PipeMigration, String> {
    let mut baseline =
        chelis_deep::parser::parse_pipe_migration_raw(previous_deep).map_err(|e| e.to_string())?;
    let mut dtypes = BTreeMap::new();
    for expr in &baseline {
        literal_dtypes(expr, &mut dtypes);
    }
    let mut decls = crate::parser::parse_pipe_migration(source).map_err(|e| e.to_string())?;
    let mut edits = BTreeMap::new();
    let mut failure = None;
    pipe_sugar::visit_program_mut(&mut decls, &mut |expr| {
        let Expr::Pipe(seed, _, _) = expr else {
            return;
        };
        pipe_sugar::visit_expr_mut(seed, &mut |value| {
            if let Expr::Lit(Literal::Int(_) | Literal::Float(_), span) = value {
                let key = (span.offset, span.offset + span.len);
                if let Some(dtype) = dtypes.get(&key) {
                    edits.insert(key.1, dtype.clone());
                } else {
                    failure = Some(format!(
                        "previous compiler supplied no literal dtype at byte {}",
                        span.offset
                    ));
                }
            }
        });
    });
    if let Some(failure) = failure {
        return Err(failure);
    }
    let mut suffixed = source.to_string();
    for (offset, dtype) in edits.into_iter().rev() {
        suffixed.insert_str(offset, &dtype);
    }
    let decls = crate::parser::parse_pipe_migration(&suffixed).map_err(|e| e.to_string())?;
    let source = crate::format::format_pipe_migration_source(&suffixed, &decls)
        .map_err(|e| e.to_string())?;
    if crate::format::format_source(&source).map_err(|e| e.to_string())? != source {
        return Err("pipe migration did not reach a formatter fixed point".into());
    }
    for expr in &mut baseline {
        fold(expr)?;
    }
    let baseline = chelis_deep::stamp_deep_file(baseline).map_err(|e| e.to_string())?;
    Ok(PipeMigration { source, baseline })
}
