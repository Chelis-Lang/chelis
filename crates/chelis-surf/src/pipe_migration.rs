//! Explicit migration from the previous pipe grammar. This is the only reader
//! of retired raw `pipe` forms; it never admits them to normal Deep ingress.
use crate::{ast::*, pipe_sugar};
use chelis_deep::annotations::{Metadata, MetadataValue};
use chelis_deep::{Atom, DeepTag, ExprCarrier, RawAtom, RawExpr, Span};
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

fn admitted_owner_metadata(owner: RawExpr, binding_value: bool) -> Result<Metadata, String> {
    let span = owner.span();
    // The placement policy checks both the owning tag and its parent slot.
    // Preserve a real bind-value context rather than admitting maps in isolation.
    let envelope = if binding_value {
        RawExpr::List(
            vec![
                RawExpr::Atom(RawAtom::Symbol("bind".into()), span),
                RawExpr::Map(Vec::new(), span),
                RawExpr::Atom(RawAtom::Symbol("migration_value".into()), span),
                owner,
            ],
            span,
        )
    } else {
        owner
    };
    let mut stamped = chelis_deep::stamp_to_typed::stamp_exprs_lenient(vec![envelope])
        .map_err(|error| error.to_string())?;
    let owner = stamped
        .pop()
        .ok_or("previous Deep annotation owner was not admitted")?;
    let owner = if binding_value {
        match &owner {
            chelis_deep::Expr::Node(node, _) => node.children_slice().get(1),
            _ => None,
        }
        .ok_or("previous Deep annotation owner lost its bind-value context")?
    } else {
        &owner
    };
    match owner {
        chelis_deep::Expr::Node(node, _) => Ok(node.meta().clone()),
        chelis_deep::Expr::Atom(..) => Ok(Metadata::default()),
        _ => Err("previous Deep annotation owner did not produce a typed node".into()),
    }
}

fn decode_metadata(
    entries: &[(String, RawExpr)],
    span: Span,
    binding_value: bool,
) -> Result<Metadata, String> {
    // Only the retired spelling is stripped at this historical boundary.
    // Every surviving annotation uses the ordinary typed admission policy,
    // including duplicate rejection and extension preservation.
    let entries = entries
        .iter()
        .filter(|(key, _)| key != "surf_pipe_stage")
        .cloned()
        .collect();
    // Admit retired pipe annotations on the replacement block envelope used
    // when both result and owner carry ascriptions, with the original context.
    let envelope = RawExpr::List(
        vec![
            RawExpr::Atom(RawAtom::Symbol("block".into()), span),
            RawExpr::Map(entries, span),
            RawExpr::List(
                vec![
                    RawExpr::Atom(RawAtom::Symbol("var".into()), span),
                    RawExpr::Map(Vec::new(), span),
                    RawExpr::Atom(RawAtom::Symbol("migration_value".into()), span),
                ],
                span,
            ),
        ],
        span,
    );
    admitted_owner_metadata(envelope, binding_value)
}

fn node_metadata(expr: &RawExpr, binding_value: bool) -> Result<Metadata, String> {
    if !matches!(expr, RawExpr::Atom(..)) && map_entries(expr).is_none() {
        return Err(format!(
            "previous Deep `{}` requires an inline metadata map",
            head(expr).unwrap_or("annotation owner")
        ));
    }
    // Placement-sensitive annotations (literal spelling, quantified bounds,
    // etc.) must be admitted on their real owner, not a standalone map.
    let mut raw = expr.clone();
    if let RawExpr::List(items, _) = &mut raw
        && let Some(RawExpr::Map(entries, _)) = items.get_mut(1)
    {
        entries.retain(|(key, _)| key != "surf_pipe_stage");
    }
    admitted_owner_metadata(raw, binding_value)
}

fn raw_metadata(
    values: impl IntoIterator<Item = MetadataValue>,
    span: Span,
) -> Result<RawExpr, String> {
    let metadata = Metadata::try_from_values(values).map_err(|error| error.to_string())?;
    Ok(chelis_deep::Expr::Map(metadata, span).to_raw())
}

fn fold_stage(mut stage: RawExpr, carried: RawExpr) -> Result<RawExpr, String> {
    // Compaction erases the historical wrapper, so admit its annotations on
    // the real owner before any branch can discard malformed evidence.
    let metadata = node_metadata(&stage, false)?;
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
    let span_metadata = raw_metadata(metadata.span_id().cloned().map(MetadataValue::Span), span)?;
    if let RawExpr::List(items, _) = &mut stage
        && let Some(RawExpr::Map(entries, _)) = items.get_mut(1)
    {
        entries.retain(|(key, _)| key != "surf_pipe_stage");
    }
    Ok(RawExpr::List(
        vec![
            RawExpr::Atom(RawAtom::Symbol("app".into()), span),
            span_metadata,
            stage,
            carried,
        ],
        span,
    ))
}

fn is_bind_value(parent: Option<&str>, index: usize) -> bool {
    parent == Some("bind") && index >= 2 && (index - 2) % 2 == 1
}

fn contains_program_nodes(key: &str) -> bool {
    !matches!(
        chelis_deep::metadata::role(key),
        chelis_deep::metadata::MetadataRole::Preserved
            | chelis_deep::metadata::MetadataRole::BinderMap
    )
}

fn fold(expr: &mut RawExpr, binding_value: bool) -> Result<(), String> {
    let parent = head(expr).map(str::to_owned);
    if parent.as_deref() == Some("pipe") && map_entries(expr).is_none() {
        return Err("previous Deep `pipe` requires an inline metadata map".into());
    }
    match expr {
        RawExpr::List(items, _) => {
            for (index, item) in items.iter_mut().enumerate() {
                fold(item, is_bind_value(parent.as_deref(), index))?;
            }
        }
        RawExpr::Map(entries, _) => {
            entries.retain(|(key, _)| key != "surf_pipe_stage");
            for (key, value) in entries {
                if contains_program_nodes(key) {
                    fold(value, false)?;
                }
            }
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            entries.retain(|(key, _)| key != "surf_pipe_stage");
            for (key, value) in entries {
                if contains_program_nodes(key) {
                    fold(value, false)?;
                }
            }
            fold(expr, false)?;
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
        let metadata = decode_metadata(&entries, *span, binding_value)?;
        let annotations: Vec<_> = metadata
            .values()
            .filter(|value| {
                matches!(
                    value,
                    MetadataValue::Type(_) | MetadataValue::SurfBindingType(_)
                )
            })
            .cloned()
            .collect();
        if !annotations.is_empty() {
            let carried_metadata = node_metadata(&carried, binding_value)?;
            let conflict = map_entries(&carried).is_none()
                || annotations.iter().any(|value| {
                    carried_metadata
                        .values()
                        .any(|existing| existing.key() == value.key())
                });
            let annotations = raw_metadata(annotations, *span)?;
            if conflict {
                carried = RawExpr::List(
                    vec![
                        RawExpr::Atom(RawAtom::Symbol("block".into()), *span),
                        annotations,
                        carried,
                    ],
                    *span,
                );
            } else if let RawExpr::List(items, _) = &mut carried
                && let Some(RawExpr::Map(entries, _)) = items.get_mut(1)
            {
                let RawExpr::Map(annotations, _) = annotations else {
                    return Err("typed annotations did not export a metadata map".into());
                };
                entries.extend(annotations);
            }
        }
    } else {
        return Err("previous Deep `pipe` requires an inline metadata map".into());
    }
    *expr = carried;
    Ok(())
}

fn literal_dtypes(
    expr: &RawExpr,
    result: &mut BTreeMap<(usize, usize), String>,
    binding_value: bool,
) -> Result<(), String> {
    let metadata = if head(expr) == Some("lit") {
        node_metadata(expr, binding_value)?
    } else {
        Metadata::default()
    };
    if head(expr) == Some("lit")
        && let Some(span) = metadata.span_id()
        && let Some(ty) = metadata.ty()
        && let ExprCarrier::DecodedNode(DeepTag::TPrim, _, children) = ty.expression().carrier()
        && let Some(chelis_deep::Expr::Atom(Atom::Name(dtype), _)) = children.first()
        && let Some((start, end)) = span
            .value()
            .strip_prefix("surf:")
            .and_then(|span| span.split_once(".."))
        && let (Ok(start), Ok(end)) = (start.parse(), end.parse())
    {
        result.insert((start, end), dtype.into());
    }
    match expr {
        RawExpr::List(items, _) => {
            for (index, item) in items.iter().enumerate() {
                literal_dtypes(item, result, is_bind_value(head(expr), index))?;
            }
        }
        RawExpr::Map(entries, _) => {
            for (key, value) in entries {
                if contains_program_nodes(key) {
                    literal_dtypes(value, result, false)?;
                }
            }
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            for (key, value) in entries {
                if contains_program_nodes(key) {
                    literal_dtypes(value, result, false)?;
                }
            }
            literal_dtypes(expr, result, false)?;
        }
        RawExpr::Atom(..) | RawExpr::ExtensionData(_) => {}
    }
    Ok(())
}

/// Prepare grouped, explicitly typed source using the previous compiler's Deep.
/// The caller MUST compare the expanded migrated Deep with `baseline` before
/// publishing or replacing any file. A missing old dtype never defaults.
pub fn prepare(source: &str, previous_deep: &str) -> Result<PipeMigration, String> {
    let mut baseline =
        chelis_deep::parser::parse_pipe_migration_raw(previous_deep).map_err(|e| e.to_string())?;
    let mut dtypes = BTreeMap::new();
    for expr in &baseline {
        literal_dtypes(expr, &mut dtypes, false)?;
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
        fold(expr, false)?;
    }
    let baseline = chelis_deep::stamp_deep_file(baseline).map_err(|e| e.to_string())?;
    Ok(PipeMigration { source, baseline })
}
