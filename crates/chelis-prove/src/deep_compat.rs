//! Transitional compatibility for typed Deep parser output.

use chelis_deep::{Atom, DeepTag, Expr, List, MetaMap, Span};

/// Parse a Deep file through the typed boundary, then supply legacy list dispatch.
pub(crate) fn parse_file_to_lists(
    source: &str,
) -> Result<Vec<Expr>, chelis_deep::StampOrParseError> {
    chelis_deep::parse_and_stamp_file(source).map(|exprs| normalize_nodes_to_lists(&exprs))
}

/// Convert the complete typed tree without native recursion.
pub(crate) fn normalize_nodes_to_lists(exprs: &[Expr]) -> Vec<Expr> {
    exprs.iter().map(normalize_node_to_list).collect()
}

fn normalize_node_to_list(expr: &Expr) -> Expr {
    enum Action<'a> {
        Visit(&'a Expr),
        FinishList {
            element_count: usize,
            span: Span,
        },
        FinishMap {
            map: &'a MetaMap,
            span: Span,
        },
        FinishMetaExpr {
            meta: &'a chelis_deep::MetaExpr,
            span: Span,
        },
        FinishNode {
            tag: DeepTag,
            meta: &'a MetaMap,
            child_count: usize,
            span: Span,
        },
        FinishUnknownForm(&'a chelis_deep::UnknownFormData),
    }

    fn split_tail(values: &mut Vec<Expr>, count: usize) -> Vec<Expr> {
        let start = values
            .len()
            .checked_sub(count)
            .expect("normalization action and value stacks remain balanced");
        values.split_off(start)
    }

    let mut actions = vec![Action::Visit(expr)];
    let mut values = Vec::new();

    while let Some(action) = actions.pop() {
        match action {
            Action::Visit(expr) => match expr {
                Expr::Atom(atom, span) => {
                    values.push(Expr::Atom(atom.clone(), *span));
                }
                Expr::List(list, span) => {
                    actions.push(Action::FinishList {
                        element_count: list.elements.len(),
                        span: *span,
                    });
                    actions.extend(list.elements.iter().rev().map(Action::Visit));
                }
                Expr::Map(map, span) => {
                    actions.push(Action::FinishMap { map, span: *span });
                    actions.extend(
                        map.entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
                Expr::MetaExpr(meta, span) => {
                    actions.push(Action::FinishMetaExpr { meta, span: *span });
                    actions.push(Action::Visit(&meta.expr));
                    actions.extend(
                        meta.entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
                Expr::Node(node, span) => {
                    actions.push(Action::FinishNode {
                        tag: node.tag(),
                        meta: node.meta(),
                        child_count: node.child_count(),
                        span: *span,
                    });
                    actions.extend(node.children_slice().iter().rev().map(Action::Visit));
                    actions.extend(
                        node.meta()
                            .entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
                Expr::BareList(elements, span) => {
                    actions.push(Action::FinishList {
                        element_count: elements.len(),
                        span: *span,
                    });
                    actions.extend(elements.iter().rev().map(Action::Visit));
                }
                Expr::UnknownForm(data) => {
                    actions.push(Action::FinishUnknownForm(data));
                    actions.extend(data.children.iter().rev().map(Action::Visit));
                    actions.extend(
                        data.meta
                            .entries
                            .iter()
                            .rev()
                            .map(|(_, value)| Action::Visit(value)),
                    );
                }
            },
            Action::FinishList {
                element_count,
                span,
            } => {
                let elements = split_tail(&mut values, element_count);
                values.push(Expr::List(List { elements }, span));
            }
            Action::FinishMap { map, span } => {
                let normalized_values = split_tail(&mut values, map.entries.len());
                let entries = map
                    .entries
                    .iter()
                    .zip(normalized_values)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                values.push(Expr::Map(MetaMap { entries }, span));
            }
            Action::FinishMetaExpr { meta, span } => {
                let mut normalized = split_tail(&mut values, meta.entries.len() + 1);
                let normalized_expr = normalized
                    .pop()
                    .expect("MetaExpr normalization visits its expression");
                let entries = meta
                    .entries
                    .iter()
                    .zip(normalized)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                values.push(Expr::MetaExpr(
                    chelis_deep::MetaExpr {
                        entries,
                        expr: Box::new(normalized_expr),
                    },
                    span,
                ));
            }
            Action::FinishNode {
                tag,
                meta,
                child_count,
                span,
            } => {
                let mut normalized = split_tail(&mut values, meta.entries.len() + child_count);
                let children = normalized.split_off(meta.entries.len());
                let entries = meta
                    .entries
                    .iter()
                    .zip(normalized)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                let mut elements = Vec::with_capacity(child_count + 2);
                elements.push(Expr::Atom(Atom::Tag(tag), span));
                elements.push(Expr::Map(MetaMap { entries }, span));
                elements.extend(children);
                values.push(Expr::List(List { elements }, span));
            }
            Action::FinishUnknownForm(data) => {
                let mut normalized =
                    split_tail(&mut values, data.meta.entries.len() + data.children.len());
                let children = normalized.split_off(data.meta.entries.len());
                let entries = data
                    .meta
                    .entries
                    .iter()
                    .zip(normalized)
                    .map(|((key, _), value)| (key.clone(), value))
                    .collect();
                values.push(Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                    head: data.head.clone(),
                    meta: MetaMap { entries },
                    children,
                    span: data.span,
                })));
            }
        }
    }

    assert_eq!(
        values.len(),
        1,
        "one normalization root produces exactly one expression"
    );
    values.pop().expect("normalization produced its root")
}
