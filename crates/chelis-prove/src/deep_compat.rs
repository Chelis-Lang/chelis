//! Transitional compatibility for typed Deep parser output.

use chelis_deep::{DeepTag, Expr, Span};

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

fn normalize_node_to_list(expr: &chelis_deep::Expr) -> chelis_deep::Expr {
    enum Action<'a> {
        Visit(&'a chelis_deep::Expr),
        FinishList {
            element_count: usize,
            span: Span,
        },
        FinishMap {
            map: &'a chelis_deep::Metadata,
            span: Span,
        },
        FinishMetaExpr {
            meta: &'a chelis_deep::MetaExpr,
            span: Span,
        },
        FinishNode {
            tag: DeepTag,
            meta: &'a chelis_deep::Metadata,
            child_count: usize,
            span: Span,
        },
        FinishUnknownForm(&'a chelis_deep::UnknownFormData),
    }

    fn split_tail(values: &mut Vec<chelis_deep::Expr>, count: usize) -> Vec<chelis_deep::Expr> {
        let start = values
            .len()
            .checked_sub(count)
            .expect("normalization action/value stacks remain balanced");
        values.split_off(start)
    }

    fn metadata_leaves(meta: &chelis_deep::Metadata) -> Vec<&chelis_deep::Expr> {
        let mut leaves = Vec::new();
        meta.visit_expressions(&mut |value, _| leaves.push(value));
        leaves
    }
    fn rebuild_metadata(
        meta: &chelis_deep::Metadata,
        normalized: Vec<chelis_deep::Expr>,
    ) -> chelis_deep::Metadata {
        let mut normalized = normalized.into_iter();
        let result = meta
            .try_map_leaves::<chelis_deep::metadata::MetadataError>(&mut |_, _| {
                Ok(normalized
                    .next()
                    .expect("one worklist value per metadata leaf"))
            })
            .expect("carrier normalization preserves payload admission");
        assert!(
            normalized.next().is_none(),
            "all metadata worklist values consumed"
        );
        result
    }
    let mut actions = vec![Action::Visit(expr)];
    let mut values = Vec::new();

    while let Some(action) = actions.pop() {
        match action {
            Action::Visit(expr) => match expr {
                chelis_deep::Expr::Atom(atom, span) => {
                    values.push(chelis_deep::Expr::Atom(atom.clone(), *span));
                }
                chelis_deep::Expr::List(list, span) => {
                    actions.push(Action::FinishList {
                        element_count: list.elements.len(),
                        span: *span,
                    });
                    actions.extend(list.elements.iter().rev().map(Action::Visit));
                }
                chelis_deep::Expr::Map(map, span) => {
                    actions.push(Action::FinishMap { map, span: *span });
                    actions.extend(metadata_leaves(map).into_iter().rev().map(Action::Visit));
                }
                chelis_deep::Expr::MetaExpr(meta, span) => {
                    actions.push(Action::FinishMetaExpr { meta, span: *span });
                    actions.push(Action::Visit(&meta.expr));
                    actions.extend(
                        metadata_leaves(&meta.metadata)
                            .into_iter()
                            .rev()
                            .map(Action::Visit),
                    );
                }
                chelis_deep::Expr::Node(node, span) => {
                    actions.push(Action::FinishNode {
                        tag: node.tag(),
                        meta: node.meta(),
                        child_count: node.child_count(),
                        span: *span,
                    });
                    actions.extend(node.children_slice().iter().rev().map(Action::Visit));
                    actions.extend(
                        metadata_leaves(node.meta())
                            .into_iter()
                            .rev()
                            .map(Action::Visit),
                    );
                }
                chelis_deep::Expr::BareList(elements, span) => {
                    actions.push(Action::FinishList {
                        element_count: elements.len(),
                        span: *span,
                    });
                    actions.extend(elements.iter().rev().map(Action::Visit));
                }
                chelis_deep::Expr::UnknownForm(data) => {
                    actions.push(Action::FinishUnknownForm(data));
                    actions.extend(data.children.iter().rev().map(Action::Visit));
                    actions.extend(
                        metadata_leaves(&data.meta)
                            .into_iter()
                            .rev()
                            .map(Action::Visit),
                    );
                }
            },
            Action::FinishList {
                element_count,
                span,
            } => {
                let elements = split_tail(&mut values, element_count);
                values.push(chelis_deep::Expr::List(
                    chelis_deep::List { elements },
                    span,
                ));
            }
            Action::FinishMap { map, span } => {
                let normalized = split_tail(&mut values, metadata_leaves(map).len());
                values.push(chelis_deep::Expr::Map(
                    rebuild_metadata(map, normalized),
                    span,
                ));
            }
            Action::FinishMetaExpr { meta, span } => {
                let mut normalized =
                    split_tail(&mut values, metadata_leaves(&meta.metadata).len() + 1);
                let normalized_expr = normalized.pop().expect("MetaExpr visits its expression");
                values.push(chelis_deep::Expr::MetaExpr(
                    chelis_deep::MetaExpr {
                        metadata: rebuild_metadata(&meta.metadata, normalized),
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
                let metadata_count = metadata_leaves(meta).len();
                let mut normalized = split_tail(&mut values, metadata_count + child_count);
                let children = normalized.split_off(metadata_count);
                let mut elements = vec![
                    chelis_deep::Expr::Atom(chelis_deep::Atom::Tag(tag), span),
                    chelis_deep::Expr::Map(rebuild_metadata(meta, normalized), span),
                ];
                elements.extend(children);
                values.push(chelis_deep::Expr::List(
                    chelis_deep::List { elements },
                    span,
                ));
            }
            Action::FinishUnknownForm(data) => {
                let metadata_count = metadata_leaves(&data.meta).len();
                let mut normalized = split_tail(&mut values, metadata_count + data.children.len());
                let children = normalized.split_off(metadata_count);
                values.push(chelis_deep::Expr::UnknownForm(Box::new(
                    chelis_deep::UnknownFormData {
                        head: data.head.clone(),
                        meta: rebuild_metadata(&data.meta, normalized),
                        children,
                        span: data.span,
                    },
                )));
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
