//! Declaration-owned evidence for spec/04 §4.5.3 body composition.
//! Lexical targets and SCCs come exclusively from TopLevelReferenceGraph.

use super::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Default)]
pub(crate) struct RankBodyContext {
    declarations: BTreeMap<String, [u8; 32]>,
}

fn rank_body_error(def_name: &str, message: String) -> CheckError {
    CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!("rank-polymorphic def `{def_name}` {message} (spec/04-type-system.md §4.5.3)."),
        vec![],
    )
}

fn check_body(
    def_name: &str,
    expr: &deep::Expr,
    graph: &TopLevelReferenceGraph,
    available: &BTreeSet<usize>,
    component: &BTreeSet<usize>,
    context: &RankBodyContext,
    errors: &mut Vec<CheckError>,
) {
    stack_guard!("check_rank_body_discipline", expr);
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        if let deep::Expr::MetaExpr(meta, _) = expr {
            check_body(
                def_name, &meta.expr, graph, available, component, context, errors,
            );
        }
        return;
    };
    match tag {
        DeepTag::Grad | DeepTag::Vmap => errors.push(rank_body_error(
            def_name,
            format!("may not use `{}` in its body: no body evidence transports its function across the opaque rank", tag.as_str()),
        )),
        DeepTag::App => match graph.applications.get(&expr_key(expr)) {
            Some(ResolvedApplication::Definition(vertex))
                if available.contains(vertex) || component.contains(vertex) => {}
            Some(ResolvedApplication::Definition(vertex)) => errors.push(rank_body_error(
                def_name,
                format!("may not call `{}`: checked name-preserving body evidence is unavailable", graph.definition(*vertex).name),
            )),
            Some(ResolvedApplication::Unshadowed(name)) if context.declarations.contains_key(name) => {}
            Some(ResolvedApplication::Unshadowed(name)) => match builtins::shape_class(name) {
                Some(builtins::ShapeClass::Identity | builtins::ShapeClass::NameTracked
                    | builtins::ShapeClass::OrderedPrefix | builtins::ShapeClass::Inert) => {}
                Some(builtins::ShapeClass::Untracked) => errors.push(rank_body_error(
                    def_name,
                    format!("may not call builtin `{name}`: it is not name-trackable; only shape-identity, named-axis, ordered-prefix and inert builtins are admitted"),
                )),
                None => errors.push(rank_body_error(
                    def_name,
                    format!("may not call `{name}`: checked name-preserving body evidence is unavailable"),
                )),
            },
            Some(ResolvedApplication::Local(name)) => errors.push(rank_body_error(
                def_name,
                format!("may not apply local callable `{name}`: its function type supplies no body evidence"),
            )),
            Some(ResolvedApplication::Computed) | None => errors.push(rank_body_error(
                def_name,
                "may not apply a computed callee: resolved declaration-owned body evidence is unavailable".to_string(),
            )),
        },
        _ => {}
    }
    for child in kids {
        check_body(
            def_name, child, graph, available, component, context, errors,
        );
    }
}

/// Check complete bodies against the existing inference/row-unification
/// result, accepting SCCs as one unit. This returns staged evidence; its
/// caller's empty-errors/totality boundary owns publication.
pub(super) fn validate_rank_bodies(
    exprs: &[deep::Expr],
    product: &InferenceProduct,
    graph: &TopLevelReferenceGraph,
    base: &RankBodyContext,
    errors: &mut DiagnosticSink<'_>,
) -> RankBodyContext {
    let items = top_level_decl_items(exprs);
    let mut context = base.clone();
    for expr in &items {
        if let Some(name) = top_level_decl_name(expr) {
            context.declarations.remove(name);
        }
    }
    let components = graph.inference_components();
    if !components.complete {
        return context;
    }
    let cancel = crate::cancel::current_cancel_token();
    let mut accepted = BTreeSet::new();
    for component in components.components {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            break;
        }
        let mut candidate = BTreeSet::new();
        for index in &component.members {
            if let Some(vertex) = graph.definition_vertex_by_item[*index]
                && graph.definition(vertex).kind == TopLevelDefinitionKind::Function
            {
                candidate.insert(vertex);
            }
        }
        let mut checked = Vec::new();
        let mut component_valid = true;
        for index in &component.members {
            let Some(vertex) = graph.definition_vertex_by_item[*index] else {
                continue;
            };
            let definition = graph.definition(vertex);
            let Some((DeepTag::Def, _, def_kids)) = stamped_parts(items[*index]) else {
                continue;
            };
            let Some((DeepTag::Fn, _, fn_kids)) = def_kids.get(1).and_then(stamped_parts) else {
                component_valid = false;
                continue;
            };
            let Some(body) = fn_kids.get(1) else {
                component_valid = false;
                continue;
            };
            let mut failures = Vec::new();
            check_body(
                &definition.name,
                body,
                graph,
                &accepted,
                &candidate,
                &context,
                &mut failures,
            );
            component_valid &= failures.is_empty();
            checked.push((vertex, *index, failures));
        }
        for (vertex, index, failures) in checked {
            let name = &graph.definition(vertex).name;
            let (_, _, def_kids) = stamped_parts(items[index]).expect("checked definition");
            let required = product
                .owner_type(&def_kids[1], "rank-body function", errors)
                .is_some_and(|ty| type_contains_rank(&ty))
                && !is_exact_op35_wrapper(name);
            if component_valid {
                accepted.insert(vertex);
                let canonical =
                    chelis_deep::printer::print_canonical_flat(std::slice::from_ref(items[index]));
                context
                    .declarations
                    .insert(name.clone(), Sha256::digest(canonical.as_bytes()).into());
            } else if required {
                if failures.is_empty() {
                    errors.push(rank_body_error(name, "has no body evidence: a member of its recursive component is not name-trackable".to_string()));
                } else {
                    for error in failures {
                        errors.push(error);
                    }
                }
            }
        }
    }
    context
}
