//! Checked Deep type annotation.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn annotate_ir_program(
    exprs: &[deep::Expr],
    product: &InferenceProduct,
    errors: &mut DiagnosticSink<'_>,
) -> Vec<deep::Expr> {
    let items = top_level_decl_items_with_modules(exprs);
    let declared_signatures = collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr));
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    exprs
        .iter()
        .map(|expr| annotate_expr_with_scope(expr, product, annotation_context, errors))
        .collect()
}

pub(super) fn annotate_expr_with_scope(
    expr: &deep::Expr,
    product: &InferenceProduct,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> deep::Expr {
    // Bail value is the identity (unannotated) expr: annotation is a
    // best-effort pass and the entry boundary fails the check anyway.
    stack_guard!("annotate_expr_with_scope", expr, expr.clone());
    match expr {
        deep::Expr::Atom(_, _) => expr.clone(),
        // Metadata values are source/compiler context, not runtime children
        // owned by expression inference. Preserve them verbatim.
        deep::Expr::Map(map, span) => deep::Expr::Map(map.clone(), *span),
        deep::Expr::MetaExpr(meta, span) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                expr: Box::new(annotate_expr_with_scope(
                    &meta.expr,
                    product,
                    annotation_context,
                    errors,
                )),
                entries: meta.entries.clone(),
            },
            *span,
        ),
        deep::Expr::List(list, span) => {
            if !matches!(list.elements.get(1), Some(deep::Expr::Map(_, _))) {
                return deep::Expr::List(
                    deep::List {
                        elements: list
                            .elements
                            .iter()
                            .map(|element| {
                                annotate_expr_with_scope(
                                    element,
                                    product,
                                    annotation_context,
                                    errors,
                                )
                            })
                            .collect(),
                    },
                    *span,
                );
            }

            let tag = get_tag(list);
            let def_name = (tag == Some(DeepTag::Def))
                .then(|| children(list).first().and_then(symbol_name))
                .flatten();
            let declared_sig =
                def_name.and_then(|name| annotation_context.declared_signature(name));
            let (annotated_children, fn_ty_override) = match tag {
                Some(DeepTag::Fn) => {
                    let (kids, fn_ty) =
                        annotate_fn_children(list, expr, product, None, annotation_context, errors);
                    (kids, Some(fn_ty))
                }
                // `(def name (fn ...))`: when the def has a separate
                // `defsig`, its declared parameter type expressions
                // (captured verbatim in `annotation_context`)
                // preserve `&` borrow wrappers that neither the bare
                // `fn` literal nor the inferred function type carry.
                // Stamp those onto the def's `(params ...)` node so a
                // standalone-lowered borrowed param keeps its borrow
                // and IR lowering sees a non-rank-0 type. Defs without
                // a `defsig` are absent from the map and fall through
                // to the plain recursion (bare params, owned by
                // `infer_signature_metadata`).
                Some(DeepTag::Def) => {
                    let kids = children(list);
                    let declared_param_types =
                        declared_sig.map(|metadata| metadata.param_types.as_slice());
                    let annotated: Vec<deep::Expr> = kids
                        .iter()
                        .enumerate()
                        .map(|(index, child)| {
                            let role = child_stamp_role(DeepTag::Def, index, kids.len());
                            if let (
                                ChildStampRole::RuntimeExpr,
                                Some(declared),
                                deep::Expr::List(fn_list, fn_span),
                            ) = (role, declared_param_types.as_ref(), child)
                                && get_tag(fn_list) == Some(DeepTag::Fn)
                            {
                                let (fn_kids, fn_ty) = annotate_fn_children(
                                    fn_list,
                                    child,
                                    product,
                                    Some(declared),
                                    annotation_context,
                                    errors,
                                );
                                let mut elements = vec![
                                    fn_list.elements[0].clone(),
                                    annotated_meta_map_with_override(
                                        fn_list,
                                        child,
                                        product,
                                        Some(fn_ty),
                                        errors,
                                    ),
                                ];
                                elements.extend(fn_kids);
                                deep::Expr::List(deep::List { elements }, *fn_span)
                            } else {
                                annotate_child_for_role(
                                    DeepTag::Def,
                                    index,
                                    kids.len(),
                                    child,
                                    product,
                                    annotation_context,
                                    errors,
                                )
                            }
                        })
                        .collect();
                    (annotated, None)
                }
                Some(tag) => (
                    annotate_children_by_role(tag, list, product, annotation_context, errors),
                    None,
                ),
                None => (children(list).to_vec(), None),
            };

            let mut elements = vec![
                list.elements[0].clone(),
                annotated_meta_map_with_override(list, expr, product, fn_ty_override, errors),
            ];
            elements.extend(annotated_children);
            deep::Expr::List(deep::List { elements }, *span)
        }
        deep::Expr::Node(node, span) => {
            let tag = node.tag();
            let children = node.children_slice();
            let fn_ty_override = (tag == DeepTag::Fn)
                .then(|| product.owner_type(expr, "function node", errors))
                .flatten();
            let annotated_children = children
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    annotate_child_for_role(
                        tag,
                        index,
                        children.len(),
                        child,
                        product,
                        annotation_context,
                        errors,
                    )
                })
                .collect();
            let meta = annotated_node_meta_with_override(
                tag,
                node.meta(),
                expr,
                fn_ty_override,
                product,
                errors,
            );
            deep::Expr::node(tag, meta, annotated_children, *span)
        }
        deep::Expr::BareList(elems, span) => {
            let annotated: Vec<deep::Expr> = elems
                .iter()
                .map(|child| annotate_expr_with_scope(child, product, annotation_context, errors))
                .collect();
            deep::Expr::BareList(annotated, *span)
        }
        deep::Expr::UnknownForm(data) => {
            // Preserve unknown forms unchanged — downstream diagnostics handle them.
            deep::Expr::UnknownForm(data.clone())
        }
    }
}

pub(super) fn annotate_children_by_role(
    tag: DeepTag,
    list: &deep::List,
    product: &InferenceProduct,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> Vec<deep::Expr> {
    let kids = children(list);
    kids.iter()
        .enumerate()
        .map(|(index, child)| {
            annotate_child_for_role(
                tag,
                index,
                kids.len(),
                child,
                product,
                annotation_context,
                errors,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn annotate_child_for_role(
    parent_tag: DeepTag,
    index: usize,
    arity: usize,
    child: &deep::Expr,
    product: &InferenceProduct,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> deep::Expr {
    // Decode-once: `child_stamp_role` is total over `DeepTag`, so the old
    // "no child ownership classification" version-skew arm is
    // unrepresentable and was removed.
    match child_stamp_role(parent_tag, index, arity) {
        ChildStampRole::RuntimeExpr | ChildStampRole::ExplicitInferenceBypass => {
            annotate_expr_with_scope(child, product, annotation_context, errors)
        }
        ChildStampRole::Syntax
        | ChildStampRole::Selector
        | ChildStampRole::EffectHandler
        | ChildStampRole::Binder
        | ChildStampRole::Type => child.clone(),
    }
}

/// True for the `(t-var _)` placeholder that `desugar_fun_def` emits
/// in a synthesized sig for a parameter that had no declared type.
pub(super) fn is_wildcard_tvar_expr(expr: &deep::Expr) -> bool {
    let deep::Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some(DeepTag::TVar)
        && children(list).first().and_then(symbol_name) == Some("_")
}

/// Rebuild a `(params ...)` node so every previously-bare parameter
/// symbol carries a `{type: ...}` metadata entry, using the declared
/// signature's parameter type *expressions* (`declared_param_type_exprs`).
///
/// The desugarer only attaches type metadata to parameters with an
/// inline annotation (`def f(x: T)`); parameters whose types come from
/// a separate `sig` declaration desugar to bare symbols. IR lowering's
/// `lower_fn` reads param types straight off this node, so without this
/// step a standalone-lowered def's params fall back to a rank-0
/// `default_type()` and any shape-sensitive op on them panics in
/// `tier2`.
///
/// The declared type *expressions* are copied verbatim, so `(t-ref ...)`
/// borrow wrappers survive intact (a `Type` round-trip via the inferred
/// function type drops them, which would make the linearity checker
/// treat a borrowed param as owned).
///
/// Parameters that already carry a type annotation are left untouched.
pub(super) fn annotate_params_node(
    params_expr: &deep::Expr,
    declared_param_type_exprs: &[deep::Expr],
) -> deep::Expr {
    let deep::Expr::List(list, span) = params_expr else {
        return params_expr.clone();
    };
    if get_tag(list) != Some(DeepTag::Params) {
        return params_expr.clone();
    }
    let mut elements = vec![list.elements[0].clone(), list.elements[1].clone()];
    for (index, param) in children(list).iter().enumerate() {
        match param {
            deep::Expr::Atom(deep::Atom::Name(name), atom_span) => {
                // A synthesized sig from `desugar_fun_def` uses
                // `(t-var _)` as the placeholder for a parameter with
                // no declared type. That is not a real declared type:
                // stamping it would pre-empt `infer_signature_metadata`'s
                // read-only/borrow inference, so the param is left bare.
                let declared = declared_param_type_exprs
                    .get(index)
                    .filter(|expr| !is_wildcard_tvar_expr(expr));
                match declared {
                    Some(type_expr) => {
                        elements.push(deep::Expr::List(
                            deep::List {
                                elements: vec![
                                    deep::Expr::Atom(deep::Atom::Name(name.clone()), *atom_span),
                                    deep::Expr::Map(
                                        deep::MetaMap {
                                            entries: vec![("type".to_string(), type_expr.clone())],
                                        },
                                        *atom_span,
                                    ),
                                ],
                            },
                            *atom_span,
                        ));
                    }
                    None => elements.push(param.clone()),
                }
            }
            // Already-typed params (MetaExpr / List forms) are left as-is.
            _ => elements.push(param.clone()),
        }
    }
    deep::Expr::List(deep::List { elements }, *span)
}

/// Annotate the children of a `(fn ...)` node.
///
/// `declared_param_type_exprs`, when present, is the parameter type
/// expression list from the enclosing def's declared `sig`. It is used
/// to stamp the `(params ...)` node, preserving `(t-ref ...)` borrow
/// wrappers verbatim. `fn` literals with no declared signature pass
/// `None` and keep bare params.
pub(super) fn annotate_fn_children(
    list: &deep::List,
    owner_expr: &deep::Expr,
    product: &InferenceProduct,
    declared_param_type_exprs: Option<&[deep::Expr]>,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> (Vec<deep::Expr>, Type) {
    let kids = children(list);
    if kids.is_empty() {
        return (vec![], Type::Unit);
    }

    let resolved_fn_ty = product
        .owner_type(owner_expr, "function node", errors)
        .unwrap_or(Type::Unit);
    // Issue #319/#773: declared parameter syntax is copied from the owning
    // `sig` so borrow wrappers and shared symbolic variables remain exact.
    // Body expression types are not recomputed here: they come from the
    // completed `InferenceProduct` epoch, eliminating the former fresh-var /
    // fresh-substitution annotation collision. A bare `fn` with no declared
    // signature keeps bare parameters so signature metadata can still infer
    // its read-only display form.
    let annotated_params = match declared_param_type_exprs {
        Some(declared) => annotate_params_node(&kids[0], declared),
        None => kids[0].clone(),
    };
    debug_assert_eq!(
        child_stamp_role(DeepTag::Fn, 0, kids.len()),
        ChildStampRole::Binder
    );
    // The params node is the binder owner's completed output. A typed
    // parameter is represented as `(name {type: ...})`, which is binder
    // syntax rather than an expression tagged `name`; recursively feeding it
    // back through expression annotation would misreport every valid stamped
    // parameter as `UnknownForm`. Malformed params are rejected by
    // the primary parameter owner, exactly once, through the session sink.
    let mut result = vec![annotated_params];
    if let Some(body) = kids.get(1) {
        result.push(annotate_child_for_role(
            DeepTag::Fn,
            1,
            kids.len(),
            body,
            product,
            annotation_context,
            errors,
        ));
    }
    (result, resolved_fn_ty)
}

pub(super) fn annotated_meta_map_with_override(
    list: &deep::List,
    expr: &deep::Expr,
    product: &InferenceProduct,
    precomputed_ty: Option<Type>,
    errors: &mut DiagnosticSink<'_>,
) -> deep::Expr {
    let meta_span = match list.elements.get(1) {
        Some(deep::Expr::Map(_, span)) => *span,
        _ => span_of_expr(expr),
    };
    let mut entries = get_meta(list)
        .map(|meta| meta.entries.clone())
        .unwrap_or_default();

    let ty_for_meta = if let Some(tag) = get_tag(list) {
        match (tag, precomputed_ty) {
            (DeepTag::Fn, Some(ty)) => Some(ty),
            (DeepTag::PatVar | DeepTag::PatAs, _) => {
                product.owner_type(expr, "pattern binding", errors)
            }
            (t, _) if should_attach_type_metadata(t) => {
                product.owner_type(expr, "metadata-eligible expression", errors)
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(ty) = ty_for_meta
        && !matches!(ty, Type::Error(_))
    {
        let ty_expr = type_to_legacy_deep_expr(&ty);
        if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == "type") {
            *existing = ty_expr;
        } else {
            entries.push(("type".to_string(), ty_expr));
        }
    }

    deep::Expr::Map(deep::MetaMap { entries }, meta_span)
}

pub(super) fn annotated_node_meta_with_override(
    tag: DeepTag,
    meta: &deep::MetaMap,
    expr: &deep::Expr,
    precomputed_ty: Option<Type>,
    product: &InferenceProduct,
    errors: &mut DiagnosticSink<'_>,
) -> deep::MetaMap {
    let mut entries = meta.entries.clone();
    let ty_for_meta = match (tag, precomputed_ty) {
        (DeepTag::Fn, Some(ty)) => Some(ty),
        (DeepTag::PatVar | DeepTag::PatAs, _) => {
            product.owner_type(expr, "pattern binding", errors)
        }
        (t, _) if should_attach_type_metadata(t) => {
            product.owner_type(expr, "metadata-eligible expression", errors)
        }
        _ => None,
    };

    if let Some(ty) = ty_for_meta
        && !matches!(ty, Type::Error(_))
    {
        let ty_expr = type_to_deep_expr(&ty);
        if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == "type") {
            *existing = ty_expr;
        } else {
            entries.push(("type".to_string(), ty_expr));
        }
    }

    deep::MetaMap { entries }
}

pub(super) fn should_attach_type_metadata(tag: DeepTag) -> bool {
    // chelis#731 Phase 3 / rt-c2e7c23d F5: a TOTAL match, not `!matches!(..)`.
    // As a negated match this policy silently answered `true` for any tag
    // added later, so the `Scratch63` mutation oracle could not reach it: a
    // new variant inherited metadata-eligibility instead of forcing a
    // decision. Exhaustive here means a new tag is a compile error at this
    // policy too, not only at the dispatch sites.
    match tag {
        // Structural, binding, pattern, type and dimension syntax: these
        // carry no inferred expression type, so no metadata is attached.
        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Let
        | DeepTag::Fn
        | DeepTag::Var
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Arm
        | DeepTag::Effects
        | DeepTag::Resource
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TRef
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit => false,
        // Expression-bearing forms: these own a type and receive the stamp.
        DeepTag::Def
        | DeepTag::App
        | DeepTag::Match
        | DeepTag::If
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Pipe
        | DeepTag::Block
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::HandleEffect
        | DeepTag::Borrow
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice => true,
    }
}
