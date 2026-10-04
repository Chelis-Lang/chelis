//! Checked Deep type annotation.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// Explicit annotation-time declaration context. The declared signature map
/// belongs to one annotation unit; `current_type_binders` is narrowed to the
/// `def` whose children are being annotated and is passed through every
/// recursive annotation call.
#[derive(Clone, Copy)]
pub(super) struct AnnotationResolutionContext<'a> {
    declared_signatures: &'a UnordMap<String, DeclaredSigMetadata>,
}

impl<'a> AnnotationResolutionContext<'a> {
    pub(super) fn root(declared_signatures: &'a UnordMap<String, DeclaredSigMetadata>) -> Self {
        Self {
            declared_signatures,
        }
    }

    pub(super) fn declared_signature(self, name: &str) -> Option<&'a DeclaredSigMetadata> {
        self.declared_signatures.get(name)
    }
}

/// Semantic role of one tagged Deep node's child in checker-owned type
/// stamping. This is deliberately distinct from the child's syntactic tag:
/// a `lit` is a runtime expression under `app`, but the same shape is selector
/// syntax in `tuple-get`, `grad`, or `vmap`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ChildStampRole {
    /// Traversed by ordinary expression inference.
    RuntimeExpr,
    /// Compiler/source syntax which is preserved verbatim.
    Syntax,
    /// A field, axis, projection, or transform selector.
    Selector,
    /// Handler payload syntax whose literal-form contract is owned by
    /// `chelis-effects`, not expression inference.
    EffectHandler,
    /// A declaration, parameter, or binding name.
    Binder,
    /// Type/dimension syntax resolved by its owning type consumer.
    Type,
    /// Traversed by a dedicated inference owner rather than `infer_expr` on
    /// the structural parent (module declarations, patterns, helper nodes,
    /// and synthesized pipe stages).
    ExplicitInferenceBypass,
}

/// Exhaustive child-role table for the canonical closed Deep vocabulary.
///
/// Returning `None` is a loud version-skew signal, never permission to treat
/// an unknown child as a runtime expression. The completeness test below
/// iterates `chelis_deep::validate::VALID_TAGS`, the grammar's single source
/// of truth, so adding a tag requires an explicit ownership decision here.
pub(super) fn child_stamp_role(tag: DeepTag, index: usize, _arity: usize) -> ChildStampRole {
    use ChildStampRole::{
        Binder, EffectHandler, ExplicitInferenceBypass, RuntimeExpr, Selector, Syntax, Type,
    };

    match tag {
        // Module wrappers are not inferred as one expression. Their
        // declarations each own a separate inference epoch.
        DeepTag::Module => {
            if index == 0 {
                Binder
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Import | DeepTag::ImportAll | DeepTag::Export => Syntax,

        // Declarations.
        DeepTag::Def => {
            if index == 0 {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Defsig => match (index, _arity) {
            (0, _) => Binder,
            (1, 3..) => Syntax,
            _ => Type,
        },
        DeepTag::Deftype | DeepTag::Typealias => {
            if index == 0 {
                Binder
            } else if index == 1 {
                Syntax
            } else {
                Type
            }
        }
        DeepTag::Variant | DeepTag::Field => {
            if index == 0 {
                Binder
            } else {
                Type
            }
        }
        DeepTag::Defdim => Binder,

        // Expressions and their structural helper positions.
        DeepTag::Fn => {
            if index == 0 {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::App
        | DeepTag::If
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::Par
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Copy
        | DeepTag::Borrow
        | DeepTag::Unquote
        | DeepTag::Splice => RuntimeExpr,
        DeepTag::HandleEffect => {
            if index == 0 {
                EffectHandler
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Let => {
            if index == 0 {
                ExplicitInferenceBypass
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Match => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Arm => {
            if index == 0 {
                ExplicitInferenceBypass
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Var | DeepTag::Lit => Syntax,
        DeepTag::Record => {
            if index == 0 {
                Type
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Access => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::TupleGet => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::RecordUpdate => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }

        // Pattern nodes are consumed by the primary pattern traversal.
        DeepTag::PatVar => Binder,
        DeepTag::PatLit => Syntax,
        DeepTag::PatCtor | DeepTag::PatRecord => {
            if index == 0 {
                Selector
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::PatTuple => ExplicitInferenceBypass,
        DeepTag::PatWild => Syntax,
        DeepTag::PatAs => {
            if index == 0 {
                Binder
            } else {
                ExplicitInferenceBypass
            }
        }

        // Type and dimension nodes are owned recursively by DeepTypeResolver,
        // never by expression annotation.
        DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TRef
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank => Type,

        // Transform-specific selector/type positions.
        DeepTag::Grad | DeepTag::Vmap => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::Cast => {
            if index == 0 {
                RuntimeExpr
            } else {
                Type
            }
        }

        // Quoted children and effect/resource payloads are syntax data.
        DeepTag::Quote | DeepTag::Effects | DeepTag::Resource => Syntax,

        // Structural helper nodes. `kv` is also used by pattern records, so
        // its value/pattern slot is an explicit owning traversal in both
        // contexts; canonical runtime values still record their normal stamp.
        DeepTag::Params => Binder,
        DeepTag::Bind => {
            if index.is_multiple_of(2) {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Kv => {
            if index == 0 {
                Selector
            } else {
                ExplicitInferenceBypass
            }
        }
    }
}

pub(super) fn annotate_ir_program(
    exprs: &[deep::Expr],
    product: &InferenceProduct,
    errors: &mut DiagnosticSink<'_>,
) -> Vec<deep::Expr> {
    let items = top_level_decl_items_with_modules(exprs);
    let declared_signatures = collect_declared_sig_metadata(items.iter().map(|(_, expr)| *expr));
    let annotation_context = AnnotationResolutionContext::root(&declared_signatures);
    annotate_top_levels(exprs, product, annotation_context, errors)
}

/// Annotate each top-level expression, stopping early if cancellation was
/// requested (chelis#930).
///
/// Annotation is the third front-end pass whose cost scales with declaration
/// count — measured at ~13.8s on the Coral library in the perf baseline — so
/// it polls per top level for the same reason inference does. A short result
/// vector is never consumed: every caller runs a `cancellation_gate`
/// immediately after and returns the hard failure.
pub(super) fn annotate_top_levels(
    exprs: &[deep::Expr],
    product: &InferenceProduct,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> Vec<deep::Expr> {
    let cancel = crate::cancel::current_cancel_token();
    let mut annotated = Vec::with_capacity(exprs.len());
    for expr in exprs {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            break;
        }
        annotated.push(annotate_expr_with_scope(
            expr,
            product,
            annotation_context,
            errors,
        ));
    }
    annotated
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
    // An exhaustive representation-preserving transformer: every arm
    // rebuilds the same carrier it received.
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
                metadata: meta.metadata.clone(),
            },
            *span,
        ),
        deep::Expr::Node(node, span) => {
            let tag = node.tag();
            let children = node.children_slice();
            let fn_ty_override = (tag == DeepTag::Fn)
                .then(|| product.owner_type(expr, "function node", errors))
                .flatten();
            let declared_param_types = (tag == DeepTag::Def)
                .then(|| {
                    children
                        .first()
                        .and_then(symbol_name)
                        .and_then(|name| annotation_context.declared_signature(name))
                        .map(|metadata| metadata.param_types.as_slice())
                })
                .flatten();
            let annotated_children = children
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    let role = child_stamp_role(tag, index, children.len());
                    if let (DeepTag::Def, ChildStampRole::RuntimeExpr, Some(declared)) =
                        (tag, role, declared_param_types)
                        && let Some(annotated) = annotate_declared_fn_child(
                            child,
                            expr,
                            declared,
                            product,
                            annotation_context,
                            errors,
                        )
                    {
                        annotated
                    } else {
                        annotate_child_for_role(
                            tag,
                            index,
                            children.len(),
                            child,
                            product,
                            annotation_context,
                            errors,
                        )
                    }
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

/// True for a whole-slot inference hole in a signature or parameter annotation.
pub(super) fn is_wildcard_tvar_expr(expr: &deep::Expr) -> bool {
    // chelis#1107 amendment: carrier-preserving read.
    matches!(stamped_parts(expr), Some((DeepTag::TVar, _, kids))
        if matches!(kids, [name] if symbol_name(name) == Some("_")))
}

/// Rebuild a `(params ...)` node so every previously-bare parameter
/// symbol carries a `{type: ...}` metadata entry, using the declared
/// signature's parameter type *expressions* (`declared_param_type_exprs`).
///
/// Inline annotations now leave whole-slot holes on parameters, with the
/// actual types in the signature. Standalone signatures can have bare
/// parameters. Both forms need the declared type here. IR lowering's
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
/// Real annotations remain untouched. Only whole-slot holes receive the
/// corresponding declared type, with all other binder metadata intact.
pub(super) fn annotate_params_node(
    params_expr: &deep::Expr,
    declared_param_type_exprs: &[deep::Expr],
) -> deep::Expr {
    // A newly annotated name-headed binder is a structural `BareList`.
    let (metadata, children) = match params_expr.carrier() {
        deep::ExprCarrier::DecodedNode(DeepTag::Params, metadata, children) => (metadata, children),
        deep::ExprCarrier::DecodedNode(_, _, _)
        | deep::ExprCarrier::StructuralList(_)
        | deep::ExprCarrier::UndecodableHead(_, _, _)
        | deep::ExprCarrier::Atom(_)
        | deep::ExprCarrier::MetadataMap(_)
        | deep::ExprCarrier::MetadataExpression(_) => return params_expr.clone(),
    };
    let mut annotated_children = Vec::with_capacity(children.len());
    for (index, param) in children.iter().enumerate() {
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
                        let elements = vec![
                            deep::Expr::Atom(deep::Atom::Name(name.clone()), *atom_span),
                            deep::Expr::Map(
                                deep::Metadata::from(
                                    chelis_deep::annotations::MetadataValue::Type(
                                        chelis_deep::annotations::TypeSyntax::try_new(
                                            type_expr.clone(),
                                        )
                                        .expect("declared parameter type syntax"),
                                    ),
                                ),
                                *atom_span,
                            ),
                        ];
                        annotated_children.push(deep::Expr::BareList(elements, *atom_span));
                    }
                    None => annotated_children.push(param.clone()),
                }
            }
            _ => annotated_children.push(
                match declared_param_type_exprs
                    .get(index)
                    .filter(|ty| !is_wildcard_tvar_expr(ty))
                {
                    Some(declared) => annotate_parameter_hole(param, declared),
                    None => param.clone(),
                },
            ),
        }
    }

    deep::Expr::node(
        DeepTag::Params,
        metadata.clone(),
        annotated_children,
        params_expr.span(),
    )
}

fn annotate_parameter_hole(param: &deep::Expr, declared: &deep::Expr) -> deep::Expr {
    let mut annotated = param.clone();
    // Producer-side carrier preservation: this mutates a private clone and
    // returns the same representation. It does not decide whether an input
    // carrier is semantically readable.
    let metadata = match &mut annotated {
        deep::Expr::MetaExpr(meta, _) => Some(&mut meta.metadata),
        deep::Expr::BareList(elements, _) => match elements.get_mut(1) {
            Some(deep::Expr::Map(meta, _)) => Some(meta),
            _ => None,
        },
        _ => None,
    };
    if let Some(metadata) = metadata
        && metadata
            .ty()
            .is_some_and(|ty| is_wildcard_tvar_expr(ty.expression()))
    {
        metadata.replace(chelis_deep::annotations::MetadataValue::Type(
            chelis_deep::annotations::TypeSyntax::try_new(declared.clone())
                .expect("declared parameter type syntax"),
        ));
    }
    annotated
}

/// Annotate the children of a `(fn ...)` node.
///
/// `declared_param_type_exprs`, when present, is the parameter type
/// expression list from the enclosing def's declared `sig`. It is used
/// to stamp the `(params ...)` node, preserving `(t-ref ...)` borrow
/// wrappers verbatim. `fn` literals with no declared signature pass
/// `None` and keep bare params.
pub(super) fn annotate_fn_children(
    fn_expr: &deep::Expr,
    product: &InferenceProduct,
    declared_param_type_exprs: Option<&[deep::Expr]>,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> (Vec<deep::Expr>, Type) {
    let deep::ExprCarrier::DecodedNode(DeepTag::Fn, _, kids) = fn_expr.carrier() else {
        return (vec![], Type::Unit);
    };
    if kids.is_empty() {
        return (vec![], Type::Unit);
    }

    let resolved_fn_ty = product
        .owner_type(fn_expr, "function node", errors)
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

fn annotate_declared_fn_child(
    child: &deep::Expr,
    declaration_expr: &deep::Expr,
    declared_param_type_exprs: &[deep::Expr],
    product: &InferenceProduct,
    annotation_context: AnnotationResolutionContext<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> Option<deep::Expr> {
    let deep::ExprCarrier::DecodedNode(DeepTag::Fn, metadata, _) = child.carrier() else {
        return None;
    };
    let (annotated_children, inferred_fn_ty) = annotate_fn_children(
        child,
        product,
        Some(declared_param_type_exprs),
        annotation_context,
        errors,
    );
    // The declaration owner's completed type retains the authored result
    // claim; the body owner remains the independent inferred type used to
    // check that claim.
    let fn_ty = product
        .owner_type(declaration_expr, "declared function", errors)
        .unwrap_or(inferred_fn_ty);

    Some(deep::Expr::node(
        DeepTag::Fn,
        annotated_node_meta_with_override(
            DeepTag::Fn,
            metadata,
            child,
            Some(fn_ty),
            product,
            errors,
        ),
        annotated_children,
        child.span(),
    ))
}

pub(super) fn annotated_node_meta_with_override(
    tag: DeepTag,
    meta: &deep::Metadata,
    expr: &deep::Expr,
    precomputed_ty: Option<Type>,
    product: &InferenceProduct,
    errors: &mut DiagnosticSink<'_>,
) -> deep::Metadata {
    let mut metadata = meta.clone();
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

    if let Some(ty) = ty_for_meta {
        write_type_metadata_monotone(&mut metadata, ty, type_to_deep_expr);
    }

    metadata
}

/// Write checker-owned metadata only when doing so preserves or increases
/// information (#783). An unresolved, error, or partially resolved candidate
/// never replaces existing metadata; a safe resolved candidate may refresh it.
fn write_type_metadata_monotone(
    metadata: &mut deep::Metadata,
    ty: Type,
    encode: fn(&Type) -> deep::Expr,
) {
    let write = if metadata.ty().is_some() {
        type_is_safe_annotation_stamp(&ty)
    } else {
        !matches!(ty, Type::Error(_))
    };
    if write {
        metadata.replace(chelis_deep::annotations::MetadataValue::Type(
            chelis_deep::annotations::TypeSyntax::try_new(encode(&ty))
                .expect("checker emits valid type syntax"),
        ));
    }
}

fn type_is_safe_annotation_stamp(ty: &Type) -> bool {
    match ty {
        Type::Prim(_) | Type::Unit => true,
        Type::Fn(args, ret) => {
            args.iter().all(type_is_safe_annotation_stamp) && type_is_safe_annotation_stamp(ret)
        }
        Type::Ref(inner) => type_is_safe_annotation_stamp(inner),
        Type::Tensor(dims, precision) => {
            matches!(precision, TensorPrec::Concrete(_))
                && dims
                    .iter()
                    .all(|dim| matches!(dim, Dim::Name(_) | Dim::Lit(_)))
        }
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().all(type_is_safe_annotation_stamp),
        Type::KindedAdt(_, args) => args.iter().all(|argument| match argument {
            NominalArg::Type(ty) => type_is_safe_annotation_stamp(ty),
            NominalArg::Dimension(dim) => matches!(dim, Dim::Name(_) | Dim::Lit(_)),
        }),
        Type::Var(_) | Type::Error(_) => false,
    }
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

#[cfg(test)]
mod monotone_writeback_tests {
    use super::*;

    #[test]
    fn unresolved_dimension_variable_cannot_replace_concrete_metadata() {
        let concrete = Type::Tensor(
            vec![Dim::Lit(4), Dim::Lit(4)],
            TensorPrec::Concrete(Prim::F32),
        );
        let original = type_to_deep_expr(&concrete);
        let mut metadata = deep::Metadata::default();
        metadata
            .insert(chelis_deep::annotations::MetadataValue::Type(
                chelis_deep::annotations::TypeSyntax::try_new(original.clone()).unwrap(),
            ))
            .unwrap();

        write_type_metadata_monotone(
            &mut metadata,
            Type::Tensor(
                vec![Dim::Var(DimVar(9001)), Dim::Lit(4)],
                TensorPrec::Concrete(Prim::F32),
            ),
            type_to_deep_expr,
        );

        assert_eq!(
            metadata.ty().map(|ty| ty.expression()),
            Some(&original),
            "an unresolved dimension is less informative than an existing concrete shape"
        );
    }

    #[test]
    fn unresolved_dimension_can_be_written_when_no_annotation_exists() {
        let candidate = Type::Tensor(
            vec![Dim::Var(DimVar(9002)), Dim::Lit(4)],
            TensorPrec::Concrete(Prim::F32),
        );
        let mut metadata = deep::Metadata::default();

        write_type_metadata_monotone(&mut metadata, candidate, type_to_deep_expr);

        assert_eq!(
            metadata.values().count(),
            1,
            "symbolic inferred metadata still has an owner"
        );
    }
}
