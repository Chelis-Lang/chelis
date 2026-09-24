//! Function, definition, let, conditional, and pipe inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// Recover the exact Surf source range stamped onto a desugared Deep node.
///
/// Ordinary desugared expressions retain a zero structural span, while their
/// authored range travels in canonical `surf:<start>..<end>` metadata.
/// Programmatic or non-Surf Deep inputs fall back to the structural span.
fn surf_source_span(expr: &deep::Expr) -> Span {
    let structural_span = expr.span();
    let Some((_, meta, _)) = stamped_parts(expr) else {
        return structural_span;
    };
    let Some(span_id) = meta.span_id() else {
        return structural_span;
    };
    let Some(range) = span_id.value().strip_prefix("surf:") else {
        return structural_span;
    };
    let Some((start, end)) = range.split_once("..") else {
        return structural_span;
    };
    let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) else {
        return structural_span;
    };
    let Some(len) = end.checked_sub(start) else {
        return structural_span;
    };
    Span::new(start, len)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_fn(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Type {
    let kids = node.children_slice();
    if kids.is_empty() {
        return malformed_form(node, "fn", "parameters and a body", errors);
    }

    // kids[0] = (params {} x1 ... xn)
    // kids[1] = body
    let params = extract_params_with_ownership(
        &kids[0],
        &DefParameterAnnotationOwnership::independent(),
        vg,
        adt_reg,
        errors,
        annotation_binder_mode(env),
        declaration_diagnostic_owner,
    );
    let mut param_types = Vec::new();
    let mut fn_env = env.clone();

    for (pname, ty_ann) in &params {
        let ty = ty_ann.clone().unwrap_or_else(|| vg.fresh_type());
        product.record_inferred_contract(&format!("function parameter `{pname}`"), &ty, env, subst);
        fn_env.bind_lexical(pname.clone(), Scheme::mono(ty.clone()));
        // chelis#397/#469: a parameter is a fresh runtime binding with no
        // size provenance. Clear any entry inherited (through the derived
        // `Clone` of `env`) from an outer name it shadows, so a sourceless
        // value parameter `d` shadowing an outer shape-sourced `d` (BLOCKER C)
        // is not wrongly treated as a materializable extent.
        fn_env.clear_size_provenance(pname);
        // chelis#631: same for a shadowed list-literal length.
        fn_env.clear_list_literal_len(pname);
        param_types.push(ty);
    }

    // Snapshot the dimension variables introduced by the declared parameter
    // signatures. The post-body `check_declared_dvars_rigid` call flags both
    // (a) a declared dim forced to a concrete literal (Nautilus Bug 2) and
    // (b) two distinct declared dims collapsed into one another by the body
    // (Path B of TypeCheck-FreeDimVarUnification-F1). We flag this post-body
    // so legitimate polymorphic uses (where each dvar stays unbound and
    // distinct) still type-check.
    let mut declared_dvars: Vec<DimVar> = Vec::new();
    for t in &param_types {
        for dv in crate::env::free_dvars(t) {
            if !declared_dvars.contains(&dv) {
                declared_dvars.push(dv);
            }
        }
    }

    subst.protect_dimensions(declared_dvars.iter().copied());
    let body = if kids.len() > 1 {
        &kids[1]
    } else {
        return malformed_form(node, "fn", "a body expression", errors);
    };
    let body_ty = infer_expr(body, &mut fn_env, vg, subst, adt_reg, errors, product);

    // chelis#260: no recorded names on this path. A dim binder has to be
    // declared as `def f[n, m]`, which routes through the `Defsig` arm and
    // then the annotated-def site; an `fn` whose annotation mentions `n`
    // without such a declaration is rejected earlier as an undeclared
    // dimension variable. So this call cannot currently reach a named
    // collapse, and passing an empty map renders the internal id rather
    // than inventing a name.
    check_declared_dvars_rigid(None, &declared_dvars, &UnordMap::new(), subst, errors);

    let resolved_params: Vec<Type> = param_types.iter().map(|t| subst.apply(t)).collect();
    let resolved_body = subst.apply(&body_ty);

    Type::Fn(resolved_params, Box::new(resolved_body))
}

/// Whether one annotated `def` parameter is an independent contract or a
/// structurally verified copy of its adjacent signature slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DefParameterSlotOwnership {
    Independent,
    VerifiedSignatureCopy,
}

/// Position-aligned ownership for the supplied parameters of one `def`.
///
/// An absent entry is independent. That makes a malformed outer carrier fail
/// closed, while an arity disagreement cannot revoke ownership already proved
/// for another canonical parameter position.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct DefParameterAnnotationOwnership {
    slots: Vec<DefParameterSlotOwnership>,
}

impl DefParameterAnnotationOwnership {
    fn independent() -> Self {
        Self::default()
    }

    fn is_verified_signature_copy(&self, index: usize) -> bool {
        self.slots.get(index) == Some(&DefParameterSlotOwnership::VerifiedSignatureCopy)
    }
}

/// Classify generated property copies from their raw Deep syntax.
///
/// Source spans are deliberately excluded, while semantic metadata remains
/// part of equality. Any malformed or noncanonical carrier stays independent.
pub(super) fn classify_property_parameter_annotations(
    body: &deep::Expr,
    declared_param_types: &[deep::Expr],
) -> DefParameterAnnotationOwnership {
    let Some((DeepTag::Fn, _, fn_children)) = stamped_parts(body) else {
        return DefParameterAnnotationOwnership::independent();
    };
    let Some(params) = fn_children.first().and_then(canonical_parameter_elements) else {
        return DefParameterAnnotationOwnership::independent();
    };
    let slots = params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let verified = declared_param_types.get(index).is_some_and(|declared| {
                parameter_type_syntax(param).is_some_and(|annotation| {
                    chelis_deep::metadata::same_semantic_type_syntax(annotation, declared)
                })
            });
            if verified {
                DefParameterSlotOwnership::VerifiedSignatureCopy
            } else {
                DefParameterSlotOwnership::Independent
            }
        })
        .collect();
    DefParameterAnnotationOwnership { slots }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_declared_def_body(
    body: &deep::Expr,
    decl_ty: &Type,
    declaration_meta: &deep::Metadata,
    declared_signature: Option<&DeclaredSigMetadata>,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Type {
    let property_has_quantifier_carriers = declaration_meta
        .chelis_role()
        .is_some_and(|role| role.value() == "property")
        && declaration_meta.property_quantifiers().is_some();
    let parameter_annotation_ownership = if property_has_quantifier_carriers {
        declared_signature.map_or_else(DefParameterAnnotationOwnership::independent, |signature| {
            classify_property_parameter_annotations(body, &signature.param_types)
        })
    } else {
        DefParameterAnnotationOwnership::independent()
    };
    let inferred = infer_def_body_with_sig(
        body,
        decl_ty,
        &parameter_annotation_ownership,
        env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
        declaration_diagnostic_owner,
    );
    product.record_bypass(
        body,
        inferred.clone(),
        "declared-signature function inference",
    );
    inferred
}

/// WS-A7: infer a `def`'s body when a declared signature is available, seeding
/// any bare-arg parameters of an outer `(fn ...)` body with the declared
/// signature's param types. This eliminates the bare-arg + sig-with-borrows
/// return-type miscompile where call-site auto-borrow on an unconstrained
/// param tvar collapses the param-side and return-side of the callee scheme
/// (e.g. `add: (&t, &t) -> t`) into the same equivalence class.
///
/// Falls back to the standard `infer_expr` path when the body is not a
/// `(fn ...)` or the declared type is not a `Fn`. The post-body unification
/// reports arity disagreements. Real parameter annotations are not overridden.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_def_body_with_sig(
    body: &deep::Expr,
    decl_ty: &Type,
    parameter_annotation_ownership: &DefParameterAnnotationOwnership,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Type {
    // Match: body is `(fn (params ...) body-expr)` AND decl is `Fn(args, ret)`.
    let Some((DeepTag::Fn, _, kids)) = stamped_parts(body) else {
        return infer_expr_with_declaration_diagnostic_owner(
            body,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            product,
            declaration_diagnostic_owner,
        );
    };
    let (decl_args, decl_ret) = match decl_ty {
        Type::Fn(args, ret) => (args, ret.as_ref()),
        _ => {
            return infer_expr_with_declaration_diagnostic_owner(
                body,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                product,
                declaration_diagnostic_owner,
            );
        }
    };

    let params = extract_params_with_ownership(
        &kids[0],
        parameter_annotation_ownership,
        vg,
        adt_reg,
        errors,
        annotation_binder_mode(env),
        declaration_diagnostic_owner,
    );
    let mut param_types = Vec::with_capacity(params.len());
    let mut fn_env = env.clone();
    for (index, (pname, ty_ann)) in params.iter().enumerate() {
        // Resolve each annotation once, even when arities disagree. Existing
        // slots seed bare/hole parameters. An extra parameter has no declared
        // slot, so its body determines its type. Post-body unification still
        // reports the arity mismatch instead of replaying the function.
        let ty = ty_ann
            .clone()
            .or_else(|| decl_args.get(index).cloned())
            .unwrap_or_else(|| vg.fresh_type());
        product.record_inferred_contract(&format!("function parameter `{pname}`"), &ty, env, subst);
        fn_env.bind_lexical(pname.clone(), Scheme::mono(ty.clone()));
        // chelis#397/#469: a fresh parameter has no size provenance; clear any
        // entry inherited from an outer name it shadows (BLOCKER C).
        fn_env.clear_size_provenance(pname);
        // chelis#631: same for a shadowed list-literal length.
        fn_env.clear_list_literal_len(pname);
        param_types.push(ty);
    }

    // Note: the declared-dim rigidity check (both the Var->Lit pin and
    // the Var->Var collapse of TypeCheck-FreeDimVarUnification-F1) runs
    // at the caller's defsig site, *after* the post-body sig-unify. The
    // sig-unify is where two distinct declared dims actually collapse
    // for an annotated-param body like
    // `def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) ->
    // tensor[n, f32] = y`, so checking here (pre-sig-unify) would miss
    // it. Running it only at the caller also avoids double-reporting.
    // Annotated parameters have their own fresh resolution IDs; protecting
    // only the separate defsig instantiation would be too late for the body.
    subst.protect_dimensions(param_types.iter().flat_map(crate::env::free_dvars));
    let body_expr = &kids[1];
    let body_ty = infer_expr_with_expected(
        body_expr,
        decl_ret,
        &mut fn_env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
    );

    let resolved_params: Vec<Type> = param_types.iter().map(|t| subst.apply(t)).collect();
    let resolved_body = subst.apply(&body_ty);

    Type::Fn(resolved_params, Box::new(resolved_body))
}

fn canonical_parameter_elements(expr: &deep::Expr) -> Option<&[deep::Expr]> {
    match expr {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => Some(node.children_slice()),
        _ => None,
    }
}

fn parameter_elements(expr: &deep::Expr) -> Option<&[deep::Expr]> {
    match expr {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => Some(node.children_slice()),
        deep::Expr::BareList(elements, _) => Some(elements.as_slice()),
        _ => None,
    }
}

fn parameter_type_syntax(expr: &deep::Expr) -> Option<&deep::Expr> {
    match expr {
        deep::Expr::MetaExpr(meta, _) => meta.metadata.ty().map(|value| value.expression()),
        deep::Expr::BareList(elements, _) => elements.get(1).and_then(|meta| {
            let deep::Expr::Map(meta, _) = meta else {
                return None;
            };
            meta.ty().map(|value| value.expression())
        }),
        _ => None,
    }
}

/// Extract parameter names (and optional type annotations) from (params {} x1 ... xn).
/// Each param can be a bare symbol, a metadata-annotated symbol, or a legacy
/// `(name {type: T})` helper pair.
fn extract_params_with_ownership(
    expr: &deep::Expr,
    ownership: &DefParameterAnnotationOwnership,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    binder_mode: BinderMode<'_>,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Vec<(String, Option<Type>)> {
    let Some(elems) = parameter_elements(expr) else {
        return Vec::new();
    };
    let mut resolver = DeepTypeResolver::new(
        TypeUseSite::Annotation,
        binder_mode,
        adt_reg.resolution_env(),
        vg,
        errors,
    )
    .with_declaration_diagnostic_owner(declaration_diagnostic_owner);
    let mut resolve_annotation =
        |annotation: &deep::Expr| match resolver.resolve_parameter(annotation) {
            Ok(ty) => ty.map(|ty| ty.into_type()),
            Err(witness) => Some(propagate(&witness)),
        };
    let mut params = Vec::with_capacity(elems.len());

    for (index, expr) in elems.iter().enumerate() {
        let verified_copy = ownership.is_verified_signature_copy(index);
        match expr {
            deep::Expr::Atom(deep::Atom::Name(name), _) => {
                params.push((name.to_string(), None));
            }
            deep::Expr::MetaExpr(meta, _) => {
                let deep::Expr::Atom(deep::Atom::Name(name), _) = meta.expr.as_ref() else {
                    continue;
                };
                let annotation = (!verified_copy)
                    .then(|| {
                        meta.metadata
                            .ty()
                            .and_then(|value| resolve_annotation(value.expression()))
                    })
                    .flatten();
                params.push((name.to_string(), annotation));
            }
            // Typed param: (name {type: T}).
            deep::Expr::BareList(elements, _) => {
                let Some(name) = elements.first().and_then(symbol_name) else {
                    continue;
                };
                let annotation = (!verified_copy)
                    .then(|| match elements.get(1) {
                        Some(deep::Expr::Map(meta, _)) => meta
                            .ty()
                            .and_then(|value| resolve_annotation(value.expression())),
                        _ => None,
                    })
                    .flatten();
                params.push((name.to_string(), annotation));
            }
            _ => {}
        }
    }
    drop(resolver);
    let mut aliases = AliasExpansionSession::new(adt_reg, vg);
    for (_, annotation) in &mut params {
        if let Some(ty) = annotation {
            *ty = aliases.resolve(ty);
        }
    }
    params
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_let(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    if kids.len() < 2 {
        return malformed_form(node, "let", "bindings and a body", errors);
    }

    // kids[0] = (bind {} x1 e1 x2 e2 ...)
    // kids[1] = body
    let mut let_env = env.clone();

    if let Some((DeepTag::Bind, _, bind_children)) = stamped_parts(&kids[0]) {
        // Process pairs: name, expr
        let mut i = 0;
        while i + 1 < bind_children.len() {
            if let Some(name) = symbol_name(&bind_children[i]) {
                let rhs_expr = &bind_children[i + 1];
                let rhs_level = subst.enter_level(vg);
                let shape_checkpoint = product.deferred_shape_checkpoint();
                let literal_pattern_checkpoint = product.deferred_literal_pattern_checkpoint();
                let contract_checkpoint = product.admission_contract_checkpoint();
                let mut rhs_type_metadata_resolution = stamped_parts(rhs_expr)
                    .and_then(|(_, meta, _)| {
                        let authored = meta
                            .surf_binding_type()
                            .is_none_or(|origin| *origin.value() == BindingTypeOrigin::Explicit);
                        authored.then(|| meta.ty().map(|value| value.expression()))?
                    })
                    .map(|declared_ty_expr| {
                        match resolve_deep_type_with_diagnostic_owner(
                            declared_ty_expr,
                            vg,
                            adt_reg,
                            TypeUseSite::Annotation,
                            annotation_binder_mode(&let_env),
                            let_env.type_resolution_diagnostic_owner(),
                            errors,
                        ) {
                            Ok(ty) => OwnedTypeMetadataResolution::Resolved(ty),
                            Err(witness) => OwnedTypeMetadataResolution::Failed(witness),
                        }
                    });
                let expr_ty = infer_expr_with_type_metadata_ownership(
                    rhs_expr,
                    &mut let_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    None,
                    Some(&mut rhs_type_metadata_resolution),
                    None,
                );
                let rhs_type_before_ascription = subst.apply(&expr_ty);
                let local_ascription_origin = match stamped_parts(rhs_expr)
                    .and_then(|(_, meta, _)| meta.surf_binding_type())
                    .map(|origin| *origin.value())
                {
                    Some(BindingTypeOrigin::Explicit) => {
                        Some(LocalTensorAscriptionOrigin::SurfExplicit)
                    }
                    Some(BindingTypeOrigin::Inferred) => None,
                    None => Some(LocalTensorAscriptionOrigin::DeepTypeMetadata),
                };
                let mut checked_local_ascription = None;

                // chelis#159: block-scoped `let name: T = expr` desugars
                // inject the declared type `T` as a `"type"` metadata
                // entry on the RHS node (crates/chelis-surf/src/desugar.rs:1374-1379
                // via `inject_type_metadata`). Pre-fix, infer_let did
                // not consult that metadata, so a generic-builtin RHS
                // like `to_tensor(...)` left its output type var free
                // and the ascription was silently dropped. Unify the
                // inferred RHS type against the declared type so the
                // ascription propagates into downstream sig calls.
                let final_ty = if let Some(declared_ty_expr) = stamped_parts(rhs_expr)
                    .and_then(|(_, meta, _)| meta.ty().map(|v| v.expression()))
                {
                    let declared_ty = match &rhs_type_metadata_resolution {
                        // A root metadata-aware RHS consumer records the exact
                        // result it owns. Reuse that result here so the same
                        // ascription is neither resolved nor reported twice.
                        Some(OwnedTypeMetadataResolution::Resolved(ty)) => ty.clone(),
                        Some(OwnedTypeMetadataResolution::Failed(witness)) => propagate(witness),
                        // No ownership record means any `expr_ty` error came
                        // from the RHS itself. Resolve the ascription as its
                        // own root so two independent failures both surface.
                        None => match resolve_deep_type_with_diagnostic_owner(
                            declared_ty_expr,
                            vg,
                            adt_reg,
                            TypeUseSite::Annotation,
                            annotation_binder_mode(&let_env),
                            let_env.type_resolution_diagnostic_owner(),
                            errors,
                        ) {
                            Ok(ty) => ty,
                            Err(witness) => propagate(&witness),
                        },
                    };
                    match unify(&expr_ty, &declared_ty, subst) {
                        Ok(()) if local_ascription_origin.is_some() => {
                            checked_local_ascription = Some((
                                local_ascription_origin.expect("checked above"),
                                declared_ty_expr.clone(),
                                declared_ty.clone(),
                            ));
                        }
                        Ok(()) => {}
                        Err(e) => {
                            let mut diagnostic = CheckError::new(
                                check_error_kind_from_type_error_kind(&e.kind),
                                format!(
                                    "let-binding `{name}` ascription does not match RHS: {}",
                                    e.message
                                ),
                                vec![format!(
                                    "Declared type for `{name}` is {declared_ty}; \
                                     RHS inferred to {expr_ty}"
                                )],
                            );
                            if let Some(location) =
                                TypeDiagnosticLocation::from_expr(declared_ty_expr)
                                    .or_else(|| TypeDiagnosticLocation::from_expr(rhs_expr))
                            {
                                diagnostic = location.attach(diagnostic);
                            }
                            errors.push(diagnostic);
                        }
                    }
                    // On unify failure, bind `name` to the declared
                    // type rather than the inferred RHS type. This
                    // produces a cleaner error cascade: downstream uses
                    // of `name` see what the user said they meant, not
                    // what the (already-rejected) RHS inferred to, so
                    // a single ascription-mismatch diagnostic stands
                    // alone instead of fanning out into multiple
                    // downstream errors. The trade-off: pathological
                    // bodies where the user's ascription is *also*
                    // independently wrong against later code may have
                    // a second mismatch masked. The single-error
                    // cascade is the better default for chelis#159's
                    // user-facing diagnostic ergonomics.
                    declared_ty
                } else {
                    expr_ty
                };
                if let Some((origin, authored_type, declared_type)) = checked_local_ascription {
                    product.record_local_tensor_ascription(
                        origin,
                        name,
                        bind_children[i].span(),
                        authored_type.span(),
                        surf_source_span(rhs_expr),
                        authored_type,
                        declared_type,
                        &rhs_type_before_ascription,
                        errors,
                    );
                }

                subst.leave_level(rhs_level, vg);

                let scheme = if product.has_pending_shape_check_since(shape_checkpoint)
                    || product.has_pending_literal_pattern_since(literal_pattern_checkpoint, subst)
                    || product.has_pending_admission_contract_since(contract_checkpoint, subst)
                {
                    // [04-INF-1]: semantic shape or literal-pattern
                    // obligations retain the exact inference variables
                    // captured by this lambda until its first application.
                    subst.lower_type_to_current(&final_ty);
                    Scheme::mono(subst.apply(&final_ty))
                } else {
                    let_env.generalize(&final_ty, subst)
                };
                // chelis#397/#469: record the size provenance of this binding
                // BEFORE binding it (so `classify_expand_size` resolves it
                // against the binding's RHS, not its own name) so a later
                // `expand(b, 0, name)` can recover whether `name` is a
                // materializable extent (static / shape-sourced) or a
                // sourceless runtime scalar. Bound BEFORE `let_env.bind` so
                // the RHS is classified against the pre-binding scope, and
                // transitively through earlier bindings in the same block.
                // The `Sourceless`/`Unknown` arm CLEARS any stale provenance so
                // a re-bind to a sourceless RHS — `len = shape(x, 0); len = k`
                // (BLOCKER B) — does not inherit the earlier shape-sourced entry.
                match classify_expand_size(rhs_expr, &let_env, adt_reg, subst) {
                    SizeClass::Static => {
                        if let Some(value) =
                            fold_static_int_expr(rhs_expr, |bound| let_env.static_size_value(bound))
                        {
                            let_env.mark_static_size_value(name, value);
                        } else {
                            let_env.mark_size_provenance(name, crate::env::SizeProvenance::Static);
                        }
                    }
                    SizeClass::ShapeSourced => {
                        let_env
                            .mark_size_provenance(name, crate::env::SizeProvenance::ShapeSourced);
                    }
                    SizeClass::Sourceless | SizeClass::Unknown => {
                        let_env.clear_size_provenance(name)
                    }
                }
                // chelis#631: same discipline for list-literal lengths.
                note_list_literal_binding(&mut let_env, name, rhs_expr);
                let_env.bind_lexical(name.to_string(), scheme);
            }
            i += 2;
        }
    }

    infer_expr(&kids[1], &mut let_env, vg, subst, adt_reg, errors, product)
}

/// The obligation every condition position owes: an `if` condition and a
/// `match` arm guard each require a `bool`, discharged by unifying with it.
///
/// chelis#2444: the guard used to apply the substitution and require the
/// result to already be `bool`, so a condition whose type was still pending
/// (an `eq` over a dtype-binder operand, which could yet be a `bool` scalar or
/// a `bool` tensor) was rejected as a guard and accepted as an `if` condition.
/// One function makes the two positions one rule; only the position named in
/// the diagnostic differs.
pub(super) fn require_bool_condition(
    cond_ty: &Type,
    position: &str,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    if unify(cond_ty, &Type::Prim(Prim::Bool), subst).is_err() {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("{position} must be bool, got {}", subst.apply(cond_ty)),
            vec![],
        ));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_if(
    expr: &deep::Expr,
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    if kids.len() < 3 {
        return malformed_form(
            node,
            "if",
            "a condition, a then-branch, and an else-branch",
            errors,
        );
    }

    let cond_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    require_bool_condition(&cond_ty, "if condition", subst, errors);

    let then_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let else_ty = infer_expr(&kids[2], env, vg, subst, adt_reg, errors, product);

    match unify(&then_ty, &else_ty, subst) {
        Ok(()) => subst.apply(&then_ty),
        Err(te) => {
            let mut e: CheckError = te.into();
            if let Some(id) = node_span_id(node) {
                e.span_offset = parse_span_offset(id);
                e.span_id = Some(id.to_string());
            } else {
                let off = expr.span().offset;
                if off > 0 {
                    e.span_offset = Some(off);
                }
            }
            errors.push(e);
            subst.apply(&then_ty)
        }
    }
}
