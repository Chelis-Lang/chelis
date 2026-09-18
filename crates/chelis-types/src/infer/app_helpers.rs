//! Common application helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn auto_borrow_call_arg_types(
    func_ty: &Type,
    arg_tys: Vec<Type>,
    subst: &Subst,
) -> Vec<Type> {
    let Type::Fn(params, _) = subst.apply(func_ty) else {
        return arg_tys;
    };
    arg_tys
        .into_iter()
        .enumerate()
        .map(
            |(index, actual)| match params.get(index).map(|param| subst.apply(param)) {
                Some(Type::Ref(_)) if !matches!(subst.apply(&actual), Type::Ref(_)) => {
                    Type::Ref(Box::new(actual))
                }
                _ => actual,
            },
        )
        .collect()
}

/// Apply the checked function contract shared by ordinary and specialized
/// application paths.
///
/// Procedural routes may compute a more precise result than their fallback HM
/// scheme, but they must still consume the scheme's argument restrictions.
/// Keeping unification and family-failure cleanup here prevents such a route
/// from becoming a second admission mechanism.
pub(super) fn unify_checked_call_contract(
    list: &deep::List,
    func_ty: &Type,
    arg_tys: &[Type],
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Result<Type, Type> {
    let ret_ty = vg.fresh_type();
    let unify_arg_tys = auto_borrow_call_arg_types(func_ty, arg_tys.to_vec(), subst);
    let expected_fn = Type::Fn(unify_arg_tys.clone(), Box::new(ret_ty.clone()));

    match unify(func_ty, &expected_fn, subst) {
        Ok(()) => Ok(ret_ty),
        Err(te) => {
            let family_mismatch = matches!(te.kind, TypeErrorKind::DtypeFamilyMismatch);
            if family_mismatch
                && let Some(rejected) = product.report_preceding_shape_error_for_failed_family_call(
                    func_ty,
                    &unify_arg_tys,
                    vg,
                    subst,
                    errors,
                )
            {
                if let Type::Error(witness) = rejected {
                    product.cancel_shape_checks_for_failed_family_call(func_ty, subst, witness);
                }
                return Err(rejected);
            }
            let mut error: CheckError = te.into();
            if let Some(id) = list_span_id(list) {
                error.span_offset = parse_span_offset(id);
                error.span_id = Some(id.to_string());
            } else {
                let offset = span_of_list(list).offset;
                if offset > 0 {
                    error.span_offset = Some(offset);
                }
            }
            let rejected = report(errors, error);
            if family_mismatch && let Type::Error(witness) = rejected {
                product.cancel_shape_checks_for_failed_family_call(func_ty, subst, witness);
            }
            Err(rejected)
        }
    }
}

/// Implicit-copy fan-out v3 Shape A relaxation: when a def's body's
/// tail-position expression is a bare `(var x)` reference (possibly
/// wrapped in `let`, `if`, or `match` structures whose sibling branches
/// all return the same name) and the body's inferred return type is
/// `Ref(R)` while the declared return is owned `R`, return a relaxed
/// declared type `Fn(params, Ref(R))` so the def-body unify can succeed.
/// Returns `None` for any other body shape; the caller surfaces the
/// existing TypeMismatch in that case.
///
/// The PR #91 (W4-A) version of this helper accepted only a bare
/// `(fn (params...) (var x))` body.  0.7.9 broadens the gate to walk
/// `let`/`if`/`match` tail-position structures via
/// `descend_to_tail_var`, closing `Linearity-ShapeABroadReturn-F1`.
pub(super) fn shape_a_relaxed_return(
    body_expr: &deep::Expr,
    body_ty: &Type,
    decl_ty: &Type,
) -> Option<Type> {
    // body is the def's body, which the desugarer wraps as
    // `(fn (params ...) body_inner)` whenever the def has params.  Walk
    // the inner expression's tail position to confirm it resolves to a
    // bare `(var name)` reference across every reachable sibling.
    let body_children = match body_expr.carrier() {
        deep::ExprCarrier::DecodedNode(DeepTag::Fn, _, children) => children,
        deep::ExprCarrier::DecodedNode(_, _, _)
        | deep::ExprCarrier::StructuralList(_)
        | deep::ExprCarrier::UndecodableHead(_, _, _)
        | deep::ExprCarrier::Atom(_)
        | deep::ExprCarrier::MetadataMap(_)
        | deep::ExprCarrier::MetadataExpression(_)
        | deep::ExprCarrier::MalformedLegacyList(_) => return None,
    };
    let inner = body_children.get(1)?;
    descend_to_tail_var(inner)?;

    // The body's inferred type and the declared type both must be
    // `Fn(params, ret)` with matching params and a return-position
    // mismatch of exactly `Ref(R)` (body) vs `R` (decl).
    let (Type::Fn(body_params, body_ret), Type::Fn(decl_params, decl_ret)) = (body_ty, decl_ty)
    else {
        return None;
    };
    if body_params.len() != decl_params.len() {
        return None;
    }
    let Type::Ref(body_inner_ret) = body_ret.as_ref() else {
        return None;
    };
    // The body's unwrapped return must match the declared return
    // structurally before we allow the relaxation; otherwise the
    // relaxed unify would still fail and only the loop would change.
    if !types_structurally_equal(body_inner_ret.as_ref(), decl_ret.as_ref()) {
        return None;
    }
    Some(Type::Fn(
        decl_params.clone(),
        Box::new(Type::Ref(Box::new(decl_ret.as_ref().clone()))),
    ))
}

/// Descend through `let`, `if`, and `match` to a tail-position
/// `(var name)` reference.  Returns `Some(name)` when every sibling
/// branch resolves to the same bare-var name, `None` otherwise.
///
/// This is the broader-Shape-A coverage closure for
/// `Linearity-ShapeABroadReturn-F1`.  The rules:
///
/// - `(var x)` returns `Some("x")` (the leaf case from PR #91).
/// - `(let bind body)` recurses into `body` (the second child).
/// - `(if cond then_e else_e)` recurses into both branches; both must
///   resolve to the same name.
/// - `(match scrutinee arm ...)` recurses into every arm body (the
///   third child of each `(arm pattern guard body)` triple); all arms
///   must resolve to the same name.
/// - Otherwise returns `None`.
///
/// The descent is type-agnostic; the surrounding logic in
/// `shape_a_relaxed_return` already verifies that the body's inferred
/// return type is `Ref(R)` and the declared return is `R` structurally.
///
/// The "same name across siblings" requirement is intentional: the
/// existing relaxation is justified by the caller's borrow lifetime
/// already covering the parameter being returned.  Heterogeneous
/// bare-var returns would extend the relaxation beyond v3 scope and
/// need a richer coercion story.
pub(super) fn descend_to_tail_var(expr: &deep::Expr) -> Option<&str> {
    stack_guard!("descend_to_tail_var", expr, None);
    let (tag, children) = match expr.carrier() {
        deep::ExprCarrier::DecodedNode(tag, _, children) => (tag, children),
        deep::ExprCarrier::StructuralList(_)
        | deep::ExprCarrier::UndecodableHead(_, _, _)
        | deep::ExprCarrier::Atom(_)
        | deep::ExprCarrier::MetadataMap(_)
        | deep::ExprCarrier::MetadataExpression(_)
        | deep::ExprCarrier::MalformedLegacyList(_) => return None,
    };
    match tag {
        DeepTag::Var => children.first().and_then(symbol_name),
        DeepTag::Let => {
            let body = children.get(1)?;
            descend_to_tail_var(body)
        }
        DeepTag::If => {
            let then_e = children.get(1)?;
            let else_e = children.get(2)?;
            let then_name = descend_to_tail_var(then_e)?;
            let else_name = descend_to_tail_var(else_e)?;
            if then_name == else_name {
                Some(then_name)
            } else {
                None
            }
        }
        DeepTag::Match => {
            // Skip the scrutinee (first child); every remaining child is
            // expected to be an `(arm pattern guard body)` triple.
            let arms = children.get(1..)?;
            if arms.is_empty() {
                return None;
            }
            let mut name: Option<&str> = None;
            for arm in arms {
                let arm_children = match arm.carrier() {
                    deep::ExprCarrier::DecodedNode(DeepTag::Arm, _, children) => children,
                    deep::ExprCarrier::DecodedNode(_, _, _)
                    | deep::ExprCarrier::StructuralList(_)
                    | deep::ExprCarrier::UndecodableHead(_, _, _)
                    | deep::ExprCarrier::Atom(_)
                    | deep::ExprCarrier::MetadataMap(_)
                    | deep::ExprCarrier::MetadataExpression(_)
                    | deep::ExprCarrier::MalformedLegacyList(_) => return None,
                };
                let arm_body = arm_children.get(2)?;
                let arm_name = descend_to_tail_var(arm_body)?;
                match name {
                    None => name = Some(arm_name),
                    Some(prev) if prev == arm_name => {}
                    Some(_) => return None,
                }
            }
            name
        }
        _ => None,
    }
}

/// Two tensor dimensions are *identical* for structural-equality
/// purposes iff they denote the same dimension: same concrete
/// `Dim::Name`, same `Dim::Lit`, or the same `Dim::Var`.  `Dim::Wildcard`
/// matches anything on either side (it is the permissive "unknown"
/// sentinel, consistent with `unify_dim`).
///
/// Two *distinct* symbolic dim variables (`n` vs `m`) are NOT identical
/// even though both contribute rank 1.  This is the soundness fix for
/// `TypeCheck-FreeDimVarUnification-F1` (SR-LEAK-A): the Shape A
/// relaxed-retry must not accept a body whose return dim diverges from
/// the declared return dim.
///
/// Name <-> Lit (issue Chelis-Lang/chelis#219, Option A): mirrors
/// `unify_dim`'s permissive Name <-> Lit arm. The Shape A relaxed-
/// retry's structural check must agree with `unify_dim` so the
/// retry path doesn't silently reject a callee-shape pairing that
/// the call-site unification would accept.
pub(super) fn dims_identical(d1: &Dim, d2: &Dim) -> bool {
    match (d1, d2) {
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => true,
        (Dim::Name(n1), Dim::Name(n2)) => n1 == n2,
        (Dim::Lit(l1), Dim::Lit(l2)) => l1 == l2,
        (Dim::Var(v1), Dim::Var(v2)) => v1 == v2,
        // Issue #219 Option A: Name and Lit count as identical for
        // the Shape A relaxed-retry's structural check.
        (Dim::Name(_), Dim::Lit(_)) | (Dim::Lit(_), Dim::Name(_)) => true,
        _ => false,
    }
}

/// `where`/`clamp` require identical shapes, unlike permissive dimension
/// unification or scatter containment. Names do not equal bare literals;
/// known contradictions take precedence over a matching semantic label.
pub(super) fn elementwise_shapes_match(a: &[Dim], b: &[Dim], subst: &Subst) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            let a = subst.observe_dim(a);
            let b = subst.observe_dim(b);
            if matches!((a.known_extent(), b.known_extent()), (Some(x), Some(y)) if x != y) {
                return false;
            }
            if a.name().is_some() || b.name().is_some() {
                return a.name() == b.name();
            }
            (a.known_extent().is_some() && a.known_extent() == b.known_extent())
                || (a.variable().is_some() && a.variable() == b.variable())
                || (a.rank().is_some() && a.rank() == b.rank())
                || (a.is_wildcard() && b.is_wildcard())
        })
}

/// Structural type equality used by `shape_a_relaxed_return` to guard
/// the relaxed retry: the relaxation is only safe when the body and
/// declared return differ exactly by a top-level `Ref` wrapper, so the
/// dim structure underneath must match by *identity*, not just by rank.
///
/// Tensor dims are compared with `dims_identical`: same concrete name,
/// same literal, or the same dim variable.  Distinct dim variables do
/// not match (`TypeCheck-FreeDimVarUnification-F1`).
pub(super) fn types_structurally_equal(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Unit, Type::Unit) => true,
        (Type::Prim(p1), Type::Prim(p2)) => p1 == p2,
        (Type::Ref(i1), Type::Ref(i2)) => types_structurally_equal(i1, i2),
        (Type::Tensor(d1, p1), Type::Tensor(d2, p2)) => {
            p1 == p2
                && d1.len() == d2.len()
                && d1.iter().zip(d2.iter()).all(|(x, y)| dims_identical(x, y))
        }
        (Type::Tuple(es1), Type::Tuple(es2)) => {
            es1.len() == es2.len()
                && es1
                    .iter()
                    .zip(es2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
        }
        (Type::Adt(n1, a1), Type::Adt(n2, a2)) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
        }
        (Type::KindedAdt(n1, a1), Type::KindedAdt(n2, a2)) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1.iter().zip(a2).all(|(left, right)| match (left, right) {
                    (NominalArg::Type(left), NominalArg::Type(right)) => {
                        types_structurally_equal(left, right)
                    }
                    (NominalArg::Dimension(left), NominalArg::Dimension(right)) => {
                        dims_identical(left, right)
                    }
                    _ => false,
                })
        }
        (Type::Fn(p1, r1), Type::Fn(p2, r2)) => {
            p1.len() == p2.len()
                && p1
                    .iter()
                    .zip(p2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
                && types_structurally_equal(r1, r2)
        }
        (Type::Var(_), Type::Var(_)) => true,
        (Type::Error(_), _) | (_, Type::Error(_)) => true,
        _ => false,
    }
}

pub(super) fn type_for_readonly_check(ty: &Type, subst: &Subst) -> Type {
    match subst.apply(ty) {
        Type::Ref(inner) => subst.apply(&inner),
        other => other,
    }
}

#[cfg(test)]
mod dimension_observation_tests {
    use super::*;

    fn labelled(subst: &mut Subst, id: u32, name: &str, extent: Option<i64>) -> Dim {
        let dim = Dim::Var(DimVar(id));
        subst.protect_dimensions([DimVar(id)]);
        unify_dim(&dim, &Dim::Name(name.into()), subst).unwrap();
        if let Some(extent) = extent {
            unify_dim(&dim, &Dim::Lit(extent), subst).unwrap();
        }
        dim
    }

    #[test]
    fn observations_keep_constraint_label_and_identity_policies_separate() {
        let mut subst = Subst::new();
        let a = labelled(&mut subst, 0, "row", Some(2));
        let b = labelled(&mut subst, 1, "row", Some(3));
        let c = labelled(&mut subst, 2, "other", Some(2));
        let symbolic = labelled(&mut subst, 3, "row", None);
        let alias = Dim::Var(DimVar(4));
        unify_dim(&alias, &symbolic, &mut subst).unwrap();
        for (x, y) in [(&a, &b), (&b, &a), (&a, &c), (&c, &a), (&a, &Dim::Lit(2))] {
            assert!(!elementwise_shapes_match(
                std::slice::from_ref(x),
                std::slice::from_ref(y),
                &subst
            ));
        }
        assert!(elementwise_shapes_match(
            std::slice::from_ref(&a),
            std::slice::from_ref(&symbolic),
            &subst
        ));
        assert_eq!(
            subst.observe_dim(&alias).variable(),
            subst.observe_dim(&symbolic).variable()
        );
        assert!(subst.observe_dim(&alias).is_protected());
        assert_eq!(subst.observe_dim(&a).known_extent(), Some(2));
        assert_eq!(subst.observe_dim(&a).literal_extent(), None);
        assert_eq!(
            subst.static_dim_products_match(std::slice::from_ref(&a), &[Dim::Lit(3)]),
            None
        );
        assert_eq!(select_diagonal_extent(&a, &b, &subst).dim, Dim::Lit(2));
        assert_eq!(select_diagonal_extent(&b, &a, &subst).dim, Dim::Lit(2));
        assert_eq!(
            select_diagonal_extent(&symbolic, &alias, &subst).dim,
            symbolic
        );
        // Shape A still compares authored identity, not equal labels.
        assert!(!dims_identical(&a, &symbolic));
        assert!(dims_identical(&a, &a));
        let serialized = bincode::serialize(&subst).unwrap();
        let decoded: Subst = bincode::deserialize(&serialized).unwrap();
        assert_eq!(decoded.observe_dim(&a).known_extent(), Some(2));
        assert_eq!(decoded.observe_dim(&alias).name(), Some("row"));
        assert!(!decoded.observe_dim(&alias).is_protected());
    }

    /// This proves only the inference formula seam. The original labelled
    /// source controls still encounter the pre-existing concrete-metadata
    /// admission restriction in validate.rs; do not claim source conv coverage.
    #[test]
    fn conv_formula_reads_known_constraints_without_inventing_unknown_extents() {
        let source =
            chelis_surf::parser::parse_str("def f() = conv(x,k,[1i64],[(0i64,0i64)])").unwrap();
        let program = chelis_surf::desugar::desugar_program(&source);
        fn conv_args(expr: &deep::Expr) -> Option<&[deep::Expr]> {
            let (tag, _, kids) = stamped_parts(expr)?;
            if tag == DeepTag::App
                && kids
                    .first()
                    .and_then(|e| stamped_parts(e))
                    .is_some_and(|(tag, _, xs)| {
                        tag == DeepTag::Var && xs.first().and_then(symbol_name) == Some("conv")
                    })
            {
                Some(&kids[1..])
            } else {
                kids.iter().find_map(conv_args)
            }
        }
        let args = program.iter().find_map(conv_args).unwrap();
        let mut subst = Subst::new();
        let known = labelled(&mut subst, 0, "width", Some(5));
        let unknown = labelled(&mut subst, 1, "width", None);
        let kernel = [Dim::Lit(3), Dim::Lit(2), Dim::Lit(3)];
        assert_eq!(
            compute_concrete_conv_spatial(
                args,
                &[Dim::Lit(1), Dim::Lit(2), known],
                &kernel,
                &subst
            ),
            Some(vec![3])
        );
        assert_eq!(
            compute_concrete_conv_spatial(
                args,
                &[Dim::Lit(1), Dim::Lit(2), unknown],
                &kernel,
                &subst
            ),
            None
        );
        assert_eq!(
            compute_concrete_conv_spatial(
                args,
                &[Dim::Lit(1), Dim::Lit(2), Dim::Lit(2)],
                &kernel,
                &subst
            ),
            None
        );
    }
}
