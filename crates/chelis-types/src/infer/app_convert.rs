//! List and tensor conversion rules.
//!
//! One responsibility: the post-unification rules for `to_tensor`, `to_list`,
//! `pad_sequences` and `pad_sequences_to`, which move values between nested
//! `List`s and tensors.

use super::*;

/// List-to-tensor and tensor-to-list conversion rules after generic
/// application unification: `to_tensor`, `to_list`, `pad_sequences` and
/// `pad_sequences_to`. [`finish_unified_app`] calls this for any callee its
/// own dispatch does not name; `None` means `fname` is not one of these
/// operations, or its rule had nothing to decide, and the dispatcher
/// continues exactly as when the arms were inline.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_conversion_app(
    fname: &str,
    source_site: CheckSite<'_>,
    node: &DeepNode,
    kids: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    site: &UnresolvedOperandSite<'_>,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Option<Type> {
    match fname {
        "to_tensor" => {
            if kids
                .get(1)
                .is_some_and(|arg| static_to_tensor_shape_status(arg).is_err())
            {
                let mut error = CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_node_provenance(node, "to_tensor requires rectangular child shapes; statically inconsistent literal dimensions ([05-OP-57])".to_string()),
                    vec![],
                );
                if let Some(id) = node_span_id(node) {
                    error.span_offset = parse_span_offset(id);
                    error.span_id = Some(id.to_string());
                }
                return Some(report(errors, error));
            }
            if let Some(first_arg) = arg_tys.first() {
                // Bucket 4b: support arbitrarily-nested numeric/bool
                // lists. Each enclosing `List` adds one outer
                // dimension, and the innermost element type must
                // be a numeric or bool primitive.
                //
                // Issue Chelis-Lang/chelis#218 (R2 HIGH-A from
                // PR #211): when the argument is a statically-
                // resolvable Cons-chain literal, emit concrete
                // `Dim::Lit(n)` per axis instead of wildcards.
                // The wildcard fallback only fires when the
                // argument is variable-fed (e.g.
                // `to_tensor(items)`), where the shape is
                // genuinely unknown at type-check time. Emitting
                // concrete dims at this single source point
                // means every downstream consumer (reductions,
                // elementwise activations, anything that reads
                // the to_tensor app's `type:` metadata) sees a
                // sound shape instead of `Dim::Wildcard`.
                let resolved = subst.apply(first_arg);
                if matches!(resolved, Type::Error(_)) {
                    return Some(result_ty.clone());
                }
                if matches!(resolved, Type::Var(_)) {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                match peel_to_tensor_argument(&resolved) {
                    ToTensorPeel::Ok { rank, precision } => {
                        if rank == 0 {
                            // Defensive: a bare scalar should never
                            // hit this branch (the typer requires
                            // a `List` head), but guard anyway.
                            return Some(report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!("to_tensor expects List input, got {resolved}"),
                                    ),
                                    vec![],
                                ),
                            ));
                        }
                        // R2 HIGH-A: try the static-shape walker
                        // on the actual argument expression
                        // first. `kids[0]` is the `(var
                        // to_tensor)` callee; `kids[1]` is the
                        // argument expression. If the walker
                        // can't resolve a uniform shape
                        // (variable-fed argument, ragged
                        // literal, or unrecognized leaf), fall
                        // back to the legacy wildcard rank.
                        let dims = kids
                            .get(1)
                            .and_then(|arg| static_to_tensor_shape(arg, rank))
                            .unwrap_or_else(|| vec![Dim::Wildcard; rank]);
                        return Some(Type::Tensor(dims, TensorPrec::Concrete(precision)));
                    }
                    // [05-OP-57]: the leaf of `to_tensor`'s List is one
                    // scalar tensor-element dtype. A leaf variable restricted
                    // to the `Float`, `Int` or `Numeric` family can only be
                    // such a scalar, so the nesting depth already fixes the
                    // rank and the variable is the result precision, exactly
                    // as `to_list` maps `tensor[n, p]` to `List[p]`. Returning
                    // the scheme's opaque result here left every consumer of
                    // a generic `to_tensor` (a `reshape`, say) suspended past
                    // its declaration boundary.
                    // Any dtype bound of spec/04 §5.9 resolves the element,
                    // including the explicit set form: a set restricts WHICH
                    // dtypes are admissible, never which programs type-check
                    // (chelis#2443). Listing only the three families left a
                    // set-bounded `to_tensor` suspended, so `[p: {f32, f64}]`
                    // was rejected where the equivalent `Float` was accepted.
                    ToTensorPeel::Pending { rank, element }
                        if rank > 0
                            && subst
                                .tvar_restriction(element)
                                .is_some_and(|restriction| !restriction.is_value_constraint()) =>
                    {
                        let dims = kids
                            .get(1)
                            .and_then(|arg| static_to_tensor_shape(arg, rank))
                            .unwrap_or_else(|| vec![Dim::Wildcard; rank]);
                        return Some(Type::Tensor(dims, TensorPrec::Var(element)));
                    }
                    // chelis#2523: `to_tensor` maps `List` nested `r`
                    // deep around a numeric or bool dtype `p` to a rank-`r`
                    // tensor of `p`, so an element type that is still a
                    // variable is determined by a result whose rank and
                    // precision are known, as a declared field does for
                    // `IntCol(to_tensor([]))`. Otherwise the call waits for
                    // either one to bind, and the declaration boundary
                    // decides it if neither does.
                    ToTensorPeel::Pending { rank, element } => {
                        if let Type::Tensor(result_dims, precision) = &result_ty
                            && result_dims.len() >= rank
                        {
                            let scalar = match precision {
                                TensorPrec::Concrete(prim) => Type::Prim(*prim),
                                TensorPrec::Var(var) => Type::Var(*var),
                            };
                            let inner = (rank..result_dims.len()).fold(scalar, |inner, _| {
                                Type::Adt("List".to_string(), vec![inner])
                            });
                            if let Err(te) = unify(&Type::Var(element), &inner, subst) {
                                return Some(report(errors, te.into()));
                            }
                            // A concrete precision settles the call; a
                            // precision variable is decided when it binds,
                            // or at its instantiations.
                            if let TensorPrec::Concrete(prim) = precision
                                && (prim.is_numeric() || *prim == Prim::Bool)
                            {
                                return Some(result_ty.clone());
                            }
                        }
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    ToTensorPeel::Poisoned => return Some(result_ty.clone()),
                    ToTensorPeel::BadInner(inner) => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!(
                                        "to_tensor expects numeric or bool elements at the innermost level, got {inner}"
                                    ),
                                ),
                                vec![],
                            ),
                        ));
                    }
                    ToTensorPeel::NotList => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("to_tensor expects List input, got {resolved}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "to_list" => {
            if let Some(first_arg) = arg_tys.first() {
                match type_for_readonly_check(first_arg, subst) {
                    Type::Tensor(dims, precision) => {
                        if dims.len() != 1 {
                            return Some(report_at_check_site(
                                errors,
                                CheckError::with_types(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "to_list argument 1: expected rank-1 tensor, got rank-{} tensor",
                                            dims.len()
                                        ),
                                    ),
                                    "rank-1 tensor".to_string(),
                                    format!("rank-{} tensor", dims.len()),
                                    vec![],
                                ),
                                site.source_site,
                            ));
                        }
                        let precision = match precision {
                            TensorPrec::Concrete(p) => p,
                            // [05-OP-57] fixes the result constructor and
                            // leaf relation independently of whether the
                            // authored precision has been instantiated:
                            // tensor[n, p] becomes List[p]. Returning the
                            // builtin scheme's opaque result variable here
                            // erased the known List constructor and made a
                            // following index look unresolved at [04-INF-9]'s
                            // declaration boundary (chelis#2126).
                            TensorPrec::Var(variable) => {
                                return Some(Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Var(variable)],
                                ));
                            }
                        };
                        if !precision.is_numeric() && !matches!(precision, Prim::Bool) {
                            return Some(report_at_check_site(
                                errors,
                                CheckError::with_types(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "to_list argument 1: expected numeric or bool tensor, got tensor[{}, {}]",
                                            dims[0],
                                            precision.name()
                                        ),
                                    ),
                                    "numeric or bool tensor".to_string(),
                                    format!("tensor[{}, {}]", dims[0], precision.name()),
                                    vec![],
                                ),
                                site.source_site,
                            ));
                        }
                        return Some(Type::Adt("List".to_string(), vec![Type::Prim(precision)]));
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report_at_check_site(
                            errors,
                            CheckError::with_types(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("to_list argument 1: expected tensor, got {other}"),
                                ),
                                "tensor".to_string(),
                                other.to_string(),
                                vec![],
                            ),
                            site.source_site,
                        ));
                    }
                }
            }
        }
        "pad_sequences" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            let seqs_ty = subst.apply(&arg_tys[0]);
            let pad_ty = subst.apply(&arg_tys[1]);
            match seqs_ty {
                Type::Adt(outer_name, outer_args)
                    if outer_name == "List" && outer_args.len() == 1 =>
                {
                    match &outer_args[0] {
                        Type::Adt(inner_name, inner_args)
                            if inner_name == "List" && inner_args.len() == 1 =>
                        {
                            if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                return Some(report(errors, te.into()));
                            }
                            match subst.apply(&inner_args[0]) {
                                Type::Prim(precision) if precision.is_data_element_dtype() => {
                                    return Some(Type::Tensor(
                                        vec![Dim::Wildcard, Dim::Wildcard],
                                        TensorPrec::Concrete(precision),
                                    ));
                                }
                                Type::Error(_) => return Some(result_ty.clone()),
                                Type::Var(_) => {
                                    return Some(site.defer(
                                        arg_tys,
                                        result_ty,
                                        product,
                                        result_ty.clone(),
                                    ));
                                }
                                other => {
                                    return Some(report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_node_provenance(
                                                node,
                                                format!(
                                                    "pad_sequences expects nested lists of a data element dtype, got {other}"
                                                ),
                                            ),
                                            vec![],
                                        ),
                                    ));
                                }
                            }
                        }
                        other => {
                            return Some(report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "pad_sequences expects List[List[T]], got List[{other}]"
                                        ),
                                    ),
                                    vec![],
                                ),
                            ));
                        }
                    }
                }
                Type::Error(_) => return Some(result_ty.clone()),
                Type::Var(_) => {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                other => {
                    return Some(report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!("pad_sequences expects List[List[T]] input, got {other}"),
                            ),
                            vec![],
                        ),
                    ));
                }
            }
        }
        "pad_sequences_to" => {
            if arg_tys.len() != 3 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    3,
                    arg_tys.len(),
                ));
            }
            let seqs_ty = subst.apply(&arg_tys[0]);
            let width_ty = subst.apply(&arg_tys[1]);
            let pad_ty = subst.apply(&arg_tys[2]);
            if let Err(te) = unify(&width_ty, &Type::Prim(Prim::Int64), subst) {
                return Some(report(errors, te.into()));
            }
            // The padded (axis-1) dimension equals the `width`
            // argument. When `width` is a literal — including
            // `cast(N, i64)`, the form every caller uses —
            // propagate `Dim::Lit(N)` so the padded width is a
            // concrete dim that participates in shape checking.
            // A non-literal or non-positive width stays
            // `Dim::Wildcard` (the runtime validates the value).
            // `extract_int_for_dim` (not `extract_int_literal`)
            // is the cast-aware extractor used for dim contexts.
            let width_dim = node
                .children_slice()
                .get(2)
                .and_then(extract_int_for_dim)
                .filter(|width| *width > 0)
                .map_or(Dim::Wildcard, Dim::Lit);
            match seqs_ty {
                Type::Adt(outer_name, outer_args)
                    if outer_name == "List" && outer_args.len() == 1 =>
                {
                    match &outer_args[0] {
                        Type::Adt(inner_name, inner_args)
                            if inner_name == "List" && inner_args.len() == 1 =>
                        {
                            if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                return Some(report(errors, te.into()));
                            }
                            match subst.apply(&inner_args[0]) {
                                Type::Prim(precision) if precision.is_data_element_dtype() => {
                                    return Some(Type::Tensor(
                                        vec![Dim::Wildcard, width_dim],
                                        TensorPrec::Concrete(precision),
                                    ));
                                }
                                Type::Error(_) => return Some(result_ty.clone()),
                                Type::Var(_) => {
                                    return Some(site.defer(
                                        arg_tys,
                                        result_ty,
                                        product,
                                        result_ty.clone(),
                                    ));
                                }
                                other => {
                                    return Some(report(
                                        errors,
                                        CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_node_provenance(
                                                node,
                                                format!(
                                                    "pad_sequences_to expects nested lists of a data element dtype, got {other}"
                                                ),
                                            ),
                                            vec![],
                                        ),
                                    ));
                                }
                            }
                        }
                        other => {
                            return Some(report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_node_provenance(
                                        node,
                                        format!(
                                            "pad_sequences_to expects List[List[T]], got List[{other}]"
                                        ),
                                    ),
                                    vec![],
                                ),
                            ));
                        }
                    }
                }
                Type::Error(_) => return Some(result_ty.clone()),
                Type::Var(_) => {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                other => {
                    return Some(report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "pad_sequences_to expects List[List[T]] input, got {other}"
                                ),
                            ),
                            vec![],
                        ),
                    ));
                }
            }
        }
        _ => {}
    }
    None
}

/// spec/05 [05-OP-57] and spec/03 §6.4: the dtype child of
/// `to_tensor(xs, T)`, a type node in the call's third position. The call
/// checks as `to_tensor(xs)` whose List leaf dtype is `T`.
pub(super) fn to_tensor_dtype_child<'a>(
    kids: &'a [deep::Expr],
    env: &Env,
) -> Option<&'a deep::Expr> {
    let [callee, _, dtype] = kids else {
        return None;
    };
    let (DeepTag::Var, _, parts) = stamped_parts(callee)? else {
        return None;
    };
    if parts.first().and_then(symbol_name) != Some("to_tensor")
        || env.is_lexically_bound("to_tensor")
    {
        return None;
    }
    matches!(
        stamped_parts(dtype),
        Some((DeepTag::TPrim | DeepTag::TVar, _, _))
    )
    .then_some(dtype)
}

/// spec/03 §6.4: Deep has no dtype-stating constructs, so a numeric literal
/// element of a `to_tensor` argument with no `type` metadata has no dtype.
/// Returns the first such element of a literal `Cons` chain, recursively
/// through nested chains.
pub(super) fn untyped_to_tensor_element(argument: &deep::Expr) -> Option<&deep::Expr> {
    stack_guard!("untyped_to_tensor_element", argument, None);
    let mut tail = argument;
    loop {
        let (DeepTag::App, _, parts) = stamped_parts(tail)? else {
            return None;
        };
        let [cons, head, rest] = parts else {
            return None;
        };
        let (DeepTag::Var, _, callee) = stamped_parts(cons)? else {
            return None;
        };
        if callee.first().and_then(symbol_name) != Some("Cons") {
            return None;
        }
        match stamped_parts(head) {
            Some((
                DeepTag::Lit,
                meta,
                [deep::Expr::Atom(deep::Atom::Int(_) | deep::Atom::Float(_), _)],
            )) if meta.ty().is_none() => {
                return Some(head);
            }
            Some((DeepTag::App, _, _)) => {
                if let Some(element) = untyped_to_tensor_element(head) {
                    return Some(element);
                }
            }
            _ => {}
        }
        tail = rest;
    }
}

/// The diagnostic for an untyped Deep literal element of `to_tensor`.
pub(super) fn untyped_to_tensor_element_error(element: &deep::Expr) -> CheckError {
    let error = CheckError::new(
        CheckErrorKind::TypeMismatch,
        "Deep literal element of `to_tensor` has no `type` metadata; tensor elements have no \
         default dtype (spec/03-deep-syntax.md §6.4)"
            .to_string(),
        vec!["give the literal a `type`, such as `(lit {type: (t-prim {} f32)} 1.5)`".to_string()],
    );
    match TypeDiagnosticLocation::from_expr(element) {
        Some(location) => location.attach(error),
        None => error,
    }
}

/// [05-OP-57]: a written `T` states the leaf dtype, which the List's leaf
/// SHALL equal; `to_tensor` never converts. The leaf of a List whose element
/// type is still open takes `T`, which is how an empty List gets its dtype.
#[allow(clippy::too_many_arguments)]
pub(super) fn require_to_tensor_leaf(
    expr: &deep::Expr,
    dtype_child: &deep::Expr,
    argument: &Type,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> Result<(), Type> {
    let dtype = {
        let mut resolver = DeepTypeResolver::new(
            TypeUseSite::Annotation,
            annotation_binder_mode(env),
            adt_reg.resolution_env(),
            vg,
            errors,
        )
        .with_diagnostic_owner(expr);
        match resolver.resolve(dtype_child) {
            Ok(dtype) => dtype.into_type(),
            Err(witness) => return Err(propagate(&witness)),
        }
    };
    // A `t-var` child names a dtype binder; a `t-prim` child must name a
    // tensor-element primitive.
    let binder = matches!(stamped_parts(dtype_child), Some((DeepTag::TVar, _, _)));
    let is_dtype =
        binder || matches!(&dtype, Type::Prim(prim) if prim.is_numeric() || *prim == Prim::Bool);
    let at_expr = |error: CheckError| match TypeDiagnosticLocation::from_expr(expr) {
        Some(location) => location.attach(error),
        None => error,
    };
    if !is_dtype {
        return Err(report(
            errors,
            at_expr(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "to_tensor's dtype argument must be a tensor-element dtype or a dtype \
                     binder, not {dtype} ([05-OP-57])"
                ),
                vec![],
            )),
        ));
    }
    let mut leaf = subst.apply(argument);
    let mut depth = 0usize;
    while let Type::Adt(name, arguments) = &leaf
        && name == "List"
        && let [element] = arguments.as_slice()
    {
        leaf = subst.apply(element);
        depth += 1;
    }
    // A non-List argument is the conversion rule's to reject or defer.
    if depth == 0 {
        return Ok(());
    }
    // A dtype binder ranges over every member of its bound, so a concrete
    // leaf never equals it; unifying would narrow the binder instead.
    let binder_against_concrete = binder && matches!(leaf, Type::Prim(_));
    if binder_against_concrete || unify(&leaf, &dtype, subst).is_err() {
        let leaf = subst.apply(&leaf);
        let dtype = subst.apply(&dtype);
        return Err(report(
            errors,
            at_expr(CheckError::with_types(
                CheckErrorKind::TypeMismatch,
                format!(
                    "to_tensor's dtype argument states {dtype}, but the List's leaf dtype is \
                     {leaf}, and to_tensor never converts ([05-OP-57]); write \
                     to_tensor(xs, {leaf}), or convert the result with \
                     cast(to_tensor(xs, {leaf}), {dtype})"
                ),
                dtype.to_string(),
                leaf.to_string(),
                vec![],
            )),
        ));
    }
    Ok(())
}
