//! Shape operation inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// chelis#339 Part 2: infer a variadic named-axis reduction
/// `sum(x, seq, head)` (spec/04-type-system.md §4.5.3). The reduction HM
/// schemes are arity-2, so the 3+-arg form bypasses the generic arity
/// check (the `infer_permute_app` pattern); the existing
/// `check_reduction_signature` named loop validates every axis and
/// computes the symbolic output. The variadic form is defined for the
/// value reductions only — `argmax_reduce`/`argmin_reduce` produce
/// indices along ONE axis, which a second reduction cannot compose, so
/// they are rejected here with a targeted diagnostic.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_reduction_app(
    list: &deep::List,
    fname: &str,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    if fname == "argmax_reduce" || fname == "argmin_reduce" {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                format!(
                    "{fname} is an index-returning reduction and has no variadic \
                 named-axis form: an index along one axis is not composable with a \
                 second reduction (spec/04-type-system.md \u{00a7}4.5.3). Reduce one \
                 axis at a time."
                ),
                vec![],
            ),
        );
    }

    let kids = children(list);
    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // Axis slots carry dimension names, typed as axes (`int32`)
            // rather than inferred as values — the named-reduction exemption
            // from the generic path.
            if index >= 1 && symbolic_dim_ref_name(arg).is_some() {
                Type::Prim(Prim::Int32)
            } else {
                infer_expr(arg, env, vg, subst, adt_reg, errors, product)
            }
        })
        .collect();
    if let Some(err) = propagate_if_error(&arg_tys) {
        return err;
    }

    // [05-DIM-3] / [05-OP-29]: this early variadic route bypasses the
    // generic application's registered axis-dtype gate. Apply the same
    // registry here so every Count axis is int32, including concrete
    // multi-axis calls whose constant values are otherwise extractable.
    if let Err(rejected) = enforce_registered_axis_dtypes(fname, &arg_tys, list, errors) {
        return rejected;
    }

    let result_ty = Type::Var(vg.fresh_tvar());
    check_reduction_signature(fname, &kids[1..], &arg_tys, &result_ty, subst, errors)
}

/// chelis#339: infer the 4-arg anchored named-axis expand form
/// `expand(x, new, size, anchor)` (spec/04-type-system.md §4.5.3). The
/// builtin scheme is arity-3, so this form bypasses the generic HM arity
/// check (the `infer_permute_app` pattern). The `new` and `anchor` slots
/// carry dimension *names*, not bound values — like a named reduction
/// axis they are typed as `int32` axes rather than inferred, and
/// `check_expand_signature` reads the actual names back from the arg
/// exprs.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_expand_app(
    callee: &'static str,
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() != 5 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                format!(
                    "{callee} expects (tensor, axis, size) or the named-axis form \
                 (tensor, name, size, anchor), got {} arguments",
                    kids.len() - 1
                ),
                vec![],
            ),
        );
    }

    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // The name/size/anchor slots may carry dim names; a name bound in
            // the value environment is a runtime value instead (issue #259
            // scope discrimination, as in the generic-path exemption). A dim
            // name in the size slot is an extent-domain value ([05-DIM-1]),
            // so it types int64; the name/anchor slots are axis-domain.
            if index >= 1
                && symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none())
            {
                if index == 2 {
                    Type::Prim(Prim::Int64)
                } else {
                    Type::Prim(Prim::Int32)
                }
            } else {
                infer_expr(arg, env, vg, subst, adt_reg, errors, product)
            }
        })
        .collect();
    // chelis#530: the size slot (index 2 / `kids[3]`) is exempt from the
    // error-propagation short-circuit so an inline tuple-get / `match`/`if`
    // size that infers to `Type::Error` still reaches the named-axis literal
    // check below (`check_named_expand_signature` reads the size from the raw
    // AST), rather than being silently accepted. Mirrors the generic-path
    // exemption in `infer_app`.
    if let Some(err) = propagate_if_error(
        arg_tys
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != 2)
            .map(|(_, ty)| ty),
    ) {
        return err;
    }
    // The size slot must be an int64 (a literal, a symbolic dim, or a
    // runtime int64 expression; extent-domain under [05-DIM-1], §4.7.2);
    // a non-int64 size is a type error the arity-3 scheme would otherwise
    // have caught.
    let size_ty = subst.apply(&arg_tys[2]);
    match size_ty {
        Type::Prim(Prim::Int64) | Type::Var(_) | Type::Error(_) => {}
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "{callee} expects an int64 size (write Ni64 or cast(N, int64)), got {other}"
                        ),
                    ),
                    vec![],
                ),
            );
        }
    }

    let axis_is_dim_name = kids.get(2).is_some_and(|arg| {
        symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none())
    });
    // chelis#397/#469: classify the size slot by PROVENANCE, following
    // `let`/`cast`/arithmetic to a tensor shape source. The positive-rank
    // path would otherwise stamp a sourceless runtime scalar as a `Dim::Name`,
    // type-check clean, and then die at build/eval with the §4.7.2
    // sourceless-size rejection (chelis#469: "no tensor in scope carries it").
    // Rejecting it at CHECK keeps check↔build↔eval in sync (a check-clean
    // program must build); a literal/static/shape-sourced size is materializable
    // and accepted, uniformly across the bare-`var`, `cast`-wrapped, `let`-bound,
    // and arithmetic spellings.
    let size_class = kids
        .get(3)
        .map(|arg| classify_expand_size(arg, env))
        .unwrap_or(SizeClass::Unknown);
    let result_ty = Type::Var(vg.fresh_tvar());
    check_expand_signature(
        callee,
        &kids[1..],
        &arg_tys,
        &result_ty,
        axis_is_dim_name,
        size_class,
        env,
        subst,
        errors,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_permute_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                "permute expects a tensor followed by one or more axis indices".to_string(),
                vec![],
            ),
        );
    }

    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let input_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let axis_tys: Vec<Type> = kids[2..]
        .iter()
        .map(|arg| infer_expr(arg, env, vg, subst, adt_reg, errors, product))
        .collect();

    if let Some(err) = propagate_if_error(std::iter::once(&input_ty).chain(axis_tys.iter())) {
        return err;
    }

    for axis_ty in &axis_tys {
        let resolved = subst.apply(axis_ty);
        match resolved {
            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error(_) => {}
            other => {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!("permute expects int32 axis indices, got {other}"),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }

    let input_ty = type_for_readonly_check(&input_ty, subst);
    let Type::Tensor(dims, prec) = input_ty else {
        if matches!(input_ty, Type::Var(_) | Type::Error(_)) {
            return input_ty;
        }
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("permute expects tensor input, got {input_ty}"),
                vec![],
            ),
        );
    };

    // Uses `extract_int_for_dim` so `cast(N, int32)`-wrapped literal
    // axes reach the OOB-axis check and the unique-axis check at infer
    // time instead of silently falling back to original-dim order (red
    // team round 3 sibling sweep within the spec section 2.4 movement
    // family).
    let Some(axes) = kids[2..]
        .iter()
        .map(extract_int_for_dim)
        .collect::<Option<Vec<_>>>()
    else {
        return Type::Tensor(dims, prec);
    };

    if axes.len() != dims.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                format!(
                    "permute expects {} axis indices for rank {} tensor, got {}",
                    dims.len(),
                    dims.len(),
                    axes.len()
                ),
                vec![],
            ),
        );
    }

    let mut seen = UnordSet::new();
    let mut reordered = Vec::with_capacity(dims.len());
    for axis in axes {
        if axis < 0 || axis as usize >= dims.len() {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "permute axis {axis} is out of bounds for rank {} tensor",
                        dims.len()
                    ),
                    vec![],
                ),
            );
        }
        let axis = axis as usize;
        if !seen.insert(axis) {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("permute axis {axis} appears more than once"),
                    vec![],
                ),
            );
        }
        reordered.push(dims[axis].clone());
    }

    Type::Tensor(reordered, prec)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_reshape_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 || kids.len() > 3 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                "reshape expects a tensor and an optional shape list".to_string(),
                vec![],
            ),
        );
    }

    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let input_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let input_var_name = symbolic_dim_ref_name(&kids[1]).map(|s| s.to_string());
    match type_for_readonly_check(&input_ty, subst) {
        Type::Prim(precision) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(shape_expr, env, vg, subst, adt_reg, errors, product);
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(_te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    // spec/04 §4.7.5: the slot-mismatch diagnostic names the
                    // fix, and its direction states the slot's demand rather
                    // than a unification-order artifact (chelis#916).
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::PrecisionMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "reshape expects an int64 shape list (write i64-suffixed \
                                     elements, e.g. 2i64, or cast(..., int64)), got {}",
                                    subst.apply(&shape_ty)
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
                let dims = reshape_output_dims(shape_expr, input_var_name.as_deref(), &[], subst);
                if let Err(error) = validate_reshape_target_dims(&dims, subst) {
                    return report(errors, error.into());
                }
                return Type::Tensor(dims, TensorPrec::Concrete(precision));
            }

            Type::Tensor(vec![Dim::Wildcard], TensorPrec::Concrete(precision))
        }
        Type::Tensor(input_dims, precision) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(shape_expr, env, vg, subst, adt_reg, errors, product);
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(_te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    // spec/04 §4.7.5: the slot-mismatch diagnostic names the
                    // fix, and its direction states the slot's demand rather
                    // than a unification-order artifact (chelis#916).
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::PrecisionMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "reshape expects an int64 shape list (write i64-suffixed \
                                     elements, e.g. 2i64, or cast(..., int64)), got {}",
                                    subst.apply(&shape_ty)
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
                let dims =
                    reshape_output_dims(shape_expr, input_var_name.as_deref(), &input_dims, subst);
                if let Err(error) = validate_reshape_target_dims(&dims, subst) {
                    return report(errors, error.into());
                }
                if subst.static_dim_products_match(&input_dims, &dims) == Some(false) {
                    let input_numel = subst.static_dim_product(&input_dims);
                    let target_numel = subst.static_dim_product(&dims);
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            match (target_numel, input_numel) {
                                (Some(target), Some(input)) => format!(
                                    "reshape target has {target} elements but input tensor has {input}"
                                ),
                                _ => "reshape target element count does not match input tensor"
                                    .to_string(),
                            },
                            vec![],
                        ),
                    );
                }
                return Type::Tensor(dims, precision);
            }

            Type::Tensor(vec![Dim::Wildcard], precision)
        }
        Type::Var(_) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(shape_expr, env, vg, subst, adt_reg, errors, product);
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(_te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    // spec/04 §4.7.5: the slot-mismatch diagnostic names the
                    // fix, and its direction states the slot's demand rather
                    // than a unification-order artifact (chelis#916).
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::PrecisionMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "reshape expects an int64 shape list (write i64-suffixed \
                                     elements, e.g. 2i64, or cast(..., int64)), got {}",
                                    subst.apply(&shape_ty)
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
                // Inferring the shape list may have bound the input's own
                // type through a `shape(input, axis)` element. Re-read the
                // input before deriving the output.
                if let Type::Tensor(input_dims, precision) = subst.apply(&input_ty) {
                    let dims = reshape_output_dims(
                        shape_expr,
                        input_var_name.as_deref(),
                        &input_dims,
                        subst,
                    );
                    if let Err(error) = validate_reshape_target_dims(&dims, subst) {
                        return report(errors, error.into());
                    }
                    if subst.static_dim_products_match(&input_dims, &dims) == Some(false) {
                        let input_numel = subst.static_dim_product(&input_dims);
                        let target_numel = subst.static_dim_product(&dims);
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::DimensionMismatch,
                                match (target_numel, input_numel) {
                                    (Some(target), Some(input)) => format!(
                                        "reshape target has {target} elements but input tensor has {input}"
                                    ),
                                    _ => "reshape target element count does not match input tensor"
                                        .to_string(),
                                },
                                vec![],
                            ),
                        );
                    }
                    return Type::Tensor(dims, precision);
                }

                let dims = reshape_output_dims(shape_expr, input_var_name.as_deref(), &[], subst);
                if let Err(error) = validate_reshape_target_dims(&dims, subst) {
                    return report(errors, error.into());
                }
            }
            input_ty
        }
        Type::Error(_) => input_ty,
        _ => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                "reshape expects tensor input".to_string(),
                vec![],
            ),
        ),
    }
}

/// `shrink(&x, [[s0, e0], [s1, e1], ...]) -> tensor[e0-s0, e1-s1, ..., p]`
///
/// Per spec/05-risc-primitives.md §2.4, `shrink` slices a sub-tensor whose
/// rank matches the input and whose i-th axis dim is `end_i - start_i`.
/// The second argument is a list-of-pair-of-int32 with one entry per input
/// axis. Each pair is `[start, end]` with `0 <= start < end <= input_dim[i]`.
///
/// Closes issue Chelis-Lang/chelis#187 on the type-system side: before this
/// path was added, `shrink` was registered as `tensor_unop` (1-arg
/// `&tensor -> tensor`) so `shrink(&x, bounds)` failed with
/// `function arity mismatch: expected 1 args` even though the IR lowering
/// at `crates/chelis-ir/src/lower.rs:4635-4647` reads bounds from `args[1]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_shrink_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() != 3 {
        return report(errors, CheckError::new(
            CheckErrorKind::ArityMismatch,
            "shrink expects a tensor and a list of [start, end] bounds pairs, one pair per axis"
                .to_string(),
            vec![],
        ));
    }

    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let input_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let bounds_ty = infer_expr(&kids[2], env, vg, subst, adt_reg, errors, product);

    if let Some(err) = propagate_if_error([&input_ty, &bounds_ty]) {
        return err;
    }

    // The bounds argument must be a `List[List[Int64]]` (extent-domain
    // under [05-DIM-1]).
    let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
    let expected_bounds_ty = Type::Adt("List".to_string(), vec![int_list]);
    if let Err(_te) = unify(&bounds_ty, &expected_bounds_ty, subst) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "shrink expects a list of [start, end] int64 bounds pairs \
                         (write 0i64 or cast(..., int64) on each bound), got {}",
                        subst.apply(&bounds_ty)
                    ),
                ),
                vec![],
            ),
        );
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(&input_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("shrink expects tensor input, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    // Classify the (already desugared) Cons/Nil chain. The three-way
    // result distinguishes "concrete literals" (validate precisely)
    // from "structure looks fine but elements are non-literal" (defer
    // to runtime, output wildcards) from "structurally malformed"
    // (reject at infer with a clear axis-tagged message). See PR #214
    // red team round 1 finding R1-F1.
    let bounds: Vec<Option<(i64, i64)>> = match cons_chain_int_pairs(&kids[2]) {
        PairListShape::Literal(pairs) => pairs.into_iter().map(Some).collect(),
        // chelis#616: per-axis mixing — a literal pair keeps its precise
        // extent (and its infer-time validation); only a RUNTIME pair's own
        // axis becomes a wildcard. Collapsing every axis let unification
        // fill a runtime axis from a sibling literal axis, which the C
        // backend then baked as a wrong, unguarded allocation extent.
        PairListShape::Mixed(pairs) => pairs,
        PairListShape::Unknown => {
            return Type::Tensor(vec![Dim::Wildcard; dims.len()], prec);
        }
        PairListShape::Malformed { axis, reason } => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("shrink axis {axis} pair {reason}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    if bounds.len() != dims.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "shrink expects {} bounds pairs for rank {} tensor, got {}",
                        dims.len(),
                        dims.len(),
                        bounds.len()
                    ),
                ),
                vec![],
            ),
        );
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (pair, dim)) in bounds.iter().zip(dims.iter()).enumerate() {
        let Some((start, end)) = pair else {
            // Runtime bounds on this axis: extent known only at run time.
            // chelis#632 note: unlike stride/pad, shrink has no
            // checker-detectable IDENTITY form for a symbolic axis — the
            // IR-side full-axis sentinel (`start 0, end ToEnd`) has no
            // `PairListShape` counterpart, because a full-axis slice of a
            // symbolic dim necessarily spells its end as a runtime value
            // and lands here. Identity-precision for shrink is a possible
            // future refinement, not part of the chelis#632 fix.
            out_dims.push(Dim::Wildcard);
            continue;
        };
        if *start < 0 || *end < 0 {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("shrink axis {axis} bound [{start}, {end}] has negative endpoint"),
                    ),
                    vec![],
                ),
            );
        }
        if *start >= *end {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "shrink axis {axis} bound [{start}, {end}] is empty or inverted (start >= end)"
                        ),
                    ),
                    vec![],
                ),
            );
        }
        if let Dim::Lit(input_dim) = dim
            && *end > *input_dim
        {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "shrink axis {axis} bound [{start}, {end}] is out of range for input dim {input_dim}"
                        ),
                    ),
                    vec![],
                ),
            );
        }
        out_dims.push(Dim::Lit(end - start));
    }

    Type::Tensor(out_dims, prec)
}

/// `stride(&x, s0, s1, ...) -> tensor[ceil_div(d0, s0), ...]`
///
/// Per spec/05-risc-primitives.md §2.4, `stride` takes every `s_i`-th
/// element along axis i; the i-th output dim is `ceil(input_dim[i] /
/// s_i)`. The strides are passed as variadic int32 args, one per input
/// axis. Zero or negative strides are rejected.
///
/// Closes issue Chelis-Lang/chelis#187 on the type-system side -- before
/// this path was added, `stride` was registered as `tensor_unop` (arity
/// 1) so `stride(&x, 1, 2)` failed with `function arity mismatch:
/// expected 1 args`.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_stride_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() < 3 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                "stride expects a tensor followed by one positive int32 stride per axis"
                    .to_string(),
                vec![],
            ),
        );
    }

    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let input_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let stride_tys: Vec<Type> = kids[2..]
        .iter()
        .map(|arg| infer_expr(arg, env, vg, subst, adt_reg, errors, product))
        .collect();

    if let Some(err) = propagate_if_error(std::iter::once(&input_ty).chain(stride_tys.iter())) {
        return err;
    }

    for stride_ty in &stride_tys {
        let resolved = subst.apply(stride_ty);
        match resolved {
            // Stride steps are extent-domain ([05-DIM-1]): int64.
            Type::Prim(Prim::Int64) | Type::Var(_) | Type::Error(_) => {}
            other => {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!("stride expects int64 strides (write 2i64), got {other}"),
                        ),
                        vec![],
                    ),
                );
            }
        }
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(&input_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("stride expects tensor input, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    // Per-axis extraction (chelis#616): a literal (possibly cast-wrapped)
    // step keeps its precise infer-time validation and extent math; only a
    // RUNTIME step's own axis becomes a wildcard. Collapsing every axis to
    // a wildcard let unification fill a runtime axis's extent from a
    // sibling literal axis (see the shrink arm). Uses
    // `extract_int_for_dim` so `cast(N, int32)`-wrapped literal strides
    // reach the positive-stride check at infer time instead of falling
    // back to host runtime (red team round 3 finding R3-HIGH1).
    let strides: Vec<Option<i64>> = kids[2..].iter().map(extract_int_for_dim).collect();

    if strides.len() != dims.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "stride expects {} strides for rank {} tensor, got {}",
                        dims.len(),
                        dims.len(),
                        strides.len()
                    ),
                ),
                vec![],
            ),
        );
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (step, dim)) in strides.iter().zip(dims.iter()).enumerate() {
        let Some(step) = step else {
            // Runtime step on this axis: extent known only at run time.
            out_dims.push(Dim::Wildcard);
            continue;
        };
        if *step <= 0 {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "stride axis {axis} step {step} must be positive (zero or negative strides are not allowed)"
                        ),
                    ),
                    vec![],
                ),
            );
        }
        let step_us = *step as usize;
        match dim {
            Dim::Lit(input_dim) => {
                let out = (*input_dim as usize).div_ceil(step_us);
                out_dims.push(Dim::Lit(out as i64));
            }
            // chelis#632: only an IDENTITY step (1) passes a symbolic dim
            // through, mirroring the identity-only rule in
            // `chelis_ir::dag::shape_source_for_axis`. A non-identity
            // step changes the extent to `ceil(d/step)`, so keeping the
            // input symbol was an annotation-level lie that also falsely
            // tripped §4.4.1 rigidity on sig-symbol direct returns; the
            // fresh wildcard's extent is op-declared and runtime-guarded
            // by the chelis#616 machinery.
            other => out_dims.push(if step_us == 1 {
                other.clone()
            } else {
                Dim::Wildcard
            }),
        }
    }

    Type::Tensor(out_dims, prec)
}

/// `pad(&x, [[lo_0, hi_0], [lo_1, hi_1], ...], fill) -> tensor[d_0 + lo_0
/// + hi_0, ..., p]`
///
/// Per spec/05-risc-primitives.md §2.4, `pad` widens each axis by the
/// `(lo, hi)` padding amounts and fills the inserted region with `fill`.
/// Same structural antipattern as `shrink`: the `tensor_unop` registration
/// said 1-arg, but the IR lowering at `crates/chelis-ir/src/lower.rs:4616-4634`
/// reads padding from `args[1]` and fill from `args[2]`. Sibling sweep
/// finding for issue Chelis-Lang/chelis#187.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_pad_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() != 4 {
        return report(errors, CheckError::new(
            CheckErrorKind::ArityMismatch,
            "pad expects a tensor, a list of [lo, hi] padding pairs (one per axis), and a fill scalar".to_string(),
            vec![],
        ));
    }

    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let input_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let padding_ty = infer_expr(&kids[2], env, vg, subst, adt_reg, errors, product);
    let fill_ty = infer_expr(&kids[3], env, vg, subst, adt_reg, errors, product);

    if let Some(err) = propagate_if_error([&input_ty, &padding_ty]) {
        return err;
    }

    // Padding pairs are extent-domain ([05-DIM-1]): List[List[Int64]].
    let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
    let expected_padding_ty = Type::Adt("List".to_string(), vec![int_list]);
    if let Err(_te) = unify(&padding_ty, &expected_padding_ty, subst) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "pad expects a list of [lo, hi] int64 padding pairs \
                         (write 1i64 or cast(..., int64) on each amount), got {}",
                        subst.apply(&padding_ty)
                    ),
                ),
                vec![],
            ),
        );
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(&input_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("pad expects tensor input, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    // R1-F2: enforce the fill arg is a scalar of the input tensor
    // precision. The previous code dropped `fill_ty` on the floor, so a
    // list, tuple, bool, or wrong-precision scalar would slip through to
    // host-runtime. Per spec/05-risc-primitives.md §2.4, `pad`'s fill
    // value is a single scalar of the input precision.
    //
    // Use unification rather than a hard match so polymorphic-precision
    // tensors (precision still a `TensorPrec::Var`) generate the
    // constraint cleanly instead of being rejected. The expected scalar
    // type is `Type::Prim(p)` where `p` is the tensor's element
    // precision.
    let expected_fill_ty = match prec {
        TensorPrec::Concrete(p) => Type::Prim(p),
        TensorPrec::Var(_) => {
            // Precision is still polymorphic; introduce a fresh tvar and
            // let unification tie it to whatever the tensor lands on.
            Type::Var(vg.fresh_tvar())
        }
    };
    if let Err(_te) = unify(&fill_ty, &expected_fill_ty, subst) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "pad fill must be a scalar of the input tensor precision ({expected_fill_ty}), got {}",
                        subst.apply(&fill_ty)
                    ),
                ),
                vec![],
            ),
        );
    }

    let padding: Vec<Option<(i64, i64)>> = match cons_chain_int_pairs(&kids[2]) {
        PairListShape::Literal(pairs) => pairs.into_iter().map(Some).collect(),
        // chelis#616: per-axis mixing — only a RUNTIME pair's own axis
        // wildcards (see the shrink arm for the mis-size this prevents).
        PairListShape::Mixed(pairs) => pairs,
        PairListShape::Unknown => {
            return Type::Tensor(vec![Dim::Wildcard; dims.len()], prec);
        }
        PairListShape::Malformed { axis, reason } => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("pad axis {axis} pair {reason}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    if padding.len() != dims.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "pad expects {} padding pairs for rank {} tensor, got {}",
                        dims.len(),
                        dims.len(),
                        padding.len()
                    ),
                ),
                vec![],
            ),
        );
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (pair, dim)) in padding.iter().zip(dims.iter()).enumerate() {
        let Some((lo, hi)) = pair else {
            // Runtime padding on this axis: extent known only at run time.
            out_dims.push(Dim::Wildcard);
            continue;
        };
        if *lo < 0 || *hi < 0 {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("pad axis {axis} padding [{lo}, {hi}] has negative entry"),
                    ),
                    vec![],
                ),
            );
        }
        match dim {
            Dim::Lit(input_dim) => {
                out_dims.push(Dim::Lit(input_dim + lo + hi));
            }
            // chelis#632: only ZERO padding passes a symbolic dim through
            // (identity; mirrors `shape_source_for_axis`'s zero-pad arm).
            // Non-zero padding widens the extent to `d + lo + hi`; see
            // the stride arm above for the full rationale.
            other => out_dims.push(if *lo == 0 && *hi == 0 {
                other.clone()
            } else {
                Dim::Wildcard
            }),
        }
    }

    Type::Tensor(out_dims, prec)
}

/// `reduce_window_*(&x, window_shape, strides)` infer.
///
/// Per `spec/05-risc-primitives.md` §2.3.1:
/// - `window_shape` and `strides` are `List[int32]` of equal length
///   `n >= 1`.
/// - The trailing `n` axes of the input are the windowed axes; leading
///   `rank - n` axes pass through.
/// - Each window/stride entry must be a positive int32 literal at
///   check time (non-literal arguments fall back to a wildcard output
///   shape so runtime checks can still apply).
/// - Output rank equals input rank. Trailing dim i is
///   `floor((input_dims[rank - n + i] - window_shape[i]) / strides[i]) + 1`.
///   A non-positive result is rejected as a `DimensionMismatch` per
///   §2.3.1.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_reduce_window_app(
    list: &deep::List,
    name: &str,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() != 4 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                format!(
                    "{name} expects 3 arguments (tensor, window_shape, strides), got {}",
                    kids.len().saturating_sub(1)
                ),
                vec![],
            ),
        );
    }
    let _func_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let input_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    let window_ty = infer_expr(&kids[2], env, vg, subst, adt_reg, errors, product);
    let stride_ty = infer_expr(&kids[3], env, vg, subst, adt_reg, errors, product);

    if let Some(err) = propagate_if_error([&input_ty, &window_ty, &stride_ty]) {
        return err;
    }

    let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
    if let Err(_te) = unify(&window_ty, &int_list, subst) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "{name} expects window_shape to be List[int64], got {}",
                        subst.apply(&window_ty)
                    ),
                ),
                vec![],
            ),
        );
    }
    if let Err(_te) = unify(&stride_ty, &int_list, subst) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "{name} expects strides to be List[int64], got {}",
                        subst.apply(&stride_ty)
                    ),
                ),
                vec![],
            ),
        );
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(&input_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("{name} expects tensor input, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };

    // Extract literal window / stride entries. Non-literal arguments
    // are accepted at infer time (the type is still `List[int64]`) but
    // the output shape collapses to wildcards so the host runtime can
    // do the final shape check.
    let window_lit = cons_chain_int_list(&kids[2]);
    let strides_lit = cons_chain_int_list(&kids[3]);
    let (Some(window_shape), Some(strides)) = (window_lit, strides_lit) else {
        return Type::Tensor(vec![Dim::Wildcard; dims.len()], prec);
    };

    if window_shape.is_empty() || strides.is_empty() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{name} requires a non-empty window_shape and strides"),
                ),
                vec![],
            ),
        );
    }
    if window_shape.len() != strides.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "{name} window_shape (len {}) and strides (len {}) must agree",
                        window_shape.len(),
                        strides.len()
                    ),
                ),
                vec![],
            ),
        );
    }
    let n = window_shape.len();
    if dims.len() < n {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{name} window arity {n} exceeds tensor rank {}", dims.len()),
                ),
                vec![],
            ),
        );
    }
    for (i, &w) in window_shape.iter().enumerate() {
        if w <= 0 {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("{name} window_shape[{i}] = {w} must be >= 1"),
                    ),
                    vec![],
                ),
            );
        }
    }
    for (i, &s) in strides.iter().enumerate() {
        if s <= 0 {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("{name} strides[{i}] = {s} must be >= 1"),
                    ),
                    vec![],
                ),
            );
        }
    }
    let leading = dims.len() - n;
    let mut out_dims = Vec::with_capacity(dims.len());
    out_dims.extend(dims[..leading].iter().map(|d| subst.apply_dim(d)));
    for i in 0..n {
        let resolved = subst.apply_dim(&dims[leading + i]);
        match &resolved {
            Dim::Lit(in_dim) => {
                let w = window_shape[i];
                let s = strides[i];
                if *in_dim < w {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "{name} axis {} input dim {in_dim} < window_shape[{i}] = {w}",
                                    leading + i
                                ),
                            ),
                            vec![],
                        ),
                    );
                }
                // `in_dim >= w` (checked above) and `s >= 1` guarantee
                // `out = floor((in_dim - w) / s) + 1 >= 1`, so the Valid
                // output extent is always positive here — the `in_dim < w`
                // guard above is what rejects the empty-window case.
                let out = (*in_dim - w) / s + 1;
                out_dims.push(Dim::Lit(out));
            }
            _ => out_dims.push(Dim::Wildcard),
        }
    }

    Type::Tensor(out_dims, prec)
}

/// Walk a `Cons(a, Cons(b, ..., Nil))` chain and return the literal
/// integer entries (cast-aware via `extract_int_for_dim`). Returns
/// `None` when any element is non-literal or when the structure does
/// not terminate cleanly in `Nil`.
///
/// Shares the cons-chain walk with `collect_cons_chain_for_shape`
/// (the structural recognizer) and only adds the per-element
/// integer-literal extraction on top.
pub(super) fn cons_chain_int_list(expr: &deep::Expr) -> Option<Vec<i64>> {
    collect_cons_chain_for_shape(expr)?
        .iter()
        .map(|e| extract_int_for_dim(e))
        .collect()
}

/// Three-way result of inspecting a `[[s_0, e_0], [s_1, e_1], ...]` list
/// literal arg: well-formed concrete literals, structurally malformed
/// (wrong inner length, missing `Nil`, etc.), or "structure looks fine
/// but inner entries are non-literal" (e.g. variables) so the caller
/// should fall back to a wildcard output shape.
///
/// Red team round 1 on PR #214 found that `cons_chain_int_pairs`
/// returning a plain `Option` couldn't distinguish "user wrote a triple"
/// from "user wrote a variable" -- both became `None`, both fell through
/// to `Dim::Wildcard`, so malformed input silently slipped past
/// `chelis check` and only failed at host-runtime or IR-verifier time.
pub(super) enum PairListShape {
    /// Top-level chain closed by `Nil`, every entry was a literal
    /// `Cons(start, Cons(end, Nil))` pair.
    Literal(Vec<(i64, i64)>),
    /// Top-level chain closed by `Nil` and every entry was structurally
    /// a `Cons(_, Cons(_, Nil))`, but at least one inner element was a
    /// non-literal (variable, call, etc.). Carries the PER-AXIS
    /// classification: `Some((start, end))` for a literal pair (which
    /// the caller must still validate and size precisely), `None` for a
    /// runtime pair (that axis alone becomes a wildcard; runtime
    /// validates it). chelis#616 red-team finding: collapsing EVERY
    /// axis to a wildcard here let downstream unification fill a
    /// runtime axis's extent from a sibling literal axis, and the C
    /// backend baked the wrong literal with no guard — a silent
    /// mis-size.
    Mixed(Vec<Option<(i64, i64)>>),
    /// At least one inner entry has the wrong structural shape (wrong
    /// number of elements, missing `Nil` close, etc.). The caller MUST
    /// emit an infer-time error naming the offending axis.
    Malformed { axis: usize, reason: String },
    /// The top-level chain is well-typed as `List[List[Int64]]` but
    /// isn't a literal Cons/Nil chain (e.g. it's a variable resolved by
    /// the type system). Caller falls back to wildcard output shape.
    Unknown,
}

/// Walk a `Cons(Cons(start_i, Cons(end_i, Nil)), ..., Nil)` chain — the
/// desugared form of a Surf `[[start_0, end_0], [start_1, end_1], ...]`
/// list-of-pair literal — and classify it via [`PairListShape`].
pub(super) fn cons_chain_int_pairs(expr: &deep::Expr) -> PairListShape {
    let mut pairs: Vec<Option<(i64, i64)>> = Vec::new();
    let mut any_non_literal = false;
    let mut cursor = expr;
    let mut axis = 0usize;
    loop {
        // chelis#1107 amendment: carrier-preserving read. A `List`-only
        // destructure classified every stamped pad/crop pair-list as
        // `Unknown`, so the bounds check never ran on the typed ingress.
        let (outer_tag, outer_kids) = match stamped_parts(cursor) {
            Some((tag, _, kids)) => (Some(tag), kids),
            None => return PairListShape::Unknown,
        };
        match outer_tag {
            Some(DeepTag::Var) => {
                let name = match outer_kids.first().and_then(symbol_name) {
                    Some(name) => name,
                    None => return PairListShape::Unknown,
                };
                if name == "Nil" {
                    if any_non_literal {
                        return PairListShape::Mixed(pairs);
                    }
                    return PairListShape::Literal(
                        pairs
                            .into_iter()
                            .map(|pair| pair.expect("all literal"))
                            .collect(),
                    );
                }
                return PairListShape::Unknown;
            }
            Some(DeepTag::App) => {
                let app_children = outer_kids;
                let func = match app_children.first() {
                    Some(func) => func,
                    None => return PairListShape::Unknown,
                };
                if !is_builtin_var(func, "Cons") {
                    return PairListShape::Unknown;
                }
                let pair_expr = match app_children.get(1) {
                    Some(p) => p,
                    None => return PairListShape::Unknown,
                };
                let tail = match app_children.get(2) {
                    Some(t) => t,
                    None => return PairListShape::Unknown,
                };
                match cons_chain_two_ints(pair_expr, axis) {
                    InnerPairShape::Literal(pair) => pairs.push(Some(pair)),
                    InnerPairShape::NonLiteral => {
                        any_non_literal = true;
                        pairs.push(None);
                    }
                    InnerPairShape::Malformed { reason } => {
                        return PairListShape::Malformed { axis, reason };
                    }
                    InnerPairShape::Unknown => return PairListShape::Unknown,
                }
                cursor = tail;
                axis += 1;
            }
            _ => return PairListShape::Unknown,
        }
    }
}

/// Classification of a single inner pair expression. Distinguishes the
/// "wrong shape" case (must be reported at infer) from the "right shape,
/// non-literal element" case (defer to runtime).
pub(super) enum InnerPairShape {
    Literal((i64, i64)),
    NonLiteral,
    Malformed { reason: String },
    Unknown,
}

/// Walk a `Cons(start, Cons(end, Nil))` chain and classify it. Counts
/// the actual number of elements in the inner list so the error message
/// can name the bad arity explicitly (e.g. "got 3-element list").
///
/// Bare `(var Nil)` at the top level is the desugared form of `[]` --
/// a zero-element list literal. That is just as malformed as a triple
/// or singleton (it has zero of the required two endpoints), so it
/// must surface as `Malformed { reason: "got 0-element list" }` rather
/// than `NonLiteral` (red team round 2 finding R2-M1). Other `var` tags
/// represent opaque `List[Int32]` references the type system already
/// constrained; those still defer to runtime via `NonLiteral`.
///
/// Inner head values are extracted via [`extract_int_for_dim`], which
/// peels `cast(N, int32)` / `cast(N, int64)` -- so cast-wrapped int
/// literals participate in the infer-time bounds check rather than
/// silently falling back to `NonLiteral` (red team round 2 finding
/// R2-L1; mirrors how reshape extracts dim literals).
pub(super) fn cons_chain_two_ints(expr: &deep::Expr, _axis: usize) -> InnerPairShape {
    // chelis#1107 amendment: carrier-preserving read.
    let (outer_tag, outer_kids) = match stamped_parts(expr) {
        Some((tag, _, kids)) => (Some(tag), kids),
        None => return InnerPairShape::Unknown,
    };
    if outer_tag != Some(DeepTag::App) {
        // Inner element is not a Cons-chain. The `Nil` case (zero-element
        // list literal) is malformed; any other `var` is an opaque
        // `List[Int32]` reference whose contents the runtime will check.
        if matches!(outer_tag, Some(DeepTag::Var)) {
            let is_nil = outer_kids
                .first()
                .and_then(symbol_name)
                .map(|name| name == "Nil")
                .unwrap_or(false);
            if is_nil {
                return InnerPairShape::Malformed {
                    reason: "expects a pair [start, end] of two int literals, got 0-element list"
                        .to_string(),
                };
            }
            return InnerPairShape::NonLiteral;
        }
        return InnerPairShape::Unknown;
    }
    // Count the elements in the inner list so we can give a precise
    // "got N-element list" diagnostic. Walk the chain element-by-element.
    let mut elements_seen = 0usize;
    let mut head_values: Vec<Option<i64>> = Vec::new();
    let mut inner_cursor: &deep::Expr = expr;
    loop {
        // chelis#1107 amendment: carrier-preserving read.
        let (inner_tag, inner_kids) = match stamped_parts(inner_cursor) {
            Some((tag, _, kids)) => (Some(tag), kids),
            None => return InnerPairShape::Unknown,
        };
        match inner_tag {
            Some(DeepTag::Var) => {
                let name = match inner_kids.first().and_then(symbol_name) {
                    Some(n) => n,
                    None => return InnerPairShape::Unknown,
                };
                if name != "Nil" {
                    return InnerPairShape::Unknown;
                }
                if elements_seen != 2 {
                    return InnerPairShape::Malformed {
                        reason: format!(
                            "expects a pair [start, end] of two int literals, got {}-element list",
                            elements_seen
                        ),
                    };
                }
                let start = match head_values[0] {
                    Some(v) => v,
                    None => return InnerPairShape::NonLiteral,
                };
                let end = match head_values[1] {
                    Some(v) => v,
                    None => return InnerPairShape::NonLiteral,
                };
                return InnerPairShape::Literal((start, end));
            }
            Some(DeepTag::App) => {
                let app_children = inner_kids;
                let func = match app_children.first() {
                    Some(f) => f,
                    None => return InnerPairShape::Unknown,
                };
                if !is_builtin_var(func, "Cons") {
                    return InnerPairShape::Unknown;
                }
                let head_expr = match app_children.get(1) {
                    Some(h) => h,
                    None => return InnerPairShape::Unknown,
                };
                let tail = match app_children.get(2) {
                    Some(t) => t,
                    None => return InnerPairShape::Unknown,
                };
                head_values.push(extract_int_for_dim(head_expr));
                elements_seen += 1;
                inner_cursor = tail;
                // Guard against extra trailing elements: if we already
                // saw a [start, end] pair but the chain continues past
                // `Nil`, report malformed. The Nil arm above catches the
                // n==2 happy path before we get here on subsequent
                // iterations, so just keep walking and the count check
                // at Nil-time will catch it.
            }
            _ => return InnerPairShape::Unknown,
        }
    }
}

pub(super) fn list_literal_len(expr: &deep::Expr) -> Option<usize> {
    // The host-lane list-literal spelling is compiler-internal and outside
    // the closed vocabulary; it stays symbol-headed (raw-string boundary).
    //
    // chelis#1107 amendment: because that head never decodes, the stamp pass
    // carries it as `Expr::UnknownForm`, not `Expr::Node` -- so this reader
    // needs an `UnknownForm` arm rather than `stamped_parts`. Without it the
    // rank fell back to 1 on the typed ingress while the normalizing ingress
    // read the real element count.
    match expr {
        deep::Expr::List(list, _) if list.unknown_tag_symbol() == Some("list") => {
            Some(children(list).len())
        }
        deep::Expr::UnknownForm(data) if data.head == "list" => Some(data.children.len()),
        _ => None,
    }
}

/// Build the output dim list for `reshape(input, shape_list)`.
///
/// Walks `shape_expr` element by element. For each element, the first
/// recognizer that matches wins:
///
/// 1. concrete int literal (or `cast(N, int{32,64})`) → `Dim::Lit(N)`;
/// 2. `cast(shape(input, lit_axis), int64)` where the inner var matches
///    the reshape input by name and `lit_axis` is a valid axis of the
///    input → the input's dim at that axis (resolved through `subst`);
/// 3. fallback → `Dim::Wildcard`.
///
/// The shape list itself may surface as the explicit `(list ...)` tag
/// form or as a `Cons(head, ..., Nil)` chain after desugaring; both are
/// recognized. If neither form is matched, the rank is inferred from
/// `list_literal_len` (best-effort), and the whole output is filled
/// with `Wildcard`s -- the pre-fix behavior.
///
/// `input_var_name` is the name of the reshape input expression when it
/// is a bare `(var {} NAME)`, otherwise `None`. The symbolic-dim
/// recognizer requires this to match; with a non-var reshape input the
/// rule conservatively falls back to `Wildcard`.
///
/// Pre-fix the body of `infer_reshape_app` ran a literal-only
/// recognizer (the now-removed `list_literal_dims`) and fell back to
/// `vec![Wildcard; rank]` for anything else, including the common
/// runtime-batch pattern `cast(shape(x, axis), int64)`. That blind spot
/// is chelis#206; this helper closes it. The earlier `Dim::Lit`-only
/// behavior is also still covered (see the RT-A1W1 CRIT root cause for
/// chelis#35: `reshape(t, [2, 1, 3])` must yield
/// `tensor[Lit(2), Lit(1), Lit(3), p]`, not `tensor[Wildcard, ..., p]`).
pub(super) fn reshape_output_dims(
    shape_expr: &deep::Expr,
    input_var_name: Option<&str>,
    input_dims: &[Dim],
    subst: &Subst,
) -> Vec<Dim> {
    let elements = match collect_shape_list_elements(shape_expr) {
        Some(elems) => elems,
        None => {
            let rank = list_literal_len(shape_expr).unwrap_or(1);
            return vec![Dim::Wildcard; rank];
        }
    };
    elements
        .into_iter()
        .map(|elem| reshape_output_dim(elem, input_var_name, input_dims, subst))
        .collect()
}

fn validate_reshape_target_dims(dims: &[Dim], subst: &Subst) -> Result<(), TypeError> {
    if let Some(value) = dims.iter().find_map(|dim| match subst.apply_dim(dim) {
        Dim::Lit(value) if value < 0 => Some(value),
        _ => None,
    }) {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!("reshape target extents must be non-negative, got {value}"),
        });
    }
    Ok(())
}

/// Recognize a single dim-list element from a reshape shape list.
pub(super) fn reshape_output_dim(
    elem: &deep::Expr,
    input_var_name: Option<&str>,
    input_dims: &[Dim],
    subst: &Subst,
) -> Dim {
    if let Some(n) = extract_int_for_dim(elem) {
        return Dim::Lit(n);
    }
    if input_var_name.is_some()
        && let Some(axis) = extract_shape_axis_of(elem, input_var_name)
        && let Some(dim) = input_dims.get(axis)
    {
        // Resolve through current substitution so a recently-bound dim
        // var surfaces as its concrete name/lit.
        return subst.apply_dim(dim);
    }
    Dim::Wildcard
}

/// Collect the elements of a reshape shape-list argument as a flat
/// `Vec` of expressions, handling both the `(list ...)` tag form and
/// the desugared `Cons(head, ..., Nil)` chain. Returns `None` if the
/// shape arg is not a recognized list form (in which case the caller
/// falls back to all-wildcards with rank inferred from `list_literal_len`).
pub(super) fn collect_shape_list_elements(expr: &deep::Expr) -> Option<Vec<&deep::Expr>> {
    // chelis#1107 (measured, deliberately NOT extended to `UnknownForm`):
    // `(list ...)` is outside the 62-tag vocabulary, so in expression position
    // BOTH ingresses reject the program before this shape ever matters --
    // `infer_expr`'s `UnknownForm` arm fires on each. The stamp pass carries
    // it as `Expr::UnknownForm` and `normalize_nodes_to_lists` preserves that,
    // so both lanes miss this `List`-only arm identically and both fall back
    // to the same wildcard shape. The arm is symmetric across carriers and
    // cannot produce an ingress divergence; it stays live only for
    // programmatically built `Expr::List` trees, where both lanes see a
    // `List`. Teaching it `UnknownForm` would make the checker derive a shape
    // for a form it has already ruled invalid, which is not an improvement.
    if let deep::Expr::List(list, _) = expr
        && list.unknown_tag_symbol() == Some("list")
    {
        return Some(children(list).iter().collect());
    }
    let mut elems = Vec::new();
    let mut cursor = expr;
    loop {
        // chelis#1107 amendment (the red team's confirmed member): a
        // `List`-only destructure gave up on every stamped Cons chain, so
        // `reshape_output_dims` collapsed to a rank-1 wildcard on the typed
        // ingress while the normalizing ingress read the literal shape --
        // bidirectionally divergent, and user-reachable through `chelis
        // prove`.
        let (tag, _, kids) = stamped_parts(cursor)?;
        match tag {
            DeepTag::Var => {
                let name = kids.first().and_then(symbol_name)?;
                if name == "Nil" {
                    return Some(elems);
                }
                return None;
            }
            DeepTag::App => {
                let func = kids.first()?;
                if !is_builtin_var(func, "Cons") {
                    return None;
                }
                elems.push(kids.get(1)?);
                cursor = kids.get(2)?;
            }
            _ => return None,
        }
    }
}

/// If `expr` has one of the spec/04 §4.7.3 recognized forms
/// `shape(<var named input_var_name>, <concrete int axis>)` or
/// `cast(shape(<var named input_var_name>, <concrete int axis>), int64)`
/// (the second is an identity cast under [05-DIM-2], kept so the
/// pre-[05-DIM-2] spelling retains its propagation), return the axis.
/// Both `cast` and `shape` may surface either as the dedicated tag
/// (`(cast {} ...)`, ...) or as `(app {} (var {} cast) ...)`. The axis
/// expression matches `extract_int_for_dim` -- it accepts `N`, `lit N`,
/// and `cast(N, int{32,64})`.
///
/// Returns `None` when:
/// - a cast wrapper is present but its target is not `int64`,
/// - the candidate expression is not a `shape(...)` call,
/// - the shape's tensor arg is not a `var` matching `input_var_name`,
/// - the axis is not a concrete non-negative int.
pub(super) fn extract_shape_axis_of(
    expr: &deep::Expr,
    input_var_name: Option<&str>,
) -> Option<usize> {
    let input = input_var_name?;
    let candidate = match peel_cast(expr) {
        Some((inner, target_ty)) => {
            if !is_target_ty(target_ty, Prim::Int64) {
                return None;
            }
            inner
        }
        None => expr,
    };
    let shape_args = app_children_of(candidate)?;
    let func = shape_args.first()?;
    if !is_builtin_var(func, "shape") {
        return None;
    }
    let tensor_arg = shape_args.get(1)?;
    let arg_name = symbolic_dim_ref_name(tensor_arg)?;
    if arg_name != input {
        return None;
    }
    let axis_expr = shape_args.get(2)?;
    let axis = extract_int_for_dim(axis_expr)?;
    if axis < 0 {
        return None;
    }
    Some(axis as usize)
}

/// Strip one layer of `cast` (tag-form or `app`-form) and return
/// (inner_expr, target_type_expr).
pub(super) fn peel_cast(expr: &deep::Expr) -> Option<(&deep::Expr, &deep::Expr)> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(expr)?;
    match tag {
        DeepTag::Cast => Some((kids.first()?, kids.get(1)?)),
        DeepTag::App => {
            let func = kids.first()?;
            if !is_builtin_var(func, "cast") {
                return None;
            }
            Some((kids.get(1)?, kids.get(2)?))
        }
        _ => None,
    }
}

/// Treat `(t-prim {} <name>)` as the target type marker emitted by
/// `cast(..., int64)` etc. Returns true iff the marker matches `prim`.
pub(super) fn is_target_ty(expr: &deep::Expr, prim: Prim) -> bool {
    // chelis#1107 amendment: carrier-preserving read.
    let Some((DeepTag::TPrim, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    let Some(name_expr) = kids.first() else {
        return false;
    };
    symbol_name(name_expr) == Some(prim.name())
}

/// Treat `expr` as an `(app {} ...)` node and return its children, or
/// `None` if it isn't one.
///
/// chelis#1107 amendment: this used to hand back a `&deep::List`, which is
/// unrepresentable for a stamped `Expr::Node`. Returning the children slice
/// is carrier-agnostic and is all the single caller ever wanted. The old
/// `is_shape_app` helper went with it -- its sole call site immediately
/// re-checked the same `is_builtin_var(func, "shape")` condition itself.
pub(super) fn app_children_of(expr: &deep::Expr) -> Option<&[deep::Expr]> {
    match stamped_parts(expr)? {
        (DeepTag::App, _, kids) => Some(kids),
        _ => None,
    }
}

/// Extract an int literal from a Deep expr, looking through `cast(N, int64)`
/// and `cast(N, int32)` — both are common in Chelis dim lists since integer
/// literals default to int32 and require an explicit cast for int64 contexts.
/// `cast` may surface either as the `(cast {} ... ...)` tag or as an `app`
/// of the `cast` var, depending on how far desugaring has progressed.
///
/// Note: callers in the spec section 2.4 movement family (shrink, pad,
/// stride, permute, expand) typically unify the surrounding bounds /
/// strides argument against an `int32`-pinned expected type before
/// reaching this extractor, so a `cast(N, int64)` endpoint is rejected
/// at the outer unification step rather than slipping through to here
/// (red team round 3 HIGH-2 contract note).
///
/// The cast arm recurses through `extract_int_for_dim` so that
/// `cast(cast(N, int32), int32)` and other doubly-nested forms peel to
/// their literal at any depth (red team round 3 finding R3-MED2).
/// Termination is bounded: each recursive call strictly reduces the
/// expression depth (peels one wrapper layer).
///
/// Audit catalog of issue #216 sites that use this cast-aware extractor
/// (one row per infer-time int-literal extraction that gates a
/// user-facing validation check). Each row also notes any host-runtime
/// defense-in-depth so a regression here does not silently corrupt
/// runtime behavior, only the diagnostic layer.
///
/// | Domain               | Site (approx)                        | User-reachable cast? | Validation                | Host-runtime defense |
/// |----------------------|--------------------------------------|----------------------|---------------------------|----------------------|
/// | trace/diagonal axis  | `resolve_axis_pair_member` (~4110)   | yes                  | rank bounds + diagnostic  | yes (eval)           |
/// | builtin axis         | `resolve_builtin_axis` (~4154)       | yes                  | rank bounds + diagnostic  | yes (eval)           |
/// | conv output type   | `derive_conv_output_type` (~5720)  | yes (`stride=cast`)  | positivity + spatial dim  | yes (validator arm)  |
/// | conv output type   | `derive_conv_output_type` (~5721)  | yes (`padding=cast`) | non-neg + spatial dim     | yes (validator arm)  |
/// | conv validator     | `conv_parameters`| yes                  | literal-int + then >0/>=0 | yes (codegen panic)  |
/// | conv axis-dim      | `ir_builtin_axis_dim` (~5907)        | yes                  | rank bounds via normalize | yes (eval)           |
/// | softmax axis         | softmax arm (~7630)                  | yes (`axis=cast`)    | rank bounds + diagnostic  | yes (eval)           |
/// | shape axis           | shape arm (~8238)                    | yes (issue #206)     | non-neg + rank bounds     | yes (eval)           |
/// | split axis           | split arm (~8959)                    | yes                  | rank bounds + diagnostic  | yes (eval)           |
/// | conv spatial out   | `compute_concrete_conv_spatial`(11722)| yes                | positivity + spatial dim  | yes (validator arm)  |
/// | conv spatial out   | `compute_concrete_conv_spatial`(11723)| yes                | non-neg + spatial dim     | yes (validator arm)  |
/// | reduction axis       | `check_reduce_signature` (~12076)    | yes (`axis=cast`)    | rank bounds + diagnostic  | yes (eval)           |
/// | grad wrt tuple       | `grad_wrt_indices` (~13793)          | NO (surf desugar)    | int-type + non-neg        | yes (AD pass)        |
/// | grad wrt single      | `grad_wrt_indices` (~13814)          | NO (surf desugar)    | int-type + non-neg        | yes (AD pass)        |
/// | vmap axis            | `infer_vmap` (~13866)                | NO (surf parser)     | non-neg + diagnostic      | yes (eval)           |
///
/// The three "NO" rows -- vmap axis, both grad wrt sites -- have no
/// idiomatic Surf cast-wrapping pattern because the Surf parser /
/// desugarer normalizes them to bare literal ints before reaching the
/// extractor. They are reachable only through direct Deep input
/// (decompiler output, custom tooling, macro expansion). The swap there
/// is defense-in-depth on Deep-direct paths; the post-fix tests use
/// `parse_deep` rather than the Surf parser.
pub(super) fn extract_int_for_dim(expr: &deep::Expr) -> Option<i64> {
    stack_guard!("extract_int_for_dim", expr, None);
    if let Some(value) = extract_int_literal(expr) {
        return Some(value);
    }
    // chelis#1107: carrier-preserving read. A `List`-only destructure meant a
    // stamped `cast`-wrapped dimension read as "not statically known" on
    // `check_typed_program` while `check_ir_program` extracted it.
    let (tag, _, kids) = stamped_parts(expr)?;
    match tag {
        DeepTag::Cast => extract_int_for_dim(kids.first()?),
        DeepTag::App => {
            let func = kids.first()?;
            if !is_builtin_var(func, "cast") {
                return None;
            }
            extract_int_for_dim(kids.get(1)?)
        }
        _ => None,
    }
}
