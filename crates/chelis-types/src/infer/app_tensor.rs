//! Tensor operation signature checks.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn check_layer_norm_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    _vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if arg_tys.len() != 4 {
        return report_builtin_arity_bare(errors, "layer_norm", "4 arguments", arg_tys.len());
    }

    let x_ty = type_for_readonly_check(&arg_tys[0], subst);
    let gamma_ty = type_for_readonly_check(&arg_tys[1], subst);
    let beta_ty = type_for_readonly_check(&arg_tys[2], subst);

    let (x_dims, x_prec) = match x_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("layer_norm expects tensor input, got {other}"),
                    vec![],
                ),
            );
        }
    };

    let (gamma_dims, gamma_prec) = match gamma_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("layer_norm expects tensor gamma, got {other}"),
                    vec![],
                ),
            );
        }
    };

    let (beta_dims, beta_prec) = match beta_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("layer_norm expects tensor beta, got {other}"),
                    vec![],
                ),
            );
        }
    };

    if x_dims.is_empty() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                "layer_norm expects rank >= 1 input tensor".to_string(),
                vec![],
            ),
        );
    }
    if gamma_dims.len() != 1 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "layer_norm expects rank-1 gamma, got rank {}",
                    gamma_dims.len()
                ),
                vec![],
            ),
        );
    }
    if beta_dims.len() != 1 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "layer_norm expects rank-1 beta, got rank {}",
                    beta_dims.len()
                ),
                vec![],
            ),
        );
    }
    if x_prec != gamma_prec || x_prec != beta_prec {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "layer_norm requires matching precisions, got {}, {}, {}",
                    x_prec.name(),
                    gamma_prec.name(),
                    beta_prec.name()
                ),
                vec!["Insert explicit cast".to_string()],
            ),
        );
    }

    if let TensorPrec::Concrete(prim) = x_prec {
        if !prim.is_float() {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    "layer_norm requires one active float dtype".to_string(),
                    vec![],
                ),
            );
        }
        let epsilon_ty = type_for_readonly_check(&arg_tys[3], subst);
        if let Err(error) = unify(&epsilon_ty, &Type::Prim(prim), subst) {
            return report(errors, error.into());
        }
    }

    let hidden_dim = x_dims.last().cloned().expect("checked non-empty");
    if subst.observe_dim(&hidden_dim).known_extent() == Some(0) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                "layer_norm requires a positive hidden extent".to_string(),
                vec![],
            ),
        );
    }
    if let Err(te) = unify_dim(&hidden_dim, &gamma_dims[0], subst) {
        return report(errors, te.into());
    }
    if let Err(te) = unify_dim(&hidden_dim, &beta_dims[0], subst) {
        return report(errors, te.into());
    }

    let canonical = Type::Tensor(
        x_dims.into_iter().map(|d| subst.apply_dim(&d)).collect(),
        x_prec,
    );
    if let Err(te) = unify(result_ty, &canonical, subst) {
        return report(errors, te.into());
    }
    subst.apply(&canonical)
}

/// Literal per-axis metadata. Unknown values remain shape obligations; malformed
/// literal values are errors, never defaults. The declared scheme owns types.
pub(super) type ConvAxisParameters = (i64, i64, i64);

pub(super) fn conv_parameters(
    strides: &deep::Expr,
    padding: &deep::Expr,
    rank: usize,
) -> Result<Option<Vec<ConvAxisParameters>>, String> {
    let strides = collect_shape_list_elements(strides);
    let padding = collect_shape_list_elements(padding);
    if strides.as_ref().is_some_and(|xs| xs.len() != rank)
        || padding.as_ref().is_some_and(|xs| xs.len() != rank)
    {
        return Err(format!(
            "conv requires exactly {rank} stride and padding entries"
        ));
    }
    let (Some(strides), Some(padding)) = (strides, padding) else {
        return Ok(None);
    };
    let mut result = Vec::with_capacity(rank);
    let mut concrete = true;
    for (axis, (stride, pair)) in strides.into_iter().zip(padding).enumerate() {
        let pair = stamped_parts(pair)
            .filter(|(tag, _, kids)| *tag == DeepTag::Tuple && kids.len() == 2)
            .map(|(_, _, kids)| kids);
        let stride = extract_int_for_dim(stride);
        let low = pair.and_then(|xs| extract_int_for_dim(&xs[0]));
        let high = pair.and_then(|xs| extract_int_for_dim(&xs[1]));
        if stride.is_some_and(|s| s <= 0) {
            return Err(format!(
                "conv requires a positive stride, got {} (spatial axis {axis})",
                stride.expect("known nonpositive stride")
            ));
        }
        if low.is_some_and(|p| p < 0) || high.is_some_and(|p| p < 0) {
            return Err(format!(
                "conv requires non-negative padding at spatial axis {axis}"
            ));
        }
        if let (Some(stride), Some(low), Some(high)) = (stride, low, high) {
            result.push((stride, low, high));
        } else {
            concrete = false;
        }
    }
    Ok(concrete.then_some(result))
}

pub(super) fn compute_concrete_conv_spatial(
    arg_exprs: &[deep::Expr],
    input_dims: &[Dim],
    kernel_dims: &[Dim],
    subst: &Subst,
) -> Option<Vec<i64>> {
    let rank = input_dims.len().checked_sub(2)?;
    if rank == 0 || kernel_dims.len() != input_dims.len() {
        return None;
    }
    let params = conv_parameters(arg_exprs.get(2)?, arg_exprs.get(3)?, rank).ok()??;
    input_dims[2..]
        .iter()
        .zip(&kernel_dims[2..])
        .zip(params)
        .map(|((input, kernel), (stride, low, high))| {
            let input = subst.observe_dim(input).known_extent()?;
            let kernel = subst.observe_dim(kernel).known_extent()?;
            conv_output_extent(input, kernel, stride, low, high)
        })
        .collect()
}

pub(super) fn check_conv_signature(
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if arg_tys.len() != 4 {
        return report_builtin_arity_bare(errors, "conv", "4 arguments", arg_tys.len());
    }
    let input = type_for_readonly_check(&arg_tys[0], subst);
    let kernel = type_for_readonly_check(&arg_tys[1], subst);
    let (input_dims, input_prec, kernel_dims, kernel_prec) = match (&input, &kernel) {
        (Type::Var(_) | Type::Error(_), _) | (_, Type::Var(_) | Type::Error(_)) => {
            return subst.apply(result_ty);
        }
        (Type::Tensor(ds, p), Type::Tensor(ks, q)) => (ds, p, ks, q),
        _ => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "conv requires tensor input and kernel".to_string(),
                    vec![],
                ),
            );
        }
    };
    if input_dims.len() < 3 || kernel_dims.len() != input_dims.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "conv requires equal input/kernel ranks of at least 3, got {} and {}",
                    input_dims.len(),
                    kernel_dims.len()
                ),
                vec![],
            ),
        );
    }
    if input_prec != kernel_prec {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                "conv requires matching input/kernel precision".to_string(),
                vec![],
            ),
        );
    }
    if let TensorPrec::Concrete(p) = input_prec
        && !p.is_float()
    {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                "conv requires an active float dtype".to_string(),
                vec![],
            ),
        );
    }
    if let Err(error) = unify_dim(&input_dims[1], &kernel_dims[1], subst) {
        return report(errors, error.into());
    }
    let rank = input_dims.len() - 2;
    let spatial = compute_concrete_conv_spatial(arg_exprs, input_dims, kernel_dims, subst);
    let mut output_dims = vec![
        subst.apply_dim(&input_dims[0]),
        subst.apply_dim(&kernel_dims[0]),
    ];
    output_dims.extend(match spatial {
        Some(dims) => dims.into_iter().map(Dim::Lit).collect::<Vec<_>>(),
        None => (0..rank).map(|_| Dim::Var(vg.fresh_dvar())).collect(),
    });
    let output = Type::Tensor(output_dims, input_prec.clone());
    if let Err(error) = unify(result_ty, &output, subst) {
        return report(errors, error.into());
    }
    subst.apply(&output)
}

pub(super) fn check_matmul_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if arg_tys.len() != 2 {
        return report_builtin_arity_bare(errors, "matmul", "2 arguments", arg_tys.len());
    }

    let lhs = type_for_readonly_check(&arg_tys[0], subst);
    let rhs = type_for_readonly_check(&arg_tys[1], subst);

    let (lhs_dims, lhs_prec) = match lhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("matmul expects tensor lhs, got {other}"),
                    vec![],
                ),
            );
        }
    };
    let (rhs_dims, rhs_prec) = match rhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("matmul expects tensor rhs, got {other}"),
                    vec![],
                ),
            );
        }
    };

    if lhs_prec != rhs_prec {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "matmul requires matching precisions, got {} and {}",
                    lhs_prec.name(),
                    rhs_prec.name()
                ),
                vec!["Insert explicit cast".to_string()],
            ),
        );
    }
    // RT-2 fixup B6: per spec/04-type-system.md §5.7.2, the active
    // matmul signature does not admit integer operand precisions
    // (i8, i16, i32, i64). Reject upfront at the call site
    // with a §5.7.2-citing diagnostic so users see the spec rule
    // here, not as a downstream IR-verify or codegen failure. The
    // verify-layer F1 guard remains as defense in depth.
    if lhs_prec.is_integer() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "matmul on integer operand precision `{}` is not admitted in this \
                 cycle per spec/04-type-system.md §5.7.2: integer matmul not admitted \
                 (the spec deliberately defers the integer-matmul accumulator rule; \
                 use reduce_sum over an explicit expand+mul lowering for integer \
                 inner products)",
                    lhs_prec.name()
                ),
                vec![
                    "spec/04-type-system.md §5.7.2: there is no current backend that \
                 supports integer BLAS, and an integer-matmul surface raises \
                 questions (saturating vs wrapping accumulator, signed-vs-unsigned \
                 interaction with §1.1.2) that are out of scope here. Integer \
                 reduce_sum is supported per §5.7.1."
                        .to_string(),
                ],
            ),
        );
    }
    if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "matmul expects tensors of rank >= 2, got rank {} and {}",
                    lhs_dims.len(),
                    rhs_dims.len()
                ),
                vec![],
            ),
        );
    }
    if let Err(te) = unify_dim(
        &lhs_dims[lhs_dims.len() - 1],
        &rhs_dims[rhs_dims.len() - 2],
        subst,
    ) {
        return report(errors, te.into());
    }

    let lhs_lead = &lhs_dims[..lhs_dims.len() - 2];
    let rhs_lead = &rhs_dims[..rhs_dims.len() - 2];
    let lead_len = lhs_lead.len().max(rhs_lead.len());
    let mut out_dims = Vec::with_capacity(lead_len + 2);
    for offset in 0..lead_len {
        let lhs_idx = lhs_lead.len().checked_sub(lead_len - offset);
        let rhs_idx = rhs_lead.len().checked_sub(lead_len - offset);
        let dim = match (
            lhs_idx.map(|idx| &lhs_lead[idx]),
            rhs_idx.map(|idx| &rhs_lead[idx]),
        ) {
            (Some(lhs_dim), Some(rhs_dim)) => {
                let lhs_applied = subst.apply_dim(lhs_dim);
                let rhs_applied = subst.apply_dim(rhs_dim);
                if subst.observe_dim(lhs_dim).known_extent() == Some(1) {
                    rhs_applied
                } else if subst.observe_dim(rhs_dim).known_extent() == Some(1) {
                    lhs_applied
                } else {
                    if let Err(te) = unify_dim(&lhs_applied, &rhs_applied, subst) {
                        return report(errors, te.into());
                    }
                    subst.apply_dim(&lhs_applied)
                }
            }
            (Some(lhs_dim), None) => subst.apply_dim(lhs_dim),
            (None, Some(rhs_dim)) => subst.apply_dim(rhs_dim),
            (None, None) => unreachable!(),
        };
        out_dims.push(dim);
    }
    out_dims.push(subst.apply_dim(&lhs_dims[lhs_dims.len() - 2]));
    out_dims.push(subst.apply_dim(&rhs_dims[rhs_dims.len() - 1]));

    let canonical = Type::Tensor(out_dims, lhs_prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        return report(errors, te.into());
    }
    subst.apply(&canonical)
}

pub(super) fn check_reduction_signature(
    name: &str,
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if arg_tys.len() < 2 {
        return report_builtin_arity_bare(errors, name, "at least 2 arguments", arg_tys.len());
    }

    let input_ty = type_for_readonly_check(&arg_tys[0], subst);
    let (dims, prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("{name} expects tensor input, got {other}"),
                    vec![],
                ),
            );
        }
    };

    if name == "count" && !matches!(prec, TensorPrec::Concrete(Prim::Bool)) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "count expects exactly a bool tensor, got tensor precision {}",
                    prec.render()
                ),
                vec!["Use count for bool tensors; numeric reductions use sum/prod_reduce.".into()],
            ),
        );
    }

    // Resolve which axis (or axes) the reduction removes. Two modes:
    //
    //  * Positional (legacy): a single compile-time-constant integer axis on a
    //    *concrete-rank* operand (`sum(x, 0)` / `sum(x, cast(-1, i32))`).
    //    `normalize_static_axis` handles negative indexing and bounds (issue
    //    #216), consistent with gather/scatter and IR lowering's
    //    `normalize_axis`.
    //
    //  * Named (Tier-3, spec/04-type-system.md §4.5.3): one or more axes named by
    //    the dimension they remove (`sum(x, seq)` / `sum(x, seq, head)`). Named
    //    axes are the only valid form on a rank-spread operand — a positional
    //    index is meaningless at symbolic rank — and they preserve the
    //    surviving named axes in the output.
    let axis_exprs = &arg_exprs[1..];
    let has_spread = dims.iter().any(|d| matches!(d, Dim::Rank(_)));

    // The reduction HM schemes are arity-2, but the variadic named-axis form
    // (`sum(x, seq, head)`, chelis#339) reaches this arm through the
    // `infer_reduction_app` dispatcher with N axis exprs; the loop below
    // resolves each named axis and rejects positional integers, unknown
    // names, ambiguity, and duplicates. Composition
    // (`sum(sum(x, head), seq)`) remains equivalent and order-insensitive.
    let mut remove: Vec<usize> = Vec::new();
    if !has_spread
        && (axis_exprs.len() == 1 || name == "count")
        && axis_exprs
            .iter()
            .all(|axis| extract_int_for_dim(axis).is_some())
    {
        for axis_expr in axis_exprs {
            let raw = extract_int_for_dim(axis_expr).expect("guarded static axis");
            let Some(axis) = normalize_static_axis(dims.len(), raw) else {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name} axis {raw} is out of bounds for rank {} tensor",
                            dims.len()
                        ),
                        vec![],
                    ),
                );
            };
            if remove.contains(&axis) {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name}: duplicate reduction axis {raw}; each normalized axis may appear at most once"
                        ),
                        vec![],
                    ),
                );
            }
            remove.push(axis);
        }
    } else {
        let selects_concrete_named_axis = name == "count"
            && !has_spread
            && axis_exprs.iter().any(|axis| {
                symbolic_dim_ref_name(axis).is_some_and(|axis_name| {
                    dims.iter()
                        .any(|dim| matches!(subst.semantic_dim(dim), Dim::Name(name) if name == axis_name))
                })
            });
        if selects_concrete_named_axis {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    "count on a concrete-rank operand requires one or more positional i32 axes; named axes are reserved for rank-polymorphic operands".to_string(),
                    vec!["Use the selected dimensions' positional indices, or make the operand rank-polymorphic and name every selected axis.".to_string()],
                ),
            );
        }
        for ax in axis_exprs {
            // A positional integer that reaches the named path: either the
            // operand is rank-spread (index meaningless at symbolic rank) or it
            // is mixed with other axes. Direct the user to name each axis.
            if extract_int_for_dim(ax).is_some() {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name}: positional and named axes cannot be mixed, and positional axes \
                         require a concrete-rank operand; name every selected axis on a \
                         rank-spread operand (e.g. `{name}(x, seq, head)`) \
                         (spec/04-type-system.md \u{00a7}4.5.3)"
                        ),
                        vec![],
                    ),
                );
            }
            // Issue #259: a non-literal, non-name axis (a runtime `i32`
            // binding) cannot determine which dimension is removed; emit the
            // targeted compile-time-constant diagnostic rather than leaking an
            // unresolved output type downstream.
            let Some(axis_name) = symbolic_dim_ref_name(ax) else {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name} axis must be a compile-time constant or a named axis of the \
                         operand, got {}",
                            describe_axis_arg(Some(ax)),
                        ),
                        vec![format!(
                            "Pass a literal axis (e.g. `{name}(x, 0)`) on a concrete-rank operand, or \
                         name the axis (e.g. `{name}(x, seq)`) to reduce by name."
                        )],
                    ),
                );
            };
            let hits: Vec<usize> = dims
                .iter()
                .enumerate()
                .filter(|(_, d)| matches!(subst.semantic_dim(d), Dim::Name(n) if n == axis_name))
                .map(|(i, _)| i)
                .collect();
            match hits.as_slice() {
                [i] => {
                    // chelis#339: a duplicate axis name in the variadic list
                    // is a hard error, never a silent deduplication.
                    if remove.contains(i) {
                        return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::DimensionMismatch,
                                format!(
                                    "{name}: duplicate reduction axis `{axis_name}`; each named \
                                 axis may appear at most once in a variadic reduction \
                                 (spec/04-type-system.md \u{00a7}4.5.3)"
                                ),
                                vec![],
                            ),
                        );
                    }
                    remove.push(*i);
                }
                [] if has_spread => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{name}: rank-spread operand has no named `{axis_name}` axis to \
                             reduce (spec/04-type-system.md \u{00a7}4.5.3)"
                            ),
                            vec![],
                        ),
                    );
                }
                [] => {
                    // Concrete operand: `axis_name` is neither a literal nor a
                    // named axis of the operand. Two causes share this arm — a
                    // runtime `i32` binding (issue #259) and a mistyped/absent
                    // axis name — so the message stays neutral between them
                    // rather than asserting "runtime value".
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{name} axis `{axis_name}` is neither a compile-time constant nor a \
                             named axis of the operand: a reduction axis must be a literal or \
                             `cast(N, i32)` constant, or the name of an existing axis"
                            ),
                            vec![format!(
                                "Pass a literal axis (e.g. `{name}(x, 0)`) or `cast(N, i32)`, or \
                             name an existing axis of the operand (e.g. `{name}(x, seq)`)."
                            )],
                        ),
                    );
                }
                _ => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{name}: named axis `{axis_name}` is ambiguous; it appears more than \
                             once in the operand shape"
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }
    }

    // Build the output by dropping the selected axes (descending so earlier
    // indices stay valid). Surviving named/spread dims keep identity and
    // order. Duplicates were rejected loudly above (chelis#339), so no
    // silent dedup happens here.
    let mut out_dims = dims;
    remove.sort_unstable();
    for &idx in remove.iter().rev() {
        out_dims.remove(idx);
    }

    // RT-2 fixup B1: per spec/04-type-system.md §5.7.1, the result
    // precision of `reduce_sum` follows the §5.7.1 table — i8/i16
    // operand → i32 result, i32/i64/f32/f64 → operand precision,
    // bf16/f16 → operand precision (the f32 accumulator is consumed
    // inside the op and downcast on output). For `max_reduce`,
    // `min_reduce`, `prod_reduce`, and `mean` the result precision is
    // the operand precision.
    //
    // Issue #230: `argmax_reduce` and `argmin_reduce` are index-returning
    // reductions — they produce element indices, not reduced operand
    // values. Their result precision is canonically `i64`, regardless
    // of the input dtype. The std-package signatures in
    // `packages/chelis-std/src/tensor/reduce.ch` pin this (`tensor[b,
    // i64]`); the type checker was returning the input precision and
    // diverging from std. (The host-runtime/backend still stores
    // integer-valued floats internally per the Phase 3j-pre Batch 1
    // caveat documented on `RiscOp::Argmax`; the i64 label is the
    // declarative output type.)
    //
    // WS-A5: the §5.7.1 widening rule is defined over a known operand
    // precision. If the operand precision is still polymorphic
    // (TensorPrec::Var), defer the decision until the precision is
    // resolved by unification — return the canonical-but-still-poly
    // result type and let the standard unify path proceed.
    let result_prec: TensorPrec = if name == "count" {
        TensorPrec::Concrete(Prim::Int64)
    } else if name == "sum" {
        match &prec {
            TensorPrec::Concrete(p) => match p.default_reduce_sum_result_precision() {
                Ok(rp) => TensorPrec::Concrete(rp),
                Err(msg) => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!("sum: {msg}"),
                            vec![],
                        ),
                    );
                }
            },
            TensorPrec::Var(_) => prec.clone(),
        }
    } else if name == "argmax_reduce" || name == "argmin_reduce" {
        TensorPrec::Concrete(Prim::Int64)
    } else {
        prec.clone()
    };
    // RT-2 fixup B1: emit a §5.7.1-citing diagnostic at the call site
    // before falling back to the generic unify error, so users binding
    // `sum(i8 tensor)` to `tensor[i8]` see the spec-row hint
    // instead of the opaque "doesn't match declared signature" trail.
    if name == "sum" && result_prec != prec {
        let resolved_result = subst.apply(result_ty);
        if let Type::Tensor(_, declared_prec) = resolved_result
            && declared_prec != result_prec
        {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    format!(
                        "sum on operand precision `{}` produces result precision `{}` per \
                     spec/04-type-system.md §5.7.1 (the §5.7.1 result-precision table \
                     widens narrow integer operands to i32 to prevent silent overflow); \
                     declared result precision `{}` is incompatible. Use `tensor[{}]` or \
                     omit the result type to accept the spec default.",
                        prec.render(),
                        result_prec.render(),
                        declared_prec.render(),
                        result_prec.render(),
                    ),
                    vec![format!(
                        "spec/04-type-system.md §5.7.1: `reduce_sum` on `{}` operands \
                     produces a `{}` result by default to prevent silent overflow",
                        prec.render(),
                        result_prec.render(),
                    )],
                ),
            );
        }
    }
    let canonical = Type::Tensor(out_dims, result_prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        return report(errors, te.into());
    }
    subst.apply(&canonical)
}

/// `expand`'s unit-extent claim, statically refuted.
///
/// spec/05-risc-primitives.md §2.4.1: "`expand` sets the extent at `axis` and
/// is well formed only when the operand's extent at `axis` is 1 (the size-1
/// broadcast of §2.4's table); the operation is a claim that the operand's
/// extent at `axis` is 1. A literal operand extent at `axis` other than 1 is
/// a type error. A symbolic or runtime operand extent at `axis` other than 1
/// fails that claim's runtime extent guard and traps `Domain`."
///
/// So this returns an error only for a literal that refutes the claim.
/// Everything else, a named dim or an unconstrained extent, is admitted here
/// and carries the claim into the IR, where the §4.7 runtime extent guard
/// compares it against the value observed.
fn unit_extent_claim_error(
    builtin: &str,
    input_dims: &[Dim],
    axis: usize,
    subst: &Subst,
) -> Option<CheckError> {
    match input_dims
        .get(axis)
        .and_then(|d| subst.observe_dim(d).literal_extent())
    {
        Some(1) => None,
        Some(extent) => Some(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "{builtin} requires the operand's extent at axis {axis} to be 1, got \
                 {extent}: {builtin} broadcasts a size-1 axis and cannot replace an \
                 axis that already carries data (spec/05-risc-primitives.md \u{00a7}2.4.1)"
            ),
            vec![],
        )),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check_expand_signature(
    builtin: &'static str,
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    axis_is_dim_name: bool,
    env: &Env,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    // `insert` is this route with one result form instead of two. Everything
    // before the result typing, including what happens to a pending operand,
    // is shared byte for byte.
    let inserts_only = builtin == "insert";
    if arg_tys.len() != 3 && arg_tys.len() != 4 {
        return report_builtin_arity_bare(errors, builtin, "3 or 4 arguments", arg_tys.len());
    }

    let input_ty = type_for_readonly_check(&arg_tys[0], subst);
    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error(_) => return subst.apply(result_ty),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("{builtin} expects tensor input, got {other}"),
                    vec![],
                ),
            );
        }
    };

    let has_spread = input_dims.iter().any(|d| matches!(d, Dim::Rank(_)));

    // chelis#339 named-axis insert (spec/04-type-system.md §4.5.3): when the
    // axis argument is a dimension NAME rather than an integer, the call
    // inserts a new named axis, at the trailing end (3-arg form) or
    // immediately before an existing named anchor (4-arg form). This is the
    // only valid form on a rank-spread operand. `axis_is_dim_name` is
    // scope-discriminated by the caller: a bare var bound in the value
    // environment is a runtime value (issue #259), not a dim name, and falls
    // through to the compile-time-constant rejection below.
    //
    // The form belongs to `insert`. spec/05-risc-primitives.md §2.4: "The
    // named-axis form (`insert(x, new, size)` with a dimension name) and the
    // four-argument anchored form belong to `insert`", and §4.5.3 is written
    // in `insert` throughout. `expand` therefore names the fix rather than
    // adopting a form that would raise the rank it must leave alone.
    if axis_is_dim_name
        && arg_exprs.get(1).and_then(extract_int_for_dim).is_none()
        && let Some(new_name) = arg_exprs.get(1).and_then(symbolic_dim_ref_name)
    {
        if !inserts_only {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "{builtin} takes a positional i32 axis, not the dimension \
                         name `{new_name}`: the named-axis form adds an axis and \
                         belongs to `insert`. Write `insert(x, {new_name}, size)` to \
                         add a named axis, or `{builtin}(x, <i32 axis>, size)` to \
                         broadcast an existing size-1 axis \
                         (spec/05-risc-primitives.md \u{00a7}2.4)"
                    ),
                    vec![],
                ),
            );
        }
        return check_named_expand_signature(
            builtin,
            new_name,
            arg_exprs,
            &input_dims,
            has_spread,
            input_prec,
            result_ty,
            subst,
            errors,
        );
    }

    // From here on the call is the positional concrete-rank form. A fourth
    // (anchor) argument is only meaningful in the named-axis form, which is
    // `insert`'s.
    if arg_exprs.len() == 4 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                if inserts_only {
                    format!(
                        "{builtin} takes a fourth (anchor) argument only in the named-axis \
                         form `{builtin}(x, new, size, anchor)`, where `new` names the \
                         inserted axis (spec/04-type-system.md \u{00a7}4.5.3)"
                    )
                } else {
                    format!(
                        "{builtin} takes exactly three arguments `(x, axis, size)`. The \
                         four-argument anchored form adds a named axis and belongs to \
                         `insert`: write `insert(x, new, size, anchor)` \
                         (spec/05-risc-primitives.md \u{00a7}2.4)"
                    )
                },
                vec![],
            ),
        );
    }
    // A positional index is meaningless at symbolic rank: against a spread
    // there is no fixed position. For `insert`, spec/04-type-system.md §4.5.3
    // says so and offers the named form. `expand` has no named form, and no
    // sentence states its verdict on a spread operand, so it keeps the same
    // rejection and says only what is decided: the axis it broadcasts must be
    // an axis the operand's static rank has. Tracked as a spec gap by
    // chelis#1575.
    if has_spread {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                if inserts_only {
                    format!(
                        "{builtin}: a positional integer axis is only valid on a \
                         concrete-rank operand; on a rank-spread operand, name the \
                         inserted axis (e.g. `{builtin}(x, one, 1)` for a trailing insert, \
                         or `{builtin}(x, c, n, seq)` to insert before the named `seq` \
                         anchor) so the insertion point stays name-anchored \
                         (spec/04-type-system.md \u{00a7}4.5.3)"
                    )
                } else {
                    format!(
                        "{builtin}: a positional integer axis is only valid on a \
                         concrete-rank operand, because the axis it broadcasts must be \
                         one the operand's static rank has; a rank-spread operand has \
                         no fixed position. `{builtin}` has no named-axis form \
                         (spec/05-risc-primitives.md \u{00a7}2.4)"
                    )
                },
                vec![],
            ),
        );
    }

    // Uses `extract_int_for_dim` so a `cast(N, i32)`-wrapped literal axis
    // reaches the non-negative-axis check. Extent folding has its own exact
    // i64 path below.
    let axis = match arg_exprs.get(1).and_then(extract_int_for_dim) {
        Some(axis) if axis >= 0 => axis as usize,
        Some(axis) => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("{builtin} requires non-negative axis, got {axis}"),
                    vec![],
                ),
            );
        }
        // Issue #259: the input is a concrete tensor (past the
        // `Var | Error` guard above), so the output shape is determinable
        // once the insert axis is known. When the axis arg is not a
        // compile-time constant, `extract_int_for_dim` returns `None` and
        // we cannot place the inserted dimension. Pre-fix this arm returned
        // the still-unresolved `Type::Var(out)` from `tensor_expand_to_out`,
        // which leaked downstream and surfaced as a misleading
        // `borrow requires tensor or tensor-carrying input, got ?N`. Emit a
        // targeted diagnostic at the expand call site naming the root cause.
        None => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "{builtin} axis must be a compile-time constant of type i32 for the output \
                     shape to be inferable, got {}",
                        describe_axis_arg(arg_exprs.get(1)),
                    ),
                    vec![format!(
                        "Pass a literal axis (e.g. `{builtin}(x, 0, n)`) or a \
                             `cast(N, i32)` literal. The axis selects where the new \
                             dimension is inserted, so it must be known at compile time."
                    )],
                ),
            );
        }
    };
    // The two operations have different axis ranges, and
    // spec/04-type-system.md §4.7.2 states both: "`expand` requires `axis`
    // within `rank(x)` [...] `insert` admits `axis` in `0..=rank(x)`, so
    // `axis == rank(x)` appends a trailing axis. An axis outside its
    // operation's range is a type error."
    //
    // Diagnose it before consulting an expected result: expected-result
    // propagation may carry an independently wrong rank, but it must not mask
    // the more local invalid-axis reason (chelis#579/#942).
    // `saturating_sub` would let axis 0 through on a rank-0 operand, which has
    // no axis at all, so the two bounds are written as the conditions §4.7.2
    // states rather than as one arithmetic limit.
    let axis_out_of_range = if inserts_only {
        axis > input_dims.len()
    } else {
        axis >= input_dims.len()
    };
    if axis_out_of_range {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                if inserts_only {
                    format!(
                        "{builtin} axis {axis} is out of bounds for rank {} tensor",
                        input_dims.len()
                    )
                } else {
                    format!(
                        "{builtin} axis {axis} is out of bounds for rank {} tensor: \
                         {builtin} broadcasts an existing axis, so its axis must be \
                         within the operand's rank. Use `insert` to add an axis \
                         (spec/04-type-system.md \u{00a7}4.7.2)",
                        input_dims.len()
                    )
                },
                vec![],
            ),
        );
    }
    let size = match arg_exprs
        .get(2)
        .and_then(|expr| fold_static_int_expr(expr, |name| env.static_size_value(name)))
    {
        Some(size) if size >= 0 => Dim::Lit(size),
        Some(size) => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("{builtin} requires non-negative size, got {size}"),
                    vec![],
                ),
            );
        }
        // A non-literal runtime size. spec/04-type-system.md section 4.7.2
        // admits any `i64` size and forbids rejecting an extent because of
        // its provenance (chelis#469), so no spelling is refused here: a
        // parameter, binding, cast, call result or arithmetic size is a
        // fresh runtime extent, and a literal or named claim over it is
        // checked at run time.
        None => {
            // A bare `var` naming a genuine §4.7.2 Form-2 symbolic dim — a
            // declared dim parameter (not a value binding) or a dim carried
            // by an in-scope tensor — stamps the named dim into the output so
            // declared results refer to it by name. Every other materializable
            // spelling (`shape(...)` reads, static arithmetic, `cast`-wrapped,
            // and `let`-bound sizes) defers the output dim slot to
            // the declared return-type / call-context via unification.
            match arg_exprs.get(2).and_then(symbolic_dim_ref_name) {
                Some(name)
                    if env.lookup(name).is_none()
                        || env.tensor_carries_dim_with_subst(name, subst) =>
                {
                    Dim::Name(name.to_string())
                }
                _ => Dim::Wildcard,
            }
        }
    };

    let resolved_result = subst.apply(result_ty);
    let canonical = match resolved_result {
        Type::Tensor(out_dims, out_prec) => {
            if out_prec != input_prec {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::PrecisionMismatch,
                        format!(
                            "{builtin} output precision {} does not match input precision {}",
                            out_prec.name(),
                            input_prec.name()
                        ),
                        vec![],
                    ),
                );
            }
            // One result shape per operation (spec/04-type-system.md §4.7.2:
            // "Each operation has exactly one result shape"), so the declared
            // rank agrees with the operation or the program is rejected.
            let expected_rank = if inserts_only {
                input_dims.len() + 1
            } else {
                input_dims.len()
            };
            if out_dims.len() != expected_rank {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        if inserts_only {
                            format!(
                                "{builtin} output rank {} must equal input rank {} plus one",
                                out_dims.len(),
                                input_dims.len()
                            )
                        } else {
                            format!(
                                "{builtin} output rank {} must equal input rank {}: \
                                 {builtin} broadcasts an existing size-1 axis and leaves \
                                 the rank alone. Use `insert` to add an axis \
                                 (spec/04-type-system.md \u{00a7}4.7.2)",
                                out_dims.len(),
                                input_dims.len()
                            )
                        },
                        vec![],
                    ),
                );
            }
            if inserts_only {
                let mut expected = input_dims.clone();
                expected.insert(axis, size.clone());
                Type::Tensor(expected, input_prec)
            } else {
                if let Some(error) = unit_extent_claim_error(builtin, &input_dims, axis, subst) {
                    return report(errors, error);
                }
                let mut expected = input_dims.clone();
                expected[axis] = size.clone();
                Type::Tensor(expected, input_prec)
            }
        }
        Type::Var(_) => {
            // spec/04-type-system.md §4.7.2: "No result is deferred, no
            // consumer selects between shapes, and no context supplies a
            // default." Each operation's one shape follows from the operand,
            // the axis and the size alone, with nothing to record.
            if inserts_only {
                let mut expected = input_dims.clone();
                expected.insert(axis, size.clone());
                Type::Tensor(expected, input_prec)
            } else {
                if let Some(error) = unit_extent_claim_error(builtin, &input_dims, axis, subst) {
                    return report(errors, error);
                }
                let mut expected = input_dims.clone();
                expected[axis] = size.clone();
                Type::Tensor(expected, input_prec)
            }
        }
        Type::Error(witness) => return propagate(&witness),
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("{builtin} expects tensor output, got {other}"),
                    vec![],
                ),
            );
        }
    };

    if let Err(te) = unify(result_ty, &canonical, subst) {
        return report(errors, te.into());
    }
    subst.apply(&canonical)
}

/// [05-OP-71]: `split_keys(k, n)`'s result extent follows
/// spec/04-type-system.md §4.7.2's rule for `expand` and `insert`. A count
/// that folds to a literal gives that literal extent and a negative one is a
/// type error; any other count gives a fresh runtime extent `*`, which the
/// lowered graph checks for equality wherever a declared or shared extent
/// meets it. The scheme's free result dimension is never the answer: it
/// would take whatever extent the context offers, whatever the count.
pub(super) fn check_split_keys_signature(
    arg_exprs: &[deep::Expr],
    result_ty: &Type,
    env: &Env,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let rows = match arg_exprs
        .get(1)
        .and_then(|expr| fold_static_int_expr(expr, |name| env.static_size_value(name)))
    {
        Some(count) if count >= 0 => Dim::Lit(count),
        Some(count) => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("split_keys requires a non-negative count, got {count} ([05-OP-71])"),
                    vec![],
                ),
            );
        }
        None => Dim::Wildcard,
    };
    let mut dims = match subst.apply(result_ty) {
        Type::Tensor(dims, _) => dims,
        _ => vec![Dim::Wildcard],
    };
    let Some(last) = dims.last_mut() else {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                "split_keys must append a count axis".to_string(),
                vec![],
            ),
        );
    };
    *last = rows;
    let canonical = Type::Tensor(dims, TensorPrec::Concrete(Prim::Key));
    if let Err(error) = unify(result_ty, &canonical, subst) {
        return report(errors, error.into());
    }
    canonical
}

/// The named-axis expand arm (chelis#339, spec/04-type-system.md §4.5.3):
/// insert a new axis named `new_name` into the operand's row form — at the
/// trailing end (3-arg form) or immediately before the named `anchor` axis
/// (4-arg form). Insertion is unitary: the position is an end of the row or
/// fixed by an anchor located uniquely in the operand; everything else is a
/// hard error, never a guessed placement. The inserted dim enters the
/// symbolic output as `Dim::Name`, so declared results refer to it by name
/// and call-site monomorphization carries it through.
#[allow(clippy::too_many_arguments)]
pub(super) fn check_named_expand_signature(
    builtin: &'static str,
    new_name: &str,
    arg_exprs: &[deep::Expr],
    input_dims: &[Dim],
    has_spread: bool,
    input_prec: TensorPrec,
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    // The inserted name must not collide with an existing named axis: a
    // duplicate dim name would make every later by-name lookup (reduction,
    // anchor location) ambiguous.
    if input_dims
        .iter()
        .any(|d| matches!(subst.semantic_dim(d), Dim::Name(n) if n == new_name))
    {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "{builtin}: inserted axis `{new_name}` already names an axis of the operand; \
                 a duplicate dim name would make later by-name axis lookups ambiguous \
                 (spec/04-type-system.md \u{00a7}4.5.3). Pick a fresh name for the inserted \
                 axis."
                ),
                vec![],
            ),
        );
    }

    // The named-insert size must be a non-negative compile-time literal (an
    // `Ni64` literal or `cast(N, i64)`; extent-domain under [05-DIM-1]).
    // A symbolic-dim or runtime i64 size cannot
    // be stamped onto the inserted named dim at lowering: the eval lane has
    // no extent to stage and the C backend would emit an undeclared dim
    // symbol (silent shape-0 output) — both verified failure modes, so the
    // checker rejects the form outright rather than letting a check-clean
    // program break downstream (chelis#339).
    let Some(size) = arg_exprs
        .get(2)
        .and_then(|expr| fold_static_int_expr(expr, |_| None))
    else {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "{builtin}: the named-axis insert form requires a compile-time literal size \
                 (an Ni64 literal or `cast(N, i64)` constant), got {}; the inserted axis's \
                 extent must be stampable onto the new named dim at lowering \
                 (spec/04-type-system.md \u{00a7}4.5.3)",
                    describe_axis_arg(arg_exprs.get(2)),
                ),
                vec![],
            ),
        );
    };
    if size < 0 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("{builtin} requires non-negative size, got {size}"),
                vec![],
            ),
        );
    }

    // Resolve the insertion point: before the unique named anchor (4-arg
    // form) or at the trailing end of the row (3-arg form).
    let insert_at = match arg_exprs.get(3) {
        Some(anchor_expr) => {
            let Some(anchor) = symbolic_dim_ref_name(anchor_expr) else {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{builtin} anchor must name an existing axis of the operand, got {} \
                         (spec/04-type-system.md \u{00a7}4.5.3)",
                            describe_axis_arg(arg_exprs.get(3)),
                        ),
                        vec![],
                    ),
                );
            };
            let hits: Vec<usize> = input_dims
                .iter()
                .enumerate()
                .filter(|(_, d)| matches!(subst.semantic_dim(d), Dim::Name(n) if n == anchor))
                .map(|(i, _)| i)
                .collect();
            match hits.as_slice() {
                [i] => *i,
                [] if has_spread => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{builtin}: rank-spread operand has no named `{anchor}` axis to \
                             anchor the insertion; insertion strictly inside an opaque \
                             spread has no anchor and is rejected \
                             (spec/04-type-system.md \u{00a7}4.5.3)"
                            ),
                            vec![],
                        ),
                    );
                }
                [] => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{builtin} anchor `{anchor}` is not a named axis of the operand: \
                             the named-axis form inserts at the trailing end or immediately \
                             before an existing named anchor (spec/04-type-system.md \
                             \u{00a7}4.5.3)"
                            ),
                            vec![],
                        ),
                    );
                }
                _ => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{builtin}: anchor `{anchor}` is ambiguous; it appears more than \
                             once in the operand shape (spec/04-type-system.md \u{00a7}4.5.3)"
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }
        None => input_dims.len(),
    };

    // Symbolic output: insert into the row form. Surviving dims (spreads
    // included) keep identity and order; the new axis is a named dim.
    let mut out_dims = input_dims.to_vec();
    out_dims.insert(insert_at, Dim::Name(new_name.to_string()));
    let canonical = Type::Tensor(out_dims, input_prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        return report(errors, te.into());
    }
    subst.apply(&canonical)
}

/// Result of peeling nested `List<...>` wrappers from a `to_tensor`
/// argument. Bucket 4b: previously the typer only accepted a single
/// `List<numeric|bool>` and rejected `List<List<f32>>` outright; now
/// we walk down through arbitrarily many `List` heads, count the rank,
/// and require the innermost element to be a numeric or bool prim.
pub(super) enum ToTensorPeel<'a> {
    /// Successfully peeled `rank` `List` layers down to a `Prim`.
    Ok { rank: usize, precision: Prim },
    /// The type `rank` `List` layers in is still a variable. The caller
    /// decides whether that variable's restriction already makes it a scalar
    /// leaf dtype ([05-OP-57]); otherwise the leaf is still pending.
    Pending { rank: usize, element: TypeVar },
    /// Some inner type is an `Error` that already owns its diagnostic.
    Poisoned,
    /// Reached a non-`List`, non-prim leaf — the innermost element is
    /// not numeric or bool, so emit a typed diagnostic.
    BadInner(&'a Type),
    /// The argument is not a `List` at all.
    NotList,
}

pub(super) fn peel_to_tensor_argument(ty: &Type) -> ToTensorPeel<'_> {
    let mut current = ty;
    let mut rank = 0;
    loop {
        match current {
            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                rank += 1;
                current = &args[0];
            }
            Type::Prim(precision) if precision.is_numeric() || matches!(precision, Prim::Bool) => {
                return ToTensorPeel::Ok {
                    rank,
                    precision: *precision,
                };
            }
            Type::Var(element) => {
                return ToTensorPeel::Pending {
                    rank,
                    element: *element,
                };
            }
            Type::Error(_) => return ToTensorPeel::Poisoned,
            other => {
                if rank == 0 {
                    return ToTensorPeel::NotList;
                }
                return ToTensorPeel::BadInner(other);
            }
        }
    }
}

/// Extract an int literal from a Deep expr, recognizing the canonical
/// literal forms (`Atom::Int`, `(lit {type: ...} N)`) and the `neg` app
/// wrapper. Float-in-cast intentionally is not recognized: the spec
/// says integer literals default to `i32` and require explicit
/// notation for other widths, so a float wrapped in a cast to an int
/// dtype is a precision-narrowing operation that the runtime should
/// validate -- not a literal int (round 3 LOW-2 design note).
///
/// The `neg` arm recurses through `extract_int_for_dim` so that
/// `neg(cast(N, i32))` peels both wrappers and resolves to `-N` at
/// infer time (red team round 3 finding R3-MED1). Mutual recursion
/// with `extract_int_for_dim` is bounded: each call strictly reduces
/// the expression depth (peels one wrapper layer).
pub(super) fn extract_int_literal(expr: &deep::Expr) -> Option<i64> {
    if let deep::Expr::Atom(deep::Atom::Int(n), _) = expr {
        return Some(*n);
    }
    // chelis#1107: carrier-preserving read. The `List`-only arms below never
    // matched a stamped `lit`/`app`, so every literal axis and dimension read
    // as "not a compile-time constant" on `check_typed_program` -- an
    // OVER-rejection: `mean(x, 0)` was rejected there and accepted by
    // `check_ir_program`.
    let (tag, _, kids) = stamped_parts(expr)?;
    match tag {
        DeepTag::Lit => kids.first().and_then(|child| match child {
            deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
            _ => None,
        }),
        DeepTag::App => match (kids.first(), kids.get(1)) {
            (Some(func), Some(arg)) if is_builtin_var(func, "neg") => {
                extract_int_for_dim(arg).and_then(i64::checked_neg)
            }
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn is_builtin_var(expr: &deep::Expr, expected: &str) -> bool {
    // chelis#1107: carrier-preserving read. A `List`-only destructure made this
    // return `false` for EVERY stamped `(var {} name)`, so each of its callers
    // -- the `cast`/`neg`/`shape`/`Cons` recognizers in the static shape
    // readers -- silently failed to recognize its form on the typed ingress.
    matches!(stamped_parts(expr), Some((DeepTag::Var, _, kids))
        if kids.first().and_then(symbol_name) == Some(expected))
}
