//! Static-value validation.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum StaticValue {
    Unknown,
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<StaticValue>),
    Tensor(StaticTensor),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct StaticTensor {
    shape: Vec<usize>,
    int_values: Option<Vec<i64>>,
}

pub(super) fn bind_fn_params_unknown(fn_list: &deep::List, env: &mut HashMap<String, StaticValue>) {
    let Some(params_expr) = children(fn_list).first() else {
        return;
    };
    // chelis#1107 amendment: carrier-preserving read.
    let Some((DeepTag::Params, _, param_entries)) = stamped_parts(params_expr) else {
        return;
    };
    for param in param_entries {
        match param {
            deep::Expr::Atom(deep::Atom::Name(name), _) => {
                env.insert(name.clone(), StaticValue::Unknown);
            }
            deep::Expr::List(param_list, _) => {
                if let Some(name) = param_list.elements.first().and_then(symbol_name) {
                    env.insert(name.to_string(), StaticValue::Unknown);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn literal_static_value(expr: &deep::Expr) -> StaticValue {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(value), _) => StaticValue::Int(*value),
        deep::Expr::Atom(deep::Atom::Float(value), _) => StaticValue::Float(*value),
        deep::Expr::Atom(deep::Atom::Bool(value), _) => StaticValue::Bool(*value),
        deep::Expr::Atom(deep::Atom::Str(value), _) => StaticValue::String(value.clone()),
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Lit) => children(list)
            .first()
            .map(literal_static_value)
            .unwrap_or(StaticValue::Unknown),
        _ => StaticValue::Unknown,
    }
}

pub(super) fn app_builtin_name(expr: &deep::Expr) -> Option<&str> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::Var {
        return None;
    }
    kids.first().and_then(symbol_name)
}

pub(super) fn validate_static_builtin_application(
    name: &str,
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    match name {
        "pad_sequences" => static_pad_sequences(args),
        "to_tensor" => static_to_tensor(args),
        "concat" => static_concat(args, expr, errors),
        "split" => static_split(args, expr, errors),
        "gather" => static_gather(args, expr, errors),
        "scatter" => static_scatter(args, expr, errors),
        "scatter_replace" => static_scatter_replace(args, expr, errors),
        "clamp" => static_clamp(args, expr, errors),
        "einsum" => static_einsum(args, expr, errors),
        _ => StaticValue::Unknown,
    }
}

pub(super) fn static_pad_sequences(args: &[StaticValue]) -> StaticValue {
    let Some(StaticValue::List(rows)) = args.first() else {
        return StaticValue::Unknown;
    };
    let mut width = 0usize;
    for row in rows {
        let StaticValue::List(items) = row else {
            return StaticValue::Unknown;
        };
        width = width.max(items.len());
    }
    StaticValue::Tensor(StaticTensor {
        shape: vec![rows.len(), width],
        int_values: None,
    })
}

pub(super) fn static_to_tensor(args: &[StaticValue]) -> StaticValue {
    let Some(StaticValue::List(items)) = args.first() else {
        return StaticValue::Unknown;
    };
    let mut ints = Vec::with_capacity(items.len());
    for item in items {
        match item {
            StaticValue::Int(value) => ints.push(*value),
            StaticValue::Float(_) | StaticValue::Bool(_) => {
                return StaticValue::Tensor(StaticTensor {
                    shape: vec![items.len()],
                    int_values: None,
                });
            }
            _ => return StaticValue::Unknown,
        }
    }
    StaticValue::Tensor(StaticTensor {
        shape: vec![items.len()],
        int_values: Some(ints),
    })
}

pub(super) fn static_concat(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (Some(StaticValue::List(parts)), Some(StaticValue::Int(axis))) =
        (args.first(), args.get(1))
    else {
        return StaticValue::Unknown;
    };
    if *axis < 0 {
        return StaticValue::Unknown;
    }
    let tensors = parts
        .iter()
        .map(|value| match value {
            StaticValue::Tensor(tensor) => Some(tensor.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let Some(tensors) = tensors else {
        return StaticValue::Unknown;
    };
    let Some(first) = tensors.first() else {
        return StaticValue::Unknown;
    };
    let axis = *axis as usize;
    if axis >= first.shape.len() {
        return StaticValue::Unknown;
    }
    let mut shape = first.shape.clone();
    let mut axis_total = shape[axis];
    for tensor in tensors.iter().skip(1) {
        if tensor.shape.len() != shape.len() {
            push_static_runtime_error(
                expr,
                errors,
                "concat expects matching tensor rank".to_string(),
            );
            return StaticValue::Unknown;
        }
        for (dim, expected_extent) in shape.iter().enumerate() {
            if dim != axis && tensor.shape[dim] != *expected_extent {
                push_static_runtime_error(
                    expr,
                    errors,
                    "concat expects matching non-concatenated axes".to_string(),
                );
                return StaticValue::Unknown;
            }
        }
        axis_total += tensor.shape[axis];
    }
    shape[axis] = axis_total;
    StaticValue::Tensor(StaticTensor {
        shape,
        int_values: None,
    })
}

pub(super) fn static_split(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(tensor)),
        Some(StaticValue::Int(axis)),
        Some(StaticValue::List(sizes)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    if *axis < 0 {
        return StaticValue::Unknown;
    }
    let axis = *axis as usize;
    if axis >= tensor.shape.len() {
        return StaticValue::Unknown;
    }
    let Some(size_values) = sizes
        .iter()
        .map(|value| match value {
            StaticValue::Int(size) if *size >= 0 => Some(*size as usize),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return StaticValue::Unknown;
    };
    if size_values.iter().sum::<usize>() != tensor.shape[axis] {
        push_static_runtime_error(
            expr,
            errors,
            "split sizes must sum to the selected axis extent".to_string(),
        );
    }
    StaticValue::Unknown
}

pub(super) fn static_gather(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(tensor)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Int(axis)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(tensor.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    if let Some(index_values) = &indices.int_values {
        for value in index_values {
            if *value < 0 || *value >= tensor.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("gather index {value} out of bounds"),
                );
                return StaticValue::Unknown;
            }
        }
    }
    StaticValue::Tensor(StaticTensor {
        shape: gather_result_shape(&tensor.shape, &indices.shape, axis),
        int_values: None,
    })
}

pub(super) fn static_scatter(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(base)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Tensor(updates)),
        Some(StaticValue::Int(axis)),
        Some(StaticValue::String(_mode)),
    ) = (
        args.first(),
        args.get(1),
        args.get(2),
        args.get(3),
        args.get(4),
    )
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(base.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    let expected_updates = gather_result_shape(&base.shape, &indices.shape, axis);
    if updates.shape != expected_updates {
        push_static_runtime_error(
            expr,
            errors,
            "scatter updates must match gathered tensor shape and precision".to_string(),
        );
        return StaticValue::Unknown;
    }
    if let Some(index_values) = &indices.int_values {
        for linear in 0..updates_shape_numel(&updates.shape) {
            let update_index = unravel_index(linear, &updates.shape);
            let gather_index = update_index[axis..axis + indices.shape.len()].to_vec();
            let gather_linear = ravel_index(&gather_index, &indices.shape);
            let gathered = index_values[gather_linear];
            if gathered < 0 || gathered >= base.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("scatter index {gathered} out of bounds"),
                );
                return StaticValue::Unknown;
            }
        }
    }
    StaticValue::Tensor(base.clone())
}

/// Static check for the tensor-lane `scatter_replace(base, indices,
/// updates, axis)` builtin. Mirrors `static_scatter` with `mode ==
/// "replace"` semantics: validates the updates-shape contract,
/// rejects out-of-bounds indices when statically knowable, and permits
/// duplicate target indices under [05-SPARSE-2]'s deterministic
/// last-write-wins rule. Differs from `static_scatter` in that there is no
/// `mode` argument.
pub(super) fn static_scatter_replace(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(base)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Tensor(updates)),
        Some(StaticValue::Int(axis)),
    ) = (args.first(), args.get(1), args.get(2), args.get(3))
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(base.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    let expected_updates = gather_result_shape(&base.shape, &indices.shape, axis);
    if updates.shape != expected_updates {
        push_static_runtime_error(
            expr,
            errors,
            "scatter_replace updates must match gathered tensor shape and precision".to_string(),
        );
        return StaticValue::Unknown;
    }
    if let Some(index_values) = &indices.int_values {
        for linear in 0..updates_shape_numel(&updates.shape) {
            let update_index = unravel_index(linear, &updates.shape);
            let gather_index = update_index[axis..axis + indices.shape.len()].to_vec();
            let gather_linear = ravel_index(&gather_index, &indices.shape);
            let gathered = index_values[gather_linear];
            if gathered < 0 || gathered >= base.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("scatter_replace index {gathered} out of bounds"),
                );
                return StaticValue::Unknown;
            }
        }
    }
    StaticValue::Tensor(base.clone())
}

pub(super) fn static_clamp(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(input)),
        Some(StaticValue::Tensor(low)),
        Some(StaticValue::Tensor(high)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let low_ok = low.shape.is_empty() || low.shape == input.shape;
    let high_ok = high.shape.is_empty() || high.shape == input.shape;
    if !low_ok || !high_ok {
        push_static_runtime_error(
            expr,
            errors,
            "clamp expects scalar bounds or matching-shape tensor bounds".to_string(),
        );
        return StaticValue::Unknown;
    }
    StaticValue::Tensor(input.clone())
}

pub(super) fn static_einsum(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    let (
        Some(StaticValue::String(equation)),
        Some(StaticValue::Tensor(lhs)),
        Some(StaticValue::Tensor(rhs)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let Some((inputs, output)) = equation.split_once("->") else {
        return StaticValue::Unknown;
    };
    let mut input_groups = inputs.split(',');
    let lhs_labels = input_groups
        .next()
        .unwrap_or_default()
        .chars()
        .collect::<Vec<_>>();
    let rhs_labels = input_groups
        .next()
        .unwrap_or_default()
        .chars()
        .collect::<Vec<_>>();
    if input_groups.next().is_some()
        || lhs_labels.len() != lhs.shape.len()
        || rhs_labels.len() != rhs.shape.len()
    {
        return StaticValue::Unknown;
    }
    let mut extents = HashMap::<char, usize>::new();
    for (label, extent) in lhs_labels.iter().zip(&lhs.shape) {
        if let Some(existing) = extents.insert(*label, *extent)
            && existing != *extent
        {
            push_static_runtime_error(
                expr,
                errors,
                format!("einsum label `{label}` has inconsistent extents"),
            );
            return StaticValue::Unknown;
        }
    }
    for (label, extent) in rhs_labels.iter().zip(&rhs.shape) {
        if let Some(existing) = extents.insert(*label, *extent)
            && existing != *extent
        {
            push_static_runtime_error(
                expr,
                errors,
                format!("einsum label `{label}` has inconsistent extents"),
            );
            return StaticValue::Unknown;
        }
    }
    let mut out_shape = Vec::new();
    for label in output.chars() {
        let Some(extent) = extents.get(&label).copied() else {
            return StaticValue::Unknown;
        };
        out_shape.push(extent);
    }
    StaticValue::Tensor(StaticTensor {
        shape: out_shape,
        int_values: None,
    })
}

pub(super) fn normalize_static_axis(rank: usize, axis: i64) -> Option<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { rank + axis } else { axis };
    (0..rank).contains(&axis).then_some(axis as usize)
}

/// Reject an axis argument whose resolved type is not `int32`.
///
/// An axis names a rank position and is int32 in every enforced surface
/// (`sum`, `permute`, `shape`). chelis#1113 owns the numbered-atom
/// classification; until it lands this keeps the acceptance closed so
/// no axis-taking builtin silently admits an int64 axis while `sum`
/// rejects one. `Var` and `Error` pass through: an axis that is still
/// unresolved carries no dtype to judge, and one that already failed
/// must not produce a second diagnostic for the same cause.
///
/// The `Err` value is the witness-carrying `Type::Error` from `report`
/// (chelis#731 §C3), so a caller returning `Type` propagates it with a
/// plain `return` rather than minting a fresh one.
pub(super) fn reject_non_int32_axis(
    op: &str,
    axis_ty: &Type,
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Result<(), Type> {
    match axis_ty {
        Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error(_) => Ok(()),
        other => Err(report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{op} expects int32 axis, got {other}"),
                ),
                vec![],
            ),
        )),
    }
}

/// Enforce every axis slot registered for `op` against already-inferred
/// argument types. The registration is the class mechanism for [05-DIM-3]:
/// fixed and variadic axis layouts share one dtype gate, while each
/// operation keeps its own value/rank checks.
pub(super) fn enforce_registered_axis_dtypes(
    op: &str,
    arg_tys: &[Type],
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Result<(), Type> {
    let Some(layout) = builtins::axis_argument_layout(op) else {
        return Ok(());
    };
    match layout {
        builtins::AxisArgumentLayout::NoAxes => {}
        builtins::AxisArgumentLayout::Fixed(slots) => {
            for &slot in slots {
                if let Some(axis_ty) = arg_tys.get(slot) {
                    reject_non_int32_axis(op, axis_ty, list, errors)?;
                }
            }
        }
        builtins::AxisArgumentLayout::VariadicFrom(first) => {
            for axis_ty in arg_tys.iter().skip(first) {
                reject_non_int32_axis(op, axis_ty, list, errors)?;
            }
        }
    }
    Ok(())
}

/// Resolve one member of a two-axis builtin (`trace`, `diagonal`)
/// against the operand `tensor_ty`. Like [`resolve_builtin_axis`], a
/// negative literal indexes from the end; a still-out-of-range axis
/// pushes a diagnostic and returns `Err(())`. When the axis argument
/// is absent (not a literal) the historical positional `default` is
/// used so the prior `unwrap_or(0)` / `unwrap_or(1)` behavior is
/// preserved for the no-arg case.
///
/// `axis_ty` is the member's resolved (subst-applied) type and is
/// screened by [`reject_non_int32_axis`] before extraction. Without
/// that screen the fallback above is a fail-open: a non-integer axis
/// extracts to `None` and silently becomes `default`, so
/// `diagonal(m, 9.0, 0)` reported "axes 0 and 0" for an axis the
/// caller never wrote, and a string axis checked clean.
pub(super) fn resolve_axis_pair_member(
    op: &str,
    axis_expr: Option<&deep::Expr>,
    axis_ty: &Type,
    tensor_ty: &Type,
    default: usize,
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Result<usize, Type> {
    reject_non_int32_axis(op, axis_ty, list, errors)?;
    // Issue #216: use the cast-aware extractor so `cast(N, int32)`-wrapped
    // axis literals trip the infer-time bounds check instead of slipping
    // through to host-runtime defense-in-depth.
    // chelis#731 §C3: the out-of-bounds `Err` now carries the
    // witness-`Type::Error` from `report`, so the caller propagates it.
    match axis_expr.and_then(extract_int_for_dim) {
        Some(raw) => match tensor_ty {
            Type::Tensor(dims, _) => match normalize_static_axis(dims.len(), raw) {
                Some(axis) => Ok(axis),
                None => Err(report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!("{op} axis {raw} out of bounds for rank {}", dims.len()),
                        ),
                        vec![],
                    ),
                )),
            },
            // Non-tensor operand: keep a non-negative literal verbatim
            // and let the downstream shape checker surface the real
            // mismatch.
            _ => Ok(if raw >= 0 { raw as usize } else { default }),
        },
        None => Ok(default),
    }
}

/// Resolve the axis argument of an axis-taking builtin against the
/// operand `tensor_ty`, normalizing a negative literal to index from
/// the end (`-1` is the last axis). On an axis still out of range
/// after normalization, push an out-of-bounds diagnostic and return
/// `None` so the caller bails to `Type::Error`.
///
/// When the operand is not a concrete tensor or the axis is not a
/// literal, falls back to the prior lenient behavior: a non-negative
/// literal is taken verbatim, anything else defaults to `0` and the
/// downstream shape checker surfaces any real mismatch. The negative
/// normalization itself only needs the operand rank, which a
/// `Type::Tensor` always carries.
/// Resolve a builtin's axis argument. On the out-of-bounds failure path it
/// reports a diagnostic and returns the witness-carrying `Type::Error` as the
/// `Err` value (chelis#731 §C3), so the caller propagates it with a plain
/// `return` rather than minting a fresh `Type::Error`.
///
/// `axis_ty` is the axis argument's resolved (subst-applied) type,
/// screened by [`reject_non_int32_axis`] before extraction so no caller
/// silently admits an int64 axis the way `cumsum`/`concat` once did
/// while `sum` rejected one.
pub(super) fn resolve_builtin_axis(
    op: &str,
    axis_expr: Option<&deep::Expr>,
    axis_ty: &Type,
    tensor_ty: &Type,
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Result<usize, Type> {
    reject_non_int32_axis(op, axis_ty, list, errors)?;
    // Issue #216: cast-aware extractor; see `resolve_axis_pair_member`.
    let raw_axis = axis_expr.and_then(extract_int_for_dim);
    match (tensor_ty, raw_axis) {
        (Type::Tensor(dims, _), Some(raw)) => match normalize_static_axis(dims.len(), raw) {
            Some(axis) => Ok(axis),
            None => Err(report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("{op} axis {raw} out of bounds for rank {}", dims.len()),
                    ),
                    vec![],
                ),
            )),
        },
        _ => Ok(raw_axis
            .filter(|raw| *raw >= 0)
            .map(|raw| raw as usize)
            .unwrap_or(0)),
    }
}

pub(super) fn gather_result_shape(base: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    let mut shape = Vec::with_capacity(base.len().saturating_sub(1) + indices.len());
    shape.extend_from_slice(&base[..axis]);
    shape.extend_from_slice(indices);
    shape.extend_from_slice(&base[axis + 1..]);
    shape
}

pub(super) fn updates_shape_numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
}

pub(super) fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        out[dim] = linear % shape[dim];
        linear /= shape[dim];
    }
    out
}

pub(super) fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (index, extent) in indices.iter().zip(shape.iter()).rev() {
        flat += index * stride;
        stride *= extent;
    }
    flat
}

pub(super) fn push_static_runtime_error(
    expr: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
    message: String,
) {
    errors.push(CheckError::new(
        CheckErrorKind::Other,
        with_macro_provenance(expr, message),
        vec![],
    ));
}
