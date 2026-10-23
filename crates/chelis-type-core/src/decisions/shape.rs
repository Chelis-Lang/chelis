//! Shape and reduction result decisions.

use super::SourceLabel;
use crate::model::{Dim, TensorPrec, Type, TypeVar, VarGen};
use crate::unify::Subst;
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};
use chelis_numeric_core::prim::Prim;

pub fn normalize_static_axis(rank: usize, axis: i64) -> Option<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { rank + axis } else { axis };
    (0..rank).contains(&axis).then_some(axis as usize)
}

pub fn shape_route_result<E: Clone + PartialEq>(
    route: &crate::unify::ShapeRoute<E>,
    operand: &Type<E>,
    slot: SumResultSlot<'_, E>,
    location: Option<SourceLabel>,
    subst: &Subst<E>,
) -> Result<(Type<E>, Option<Type<E>>), String> {
    use crate::unify::ShapeRoute;
    let operand = match operand {
        Type::Ref(inner) if route.normalizes_borrow() => inner.as_ref(),
        other => other,
    };
    match route {
        ShapeRoute::Gather {
            op,
            indices,
            raw_axis,
            updates,
            mode,
        } => {
            let axis = settled_axis(op, operand, *raw_axis, 0)?;
            if let Some(mode) = mode
                && !matches!(mode.as_str(), "replace" | "add")
            {
                return Err(format!("{op} mode must be \"replace\" or \"add\""));
            }
            let result = infer_gather_result_type(op, operand, indices.as_ref(), axis)?;
            match updates {
                Some(_) => Ok((operand.clone(), Some(result))),
                None => Ok((result, None)),
            }
        }
        ShapeRoute::Trace {
            raw_axis1,
            raw_axis2,
        } => {
            let axis1 = settled_axis("trace", operand, *raw_axis1, 0)?;
            let axis2 = settled_axis("trace", operand, *raw_axis2, 1)?;
            Ok((
                infer_trace_result_type(operand, axis1, axis2, slot, location, subst)?,
                None,
            ))
        }
    }
}

fn settled_axis<E>(
    op: &str,
    operand: &Type<E>,
    raw: Option<i64>,
    default: usize,
) -> Result<usize, String> {
    let Type::Tensor(dims, _) = operand else {
        return Ok(default);
    };
    match raw {
        Some(raw) => normalize_static_axis(dims.len(), raw)
            .ok_or_else(|| format!("{op} axis {raw} out of bounds for rank {}", dims.len())),
        None => Ok(default),
    }
}

pub fn infer_gather_result_type<E>(
    op: &str,
    tensor_ty: &Type<E>,
    indices_ty: &Type<E>,
    axis: usize,
) -> Result<Type<E>, String> {
    let Type::Tensor(tensor_dims, tensor_precision) = tensor_ty else {
        return Err(format!("{op} expects tensor input, got {tensor_ty}"));
    };
    let Type::Tensor(index_dims, index_precision) = indices_ty else {
        return Err(format!(
            "{op} expects integer tensor indices, got {indices_ty}"
        ));
    };
    if !index_precision.is_integer() {
        return Err(format!(
            "{op} expects integer tensor indices, got tensor[..., {}]",
            index_precision.name()
        ));
    }
    if axis >= tensor_dims.len() {
        return Err(format!(
            "{op} axis {axis} out of bounds for rank {}",
            tensor_dims.len()
        ));
    }
    let mut out_dims = tensor_dims[..axis].to_vec();
    out_dims.extend(index_dims.clone());
    out_dims.extend_from_slice(&tensor_dims[axis + 1..]);
    Ok(Type::Tensor(out_dims, tensor_precision.clone()))
}

pub fn infer_trace_result_type<E: Clone + PartialEq>(
    tensor_ty: &Type<E>,
    axis1: usize,
    axis2: usize,
    slot: SumResultSlot<'_, E>,
    location: Option<SourceLabel>,
    subst: &Subst<E>,
) -> Result<Type<E>, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("trace expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "trace expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    let out_dims = dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| ((index != axis1) && (index != axis2)).then_some(dim.clone()))
        .collect();
    let precision = default_sum_result_precision("trace", precision, subst)?;
    Ok(publish_sum_result(
        "trace", out_dims, precision, slot, location, subst,
    ))
}

pub fn infer_einsum_result_type<E: Clone + PartialEq>(
    equation: &str,
    left: &Type<E>,
    right: &Type<E>,
    slot: SumResultSlot<'_, E>,
    location: Option<SourceLabel>,
    subst: &Subst<E>,
) -> Result<Type<E>, String> {
    let (Type::Tensor(left_dims, precision), Type::Tensor(right_dims, _)) = (left, right) else {
        return Err(format!(
            "einsum expects two tensor operands, got {left} and {right}"
        ));
    };
    let grammar = || {
        format!(
            "einsum equation `{equation}` must match `[a-z]*,[a-z]*->[a-z]*` \
             (spec/05-risc-primitives.md [05-OP-33])"
        )
    };
    let Some((inputs, output)) = equation.split_once("->") else {
        return Err(grammar());
    };
    let Some((left_labels, right_labels)) = inputs.split_once(',') else {
        return Err(grammar());
    };
    if [left_labels, right_labels, output]
        .iter()
        .any(|labels| !labels.chars().all(|label| label.is_ascii_lowercase()))
    {
        return Err(grammar());
    }
    for (side, labels, dims) in [
        ("left", left_labels, left_dims),
        ("right", right_labels, right_dims),
    ] {
        if dims.iter().any(|dim| matches!(dim, Dim::Rank(_))) {
            return Err(format!(
                "einsum equation `{equation}` fixes the {side} operand's rank at {}, but \
                 the operand has a rank spread, which denotes every rank \
                 ([05-OP-33], [04-INF-6])",
                labels.len()
            ));
        }
        if labels.len() != dims.len() {
            return Err(format!(
                "einsum equation `{equation}` gives the {side} operand {} labels, but it \
                 has rank {} ([05-OP-33])",
                labels.len(),
                dims.len()
            ));
        }
    }
    let mut dims = Vec::with_capacity(output.len());
    for (index, label) in output.chars().enumerate() {
        if output.chars().take(index).any(|earlier| earlier == label) {
            return Err(format!(
                "einsum output label `{label}` must occur exactly once in `{equation}` \
                 ([05-OP-33])"
            ));
        }
        let dim = left_labels
            .chars()
            .position(|candidate| candidate == label)
            .map(|position| &left_dims[position])
            .or_else(|| {
                right_labels
                    .chars()
                    .position(|candidate| candidate == label)
                    .map(|position| &right_dims[position])
            })
            .ok_or_else(|| {
                format!(
                    "einsum output label `{label}` must occur in an input of `{equation}` \
                     ([05-OP-33])"
                )
            })?;
        dims.push(dim.clone());
    }
    let precision = default_sum_result_precision("einsum", precision, subst)?;
    Ok(publish_sum_result(
        "einsum", dims, precision, slot, location, subst,
    ))
}

pub enum SumResultPrecision {
    Decided(TensorPrec),
    Pending(TypeVar),
}

pub fn default_sum_result_precision<E: Clone + PartialEq>(
    op: &str,
    precision: &TensorPrec,
    subst: &Subst<E>,
) -> Result<SumResultPrecision, String> {
    match precision {
        TensorPrec::Concrete(prim) => settled_sum_result_precision::<E>(op, &Type::Prim(*prim))
            .map(|result| SumResultPrecision::Decided(TensorPrec::Concrete(result))),
        TensorPrec::Var(variable) => Ok(match bound_sum_result_precision(op, *variable, subst) {
            Ok(result) => SumResultPrecision::Decided(result),
            Err(_) => SumResultPrecision::Pending(*variable),
        }),
    }
}

pub enum SumResultSlot<'a, E> {
    Fresh(&'a mut VarGen),
    Existing(&'a Type<E>),
}

pub fn publish_sum_result<E: Clone + PartialEq>(
    op: &str,
    dims: Vec<Dim>,
    precision: SumResultPrecision,
    slot: SumResultSlot<'_, E>,
    location: Option<SourceLabel>,
    subst: &Subst<E>,
) -> Type<E> {
    match precision {
        SumResultPrecision::Decided(precision) => Type::Tensor(dims, precision),
        SumResultPrecision::Pending(variable) => {
            let published = match slot {
                SumResultSlot::Fresh(vg) => {
                    Type::Tensor(dims.clone(), TensorPrec::Var(vg.fresh_tvar()))
                }
                SumResultSlot::Existing(published) => published.clone(),
            };
            subst.record_deferred_tensor_operand(
                variable,
                crate::unify::DeferredOperandGate::SumResult {
                    op: op.to_string(),
                    dims,
                    result: Box::new(published.clone()),
                    location,
                },
            );
            published
        }
    }
}

pub fn settled_sum_result_precision<E>(op: &str, resolved: &Type<E>) -> Result<Prim, String> {
    match resolved {
        Type::Prim(prim) => prim
            .default_reduce_sum_result_precision()
            .map_err(|message| {
                format!(
                    "{op} expects a tensor precision with a default sum accumulator, so that its \
                 result sum_result(p, default(p)) exists (spec/04-type-system.md §5.7.1), \
                 got {}; {message}",
                    prim.name()
                )
            }),
        other => Err(format!(
            "{op} expects a tensor precision with a default sum accumulator, so that its \
             result sum_result(p, default(p)) exists (spec/04-type-system.md §5.7.1), \
             got {other}"
        )),
    }
}

pub fn bound_sum_result_precision<E: Clone + PartialEq>(
    op: &str,
    variable: TypeVar,
    subst: &Subst<E>,
) -> Result<TensorPrec, String> {
    let restriction = subst.tvar_restriction(variable);
    let members = Prim::ACTIVE_FLOATS
        .into_iter()
        .chain(Prim::ACTIVE_INTEGERS)
        .chain([Prim::Bool])
        .filter(|prim| restriction.is_none_or(|bound| bound.admits(*prim)))
        .collect::<Vec<_>>();
    let results = members
        .iter()
        .map(|member| member.default_reduce_sum_result_precision().ok())
        .collect::<Vec<_>>();
    if members
        .iter()
        .zip(&results)
        .all(|(member, result)| *result == Some(*member))
    {
        return Ok(TensorPrec::Var(variable));
    }
    if let Some((Some(first), rest)) = results.split_first()
        && rest.iter().all(|result| *result == Some(*first))
    {
        return Ok(TensorPrec::Concrete(*first));
    }
    let bound = match restriction {
        Some(bound) => format!("its bound `{}`", bound.bound_spelling()),
        None => "an unbounded variable".to_string(),
    };
    Err(format!(
        "{op} over a tensor whose precision is a type variable has no single result dtype, \
         because spec/04 §5.7.1 sums an i8 or i16 operand in i32 and returns that i32 total \
         but returns i32, i64 and the floats at their own dtype, so sum_result(p, default(p)) \
         is not one dtype across the dtypes {bound} admits ([04-INF-6]); bound the variable to dtypes that \
         share one result (`Float`, `{{i32, i64}}`, or `{{i8, i16}}` with an i32 result), \
         or cast the operand to a concrete dtype and declare the widened result"
    ))
}

pub fn sum_result_widening_note<E>(
    op: Option<&str>,
    left: &Type<E>,
    right: &Type<E>,
) -> Option<String> {
    fn result_prim<E>(ty: &Type<E>) -> Option<Prim> {
        match ty {
            Type::Prim(prim) | Type::Tensor(_, TensorPrec::Concrete(prim)) => Some(*prim),
            Type::Fn(_, result) => result_prim(result),
            _ => None,
        }
    }
    let (left, right) = (result_prim(left)?, result_prim(right)?);
    let widened = |operand: Prim, result: Prim| {
        operand.default_reduce_sum_result_precision() == Ok(result) && operand != result
    };
    let (operand, result) = if widened(left, right) {
        (left.name(), right.name())
    } else if widened(right, left) {
        (right.name(), left.name())
    } else {
        return None;
    };
    Some(match op {
        Some(op @ ("sum" | "einsum")) => format!(
            "`{op}` over {operand} returns {result}, because spec/04 §5.7.1 sums {operand} in \
             {result} and returns that {result} total; declare the result as {result} (or pass \
             `accumulator=i64` and declare i64), or narrow it explicitly with \
             `cast(..., {operand})`"
        ),
        Some(op) => format!(
            "`{op}` over {operand} returns {result}, because spec/04 §5.7.1 sums {operand} in \
             {result} and returns that {result} total; declare the result as {result}, or \
             narrow it explicitly with `cast(..., {operand})`"
        ),
        None => format!(
            "if the {result} value is the result of `sum`, `cumsum`, `trace` or `einsum` over \
             {operand}, spec/04 §5.7.1 widened it, because those operations sum {operand} in \
             {result} and return that {result} total; declare the result as {result} (or, for \
             `sum` and `einsum`, pass `accumulator=i64` and declare i64), or narrow it \
             explicitly with `cast(..., {operand})`"
        ),
    })
}
pub fn explicit_accumulator_result(
    operation: &str,
    operand: Prim,
    accumulator: Prim,
) -> Result<Prim, String> {
    use chelis_numeric_core::prim::Prim::*;
    let permitted: &[Prim] = match (operation, operand) {
        (_, Bf16 | F16 | F32) => &[F32, F64],
        (_, F64) => &[F64],
        ("matmul", _) => &[],
        (_, Int8 | Int16 | Int32) => &[Int32, Int64],
        (_, Int64) => &[Int64],
        _ => &[],
    };
    if permitted.contains(&accumulator) {
        return Ok(match (operation, operand) {
            ("matmul", _) | (_, Bf16 | F16) => operand,
            _ => accumulator,
        });
    }
    let names = permitted
        .iter()
        .map(|prim| format!("`accumulator={}`", prim.name()))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(format!(
            "`{operation}` over {} admits no accumulator (spec/04 §5.7.1)",
            operand.name()
        ));
    }
    let reason = if accumulator.is_integer() != operand.is_integer()
        || accumulator.is_float() != operand.is_float()
    {
        "an accumulator has its operand's numeric kind"
    } else {
        "an accumulator is at least as wide as its operand and its default"
    };
    Err(format!(
        "`{operation}` over {} does not admit `accumulator={}`, because {reason} \
         (spec/04 §5.7.1); omit the argument to accumulate in the default, or write {}",
        operand.name(),
        accumulator.name(),
        names.join(" or ")
    ))
}

pub fn sum_result_bound_note(admits_small_integer: bool, prim: Prim) -> Option<String> {
    (admits_small_integer && prim == Prim::Int32).then(|| {
        "if this i32 is the result of `sum`, `cumsum`, `trace` or `einsum` over i8 or i16, \
         spec/04 §5.7.1 widened it, because those operations sum i8 and i16 in i32 and return \
         that i32 total; declare that result as i32 (or, for `sum` and `einsum`, pass \
         `accumulator=i64` and declare i64), or bound the binder to dtypes that share one \
         sum result"
            .to_string()
    })
}
