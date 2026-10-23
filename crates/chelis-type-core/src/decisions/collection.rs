//! Collection decisions with explicit call evidence.

use super::shape::normalize_static_axis;
use super::{DecisionFailure, DecisionFailureKind};
use crate::model::{CollectionConstraint, Dim, ResultConstraint, TensorPrec, Type};
use crate::unify::{Subst, TypeError, unify};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use chelis_numeric_core::prim::Prim;

/// Stored-value equality rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateRule {
    Append,
    Concat,
    DictInsert,
    DictMerge,
    Fold,
    Scan,
}

impl AggregateRule {
    pub fn decide<E: Clone + PartialEq>(
        self,
        operands: &[Type<E>],
        result: &Type<E>,
        subst: &Subst<E>,
    ) -> Result<Option<ResultConstraint<E>>, Box<DecisionFailure>> {
        let arity = match self {
            AggregateRule::Append
            | AggregateRule::Concat
            | AggregateRule::DictMerge
            | AggregateRule::Fold
            | AggregateRule::Scan => 2,
            AggregateRule::DictInsert => 3,
        };
        let name = match self {
            AggregateRule::Append => "append",
            AggregateRule::Concat => "concat",
            AggregateRule::DictInsert => "dict_insert",
            AggregateRule::DictMerge => "dict_merge",
            AggregateRule::Fold => "fold",
            AggregateRule::Scan => "scan",
        };
        if operands.len() != arity {
            return Err(Box::new(DecisionFailure::new(
                DecisionFailureKind::TypeMismatch,
                format!("{name} expects {arity} arguments, got {}", operands.len()),
                vec![],
            )));
        }
        let operands = operands
            .iter()
            .map(|ty| subst.apply(ty))
            .collect::<Vec<_>>();
        if operands.iter().any(|ty| matches!(ty, Type::Error(_))) {
            return Ok(None);
        }
        let inputs = match (self, operands.as_slice()) {
            (AggregateRule::Fold | AggregateRule::Scan, [initial, callback_result]) => {
                vec![initial.clone(), callback_result.clone()]
            }
            (AggregateRule::Append, [list @ Type::Adt(name, args), value])
                if name == "List" && args.len() == 1 =>
            {
                vec![
                    list.clone(),
                    Type::adt("List".to_string(), vec![value.clone()]),
                ]
            }
            (AggregateRule::Concat, [left @ Type::Adt(a, aa), right @ Type::Adt(b, ba)])
                if a == "List" && b == "List" && aa.len() == 1 && ba.len() == 1 =>
            {
                vec![left.clone(), right.clone()]
            }
            (AggregateRule::DictInsert, [dict @ Type::Adt(name, args), key, value])
                if name == "Dict" && args.len() == 2 =>
            {
                vec![
                    dict.clone(),
                    Type::adt("Dict".to_string(), vec![key.clone(), value.clone()]),
                ]
            }
            (AggregateRule::DictMerge, [left @ Type::Adt(a, aa), right @ Type::Adt(b, ba)])
                if a == "Dict" && b == "Dict" && aa.len() == 2 && ba.len() == 2 =>
            {
                vec![left.clone(), right.clone()]
            }
            (_, values) if values.iter().any(|ty| matches!(ty, Type::Var(_))) => return Ok(None),
            _ => {
                let message = match (self, operands.as_slice()) {
                    (AggregateRule::Append, [list, _]) => {
                        format!("append expects List input, got {list}")
                    }
                    (AggregateRule::Concat, [left, right]) => {
                        format!("concat expects matching List inputs, got {left} and {right}")
                    }
                    (AggregateRule::DictInsert, [dict, key, value]) => format!(
                        "dict_insert expects Dict[K, V], K, and V, got {dict}, {key}, and {value}"
                    ),
                    (AggregateRule::DictMerge, [left, right]) => {
                        format!("dict_merge expects matching Dict inputs, got {left} and {right}")
                    }
                    _ => format!(
                        "{name} expects compatible collection operands; got {}",
                        operands
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                };
                return Err(Box::new(DecisionFailure::new(
                    DecisionFailureKind::TypeMismatch,
                    message,
                    vec![],
                )));
            }
        };
        let mut compatibility = subst.clone();
        for input in &inputs[1..] {
            unify(&inputs[0], input, &mut compatibility)
                .map_err(|error| {
                    let message = match self {
                    AggregateRule::Fold | AggregateRule::Scan => format!("{name} expects a callback whose accumulator/result type matches the initial accumulator; {}", error.message),
                    AggregateRule::Append => format!("append expects a value of the list's element type; {}", error.message),
                    AggregateRule::Concat => format!("concat expects matching List inputs; {}", error.message),
                    AggregateRule::DictInsert | AggregateRule::DictMerge => format!("{name} requires matching stored value types; {}", error.message),
                    };
                    let mut diagnostic: DecisionFailure = error.into();
                    diagnostic.original_message =
                        Some(core::mem::replace(&mut diagnostic.message, message));
                    Box::new(diagnostic)
                })?;
        }
        Ok(Some(ResultConstraint::Join {
            inputs,
            result: result.clone(),
        }))
    }
}

pub enum CollectionDecision<E> {
    ResultOrigin(ResultConstraint<E>),
    RequiredEquality {
        actual: Type<E>,
        expected: Type<E>,
    },
    ProducedResult {
        declared: Type<E>,
        produced: Type<E>,
    },
}

impl<E: Clone + PartialEq> CollectionDecision<E> {
    pub fn publish(self, subst: &mut Subst<E>) -> Result<(), TypeError> {
        match self {
            Self::ResultOrigin(equation) => {
                subst.record_result_constraint(equation);
                Ok(())
            }
            Self::RequiredEquality { actual, expected } => unify(&actual, &expected, subst),
            Self::ProducedResult { declared, produced } => unify(&declared, &produced, subst),
        }
    }

    pub fn produced_result(&self) -> Option<Type<E>> {
        match self {
            Self::ProducedResult { produced, .. } => Some(produced.clone()),
            Self::ResultOrigin(_) | Self::RequiredEquality { .. } => None,
        }
    }
}

pub fn decide_collection_constraint<E: Clone + PartialEq>(
    constraint: &CollectionConstraint<E>,
    evidence: Option<&CollectionCallEvidence>,
    subst: &Subst<E>,
) -> Result<Option<CollectionDecision<E>>, String> {
    let result = constraint.result().clone();
    let joined = |inputs| {
        Some(CollectionDecision::ResultOrigin(ResultConstraint::Join {
            inputs,
            result: result.clone(),
        }))
    };
    let applied = constraint.map_types(|ty| subst.apply(ty));
    if applied
        .operands()
        .iter()
        .any(|ty| matches!(ty, Type::Error(_)))
    {
        return Ok(None);
    }
    match &applied {
        CollectionConstraint::FloatClassify { kind, operand, .. } => {
            let operand = match operand {
                Type::Ref(inner) => inner.as_ref(),
                other => other,
            };
            let expected = match operand {
                Type::Tensor(dims, _) => {
                    Type::Tensor(dims.clone(), TensorPrec::Concrete(Prim::Bool))
                }
                Type::Prim(p) if p.is_float() => Type::Prim(Prim::Bool),
                Type::Var(v)
                    if subst
                        .tvar_restriction(*v)
                        .is_some_and(|bound| !bound.is_value_constraint()) =>
                {
                    Type::Prim(Prim::Bool)
                }
                Type::Var(_) | Type::Error(_) => return Ok(None),
                other => {
                    return Err(format!(
                        "{} expects a float scalar or tensor, got {other} ({})",
                        kind.name(),
                        kind.atom()
                    ));
                }
            };
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: result.clone(),
                expected,
            }))
        }

        CollectionConstraint::KeyFromSeed {
            operand: Type::Var(_),
            ..
        }
        | CollectionConstraint::SplitKey {
            operand: Type::Var(_),
            ..
        }
        | CollectionConstraint::SplitKeys {
            operand: Type::Var(_),
            ..
        }
        | CollectionConstraint::FoldIn {
            operand: Type::Var(_),
            ..
        } => Ok(None),
        CollectionConstraint::KeyFromSeed { operand, .. } => {
            key_operation_surface(operand, Prim::Int64).map(|ty| {
                Some(CollectionDecision::RequiredEquality {
                    actual: result.clone(),
                    expected: ty,
                })
            })
        }
        CollectionConstraint::SplitKey { operand, .. } => {
            let half = key_operation_surface(operand, Prim::Key)?;
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: result.clone(),
                expected: Type::tuple(vec![half.clone(), half]),
            }))
        }
        CollectionConstraint::SplitKeys { operand, count, .. } => {
            let key = key_operation_surface(operand, Prim::Key)?;
            let mut compatibility = subst.clone();
            unify(count, &Type::Prim(Prim::Int64), &mut compatibility).map_err(|e| e.message)?;
            let mut dims = match key {
                Type::Tensor(dims, _) => dims,
                _ => vec![],
            };
            dims.push(match evidence {
                Some(CollectionCallEvidence::SplitKeysCount(Some(count))) if *count < 0 => {
                    return Err(format!(
                        "split_keys argument 2: expected non-negative count, got {count} \
                         ([05-OP-71])"
                    ));
                }
                Some(CollectionCallEvidence::SplitKeysCount(Some(count))) => Dim::Lit(*count),
                _ => Dim::Wildcard,
            });
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: Type::tuple(vec![count.clone(), result.clone()]),
                expected: Type::tuple(vec![
                    Type::Prim(Prim::Int64),
                    Type::Tensor(dims, TensorPrec::Concrete(Prim::Key)),
                ]),
            }))
        }
        CollectionConstraint::FoldIn { operand, index, .. } => {
            let key = key_operation_surface(operand, Prim::Key)?;
            let expected = match &key {
                Type::Tensor(dims, _) => {
                    Type::Tensor(dims.clone(), TensorPrec::Concrete(Prim::Int64))
                }
                _ => Type::Prim(Prim::Int64),
            };
            let mut compatibility = subst.clone();
            unify(index, &expected, &mut compatibility).map_err(|e| format!("fold_in requires exactly equal shapes and scalar/tensor surfaces ([05-OP-72]): {}", e.message))?;
            Ok(Some(CollectionDecision::RequiredEquality {
                actual: Type::tuple(vec![index.clone(), result.clone()]),
                expected: Type::tuple(vec![expected, key]),
            }))
        }
        CollectionConstraint::Len { operand, .. } => match queried_container(operand) {
            Type::Var(_) => Ok(None),
            Type::Adt(name, _) if name == "List" || name == "Dict" => {
                Ok(joined(vec![Type::Prim(Prim::Int64)]))
            }
            _ => Err(format!("len expects List or Dict input, got {operand}")),
        },
        CollectionConstraint::Index { list, index, .. } => {
            match index {
                Type::Var(_) => return Ok(None),
                Type::Prim(Prim::Int64) => {}
                other => return Err(format!("index expects i64 index, got {other}")),
            }
            match queried_container(list) {
                Type::Var(_) => Ok(None),
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    Ok(joined(vec![args[0].clone()]))
                }
                _ => Err(format!("index expects List input, got {list}")),
            }
        }
        CollectionConstraint::Append { list, value, .. } => AggregateRule::Append
            .decide(&[list.clone(), value.clone()], &result, subst)
            .map(|decision| decision.map(CollectionDecision::ResultOrigin))
            .map_err(|error| error.message),
        CollectionConstraint::Concat { lhs, rhs, .. } => match (lhs, rhs) {
            (Type::Var(_), _) | (_, Type::Var(_)) => Ok(None),
            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                if lhs_name == "List"
                    && rhs_name == "List"
                    && lhs_args.len() == 1
                    && rhs_args.len() == 1 =>
            {
                AggregateRule::Concat
                    .decide(&[lhs.clone(), rhs.clone()], &result, subst)
                    .map(|decision| decision.map(CollectionDecision::ResultOrigin))
                    .map_err(|error| error.message)
            }
            (Type::Adt(lhs_name, lhs_args), Type::Prim(Prim::Int32))
                if lhs_name == "List" && lhs_args.len() == 1 =>
            {
                let (raw_axis, list_info) = match evidence {
                    Some(CollectionCallEvidence::TensorConcat(evidence)) => {
                        (evidence.raw_axis, evidence.list_info.clone())
                    }
                    _ => (None, ConcatListInfo::BindingLen(None)),
                };
                tensor_concat_result_type(&lhs_args[0], raw_axis, list_info, subst).map(
                    |produced| {
                        Some(CollectionDecision::ProducedResult {
                            declared: result.clone(),
                            produced,
                        })
                    },
                )
            }
            (lhs, rhs) => Err(format!(
                "concat expects matching List inputs, got {lhs} and {rhs}"
            )),
        },
    }
}

pub fn queried_container<E>(ty: &Type<E>) -> &Type<E> {
    match ty {
        Type::Ref(referent) => referent,
        other => other,
    }
}
fn key_operation_surface<E>(operand: &Type<E>, input: Prim) -> Result<Type<E>, String> {
    match operand {
        Type::Prim(p) if *p == input => Ok(Type::Prim(Prim::Key)),
        Type::Tensor(dims, TensorPrec::Concrete(p)) if *p == input => {
            Ok(Type::Tensor(dims.clone(), TensorPrec::Concrete(Prim::Key)))
        }
        other => Err(format!(
            "key operation expects {} or a tensor of {}, got {other}",
            input.name(),
            input.name()
        )),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorConcatCallEvidence {
    pub raw_axis: Option<i64>,
    pub list_info: ConcatListInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConcatListInfo {
    Direct(Vec<Vec<Dim>>),
    BindingLen(Option<usize>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CollectionCallEvidence {
    TensorConcat(TensorConcatCallEvidence),
    SplitKeysCount(Option<i64>),
}
pub fn tensor_concat_result_type<E: Clone + PartialEq>(
    element_ty: &Type<E>,
    raw_axis: Option<i64>,
    list_info: ConcatListInfo,
    subst: &Subst<E>,
) -> Result<Type<E>, String> {
    let Type::Tensor(dims, precision) = element_ty else {
        return Err(format!(
            "concat expects List[tensor[...]] for tensor concatenation, got {element_ty}"
        ));
    };
    if dims.is_empty() {
        return Err("concat expects tensor inputs with at least one axis".to_string());
    }
    let mut out_dims = dims.clone();
    let Some(raw) = raw_axis else {
        out_dims.fill(Dim::Wildcard);
        return Ok(Type::Tensor(out_dims, precision.clone()));
    };
    let rank = out_dims.len();
    let Some(axis) = normalize_static_axis(rank, raw) else {
        return Err(format!("concat axis {raw} out of bounds for rank {rank}"));
    };
    out_dims[axis] = match list_info {
        ConcatListInfo::Direct(elements) if !elements.is_empty() => elements
            .iter()
            .try_fold(0i64, |total, dims| {
                match dims
                    .get(axis)
                    .and_then(|d| subst.observe_dim(d).literal_extent())
                {
                    Some(k) if dims.len() == rank => total.checked_add(k),
                    _ => None,
                }
            })
            .map(Dim::Lit)
            .unwrap_or(Dim::Wildcard),
        ConcatListInfo::BindingLen(Some(n)) if n >= 1 => {
            match subst.observe_dim(&out_dims[axis]).literal_extent() {
                Some(k) => match k.checked_mul(n as i64) {
                    Some(total) => Dim::Lit(total),
                    None => Dim::Wildcard,
                },
                _ => Dim::Wildcard,
            }
        }
        _ => Dim::Wildcard,
    };
    Ok(Type::Tensor(out_dims, precision.clone()))
}
