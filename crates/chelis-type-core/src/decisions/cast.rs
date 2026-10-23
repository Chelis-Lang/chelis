//! Cast decisions without host diagnostic authority.

use super::{DecisionFailure, DecisionFailureKind};
use crate::model::{TensorPrec, Type, TypeVar, TypeVarRestriction};
use crate::unify::Subst;
use alloc::{boxed::Box, format, string::ToString, vec};
use chelis_deep::CastMode;
use chelis_numeric_core::named_cast::NamedCastMode;
use chelis_numeric_core::prim::Prim;

pub fn key_cast_source_error<E>(source: &Type<E>) -> Option<DecisionFailure> {
    let mut peeled = source;
    while let Type::Ref(inner) = peeled {
        peeled = inner.as_ref();
    }
    matches!(
        peeled,
        Type::Prim(Prim::Key) | Type::Tensor(_, TensorPrec::Concrete(Prim::Key))
    )
    .then(|| {
        DecisionFailure::new(
            DecisionFailureKind::UnsupportedTensorPrecision,
            format!(
                "cast has no `key` source: a random key has no numeric value to convert \
                 (spec/04-type-system.md section 1.1), got {source}"
            ),
            vec!["Derive keys with `split_key`, `split_keys` or `fold_in` instead".to_string()],
        )
    })
}

pub fn read_through_tensor_borrow<E: Clone + PartialEq>(
    source: Type<E>,
    subst: &Subst<E>,
) -> Type<E> {
    match source {
        Type::Ref(inner) if matches!(subst.apply(&inner), Type::Tensor(..)) => subst.apply(&inner),
        other => other,
    }
}

pub fn cast_result_from_settled_source<E: Clone + PartialEq>(
    resolved: Type<E>,
    new_prec: Prim,
    mode: CastMode,
    subst: &Subst<E>,
) -> Result<Type<E>, Box<DecisionFailure>> {
    let resolved = read_through_tensor_borrow(resolved, subst);
    if let Some(error) = key_cast_source_error(&resolved) {
        return Err(Box::new(error));
    }
    match resolved {
        Type::Tensor(dims, src_prec) => {
            if !new_prec.is_data_element_dtype() {
                return Err(Box::new(unsupported_precision_error(
                    new_prec, /* tensor = */ true,
                )));
            }
            if let Some(named) = mode.named() {
                let source = match src_prec {
                    TensorPrec::Concrete(p) => Some(p),
                    TensorPrec::Var(_) => None,
                };
                if let Some(error) = named_cast_pair_error(named, source, new_prec) {
                    return Err(Box::new(error));
                }
                if let TensorPrec::Var(precision) = src_prec
                    && let Some(error) = require_cast_source_family(
                        precision,
                        named_cast_source_family(named),
                        mode,
                        subst,
                    )
                {
                    return Err(Box::new(error));
                }
            }
            Ok(Type::Tensor(dims, TensorPrec::Concrete(new_prec)))
        }
        Type::Prim(src_prec) => {
            if !new_prec.is_valid_scalar_cast_target() {
                return Err(Box::new(unsupported_precision_error(
                    new_prec, /* tensor = */ false,
                )));
            }
            if !src_prec.is_data_element_dtype() {
                return Err(Box::new(DecisionFailure::with_types(
                    DecisionFailureKind::CastNonTensor,
                    format!(
                        "cast argument 1: expected numeric or bool scalar, got {} ([05-OP-63])",
                        src_prec.name()
                    ),
                    "numeric or bool scalar".to_string(),
                    src_prec.name().to_string(),
                    vec![],
                )));
            }
            if let Some(named) = mode.named()
                && let Some(error) = named_cast_pair_error(named, Some(src_prec), new_prec)
            {
                return Err(Box::new(error));
            }
            Ok(Type::Prim(new_prec))
        }
        Type::Error(w) => Ok(Type::Error(w)),
        other => Err(Box::new(DecisionFailure::with_types(
            DecisionFailureKind::CastNonTensor,
            format!("cast argument 1: expected tensor or numeric/bool scalar, got {other}"),
            "tensor or numeric/bool scalar".to_string(),
            other.to_string(),
            vec![],
        ))),
    }
}

pub fn binder_cast_result_from_settled_source<E: Clone + PartialEq>(
    resolved: Type<E>,
    target: TypeVar,
    subst: &Subst<E>,
) -> Result<Type<E>, Box<DecisionFailure>> {
    match read_through_tensor_borrow(resolved, subst) {
        Type::Tensor(dims, _) if subst.tvar_restriction(target).is_some() => {
            Ok(Type::Tensor(dims, TensorPrec::Var(target)))
        }
        Type::Prim(source) if source.is_data_element_dtype() => Ok(Type::Var(target)),
        Type::Error(_) => Ok(Type::Var(target)),
        other => Err(Box::new(DecisionFailure::with_types(
            DecisionFailureKind::CastNonTensor,
            format!(
                "cast argument 1: expected numeric or bool scalar, got {other} (quantified target)"
            ),
            "numeric or bool scalar".to_string(),
            other.to_string(),
            vec![],
        ))),
    }
}
pub fn bounded_scalar_cast_result<E>(
    bound: TypeVarRestriction,
    new_prec: Prim,
    mode: CastMode,
) -> Option<Result<Type<E>, Box<DecisionFailure>>> {
    if !new_prec.is_valid_scalar_cast_target() {
        return Some(Err(Box::new(unsupported_precision_error(
            new_prec, /* tensor = */ false,
        ))));
    }
    if let Some(named) = mode.named() {
        if let Some(error) = named_cast_pair_error(named, None, new_prec) {
            return Some(Err(Box::new(error)));
        }
        if !bound.admits_only_within(named_cast_source_family(named)) {
            return None;
        }
    }
    Some(Ok(Type::Prim(new_prec)))
}

pub fn named_cast_source_family(mode: NamedCastMode) -> TypeVarRestriction {
    match mode {
        NamedCastMode::Trunc => TypeVarRestriction::ActiveFloat,
        NamedCastMode::Saturate => TypeVarRestriction::ActiveNumeric,
        NamedCastMode::Wrap => TypeVarRestriction::ActiveInt,
    }
}

pub fn named_cast_source_noun(mode: NamedCastMode) -> &'static str {
    match mode {
        NamedCastMode::Trunc => "a float",
        NamedCastMode::Saturate => "a signed integer or float",
        NamedCastMode::Wrap => "a signed integer",
    }
}

pub fn named_cast_pair_error(
    mode: NamedCastMode,
    source: Option<Prim>,
    target: Prim,
) -> Option<DecisionFailure> {
    let keyword = mode.keyword();
    let hint = match mode {
        NamedCastMode::Trunc => "`cast_trunc` truncates a float toward zero into an integer \
                                 width ([05-OP-6]); use `cast` for every other conversion"
            .to_string(),
        NamedCastMode::Saturate => "`cast_saturate` clamps a signed integer or a float \
                                    truncated toward zero into an integer width ([05-OP-23]); \
                                    use `cast` for every other conversion"
            .to_string(),
        NamedCastMode::Wrap => "`cast_wrap` wraps a signed integer into an integer width \
                                ([05-OP-24]); use `cast_saturate` for a float source and \
                                `cast` for every other conversion"
            .to_string(),
    };
    if !target.is_integer() {
        return Some(DecisionFailure::new(
            DecisionFailureKind::CastNonTensor,
            format!(
                "`{keyword}` target `{}` is not an integer dtype",
                target.name()
            ),
            vec![hint],
        ));
    }
    match source {
        Some(prim) if !named_cast_source_family(mode).admits(prim) => Some(DecisionFailure::new(
            DecisionFailureKind::CastNonTensor,
            format!(
                "`{keyword}` source `{}` is not {} dtype",
                prim.name(),
                named_cast_source_noun(mode)
            ),
            vec![hint],
        )),
        _ => None,
    }
}

pub fn require_cast_source_family<E: Clone + PartialEq>(
    variable: TypeVar,
    required: TypeVarRestriction,
    mode: CastMode,
    subst: &Subst<E>,
) -> Option<DecisionFailure> {
    let operation = mode.keyword();
    let atom = cast_mode_atom(mode);
    let hint = format!(
        "Declare the source's binder with the `{}` bound, or convert the source with `cast` \
         first ({atom})",
        required.bound_spelling()
    );
    match subst.apply(&Type::Var(variable)) {
        Type::Var(variable) => {
            if subst.narrow_tvar_restriction(variable, required).is_ok() {
                return None;
            }
            let bound = subst.tvar_restriction(variable)?;
            Some(DecisionFailure::new(
                DecisionFailureKind::PrecisionMismatch,
                format!(
                    "`{operation}` requires a source of dtype family `{}` ({atom}), but the \
                     source dtype is bounded by {} ({}), which shares no dtype \
                     with it (spec/04-type-system.md §5.9 [04-DTYPE-2])",
                    required.bound_spelling(),
                    bound.bound_description(),
                    bound.membership_gloss(),
                ),
                vec![hint],
            ))
        }
        Type::Prim(prim) if !required.admits(prim) => Some(DecisionFailure::new(
            DecisionFailureKind::PrecisionMismatch,
            format!(
                "`{operation}` requires a source of dtype family `{}` ({atom}), got `{}`",
                required.bound_spelling(),
                prim.name(),
            ),
            vec![hint],
        )),
        _ => None,
    }
}
pub fn cast_mode_atom(mode: CastMode) -> &'static str {
    match mode {
        CastMode::Checked => "[05-OP-63]",
        CastMode::Named(named) => named.atom(),
    }
}
pub fn unsupported_precision_error(new_prec: Prim, tensor: bool) -> DecisionFailure {
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, i8, i16, i32, i64";
    if matches!(new_prec, Prim::F8e4m3) {
        DecisionFailure::new(
            DecisionFailureKind::UnsupportedTensorPrecision,
            format!(
                "cannot cast {surface} to `f8e4m3`: f8e4m3 is deferred per \
                 spec/04-type-system.md §1.1.1 and is not part of the active \
                 numeric primitive set ({active_set})"
            ),
            vec![format!(
                "f8e4m3 has no active backend in this cycle; cast to one of \
                 {active_set} instead, or follow spec/04-type-system.md §1.1.1 \
                 for the deferral rationale"
            )],
        )
    } else {
        DecisionFailure::new(
            DecisionFailureKind::UnsupportedTensorPrecision,
            format!(
                "cannot cast {surface} to unsupported precision `{}` \
                 (supported: {active_set})",
                new_prec.name()
            ),
            vec![format!("Use a supported {surface} precision")],
        )
    }
}
