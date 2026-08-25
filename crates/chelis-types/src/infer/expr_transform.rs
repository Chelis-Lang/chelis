//! Gradient and vector-map inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_grad(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return malformed_form(list, "grad", "a function argument to differentiate", errors);
    }

    let f_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let ret = *ret;
            if !grad_output_supported(&ret) {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::Other,
                        format!("grad requires a scalar floating output, got {}", ret),
                        vec![
                            "Reduce the function result to a scalar before applying grad"
                                .to_string(),
                        ],
                    ),
                );
            }

            match grad_result_type(list, &args, adt_reg, errors) {
                Some(grad_ret) => Type::Fn(args, Box::new(grad_ret)),
                None => vg.fresh_type(),
            }
        }
        Type::Error(w) => propagate(&w),
        _ => {
            // Can't determine function structure, return fresh var
            vg.fresh_type()
        }
    }
}

pub(super) fn grad_output_supported(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => prim.is_float(),
        Type::Tensor(dims, prim) => dims.is_empty() && prim.is_float(),
        _ => false,
    }
}

pub(super) fn grad_result_type(
    list: &deep::List,
    args: &[Type],
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let targets = if let Some(indices) = grad_wrt_indices(list, errors)? {
        let mut selected = Vec::with_capacity(indices.len());
        for index in indices {
            let Some(arg) = args.get(index) else {
                errors.push(CheckError::new(
                    CheckErrorKind::ArityMismatch,
                    format!(
                        "grad `wrt` index {} is out of bounds for function with {} parameters",
                        index,
                        args.len()
                    ),
                    vec![],
                ));
                return None;
            };
            let Some(grad_ty) = grad_argument_type(arg, adt_reg) else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("grad `wrt` index {index} is not differentiable"),
                    vec!["Select floating scalar or tensor parameters in `wrt`".to_string()],
                ));
                return None;
            };
            selected.push(grad_ty);
        }
        selected
    } else {
        args.iter()
            .filter_map(|arg| grad_argument_type(arg, adt_reg))
            .collect()
    };

    // chelis#520 D2: an ADT gradient target is supported alongside plain
    // tensor/scalar targets in a multi-argument call. `grad_argument_type`
    // has already mapped each selected parameter to its gradient type (an
    // all-float-field ADT maps to itself; a tensor/scalar to itself; a
    // non-differentiable payload was skipped or rejected). The result type
    // is the per-target tuple, whose ADT slot is the field-wise gradient
    // struct (the pytree contract). The eval-lane marshalling packs the
    // flat gradient roots back into this exact structure per argument.
    Some(match targets.as_slice() {
        [] => Type::Unit,
        [single] => single.clone(),
        _ => Type::Tuple(targets),
    })
}

pub(super) fn grad_wrt_indices(
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Option<Vec<usize>>> {
    let kids = children(list);
    let Some(wrt_expr) = kids.get(1) else {
        return Some(None);
    };

    // Issue #216: cast-aware so a Deep-direct grad node with cast-wrapped
    // wrt indices peels to the underlying int and trips the
    // non-negative-index check at infer time. Surf desugar resolves
    // parameter names to bare literal ints before reaching here, so the
    // swap is defense-in-depth for Deep-direct callers (decompiler,
    // macro output, custom tooling).
    match wrt_expr {
        deep::Expr::List(tuple, _) if get_tag(tuple) == Some(DeepTag::Tuple) => {
            let mut indices = Vec::new();
            for item in children(tuple) {
                let Some(index) = extract_int_for_dim(item) else {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        "grad `wrt` tuple must contain integer parameter indices".to_string(),
                        vec![],
                    ));
                    return None;
                };
                if index < 0 {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!("grad `wrt` index must be non-negative, got {index}"),
                        vec![],
                    ));
                    return None;
                }
                indices.push(index as usize);
            }
            Some(Some(indices))
        }
        other => {
            let Some(index) = extract_int_for_dim(other) else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "grad `wrt` must be an integer parameter index or tuple of indices".to_string(),
                    vec![],
                ));
                return None;
            };
            if index < 0 {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("grad `wrt` index must be non-negative, got {index}"),
                    vec![],
                ));
                return None;
            }
            Some(Some(vec![index as usize]))
        }
    }
}

pub(super) fn grad_argument_type(arg: &Type, adt_reg: &AdtRegistry) -> Option<Type> {
    match arg {
        Type::Prim(prim) if prim.is_float() => Some(Type::Prim(*prim)),
        // WS-A5: a polymorphic precision (TensorPrec::Var) is not yet
        // known to be float, so reject it here. Once monomorphization
        // resolves the precision, the rule re-fires on the concrete
        // instantiation. `is_float()` returns false for Var precisions.
        Type::Tensor(dims, prec) if prec.is_float() => {
            Some(Type::Tensor(dims.clone(), prec.clone()))
        }
        // chelis#520 D2 slice: an ADT whose every variant carries only
        // float tensors / float scalars gets a field-wise gradient of
        // the same constructor shape (spec/06-transformations.md
        // §2.10.1). Mixed or non-tensor payloads stay
        // non-differentiable, so the arg is skipped (no `wrt`) or
        // rejected (`wrt`-selected) exactly as before. Generic ADTs
        // fall out naturally: an uninstantiated param var is not a
        // float tensor.
        Type::Adt(name, args) => {
            let def = adt_reg.defs.get(name)?;
            let all_float_fields = def.variants.iter().all(|variant| {
                variant.fields.iter().all(|(_, field_ty)| match field_ty {
                    Type::Prim(prim) => prim.is_float(),
                    Type::Tensor(_, prec) => prec.is_float(),
                    _ => false,
                })
            });
            // A pure enum (no fields in any variant) carries no
            // continuous payload: there is nothing to differentiate,
            // and typing its gradient as the enum itself would claim a
            // gradient value the runtime cannot produce. Keep it
            // non-differentiable (unit payload), the pre-#520 typing.
            let has_any_field = def
                .variants
                .iter()
                .any(|variant| !variant.fields.is_empty());
            (all_float_fields && has_any_field).then(|| Type::Adt(name.clone(), args.clone()))
        }
        Type::Ref(inner) => grad_argument_type(inner, adt_reg),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_vmap(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return malformed_form(list, "vmap", "a function argument to map", errors);
    }

    // Issue #216: cast-aware so a Deep-direct vmap node with a cast-
    // wrapped axis literal peels to the underlying int and trips the
    // non-negative check. Surf parser restricts vmap's axis to bare
    // ints, so this is defense-in-depth for Deep-direct callers.
    let axis = kids.get(1).and_then(extract_int_for_dim).unwrap_or(0);
    if axis < 0 {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("vmap axis must be non-negative, got {axis}"),
                vec!["Use `vmap(f)` or `vmap(f, axis=n)` with n >= 0".to_string()],
            ),
        );
    }
    let axis = axis as usize;

    let f_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let batch_dim = Dim::Var(vg.fresh_dvar());
            let args = args
                .iter()
                .map(|arg| vmap_transform_param_type(arg, axis, &batch_dim))
                .collect::<Result<Vec<_>, _>>();
            let ret = vmap_transform_result_type(&ret, axis, &batch_dim);

            match (args, ret) {
                (Ok(args), Ok(ret)) => Type::Fn(args, Box::new(ret)),
                (Err(message), _) | (_, Err(message)) => report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        message,
                        vec![
                            "Choose an axis that is in bounds for every vmapped tensor".to_string(),
                        ],
                    ),
                ),
            }
        }
        Type::Error(w) => propagate(&w),
        other => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("vmap expects a function, got {other}"),
                vec!["Apply `vmap` to a named function or inline lambda".to_string()],
            ),
        ),
    }
}

pub(super) fn vmap_transform_param_type(
    ty: &Type,
    axis: usize,
    batch_dim: &Dim,
) -> Result<Type, String> {
    match ty {
        Type::Ref(inner) => Ok(Type::Ref(Box::new(vmap_transform_param_type(
            inner, axis, batch_dim,
        )?))),
        Type::Tensor(dims, precision) => {
            if axis > dims.len() {
                return Err(format!(
                    "vmap axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ));
            }
            let mut dims = dims.clone();
            dims.insert(axis, batch_dim.clone());
            Ok(Type::Tensor(dims, precision.clone()))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| vmap_transform_param_type(element, axis, batch_dim))
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

pub(super) fn vmap_transform_result_type(
    ty: &Type,
    axis: usize,
    batch_dim: &Dim,
) -> Result<Type, String> {
    match ty {
        Type::Ref(inner) => Ok(Type::Ref(Box::new(vmap_transform_result_type(
            inner, axis, batch_dim,
        )?))),
        Type::Prim(precision) => Ok(Type::Tensor(
            vec![batch_dim.clone()],
            TensorPrec::Concrete(*precision),
        )),
        Type::Tensor(dims, precision) => {
            if axis > dims.len() {
                return Err(format!(
                    "vmap axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ));
            }
            let mut dims = dims.clone();
            dims.insert(axis, batch_dim.clone());
            Ok(Type::Tensor(dims, precision.clone()))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| vmap_transform_result_type(element, axis, batch_dim))
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_def(
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
        return malformed_form(list, "def", "a name and a body expression", errors);
    }

    let name = match symbol_name(&kids[0]) {
        Some(n) => n.to_string(),
        None => return malformed_form(list, "def", "a symbol name as its first child", errors),
    };

    let body_level = subst.enter_level(vg);
    let body_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    subst.leave_level(body_level, vg);
    let scheme = env.generalize(&body_ty, subst);
    // chelis#397/#469: record the size provenance (see `infer_top_level` /
    // `infer_let`) so a later `expand` size built from this binding can be
    // checked for materializability. Classified against the pre-binding scope.
    match classify_expand_size(&kids[1], env) {
        SizeClass::Static => {
            env.mark_size_provenance(&name, crate::env::SizeProvenance::Static);
        }
        SizeClass::ShapeSourced => {
            env.mark_size_provenance(&name, crate::env::SizeProvenance::ShapeSourced);
        }
        SizeClass::Sourceless | SizeClass::Unknown => env.clear_size_provenance(&name),
    }
    // chelis#631: same discipline for list-literal lengths.
    note_list_literal_binding(env, &name, &kids[1]);
    env.bind(name, scheme);
    body_ty
}
