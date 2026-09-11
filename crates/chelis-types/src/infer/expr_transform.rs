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
        // An operand whose type is still a variable decides nothing yet: it is
        // not KNOWN to be a non-function, and reporting here would invent a
        // rejection against an undecided type (chelis#731 §C3). Deferring it is
        // what the checker does with unresolved variables elsewhere, and it is
        // the one input the arm below must not claim.
        Type::Var(_) => vg.fresh_type(),
        // chelis#874 R4 / [04-TOT-1]: this arm used to be
        // `_ => vg.fresh_type()`, commented "Can't determine function
        // structure, return fresh var". A resolved non-function IS determined,
        // and the sibling `infer_vmap` rejects the identical input with the
        // message below; the two were written to the same template and only one
        // kept a disposition, so an `f32`-typed `grad` operand scored 1.0.
        other => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("grad expects a function, got {other}"),
                vec!["Apply `grad` to a named function or inline lambda".to_string()],
            ),
        ),
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
    // `grad_result_type` returns `Option<Type>` where `None` already means "a
    // diagnostic was pushed", the pre-existing convention at this boundary.
    // `.ok()?` preserves it exactly; threading the witness further is plumbing
    // this change does not take on.
    let targets = if let Some(indices) = grad_wrt_indices(list, errors).ok()? {
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

/// What the `grad` `wrt` slot can legitimately hold: a tuple of parameter
/// indices, or a single one.
///
/// chelis#874 Slice 2: this is the shape the seam reads at `grad` child 1.
/// The tuple's OWN children are `RuntimeExpr`-role, not selector slots, so
/// their per-element diagnostics below stay where they are; the seam decides
/// only whether the slot itself is readable, exactly as it decides `vmap`'s
/// axis while the non-negativity check stays a separate value check.
pub(super) enum WrtSelector<'a> {
    Tuple(&'a deep::List),
    Index(i64),
}

fn wrt_selector(expr: &deep::Expr) -> Option<WrtSelector<'_>> {
    // Carrier note, CONFIRMED by execution rather than inferred, and preserved
    // from the pre-migration code deliberately so the migration changes no
    // verdict.
    //
    // This `Expr::List`-only match means a `(tuple {} ..)` at this slot does
    // not match on the STAMPED ingress, where it arrives as `Expr::Node`. The
    // consequence is a fail-closed OVER-REJECTION, not a silent fallback:
    // `extract_int_for_dim` returns `None` for a tuple node, so
    // `check_typed_program` REJECTS a well-formed multi-index `grad` that
    // `check_ir_program` accepts. Nothing quietly takes the single-index path.
    //
    //     (defsig {} pair2 (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
    //     (def {} pair2 (fn {} (params {} x y) (var {} x)))
    //     (def {} g (grad {} (var {} pair2) (tuple {} 0 1)))
    //
    // The same divergence exists before this migration, with `TypeMismatch`
    // in place of `MalformedForm`: the accept/reject verdicts on both
    // ingresses are unchanged and only the kind and the text move. No CLI
    // surface reaches it; the callers that can are `chelis-cli`'s prove paths
    // and `chelis-backend-c`.
    //
    // It is a chelis#1107-class carrier question on chelis#1125's [04-TOT-5]
    // ingress-parity axis, not this seam's class. Filed as chelis#1618, a
    // sub-issue of chelis#1125, rather than fixed inside a migration that
    // claims to change no verdict.
    if let deep::Expr::List(tuple, _) = expr
        && get_tag(tuple) == Some(DeepTag::Tuple)
    {
        return Some(WrtSelector::Tuple(tuple));
    }
    extract_int_for_dim(expr).map(WrtSelector::Index)
}

pub(super) fn grad_wrt_indices(
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Option<Vec<usize>>, ErrorWitness> {
    let kids = children(list);

    // Issue #216: cast-aware so a Deep-direct grad node with cast-wrapped
    // wrt indices peels to the underlying int and trips the
    // non-negative-index check at infer time. Surf desugar resolves
    // parameter names to bare literal ints before reaching here, so the
    // swap is defense-in-depth for Deep-direct callers (decompiler,
    // macro output, custom tooling).
    //
    // chelis#874 Slice 2: the slot read runs through the shared seam. An
    // ABSENT `wrt` is `grad(f)`'s documented every-parameter default and stays
    // `Ok(None)`; a PRESENT child that is neither a tuple form nor an integer
    // is `Err`, with the seam's `MalformedForm` already pushed.
    let selector = match read_optional_slot(
        kids,
        DeepTag::Grad,
        1,
        SlotShape::ParameterIndices,
        wrt_selector,
        errors,
    )? {
        Some(selector) => selector,
        None => return Ok(None),
    };

    match selector {
        WrtSelector::Tuple(tuple) => {
            let mut indices = Vec::new();
            for item in children(tuple) {
                let Some(index) = extract_int_for_dim(item) else {
                    return Err(report_witness(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            "grad `wrt` tuple must contain integer parameter indices".to_string(),
                            vec![],
                        ),
                    ));
                };
                if index < 0 {
                    return Err(report_witness(
                        errors,
                        CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!("grad `wrt` index must be non-negative, got {index}"),
                            vec![],
                        ),
                    ));
                }
                indices.push(index as usize);
            }
            Ok(Some(indices))
        }
        // A readable index that is out of range is a VALUE error, not a
        // malformed slot, and keeps its own diagnostic -- the same split
        // `vmap`'s negative axis keeps after Slice 1.
        WrtSelector::Index(index) => {
            if index < 0 {
                return Err(report_witness(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!("grad `wrt` index must be non-negative, got {index}"),
                        vec![],
                    ),
                ));
            }
            Ok(Some(vec![index as usize]))
        }
    }
}

pub(super) fn grad_argument_type(arg: &Type, adt_reg: &AdtRegistry) -> Option<Type> {
    fn cotangent(arg: &Type, adt_reg: &AdtRegistry, visiting: &mut Vec<Type>) -> (Type, bool) {
        match arg {
            Type::Prim(prim) if prim.is_float() => (Type::Prim(*prim), true),
            Type::Prim(_) => (Type::Unit, false),
            // WS-A5: a polymorphic precision (TensorPrec::Var) is not yet
            // known to be float, so reject it here. Once monomorphization
            // resolves the precision, the rule re-fires on the concrete
            // instantiation. `is_float()` returns false for Var precisions.
            Type::Tensor(dims, prec) if prec.is_float() => {
                (Type::Tensor(dims.clone(), prec.clone()), true)
            }
            Type::Tensor(_, _) => (Type::Unit, false),
            // [06] §2.1: List is a recursive cotangent carrier. Preserve
            // every container layer, but only admit the argument as a grad
            // target when its element type recursively contains a float leaf.
            // A recursively all-discrete List is forward-only.
            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                let (element, has_float) = cotangent(&args[0], adt_reg, visiting);
                (Type::Adt(name.clone(), vec![element]), has_float)
            }
            Type::Tuple(items) => {
                let mapped = items
                    .iter()
                    .map(|item| cotangent(item, adt_reg, visiting))
                    .collect::<Vec<_>>();
                let has_float = mapped.iter().any(|(_, has_float)| *has_float);
                (
                    Type::Tuple(mapped.into_iter().map(|(ty, _)| ty).collect()),
                    has_float,
                )
            }
            Type::Adt(name, args) => {
                // Alias transparency and nominal type-argument substitution
                // must happen before classifying reachable fields. Without
                // this, `Wrapper[f32]` appears to contain only its stored
                // registration-time type variable and is incorrectly
                // rejected as non-differentiable.
                if visiting.contains(arg) {
                    return (arg.clone(), false);
                }
                visiting.push(arg.clone());
                let result = if let Some(expanded) = adt_reg.instantiate_alias(name, args) {
                    cotangent(&expanded, adt_reg, visiting)
                } else {
                    let has_float = adt_reg.defs.get(name).is_some_and(|def| {
                        let substitutions = def
                            .param_vars
                            .iter()
                            .copied()
                            .zip(args.iter().cloned())
                            .collect::<chelis_unord::UnordMap<_, _>>();
                        def.variants.iter().any(|variant| {
                            variant.fields.iter().any(|(_, field_ty)| {
                                let instantiated =
                                    crate::adt::substitute_alias_type(field_ty, &substitutions);
                                cotangent(&instantiated, adt_reg, visiting).1
                            })
                        })
                    });
                    // ADTs are nominal at the checker boundary. The executed
                    // constructor is preserved at runtime, while recursive
                    // field cotangents replace discrete leaves with unit as
                    // required by spec/06 section 2.1.
                    (Type::Adt(name.clone(), args.clone()), has_float)
                };
                debug_assert_eq!(visiting.pop().as_ref(), Some(arg));
                result
            }
            Type::KindedAdt(name, args) => {
                if visiting.contains(arg) {
                    return (arg.clone(), false);
                }
                visiting.push(arg.clone());
                let result = if let Some(expanded) = adt_reg.instantiate_nominal_alias(name, args) {
                    cotangent(&expanded, adt_reg, visiting)
                } else {
                    let has_float = adt_reg.defs.get(name).is_some_and(|def| {
                        let Some((type_subst, dim_subst)) =
                            crate::adt::nominal_substitutions(&def.param_args, args)
                        else {
                            return false;
                        };
                        def.variants.iter().any(|variant| {
                            variant.fields.iter().any(|(_, field_ty)| {
                                let instantiated = crate::adt::substitute_nominal_type(
                                    field_ty,
                                    &type_subst,
                                    &dim_subst,
                                );
                                cotangent(&instantiated, adt_reg, visiting).1
                            })
                        })
                    });
                    (Type::KindedAdt(name.clone(), args.clone()), has_float)
                };
                debug_assert_eq!(visiting.pop().as_ref(), Some(arg));
                result
            }
            Type::Ref(inner) => {
                let (inner, has_float) = cotangent(inner, adt_reg, visiting);
                (inner, has_float)
            }
            Type::Fn(_, _) | Type::Unit | Type::Var(_) | Type::Error(_) => (Type::Unit, false),
        }
    }

    let (result, has_float) = cotangent(arg, adt_reg, &mut Vec::new());
    has_float.then_some(result)
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
    //
    // chelis#874 R1 / [04-TOT-4]: this read used to be
    // `kids.get(1).and_then(extract_int_for_dim).unwrap_or(0)`, one
    // `unwrap_or` serving two different inputs. `kids.get(1) == None` is
    // spec/02 §0.1's bare `vmap(f)`, where the zero default is correct;
    // `Some(child)` that cannot be read is a node the program submitted and
    // the checker discarded, which scored a perfect 1.0. The seam keeps the
    // first and rejects the second.
    let axis = match read_optional_slot(
        kids,
        DeepTag::Vmap,
        1,
        SlotShape::IntegerAxis,
        extract_int_for_dim,
        errors,
    ) {
        Ok(Some(axis)) => axis,
        Ok(None) => 0,
        Err(witness) => return propagate(&witness),
    };
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
    match classify_expand_size(&kids[1], env, adt_reg) {
        SizeClass::Static => {
            if let Some(value) =
                fold_static_int_expr(&kids[1], |bound| env.static_size_value(bound))
            {
                env.mark_static_size_value(&name, value);
            } else {
                env.mark_size_provenance(&name, crate::env::SizeProvenance::Static);
            }
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

#[cfg(test)]
mod grad_argument_type_tests {
    use super::*;

    #[test]
    fn list_cotangent_recurses_and_preserves_the_container_shape() {
        let registry = AdtRegistry::default();
        let floats = Type::Adt("List".to_string(), vec![Type::Prim(Prim::F32)]);
        let nested = Type::Adt("List".to_string(), vec![floats.clone()]);

        assert_eq!(grad_argument_type(&floats, &registry), Some(floats));
        assert_eq!(grad_argument_type(&nested, &registry), Some(nested));
    }

    #[test]
    fn recursively_all_discrete_list_is_not_a_gradient_target() {
        let registry = AdtRegistry::default();
        let ints = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);

        assert_eq!(grad_argument_type(&ints, &registry), None);
    }
}
