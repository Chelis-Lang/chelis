//! Gradient and vector-map inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use crate::unsupported::{SpanRef, Stage, Unsupported, UnsupportedKind};

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_grad(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    if kids.is_empty() {
        return malformed_form(node, "grad", "a function argument to differentiate", errors);
    }

    let f_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let resolved = subst.apply(&f_ty);
    if let Type::Error(w) = &resolved {
        return propagate(w);
    }
    // chelis#2626: the `wrt` slot names parameter positions and depends on no
    // type, so it is read before anything about the operand is decided. A
    // program's diagnostics then do not depend on whether the operand's type
    // was already known here. `Err` already means a diagnostic was pushed.
    let wrt = match grad_wrt_indices(node, errors) {
        Ok(wrt) => wrt,
        Err(_) => return vg.fresh_type(),
    };

    match &resolved {
        Type::Fn(args, _) => {
            let awaits_group =
                |ty: &Type, subst: &Subst| product.awaits_group_completion(ty, subst);
            match decide_grad(
                &resolved,
                wrt.as_deref(),
                &awaits_group,
                true,
                adt_reg,
                subst,
            ) {
                GradDecision::Decided(Ok(grad_ty)) => grad_ty,
                GradDecision::Decided(Err(error)) => report(errors, *error),
                // Publish the same parameters with a fresh gradient result,
                // so application can bind the types before cotangent selection.
                // This also covers recursive-group inference (chelis#2626).
                GradDecision::Awaits { operand, awaited } => {
                    let published = Type::Fn(args.clone(), Box::new(vg.fresh_type()));
                    let mut operands = vec![operand];
                    operands.extend(awaited);
                    product.defer_shape_check(
                        DeferredShapeRule::Derivation(TypeDerivation::Grad { wrt }),
                        Vec::new(),
                        operands,
                        published.clone(),
                    );
                    published
                }
            }
        }
        // chelis#2626: an operand whose type is still a variable is not KNOWN
        // to be a non-function (chelis#731 §C3), and it is not known to be a
        // function either, so neither answer can be published. It used to
        // publish a fresh variable that nothing ever checked, so a `grad` of
        // an unresolved operand was never decided at all. The rule is
        // suspended on the deferred ledger instead, against a fresh variable
        // that stands for the gradient's type: decided once the operand binds,
        // or at the declaration boundary if it never does.
        Type::Var(_) => {
            let published = vg.fresh_type();
            product.defer_shape_check(
                DeferredShapeRule::Derivation(TypeDerivation::Grad { wrt }),
                Vec::new(),
                vec![resolved.clone()],
                published.clone(),
            );
            published
        }
        // chelis#874 R4 / [04-TOT-1]: this arm used to be
        // `_ => vg.fresh_type()`, commented "Can't determine function
        // structure, return fresh var". A resolved non-function IS determined,
        // and the sibling `infer_vmap` rejects the identical input with the
        // message below; the two were written to the same template and only one
        // kept a disposition, so an `f32`-typed `grad` operand scored 1.0.
        other => report(errors, *grad_expects_a_function(other)),
    }
}

/// What the `grad` rule concludes about an operand, or what it still waits on.
pub(super) enum GradDecision {
    /// The gradient's function type, or the rule's own rejection.
    Decided(Result<Type, Box<CheckError>>),
    /// The rule waits on the variables whose final types determine the
    /// gradient payload. Bare operands let the ledger detect readiness.
    Awaits { operand: Type, awaited: Vec<Type> },
}

/// Decide the gradient using settled parameter types (spec/06 sections 2.1
/// and 2.2). Unknown parameter types never mean non-differentiable: wait for
/// application or recursive-group inference to bind them, then replay this
/// same rule. An unresolved parameter at declaration close is a type error
/// under [04-INF-1], rather than a guessed gradient shape (chelis#2647).
///
/// Output admission retains chelis#2626's recursive-group deferral. Generic
/// output instantiation and its backend implementation remain chelis#2460.
pub(super) fn decide_grad(
    operand: &Type,
    wrt: Option<&[usize]>,
    awaits_group: &dyn Fn(&Type, &Subst) -> bool,
    defer_parameters: bool,
    adt_reg: &AdtRegistry,
    subst: &Subst,
) -> GradDecision {
    let target = resolved(operand, subst);
    let Type::Fn(args, ret) = &target else {
        return GradDecision::Decided(Err(grad_expects_a_function(&target)));
    };
    let mut awaited: Vec<Type> = grad_output_variables(ret)
        .into_iter()
        .map(Type::Var)
        .filter(|variable| awaits_group(variable, subst))
        .collect();
    if awaited.is_empty() && !grad_output_supported(ret) {
        return GradDecision::Decided(Err(grad_output_rejection(ret)));
    }
    let selected: Vec<usize> = match wrt {
        Some(indices) => indices.to_vec(),
        None => (0..args.len()).collect(),
    };
    for index in selected {
        let Some(arg) = args.get(index) else {
            continue;
        };
        for var in grad_argument_variables(arg) {
            let variable = Type::Var(var);
            if !defer_parameters {
                return GradDecision::Decided(Err(Box::new(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("grad parameter {index} has an unresolved type at the declaration boundary"),
                    vec!["Annotate the parameter or apply the gradient within the enclosing declaration so its parameter types are determined".to_string()],
                ))));
            }
            if !awaited.contains(&variable) {
                awaited.push(variable);
            }
        }
    }
    if !awaited.is_empty() {
        return GradDecision::Awaits {
            operand: target,
            awaited,
        };
    }
    GradDecision::Decided(grad_function_type(args, ret, wrt, adt_reg))
}

/// The variables the output rule reads: the output when it is a variable, or
/// a rank-0 tensor's precision. Every other constructor decides the rule
/// whatever variables it holds.
fn grad_output_variables(ret: &Type) -> Vec<TypeVar> {
    match ret {
        Type::Tensor(dims, _) if !dims.is_empty() => Vec::new(),
        Type::Fn(..) | Type::Tuple(_) | Type::Adt(..) | Type::KindedAdt(..) | Type::Ref(_) => {
            Vec::new()
        }
        leaf => crate::env::free_tvars(leaf),
    }
}

/// The variables [`grad_argument_type`] reads in a parameter: every variable
/// outside a function type, which is not differentiable whatever it holds.
fn grad_argument_variables(arg: &Type) -> Vec<TypeVar> {
    match arg {
        Type::Fn(..) => Vec::new(),
        Type::Tuple(items) | Type::Adt(_, items) => {
            items.iter().flat_map(grad_argument_variables).collect()
        }
        Type::KindedAdt(_, args) => args
            .iter()
            .filter_map(|argument| match argument {
                NominalArg::Type(ty) => Some(ty),
                NominalArg::Dimension(_) => None,
            })
            .flat_map(grad_argument_variables)
            .collect(),
        Type::Ref(inner) => grad_argument_variables(inner),
        leaf => crate::env::free_tvars(leaf),
    }
}

/// The `grad` rule after parameter classification has settled: reject a
/// non-floating-scalar output; skip a non-differentiable parameter without
/// `wrt`, or reject it when explicitly selected.
fn grad_function_type(
    args: &[Type],
    ret: &Type,
    wrt: Option<&[usize]>,
    adt_reg: &AdtRegistry,
) -> Result<Type, Box<CheckError>> {
    if !grad_output_supported(ret) {
        return Err(grad_output_rejection(ret));
    }
    grad_result_type(args, wrt, adt_reg).map(|grad_ret| Type::Fn(args.to_vec(), Box::new(grad_ret)))
}

fn grad_output_rejection(ret: &Type) -> Box<CheckError> {
    Box::new(CheckError::new(
        CheckErrorKind::Other,
        format!("grad requires a scalar floating output, got {ret}"),
        vec!["Reduce the function result to a scalar before applying grad".to_string()],
    ))
}

pub(super) fn grad_expects_a_function(operand: &Type) -> Box<CheckError> {
    Box::new(CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!("grad expects a function, got {operand}"),
        vec!["Apply `grad` to a named function or inline lambda".to_string()],
    ))
}

pub(super) fn grad_output_supported(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => prim.is_float(),
        Type::Tensor(dims, prim) => dims.is_empty() && prim.is_float(),
        _ => false,
    }
}

/// The gradient's result type for a function with parameters `args`, or the
/// rejection of a `wrt` index that is out of range or not differentiable.
fn grad_result_type(
    args: &[Type],
    wrt: Option<&[usize]>,
    adt_reg: &AdtRegistry,
) -> Result<Type, Box<CheckError>> {
    let targets = if let Some(indices) = wrt {
        let mut selected = Vec::with_capacity(indices.len());
        for &index in indices {
            let Some(arg) = args.get(index) else {
                return Err(Box::new(CheckError::new(
                    CheckErrorKind::ArityMismatch,
                    format!(
                        "grad `wrt` index {} is out of bounds for function with {} parameters",
                        index,
                        args.len()
                    ),
                    vec![],
                )));
            };
            let Some(grad_ty) = grad_argument_type(arg, adt_reg) else {
                return Err(Box::new(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("grad `wrt` index {index} is not differentiable"),
                    vec!["Select floating scalar or tensor parameters in `wrt`".to_string()],
                )));
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
    Ok(match targets.as_slice() {
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
    Tuple(&'a [deep::Expr]),
    Index(i64),
}

fn wrt_selector(expr: &deep::Expr) -> Option<WrtSelector<'_>> {
    if let Some(items) = tagged_children(expr, DeepTag::Tuple) {
        return Some(WrtSelector::Tuple(items));
    }
    extract_int_for_dim(expr).map(WrtSelector::Index)
}

pub(super) fn grad_wrt_indices(
    node: &DeepNode,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Option<Vec<usize>>, ErrorWitness> {
    let kids = node.children_slice();

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
        WrtSelector::Tuple(items) => {
            if items.is_empty() {
                return Err(report_witness(
                    errors,
                    CheckError::new(
                        CheckErrorKind::MalformedForm,
                        "grad `wrt` tuple must contain at least one parameter index \
                         (spec/03-deep-syntax.md; spec/06-transformations.md §2.7)"
                            .to_string(),
                        vec![],
                    ),
                ));
            }
            let mut indices = Vec::new();
            for item in items {
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
            // WS-A5: a precision that is still a variable is not known to be
            // floating (`is_float()` is false for it), so the parameter is not
            // differentiable where this reads it. `decide_grad` reads it only
            // after a recursive group that determines it has done so
            // (chelis#2626).
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
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    if kids.is_empty() {
        return malformed_form(node, "vmap", "a function argument to map", errors);
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
    // chelis#1603: reading the axis child is not admitting it. `infer_lit`
    // owns [04-LIT-1]'s atom/primitive matrix for every `lit` the walk
    // VISITS, and `infer_app` visits each operand, so a bool-stamped `1` is
    // already rejected in a builtin's axis operand. `vmap` reads its axis
    // through the slot seam and never visits the child, so the same node
    // reached axis 1 here with no diagnostic. Visit it so the one boundary
    // that owns the rule decides, rather than re-deriving the matrix at this
    // reader.
    if let Some(child) = kids.get(1) {
        infer_expr(child, env, vg, subst, adt_reg, errors, product);
    }
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
    if let Some(member) = vmap_batches_a_group_variable(&f_ty, subst, product) {
        return report(errors, vmap_group_member_fence(node, member));
    }
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let batch_var = vg.fresh_dvar();
            subst.mark_mapped_axis(batch_var, axis);
            let mut mapped_axis_renaming = UnordMap::new();
            let args = args
                .iter()
                .map(|arg| {
                    vmap_transform_param_type(
                        arg,
                        axis,
                        batch_var,
                        vg,
                        subst,
                        &mut mapped_axis_renaming,
                    )
                })
                .collect::<Result<Vec<_>, _>>();
            let ret = vmap_transform_result_type(
                &ret,
                axis,
                batch_var,
                vg,
                subst,
                &mut mapped_axis_renaming,
            );

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

/// chelis#2651: the recursive-group member whose types the group has yet to
/// determine and that a position `vmap` batches stands for: a parameter or
/// the result of the mapped function type `f_ty`, through a reference or a
/// tuple, that is a type variable the group's completion links
/// ([`InferenceProduct::group_variable_owner`]). `vmap` decides whether it
/// batches such a position where it is inferred, and passes a variable
/// through unbatched; the group's completion can then make it a tensor or a
/// scalar that should have been batched.
fn vmap_batches_a_group_variable<'a>(
    f_ty: &Type,
    subst: &Subst,
    product: &'a InferenceProduct,
) -> Option<&'a str> {
    fn owner<'a>(ty: &Type, subst: &Subst, product: &'a InferenceProduct) -> Option<&'a str> {
        match subst.apply(ty) {
            Type::Ref(inner) => owner(&inner, subst, product),
            Type::Tuple(elements) => elements
                .iter()
                .find_map(|element| owner(element, subst, product)),
            var @ Type::Var(_) => product.group_variable_owner(&var, subst),
            _ => None,
        }
    }
    let Type::Fn(args, ret) = subst.apply(f_ty) else {
        return None;
    };
    args.iter()
        .chain(std::iter::once(ret.as_ref()))
        .find_map(|position| owner(position, subst, product))
}

/// chelis#2651: `vmap` decides which parameters and result it batches from
/// the mapped function's type where it is inferred. Inside a recursive group,
/// a reference to a sibling is typed at a copy of the sibling's type that the
/// group links when it completes, so a position `vmap` batches can still be a
/// variable that the group then determines. Deciding the batching once the
/// group completes is not implemented, so the case is rejected. A body sees
/// such a variable unbound in every declaration order
/// (`group_link::sibling_instance`), so the rejection is the same in every
/// order, whatever binding the function reached the operand through.
fn vmap_group_member_fence(node: &DeepNode, member: &str) -> CheckError {
    let unsupported = Unsupported::new(
        UnsupportedKind::Construct(format!(
            "`vmap` over a function whose type the recursive-group member `{member}` has yet \
             to determine"
        )),
        "the batching decision, made before the group determines that member's types",
        Stage::Checker,
        crate::unimplemented_rejection!(
            2651,
            "write the full signature of the recursive-group member that the mapped function's \
             parameter or result type depends on; `vmap` over a type its group has yet to \
             determine is not implemented"
        ),
    )
    .with_span(SpanRef {
        offset: None,
        len: None,
        span_id: node_span_id(node).map(str::to_owned),
    })
    .with_supported_alternative(format!(
        "write `{member}`'s full signature, with every parameter and result type"
    ));
    CheckError::from_unsupported(unsupported)
}

fn vmap_transform_dims(
    dims: &[Dim],
    axis: usize,
    batch_var: DimVar,
    vg: &mut VarGen,
    subst: &Subst,
    mapped_axis_renaming: &mut UnordMap<DimVar, DimVar>,
) -> Result<Vec<Dim>, String> {
    let has_rank_spread = dims.iter().any(|dim| matches!(dim, Dim::Rank(_)));
    if !has_rank_spread && axis > dims.len() {
        return Err(format!(
            "vmap axis {axis} is out of bounds for rank {} tensor",
            dims.len()
        ));
    }

    let mut transformed = dims
        .iter()
        .map(|dim| {
            let Dim::Var(var) = dim else {
                return dim.clone();
            };
            let Some(existing_axis) = subst.mapped_axis(*var) else {
                return dim.clone();
            };
            let fresh = *mapped_axis_renaming.entry(*var).or_insert_with(|| {
                let fresh = vg.fresh_dvar();
                let shifted = existing_axis + usize::from(existing_axis >= axis);
                subst.mark_mapped_axis(fresh, shifted);
                fresh
            });
            Dim::Var(fresh)
        })
        .collect::<Vec<_>>();

    let batch = Dim::Var(batch_var);
    if has_rank_spread {
        // A rank-spread token has no physical width. Storage order cannot
        // express an insertion inside it, so retain the explicit axis
        // annotation and let row unification place the boundary.
        transformed.push(batch);
    } else {
        transformed.insert(axis, batch);
    }
    Ok(transformed)
}

pub(super) fn vmap_transform_param_type(
    ty: &Type,
    axis: usize,
    batch_var: DimVar,
    vg: &mut VarGen,
    subst: &Subst,
    mapped_axis_renaming: &mut UnordMap<DimVar, DimVar>,
) -> Result<Type, String> {
    match ty {
        Type::Ref(inner) => Ok(Type::Ref(Box::new(vmap_transform_param_type(
            inner,
            axis,
            batch_var,
            vg,
            subst,
            mapped_axis_renaming,
        )?))),
        Type::Tensor(dims, precision) => {
            let dims = vmap_transform_dims(dims, axis, batch_var, vg, subst, mapped_axis_renaming)?;
            Ok(Type::Tensor(dims, precision.clone()))
        }
        // spec/design/randomness_explicit_keys.md section 3 and [04-LIN-9]:
        // `vmap` maps no other scalar formal, but a key formal is mapped. Its
        // actual is a `tensor[batch, key]` whose row `b` is application `b`'s
        // key, so a scalar key actual, which every row would consume, fails
        // to unify.
        Type::Prim(Prim::Key) => Ok(Type::Tensor(
            vec![Dim::Var(batch_var)],
            TensorPrec::Concrete(Prim::Key),
        )),
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| {
                    vmap_transform_param_type(
                        element,
                        axis,
                        batch_var,
                        vg,
                        subst,
                        mapped_axis_renaming,
                    )
                })
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

pub(super) fn vmap_transform_result_type(
    ty: &Type,
    axis: usize,
    batch_var: DimVar,
    vg: &mut VarGen,
    subst: &Subst,
    mapped_axis_renaming: &mut UnordMap<DimVar, DimVar>,
) -> Result<Type, String> {
    match ty {
        Type::Ref(inner) => Ok(Type::Ref(Box::new(vmap_transform_result_type(
            inner,
            axis,
            batch_var,
            vg,
            subst,
            mapped_axis_renaming,
        )?))),
        Type::Prim(precision) => Ok(Type::Tensor(
            vec![Dim::Var(batch_var)],
            TensorPrec::Concrete(*precision),
        )),
        Type::Tensor(dims, precision) => {
            let dims = vmap_transform_dims(dims, axis, batch_var, vg, subst, mapped_axis_renaming)?;
            Ok(Type::Tensor(dims, precision.clone()))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| {
                    vmap_transform_result_type(
                        element,
                        axis,
                        batch_var,
                        vg,
                        subst,
                        mapped_axis_renaming,
                    )
                })
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_def(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    if kids.len() < 2 {
        return malformed_form(node, "def", "a name and a body expression", errors);
    }

    let name = match symbol_name(&kids[0]) {
        Some(n) => n.to_string(),
        None => return malformed_form(node, "def", "a symbol name as its first child", errors),
    };

    let body_level = subst.enter_level(vg);
    let body_ty = infer_expr(&kids[1], env, vg, subst, adt_reg, errors, product);
    subst.leave_level(body_level, vg);
    let scheme = env.generalize(&body_ty, subst);
    subst.name_generic_parameters(&scheme, &name, &UnordMap::new());
    // chelis#397/#469: record the size provenance (see `infer_top_level` /
    // `infer_let`) so a later `expand` size built from this binding can be
    // checked for materializability. Classified against the pre-binding scope.
    match classify_expand_size(&kids[1], env, adt_reg, subst) {
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
