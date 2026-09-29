//! chelis#1836, chelis#2523: a tuple projection or record-field read whose
//! target was still a type variable when it was inferred, and chelis#2626: a
//! `grad` whose operand, output or differentiated parameter was.
//!
//! `infer_tuple_get` and `infer_access` cannot name the projected type before
//! the target's constructor is known, and this type system has no
//! row-polymorphic tuple or record to bind the target to, so the access
//! publishes a fresh variable and records the derivation that ties it to the
//! target. `infer_grad` cannot name the gradient's type, or decide whether the
//! operand admits one, before those types are known, so it does the same. The
//! derivation is an entry of the deferred shape ledger
//! (`DeferredShapeRule::Derivation`), so the ledger's three guarantees hold for
//! it as for a suspended call: it is decided once the target binds, a
//! `let`-bound lambda carrying one stays monomorphic until its first
//! application ([04-INF-1]), and at the declaration boundary it is decided at
//! the instantiations of a target that never bound rather than dropped.

use super::*;

/// What a deferred derivation reads off its target.
#[derive(Clone, Debug)]
pub(super) enum TypeDerivation {
    TupleProjection {
        index: usize,
    },
    RecordField {
        field: String,
    },
    /// `grad` of the target, differentiating the parameters `wrt` selects, or
    /// every parameter without it, using their final inferred types. The
    /// projected type is the type the call
    /// published: the gradient's function type.
    Grad {
        wrt: Option<Vec<usize>>,
    },
    /// The mapped function is published before every parameter and result
    /// constructor that batching reads is known. Its published function type
    /// has separate variables: a use of the mapped result cannot decide the
    /// row result, and applying it may determine an untyped row parameter.
    Vmap {
        axis: usize,
        batch_var: DimVar,
        group_owned: bool,
    },
}

impl TypeDerivation {
    /// The operation as written, for a declaration-boundary diagnostic.
    pub(super) fn operation(&self) -> String {
        match self {
            Self::TupleProjection { index } => format!("the access `.{index}`"),
            Self::RecordField { field } => format!("the access `.{field}`"),
            Self::Grad { .. } => "`grad`".to_string(),
            Self::Vmap { .. } => "`vmap`".to_string(),
        }
    }

    /// The type a rejection's witness binds, so a route waiting on the
    /// projected type inherits the rejection rather than reporting again
    /// (chelis#731 §C3). A deferred `grad` published a function, and what
    /// stands for the gradient is that function's result; its parameters are
    /// the operand's own.
    fn witness_slot(&self, projected: &Type, subst: &Subst) -> Type {
        match (self, subst.apply(projected)) {
            (Self::Grad { .. } | Self::Vmap { .. }, Type::Fn(_, result)) => *result,
            _ => projected.clone(),
        }
    }
}

/// What deciding a derivation did.
pub(super) enum DerivationStep {
    /// The target is still a variable, so nothing was decided.
    Pending,
    /// The mapped call supplied a row-parameter type. Another obligation,
    /// such as a reduction in the row function, may now become ready.
    Progress,
    /// `projected` is bound to the projected type or to a rejection's witness.
    Decided,
    /// A `grad` whose types revealed further unresolved parameter variables
    /// or recursive-group output variables. It is carried again as this
    /// derivation over these operands: the operand's type as it was read,
    /// then each awaited variable.
    Awaits {
        derivation: TypeDerivation,
        operands: Vec<Type>,
    },
}

/// Decide one derivation against its target's current type.
///
/// Returns [`DerivationStep::Pending`], deciding nothing, while the target is
/// still a variable. Otherwise binds `projected` to the projected type, or
/// reports why the target has none and binds `projected` to that report's
/// witness, and returns [`DerivationStep::Decided`]. A `grad` waits again on a
/// parameter variable, or an output variable `awaits_group` says its group
/// still determines ([`DerivationStep::Awaits`]). At declaration close,
/// `defer_parameters` and `awaits_group` both disallow further waiting.
///
/// The diagnostics are the access rule's own for every shape `infer_access`
/// and `infer_tuple_get` reject once the target is known: this runs on a target
/// that was a variable at the access, and the ordinary access rule is not
/// re-entered, so a wrong field name or a non-record binding must be reported
/// from here or nowhere. A `grad` runs [`decide_grad`], the rule `infer_grad`
/// runs, and reports its diagnostics.
///
/// Round 1 P3-1 (chelis#1836): every failing shape BINDS `projected` to the
/// reported error's witness. The eager arm gets this for free by returning
/// `report(...)`'s `Type::Error` as the access type (chelis#731 §C3 cascade
/// suppression); this pass has to do it by unification, because a route
/// waiting on `projected` is suspended on the same ledger and reads the
/// variable rather than a return value.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_type_derivation(
    derivation: &TypeDerivation,
    source: &Type,
    projected: &Type,
    awaits_group: &dyn Fn(&Type, &Subst) -> bool,
    defer_parameters: bool,
    group_open: bool,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> DerivationStep {
    if let TypeDerivation::Vmap {
        axis,
        batch_var,
        group_owned,
    } = derivation
    {
        return resolve_vmap_derivation(
            source,
            projected,
            *axis,
            *batch_var,
            *group_owned && group_open,
            defer_parameters,
            vg,
            subst,
            errors,
        );
    }
    let target = subst.apply(source);
    // One `match` with an explicit `Type::Var` arm, not an `if matches!`
    // guard: `unresolved_operand_census.rs` enumerates this site by that arm's
    // pattern, and a guard naming only `Type::Var` is a spelling its
    // recognizer cannot see.
    let resolved: Result<Type, Box<CheckError>> = match (&target, derivation) {
        // Still unbound: the ledger carries the entry to the next pass, and
        // the declaration boundary decides it if nothing ever binds it.
        (Type::Var(_), _) => return DerivationStep::Pending,
        // The target's own rejection was already reported; the access
        // inherits its witness and stays silent (§C3).
        (Type::Error(witness), _) => {
            let slot = derivation.witness_slot(projected, subst);
            let _ = unify(&slot, &propagate(witness), subst);
            return DerivationStep::Decided;
        }
        (Type::Tuple(elements), TypeDerivation::TupleProjection { index }) => {
            match elements.get(*index) {
                Some(element) => Ok(element.clone()),
                None => Err(Box::new(CheckError::new(
                    CheckErrorKind::TupleIndexOutOfBounds,
                    format!(
                        "tuple index {index} out of bounds for tuple of size {}",
                        elements.len()
                    ),
                    vec![],
                ))),
            }
        }
        (other, TypeDerivation::TupleProjection { .. }) => Err(Box::new(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("expected tuple type, got {other}"),
            vec![],
        ))),
        (
            Type::Adt(adt_name, _) | Type::KindedAdt(adt_name, _),
            TypeDerivation::RecordField { field },
        ) => record_field_type(adt_name, field, &target, vg, subst, adt_reg),
        (other, TypeDerivation::RecordField { field }) => Err(Box::new(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!(
                "field access `.{field}` expects a record value, got a value of type `{other}` (chelis#755)"
            ),
            vec![],
        ))),
        // The rule `infer_grad` runs on an operand it knows, so a `grad`
        // decided here and one decided at the call cannot disagree. What the
        // call waited on has bound, or its group has completed and nothing
        // waits any more. A bound variable can reveal another unresolved
        // component, which the rule waits on again.
        (Type::Fn(args, _), TypeDerivation::Grad { wrt }) => {
            // An operand that was a variable at the call published a variable
            // for the gradient, which applications since may have constrained.
            // Result ascriptions stay on the separate result-constraint ledger
            // until this derivation settles; they cannot supply these inputs.
            // The gradient's parameters are the operand's, as they are for a
            // call that published the function itself, so tie them first and
            // decide on what those applications determined.
            let gradient = Type::Fn(args.clone(), Box::new(vg.fresh_type()));
            if let Err(error) = unify(projected, &gradient, subst) {
                let witness = report(errors, error.into());
                let slot = derivation.witness_slot(projected, subst);
                let _ = unify(&slot, &witness, subst);
                return DerivationStep::Decided;
            }
            match decide_grad(
                source,
                wrt.as_deref(),
                awaits_group,
                defer_parameters,
                adt_reg,
                subst,
            ) {
                GradDecision::Decided(decided) => decided,
                GradDecision::Awaits { operand, awaited } => {
                    let mut operands = vec![operand];
                    operands.extend(awaited);
                    return DerivationStep::Awaits {
                        derivation: TypeDerivation::Grad { wrt: wrt.clone() },
                        operands,
                    };
                }
            }
        }
        (other, TypeDerivation::Grad { .. }) => Err(grad_expects_a_function(other)),
        (_, TypeDerivation::Vmap { .. }) => unreachable!("vmap resolved above"),
    };
    match resolved {
        Ok(ty) => {
            if let Err(error) = unify(projected, &ty, subst) {
                errors.push(error.into());
            }
        }
        Err(error) => {
            let witness = report(errors, *error);
            let slot = derivation.witness_slot(projected, subst);
            let _ = unify(&slot, &witness, subst);
        }
    }
    DerivationStep::Decided
}

/// Reconcile the row function with the mapped function. The result relation
/// is directional: a use of `vmap(f)` never determines what `f` returns.
/// A mapped *argument* may determine an untyped row parameter, since an
/// application is [04-INF-1]'s binding site for that parameter.
#[allow(clippy::too_many_arguments)]
fn resolve_vmap_derivation(
    source: &Type,
    projected: &Type,
    axis: usize,
    batch_var: DimVar,
    group_open: bool,
    defer_parameters: bool,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> DerivationStep {
    let source_ty = subst.apply(source);
    if let Type::Error(witness) = &source_ty {
        let slot = match subst.apply(projected) {
            Type::Fn(_, result) => *result,
            other => other,
        };
        let _ = unify(&slot, &propagate(witness), subst);
        return DerivationStep::Decided;
    }
    if group_open {
        return DerivationStep::Pending;
    }

    let published = subst.apply(projected);
    if let (Type::Var(_), Type::Fn(mapped_args, _)) = (&source_ty, &published) {
        let row_function = Type::Fn(
            mapped_args.iter().map(|_| vg.fresh_type()).collect(),
            Box::new(vg.fresh_type()),
        );
        if let Err(error) = unify(source, &row_function, subst) {
            return reject_vmap_derivation(projected, error.into(), subst, errors);
        }
        return DerivationStep::Progress;
    }
    let Type::Fn(row_args, _) = subst.apply(source) else {
        if !defer_parameters {
            return reject_vmap_derivation(
                projected,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "`vmap` needs a determined function type at the declaration boundary"
                        .to_string(),
                    vec![],
                ),
                subst,
                errors,
            );
        }
        return DerivationStep::Pending;
    };

    if let Type::Fn(mapped_args, _) = &published {
        if row_args.len() != mapped_args.len() {
            return reject_vmap_derivation(
                projected,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "`vmap` row function has {} parameter(s), but its mapped use has {}",
                        row_args.len(),
                        mapped_args.len()
                    ),
                    vec![],
                ),
                subst,
                errors,
            );
        }
        let mut progress = false;
        for (row, mapped) in row_args.iter().zip(mapped_args) {
            match infer_vmap_row_parameter(row, mapped, axis, batch_var, subst) {
                Ok(changed) => progress |= changed,
                Err(error) => return reject_vmap_derivation(projected, *error, subst, errors),
            }
        }
        if progress {
            return DerivationStep::Progress;
        }
    }

    let row_function = subst.apply(source);
    let Type::Fn(row_args, row_result) = row_function else {
        unreachable!("the row function was established above")
    };
    if row_args.iter().any(vmap_unknown_position) || vmap_unknown_position(&row_result) {
        if !defer_parameters {
            return reject_vmap_derivation(
                projected,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "`vmap` has an unresolved row parameter or result type at the declaration boundary"
                        .to_string(),
                    vec![
                        "Determine the row function's types within this declaration or write its full signature"
                            .to_string(),
                    ],
                ),
                subst,
                errors,
            );
        }
        return DerivationStep::Pending;
    }

    let mut renaming = UnordMap::new();
    let mapped_args = row_args
        .iter()
        .map(|row| vmap_transform_param_type(row, axis, batch_var, vg, subst, &mut renaming))
        .collect::<Result<Vec<_>, _>>();
    let mapped_result =
        vmap_transform_result_type(&row_result, axis, batch_var, vg, subst, &mut renaming);
    let decided = match (mapped_args, mapped_result) {
        (Ok(args), Ok(result)) => Type::Fn(args, Box::new(result)),
        (Err(message), _) | (_, Err(message)) => {
            return reject_vmap_derivation(
                projected,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    message,
                    vec!["Choose an axis in bounds for every mapped tensor".to_string()],
                ),
                subst,
                errors,
            );
        }
    };
    if let Err(error) = unify(projected, &decided, subst) {
        return reject_vmap_derivation(projected, error.into(), subst, errors);
    }
    DerivationStep::Decided
}

fn vmap_unknown_position(ty: &Type) -> bool {
    match ty {
        Type::Var(_) => true,
        Type::Ref(inner) => vmap_unknown_position(inner),
        Type::Tuple(items) => items.iter().any(vmap_unknown_position),
        _ => false,
    }
}

/// Infer the type of one row from a mapped actual, only where the row type
/// still has an unknown constructor. Known tensor and key formals are checked
/// by the forward rule when the complete row function is available.
fn infer_vmap_row_parameter(
    row: &Type,
    mapped: &Type,
    axis: usize,
    batch_var: DimVar,
    subst: &mut Subst,
) -> Result<bool, Box<CheckError>> {
    let row = subst.apply(row);
    let mapped = subst.apply(mapped);
    match (row, mapped) {
        (Type::Var(_), Type::Var(_)) => Ok(false),
        (row @ Type::Var(_), mapped) => {
            let Some(inferred) = unmap_vmap_parameter(&mapped, axis, batch_var, subst)? else {
                return Ok(false);
            };
            unify(&row, &inferred, subst).map_err(|error| Box::new(CheckError::from(error)))?;
            Ok(true)
        }
        (Type::Ref(row), Type::Ref(mapped)) => {
            infer_vmap_row_parameter(&row, &mapped, axis, batch_var, subst)
        }
        (Type::Tuple(rows), Type::Tuple(mapped)) if rows.len() == mapped.len() => {
            let mut progress = false;
            for (row, mapped) in rows.iter().zip(&mapped) {
                progress |= infer_vmap_row_parameter(row, mapped, axis, batch_var, subst)?;
            }
            Ok(progress)
        }
        _ => Ok(false),
    }
}

fn unmap_vmap_parameter(
    mapped: &Type,
    axis: usize,
    batch_var: DimVar,
    subst: &mut Subst,
) -> Result<Option<Type>, Box<CheckError>> {
    match mapped {
        Type::Var(_) => Ok(None),
        Type::Tensor(dims, precision) => {
            if axis >= dims.len() {
                return Err(Box::new(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("vmap axis {axis} is out of bounds for the mapped tensor"),
                    vec![],
                )));
            }
            // A one-axis key tensor might be a mapped scalar key or a mapped
            // rank-zero key tensor. The row type must settle elsewhere.
            if dims.len() == 1
                && matches!(
                    precision,
                    TensorPrec::Concrete(Prim::Key) | TensorPrec::Var(_)
                )
            {
                return Ok(None);
            }
            let mut row_dims = dims.clone();
            let batch = row_dims.remove(axis);
            // This is one dimension equation. The batch marker's position
            // belongs to the full mapped tensor, so wrapping it in a rank-one
            // tensor makes a nonzero axis appear out of bounds.
            unify_dim(&Dim::Var(batch_var), &batch, subst)
                .map_err(|error| Box::new(CheckError::from(error)))?;
            Ok(Some(Type::Tensor(row_dims, precision.clone())))
        }
        Type::Ref(inner) => Ok(unmap_vmap_parameter(inner, axis, batch_var, subst)?
            .map(|row| Type::Ref(Box::new(row)))),
        Type::Tuple(items) => {
            let mut rows = Vec::with_capacity(items.len());
            for item in items {
                let Some(row) = unmap_vmap_parameter(item, axis, batch_var, subst)? else {
                    return Ok(None);
                };
                rows.push(row);
            }
            Ok(Some(Type::Tuple(rows)))
        }
        Type::Error(_) => Ok(None),
        other => Ok(Some(other.clone())),
    }
}

fn reject_vmap_derivation(
    projected: &Type,
    error: CheckError,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> DerivationStep {
    let witness = report(errors, error);
    let slot = match subst.apply(projected) {
        Type::Fn(_, result) => *result,
        other => other,
    };
    let _ = unify(&slot, &witness, subst);
    DerivationStep::Decided
}

/// The type of `field` on a value of the ADT `adt_name`, instantiated at
/// `target`'s arguments.
fn record_field_type(
    adt_name: &str,
    field: &str,
    target: &Type,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
) -> Result<Type, Box<CheckError>> {
    let Some(variant) = single_record_variant(adt_reg, adt_name) else {
        return Err(Box::new(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!(
                "field access `.{field}` is only defined on a single-record-variant type; \
                 `{adt_name}` is a multi-variant or positional-field type (chelis#755)"
            ),
            vec!["pattern-match on the variants with `match` to read their fields".to_string()],
        )));
    };
    let Some(position) = variant
        .fields
        .iter()
        .position(|(name, _)| name.as_deref() == Some(field))
    else {
        return Err(Box::new(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("unknown record field '{field}' on {adt_name}"),
            vec![format!(
                "known fields: {:?}",
                variant
                    .fields
                    .iter()
                    .filter_map(|(name, _)| name.as_deref())
                    .collect::<Vec<_>>()
            )],
        )));
    };
    let field_types = instantiated_field_types(adt_name, variant, target, adt_reg, vg, subst);
    // `position` came from a validated hit, so a shorter list is an internal
    // inconsistency: loud, never silent (chelis#731 §C3).
    field_types.get(position).cloned().ok_or_else(|| {
        Box::new(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!(
                "internal: field `{field}` of `{adt_name}` resolved to position {position} but \
                 no instantiated field type is available (chelis#731 [04-TOT-2])"
            ),
            vec![],
        ))
    })
}
