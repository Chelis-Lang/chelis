//! chelis#1836, chelis#2523: a tuple projection or record-field read whose
//! target was still a type variable when it was inferred.
//!
//! `infer_tuple_get` and `infer_access` cannot name the projected type before
//! the target's constructor is known, and this type system has no
//! row-polymorphic tuple or record to bind the target to, so the access
//! publishes a fresh variable and records the derivation that ties it to the
//! target. The derivation is an entry of the deferred shape ledger
//! (`DeferredShapeRule::Derivation`), so the ledger's three guarantees hold for
//! it as for a suspended call: it is decided once the target binds, a
//! `let`-bound lambda carrying one stays monomorphic until its first
//! application ([04-INF-1]), and at the declaration boundary it is decided at
//! the instantiations of a target that never bound rather than dropped.

use super::*;

/// What a deferred access reads off its target.
#[derive(Clone, Debug)]
pub(super) enum TypeDerivation {
    TupleProjection { index: usize },
    RecordField { field: String },
}

impl TypeDerivation {
    /// The access as written, for a declaration-boundary diagnostic.
    pub(super) fn spelling(&self) -> String {
        match self {
            Self::TupleProjection { index } => format!(".{index}"),
            Self::RecordField { field } => format!(".{field}"),
        }
    }
}

/// Decide one derivation against its target's current type.
///
/// Returns `false`, deciding nothing, while the target is still a variable.
/// Otherwise binds `projected` to the projected type, or reports why the
/// target has none and binds `projected` to that report's witness, and
/// returns `true`.
///
/// The diagnostics are the access rule's own for every shape `infer_access`
/// and `infer_tuple_get` reject once the target is known: this runs on a target
/// that was a variable at the access, and the ordinary access rule is not
/// re-entered, so a wrong field name or a non-record binding must be reported
/// from here or nowhere.
///
/// Round 1 P3-1 (chelis#1836): every failing shape BINDS `projected` to the
/// reported error's witness. The eager arm gets this for free by returning
/// `report(...)`'s `Type::Error` as the access type (chelis#731 §C3 cascade
/// suppression); this pass has to do it by unification, because a route
/// waiting on `projected` is suspended on the same ledger and reads the
/// variable rather than a return value.
pub(super) fn resolve_type_derivation(
    derivation: &TypeDerivation,
    source: &Type,
    projected: &Type,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let target = subst.apply(source);
    // One `match` with an explicit `Type::Var` arm, not an `if matches!`
    // guard: `unresolved_operand_census.rs` enumerates this site by that arm's
    // pattern, and a guard naming only `Type::Var` is a spelling its
    // recognizer cannot see.
    let resolved: Result<Type, Box<CheckError>> = match (&target, derivation) {
        // Still unbound: the ledger carries the entry to the next pass, and
        // the declaration boundary decides it if nothing ever binds it.
        (Type::Var(_), _) => return false,
        // The target's own rejection was already reported; the access
        // inherits its witness and stays silent (§C3).
        (Type::Error(witness), _) => {
            let _ = unify(projected, &propagate(witness), subst);
            return true;
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
    };
    match resolved {
        Ok(ty) => {
            if let Err(error) = unify(projected, &ty, subst) {
                errors.push(error.into());
            }
        }
        Err(error) => {
            let witness = report(errors, *error);
            let _ = unify(projected, &witness, subst);
        }
    }
    true
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
