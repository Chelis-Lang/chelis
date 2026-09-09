//! chelis#1489: reporting for suspended operand constraints.
//!
//! Split out of `declarations.rs`, which the inference architecture guard
//! caps at 3000 lines. This is one responsibility -- rendering the verdicts
//! unification already reached -- and it owns no decision.

use super::*;

/// chelis#1489: report the operand constraints that failed, and the ones that
/// never became decidable.
///
/// This pass makes NO decisions. Every constraint that could be settled was
/// settled the moment its operand variable was bound, by
/// `DeferredOperandGate::discharge` running inside unification. What reaches
/// here is only:
///
///   - failures recorded during discharge, and
///   - constraints still suspended when the def's inference finished.
///
/// The second group is NOT the same as "never bound". A variable settled by a
/// whole-program step that runs after this pass would still be suspended here
/// and rejected for being unresolved when it is merely not resolved YET. The
/// one known instance of that, `expand`'s candidate freeze, is gone since
/// chelis#1277 S2b gave `expand` exactly one result shape; should another
/// such step appear, it is a gap rather than a design decision.
///
/// That split is the point of the design. A set of checked sites used to
/// reject an unresolved `Type::Var` at the instant their node was inferred,
/// which made the verdict depend on WHEN
/// inference resolved a variable rather than on whether the program was well
/// typed. Deciding them in a pass like
/// this one instead was the obvious repair and the wrong one: a pass has an
/// order, and a constraint settled only by another constraint's result
/// unification is decided before its settler runs. Discharging at the binding
/// itself has no order to get wrong.
///
/// A variable that is never bound is still rejected. That is the difference
/// between this and a tolerant arm, and it is load-bearing: measured, a
/// tolerant `cast` lets `def go[t](x: t) -> int32 = cast(x, int32)` check at
/// score 1.0 AND build, with the backend selecting a dtype for `t` and
/// emitting an exported entry point on that basis. Suspending the decision is
/// safe; dropping it is not.
pub(super) fn validate_deferred_tensor_operands(
    subst: &mut Subst,
    declared_type_names: &UnordMap<TypeVar, String>,
    errors: &mut DiagnosticSink<'_>,
) {
    for failure in subst.take_operand_gate_failures() {
        match failure {
            OperandGateFailure::Decision { error } => errors.push(error),
            OperandGateFailure::ResultMismatch {
                gate,
                expected,
                settled,
            } => errors.push(CheckError::new(
                gate.kind(),
                format!(
                    "{} result does not match the type this call produces once its \
                     operand is known: expected {expected}, got {settled}",
                    gate.noun()
                ),
                gate.suggestions(),
            )),
            OperandGateFailure::Rejected { gate, resolved } => errors.push(CheckError::new(
                gate.kind(),
                gate.message(&subject_for(&resolved, declared_type_names)),
                gate.suggestions(),
            )),
        }
    }
    // Whatever is left never had its operand bound to anything.
    for (tv, gate) in subst.take_deferred_tensor_operands() {
        let resolved = subst.apply(&Type::Var(tv));
        errors.push(CheckError::new(
            gate.kind(),
            gate.message(&subject_for(&resolved, declared_type_names)),
            gate.suggestions(),
        ));
    }
}

/// chelis#260 Site 2 / spec/04 [04-FIT-9]: name the declared type parameter
/// when the signature has one. The lookup is on the RESOLVED variable, which
/// is both what carries the recorded name and what the diagnostic would
/// otherwise print. [04-FIT-10]: with no recorded name the internal identity
/// still renders rather than acquiring an invented one.
fn subject_for(resolved: &Type, declared_type_names: &UnordMap<TypeVar, String>) -> String {
    match resolved {
        Type::Var(resolved_var) => match declared_type_names.get(resolved_var) {
            Some(name) => format!("`{name}`"),
            None => format!("{resolved}"),
        },
        _ => format!("{resolved}"),
    }
}
