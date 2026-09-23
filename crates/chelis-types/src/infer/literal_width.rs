//! [04-LIT-2]: a numeric literal binds at its dtype by one finalization, and a
//! literal whose value there is non-finite is rejected at every ingress.
//!
//! The expression, pattern, and dtype-binder rules share these helpers so a
//! literal is judged by the same `finalize_scalar` every lane binds it with:
//! a literal the checker admits can never evaluate to an infinity.

use super::*;
use crate::{RawScalar, finalize_scalar};

/// The value `atom` denotes once bound at the float primitive `prim`, or
/// `None` when `prim` is not a float or `atom` is not numeric. An `Int` atom is
/// [04-LIT-1]'s integer-spelled form and is finalized from its exact value.
pub(super) fn literal_value_at_float(prim: Prim, atom: &deep::Atom) -> Option<f64> {
    if !prim.is_float() {
        return None;
    }
    let raw = match atom {
        deep::Atom::Float(value) => RawScalar::Float(*value),
        deep::Atom::Int(value) => RawScalar::Int(*value),
        deep::Atom::Name(_) | deep::Atom::Bool(_) | deep::Atom::Str(_) => return None,
    };
    finalize_scalar("lit", prim, raw)
        .ok()
        .map(|scalar| scalar.as_f64_lossy())
}

/// Whether `atom` bound at the float primitive `prim` is non-finite.
pub(super) fn literal_is_non_finite_at(prim: Prim, atom: &deep::Atom) -> bool {
    literal_value_at_float(prim, atom).is_some_and(|value| !value.is_finite())
}

/// The first float member of `restriction`'s family at which `atom` is
/// non-finite. [04-INF-6] makes an adopted binder denote every admissible
/// instantiation, so one such member rejects the literal.
pub(super) fn non_finite_float_member(
    restriction: TypeVarRestriction,
    atom: &deep::Atom,
) -> Option<Prim> {
    Prim::ACTIVE_FLOATS
        .into_iter()
        .filter(|prim| restriction.admits(*prim))
        .find(|prim| literal_is_non_finite_at(*prim, atom))
}

/// The numeric atom as a diagnostic spells it: a float keeps a fractional or
/// exponent part, so `70000.0` does not read as the integer `70000`.
pub(super) fn render_numeric_atom(atom: &deep::Atom) -> String {
    match atom {
        deep::Atom::Float(value) => format!("{value:?}"),
        deep::Atom::Int(value) => value.to_string(),
        deep::Atom::Bool(value) => value.to_string(),
        deep::Atom::Str(value) => format!("{value:?}"),
        deep::Atom::Name(value) => value.clone(),
    }
}

/// The [04-LIT-2] rejection for a literal bound at one concrete float dtype.
pub(super) fn non_finite_literal_error(atom: &deep::Atom, prim: Prim) -> CheckError {
    let literal = render_numeric_atom(atom);
    let dtype = prim.name();
    let suggestion = if prim == Prim::F64 {
        "write a finite value; no dtype is wider than `f64`".to_string()
    } else {
        format!(
            "use a float dtype wide enough to hold the value, or cast a finite wider value \
             (`cast({literal}f64, {dtype})`) if an infinity is intended"
        )
    };
    CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "literal `{literal}` rounds to infinity at `{dtype}`, the dtype it binds at, \
             and no literal denotes an infinity (spec/04-type-system.md [04-LIT-2])"
        ),
        vec![suggestion],
    )
}
