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
///
/// The dtype is always narrower than `f64`: the lexer rejects a non-finite
/// `f64` body, and every `i64` atom is finite at `f64`. So a finite `f64`
/// value always exists for the suggestion's cast.
pub(super) fn non_finite_literal_error(atom: &deep::Atom, prim: Prim) -> CheckError {
    let literal = render_numeric_atom(atom);
    let dtype = prim.name();
    let suggestion = format!(
        "use a float dtype wide enough to hold the value, or cast a finite wider value \
         (`cast({literal}f64, {dtype})`) if an infinity is intended"
    );
    CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "literal `{literal}` rounds to infinity at `{dtype}`, the dtype it binds at, \
             and no literal denotes an infinity (spec/04-type-system.md [04-LIT-2])"
        ),
        vec![suggestion],
    )
}

/// A repair note for a value whose literal dtype disagrees with a declared or
/// ascribed numeric primitive (`spec/04-type-system.md` §5.3, §5.6). When the
/// value is a literal, the annotation only checks its suffix or default, so
/// the note names the suffixed spelling. When a literal stands inside the
/// value, a declaration states no dtype for it: only a literal that is the
/// whole initializer takes the declared dtype.
pub(super) fn literal_dtype_hint(value: &deep::Expr, declared: &Type) -> Option<String> {
    let Type::Prim(prim) = declared else {
        return None;
    };
    if !prim.is_numeric() {
        return None;
    }
    let admits = |atom: &deep::Atom| match atom {
        deep::Atom::Int(_) => true,
        deep::Atom::Float(_) => prim.is_float(),
        _ => false,
    };
    // An ascribed literal sits in a one-expression block that owns the
    // ascribed type.
    let inner = match stamped_parts(value) {
        Some((DeepTag::Block, _, [child])) => child,
        _ => value,
    };
    if let Some((DeepTag::Lit, _, [deep::Expr::Atom(atom, _)])) = stamped_parts(inner)
        && admits(atom)
    {
        return Some(format!(
            "an ascription or declaration checks a literal's suffix or default and does not \
             select its dtype (spec/04-type-system.md \u{00a7}5.3); write `{}{}`",
            render_numeric_atom(atom),
            prim.name()
        ));
    }
    let atom = first_numeric_literal(inner, &admits)?;
    Some(format!(
        "a declaration states the dtype only of a literal that is its whole initializer \
         (spec/04-type-system.md \u{00a7}5.6); suffix a literal inside an expression, such as \
         `{}{}`",
        render_numeric_atom(atom),
        prim.name()
    ))
}

fn first_numeric_literal<'a>(
    expr: &'a deep::Expr,
    admits: &dyn Fn(&deep::Atom) -> bool,
) -> Option<&'a deep::Atom> {
    stack_guard!("first_numeric_literal", expr, None);
    match expr {
        deep::Expr::BareList(elements, _) => {
            return elements
                .iter()
                .find_map(|element| first_numeric_literal(element, admits));
        }
        deep::Expr::MetaExpr(meta, _) => return first_numeric_literal(&meta.expr, admits),
        _ => {}
    }
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag == DeepTag::Lit {
        return match kids {
            [deep::Expr::Atom(atom, _)] if admits(atom) => Some(atom),
            _ => None,
        };
    }
    kids.iter()
        .find_map(|kid| first_numeric_literal(kid, admits))
}
