//! Structural ownership for dtype-binder cast targets and literal adoption.

use super::*;
use chelis_deep::{BinderLiteralUse, visit_binder_literal_uses};

/// P10b / spec/04 §5.6 permits binder adoption only at the direct
/// `cast(literal, p)` operand. [04-INF-6] rejects it elsewhere, while
/// [04-DTYPE-1] rejects an unbounded target on this literal-source path.
pub(super) fn validate_binder_literal_adoption_in_program(
    items: &[(Option<String>, &deep::Expr)],
    signatures: &UnordMap<String, DeclaredSigMetadata>,
    errors: &mut DiagnosticSink<'_>,
) {
    for (_, expr) in items {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
            continue;
        };
        let sig = signatures.get(name);
        if sig.is_some_and(|sig| sig.dtype_bounds.is_err()) {
            continue; // declaration collection owns this error
        }
        let bounds = sig.and_then(|sig| sig.dtype_bounds.as_ref().ok());
        visit_binder_literal_uses(body, &mut |usage| {
            match usage {
            BinderLiteralUse::CastTarget {
                binder,
                source: Some(_),
            } if !bounds.is_some_and(|bounds| bounds.contains_key(binder)) => errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "cast target `{binder}` in `{name}` does not name an active primitive dtype: \
                     [04-DTYPE-1] permits a type binder here only when its declaration has a \
                     Float, Int, or Numeric bound"
                ),
                vec![format!(
                    "declare `{binder}` with a dtype-family bound before using it as a cast target"
                )],
            )),
            BinderLiteralUse::Literal {
                binder,
                source,
                adopting_binder,
            } if sig.is_some_and(|sig| sig.binders.contains(binder))
                && !source.is_some_and(|source| {
                    adopting_binder == Some(binder)
                        && bounds.and_then(|bounds| bounds.get(binder))
                            .is_some_and(|family| source.admitted_by(*family))
                }) => errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "literal in `{name}` cannot bind directly to rigid type variable `{binder}`: \
                     [04-INF-6] permits this only as the direct operand of \
                     `cast(literal, {binder})` with a compatible dtype-family bound"
                ),
                vec![format!(
                    "write `cast(<literal>, {binder})` under a dtype-family bound, or let the literal keep its §5.3 default"
                )],
            )),
            _ => {}
        }
        });
    }
}
