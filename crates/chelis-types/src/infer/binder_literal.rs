//! Structural ownership for dtype-binder cast targets and literal adoption.

use super::*;
use chelis_deep::{Atom, BinderLiteralUse, LiteralFamilyFit, visit_binder_literal_uses};

/// P10b / spec/04 §5.6 permits binder adoption only at the direct
/// `cast(literal, p)` operand. [04-INF-6] rejects it elsewhere, while
/// [04-DTYPE-1] rejects an unbounded target on this literal-source path.
///
/// The integer range rule takes BOTH atoms together, and the diagnostic names
/// both for that reason. §5.6 says the range checks apply "at `p`" and offers
/// `cast(3000000000, i64)` as the escape hatch the position exists to
/// preserve, so a reader who follows that citation alone can conclude the
/// family-wide rule contradicts it. It does not: under [04-INF-6] `p` denotes
/// every admissible instantiation of the binder, so "at `p`" already means at
/// every member of the family. The citation was incomplete rather than wrong.
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
        let bounds = sig.map(|sig| &sig.dtype_bounds);
        visit_binder_literal_uses(body, &mut |usage| {
            match usage {
            // chelis#1558: [04-DTYPE-1] constrains the cast TARGET, not the
            // source, so every source reaches this arm. PR #1545 landed the
            // arm gated on `source: Some(_)`, which enforced it for a literal
            // operand only; a variable operand checked at 1.0 and was caught
            // late and differently by each lane. Dropping the gate is the
            // whole repair: one pass, one diagnostic, both ingresses.
            BinderLiteralUse::CastTarget { binder, source: _ }
                if !bounds.is_some_and(|bounds| bounds.contains_key(binder)) => errors.push(CheckError::new(
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
            BinderLiteralUse::Literal { binder, source, adopting_binder }
                if sig.is_some_and(|sig| sig.binders.contains(binder)) => {
                    let family = bounds.and_then(|bounds| bounds.get(binder)).copied();
                    if adopting_binder == Some(binder)
                        && let (Some(source), Some(family)) = (source, family)
                        && source.has_exact_unsuffixed_style()
                        && source.family_fit(family) == LiteralFamilyFit::IntegerOutOfRange
                    {
                        let value = match source.numeric_atom() {
                            Some(Atom::Int(value)) => *value,
                            _ => unreachable!("only integer atoms have an integer range failure"),
                        };
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "literal {value} in `{name}` cannot bind to `{binder}: {}`: \
                                 spec/04-type-system.md §5.6 applies the adopted literal's \
                                 range checks at `{binder}`, and [04-INF-6] makes `{binder}` \
                                 denote every admissible instantiation, so the literal must \
                                 fit every member of the family, including i8 [-128, 127]",
                                family.surf_name()
                            ),
                            vec![
                                "use an in-range literal or narrow the declaration's dtype domain"
                                    .to_string(),
                            ],
                        ));
                    } else if adopting_binder == Some(binder)
                        && let (Some(source), Some(family)) = (source, family)
                        && source.admitted_by(family)
                        && let Some(atom) = source.numeric_atom()
                        && let Some(member) = super::literal_width::non_finite_float_member(
                            super::declared_type::restriction_for_family(family),
                            atom,
                        )
                    {
                        // [04-LIT-2] at every instantiation: the float analogue of
                        // the integer range rule above.
                        let literal = super::literal_width::render_numeric_atom(atom);
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "literal `{literal}` in `{name}` cannot bind to `{binder}: {}`: \
                                 [04-INF-6] makes `{binder}` denote every admissible \
                                 instantiation, and at {} the literal rounds to infinity, \
                                 which no literal denotes (spec/04-type-system.md [04-LIT-2], \
                                 section 5.6)",
                                family.surf_name(),
                                member.name(),
                            ),
                            vec![format!(
                                "use a value finite at every member of the family, narrow the \
                                 declaration's dtype domain, or cast a finite `f64` value \
                                 (`cast({literal}f64, {binder})`) if an infinity is intended"
                            )],
                        ));
                    } else if !source.is_some_and(|source| {
                        adopting_binder == Some(binder)
                            && family.is_some_and(|family| source.admitted_by(family))
                    }) {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "literal in `{name}` cannot bind directly to rigid type variable `{binder}`: \
                                 [04-INF-6] permits this only as the direct operand of \
                                 `cast(literal, {binder})` with a compatible dtype-family bound"
                            ),
                            vec![format!(
                                "write `cast(<literal>, {binder})` under a dtype-family bound, or let the literal keep its §5.3 default"
                            )],
                        ));
                    }
                }
            _ => {}
        }
        });
    }
}
