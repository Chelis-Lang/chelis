//! Structural ownership for dtype-binder cast targets and literal adoption.

use super::*;
use chelis_deep::DtypeFamily;

#[derive(Clone, Copy)]
struct Scope<'a> {
    binders: Option<&'a UnordSet<String>>,
    bounds: Option<&'a UnordMap<String, DtypeFamily>>,
}

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
        let scope = match signatures.get(name) {
            Some(sig) => match &sig.dtype_bounds {
                Ok(bounds) => Scope {
                    binders: Some(&sig.binders),
                    bounds: Some(bounds),
                },
                Err(_) => continue, // declaration collection owns this error
            },
            None => Scope {
                binders: None,
                bounds: None,
            },
        };
        validate(body, name, scope, None, errors);
    }
}

fn validate(
    expr: &deep::Expr,
    declaration: &str,
    scope: Scope<'_>,
    adopting: Option<(&str, DtypeFamily)>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("validate_binder_literal_adoption", expr);
    if let Some((tag, meta, kids)) = stamped_parts(expr) {
        if tag == DeepTag::Cast {
            let binder = kids.get(1).and_then(exact_tvar);
            let adoption = binder.and_then(|name| {
                scope
                    .bounds
                    .and_then(|bounds| bounds.get(name))
                    .map(|family| (name, *family))
            });
            if let Some(name) = binder
                && adoption.is_none()
                && kids.first().is_some_and(is_literal)
            {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "cast target `{name}` in `{declaration}` does not name an active \
                         primitive dtype: [04-DTYPE-1] permits a type binder here only \
                         when its declaration has a Float, Int, or Numeric bound"
                    ),
                    vec![format!(
                        "declare `{name}` with a dtype-family bound before using it as a cast target"
                    )],
                ));
            }
            if let Some(operand) = kids.first() {
                validate(operand, declaration, scope, adoption, errors);
            }
            for child in kids.iter().skip(1) {
                validate(child, declaration, scope, None, errors);
            }
            return;
        }

        if tag == DeepTag::Lit {
            for binder in meta
                .entries
                .iter()
                .filter(|(key, _)| key == "type")
                .filter_map(|(_, ty)| exact_tvar(ty))
            {
                // `_` is [04-INF-5]'s hole; the central resolver owns names
                // outside this declaration and its single diagnostic.
                if binder == "_"
                    || !scope
                        .binders
                        .is_some_and(|binders| binders.contains(binder))
                {
                    continue;
                }
                let allowed = adopting.is_some_and(|(target, family)| {
                    target == binder && atom_admitted(kids.first(), family)
                });
                if !allowed {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "literal in `{declaration}` cannot bind directly to rigid type \
                             variable `{binder}`: [04-INF-6] permits this only as the direct \
                             operand of `cast(literal, {binder})` with a compatible dtype-family bound"
                        ),
                        vec![format!(
                            "write `cast(<literal>, {binder})` under a dtype-family bound, or let the literal keep its §5.3 default"
                        )],
                    ));
                }
            }
        }
        for child in kids {
            validate(child, declaration, scope, None, errors);
        }
        return;
    }

    if let deep::Expr::MetaExpr(meta, _) = expr {
        validate(&meta.expr, declaration, scope, adopting, errors);
        return;
    }
    if let deep::Expr::Map(meta, _) = expr {
        for (_, value) in &meta.entries {
            validate(value, declaration, scope, None, errors);
        }
        return;
    }
    let kids: &[deep::Expr] = match expr {
        deep::Expr::BareList(kids, _) => kids,
        deep::Expr::UnknownForm(data) => &data.children,
        deep::Expr::List(list, _) => &list.elements,
        deep::Expr::Node(node, _) => node.children_slice(),
        deep::Expr::Atom(_, _) | deep::Expr::Map(_, _) | deep::Expr::MetaExpr(_, _) => return,
    };
    for child in kids {
        validate(child, declaration, scope, None, errors);
    }
}

fn is_literal(expr: &deep::Expr) -> bool {
    stamped_parts(expr).is_some_and(|(tag, _, _)| tag == DeepTag::Lit)
}

fn exact_tvar(expr: &deep::Expr) -> Option<&str> {
    let (DeepTag::TVar, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    (kids.len() == 1).then(|| kids.first().and_then(symbol_name))?
}

fn atom_admitted(atom: Option<&deep::Expr>, family: DtypeFamily) -> bool {
    match atom {
        Some(deep::Expr::Atom(deep::Atom::Float(_), _)) => {
            matches!(family, DtypeFamily::Float | DtypeFamily::Numeric)
        }
        Some(deep::Expr::Atom(deep::Atom::Int(_), _)) => true,
        _ => false,
    }
}
