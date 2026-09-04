//! Structural ownership for dtype-binder cast targets and literal adoption.

use super::*;

#[derive(Clone, Copy)]
struct BinderScope<'a> {
    binders: Option<&'a UnordSet<String>>,
    bounds: Option<&'a UnordMap<String, chelis_deep::DtypeFamily>>,
}

/// P10b / spec/04 §5.6 permits a literal to adopt a binder only as the
/// direct operand of `cast(literal, p)`. [04-INF-6] keeps it rigid elsewhere,
/// while [04-DTYPE-1] requires `p` to carry a dtype-family bound.
pub(super) fn validate_binder_literal_adoption_in_program(
    items: &[(Option<String>, &deep::Expr)],
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    errors: &mut DiagnosticSink<'_>,
) {
    for (_, expr) in items {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(body)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let scope = match declared_signatures.get(name) {
            Some(signature) => match &signature.dtype_bounds {
                Ok(bounds) => BinderScope {
                    binders: Some(&signature.binders),
                    bounds: Some(bounds),
                },
                // Declaration collection owns malformed-bound diagnostics.
                Err(_) => continue,
            },
            None => BinderScope {
                binders: None,
                bounds: None,
            },
        };
        validate_expr(body, name, scope, None, errors);
    }
}

fn validate_expr(
    expr: &deep::Expr,
    declaration: &str,
    scope: BinderScope<'_>,
    adopting_cast: Option<(&str, chelis_deep::DtypeFamily)>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("validate_binder_literal_adoption", expr);

    if let Some((tag, meta, children)) = stamped_parts(expr) {
        if tag == DeepTag::Cast {
            let binder = children.get(1).and_then(exact_tvar);
            let adoption = binder.and_then(|name| {
                scope
                    .bounds
                    .and_then(|bounds| bounds.get(name))
                    .copied()
                    .map(|family| (name, family))
            });
            if let Some(name) = binder
                && adoption.is_none()
            {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "cast target `{name}` in `{declaration}` does not name an active \
                         primitive dtype: [04-DTYPE-1] permits a type binder here only \
                         when its declaration has a Float, Int, or Numeric bound"
                    ),
                    vec![format!(
                        "declare `{name}` with a dtype-family bound before using it as a \
                         cast target"
                    )],
                ));
            }
            if let Some(operand) = children.first() {
                validate_expr(operand, declaration, scope, adoption, errors);
            }
            for child in children.iter().skip(1) {
                validate_expr(child, declaration, scope, None, errors);
            }
            return;
        }

        if tag == DeepTag::Lit {
            for (_, type_expr) in meta.entries.iter().filter(|(key, _)| key == "type") {
                let Some(literal_binder) = exact_tvar(type_expr) else {
                    continue;
                };
                if literal_binder == "_"
                    || !scope
                        .binders
                        .is_some_and(|binders| binders.contains(literal_binder))
                {
                    // `_` is the legal inference hole of [04-INF-5]. A name
                    // outside this declaration is owned by the central type
                    // resolver's single undeclared-variable diagnostic.
                    continue;
                }
                let admitted = adopting_cast.is_some_and(|(cast_binder, family)| {
                    cast_binder == literal_binder && atom_admitted(children.first(), family)
                });
                if !admitted {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "literal in `{declaration}` cannot bind directly to rigid type \
                             variable `{literal_binder}`: [04-INF-6] permits this only as \
                             the direct operand of `cast(literal, {literal_binder})` with \
                             a compatible dtype-family bound"
                        ),
                        vec![format!(
                            "write `cast(<literal>, {literal_binder})` under a dtype-family \
                             bound, or let the literal keep its §5.3 default"
                        )],
                    ));
                }
            }
        }

        for child in children {
            validate_expr(child, declaration, scope, None, errors);
        }
        return;
    }

    match expr {
        deep::Expr::BareList(items, _) => {
            for item in items {
                validate_expr(item, declaration, scope, None, errors);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            validate_expr(&meta.expr, declaration, scope, adopting_cast, errors)
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                validate_expr(child, declaration, scope, None, errors);
            }
        }
        deep::Expr::List(list, _) => {
            for item in &list.elements {
                validate_expr(item, declaration, scope, None, errors);
            }
        }
        deep::Expr::Map(meta, _) => {
            for (_, value) in &meta.entries {
                validate_expr(value, declaration, scope, None, errors);
            }
        }
        deep::Expr::Node(node, _) => {
            for child in node.children_slice() {
                validate_expr(child, declaration, scope, None, errors);
            }
        }
        deep::Expr::Atom(_, _) => {}
    }
}

fn exact_tvar(expr: &deep::Expr) -> Option<&str> {
    let (DeepTag::TVar, _, children) = stamped_parts(expr)? else {
        return None;
    };
    (children.len() == 1)
        .then(|| children.first().and_then(symbol_name))
        .flatten()
}

fn atom_admitted(atom: Option<&deep::Expr>, family: chelis_deep::DtypeFamily) -> bool {
    match atom {
        Some(deep::Expr::Atom(deep::Atom::Float(_), _)) => matches!(
            family,
            chelis_deep::DtypeFamily::Float | chelis_deep::DtypeFamily::Numeric
        ),
        // An integer spelling binds exactly in either numeric family, like
        // concrete `cast(7, f64)`.
        Some(deep::Expr::Atom(deep::Atom::Int(_), _)) => true,
        _ => false,
    }
}
