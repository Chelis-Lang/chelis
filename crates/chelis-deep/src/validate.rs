use crate::ast::Expr;

const VALID_TAGS: &[&str] = &[
    // Module (4)
    "module",
    "import",
    "import-all",
    "export",
    // Declarations (7)
    "def",
    "defsig",
    "deftype",
    "typealias",
    "variant",
    "field",
    "defdim",
    // Expressions (17)
    "fn",
    "app",
    "let",
    "match",
    "arm",
    "if",
    "var",
    "lit",
    "record",
    "access",
    "pipe",
    "block",
    "tuple",
    "tuple-get",
    "par",
    "borrow",
    // Patterns (7)
    "pat-var",
    "pat-lit",
    "pat-ctor",
    "pat-tuple",
    "pat-record",
    "pat-wild",
    "pat-as",
    // Reserved / Phase 1 (1)
    "record-update",
    // Types (8)
    "t-prim",
    "t-fn",
    "t-tensor",
    "t-adt",
    "t-var",
    "t-ref",
    "t-unit",
    "t-tuple",
    // Dimensions (3)
    "d-name",
    "d-var",
    "d-lit",
    // Transforms (6)
    "grad",
    "vmap",
    "jit",
    "realize",
    "cast",
    "copy",
    "handle-effect",
    // Meta (3)
    "quote",
    "unquote",
    "splice",
    // Helpers (5)
    "params",
    "bind",
    "kv",
    "effects",
    "resource",
];

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub kind: WarningKind,
    pub offset: usize,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum WarningKind {
    UnknownTag,
    MissingMetadata,
    Structural,
    Arity,
}

pub fn validate(exprs: &[Expr]) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    for expr in exprs {
        validate_expr(expr, &mut warnings);
    }
    warnings
}

fn validate_expr(expr: &Expr, warnings: &mut Vec<ValidationWarning>) {
    match expr {
        Expr::List(list, span) => {
            if list.elements.is_empty() {
                // Empty list () — valid (e.g., empty guard)
                return;
            }

            if let Some(Expr::Atom(crate::ast::Atom::Symbol(tag), _)) = list.elements.first() {
                // Typed helper forms like `(x {type: ...})` are structural children inside
                // `(params {} ...)`, not standalone tagged Deep nodes.
                let helper_typed_name = list.elements.len() == 2
                    && matches!(list.elements.get(1), Some(Expr::Map(_, _)));
                if helper_typed_name && !VALID_TAGS.contains(&tag.as_str()) {
                    for child in &list.elements {
                        validate_expr(child, warnings);
                    }
                    return;
                }

                match list.elements.get(1) {
                    Some(Expr::Map(_, _)) => {
                        if !VALID_TAGS.contains(&tag.as_str()) {
                            warnings.push(ValidationWarning {
                                kind: WarningKind::UnknownTag,
                                offset: span.offset,
                                message: format!(
                                    "unknown tag '{tag}'. Not in the 61-tag vocabulary"
                                ),
                            });
                        } else {
                            validate_tag_shape(tag, list, span.offset, warnings);
                        }
                    }
                    _ if VALID_TAGS.contains(&tag.as_str()) => warnings.push(ValidationWarning {
                        kind: WarningKind::MissingMetadata,
                        offset: span.offset,
                        message: format!(
                            "tagged node '{tag}' must use canonical 3-tuple form `(tag {{}} ...)`"
                        ),
                    }),
                    _ => {}
                }
            }

            // Recurse into children
            for child in &list.elements {
                validate_expr(child, warnings);
            }
        }
        Expr::MetaExpr(meta, _) => {
            validate_expr(&meta.expr, warnings);
            for (_, v) in &meta.entries {
                validate_expr(v, warnings);
            }
        }
        Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                validate_expr(v, warnings);
            }
        }
        Expr::Atom(_, _) => {} // Atoms are always valid
    }
}

fn validate_tag_shape(
    tag: &str,
    list: &crate::ast::List,
    offset: usize,
    warnings: &mut Vec<ValidationWarning>,
) {
    let child_count = list.elements.len().saturating_sub(2);

    let warn_arity = |warnings: &mut Vec<ValidationWarning>, expected: &str| {
        warnings.push(ValidationWarning {
            kind: WarningKind::Arity,
            offset,
            message: format!(
                "`{tag}` has invalid arity: expected {expected}, found {child_count} child(ren)"
            ),
        });
    };

    match tag {
        "def" => validate_property_def_metadata(list, offset, warnings),
        "deftype" => validate_deftype_invariant_metadata(list, offset, warnings),
        "if" | "arm" if child_count != 3 => {
            warn_arity(warnings, "exactly 3 children");
        }
        "handle-effect" if child_count != 2 => {
            warn_arity(warnings, "exactly 2 children");
        }
        "fn" => {
            if child_count != 2 {
                warn_arity(warnings, "exactly 2 children");
                return;
            }
            if !matches!(
                list.elements.get(2),
                Some(Expr::List(params, _))
                    if matches!(params.elements.first(), Some(Expr::Atom(crate::ast::Atom::Symbol(s), _)) if s == "params")
                        && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
            ) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`fn` must use `(fn {} (params {} ...) body)`".to_string(),
                });
            }
        }
        "let" => {
            if child_count != 2 {
                warn_arity(warnings, "exactly 2 children");
                return;
            }
            if !matches!(
                list.elements.get(2),
                Some(Expr::List(bind, _))
                    if matches!(bind.elements.first(), Some(Expr::Atom(crate::ast::Atom::Symbol(s), _)) if s == "bind")
                        && matches!(bind.elements.get(1), Some(Expr::Map(_, _)))
            ) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`let` must use `(let {} (bind {} ...) body)`".to_string(),
                });
            }
        }
        "app" if child_count < 1 => {
            warn_arity(warnings, "at least 1 child");
        }
        "params"
            if !list.elements.iter().skip(2).all(|child| match child {
                Expr::Atom(crate::ast::Atom::Symbol(_), _) => true,
                Expr::MetaExpr(meta, _) => matches!(
                    meta.expr.as_ref(),
                    Expr::Atom(crate::ast::Atom::Symbol(_), _)
                ),
                Expr::List(inner, _) => {
                    inner.elements.len() == 2
                        && matches!(
                            inner.elements.first(),
                            Some(Expr::Atom(crate::ast::Atom::Symbol(_), _))
                        )
                        && matches!(inner.elements.get(1), Some(Expr::Map(_, _)))
                }
                _ => false,
            }) =>
        {
            warnings.push(ValidationWarning {
                kind: WarningKind::Structural,
                offset,
                message: "`params` must contain bare names or typed-name metadata helpers"
                    .to_string(),
            });
        }
        "bind" if !child_count.is_multiple_of(2) => {
            warnings.push(ValidationWarning {
                kind: WarningKind::Structural,
                offset,
                message: "`bind` must contain name/expression pairs".to_string(),
            });
        }
        "effects"
            if !list.elements.iter().skip(2).all(|child| {
                matches!(child, Expr::Atom(crate::ast::Atom::Symbol(_), _))
                    || matches!(
                        child,
                        Expr::List(inner, _)
                            if matches!(
                                inner.elements.first(),
                                Some(Expr::Atom(crate::ast::Atom::Symbol(tag), _)) if tag == "resource"
                            )
                    )
            }) =>
        {
            warnings.push(ValidationWarning {
                kind: WarningKind::Structural,
                offset,
                message: "`effects` must contain symbols or `(resource {} ...)` entries"
                    .to_string(),
            });
        }
        "resource" if child_count != 1 => {
            warn_arity(warnings, "exactly 1 child");
        }
        _ => {}
    }
}

fn validate_property_def_metadata(
    list: &crate::ast::List,
    offset: usize,
    warnings: &mut Vec<ValidationWarning>,
) {
    let Some(Expr::Map(meta, _)) = list.elements.get(1) else {
        return;
    };
    let is_property = meta.entries.iter().any(|(key, value)| {
        key == "chelis_role"
            && matches!(value, Expr::Atom(crate::ast::Atom::Str(value), _) if value == "property")
    });
    if !is_property {
        return;
    }
    let has_source_kind = meta.entries.iter().any(|(key, value)| {
        key == "property_source_kind" && matches!(value, Expr::Atom(crate::ast::Atom::Str(_), _))
    });
    let has_quantifiers = meta.entries.iter().any(|(key, value)| {
        key == "property_quantifiers"
            && matches!(
                value,
                Expr::List(params, _)
                    if matches!(params.elements.first(), Some(Expr::Atom(crate::ast::Atom::Symbol(tag), _)) if tag == "params")
                        && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
            )
    });
    let has_preconditions = meta.entries.iter().any(|(key, value)| {
        key == "property_preconditions"
            && matches!(
                value,
                Expr::List(tuple, _)
                    if matches!(tuple.elements.first(), Some(Expr::Atom(crate::ast::Atom::Symbol(tag), _)) if tag == "tuple")
                        && matches!(tuple.elements.get(1), Some(Expr::Map(_, _)))
            )
    });
    let body_is_fn = matches!(
        list.elements.get(3),
        Some(Expr::List(body, _))
            if matches!(body.elements.first(), Some(Expr::Atom(crate::ast::Atom::Symbol(tag), _)) if tag == "fn")
    );
    for (ok, message) in [
        (
            has_source_kind,
            "property def metadata must include string `property_source_kind`",
        ),
        (
            has_quantifiers,
            "property def metadata must include `(params {} ...)` `property_quantifiers`",
        ),
        (
            has_preconditions,
            "property def metadata must include `(tuple {} ...)` `property_preconditions`",
        ),
        (body_is_fn, "property def body must be a callable `fn`"),
    ] {
        if !ok {
            warnings.push(ValidationWarning {
                kind: WarningKind::Structural,
                offset,
                message: message.to_string(),
            });
        }
    }
}

/// Structural shape check of the opaque-invariant metadata keys on a
/// `deftype` (RFC D-META, warning-level; mirrors the property-metadata
/// check). When an `invariant` key is present it must be a predicate fn
/// `(fn {} (params {} <one symbol>) <expr>)`, and any
/// `invariant_amenability` key must be one of the four canonical strings.
/// The deeper well-formedness (grammar, value class, amenability match)
/// is the checker's job (chelis-types); this is the cheap shape gate.
fn validate_deftype_invariant_metadata(
    list: &crate::ast::List,
    offset: usize,
    warnings: &mut Vec<ValidationWarning>,
) {
    let Some(Expr::Map(meta, _)) = list.elements.get(1) else {
        return;
    };
    if let Some((_, value)) = meta.entries.iter().find(|(key, _)| key == "invariant")
        && !is_predicate_fn_shape(value)
    {
        warnings.push(ValidationWarning {
            kind: WarningKind::Structural,
            offset,
            message: "`invariant` metadata must be `(fn {} (params {} <binder>) <expr>)`"
                .to_string(),
        });
    }
    if let Some((_, value)) = meta
        .entries
        .iter()
        .find(|(key, _)| key == "invariant_amenability")
        && !is_canonical_amenability(value)
    {
        warnings.push(ValidationWarning {
            kind: WarningKind::Structural,
            offset,
            message: "`invariant_amenability` must be one of \
                      \"linear\"|\"polynomial\"|\"transcendental\"|\"opaque\""
                .to_string(),
        });
    }
}

/// Whether `expr` is a predicate fn `(fn {} (params {} <one symbol>)
/// <body>)`: a `fn` node whose first child is a `params` node holding
/// exactly one bare-symbol binder, and which has a body child.
fn is_predicate_fn_shape(expr: &Expr) -> bool {
    let Expr::List(fn_list, _) = expr else {
        return false;
    };
    let is_fn = matches!(
        fn_list.elements.first(),
        Some(Expr::Atom(crate::ast::Atom::Symbol(t), _)) if t == "fn"
    ) && matches!(fn_list.elements.get(1), Some(Expr::Map(_, _)));
    if !is_fn || fn_list.elements.len() != 4 {
        return false;
    }
    // params node with exactly one bare-symbol binder.
    let Some(Expr::List(params, _)) = fn_list.elements.get(2) else {
        return false;
    };
    matches!(
        params.elements.first(),
        Some(Expr::Atom(crate::ast::Atom::Symbol(t), _)) if t == "params"
    ) && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
        && params.elements.len() == 3
        && matches!(
            params.elements.get(2),
            Some(Expr::Atom(crate::ast::Atom::Symbol(_), _))
        )
}

fn is_canonical_amenability(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Atom(crate::ast::Atom::Str(s), _)
            if matches!(s.as_str(), "linear" | "polynomial" | "transcendental" | "opaque")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, List, MetaMap};
    use crate::span::Span;

    const ZERO: Span = Span { offset: 0, len: 0 };

    fn sym(s: &str) -> Expr {
        Expr::Atom(Atom::Symbol(s.to_string()), ZERO)
    }

    fn empty_map() -> Expr {
        Expr::Map(MetaMap::default(), ZERO)
    }

    fn make_list(elements: Vec<Expr>) -> Expr {
        Expr::List(List { elements }, ZERO)
    }

    #[test]
    fn valid_3tuple_no_warnings() {
        // (var {} x) — valid 3-tuple node
        let node = make_list(vec![sym("var"), empty_map(), sym("x")]);
        let warnings = validate(&[node]);
        assert!(
            warnings.is_empty(),
            "expected no warnings, got: {warnings:?}"
        );
    }

    #[test]
    fn unknown_tag_warning() {
        // (apply {} f x) — "apply" is not in the vocabulary
        let node = make_list(vec![sym("apply"), empty_map(), sym("f"), sym("x")]);
        let warnings = validate(&[node]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0].kind, WarningKind::UnknownTag));
        assert!(warnings[0].message.contains("apply"));
    }

    #[test]
    fn bare_list_not_validated_as_tag() {
        // (a b) — bare structural list, NOT a tagged node (no {} at [1])
        // Should produce no warnings — validator only checks 3-tuple nodes
        let node = make_list(vec![sym("a"), sym("b")]);
        let warnings = validate(&[node]);
        assert!(
            warnings.is_empty(),
            "bare list should not trigger tag validation, got: {warnings:?}"
        );
    }

    #[test]
    fn nested_valid_nodes_no_warnings() {
        // (app {} (var {} f) (var {} x))
        let inner1 = make_list(vec![sym("var"), empty_map(), sym("f")]);
        let inner2 = make_list(vec![sym("var"), empty_map(), sym("x")]);
        let outer = make_list(vec![sym("app"), empty_map(), inner1, inner2]);
        let warnings = validate(&[outer]);
        assert!(
            warnings.is_empty(),
            "expected no warnings, got: {warnings:?}"
        );
    }

    #[test]
    fn legacy_sig_unknown_tag() {
        // (sig {} x i32) — "sig" is not in vocabulary (use "defsig")
        let node = make_list(vec![sym("sig"), empty_map(), sym("x"), sym("i32")]);
        let warnings = validate(&[node]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0].kind, WarningKind::UnknownTag));
        assert!(warnings[0].message.contains("sig"));
    }

    #[test]
    fn legacy_apply_unknown_tag() {
        // (apply {} f x) — "apply" is not in vocabulary (use "app")
        let node = make_list(vec![sym("apply"), empty_map(), sym("f"), sym("x")]);
        let warnings = validate(&[node]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0].kind, WarningKind::UnknownTag));
        assert!(warnings[0].message.contains("apply"));
    }

    #[test]
    fn missing_metadata_is_reported() {
        let node = make_list(vec![sym("if"), sym("cond"), sym("then"), sym("else")]);
        let warnings = validate(&[node]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0].kind, WarningKind::MissingMetadata));
    }

    #[test]
    fn typed_param_meta_expr_is_valid() {
        let node = make_list(vec![
            sym("def"),
            empty_map(),
            sym("f"),
            make_list(vec![
                sym("fn"),
                empty_map(),
                make_list(vec![
                    sym("params"),
                    empty_map(),
                    Expr::MetaExpr(
                        crate::ast::MetaExpr {
                            entries: vec![("type".to_string(), sym("int64"))],
                            expr: Box::new(sym("let")),
                        },
                        ZERO,
                    ),
                ]),
                make_list(vec![sym("var"), empty_map(), sym("let")]),
            ]),
        ]);
        let warnings = validate(&[node]);
        assert!(
            warnings.is_empty(),
            "typed parameter metadata helper should validate cleanly, got {warnings:?}"
        );
    }

    #[test]
    fn arity_is_validated() {
        let node = make_list(vec![sym("if"), empty_map(), sym("cond"), sym("then")]);
        let warnings = validate(&[node]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0].kind, WarningKind::Arity));
    }

    #[test]
    fn t_ref_type_tag_is_valid() {
        let node = make_list(vec![
            sym("t-ref"),
            empty_map(),
            make_list(vec![
                sym("t-tensor"),
                empty_map(),
                make_list(vec![
                    sym("d-lit"),
                    empty_map(),
                    Expr::Atom(Atom::Int(4), ZERO),
                ]),
                make_list(vec![sym("t-prim"), empty_map(), sym("f32")]),
            ]),
        ]);
        let warnings = validate(&[node]);
        assert!(
            warnings.is_empty(),
            "t-ref should be part of the closed Deep type vocabulary: {warnings:?}"
        );
    }

    // ── Opaque-invariant metadata shape (RFC D-META) ─────────────────

    fn structural_messages(src: &str) -> Vec<String> {
        let exprs = crate::parser::parse_str(src).expect("deep parses");
        validate(&exprs)
            .into_iter()
            .filter(|w| matches!(w.kind, WarningKind::Structural))
            .map(|w| w.message)
            .collect()
    }

    #[test]
    fn well_formed_invariant_metadata_passes() {
        let msgs = structural_messages(
            "(deftype {opaque: true, \
                invariant: (fn {} (params {} p) \
                    (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0))), \
                invariant_amenability: \"linear\"} \
                T () (variant {} T (field {} value (t-prim {} f32))))",
        );
        assert!(
            msgs.is_empty(),
            "expected no structural warnings, got: {msgs:?}"
        );
    }

    #[test]
    fn invariant_not_a_fn_warns() {
        let msgs = structural_messages(
            "(deftype {opaque: true, invariant: (var {} p), invariant_amenability: \"linear\"} \
                T () (variant {} T (field {} value (t-prim {} f32))))",
        );
        assert!(
            msgs.iter()
                .any(|m| m.contains("`invariant` metadata must be")),
            "expected invariant-shape warning, got: {msgs:?}"
        );
    }

    #[test]
    fn invariant_multi_binder_warns() {
        // params node with two binders is not the predicate-fn shape.
        let msgs = structural_messages(
            "(deftype {opaque: true, invariant: (fn {} (params {} p q) (var {} p)), \
                invariant_amenability: \"linear\"} \
                T () (variant {} T (field {} value (t-prim {} f32))))",
        );
        assert!(
            msgs.iter()
                .any(|m| m.contains("`invariant` metadata must be")),
            "expected invariant-shape warning for multi-binder, got: {msgs:?}"
        );
    }

    #[test]
    fn invalid_amenability_string_warns() {
        let msgs = structural_messages(
            "(deftype {opaque: true, \
                invariant: (fn {} (params {} p) (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0))), \
                invariant_amenability: \"nonlinear\"} \
                T () (variant {} T (field {} value (t-prim {} f32))))",
        );
        assert!(
            msgs.iter().any(|m| m.contains("invariant_amenability")),
            "expected amenability-vocabulary warning, got: {msgs:?}"
        );
    }

    #[test]
    fn deftype_without_invariant_metadata_unaffected() {
        let msgs = structural_messages(
            "(deftype {opaque: true} T () (variant {} T (field {} value (t-prim {} f32))))",
        );
        assert!(
            msgs.is_empty(),
            "plain opaque deftype must not warn, got: {msgs:?}"
        );
    }
}
