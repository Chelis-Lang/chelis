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
    // Expressions (15)
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
    // Types (7)
    "t-prim",
    "t-fn",
    "t-tensor",
    "t-adt",
    "t-var",
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
    // Meta (3)
    "quote",
    "unquote",
    "splice",
    // Helpers (3)
    "params",
    "bind",
    "kv",
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
                                    "unknown tag '{tag}' — not in the 56-tag vocabulary"
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
        "if" | "arm" => {
            if child_count != 3 {
                warn_arity(warnings, "exactly 3 children");
            }
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
        "app" => {
            if child_count < 1 {
                warn_arity(warnings, "at least 1 child");
            }
        }
        "params" => {
            if !list.elements.iter().skip(2).all(|child| match child {
                Expr::Atom(crate::ast::Atom::Symbol(_), _) => true,
                Expr::List(inner, _) => {
                    inner.elements.len() == 2
                        && matches!(
                            inner.elements.first(),
                            Some(Expr::Atom(crate::ast::Atom::Symbol(_), _))
                        )
                        && matches!(inner.elements.get(1), Some(Expr::Map(_, _)))
                }
                _ => false,
            }) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`params` must contain bare names or typed-name helper pairs"
                        .to_string(),
                });
            }
        }
        "bind" => {
            if !child_count.is_multiple_of(2) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`bind` must contain name/expression pairs".to_string(),
                });
            }
        }
        _ => {}
    }
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
    fn arity_is_validated() {
        let node = make_list(vec![sym("if"), empty_map(), sym("cond"), sym("then")]);
        let warnings = validate(&[node]);
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0].kind, WarningKind::Arity));
    }
}
