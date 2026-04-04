use crate::ast::Expr;

const VALID_TAGS: &[&str] = &[
    // Module (4)
    "module",
    "import",
    "import-all",
    "export",
    // Declarations (6)
    "def",
    "defsig",
    "deftype",
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
    // Patterns (6)
    "pat-var",
    "pat-lit",
    "pat-ctor",
    "pat-record",
    "pat-wild",
    "pat-as",
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

            // Only validate lists that look like 3-tuple nodes:
            // element[0] is a symbol AND element[1] is a Map.
            // Bare structural lists like (a) or (a b) are not tagged nodes.
            let is_tagged_node = list.elements.len() >= 2
                && matches!(
                    list.elements.first(),
                    Some(Expr::Atom(crate::ast::Atom::Symbol(_), _))
                )
                && matches!(list.elements.get(1), Some(Expr::Map(_, _)));

            if is_tagged_node
                && let Some(Expr::Atom(crate::ast::Atom::Symbol(tag), _)) = list.elements.first()
                && !VALID_TAGS.contains(&tag.as_str())
            {
                warnings.push(ValidationWarning {
                    kind: WarningKind::UnknownTag,
                    offset: span.offset,
                    message: format!("unknown tag '{tag}' — not in the 53-tag vocabulary"),
                });
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
}
