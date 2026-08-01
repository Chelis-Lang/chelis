use crate::ast::Expr;
use crate::tag::DeepTag;

/// Canonical closed Deep tag vocabulary (`spec/03-deep-syntax.md` §2),
/// derived from the single [`DeepTag`] declaration so the string list and
/// the enum cannot drift (chelis#731 Phase 3). `chelis-lint` mirrors it
/// and asserts equality against it in tests.
pub const VALID_TAGS: &[&str] = &DeepTag::ALL_STRS;

/// Decode-once invariant walker (chelis#731 Phase 3, PERMANENT): return
/// the first location where a list's element 0 carries a closed-vocabulary
/// tag as a raw `Atom::Name` string instead of a stamped `Atom::Tag`.
/// After the parser's stamping pass (and the typed producer constructors),
/// no parsed or desugared tree may contain one; a `Some` here means a
/// producer bypassed decode-once and its node would silently miss every
/// typed dispatch arm. Keep this callable from other crates' permanent
/// suites (the parser-side and desugar-side invariant tests).
pub fn find_raw_vocabulary_tag(exprs: &[Expr]) -> Option<String> {
    fn walk(expr: &Expr) -> Option<String> {
        match expr {
            Expr::List(list, _) => {
                if let Some(symbol) = list.unknown_tag_symbol()
                    && DeepTag::parse(symbol).is_some()
                {
                    return Some(symbol.to_string());
                }
                list.elements.iter().find_map(walk)
            }
            Expr::Map(map, _) => map.entries.iter().find_map(|(_, value)| walk(value)),
            Expr::MetaExpr(meta, _) => {
                walk(&meta.expr).or_else(|| meta.entries.iter().find_map(|(_, value)| walk(value)))
            }
            Expr::Atom(_, _) => None,
            // Stamped variants: Node children are already validated;
            // BareList/UnknownForm don't carry raw vocabulary tags by
            // construction.
            Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(..) => None,
        }
    }
    exprs.iter().find_map(walk)
}

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
        // chelis#858: a top-level expression must be a tagged Deep node.
        // The pest-side executable grammar (`chelis-validate`) already
        // rejects any top-level non-node; this validator silently accepted
        // an untagged list like `((var {} f) (var {} x))`, a rejection-
        // parity divergence on exactly the input class whose checker-side
        // silent skip motivated the issue.
        if let Expr::List(list, span) = expr
            && !list.elements.is_empty()
            && list.tag().is_none()
            && list.unknown_tag_symbol().is_none()
        {
            warnings.push(ValidationWarning {
                kind: WarningKind::Structural,
                offset: span.offset,
                message: "top-level expression must be a canonical `(tag {} ...)` node                           (chelis#858)"
                    .to_string(),
            });
        }
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

            // Decode-once (chelis#731 Phase 3): the parser already stamped
            // vocabulary tags as `Atom::Tag`, so tag identity is read from
            // the typed accessor; the element-0 symbol survives only for
            // non-vocabulary heads (unknown tags, typed-name helpers, bare
            // structural lists).
            let deep_tag = list.tag();
            if deep_tag.is_some() || list.unknown_tag_symbol().is_some() {
                // Typed helper forms like `(x {type: ...})` are structural children inside
                // `(params {} ...)`, not standalone tagged Deep nodes.
                let helper_typed_name = list.elements.len() == 2
                    && matches!(list.elements.get(1), Some(Expr::Map(_, _)));
                if helper_typed_name && deep_tag.is_none() {
                    for child in &list.elements {
                        validate_expr(child, warnings);
                    }
                    return;
                }

                match (list.elements.get(1), deep_tag) {
                    (Some(Expr::Map(_, _)), Some(deep_tag)) => {
                        validate_tag_shape(deep_tag, list, span.offset, warnings);
                    }
                    (Some(Expr::Map(_, _)), None) => {
                        // The raw-string boundary (checker_totality.md
                        // §C1.2): the string never decoded into the closed
                        // vocabulary, so the loud unknown-tag arm owns it.
                        let tag = list.unknown_tag_symbol().unwrap_or("<none>");
                        warnings.push(ValidationWarning {
                            kind: WarningKind::UnknownTag,
                            offset: span.offset,
                            message: format!("unknown tag '{tag}'. Not in the 62-tag vocabulary"),
                        });
                    }
                    (_, Some(deep_tag)) => warnings.push(ValidationWarning {
                        kind: WarningKind::MissingMetadata,
                        offset: span.offset,
                        message: format!(
                            "tagged node '{}' must use canonical 3-tuple form `(tag {{}} ...)`",
                            deep_tag.as_str()
                        ),
                    }),
                    (_, None) => {}
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
        // Stamped variants: these are produced by stamp_to_typed and are
        // structurally valid by construction. No further validation needed.
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(..) => {}
    }
}

/// Per-tag shape validation over the closed vocabulary. The match is
/// exhaustive with no `_` arm (chelis#731 Phase 3, checker_totality.md
/// §C4.2): a 63rd `DeepTag` variant fails to compile here until it gets an
/// explicit shape disposition.
fn validate_tag_shape(
    deep_tag: DeepTag,
    list: &crate::ast::List,
    offset: usize,
    warnings: &mut Vec<ValidationWarning>,
) {
    let tag = deep_tag.as_str();
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

    match deep_tag {
        DeepTag::Def => validate_property_def_metadata(list, offset, warnings),
        DeepTag::Deftype => validate_deftype_invariant_metadata(list, offset, warnings),
        DeepTag::If | DeepTag::Arm => {
            if child_count != 3 {
                warn_arity(warnings, "exactly 3 children");
            }
        }
        DeepTag::HandleEffect => {
            if child_count != 2 {
                warn_arity(warnings, "exactly 2 children");
            }
        }
        DeepTag::Fn => {
            if child_count != 2 {
                warn_arity(warnings, "exactly 2 children");
                return;
            }
            if !matches!(
                list.elements.get(2),
                Some(Expr::List(params, _))
                    if params.tag() == Some(DeepTag::Params)
                        && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
            ) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`fn` must use `(fn {} (params {} ...) body)`".to_string(),
                });
            }
        }
        DeepTag::Let => {
            if child_count != 2 {
                warn_arity(warnings, "exactly 2 children");
                return;
            }
            if !matches!(
                list.elements.get(2),
                Some(Expr::List(bind, _))
                    if bind.tag() == Some(DeepTag::Bind)
                        && matches!(bind.elements.get(1), Some(Expr::Map(_, _)))
            ) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`let` must use `(let {} (bind {} ...) body)`".to_string(),
                });
            }
        }
        DeepTag::App => {
            if child_count < 1 {
                warn_arity(warnings, "at least 1 child");
            }
        }
        DeepTag::Params => {
            let all_names = list.elements.iter().skip(2).all(|child| match child {
                Expr::Atom(crate::ast::Atom::Name(_), _) => true,
                Expr::MetaExpr(meta, _) => matches!(
                    meta.expr.as_ref(),
                    Expr::Atom(crate::ast::Atom::Name(_), _)
                ),
                Expr::List(inner, _) => {
                    inner.elements.len() == 2
                        && matches!(
                            inner.elements.first(),
                            Some(Expr::Atom(crate::ast::Atom::Name(_), _))
                        )
                        && matches!(inner.elements.get(1), Some(Expr::Map(_, _)))
                }
                _ => false,
            });
            if !all_names {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`params` must contain bare names or typed-name metadata helpers"
                        .to_string(),
                });
            }
        }
        DeepTag::Bind => {
            if !child_count.is_multiple_of(2) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`bind` must contain name/expression pairs".to_string(),
                });
            }
        }
        DeepTag::Effects => {
            let all_entries = list.elements.iter().skip(2).all(|child| {
                matches!(child, Expr::Atom(crate::ast::Atom::Name(_), _))
                    || matches!(
                        child,
                        Expr::List(inner, _)
                            if inner.tag() == Some(DeepTag::Resource)
                    )
            });
            if !all_entries {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`effects` must contain symbols or `(resource {} ...)` entries"
                        .to_string(),
                });
            }
        }
        DeepTag::Resource => {
            if child_count != 1 {
                warn_arity(warnings, "exactly 1 child");
            }
        }
        // No additional shape constraint at this validator: these tags'
        // arity/shape rules are owned by the type checker (spec/03
        // §2.5.1/§2.6, §8.2) or by their enclosing form. Listed explicitly
        // rather than wildcarded so a 63rd tag forces a decision here.
        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Defsig
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Match
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Pipe
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Kv => {}
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
                    if params.tag() == Some(DeepTag::Params)
                        && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
            )
    });
    let has_preconditions = meta.entries.iter().any(|(key, value)| {
        key == "property_preconditions"
            && matches!(
                value,
                Expr::List(tuple, _)
                    if tuple.tag() == Some(DeepTag::Tuple)
                        && matches!(tuple.elements.get(1), Some(Expr::Map(_, _)))
            )
    });
    let body_is_fn = matches!(
        list.elements.get(3),
        Some(Expr::List(body, _))
            if body.tag() == Some(DeepTag::Fn)
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
    let is_fn = fn_list.tag() == Some(DeepTag::Fn)
        && matches!(fn_list.elements.get(1), Some(Expr::Map(_, _)));
    if !is_fn || fn_list.elements.len() != 4 {
        return false;
    }
    // params node with exactly one bare-symbol binder.
    let Some(Expr::List(params, _)) = fn_list.elements.get(2) else {
        return false;
    };
    params.tag() == Some(DeepTag::Params)
        && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
        && params.elements.len() == 3
        && matches!(
            params.elements.get(2),
            Some(Expr::Atom(crate::ast::Atom::Name(_), _))
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
        Expr::Atom(Atom::Name(s.to_string()), ZERO)
    }

    fn empty_map() -> Expr {
        Expr::Map(MetaMap::default(), ZERO)
    }

    fn make_list(elements: Vec<Expr>) -> Expr {
        let mut expr = Expr::List(List { elements }, ZERO);
        // Mirror the parser's decode-once stamping so these hand-built
        // trees match what every real consumer sees.
        crate::parser::stamp_tags(std::slice::from_mut(&mut expr));
        expr
    }

    /// PERMANENT decode-once invariant (chelis#731 Phase 3): a parsed
    /// tree never carries a vocabulary tag as a raw element-0 string, so
    /// a stale `Atom::Name` tag match can never half-work again.
    #[test]
    fn parsed_trees_carry_no_raw_vocabulary_tag_strings() {
        let source = "(module {} m\n  (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n  (deftype {opaque: true, invariant: (fn {} (params {} p) (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0)))} T () (variant {} T (field {} value (t-prim {} f32))))\n  (def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (var {} x)))))";
        let exprs = crate::parser::parse_str(source).expect("deep parses");
        assert_eq!(
            find_raw_vocabulary_tag(&exprs),
            None,
            "the parser must stamp every vocabulary tag, including inside metadata values"
        );
        // Negative control: an unstamped hand-built tree IS caught.
        let raw = Expr::List(
            List {
                elements: vec![sym("var"), empty_map(), sym("x")],
            },
            ZERO,
        );
        assert_eq!(find_raw_vocabulary_tag(&[raw]).as_deref(), Some("var"));
    }

    /// chelis#858 rejection parity: an untagged top-level list is a
    /// structural error here, matching the pest-side executable grammar
    /// (`chelis-validate`), which always rejected it. Both polarities:
    /// nested bare lists (empty guards, `loc` metadata values) stay legal.
    #[test]
    fn untagged_top_level_list_is_rejected() {
        let exprs = crate::parser::parse_str("((var {} f) (var {} x))").expect("lenient parse");
        let warnings = validate(&exprs);
        assert!(
            warnings
                .iter()
                .any(|w| matches!(w.kind, WarningKind::Structural)
                    && w.message
                        .contains("top-level expression must be a canonical")),
            "untagged top-level list must be rejected; got {warnings:?}"
        );
    }

    #[test]
    fn nested_bare_lists_stay_legal_after_858() {
        // The empty guard `()` inside an arm is a NESTED bare list and must
        // stay valid; only the top level requires a tagged node.
        let exprs = crate::parser::parse_str(
            "(def {} f (match {} (var {} x) (arm {} (pat-wild {}) () (var {} x))))",
        )
        .expect("lenient parse");
        let warnings = validate(&exprs);
        assert!(
            warnings.is_empty(),
            "nested empty guard must stay legal; got {warnings:?}"
        );
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
