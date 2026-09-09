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
    exprs.iter().find_map(|expr| walk(expr, true))
}

/// The construction-time half of the decode-once scan (chelis#1109):
/// identical to [`find_raw_vocabulary_tag`] except that it stops at a
/// stamped [`crate::node::Node`] boundary instead of re-walking that
/// subtree.
///
/// Soundness is inductive, not an approximation. A `Node` exists only if
/// `Node::validate` accepted it — `try_new`, `new`, and `Deserialize` all
/// route through it, `Node`'s fields are private to its module, and every
/// post-construction mutator (`try_replace_meta`, `try_replace_child`,
/// `try_replace_children`) revalidates the whole candidate node before it
/// commits. So each `Expr::Node` below the node under construction already
/// passed this same scan over its own metadata and children, and by
/// induction over construction order its whole subtree is free of raw
/// vocabulary tags. Re-walking it is pure repetition, and repeating it at
/// every ancestor made bottom-up stamping quadratic in nesting depth.
///
/// Non-`Node` carriers (`List`, `BareList`, `Map`, `MetaExpr`,
/// `UnknownForm`) carry no such guarantee — nothing validated them — so
/// the scan still descends through them to any depth, including into
/// metadata values.
pub(crate) fn find_raw_vocabulary_tag_below_gate(exprs: &[Expr]) -> Option<String> {
    exprs.iter().find_map(|expr| walk(expr, false))
}

fn walk_metadata(meta: &crate::Metadata, descend_into_nodes: bool) -> Option<String> {
    let mut result = None;
    meta.visit_expressions(&mut |value, _| {
        if result.is_none() {
            result = walk(value, descend_into_nodes);
        }
    });
    result
}
fn walk(expr: &Expr, descend_into_nodes: bool) -> Option<String> {
    match expr {
        Expr::List(list, _) => {
            if let Some(symbol) = list.unknown_tag_symbol()
                && DeepTag::parse(symbol).is_some()
            {
                return Some(symbol.to_string());
            }
            list.elements
                .iter()
                .find_map(|v| walk(v, descend_into_nodes))
        }
        Expr::Map(meta, _) => walk_metadata(meta, descend_into_nodes),
        Expr::MetaExpr(meta, _) => walk(&meta.expr, descend_into_nodes)
            .or_else(|| walk_metadata(&meta.metadata, descend_into_nodes)),
        Expr::Atom(..) => None,
        Expr::Node(node, _) => {
            if !descend_into_nodes {
                return None;
            }
            walk_metadata(node.meta(), descend_into_nodes).or_else(|| {
                node.children_slice()
                    .iter()
                    .find_map(|v| walk(v, descend_into_nodes))
            })
        }
        Expr::BareList(items, _) => items.iter().find_map(|v| walk(v, descend_into_nodes)),
        Expr::UnknownForm(data) => walk_metadata(&data.meta, descend_into_nodes).or_else(|| {
            data.children
                .iter()
                .find_map(|v| walk(v, descend_into_nodes))
        }),
    }
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

impl WarningKind {
    /// The governed wire spelling for this kind (chelis#886).
    ///
    /// A wire spelling a consumer matches on, not a rendering of the Rust
    /// identifier. `check_snippet` used `{:?}` here, which made the wire a
    /// function of the variant's name: a rename moved it with nothing
    /// objecting, which is the hazard chelis#886 exists to remove.
    /// `every_warning_kind_projects_to_its_pinned_spelling` restates all four
    /// spellings independently, so a rename fails by name instead.
    ///
    /// "Published" would overstate it: nothing normative governs these four
    /// strings AS WIRE SPELLINGS. Two of them do occur in the spec as prose --
    /// `spec/03-deep-syntax.md` §8.1 "Structural Validation" and §8.2 "Arity
    /// Validation" name the categories these variants report -- but no text
    /// anywhere enumerates the four as diagnostic identities a consumer may
    /// match on. The repo's structural mechanism for exactly that concern
    /// (`chelis-vocab::DiagnosticKind`, `[05-UNS-6]`, and the diagnostic-kind
    /// mutation oracle, which targets only `chelis-compiler-api` and
    /// `chelis-vocab`) does not cover `WarningKind`. Routing it there is the
    /// real fix and is a separate change; this table plus its test is the
    /// local one.
    pub fn wire_name(&self) -> &'static str {
        match self {
            Self::UnknownTag => "UnknownTag",
            Self::MissingMetadata => "MissingMetadata",
            Self::Structural => "Structural",
            Self::Arity => "Arity",
        }
    }
}

pub fn validate(exprs: &[Expr]) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    if let Err(error) = crate::metadata::validate_metadata(exprs) {
        warnings.push(ValidationWarning {
            kind: WarningKind::Structural,
            offset: error.span.offset,
            message: error.to_string(),
        });
    }
    for expr in exprs {
        // chelis#858: a top-level expression must be a tagged Deep node.
        // The pest-side executable grammar (`chelis-validate`) already
        // rejects any top-level non-node; this validator silently accepted
        // an untagged list like `((var {} f) (var {} x))`, a rejection-
        // parity divergence on exactly the input class whose checker-side
        // silent skip motivated the issue.
        match expr {
            Expr::List(list, span)
                if !list.elements.is_empty()
                    && list.tag().is_none()
                    && list.unknown_tag_symbol().is_none() =>
            {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset: span.offset,
                    message: "top-level expression must be a canonical `(tag {} ...)` node                           (chelis#858)"
                        .to_string(),
                });
            }
            // [03-ROLE-3] keeps a tag-word head without a metadata map a
            // structural BareList — an import name list `(copy fill)` is a
            // real program, so the stamp pass may not reinterpret the list
            // by its head alone. This warning is the recorded lint
            // mitigation for the other producer intent behind the same byte
            // shape: a node whose author forgot the `{}` at element 1. It
            // stays advisory precisely because the spelling is ambiguous.
            Expr::BareList(elems, span)
                if matches!(
                    elems.first(),
                    Some(Expr::Atom(crate::ast::Atom::Name(head), _))
                        if DeepTag::parse(head).is_some()
                ) && !matches!(elems.get(1), Some(Expr::Map(_, _))) =>
            {
                let Expr::Atom(crate::ast::Atom::Name(head), _) = &elems[0] else {
                    unreachable!("guard requires a name head")
                };
                warnings.push(ValidationWarning {
                    kind: WarningKind::MissingMetadata,
                    offset: span.offset,
                    message: format!("`{head}` node must carry a metadata map at index 1"),
                });
            }
            Expr::BareList(elems, span)
                if matches!(
                    (elems.first(), elems.get(1)),
                    (
                        Some(Expr::Atom(crate::ast::Atom::Name(_), _)),
                        Some(Expr::Map(_, _))
                    )
                ) =>
            {
                let Expr::Atom(crate::ast::Atom::Name(head), _) = &elems[0] else {
                    unreachable!("guard requires a name head")
                };
                warnings.push(ValidationWarning {
                    kind: WarningKind::UnknownTag,
                    offset: span.offset,
                    message: format!(
                        "unknown tag '{head}'. Not in the {}-tag vocabulary",
                        DeepTag::COUNT
                    ),
                });
            }
            Expr::BareList(elems, span)
                if !elems.is_empty()
                    && !matches!(
                        elems.first(),
                        Some(Expr::Atom(crate::ast::Atom::Name(_), _))
                    ) =>
            {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset: span.offset,
                    message: "top-level expression must be a canonical `(tag {} ...)` node                           (chelis#858)"
                        .to_string(),
                });
            }
            _ => {}
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
            meta.metadata
                .visit_expressions(&mut |v, _| validate_expr(v, warnings));
        }
        Expr::Map(map, _) => {
            map.visit_expressions(&mut |v, _| validate_expr(v, warnings));
        }
        Expr::Atom(_, _) => {} // Atoms are always valid
        // Stamped Node variants: structurally valid by construction (arity
        // and role gates are checked at Node::new/try_new). However, we
        // still need to validate semantic metadata shapes (e.g., invariant
        // on deftype) and recurse into children for validation.
        Expr::Node(node, span) => {
            validate_node_tag_shape(node.tag(), node, span.offset, warnings);
            // Recurse into children
            for child in node.children_slice() {
                validate_expr(child, warnings);
            }
            // Recurse into metadata values
            node.meta()
                .visit_expressions(&mut |v, _| validate_expr(v, warnings));
        }
        Expr::BareList(elems, _) => {
            for child in elems {
                validate_expr(child, warnings);
            }
        }
        Expr::UnknownForm(data) => {
            warnings.push(ValidationWarning {
                kind: WarningKind::UnknownTag,
                offset: data.span.offset,
                message: format!(
                    "unknown tag '{}'. Not in the {}-tag vocabulary",
                    data.head,
                    DeepTag::COUNT
                ),
            });
            data.meta
                .visit_expressions(&mut |v, _| validate_expr(v, warnings));
            for child in &data.children {
                validate_expr(child, warnings);
            }
        }
    }
}

/// Per-tag metadata validation for Node-form expressions. Arity is already
/// enforced at construction time (`Node::try_new`), so this only runs the
/// metadata-shape checks that `validate_tag_shape` performs on List-form.
fn validate_node_tag_shape(
    deep_tag: DeepTag,
    node: &crate::node::Node,
    offset: usize,
    warnings: &mut Vec<ValidationWarning>,
) {
    match deep_tag {
        DeepTag::Def => {}
        DeepTag::Deftype => {
            validate_type_parameter_list("deftype", node.children_slice().get(1), offset, warnings);
        }
        DeepTag::Typealias => validate_type_parameter_list(
            "typealias",
            node.children_slice().get(1),
            offset,
            warnings,
        ),
        DeepTag::Fn => {
            if !matches!(
                node.children_slice().first(),
                Some(Expr::Node(params, _)) if params.tag() == DeepTag::Params
            ) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`fn` must use `(fn {} (params {} ...) body)`".to_string(),
                });
            }
        }
        DeepTag::Let => {
            if !matches!(
                node.children_slice().first(),
                Some(Expr::Node(bind, _)) if bind.tag() == DeepTag::Bind
            ) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`let` must use `(let {} (bind {} ...) body)`".to_string(),
                });
            }
        }
        DeepTag::Params => {
            let all_names = node.children_slice().iter().all(|child| match child {
                Expr::Atom(crate::ast::Atom::Name(_), _) => true,
                Expr::MetaExpr(meta, _) => {
                    matches!(meta.expr.as_ref(), Expr::Atom(crate::ast::Atom::Name(_), _))
                }
                Expr::BareList(elements, _) => {
                    elements.len() == 2
                        && matches!(
                            elements.first(),
                            Some(Expr::Atom(crate::ast::Atom::Name(_), _))
                        )
                        && matches!(elements.get(1), Some(Expr::Map(_, _)))
                }
                Expr::List(list, _) => {
                    list.elements.len() == 2
                        && matches!(
                            list.elements.first(),
                            Some(Expr::Atom(crate::ast::Atom::Name(_), _))
                        )
                        && matches!(list.elements.get(1), Some(Expr::Map(_, _)))
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
            if !node.child_count().is_multiple_of(2) {
                warnings.push(ValidationWarning {
                    kind: WarningKind::Structural,
                    offset,
                    message: "`bind` must contain name/expression pairs".to_string(),
                });
            }
        }
        DeepTag::Effects => {
            let all_entries = node.children_slice().iter().all(|child| {
                matches!(child, Expr::Atom(crate::ast::Atom::Name(_), _))
                    || matches!(
                        child,
                        Expr::Node(resource, _) if resource.tag() == DeepTag::Resource
                    )
                    || matches!(
                        child,
                        Expr::List(resource, _)
                            if resource.tag() == Some(DeepTag::Resource)
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
        // Node construction owns these tags' arity and role checks. Listing
        // every remaining tag explicitly keeps this validator a compile-time
        // disposition ratchet when the vocabulary grows.
        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Defsig
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Match
        | DeepTag::Arm
        | DeepTag::If
        | DeepTag::App
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
        | DeepTag::HandleEffect
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
        | DeepTag::Kv
        | DeepTag::Resource => {}
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
        DeepTag::Def => {}
        DeepTag::Deftype => {
            validate_type_parameter_list("deftype", list.elements.get(3), offset, warnings);
        }
        DeepTag::Typealias => {
            validate_type_parameter_list("typealias", list.elements.get(3), offset, warnings)
        }
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
            let params_ok = match list.elements.get(2) {
                Some(Expr::List(params, _)) => {
                    params.tag() == Some(DeepTag::Params)
                        && matches!(params.elements.get(1), Some(Expr::Map(_, _)))
                }
                Some(Expr::Node(node, _)) => node.tag() == DeepTag::Params,
                _ => false,
            };
            if !params_ok {
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
            let bind_ok = match list.elements.get(2) {
                Some(Expr::List(bind, _)) => {
                    bind.tag() == Some(DeepTag::Bind)
                        && matches!(bind.elements.get(1), Some(Expr::Map(_, _)))
                }
                Some(Expr::Node(node, _)) => node.tag() == DeepTag::Bind,
                _ => false,
            };
            if !bind_ok {
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
                Expr::MetaExpr(meta, _) => {
                    matches!(meta.expr.as_ref(), Expr::Atom(crate::ast::Atom::Name(_), _))
                }
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

fn validate_type_parameter_list(
    declaration: &str,
    params: Option<&Expr>,
    offset: usize,
    warnings: &mut Vec<ValidationWarning>,
) {
    let elements = match params {
        Some(Expr::BareList(elements, _)) => Some(elements.as_slice()),
        Some(Expr::List(list, _)) => Some(list.elements.as_slice()),
        _ => None,
    };
    if !elements.is_some_and(|elements| {
        elements
            .iter()
            .all(|expr| matches!(expr, Expr::Atom(crate::ast::Atom::Name(_), _)))
    }) {
        warnings.push(ValidationWarning {
            kind: WarningKind::Structural,
            offset,
            message: format!(
                "`{declaration}` must use a type-parameter list of bare names at child 1"
            ),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, List, Metadata};
    use crate::span::Span;

    const ZERO: Span = Span { offset: 0, len: 0 };

    fn sym(s: &str) -> Expr {
        Expr::Atom(Atom::Name(s.to_string()), ZERO)
    }

    fn empty_map() -> Expr {
        Expr::Map(Metadata::default(), ZERO)
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

        // Negative controls for every stamped carrier: the permanent oracle
        // must inspect metadata and structural children rather than treating
        // the carrier tag itself as proof that the whole subtree was stamped.
        //
        // For the gated `Node` carrier the guarantee is stronger than "the
        // oracle finds it": the constructor consults this same traversal, so
        // a Node hiding a raw vocabulary tag in its metadata cannot be built
        // at all (chelis#731 Phase 3 successor acceptance).
        let raw_in_meta = Expr::List(
            List {
                elements: vec![sym("var"), empty_map(), sym("hidden")],
            },
            ZERO,
        );
        let metadata = Metadata::default();
        let error = crate::annotations::RuntimeExpression::try_new(raw_in_meta).unwrap_err();
        assert!(error.to_string().contains("var"));
        assert!(
            metadata.is_empty(),
            "no carrier can receive the rejected annotation"
        );
    }

    #[test]
    fn strict_node_validation_preserves_helper_shape_contracts() {
        crate::parser::parse_str_strict("(let {} (bind {}) (var {} x))")
            .expect("an empty sequential binding list is valid");

        let odd_bind = crate::parser::parse_str_strict(
            "(let {} (bind {} x (lit {type: (t-prim {} int64)} 1) y) (var {} x))",
        )
        .expect_err("an odd binding list must be rejected");
        assert!(
            odd_bind.to_string().contains("name/expression pairs"),
            "the strict Node path must preserve the bind-pair diagnostic: {odd_bind}"
        );

        let invalid_param = crate::parser::parse_str_strict(
            "(fn {} (params {} 1) (lit {type: (t-prim {} int64)} 1))",
        )
        .expect_err("a parameter must be a bare name or typed helper");
        assert!(
            invalid_param.to_string().contains("params"),
            "the strict Node path must preserve the parameter-shape diagnostic: {invalid_param}"
        );
    }

    #[test]
    fn strict_parser_rejects_vocabulary_head_without_metadata() {
        let error = crate::parser::parse_str_strict("(var x)")
            .expect_err("a vocabulary-headed form cannot fall through as a structural bare list");
        assert!(
            error.to_string().contains("metadata map"),
            "the rejection must identify the missing canonical metadata slot: {error}"
        );
    }

    #[test]
    fn strict_parser_rejects_unknown_forms_and_walks_their_metadata() {
        let err = crate::parser::parse_str_strict(
            "(def {} f (mystery {payload: (tuple {} (var {} x))} (var {} y)))",
        )
        .expect_err("strict mode must reject an UnknownForm");
        assert!(
            err.to_string().contains("unknown tag 'mystery'"),
            "the outer unknown form must own the first diagnostic: {err}"
        );

        let nested = Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
            head: "nested-unknown".to_string(),
            meta: Metadata::default(),
            children: Vec::new(),
            span: ZERO,
        }));
        let outer = Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
            head: "outer-unknown".to_string(),
            meta: {
                let mut metadata = Metadata::default();
                metadata
                    .insert(crate::annotations::MetadataValue::PropertySeed(
                        crate::annotations::RuntimeExpression::try_new(nested).unwrap(),
                    ))
                    .unwrap();
                metadata
            },
            children: Vec::new(),
            span: ZERO,
        }));
        let warnings = validate(&[outer]);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.message.contains("outer-unknown")),
            "the outer UnknownForm must be diagnosed: {warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.message.contains("nested-unknown")),
            "nested UnknownForm metadata must be traversed: {warnings:?}"
        );
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
                            metadata: Metadata::from(crate::annotations::MetadataValue::Type(
                                crate::annotations::TypeSyntax::try_new(make_list(vec![
                                    sym("t-prim"),
                                    empty_map(),
                                    sym("int64"),
                                ]))
                                .unwrap(),
                            )),
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
        match crate::parser::parse_str(src) {
            Ok(exprs) => validate(&exprs)
                .into_iter()
                .filter(|w| matches!(w.kind, WarningKind::Structural))
                .map(|w| w.message)
                .collect(),
            Err(error) => vec![error.to_string()],
        }
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
                .any(|m| m.contains("metadata `invariant` requires a one-binder fn")),
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
                .any(|m| m.contains("metadata `invariant` requires a one-binder fn")),
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
