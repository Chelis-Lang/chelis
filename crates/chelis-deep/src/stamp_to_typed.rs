//! Role-directed stamp pass: `Vec<RawExpr>` → `Result<Vec<Expr>, StampError>`.
//!
//! Walks top-down, consulting `child_stamp_role` at each child to decide
//! whether a list decodes as a Node, BareList, or UnknownForm.

use crate::ast::{Atom, Expr, MetaMap};
use crate::node::Node;
use crate::raw::{RawAtom, RawExpr};
use crate::role::{
    BypassExpectation, ChildStampRole, bypass_child_expectation, child_stamp_role,
    is_declaration_tag, is_pattern_tag,
};
use crate::span::Span;
use crate::tag::DeepTag;

/// An error from the stamp pass — structurally invalid input that cannot
/// produce a well-typed AST.
#[derive(Debug, Clone, PartialEq)]
pub struct StampError {
    pub kind: StampErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StampErrorKind {
    /// A bare name appeared at a RuntimeExpr position.
    NameAtExprSlot { name: String },
    /// A list at a Type position had an undecodable head.
    UndecodableTypeHead { head: String },
    /// A bypass slot required a declaration but the head was not one.
    RequiresDeclaration { head: String },
    /// A bypass slot required a specific tag but got something else.
    RequiresTag { expected: DeepTag, got: String },
    /// A bypass slot required a pattern but the head was not one.
    RequiresPattern { head: String },
    /// A list was missing the metadata map at element 1.
    MissingMetaMap,
    /// A list was empty where a tagged node was expected.
    EmptyList,
    /// Node construction failed (arity or role violation).
    NodeError(crate::node::NodeError),
}

impl std::fmt::Display for StampError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            StampErrorKind::NameAtExprSlot { name } => {
                write!(
                    f,
                    "bare name `{name}` at expression slot; use `(var {{}} {name})`"
                )
            }
            StampErrorKind::UndecodableTypeHead { head } => {
                write!(f, "undecodable type head `{head}`")
            }
            StampErrorKind::RequiresDeclaration { head } => {
                write!(f, "expected declaration, got `{head}`")
            }
            StampErrorKind::RequiresTag { expected, got } => {
                write!(f, "expected `{}`, got `{got}`", expected.as_str())
            }
            StampErrorKind::RequiresPattern { head } => {
                write!(f, "expected pattern, got `{head}`")
            }
            StampErrorKind::MissingMetaMap => write!(f, "list missing metadata map at index 1"),
            StampErrorKind::EmptyList => write!(f, "empty list where tagged node expected"),
            StampErrorKind::NodeError(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StampError {}

/// Convert raw parser output to typed AST using role-directed stamping.
///
/// Top-level expressions are treated as Module children (bypass expecting
/// declarations).
pub fn stamp_to_typed(raw_exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError> {
    let mut out = Vec::with_capacity(raw_exprs.len());
    for raw in raw_exprs {
        out.push(stamp_as_bypass_declaration(raw)?);
    }
    Ok(out)
}

/// Stamp a `.dp` file's raw expressions into typed AST.
///
/// This handles two conventions for `.dp` files:
/// 1. A single top-level `(module ...)` wrapper (produced by `chelis deep`)
/// 2. Bare declarations at top level (hand-written `.dp`)
///
/// In case 1, the module is stamped as a Node with its children as
/// declarations. In case 2, each top-level form is stamped as a declaration
/// via `stamp_to_typed`.
pub fn stamp_deep_file(raw_exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError> {
    // If there is exactly one top-level list whose head is `module`, stamp
    // it as a module node (which internally expects declaration children).
    if raw_exprs.len() == 1
        && let Some(tag) = top_level_tag(&raw_exprs[0])
        && tag == DeepTag::Module
    {
        let raw = raw_exprs.into_iter().next().unwrap();
        let span = raw.span();
        let RawExpr::List(elements, _) = raw else {
            unreachable!()
        };
        let stamped = build_node(DeepTag::Module, elements, span)?;
        return Ok(vec![stamped]);
    }
    // Otherwise, treat as bare declarations.
    stamp_to_typed(raw_exprs)
}

/// Peek at the tag of a top-level raw list expression.
fn top_level_tag(raw: &RawExpr) -> Option<DeepTag> {
    if let RawExpr::List(elements, _) = raw
        && let Some(RawExpr::Atom(RawAtom::Symbol(name), _)) = elements.first()
    {
        return DeepTag::parse(name);
    }
    None
}

/// Stamp a raw expression in a specific role context.
fn stamp_in_role(
    raw: RawExpr,
    role: ChildStampRole,
    parent_tag: DeepTag,
    index: usize,
) -> Result<Expr, StampError> {
    match role {
        ChildStampRole::RuntimeExpr => stamp_runtime_expr(raw),
        ChildStampRole::Type => stamp_type(raw),
        ChildStampRole::Syntax | ChildStampRole::Binder | ChildStampRole::Selector => {
            stamp_bare(raw)
        }
        ChildStampRole::EffectHandler => stamp_effect_handler(raw),
        ChildStampRole::ExplicitInferenceBypass => {
            let expectation = bypass_child_expectation(parent_tag, index);
            stamp_bypass(raw, expectation)
        }
    }
}

// ── RuntimeExpr ──────────────────────────────────────────────────────

fn stamp_runtime_expr(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(RawAtom::Symbol(name), span) => Err(StampError {
            kind: StampErrorKind::NameAtExprSlot { name },
            span,
        }),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => stamp_list_as_node_or_unknown(elements, span),
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Type ─────────────────────────────────────────────────────────────

fn stamp_type(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => {
            let (head_str, tag_opt) = decode_list_head(&elements, span)?;
            match tag_opt {
                Some(tag) => build_node(tag, elements, span),
                None => Err(StampError {
                    kind: StampErrorKind::UndecodableTypeHead { head: head_str },
                    span,
                }),
            }
        }
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Syntax/Binder/Selector → BareList ────────────────────────────────

fn stamp_bare(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => {
            let mut out = Vec::with_capacity(elements.len());
            for elem in elements {
                out.push(stamp_bare(elem)?);
            }
            Ok(Expr::BareList(out, span))
        }
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── EffectHandler ────────────────────────────────────────────────────

fn stamp_effect_handler(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => stamp_list_as_node_or_unknown(elements, span),
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Bypass ───────────────────────────────────────────────────────────

fn stamp_bypass(raw: RawExpr, expectation: BypassExpectation) -> Result<Expr, StampError> {
    match expectation {
        BypassExpectation::RequiresDeclaration => stamp_as_bypass_declaration(raw),
        BypassExpectation::RequiresTag(expected_tag) => stamp_as_bypass_tag(raw, expected_tag),
        BypassExpectation::RequiresPattern => stamp_as_bypass_pattern(raw),
        BypassExpectation::FormExpecting => stamp_form_expecting(raw),
        BypassExpectation::Structural => stamp_bare(raw),
    }
}

fn stamp_as_bypass_declaration(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::List(elements, span) => {
            let (head_str, tag_opt) = decode_list_head(&elements, span)?;
            match tag_opt {
                Some(tag) if is_declaration_tag(tag) => build_node(tag, elements, span),
                _ => Err(StampError {
                    kind: StampErrorKind::RequiresDeclaration { head: head_str },
                    span,
                }),
            }
        }
        RawExpr::Atom(_, span) | RawExpr::Map(_, span) | RawExpr::MetaExpr { span, .. } => {
            Err(StampError {
                kind: StampErrorKind::RequiresDeclaration {
                    head: "<non-list>".to_string(),
                },
                span,
            })
        }
    }
}

fn stamp_as_bypass_tag(raw: RawExpr, expected_tag: DeepTag) -> Result<Expr, StampError> {
    match raw {
        RawExpr::List(elements, span) => {
            let (head_str, tag_opt) = decode_list_head(&elements, span)?;
            match tag_opt {
                Some(tag) if tag == expected_tag => build_node(tag, elements, span),
                _ => Err(StampError {
                    kind: StampErrorKind::RequiresTag {
                        expected: expected_tag,
                        got: head_str,
                    },
                    span,
                }),
            }
        }
        RawExpr::Atom(_, span) | RawExpr::Map(_, span) | RawExpr::MetaExpr { span, .. } => {
            Err(StampError {
                kind: StampErrorKind::RequiresTag {
                    expected: expected_tag,
                    got: "<non-list>".to_string(),
                },
                span,
            })
        }
    }
}

fn stamp_as_bypass_pattern(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::List(elements, span) => {
            let (head_str, tag_opt) = decode_list_head(&elements, span)?;
            match tag_opt {
                Some(tag) if is_pattern_tag(tag) => build_node(tag, elements, span),
                _ => Err(StampError {
                    kind: StampErrorKind::RequiresPattern { head: head_str },
                    span,
                }),
            }
        }
        RawExpr::Atom(_, span) | RawExpr::Map(_, span) | RawExpr::MetaExpr { span, .. } => {
            Err(StampError {
                kind: StampErrorKind::RequiresPattern {
                    head: "<non-list>".to_string(),
                },
                span,
            })
        }
    }
}

fn stamp_form_expecting(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(RawAtom::Symbol(name), span) => Err(StampError {
            kind: StampErrorKind::NameAtExprSlot { name },
            span,
        }),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => stamp_list_as_node_or_unknown(elements, span),
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn decode_list_head(
    elements: &[RawExpr],
    span: Span,
) -> Result<(String, Option<DeepTag>), StampError> {
    let Some(first) = elements.first() else {
        return Err(StampError {
            kind: StampErrorKind::EmptyList,
            span,
        });
    };
    match first {
        RawExpr::Atom(RawAtom::Symbol(s), _) => {
            let tag = DeepTag::parse(s);
            Ok((s.clone(), tag))
        }
        _ => Ok(("<non-symbol>".to_string(), None)),
    }
}

/// Build a Node from a raw list whose head decoded as `tag`.
fn build_node(tag: DeepTag, elements: Vec<RawExpr>, span: Span) -> Result<Expr, StampError> {
    if elements.len() < 2 {
        return Err(StampError {
            kind: StampErrorKind::MissingMetaMap,
            span,
        });
    }
    let mut iter = elements.into_iter();
    let _head = iter.next(); // skip the tag symbol
    let meta_raw = iter.next().unwrap();
    let meta = match meta_raw {
        RawExpr::Map(entries, _) => convert_meta_map(entries)?,
        _ => {
            return Err(StampError {
                kind: StampErrorKind::MissingMetaMap,
                span,
            });
        }
    };
    let raw_children: Vec<RawExpr> = iter.collect();
    let arity = raw_children.len();
    let mut children = Vec::with_capacity(arity);
    for (index, child) in raw_children.into_iter().enumerate() {
        let role = child_stamp_role(tag, index, arity);
        children.push(stamp_in_role(child, role, tag, index)?);
    }
    let node = Node::try_new(tag, meta, children).map_err(|e| StampError {
        kind: StampErrorKind::NodeError(e),
        span,
    })?;
    Ok(Expr::Node(Box::new(node), span))
}

fn convert_atom(raw: RawAtom) -> Atom {
    match raw {
        RawAtom::Symbol(s) => Atom::Name(s),
        RawAtom::Int(n) => Atom::Int(n),
        RawAtom::Float(f) => Atom::Float(f),
        RawAtom::Str(s) => Atom::Str(s),
        RawAtom::Bool(b) => Atom::Bool(b),
    }
}

fn convert_meta_map(entries: Vec<(String, RawExpr)>) -> Result<MetaMap, StampError> {
    let mut out = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        out.push((key, stamp_bare(value)?));
    }
    Ok(MetaMap { entries: out })
}

fn stamp_map(entries: Vec<(String, RawExpr)>, span: Span) -> Result<Expr, StampError> {
    Ok(Expr::Map(convert_meta_map(entries)?, span))
}

fn stamp_meta_expr(
    entries: Vec<(String, RawExpr)>,
    expr: RawExpr,
    span: Span,
) -> Result<Expr, StampError> {
    let converted_entries = convert_meta_map(entries)?;
    let converted_expr = stamp_bare(expr)?;
    Ok(Expr::MetaExpr(
        crate::ast::MetaExpr {
            entries: converted_entries.entries,
            expr: Box::new(converted_expr),
        },
        span,
    ))
}

fn stamp_list_as_node_or_unknown(elements: Vec<RawExpr>, span: Span) -> Result<Expr, StampError> {
    let (head_str, tag_opt) = decode_list_head(&elements, span)?;
    match tag_opt {
        Some(tag) => build_node(tag, elements, span),
        None => build_unknown_form(head_str, elements, span),
    }
}

fn build_unknown_form(
    head: String,
    elements: Vec<RawExpr>,
    span: Span,
) -> Result<Expr, StampError> {
    if elements.len() < 2 {
        return Err(StampError {
            kind: StampErrorKind::MissingMetaMap,
            span,
        });
    }
    let mut iter = elements.into_iter();
    let _head_elem = iter.next(); // skip head symbol
    let meta_raw = iter.next().unwrap();
    let meta = match meta_raw {
        RawExpr::Map(entries, _) => convert_meta_map(entries)?,
        _ => {
            return Err(StampError {
                kind: StampErrorKind::MissingMetaMap,
                span,
            });
        }
    };
    let mut children = Vec::new();
    for child in iter {
        children.push(stamp_bare(child)?);
    }
    Ok(Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
        head,
        meta,
        children,
        span,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> Span {
        Span::new(0, 0)
    }

    fn raw_sym(s: &str) -> RawExpr {
        RawExpr::Atom(RawAtom::Symbol(s.to_string()), sp())
    }

    fn raw_int(n: i64) -> RawExpr {
        RawExpr::Atom(RawAtom::Int(n), sp())
    }

    fn raw_map(entries: Vec<(String, RawExpr)>) -> RawExpr {
        RawExpr::Map(entries, sp())
    }

    fn empty_map() -> RawExpr {
        raw_map(vec![])
    }

    /// A well-formed def: `(def {} name body)`
    fn raw_def(name: &str, body: RawExpr) -> RawExpr {
        RawExpr::List(vec![raw_sym("def"), empty_map(), raw_sym(name), body], sp())
    }

    #[test]
    fn stamps_simple_def_with_literal_body() {
        // (def {} f (lit {} 42))
        let lit = RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(42)], sp());
        let input = vec![raw_def("f", lit)];
        let result = stamp_to_typed(input);
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
        assert!(matches!(&exprs[0], Expr::Node(node, _) if node.tag() == DeepTag::Def));
    }

    #[test]
    fn rejects_bare_name_at_runtime_expr_slot() {
        // (def {} f x) — x is a bare name at RuntimeExpr position
        let input = vec![raw_def("f", raw_sym("x"))];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, StampErrorKind::NameAtExprSlot { .. }));
    }

    #[test]
    fn unknown_tag_becomes_unknown_form_at_runtime_expr() {
        // (def {} f (bogus {} 1))
        let bogus = RawExpr::List(vec![raw_sym("bogus"), empty_map(), raw_int(1)], sp());
        let input = vec![raw_def("f", bogus)];
        let result = stamp_to_typed(input);
        assert!(result.is_ok());
        let exprs = result.unwrap();
        // The def should be a Node, its body child should be UnknownForm
        if let Expr::Node(node, _) = &exprs[0] {
            assert_eq!(node.tag(), DeepTag::Def);
        } else {
            panic!("expected Node");
        }
    }

    #[test]
    fn type_slot_rejects_unknown_head() {
        // (defsig {} f (bogus {} x)) — defsig child 1 is Type
        let bogus_type = RawExpr::List(vec![raw_sym("bogus"), empty_map(), raw_sym("x")], sp());
        let input = vec![RawExpr::List(
            vec![raw_sym("defsig"), empty_map(), raw_sym("f"), bogus_type],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            StampErrorKind::UndecodableTypeHead { .. }
        ));
    }

    #[test]
    fn type_slot_accepts_known_type_tag() {
        // (defsig {} f (t-prim {} f32))
        let tprim = RawExpr::List(vec![raw_sym("t-prim"), empty_map(), raw_sym("f32")], sp());
        let input = vec![RawExpr::List(
            vec![raw_sym("defsig"), empty_map(), raw_sym("f"), tprim],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_ok());
    }

    #[test]
    fn top_level_non_declaration_is_error() {
        // (app {} (lit {} 1)) — app is not a declaration
        let input = vec![RawExpr::List(
            vec![
                raw_sym("app"),
                empty_map(),
                RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(1)], sp()),
            ],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            StampErrorKind::RequiresDeclaration { .. }
        ));
    }

    #[test]
    fn binder_slot_permits_name() {
        // (def {} my_name (lit {} 0))
        let lit = RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(0)], sp());
        let input = vec![raw_def("my_name", lit)];
        let result = stamp_to_typed(input);
        assert!(result.is_ok());
    }

    #[test]
    fn missing_meta_map_is_error() {
        // (def x (lit {} 0)) — missing {} after tag
        let input = vec![RawExpr::List(
            vec![
                raw_sym("def"),
                raw_sym("x"),
                RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(0)], sp()),
            ],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind,
            StampErrorKind::MissingMetaMap
        ));
    }
}
