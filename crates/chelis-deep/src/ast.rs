use serde::{Deserialize, Serialize};

use crate::span::Span;
use crate::tag::DeepTag;

/// A Deep expression — the core AST node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// An atomic value (symbol, number, string, keyword, bool).
    Atom(Atom, Span),
    /// A parenthesized list `(tag {} children...)`.
    ///
    /// **Deprecated (chelis#908):** No code path produces this variant
    /// anymore. All producers route through `stamp_to_typed` which emits
    /// `Expr::Node`. Match arms for this variant are dead code awaiting
    /// removal. Do not construct new `Expr::List` values.
    List(List, Span),
    /// An inline metadata map `{key: value, ...}` or `{}`.
    Map(MetaMap, Span),
    /// A metadata-annotated expression `^{k1 v1 ...} expr` (legacy, kept for compat).
    MetaExpr(MetaExpr, Span),
    /// A stamped vocabulary node produced by `stamp_to_typed`. The `Node`
    /// is role-gated: construction validates arity and rejects Name atoms
    /// at RuntimeExpr positions.
    Node(Box<crate::node::Node>, Span),
    /// A structural bare list (no vocabulary head decode). Produced at
    /// Syntax/Binder/Selector positions by `stamp_to_typed`.
    BareList(Vec<Expr>, Span),
    /// A list whose head symbol did not decode into the closed vocabulary
    /// at a position where decode was attempted. Preserves the head
    /// string, metadata, and recursively stamped children for downstream
    /// diagnostics.
    UnknownForm(Box<UnknownFormData>),
}

/// Data for an `Expr::UnknownForm` — boxed to keep the `Expr` enum small.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnknownFormData {
    pub head: String,
    pub meta: MetaMap,
    pub children: Vec<Expr>,
    pub span: Span,
}

impl Expr {
    /// Construct a canonical tagged node with a decoded tag. This is the
    /// typed producer entry point: programmatic Deep construction goes
    /// through here, producing an `Expr::Node` directly.
    pub fn node(tag: DeepTag, meta: MetaMap, children: Vec<Expr>, span: Span) -> Expr {
        Expr::Node(
            Box::new(crate::node::Node::new(tag, meta, children)),
            span,
        )
    }

    /// The decoded tag when this expression is a stamped vocabulary node.
    pub fn tag(&self) -> Option<DeepTag> {
        match self {
            Expr::List(list, _) => list.tag(),
            Expr::Node(node, _) => Some(node.tag()),
            _ => None,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Expr::Atom(_, s)
            | Expr::List(_, s)
            | Expr::Map(_, s)
            | Expr::MetaExpr(_, s)
            | Expr::Node(_, s)
            | Expr::BareList(_, s) => *s,
            Expr::UnknownForm(data) => data.span,
        }
    }

    /// Return the external-source span identifier carried in this node's
    /// metadata map under the `span` key, if any.
    ///
    /// Per `spec/03-deep-syntax.md` §1.1.1 and
    /// `spec/design/chelis_span_survival.md`, every Deep AST node is a
    /// 3-tuple `(tag {meta} children...)` with the metadata map at element
    /// index 1. The `span` key holds an opaque string identifier issued by
    /// an external producer (e.g., Octant's LaTeX-to-Deep translator).
    ///
    /// Returns `Some(id)` when the metadata map contains a `span` entry
    /// whose value is a string literal (`Expr::Atom(Atom::Str(_), _)`).
    ///
    /// The empty string is a valid (though unusual) span ID and is returned
    /// as `Some("")`. Lock this convention in tests; do not silently coerce
    /// `Some("")` to `None`.
    pub fn span_id(&self) -> Option<&str> {
        let meta = match self {
            Expr::Node(node, _) => node.meta(),
            Expr::List(list, _) => match list.elements.get(1)? {
                Expr::Map(m, _) => m,
                _ => return None,
            },
            _ => return None,
        };
        for (key, value) in &meta.entries {
            if key == "span"
                && let Expr::Atom(Atom::Str(s), _) = value
            {
                return Some(s.as_str());
            }
        }
        None
    }
}

/// An atomic (leaf) value in the AST.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Atom {
    Name(String),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// A parenthesized list of expressions.
///
/// **Deprecated (chelis#908):** This struct exists only for compilation
/// compatibility during the transition. No code path produces `Expr::List`
/// anymore. Do not construct new `List` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct List {
    /// All elements of the list.
    pub elements: Vec<Expr>,
}

impl List {
    /// The node's decoded closed-vocabulary tag, when element 0 carries
    /// one.
    pub fn tag(&self) -> Option<DeepTag> {
        match self.elements.first() {
            Some(Expr::Atom(Atom::Name(symbol), _)) => DeepTag::parse(symbol),
            _ => None,
        }
    }

    /// The element-0 symbol of a list that is NOT a stamped vocabulary
    /// node, for naming unknown tags in diagnostics.
    pub fn unknown_tag_symbol(&self) -> Option<&str> {
        match self.elements.first() {
            Some(Expr::Atom(Atom::Name(symbol), _)) => {
                if DeepTag::parse(symbol).is_none() {
                    Some(symbol.as_str())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Inline metadata map: `{key: value, ...}` or `{}`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MetaMap {
    /// Key-value pairs. Keys are bare identifiers.
    pub entries: Vec<(String, Expr)>,
}

/// Legacy metadata: `^{k1 v1 ...} expr` (prefix form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaExpr {
    /// Key-value pairs. Keys are keyword strings (without `:`).
    pub entries: Vec<(String, Expr)>,
    /// The expression this metadata is attached to.
    pub expr: Box<Expr>,
}

/// Return a copy of `expr` with every node's metadata map emptied and every
/// legacy `MetaExpr` wrapper dropped, so the canonical printer renders the bare
/// term `(tag {} children...)` with no producer/lowering metadata.
///
/// This is the human-display form of a synthesized or lowered term: the
/// proposition shape without the `span`/`type`/producer keys that a lowering
/// pass attaches. The chelis#436 prove-JSON obligation `goal` field uses it so
/// a consumer sees the discharged invariant predicate, not the lowering's
/// internal span annotations. The 3-tuple shape is preserved (the metadata map
/// stays present at index 1, just empty), so the result re-parses.
pub fn strip_metadata(expr: &Expr) -> Expr {
    match expr {
        Expr::Atom(..) => expr.clone(),
        Expr::Map(_, span) => Expr::Map(MetaMap::default(), *span),
        Expr::MetaExpr(meta, _) => strip_metadata(&meta.expr),
        Expr::List(list, span) => {
            let elements = list
                .elements
                .iter()
                .enumerate()
                .map(|(index, element)| {
                    if index == 1 && matches!(element, Expr::Map(..)) {
                        Expr::Map(MetaMap::default(), element.span())
                    } else {
                        strip_metadata(element)
                    }
                })
                .collect();
            Expr::List(List { elements }, *span)
        }
        Expr::Node(node, span) => {
            use crate::node::Node;
            let children: Vec<Expr> = node
                .children_iter()
                .map(|child_ref| match child_ref {
                    crate::node::ChildRef::Expr(e)
                    | crate::node::ChildRef::Syntax(e)
                    | crate::node::ChildRef::Type(e)
                    | crate::node::ChildRef::EffectHandler(e)
                    | crate::node::ChildRef::Bypass(e) => strip_metadata(e),
                    crate::node::ChildRef::Binder(s) => {
                        Expr::Atom(Atom::Name(s.to_string()), *span)
                    }
                    crate::node::ChildRef::Selector(s) => {
                        Expr::Atom(Atom::Name(s.to_string()), *span)
                    }
                })
                .collect();
            Expr::Node(
                Box::new(Node::new(node.tag(), MetaMap::default(), children)),
                *span,
            )
        }
        Expr::BareList(elems, span) => {
            let stripped = elems.iter().map(strip_metadata).collect();
            Expr::BareList(stripped, *span)
        }
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(UnknownFormData {
            head: data.head.clone(),
            meta: MetaMap::default(),
            children: data.children.iter().map(strip_metadata).collect(),
            span: data.span,
        })),
    }
}

#[cfg(test)]
mod strip_metadata_tests {
    use super::*;
    use crate::parser::parse_str;
    use crate::printer::print_expr_flat;

    fn first(source: &str) -> Expr {
        parse_str(source)
            .expect("parse")
            .into_iter()
            .next()
            .expect("one expr")
    }

    #[test]
    fn strip_metadata_empties_every_meta_map_and_keeps_the_term() {
        // A node carrying span/type metadata renders bare after stripping: the
        // 3-tuple shape is preserved (metadata map present but empty), so the
        // result re-parses and the proposition is unchanged.
        let with_meta = first(
            r#"(app {span: "surf:0..3"} (var {} gte) (var {span: "surf:1..2"} x) (lit {type: (t-prim {} f32)} 0.0))"#,
        );
        let stripped = strip_metadata(&with_meta);
        assert_eq!(
            print_expr_flat(&stripped),
            "(app {} (var {} gte) (var {} x) (lit {} 0.0))",
            "every metadata map is emptied; the term and its children survive"
        );
        // Negative parity: the original still carries its metadata (strip does
        // not mutate in place), so a caller that wants the annotated form keeps
        // it.
        assert!(
            print_expr_flat(&with_meta).contains("surf:0..3"),
            "strip_metadata clones; the source expr is unchanged"
        );
    }

    #[test]
    fn strip_metadata_recurses_into_nested_children() {
        let nested = first(
            r#"(app {span: "a"} (var {} and) (app {span: "b"} (var {} gte) (var {span: "c"} x) (lit {span: "d"} 0.0)) (var {span: "e"} y))"#,
        );
        let printed = print_expr_flat(&strip_metadata(&nested));
        assert!(
            !printed.contains("span"),
            "no span metadata survives anywhere in the tree: {printed}"
        );
        assert_eq!(
            printed,
            "(app {} (var {} and) (app {} (var {} gte) (var {} x) (lit {} 0.0)) (var {} y))"
        );
    }

    #[test]
    fn strip_metadata_leaves_bare_atoms_untouched() {
        let atom = Expr::Atom(Atom::Int(7), crate::span::Span { offset: 3, len: 1 });
        assert_eq!(strip_metadata(&atom), atom);
    }
}
