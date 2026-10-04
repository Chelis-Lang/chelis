use serde::{Deserialize, Serialize};

pub use crate::annotations::Metadata;
use crate::span::Span;
use crate::tag::DeepTag;

/// A Deep expression — the core AST node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// An atomic value (symbol, number, string, keyword, bool).
    Atom(Atom, Span),
    /// An inline metadata map `{key: value, ...}` or `{}`.
    Map(Metadata, Span),
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
    pub meta: Metadata,
    pub children: Vec<Expr>,
    pub span: Span,
}

/// A total borrowed view of an admitted Deep expression carrier.
///
/// Every carrier has a distinct variant, so a reader that declines one says
/// so in an explicit match arm rather than through a silent `Option::None`.
/// Each variant has exactly one source: a vocabulary node has the single
/// spelling `Expr::Node` whatever its ingress (chelis#1125).
#[derive(Debug, Clone, Copy)]
pub enum ExprCarrier<'a> {
    /// A decoded vocabulary node (`Expr::Node`).
    DecodedNode(DeepTag, &'a Metadata, &'a [Expr]),
    /// A structural list whose first element has no vocabulary-head role
    /// (`Expr::BareList`).
    StructuralList(&'a [Expr]),
    /// A head that was considered for vocabulary decoding but did not decode
    /// (`Expr::UnknownForm`).
    UndecodableHead(&'a str, &'a Metadata, &'a [Expr]),
    /// An atomic leaf.
    Atom(&'a Atom),
    /// A standalone metadata map.
    MetadataMap(&'a Metadata),
    /// A legacy metadata wrapper around another expression.
    MetadataExpression(&'a MetaExpr),
}

impl Expr {
    /// Construct a canonical tagged node `(tag {meta} children...)` with a
    /// decoded tag. This is the typed producer entry point (decode-once,
    /// chelis#731 Phase 3): programmatic Deep construction goes through
    /// here so the in-memory tree never carries a vocabulary tag as a
    /// string.
    pub fn node(tag: DeepTag, meta: Metadata, children: Vec<Expr>, span: Span) -> Expr {
        Expr::Node(Box::new(crate::node::Node::new(tag, meta, children)), span)
    }

    /// Borrow this expression through the carrier-total reader interface.
    ///
    /// Every admitted representation is a distinct enum variant. A semantic
    /// reader must therefore state what it does with carriers it cannot
    /// consume instead of inheriting an implicit catch-all.
    pub fn carrier(&self) -> ExprCarrier<'_> {
        match self {
            Expr::Node(node, _) => {
                ExprCarrier::DecodedNode(node.tag(), node.meta(), node.children_slice())
            }
            Expr::BareList(elements, _) => ExprCarrier::StructuralList(elements),
            Expr::UnknownForm(data) => {
                ExprCarrier::UndecodableHead(&data.head, &data.meta, &data.children)
            }
            Expr::Atom(atom, _) => ExprCarrier::Atom(atom),
            Expr::Map(metadata, _) => ExprCarrier::MetadataMap(metadata),
            Expr::MetaExpr(metadata_expr, _) => ExprCarrier::MetadataExpression(metadata_expr),
        }
    }

    /// The decoded tag when this expression is a stamped vocabulary node.
    pub fn tag(&self) -> Option<DeepTag> {
        match self.carrier() {
            ExprCarrier::DecodedNode(tag, _, _) => Some(tag),
            _ => None,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Expr::Atom(_, s)
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
    /// Returns `Some(id)` when the expression is a decoded vocabulary node or
    /// an undecodable-head carrier, and its metadata map contains a `span`
    /// entry whose value is a string literal
    /// (`Expr::Atom(Atom::Str(_), _)`).
    ///
    /// The empty string is a valid (though unusual) span ID and is returned
    /// as `Some("")`. Lock this convention in tests; do not silently coerce
    /// `Some("")` to `None`.
    ///
    /// Returns `None` for structural lists, atoms, bare maps, legacy
    /// `MetaExpr` nodes, metadata-bearing carriers whose map has no `span`
    /// key, or `span` values that are not string literals (those are shape
    /// errors callers handle separately, not a missing span).
    pub fn span_id(&self) -> Option<&str> {
        let meta = match self.carrier() {
            ExprCarrier::DecodedNode(_, metadata, _)
            | ExprCarrier::UndecodableHead(_, metadata, _) => metadata,
            _ => return None,
        };
        meta.span_id().map(|v| v.value())
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

/// Which rung of the chelis#759 cast ladder a `cast` node selects.
///
/// The rung lives in the node's optional third child (a bare selector
/// symbol, read exactly like `grad`'s index selector) rather than in the
/// metadata map, because [`strip_metadata`] empties every metadata map on
/// the canonical-display path: a mode parked there would silently turn a
/// truncating cast back into the checked default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CastMode {
    /// `(cast {} expr target-type)`: the [04-NUM-14] checked default. A
    /// fractional or non-finite float into an integer target traps
    /// `Domain`.
    #[default]
    Checked,
    /// `(cast {} expr target-type <selector>)`: one of the named lossy
    /// rungs, each its own atom with its own admitted pairs and traps.
    Named(NamedCastMode),
}

/// The named lossy rungs of the chelis#759 cast ladder. Each one is a
/// separate atom; the lowering carries the rung on one `NamedCast` op so
/// every backend, evaluator, and adjoint site states each rung's
/// disposition by exhaustive matching instead of inheriting the checked
/// default's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NamedCastMode {
    /// `cast_trunc`, [05-OP-6]: float to integer, truncated toward zero;
    /// traps `Overflow` out of range and `Domain` on a non-finite source.
    Trunc,
}

impl NamedCastMode {
    /// Every named rung, in ladder order.
    pub const ALL: &'static [NamedCastMode] = &[NamedCastMode::Trunc];

    /// The Surf keyword and the operation name traps and diagnostics carry.
    pub fn keyword(self) -> &'static str {
        match self {
            NamedCastMode::Trunc => "cast_trunc",
        }
    }

    /// The Deep mode selector symbol.
    pub fn deep_selector(self) -> &'static str {
        match self {
            NamedCastMode::Trunc => "trunc",
        }
    }

    /// The governing [05-OP-N] atom.
    pub fn atom(self) -> &'static str {
        match self {
            NamedCastMode::Trunc => "[05-OP-6]",
        }
    }
}

impl CastMode {
    /// The Surf keyword that selects this mode.
    pub fn keyword(self) -> &'static str {
        match self {
            CastMode::Checked => "cast",
            CastMode::Named(named) => named.keyword(),
        }
    }

    /// The Deep mode selector symbol, or `None` for the default rung
    /// (spelled as the plain two-child `cast` form).
    pub fn deep_selector(self) -> Option<&'static str> {
        match self {
            CastMode::Checked => None,
            CastMode::Named(named) => Some(named.deep_selector()),
        }
    }

    /// Read a Deep mode selector symbol back. An unrecognized symbol is
    /// `None` so the caller can reject it loudly rather than defaulting
    /// to the checked rung.
    pub fn from_deep_selector(symbol: &str) -> Option<Self> {
        NamedCastMode::ALL
            .iter()
            .find(|named| named.deep_selector() == symbol)
            .map(|named| CastMode::Named(*named))
    }

    /// The named rung, or `None` for the checked default.
    pub fn named(self) -> Option<NamedCastMode> {
        match self {
            CastMode::Checked => None,
            CastMode::Named(named) => Some(named),
        }
    }
}

/// The rung selected by a `cast` node's children.
///
/// `Ok(mode)` for the two-child checked form and for a recognized
/// selector; `Err(spelling)` for an unrecognized or non-symbol third
/// child, which every caller rejects rather than silently treating as
/// checked.
pub fn cast_mode_of(children: &[Expr]) -> Result<CastMode, String> {
    let Some(selector) = children.get(2) else {
        return Ok(CastMode::Checked);
    };
    let symbol = match selector {
        Expr::Atom(Atom::Name(symbol), _) => symbol.as_str(),
        other => return Err(crate::printer::print_expr_flat(other)),
    };
    CastMode::from_deep_selector(symbol).ok_or_else(|| symbol.to_string())
}

/// Legacy metadata: `^{k1 v1 ...} expr` (prefix form).
#[derive(Debug, Clone, PartialEq)]
pub struct MetaExpr {
    /// Dedicated annotations with separately owned producer extensions.
    pub metadata: Metadata,
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
        Expr::Map(_, span) => Expr::Map(Metadata::default(), *span),
        Expr::MetaExpr(meta, _) => strip_metadata(&meta.expr),
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
                Box::new(Node::new(node.tag(), Metadata::default(), children)),
                *span,
            )
        }
        Expr::BareList(elems, span) => {
            let stripped = elems.iter().map(strip_metadata).collect();
            Expr::BareList(stripped, *span)
        }
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(UnknownFormData {
            head: data.head.clone(),
            meta: Metadata::default(),
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
