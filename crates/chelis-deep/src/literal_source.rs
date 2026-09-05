//! Structural classification of the language's scalar literal-source form.
//!
//! Surf represents a negative numeral as unary minus applied to a literal
//! (`spec/02-surf-syntax.md` P10 and section 6.3). After desugaring, that is
//! the exact Deep shape `(app {} (var {} neg) (lit ...))`. Consumers deciding
//! whether an expression is the literal source named by P10b must use this
//! classifier rather than independently recognizing only one polarity.

use crate::{Atom, DeepTag, Expr, MetaMap};

/// The two exact expression shapes that can carry a scalar literal source
/// from Surf into Deep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSourceShape {
    /// A direct `(lit ...)` node.
    Direct,
    /// The canonical unary-minus spelling `(app (var neg) (lit ...))`.
    UnaryMinus,
}

/// A structurally validated scalar literal source.
#[derive(Debug, Clone, Copy)]
pub struct LiteralSource<'a> {
    literal: &'a Expr,
    metadata: &'a MetaMap,
    numeric_atom: Option<&'a Atom>,
    shape: LiteralSourceShape,
}

impl<'a> LiteralSource<'a> {
    /// The exact `lit` node carrying the source atom and its type metadata.
    pub fn literal(self) -> &'a Expr {
        self.literal
    }

    /// Metadata carried by the exact `lit` node.
    pub fn metadata(self) -> &'a MetaMap {
        self.metadata
    }

    /// The numeric atom stored by the `lit` node, when this is a numeric
    /// literal. For [`LiteralSourceShape::UnaryMinus`], the wrapper owns the
    /// negation. Boolean, string, and unit literals return `None`.
    pub fn numeric_atom(self) -> Option<&'a Atom> {
        self.numeric_atom
    }

    /// Whether the source was direct or carried by canonical unary minus.
    pub fn shape(self) -> LiteralSourceShape {
        self.shape
    }

    /// Fold the source form to the signed atom that an adopting context
    /// materializes. Integer negation that is not representable fails closed.
    pub fn folded_numeric_atom(self) -> Option<Atom> {
        match (self.shape, self.numeric_atom?) {
            (LiteralSourceShape::Direct, atom) => Some(atom.clone()),
            (LiteralSourceShape::UnaryMinus, Atom::Int(value)) => {
                value.checked_neg().map(Atom::Int)
            }
            (LiteralSourceShape::UnaryMinus, Atom::Float(value)) => Some(Atom::Float(-value)),
            (LiteralSourceShape::UnaryMinus, _) => None,
        }
    }
}

/// Classify the complete scalar literal-source form admitted by Surf.
///
/// This accepts any exact direct `lit`, plus one unary-minus application whose
/// operand is a numeric `lit`. Computed operands, nested negation, negated
/// nonnumeric literals, and malformed arities fail closed.
pub fn classify_literal_source(expr: &Expr) -> Option<LiteralSource<'_>> {
    if let Some((literal, metadata, numeric_atom)) = direct_literal(expr) {
        return Some(LiteralSource {
            literal,
            metadata,
            numeric_atom,
            shape: LiteralSourceShape::Direct,
        });
    }

    let (DeepTag::App, _, children) = node_parts(expr)? else {
        return None;
    };
    let [callee, operand] = children else {
        return None;
    };
    if exact_var_name(callee) != Some("neg") {
        return None;
    }
    let (literal, metadata, numeric_atom) = direct_literal(operand)?;
    numeric_atom?;
    Some(LiteralSource {
        literal,
        metadata,
        numeric_atom,
        shape: LiteralSourceShape::UnaryMinus,
    })
}

fn direct_literal(expr: &Expr) -> Option<(&Expr, &MetaMap, Option<&Atom>)> {
    let (DeepTag::Lit, metadata, children) = node_parts(expr)? else {
        return None;
    };
    let [value] = children else {
        return None;
    };
    let numeric_atom = match value {
        Expr::Atom(atom @ (Atom::Int(_) | Atom::Float(_)), _) => Some(atom),
        _ => None,
    };
    Some((expr, metadata, numeric_atom))
}

fn exact_var_name(expr: &Expr) -> Option<&str> {
    let (DeepTag::Var, _, children) = node_parts(expr)? else {
        return None;
    };
    let [Expr::Atom(Atom::Name(name), _)] = children else {
        return None;
    };
    Some(name)
}

fn node_parts(expr: &Expr) -> Option<(DeepTag, &MetaMap, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        Expr::List(list, _) => {
            let tag = list.tag()?;
            let Expr::Map(meta, _) = list.elements.get(1)? else {
                return None;
            };
            Some((tag, meta, list.elements.get(2..).unwrap_or_default()))
        }
        _ => None,
    }
}
