//! Structural classification of scalar literal sources and binder adoption.

use crate::{Atom, DeepTag, DtypeFamily, Expr, MetaMap};

/// The two exact Deep shapes produced for a Surf scalar literal.
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
    /// Metadata carried by the exact `lit` node.
    pub fn metadata(self) -> &'a MetaMap {
        self.metadata
    }

    /// The numeric atom; unary minus remains represented by [`Self::shape`].
    pub fn numeric_atom(self) -> Option<&'a Atom> {
        self.numeric_atom
    }

    /// Whether the source was direct or carried by canonical unary minus.
    pub fn shape(self) -> LiteralSourceShape {
        self.shape
    }

    /// Materialize the signed atom; unrepresentable integer negation fails.
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

    /// Whether this source atom can bind at a dtype-family-bounded target.
    pub fn admitted_by(self, family: DtypeFamily) -> bool {
        let mut styles = self
            .metadata
            .entries
            .iter()
            .filter_map(|(key, value)| (key == "surf_literal_style").then_some(value));
        let exact_unsuffixed = matches!(
            styles.next(),
            Some(Expr::Atom(Atom::Str(style), _)) if style == "unsuffixed"
        ) && styles.next().is_none();
        exact_unsuffixed && self.numeric_atom_admitted_by(family)
    }

    /// Whether the atom kind is compatible with a dtype family.
    pub fn numeric_atom_admitted_by(self, family: DtypeFamily) -> bool {
        match self.numeric_atom {
            Some(Atom::Float(_)) => matches!(family, DtypeFamily::Float | DtypeFamily::Numeric),
            Some(Atom::Int(_)) => true,
            _ => false,
        }
    }
}

/// A binder-sensitive literal relation found in a Deep expression.
#[derive(Debug, Clone, Copy)]
pub enum BinderLiteralUse<'a> {
    /// A cast whose target is exactly `(t-var {} <binder>)`.
    CastTarget {
        binder: &'a str,
        source: Option<LiteralSource<'a>>,
    },
    /// A literal stamped with exactly `(t-var {} <binder>)`.
    Literal {
        binder: &'a str,
        source: Option<LiteralSource<'a>>,
        adopting_binder: Option<&'a str>,
    },
}

/// Classify a direct `lit` or one exact `app(var neg, numeric-lit)` wrapper.
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

/// Visit binder-target casts and binder-stamped literals without recursion.
pub fn visit_binder_literal_uses<'a>(
    expr: &'a Expr,
    visitor: &mut impl FnMut(BinderLiteralUse<'a>),
) {
    let mut stack = vec![(expr, None)];
    while let Some((expr, adopting_binder)) = stack.pop() {
        if let Some((tag, meta, children)) = node_parts(expr) {
            if tag == DeepTag::Cast
                && let [operand, target, rest @ ..] = children
                && let Some(binder) = exact_type_variable_name(target)
            {
                let source = classify_literal_source(operand);
                visitor(BinderLiteralUse::CastTarget { binder, source });
                stack.extend(rest.iter().rev().map(|child| (child, None)));
                stack.push((target, None));
                stack.push((
                    source.map_or(operand, |source| source.literal),
                    source.map(|_| binder),
                ));
            } else {
                if tag == DeepTag::Lit
                    && let Some(binder) = meta.entries.iter().find_map(|(key, ty)| {
                        (key == "type")
                            .then(|| exact_type_variable_name(ty))
                            .flatten()
                    })
                {
                    visitor(BinderLiteralUse::Literal {
                        binder,
                        source: classify_literal_source(expr),
                        adopting_binder,
                    });
                }
                stack.extend(children.iter().rev().map(|child| (child, None)));
            }
            stack.extend(meta.entries.iter().rev().map(|(_, value)| (value, None)));
            continue;
        }
        match expr {
            Expr::MetaExpr(meta, _) => {
                stack.extend(meta.entries.iter().rev().map(|(_, value)| (value, None)));
                stack.push((&meta.expr, adopting_binder));
            }
            Expr::Map(meta, _) => {
                stack.extend(meta.entries.iter().rev().map(|(_, value)| (value, None)));
            }
            Expr::List(list, _) => {
                stack.extend(list.elements.iter().rev().map(|child| (child, None)));
            }
            Expr::BareList(children, _) => {
                stack.extend(children.iter().rev().map(|child| (child, None)));
            }
            Expr::UnknownForm(data) => {
                stack.extend(data.children.iter().rev().map(|child| (child, None)));
                stack.extend(
                    data.meta
                        .entries
                        .iter()
                        .rev()
                        .map(|(_, value)| (value, None)),
                );
            }
            Expr::Node(..) | Expr::Atom(..) => {}
        }
    }
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

/// Name carried by an exact, non-hole `t-var` node.
pub fn exact_type_variable_name(expr: &Expr) -> Option<&str> {
    let (DeepTag::TVar, _, children) = node_parts(expr)? else {
        return None;
    };
    let [Expr::Atom(Atom::Name(name), _)] = children else {
        return None;
    };
    (name != "_").then_some(name)
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
