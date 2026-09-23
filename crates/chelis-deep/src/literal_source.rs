//! Structural classification of scalar literal sources and binder adoption.

use crate::{Atom, DeepTag, DtypeFamily, Expr, Metadata};

/// Whether a numeric atom is valid at every applicable member of a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralFamilyFit {
    /// The atom kind and value are valid across the family.
    Fits,
    /// The atom kind does not adopt this family.
    IncompatibleKind,
    /// An integer atom is outside at least one active signed member's range.
    IntegerOutOfRange,
}

/// A structurally validated scalar literal source.
#[derive(Debug, Clone, Copy)]
pub struct LiteralSource<'a> {
    literal: &'a Expr,
    metadata: &'a Metadata,
    numeric_atom: Option<&'a Atom>,
}

impl<'a> LiteralSource<'a> {
    /// Metadata carried by the exact `lit` node.
    pub fn metadata(self) -> &'a Metadata {
        self.metadata
    }

    /// The numeric atom, including its sign.
    pub fn numeric_atom(self) -> Option<&'a Atom> {
        self.numeric_atom
    }

    /// Whether this source atom can bind at a dtype-family-bounded target.
    pub fn admitted_by(self, family: DtypeFamily) -> bool {
        self.has_exact_unsuffixed_style() && self.family_fit(family) == LiteralFamilyFit::Fits
    }

    /// Whether exactly one producer marker identifies an unsuffixed literal.
    pub fn has_exact_unsuffixed_style(self) -> bool {
        self.metadata
            .surf_literal_style()
            .is_some_and(|v| *v.value() == crate::annotations::LiteralStyle::Unsuffixed)
    }

    /// Classify kind and family-wide value validity independently of provenance.
    pub fn family_fit(self, family: DtypeFamily) -> LiteralFamilyFit {
        match self.numeric_atom {
            Some(Atom::Float(_)) if matches!(family, DtypeFamily::Float | DtypeFamily::Numeric) => {
                LiteralFamilyFit::Fits
            }
            Some(Atom::Int(_)) if family == DtypeFamily::Float => LiteralFamilyFit::Fits,
            Some(Atom::Int(value))
                if active_signed_integer_ranges()
                    .into_iter()
                    .all(|(min, max)| *value >= min && *value <= max) =>
            {
                LiteralFamilyFit::Fits
            }
            Some(Atom::Int(_)) => LiteralFamilyFit::IntegerOutOfRange,
            _ => LiteralFamilyFit::IncompatibleKind,
        }
    }
}

fn active_signed_integer_ranges() -> [(i64, i64); 4] {
    [
        (i8::MIN as i64, i8::MAX as i64),
        (i16::MIN as i64, i16::MAX as i64),
        (i32::MIN as i64, i32::MAX as i64),
        (i64::MIN, i64::MAX),
    ]
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

/// Classify one direct `lit`; callable applications carry no syntax provenance.
pub fn classify_literal_source(expr: &Expr) -> Option<LiteralSource<'_>> {
    let (literal, metadata, numeric_atom) = direct_literal(expr)?;
    Some(LiteralSource {
        literal,
        metadata,
        numeric_atom,
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
                    && let Some(binder) = meta
                        .ty()
                        .and_then(|ty| exact_type_variable_name(ty.expression()))
                {
                    visitor(BinderLiteralUse::Literal {
                        binder,
                        source: classify_literal_source(expr),
                        adopting_binder,
                    });
                }
                stack.extend(children.iter().rev().map(|child| (child, None)));
            }
            meta.visit_expressions(&mut |value, _| stack.push((value, None)));
            continue;
        }
        match expr {
            Expr::MetaExpr(meta, _) => {
                meta.metadata
                    .visit_expressions(&mut |value, _| stack.push((value, None)));
                stack.push((&meta.expr, adopting_binder));
            }
            Expr::Map(meta, _) => {
                meta.visit_expressions(&mut |value, _| stack.push((value, None)));
            }
            Expr::BareList(children, _) => {
                stack.extend(children.iter().rev().map(|child| (child, None)));
            }
            Expr::UnknownForm(data) => {
                stack.extend(data.children.iter().rev().map(|child| (child, None)));
                data.meta
                    .visit_expressions(&mut |value, _| stack.push((value, None)));
            }
            Expr::Node(..) | Expr::Atom(..) => {}
        }
    }
}

fn direct_literal(expr: &Expr) -> Option<(&Expr, &Metadata, Option<&Atom>)> {
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

fn node_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        _ => None,
    }
}
