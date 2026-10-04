//! Structural classification of scalar literal sources and binder adoption.

use crate::{Atom, DeepTag, DtypeFamily, Expr, Metadata};

/// Whether a numeric atom's kind adopts a family and, for an integer atom,
/// whether it fits every signed member's range. Whether a value is finite at
/// each float member is not decided here: the checker's `[04-LIT-2]` rule
/// finalizes the atom at each member with the dtype-semantics authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralFamilyFit {
    /// The atom kind adopts the family, and an integer atom is in range for
    /// every signed member.
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

    /// Whether an unsuffixed literal is admitted by either §5.9 bound form.
    pub fn admitted_by_bound(self, bound: &crate::DtypeBound) -> bool {
        self.has_exact_unsuffixed_style() && self.bound_fit(bound) == LiteralFamilyFit::Fits
    }

    /// [04-INF-6] at every member of either §5.9 bound form.
    ///
    /// The family arm keeps [`LiteralSource::family_fit`]'s behaviour. The set
    /// arm is why chelis#2443 exists: an integer literal is range-checked
    /// against the integer members the set actually lists, so `{i32, i64}`
    /// admits a value i8 could not hold, while `Int` still does not.
    pub fn bound_fit(self, bound: &crate::DtypeBound) -> LiteralFamilyFit {
        let crate::DtypeBound::Set(members) = bound else {
            let crate::DtypeBound::Family(family) = bound else {
                unreachable!("DtypeBound has exactly two forms")
            };
            return self.family_fit(*family);
        };
        // §5.9 makes an empty set a declaration error, rejected before a
        // bound is installed, so there is always at least one member.
        let every_member_is_float = members.iter().all(|member| member.is_float());
        match self.numeric_atom {
            // A float literal cannot bind at an integer member.
            Some(Atom::Float(_)) if every_member_is_float => LiteralFamilyFit::Fits,
            Some(Atom::Float(_)) => LiteralFamilyFit::IncompatibleKind,
            // An integer literal adopts a float member exactly as it does
            // under `Float`.
            Some(Atom::Int(_)) if every_member_is_float => LiteralFamilyFit::Fits,
            Some(Atom::Int(value)) => {
                if members
                    .iter()
                    .filter(|member| member.is_integer())
                    .all(|member| {
                        let (min, max) = signed_integer_range(*member);
                        *value >= min && *value <= max
                    })
                {
                    LiteralFamilyFit::Fits
                } else {
                    LiteralFamilyFit::IntegerOutOfRange
                }
            }
            _ => LiteralFamilyFit::IncompatibleKind,
        }
    }
}

/// The closed range of one active signed integer dtype.
fn signed_integer_range(member: crate::BoundDtype) -> (i64, i64) {
    match member {
        crate::BoundDtype::I8 => (i8::MIN as i64, i8::MAX as i64),
        crate::BoundDtype::I16 => (i16::MIN as i64, i16::MAX as i64),
        crate::BoundDtype::I32 => (i32::MIN as i64, i32::MAX as i64),
        crate::BoundDtype::I64 => (i64::MIN, i64::MAX),
        float => unreachable!("{} is not a signed integer", float.name()),
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
    /// A tensor literal cast to `binder` whose `element` is stamped with
    /// exactly `(t-var {} <binder>)`, reported once per literal.
    TensorLiteral {
        binder: &'a str,
        element: LiteralSource<'a>,
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
            // `x |> cast(p)` is `cast(x, p)`: a pipe's head is the operand of
            // a cast stage, and adopts its binder exactly as the call does.
            // In `[...] |> to_tensor |> cast(p)` the head and the constructor
            // stage form the tensor literal the cast stage receives.
            if tag == DeepTag::Pipe
                && let Some((operand, stage_index, binder)) = pipe_cast_operand(children)
            {
                stack.extend(
                    children[stage_index..]
                        .iter()
                        .rev()
                        .map(|child| (child, None)),
                );
                if stage_index == 2 {
                    stack.push((&children[1], None));
                    push_tensor_literal_chain(operand, binder, &mut stack, visitor);
                } else {
                    push_cast_operand(operand, binder, &mut stack, visitor);
                }
                meta.visit_expressions(&mut |value, _| stack.push((value, None)));
                continue;
            }
            if tag == DeepTag::Cast
                && let [operand, target, rest @ ..] = children
                && let Some(binder) = exact_type_variable_name(target)
            {
                visitor(BinderLiteralUse::CastTarget {
                    binder,
                    source: classify_literal_source(operand),
                });
                stack.extend(rest.iter().rev().map(|child| (child, None)));
                stack.push((target, None));
                push_cast_operand(operand, binder, &mut stack, visitor);
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

/// The binder a pipe's cast stage targets: the call-first lambda
/// `fn (x) -> cast(x, p)` the parser builds for `|> cast(p)`, with `p` a
/// `t-var`.
fn pipe_cast_stage_binder(stage: &Expr) -> Option<&str> {
    let (DeepTag::Fn, meta, [params, body]) = node_parts(stage)? else {
        return None;
    };
    meta.surf_pipe_stage()?;
    let (DeepTag::Params, _, [param]) = node_parts(params)? else {
        return None;
    };
    let param = match param {
        Expr::Atom(Atom::Name(name), _) => name.as_str(),
        other => match node_parts(other)? {
            (_, _, [Expr::Atom(Atom::Name(name), _), ..]) => name.as_str(),
            _ => return None,
        },
    };
    let (DeepTag::Cast, _, [operand, target, ..]) = node_parts(body)? else {
        return None;
    };
    let (DeepTag::Var, _, [Expr::Atom(Atom::Name(name), _)]) = node_parts(operand)? else {
        return None;
    };
    (name == param)
        .then(|| exact_type_variable_name(target))
        .flatten()
}

/// Push the operand of a cast to `binder`. A direct literal is the cast's
/// adopting literal; so is every literal element, at any nesting depth, of a
/// `to_tensor` call of a finite `Cons`/`Nil` chain (spec/04-type-system.md
/// §5.6 position 4). Anything else is traversed without an adopting binder.
fn push_cast_operand<'a>(
    operand: &'a Expr,
    binder: &'a str,
    stack: &mut Vec<(&'a Expr, Option<&'a str>)>,
    visitor: &mut impl FnMut(BinderLiteralUse<'a>),
) {
    if let Some(source) = classify_literal_source(operand) {
        stack.push((source.literal, Some(binder)));
        return;
    }
    if let Some((DeepTag::App, meta, [function, chain])) = node_parts(operand)
        && is_name(function, "to_tensor")
        && finite_chain(chain)
    {
        meta.visit_expressions(&mut |value, _| stack.push((value, None)));
        stack.push((function, None));
        push_tensor_literal_chain(chain, binder, stack, visitor);
        return;
    }
    stack.push((operand, None));
}

/// Push a tensor literal's element chain whose literals adopt `binder`, and
/// report the literal once when an element is typed at the binder.
fn push_tensor_literal_chain<'a>(
    chain: &'a Expr,
    binder: &'a str,
    stack: &mut Vec<(&'a Expr, Option<&'a str>)>,
    visitor: &mut impl FnMut(BinderLiteralUse<'a>),
) {
    let mut elements = Vec::new();
    chain_literal_elements(chain, &mut elements);
    if let Some(element) = elements.iter().copied().find_map(|element| {
        let source = classify_literal_source(element)?;
        source
            .metadata
            .ty()
            .and_then(|ty| exact_type_variable_name(ty.expression()))
            .is_some_and(|name| name == binder)
            .then_some(source)
    }) {
        visitor(BinderLiteralUse::TensorLiteral { binder, element });
    }
    push_chain(chain, binder, stack);
}

/// The cast operand inside a pipe: the head before a cast stage to a binder,
/// or the head chain of `[...] |> to_tensor |> cast(p)`. Returns the operand,
/// the index of the first stage to traverse normally, and the binder.
fn pipe_cast_operand(children: &[Expr]) -> Option<(&Expr, usize, &str)> {
    match children {
        [head, constructor, stage, ..]
            if is_name(constructor, "to_tensor") && finite_chain(head) =>
        {
            pipe_cast_stage_binder(stage).map(|binder| (head, 2, binder))
        }
        [head, stage, ..] => pipe_cast_stage_binder(stage).map(|binder| (head, 1, binder)),
        _ => None,
    }
}

fn is_name(expr: &Expr, expected: &str) -> bool {
    matches!(
        node_parts(expr),
        Some((DeepTag::Var, _, [Expr::Atom(Atom::Name(name), _)])) if name == expected
    )
}

/// Whether `expr` is a finite `Cons`/`Nil` chain.
fn finite_chain(expr: &Expr) -> bool {
    let mut tail = expr;
    loop {
        if is_name(tail, "Nil") {
            return true;
        }
        match node_parts(tail) {
            Some((DeepTag::App, _, [cons, _, rest])) if is_name(cons, "Cons") => tail = rest,
            _ => return false,
        }
    }
}

/// The literal elements of a finite chain, nested chains included.
fn chain_literal_elements<'a>(chain: &'a Expr, out: &mut Vec<&'a Expr>) {
    let mut tail = chain;
    while let Some((DeepTag::App, _, [_, element, rest])) = node_parts(tail) {
        if direct_literal(element).is_some() {
            out.push(element);
        } else if finite_chain(element) {
            chain_literal_elements(element, out);
        }
        tail = rest;
    }
}

fn push_chain<'a>(chain: &'a Expr, binder: &'a str, stack: &mut Vec<(&'a Expr, Option<&'a str>)>) {
    let Some((tag, meta, children)) = node_parts(chain) else {
        return;
    };
    meta.visit_expressions(&mut |value, _| stack.push((value, None)));
    match (tag, children) {
        (DeepTag::App, [cons, element, rest]) => {
            stack.push((cons, None));
            if direct_literal(element).is_some() {
                stack.push((element, Some(binder)));
            } else if finite_chain(element) && !is_name(element, "Nil") {
                push_chain(element, binder, stack);
            } else {
                stack.push((element, None));
            }
            push_chain(rest, binder, stack);
        }
        _ => stack.push((chain, None)),
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
