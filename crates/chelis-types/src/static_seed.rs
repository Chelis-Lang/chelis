//! Closed typed seed constants shared by source admission and lowering.
//! No variables, aliases, arithmetic folding, or guessed literal dtypes.
use crate::types::Prim;
use chelis_deep::ast::{Atom, Expr};
use chelis_deep::{DeepTag, ExprCarrier};

/// The explicit signed literal surface in spec/02 §P5. Surf encodes a minus
/// sign as one builtin `neg` application around the typed integer literal;
/// Deep may carry the negative integer directly. Casts and computed seeds
/// remain outside source admission even when lowering can evaluate them.
pub fn literal_seed(expr: &Expr, neg_is_builtin: bool) -> Option<crate::ScalarValue> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Lit, _, _) => {}
        ExprCarrier::DecodedNode(DeepTag::App, _, [callee, inner]) if neg_is_builtin => {
            if !expr_is_var_named(callee, "neg") {
                return None;
            }
            match inner.carrier() {
                ExprCarrier::DecodedNode(DeepTag::Lit, _, [Expr::Atom(Atom::Int(value), _)])
                    if *value >= 0 => {}
                ExprCarrier::DecodedNode(_, _, _)
                | ExprCarrier::StructuralList(_)
                | ExprCarrier::UndecodableHead(_, _, _)
                | ExprCarrier::Atom(_)
                | ExprCarrier::MetadataMap(_)
                | ExprCarrier::MetadataExpression(_)
                | ExprCarrier::MalformedLegacyList(_) => return None,
            }
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return None,
    }
    constant_seed(expr, neg_is_builtin)
}

/// Evaluate lowering's existing closed scalar grammar as signed-i64 seed
/// bits under [05-RNG-1]. Source admission additionally checks §P5's literal form.
/// The caller must identify whether lexical scope leaves `neg` as the builtin.
/// Returning the tagged scalar preserves the seed's type across this boundary.
pub fn constant_seed(expr: &Expr, neg_is_builtin: bool) -> Option<crate::ScalarValue> {
    let value = extract_type_checked_scalar(expr, neg_is_builtin)?;
    (value.prim() == Prim::Int64).then_some(value)
}

fn primitive(expr: &Expr) -> Option<Prim> {
    let kids = match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::TPrim, _, kids) => kids,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return None,
    };
    let [Expr::Atom(Atom::Name(name), _)] = kids else {
        return None;
    };
    Prim::parse_name(name)
}

fn expr_is_var_named(expr: &Expr, expected: &str) -> bool {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Var, _, [Expr::Atom(Atom::Name(name), _)]) => {
            name == expected
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => false,
    }
}

/// Read an optional scalar type stamp without treating malformed type
/// metadata as absence. The outer `Option` is recognition success; the inner
/// one distinguishes an unstamped expression from a stamped scalar.
fn optional_scalar_prim(expr: &Expr) -> Option<Option<Prim>> {
    let metadata = match expr.carrier() {
        ExprCarrier::DecodedNode(_, metadata, _) => metadata,
        ExprCarrier::UndecodableHead(_, _, _) if matches!(expr, Expr::List(_, _)) => {
            return None;
        }
        ExprCarrier::MalformedLegacyList(_) => return None,
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return Some(None),
    };
    match metadata.ty() {
        Some(ty) => Some(Some(primitive(ty.expression())?)),
        None => Some(None),
    }
}

/// Decode one canonical numeric `lit` under [04-LIT-1].
///
/// This is the fold's only literal ingress. It jointly validates the value
/// atom, declared primitive, and the optional provenance marker. The sole
/// cross-family form is an exact integer atom with one
/// `literal_source: integer` marker at a float dtype; `scalar_from_i64`
/// performs its one required target-width finalization without an f64 hop.
fn extract_type_checked_literal(expr: &Expr) -> Option<crate::ScalarValue> {
    use crate::{scalar_from_f64, scalar_from_i64};

    let (metadata, kids) = match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Lit, metadata, kids) => (metadata, kids),
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return None,
    };
    let [Expr::Atom(atom, _)] = kids else {
        return None;
    };
    let declared = optional_scalar_prim(expr)??;
    let literal_source = metadata.literal_source();
    let integer_source = literal_source.is_some();

    match (atom, declared, literal_source) {
        (Atom::Int(value), prim, None) if prim.is_integer() => {
            scalar_from_i64("lit", prim, *value).ok()
        }
        (Atom::Int(value), prim, Some(_)) if prim.is_float() && integer_source => {
            scalar_from_i64("lit", prim, *value).ok()
        }
        (Atom::Float(value), prim, None) if prim.is_float() => {
            scalar_from_f64("lit", prim, *value).ok()
        }
        (Atom::Bool(value), Prim::Bool, None) => {
            scalar_from_i64("lit", Prim::Bool, i64::from(*value)).ok()
        }
        (Atom::Str(_) | Atom::Name(_) | Atom::Tag(_), _, _)
        | (Atom::Int(_) | Atom::Float(_) | Atom::Bool(_), _, _) => None,
    }
}

/// Evaluate the closed static scalar grammar while preserving a checked dtype
/// at every edge.
///
/// This is stricter than `extract_numeric_leaf`, whose `RawScalar` form is
/// intentionally useful for general literal folding but cannot distinguish
/// `true` from integer `1`. A seed needs that distinction: literal payload and
/// type metadata must agree before a wrapper can consume the value, while a
/// successful explicit cast establishes its target dtype per [04-NUM-14].
fn extract_type_checked_scalar(expr: &Expr, neg_is_builtin: bool) -> Option<crate::ScalarValue> {
    match expr.carrier() {
        // A BARE atom carries no type metadata, so there is nothing for the
        // payload to agree with and this fold declines it. Stamping one here
        // would invent a dtype the source never wrote, in two ways that both
        // matter. It would widen the fold's accept set past the checker's,
        // which classifies a bare integer atom as an UNSUFFIXED seed literal
        // and rejects it (`crate::infer::expr::seed_literal_form`); and
        // any stamp it picked would contradict spec/04-type-system.md §5.3,
        // where an integer literal defaults to `i32` and a float literal to
        // `f32` rather than to the i64/f64 a seed wants. So
        // `extract_type_checked_literal` stays the fold's ONLY literal
        // ingress, as its doc says, and a bare atom reaches the caller's loud
        // rejection instead of a guessed value (chelis#794).
        ExprCarrier::Atom(_) => None,
        ExprCarrier::DecodedNode(tag, _, kids) => match tag {
            DeepTag::Lit => extract_type_checked_literal(expr),
            DeepTag::Cast => {
                let inner = extract_type_checked_scalar(kids.first()?, neg_is_builtin)?;
                let target = primitive(kids.get(1)?)?;
                if optional_scalar_prim(expr)?.is_some_and(|declared| declared != target) {
                    return None;
                }
                match chelis_deep::cast_mode_of(kids).ok()? {
                    chelis_deep::CastMode::Checked => {
                        crate::CheckedCastPlan::new(inner.prim(), target)
                            .ok()?
                            .cast_scalar("cast", inner)
                            .ok()
                    }
                    chelis_deep::CastMode::Trunc => {
                        if !inner.prim().is_float() || !target.is_integer() {
                            return None;
                        }
                        crate::cast_trunc_scalar("cast_trunc", inner, target).ok()
                    }
                }
            }
            DeepTag::App => {
                if !neg_is_builtin || kids.len() != 2 || !expr_is_var_named(&kids[0], "neg") {
                    return None;
                }
                let inner = extract_type_checked_scalar(&kids[1], neg_is_builtin)?;
                let negated = if inner.prim().is_float() {
                    crate::float_unop(crate::FloatUnOp::Neg, inner).ok()?
                } else if inner.prim().is_integer() {
                    crate::int_unop(crate::IntUnOp::Neg, inner).ok()?
                } else {
                    return None;
                };
                if optional_scalar_prim(expr)?.is_some_and(|declared| declared != negated.prim()) {
                    return None;
                }
                Some(negated)
            }
            _ => None,
        },
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::{ExprCarrier, List, MetaExpr, Metadata, Span, UnknownFormData};

    fn span() -> Span {
        Span::new(0, 0)
    }

    fn i64_literal(value: i64) -> Expr {
        let span = span();
        let prim = Expr::node(
            DeepTag::TPrim,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("i64".into()), span)],
            span,
        );
        let mut metadata = Metadata::default();
        metadata
            .insert(chelis_deep::annotations::MetadataValue::Type(
                chelis_deep::annotations::TypeSyntax::try_new(prim)
                    .expect("canonical i64 type syntax"),
            ))
            .expect("canonical i64 type metadata");
        Expr::node(
            DeepTag::Lit,
            metadata,
            vec![Expr::Atom(Atom::Int(value), span)],
            span,
        )
    }

    fn legacy_node(expr: &Expr) -> Expr {
        let Expr::Node(node, span) = expr else {
            panic!("test fixture must be a decoded node");
        };
        Expr::List(node.to_list(*span), *span)
    }

    #[test]
    fn seed_reader_matches_successor_and_legacy_carriers() {
        let successor = i64_literal(7);
        let legacy = legacy_node(&successor);

        let successor_literal = literal_seed(&successor, true);
        let legacy_literal = literal_seed(&legacy, true);
        assert!(successor_literal.is_some());
        assert_eq!(legacy_literal, successor_literal);

        let successor_constant = constant_seed(&successor, true);
        let legacy_constant = constant_seed(&legacy, true);
        assert!(successor_constant.is_some());
        assert_eq!(legacy_constant, successor_constant);
    }

    #[test]
    fn seed_reader_declines_every_nondecoded_carrier() {
        let span = span();
        let malformed = Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Lit), span),
                    Expr::Atom(Atom::Name("not-metadata".into()), span),
                    Expr::Atom(Atom::Int(7), span),
                ],
            },
            span,
        );
        let carriers = [
            Expr::BareList(vec![Expr::Atom(Atom::Int(7), span)], span),
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "future-literal".into(),
                meta: Metadata::default(),
                children: vec![Expr::Atom(Atom::Int(7), span)],
                span,
            })),
            Expr::Atom(Atom::Int(7), span),
            Expr::Map(Metadata::default(), span),
            Expr::MetaExpr(
                MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(Expr::Atom(Atom::Int(7), span)),
                },
                span,
            ),
            malformed,
        ];

        for expr in carriers {
            assert!(
                !matches!(expr.carrier(), ExprCarrier::DecodedNode(..)),
                "negative control must stay outside the decoded-node carrier"
            );
            assert_eq!(literal_seed(&expr, true), None);
            assert_eq!(constant_seed(&expr, true), None);
        }
    }

    #[test]
    fn static_seed_reader_has_no_local_optional_carrier_adapter() {
        let source = include_str!("static_seed.rs");
        let definition = ["fn stamped_", "parts"].concat();
        let call = ["stamped_", "parts("].concat();
        assert!(
            !source.contains(&definition) && !source.contains(&call),
            "E5b requires each static-seed read to disposition ExprCarrier directly"
        );
    }
}
