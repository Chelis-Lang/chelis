use chelis_deep::{
    Atom, DeepTag, Expr, LiteralSourceShape, MetaMap, Span, classify_literal_source,
};

fn node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(tag, MetaMap::default(), children, Span::new(0, 0))
}

fn lit(atom: Atom) -> Expr {
    node(DeepTag::Lit, vec![Expr::Atom(atom, Span::new(0, 0))])
}

fn neg(expr: Expr) -> Expr {
    node(
        DeepTag::App,
        vec![
            node(
                DeepTag::Var,
                vec![Expr::Atom(Atom::Name("neg".into()), Span::new(0, 0))],
            ),
            expr,
        ],
    )
}

#[test]
fn direct_and_unary_minus_numeric_literals_share_one_classification() {
    for (expr, shape, atom) in [
        (
            lit(Atom::Float(0.1)),
            LiteralSourceShape::Direct,
            Atom::Float(0.1),
        ),
        (
            neg(lit(Atom::Float(0.1))),
            LiteralSourceShape::UnaryMinus,
            Atom::Float(0.1),
        ),
        (lit(Atom::Int(1)), LiteralSourceShape::Direct, Atom::Int(1)),
        (
            neg(lit(Atom::Int(1))),
            LiteralSourceShape::UnaryMinus,
            Atom::Int(1),
        ),
    ] {
        let source = classify_literal_source(&expr).expect("numeric literal source");
        assert_eq!(source.shape(), shape);
        assert_eq!(source.numeric_atom(), Some(&atom));
        let expected = match (shape, atom) {
            (LiteralSourceShape::Direct, atom) => atom,
            (LiteralSourceShape::UnaryMinus, Atom::Float(value)) => Atom::Float(-value),
            (LiteralSourceShape::UnaryMinus, Atom::Int(value)) => Atom::Int(-value),
            (LiteralSourceShape::UnaryMinus, _) => unreachable!(),
        };
        assert_eq!(source.folded_numeric_atom(), Some(expected));
    }
}

#[test]
fn computed_and_nested_negation_are_not_literal_sources() {
    let variable = node(
        DeepTag::Var,
        vec![Expr::Atom(Atom::Name("x".into()), Span::new(0, 0))],
    );
    assert!(classify_literal_source(&neg(variable)).is_none());
    assert!(classify_literal_source(&neg(neg(lit(Atom::Float(0.1))))).is_none());
}

#[test]
fn direct_nonnumeric_literals_remain_literal_sources_but_cannot_be_negated() {
    for atom in [Atom::Bool(true), Atom::Str("text".into())] {
        let literal = lit(atom.clone());
        let source = classify_literal_source(&literal).expect("direct literal source");
        assert_eq!(source.shape(), LiteralSourceShape::Direct);
        assert_eq!(source.numeric_atom(), None);
        assert_eq!(source.folded_numeric_atom(), None);
        assert!(classify_literal_source(&neg(literal)).is_none());
    }

    let unit = node(
        DeepTag::Lit,
        vec![Expr::BareList(Vec::new(), Span::new(0, 0))],
    );
    let source = classify_literal_source(&unit).expect("unit literal source");
    assert_eq!(source.shape(), LiteralSourceShape::Direct);
    assert_eq!(source.numeric_atom(), None);
}
