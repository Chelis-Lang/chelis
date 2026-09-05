use chelis_deep::{Atom, Expr, LiteralSourceShape, classify_literal_source, parser::parse_str};

fn expr(source: &str) -> Expr {
    parse_str(source)
        .expect("Deep")
        .pop()
        .expect("one expression")
}

#[test]
fn both_numeric_polarities_share_one_classification() {
    for (deep, shape, unsigned, folded) in [
        (
            "(lit {} 0.1)",
            LiteralSourceShape::Direct,
            Atom::Float(0.1),
            Atom::Float(0.1),
        ),
        (
            "(app {} (var {} neg) (lit {} 0.1))",
            LiteralSourceShape::UnaryMinus,
            Atom::Float(0.1),
            Atom::Float(-0.1),
        ),
        (
            "(lit {} 1)",
            LiteralSourceShape::Direct,
            Atom::Int(1),
            Atom::Int(1),
        ),
        (
            "(app {} (var {} neg) (lit {} 1))",
            LiteralSourceShape::UnaryMinus,
            Atom::Int(1),
            Atom::Int(-1),
        ),
    ] {
        let expression = expr(deep);
        let source = classify_literal_source(&expression).expect("literal source");
        assert_eq!(
            (source.shape(), source.numeric_atom()),
            (shape, Some(&unsigned))
        );
        assert_eq!(source.folded_numeric_atom(), Some(folded));
    }
}

#[test]
fn computed_nested_and_nonnumeric_negation_fail_closed() {
    for deep in [
        "(app {} (var {} neg) (var {} x))",
        "(app {} (var {} neg) (app {} (var {} neg) (lit {} 0.1)))",
    ] {
        assert!(classify_literal_source(&expr(deep)).is_none());
    }
    for (direct, negative) in [
        ("(lit {} true)", "(app {} (var {} neg) (lit {} true))"),
        (
            "(lit {} \"text\")",
            "(app {} (var {} neg) (lit {} \"text\"))",
        ),
        ("(lit {} ())", "(app {} (var {} neg) (lit {} ()))"),
    ] {
        let literal = expr(direct);
        let source = classify_literal_source(&literal).expect("direct literal");
        assert_eq!(source.numeric_atom(), None);
        assert_eq!(source.folded_numeric_atom(), None);
        assert!(classify_literal_source(&expr(negative)).is_none());
    }
}
