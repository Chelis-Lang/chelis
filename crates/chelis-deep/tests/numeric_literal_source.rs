use LiteralFamilyFit::{Fits, IntegerOutOfRange};
use chelis_deep::{
    Atom, DtypeFamily, Expr, LiteralFamilyFit, classify_literal_source, parser::parse_str,
};

fn expr(source: &str) -> Expr {
    parse_str(source)
        .expect("Deep")
        .pop()
        .expect("one expression")
}

#[test]
fn both_numeric_polarities_share_one_classification() {
    for (deep, atom) in [
        ("(lit {} 0.1)", Atom::Float(0.1)),
        ("(lit {} -0.1)", Atom::Float(-0.1)),
        ("(lit {} 1)", Atom::Int(1)),
        ("(lit {} -1)", Atom::Int(-1)),
    ] {
        let expression = expr(deep);
        let source = classify_literal_source(&expression).expect("literal source");
        assert_eq!(source.numeric_atom(), Some(&atom));
    }
}

#[test]
fn family_fit_covers_every_active_integer_width_and_total_float_finalization() {
    for (deep, family, expected) in [
        ("(lit {} -128)", DtypeFamily::Int, Fits),
        ("(lit {} 127)", DtypeFamily::Numeric, Fits),
        ("(lit {} -129)", DtypeFamily::Int, IntegerOutOfRange),
        ("(lit {} 128)", DtypeFamily::Numeric, IntegerOutOfRange),
        ("(lit {} 10000000000.0)", DtypeFamily::Float, Fits),
    ] {
        let expression = expr(deep);
        assert_eq!(
            classify_literal_source(&expression)
                .expect("literal")
                .family_fit(family),
            expected
        );
    }
}

#[test]
fn computed_nested_and_nonnumeric_negation_fail_closed() {
    for deep in [
        "(app {} (var {} neg) (var {} x))",
        "(app {} (var {} neg) (lit {} 0.1))",
        "(app {} (var {} neg) (lit {} 1))",
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
        assert!(classify_literal_source(&expr(negative)).is_none());
    }
}
