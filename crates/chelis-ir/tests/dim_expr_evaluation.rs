//! [04-SHAPE-1]: finite extent projection fails instead of wrapping.
//! Capacity equality itself belongs to the opaque CapacityKey contract.

use chelis_ir::dag::DimExpr;
use chelis_unord::UnordMap;

fn mul(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    DimExpr::Mul(Box::new(lhs), Box::new(rhs))
}

fn div(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    DimExpr::Div(Box::new(lhs), Box::new(rhs))
}

#[test]
fn finite_products_and_exact_quotients_keep_their_values() {
    for (expression, expected) in [
        (mul(DimExpr::Concrete(7), DimExpr::Concrete(6)), 42),
        (div(DimExpr::Concrete(42), DimExpr::Concrete(6)), 7),
        (
            mul(DimExpr::Concrete(usize::MAX), DimExpr::Concrete(1)),
            usize::MAX,
        ),
        (mul(DimExpr::Concrete(0), DimExpr::Concrete(usize::MAX)), 0),
    ] {
        assert_eq!(expression.evaluate(&UnordMap::new()), Ok(expected));
        assert_eq!(expression.as_concrete(), Some(expected));
    }
}

#[test]
fn overflowing_concrete_product_is_not_a_projection() {
    let expression = mul(DimExpr::Concrete(usize::MAX), DimExpr::Concrete(2));
    assert_eq!(expression.as_concrete(), None);
}

#[test]
fn overflowing_evaluation_returns_an_error_in_every_profile() {
    let expression = mul(DimExpr::Sym("n".into()), DimExpr::Concrete(2));
    let bindings = UnordMap::from([("n".into(), usize::MAX)]);
    let error = expression
        .evaluate(&bindings)
        .expect_err("overflow must fail");
    assert!(error.contains("overflow"), "wrong error: {error}");
}

#[test]
fn binding_is_not_a_capacity_proof_or_an_unchecked_projection() {
    let expression = mul(DimExpr::Sym("n".into()), DimExpr::Concrete(2));
    assert_eq!(expression.as_concrete(), None);
    assert!(
        expression
            .evaluate(&UnordMap::new())
            .unwrap_err()
            .contains("missing")
    );
    assert_eq!(
        expression.evaluate(&UnordMap::from([("n".into(), 21)])),
        Ok(42)
    );
}

#[test]
fn invalid_quotient_obligations_survive_an_outer_zero_in_both_orders() {
    for invalid in [
        div(DimExpr::Concrete(0), DimExpr::Concrete(0)),
        div(DimExpr::Concrete(3), DimExpr::Concrete(2)),
    ] {
        for expression in [
            invalid.clone(),
            mul(DimExpr::Concrete(0), invalid.clone()),
            mul(invalid, DimExpr::Concrete(0)),
        ] {
            assert!(expression.evaluate(&UnordMap::new()).is_err());
            assert_eq!(expression.as_concrete(), None);
        }
    }
}
