//! Phase 0 executable reproduction for chelis#888.
//!
//! This test locks the known-bad saturating collision until Phase 1 replaces
//! `DimExprKey` with exact, overflow-safe capacity identity. When that repair
//! lands, this test must invert and move into the Phase 1 positive controls.
//!
//! This is the key-level half of the witness, and on its own it proves only
//! that two capacities compare equal. The consequence #888 actually claims,
//! that the collision reaches the memory planner and produces a wrong slot
//! assignment, is pinned at
//! `crates/chelis-backend-c/tests/issue_888_capacity_collision.rs`. Both must
//! invert together.

use chelis_ir::dag::{DimExpr, DimExprKey};

fn mul(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    DimExpr::Mul(Box::new(lhs), Box::new(rhs))
}

fn concrete(value: usize) -> DimExpr {
    DimExpr::Concrete(value)
}

fn sym(name: &str) -> DimExpr {
    DimExpr::Sym(name.into())
}

#[test]
fn phase0_reproduces_distinct_symbolic_capacities_colliding_at_usize_max() {
    let smaller = mul(
        sym("n"),
        mul(concrete(1usize << 40), concrete(1usize << 40)),
    );
    let larger = mul(
        sym("n"),
        mul(concrete(1usize << 40), concrete(1usize << 41)),
    );

    assert_eq!(smaller.as_concrete(), None);
    assert_eq!(larger.as_concrete(), None);

    let saturated = DimExprKey::Mul(vec![
        DimExprKey::Concrete(usize::MAX),
        DimExprKey::Sym("n".into()),
    ]);
    assert_eq!(smaller.normalized_key(), saturated);
    assert_eq!(larger.normalized_key(), saturated);
    assert_eq!(
        smaller.normalized_key(),
        larger.normalized_key(),
        "Phase 0 must reproduce #888's false capacity equality until Phase 1 fixes it"
    );
}
