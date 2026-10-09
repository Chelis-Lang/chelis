//! One-sided real-arithmetic range goals retain the bound's typed bits.
mod support;

use chelis_prove::discharge::{Goal, GoalShape, IntervalBox};
use chelis_types::{dtype_semantics::scalar_from_f64, types::Prim};

#[test]
fn upper_bound_goal_preserves_typed_threshold_and_rejects_invalid_boxes() {
    crate::support::isolate();
    let upper = scalar_from_f64("proof transport", Prim::F64, 3.991125645861615).unwrap();
    let tag = |value| scalar_from_f64("proof transport", Prim::F64, value).unwrap();
    let input = IntervalBox {
        dims: vec![("x".into(), tag(-1.0), tag(1.0))],
    };
    let goal = Goal::scalar_upper_bound(input, upper).unwrap();
    assert!(
        matches!(goal.shape, GoalShape::ScalarUpperBound { upper: actual, .. } if actual == upper)
    );
    for dims in [
        vec![("x".into(), tag(1.0), tag(-1.0))],
        vec![("x".into(), tag(f64::NAN), tag(1.0))],
        vec![
            ("x".into(), tag(0.0), tag(1.0)),
            ("x".into(), tag(0.0), tag(1.0)),
        ],
    ] {
        assert!(Goal::scalar_upper_bound(IntervalBox { dims }, upper).is_err());
    }
    let mixed = IntervalBox {
        dims: vec![(
            "x".into(),
            tag(0.0),
            scalar_from_f64("proof transport", Prim::F32, 1.0).unwrap(),
        )],
    };
    assert!(Goal::scalar_upper_bound(mixed, upper).is_err());
    let infinite = scalar_from_f64("proof transport", Prim::F64, f64::INFINITY).unwrap();
    assert!(Goal::scalar_upper_bound(IntervalBox { dims: vec![] }, infinite).is_err());
}
