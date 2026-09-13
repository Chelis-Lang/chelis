//! One-sided real-arithmetic range goals retain the bound's typed bits.
use chelis_prove::discharge::{Goal, GoalShape, IntervalBox};
use chelis_types::{types::Prim, dtype_semantics::scalar_from_f64};

#[test]
fn upper_bound_goal_preserves_typed_threshold_and_rejects_invalid_boxes() {
    let upper = scalar_from_f64("proof transport", Prim::F64, 3.991125645861615).unwrap();
    let input = IntervalBox { dims: vec![("x".into(), -1.0, 1.0)] };
    let goal = Goal::scalar_upper_bound(input, upper).unwrap();
    assert!(matches!(goal.shape, GoalShape::ScalarUpperBound { upper: actual, .. } if actual == upper));
    for dims in [vec![("x".into(), 1.0, -1.0)], vec![("x".into(), f64::NAN, 1.0)],
                 vec![("x".into(), 0.0, 1.0), ("x".into(), 0.0, 1.0)]] {
        assert!(Goal::scalar_upper_bound(IntervalBox { dims }, upper).is_err());
    }
    let infinite = scalar_from_f64("proof transport", Prim::F64, f64::INFINITY).unwrap();
    assert!(Goal::scalar_upper_bound(IntervalBox { dims: vec![] }, infinite).is_err());
}
