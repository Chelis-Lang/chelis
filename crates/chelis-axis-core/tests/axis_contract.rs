use chelis_axis_core::{Permutation, PermutationError, normalize_axis, reduction_survivors};

#[test]
fn signed_axis_obeys_rank_and_negative_indexing() {
    assert_eq!(normalize_axis(3, 0), Some(0));
    assert_eq!(normalize_axis(3, -1), Some(2));
    assert_eq!(normalize_axis(3, -3), Some(0));
    assert_eq!(normalize_axis(3, 3), None);
    assert_eq!(normalize_axis(3, -4), None);
    assert_eq!(normalize_axis(0, 0), None);
    assert_eq!(normalize_axis(3, i64::MIN), None);
    assert_eq!(normalize_axis(usize::MAX, -1), None);
}

#[test]
fn permutation_tracks_positions_even_when_extents_repeat() {
    let plan = Permutation::from_raw(3, &[2, 0, 1]).unwrap();
    assert_eq!(plan.axes(), &[2, 0, 1]);
    assert_eq!(plan.inverse(), vec![1, 2, 0]);
    let dims = ["n", "n", "m"];
    let reordered: Vec<_> = plan.axes().iter().map(|&i| dims[i]).collect();
    assert_eq!(reordered, ["m", "n", "n"]);
    assert_eq!(
        Permutation::from_indices(0, &[]).unwrap().inverse(),
        Vec::<usize>::new()
    );
}

#[test]
fn permutation_rejects_each_invalid_family() {
    assert_eq!(
        Permutation::from_raw(3, &[0, 1]),
        Err(PermutationError::Arity { rank: 3, got: 2 })
    );
    assert_eq!(
        Permutation::from_raw(2, &[-1, 0]),
        Err(PermutationError::NegativeAxis(-1))
    );
    assert_eq!(
        Permutation::from_raw(2, &[0, 2]),
        Err(PermutationError::OutOfBounds(2))
    );
    assert_eq!(
        Permutation::from_raw(2, &[0, 0]),
        Err(PermutationError::Duplicate(0))
    );
    assert_eq!(
        Permutation::from_indices(2, &[0, usize::MAX]),
        Err(PermutationError::OutOfBounds(usize::MAX))
    );
    assert_eq!(
        Permutation::from_indices(2, &[1, 1]),
        Err(PermutationError::Duplicate(1))
    );
}

#[test]
fn reduction_keeps_all_other_positions_in_order() {
    assert_eq!(reduction_survivors(4, 0), Some(vec![1, 2, 3]));
    assert_eq!(reduction_survivors(4, 2), Some(vec![0, 1, 3]));
    assert_eq!(reduction_survivors(1, 0), Some(vec![]));
    assert_eq!(reduction_survivors(0, 0), None);
    assert_eq!(reduction_survivors(4, 4), None);
}
