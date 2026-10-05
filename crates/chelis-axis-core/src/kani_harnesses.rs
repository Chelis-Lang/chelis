//! Independent executable restatements of the four production contracts.
//! Values retain their full machine widths; only rank and slice length are bounded.

use crate::{checked_inverse, is_permutation, normalize_axis, reduction_survivors};

const fn bound() -> usize {
    let Some(text) = option_env!("CHELIS_AXIS_KANI_BOUND") else {
        return 4;
    };
    let bytes = text.as_bytes();
    let mut result = 0;
    let mut i = 0;
    while i < bytes.len() {
        assert!(bytes[i] >= b'0' && bytes[i] <= b'9');
        result = result * 10 + (bytes[i] - b'0') as usize;
        i += 1;
    }
    result
}

const BOUND: usize = bound();
// Only the allocation-free normalization harness is calibrated past 256.
const SLICE_BOUND: usize = if BOUND > 256 { 256 } else { BOUND };

fn permutation_model(axes: &[usize], rank: usize) -> bool {
    if axes.len() != rank {
        return false;
    }
    for i in 0..axes.len() {
        if axes[i] >= rank {
            return false;
        }
        for j in 0..i {
            if axes[i] == axes[j] {
                return false;
            }
        }
    }
    true
}

#[kani::proof]
fn admission() {
    let rank: usize = kani::any();
    kani::assume(rank <= BOUND);
    let values: [usize; SLICE_BOUND + 1] = kani::any();
    let len: usize = kani::any();
    kani::assume(len <= values.len());
    let axes = &values[..len];
    kani::cover!(rank == 0 && len == 0);
    kani::cover!(len != rank);
    kani::cover!(permutation_model(axes, rank));
    kani::cover!(len == rank && axes.iter().any(|a| *a >= rank));
    kani::cover!(len >= 2 && values[0] == values[1]);
    assert_eq!(is_permutation(axes, rank), permutation_model(axes, rank));
}

#[kani::proof]
fn normalization() {
    let rank: usize = kani::any();
    kani::assume(rank <= BOUND);
    let raw: i64 = kani::any();
    let position = if raw < 0 {
        raw as i128 + rank as i128
    } else {
        raw as i128
    };
    let expected = if rank as u128 > i64::MAX as u128 || position < 0 || position >= rank as i128 {
        None
    } else {
        Some(position as usize)
    };
    kani::cover!(raw == i64::MIN);
    kani::cover!(rank > 0 && raw == -1);
    assert_eq!(normalize_axis(rank, raw), expected);
}

#[kani::proof]
fn survivors() {
    let rank: usize = kani::any();
    kani::assume(rank <= BOUND);
    let axis: usize = kani::any();
    match reduction_survivors(rank, axis) {
        None => {
            assert!(axis >= rank);
        }
        Some(positions) => {
            assert!(axis < rank);
            assert_eq!(positions.len(), rank - 1);
            for (k, position) in positions.iter().enumerate() {
                assert_eq!(*position, if k < axis { k } else { k + 1 });
            }
        }
    }
}

#[kani::proof]
fn inverse() {
    let rank: usize = kani::any();
    kani::assume(rank <= BOUND);
    let values: [usize; SLICE_BOUND + 1] = kani::any();
    let len: usize = kani::any();
    kani::assume(len <= values.len());
    let axes = &values[..len];
    match checked_inverse(axes, rank) {
        None => {
            assert!(!permutation_model(axes, rank));
        }
        Some(result) => {
            assert!(permutation_model(axes, rank));
            assert_eq!(result.len(), rank);
            for (i, axis) in axes.iter().enumerate() {
                assert_eq!(result[*axis], i);
            }
        }
    }
}

#[cfg(axis_false_postcondition)]
#[kani::proof]
fn false_permutation_postcondition() {
    // Same deliberately false accepted-result postcondition as the Verus runner.
    assert!(!is_permutation(&[], 0));
}
