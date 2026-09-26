//! Overflow-checked arithmetic assigned through an overloaded place: in
//! `v[i] = a + b` the place `*IndexMut::index_mut(&mut v, i)` is computed after
//! the checked sum, so the sum's result is used in a later block than its
//! overflow assert (`chelis_ir::eval::reduce_window`, eval.rs:1052).

/// The source-index computation of `reduce_window`'s window closure.
pub fn window_source(out_idx: &[usize], strides: &[usize], window_pos: &[usize], leading: usize) -> Vec<usize> {
    let rank = out_idx.len();
    let n = window_pos.len();
    let mut src_idx = vec![0usize; rank];
    src_idx[..leading].copy_from_slice(&out_idx[..leading]);
    for i in 0..n {
        src_idx[leading + i] = out_idx[leading + i] * strides[i] + window_pos[i];
    }
    src_idx
}

fn each_window(window_shape: &[usize], mut f: impl FnMut(&[usize])) {
    let n = window_shape.len();
    let mut window_pos = vec![0usize; n];
    loop {
        f(&window_pos);
        if n == 0 {
            break;
        }
        let mut carry = n;
        for i in (0..n).rev() {
            window_pos[i] += 1;
            if window_pos[i] < window_shape[i] {
                carry = i;
                break;
            }
            window_pos[i] = 0;
        }
        if carry == n {
            break;
        }
    }
}

/// The same update inside the `FnMut` closure of a window walk.
pub fn window_group(out_idx: &[usize], strides: &[usize], window_shape: &[usize]) -> Vec<usize> {
    let rank = out_idx.len();
    let n = window_shape.len();
    let leading = rank - n;
    let mut group = Vec::new();
    each_window(window_shape, |window_pos| {
        let mut src_idx = vec![0usize; rank];
        for i in 0..n {
            src_idx[leading + i] = out_idx[leading + i] * strides[i] + window_pos[i];
        }
        group.push(src_idx[rank - 1]);
    });
    group
}

/// A single checked sum assigned through `IndexMut`, whose overflow panics.
pub fn assign_sum(values: &[u8], left: u8, right: u8) -> Vec<u8> {
    let mut out = values.to_vec();
    out[0] = left + right;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sums_assigned_through_index_mut() {
        assert_eq!(window_source(&[1, 2, 3], &[2, 3], &[1, 0], 1), vec![1, 5, 9]);
        assert_eq!(window_source(&[4, 0], &[1], &[2], 1), vec![4, 2]);
        assert_eq!(window_group(&[0, 1], &[2], &[3]), vec![2, 3, 4]);
        assert_eq!(window_group(&[1, 1], &[1, 2], &[2, 2]), vec![2, 3, 2, 3]);
        assert_eq!(assign_sum(&[9, 9], 3, 4), vec![7, 9]);
        assert!(std::panic::catch_unwind(|| assign_sum(&[0], 200, 100)).is_err());
        assert!(std::panic::catch_unwind(|| window_source(&[usize::MAX, 0], &[2], &[0], 0)).is_err());
    }
}
