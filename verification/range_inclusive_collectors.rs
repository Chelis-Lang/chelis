// Collectors over `RangeInclusive` of a std integer (core iter/range.rs):
// these native results pin the collected items of its fold override, which
// yields the final endpoint without stepping past it (also at the type's
// maximum), the empty range, and a range resumed after next.

/// chelis-types deep_type.rs edit_distance: `(0..=right.len()).collect()`.
pub fn row(len: usize) -> Vec<usize> {
    (0..=len).collect()
}

pub fn shifted(start: i32, end: i32) -> Vec<i32> {
    (start..=end).map(|x| x * 2).collect()
}

/// The range ending at the maximum value yields it without overflowing.
pub fn to_max(start: usize) -> Vec<usize> {
    (start..=usize::MAX).collect()
}

pub fn extended(mut out: Vec<usize>, start: usize, end: usize) -> Vec<usize> {
    out.extend(start..=end);
    out
}

/// After one next, the rest of the range is collected.
pub fn rest_after_next(start: usize, end: usize) -> (Option<usize>, Vec<usize>) {
    let mut range = start..=end;
    let first = range.next();
    (first, range.collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collectors() {
        assert_eq!(row(3), vec![0, 1, 2, 3]);
        assert_eq!(row(0), vec![0]);
        assert_eq!(shifted(-1, 2), vec![-2, 0, 2, 4]);
        assert_eq!(shifted(3, 2), vec![]);
        assert_eq!(to_max(usize::MAX - 1), vec![usize::MAX - 1, usize::MAX]);
        assert_eq!(extended(vec![9], 2, 4), vec![9, 2, 3, 4]);
        assert_eq!(extended(vec![9], 4, 2), vec![9]);
        assert_eq!(rest_after_next(1, 3), (Some(1), vec![2, 3]));
        assert_eq!(rest_after_next(5, 5), (Some(5), vec![]));
        assert_eq!(rest_after_next(6, 5), (None, vec![]));
    }
}
