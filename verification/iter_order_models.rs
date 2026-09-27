//! Iterator consumers and adapters of the production crates (Aeneas.Std Core/Iter,
//! IteratorAdapters, SliceIter): `Iterator::min` (chelis-types `infer::program` and
//! `infer::declarations`: the least member or item index, over `iter()`, `map` and `filter_map`),
//! `Iterator::eq`/`ne` (chelis-ir `specialize`: `axes.iter().copied().eq(0..axes.len())`,
//! chelis-types `unify`: `left.iter().map(..).ne(right.iter().map(..))`) with the length check of
//! two `TrustedLen` iterators, and `skip_while` (chelis-types `opacity`: the module prefix).

use std::cmp::Ordering;

/// Ordered by `key` only: equal minima are told apart by `tag`.
#[derive(Debug, Clone, Copy)]
pub struct Keyed {
    pub key: u32,
    pub tag: u32,
}

impl PartialEq for Keyed {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Keyed {}

impl PartialOrd for Keyed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Keyed {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// As chelis-types `infer::program`: `let Some(&representative) = members.iter().min()`
/// (the first of equal minima).
pub fn representative(members: &[Keyed]) -> Option<(u32, u32)> {
    let Some(&least) = members.iter().min() else {
        return None;
    };
    Some((least.key, least.tag))
}

/// As chelis-types `infer::declarations`: the least item index of a component, or `usize::MAX`.
pub fn least_index(component: &[usize], item_index: &[usize]) -> usize {
    component.iter().map(|vertex| item_index[*vertex]).min().unwrap_or(usize::MAX)
}

/// `Map::min` with a mapper counting its calls through a captured borrow.
pub fn counted_min(values: &[u32], calls: &mut u32) -> Option<u32> {
    values
        .iter()
        .map(|value| {
            *calls += 1;
            *value % 7
        })
        .min()
}

/// As chelis-types `infer::declarations`: `filter_map(..).min()` over the kept items.
pub fn earliest_even_half(values: &[u32]) -> Option<u32> {
    values.iter().filter_map(|value| if *value % 2 == 0 { Some(*value / 2) } else { None }).min()
}

/// `FilterMap::min` with a closure counting its calls through a captured borrow.
pub fn counted_filter_min(values: &[u32], calls: &mut u32) -> Option<u32> {
    values
        .iter()
        .filter_map(|value| {
            *calls += 1;
            if *value % 3 == 0 { None } else { Some(*value % 5) }
        })
        .min()
}

/// As chelis-ir `specialize`: the axes are the identity permutation.
pub fn is_identity(axes: &[usize]) -> bool {
    axes.iter().copied().eq(0..axes.len())
}

/// As chelis-types `unify`: the sorted mapped axes of two rows differ.
pub fn axes_differ(left: &[(usize, u32)], right: &[(usize, u32)]) -> bool {
    left.iter().map(|(axis, _)| axis).ne(right.iter().map(|(axis, _)| axis))
}

/// `Map::eq` whose mapper panics on a zero divisor: two `TrustedLen` iterators of different
/// lengths are unequal without a call of the mapper, equal lengths compare item by item.
pub fn quotients_equal(divisors: &[u32], expected: &[u32]) -> bool {
    divisors.iter().map(|divisor| 60 / *divisor).eq(expected.iter().copied())
}

/// `Map::ne` against a `Filter` (not `TrustedLen`): no length check, item by item.
pub fn quotients_differ_from_odd(divisors: &[u32], expected: &[u32]) -> bool {
    divisors.iter().map(|divisor| 60 / *divisor).ne(expected.iter().copied().filter(|x| *x % 2 == 1))
}

/// As chelis-types `opacity`: the segments from the first capitalized one on.
pub fn module_segments<'a>(segments: &[&'a str]) -> Vec<&'a str> {
    segments
        .iter()
        .copied()
        .skip_while(|seg| seg.chars().next().is_none_or(|c| !c.is_ascii_uppercase()))
        .collect()
}

/// `skip_while` with a predicate counting its calls through a captured borrow: it is not
/// called again once an item is kept.
pub fn counted_skip(values: &[u32], bound: u32, calls: &mut u32) -> Vec<u32> {
    values
        .iter()
        .copied()
        .skip_while(|value| {
            *calls += 1;
            *value < bound
        })
        .collect()
}

/// `SkipWhile::next` one item at a time.
pub fn first_two_kept(values: &[u32], bound: u32) -> (Option<u32>, Option<u32>) {
    let mut kept = values.iter().copied().skip_while(|value| *value < bound);
    let first = kept.next();
    (first, kept.next())
}

/// Rejected (a fail-closed root, not a model): the `Iterator::eq` model takes its iterators by
/// value and returns no state, so it cannot give back the counter a mapping closure borrows.
pub fn reject_stateful_eq(left: &[u32], right: &[u32], calls: &mut u32) -> bool {
    left.iter()
        .map(|value| {
            *calls += 1;
            *value
        })
        .eq(right.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyed(pairs: &[(u32, u32)]) -> Vec<Keyed> {
        pairs.iter().map(|&(key, tag)| Keyed { key, tag }).collect()
    }

    #[test]
    fn minima() {
        assert_eq!(representative(&keyed(&[(3, 0), (1, 1), (2, 2), (1, 3)])), Some((1, 1)));
        assert_eq!(representative(&keyed(&[(5, 9)])), Some((5, 9)));
        assert_eq!(representative(&[]), None);
        assert_eq!(least_index(&[2, 0, 1], &[7, 9, 4]), 4);
        assert_eq!(least_index(&[], &[7, 9, 4]), usize::MAX);
        let mut calls = 2;
        assert_eq!(counted_min(&[20, 15, 9, 14], &mut calls), Some(0));
        assert_eq!(calls, 6);
        let mut calls = 0;
        assert_eq!(counted_min(&[], &mut calls), None);
        assert_eq!(calls, 0);
        assert_eq!(earliest_even_half(&[9, 14, 3, 8, 12]), Some(4));
        assert_eq!(earliest_even_half(&[9, 3]), None);
        let mut calls = 1;
        assert_eq!(counted_filter_min(&[9, 13, 6, 22, 7], &mut calls), Some(2));
        assert_eq!(calls, 6);
    }

    #[test]
    fn equalities() {
        assert!(is_identity(&[0, 1, 2, 3]));
        assert!(!is_identity(&[0, 2, 1]));
        assert!(is_identity(&[]));
        assert!(!is_identity(&[1]));
        assert!(!axes_differ(&[(0, 5), (2, 1)], &[(0, 9), (2, 3)]));
        assert!(axes_differ(&[(0, 5), (2, 1)], &[(0, 9), (1, 3)]));
        assert!(axes_differ(&[(0, 5)], &[(0, 9), (1, 3)]));
        assert!(!axes_differ(&[], &[]));
        assert!(quotients_equal(&[1, 2, 3], &[60, 30, 20]));
        assert!(!quotients_equal(&[1, 2, 4], &[60, 30, 20]));
        // Different lengths: false before any division, so the zero divisor does not panic.
        assert!(!quotients_equal(&[0, 1], &[60]));
        assert!(!quotients_differ_from_odd(&[4, 12], &[15, 2, 5]));
        assert!(quotients_differ_from_odd(&[4, 12], &[15, 2, 5, 7]));
        assert!(quotients_differ_from_odd(&[4], &[15, 5]));
    }

    #[test]
    #[should_panic]
    fn equalities_divide_when_the_lengths_agree() {
        quotients_equal(&[0, 1], &[60, 60]);
    }

    #[test]
    fn rejected_eq_counts_the_mapped_items() {
        let mut calls = 0;
        assert!(reject_stateful_eq(&[1, 2], &[1, 2], &mut calls));
        assert_eq!(calls, 2);
        // Two `TrustedLen` iterators of different lengths: unequal without a call.
        assert!(!reject_stateful_eq(&[1, 2], &[1], &mut calls));
        assert_eq!(calls, 2);
    }

    #[test]
    fn skips() {
        assert_eq!(module_segments(&["std", "io", "Error", "kind"]), vec!["Error", "kind"]);
        assert_eq!(module_segments(&["", "Map", "x"]), vec!["Map", "x"]);
        assert_eq!(module_segments(&["a", "b"]), Vec::<&str>::new());
        let mut calls = 3;
        assert_eq!(counted_skip(&[1, 2, 7, 1, 9], 5, &mut calls), vec![7, 1, 9]);
        assert_eq!(calls, 6);
        let mut calls = 0;
        assert_eq!(counted_skip(&[1, 2], 5, &mut calls), Vec::<u32>::new());
        assert_eq!(calls, 2);
        assert_eq!(first_two_kept(&[1, 6, 2, 8], 5), (Some(6), Some(2)));
        assert_eq!(first_two_kept(&[1, 6], 5), (Some(6), None));
        assert_eq!(first_two_kept(&[1, 2], 5), (None, None));
    }
}
