// `impl<A, V: FromIterator<A>> FromIterator<Option<A>> for Option<V>` (core
// option.rs: `iter::try_process` over a `GenericShunt`): these native results
// pin that the inner collector stops at the first `None`, that the mapper runs
// exactly up to it, that the source is left just after it, and the size_hint
// `(0, upper)` the inner collector sees (`(0, Some(0))` once a `None` is seen).

use std::collections::BTreeSet;

pub fn doubled(xs: &[u32]) -> Option<Vec<u32>> {
    xs.iter().map(|x| if *x < 100 { Some(*x * 2) } else { None }).collect()
}

fn leaf(e: &u32) -> Option<u32> {
    if *e % 2 == 0 { Some(*e / 2) } else { None }
}

/// chelis-ir lower.rs literal_to_tensor:
/// `elements.iter().map(|e| extract_numeric_leaf(e)).collect::<Option<Vec<_>>>()`.
pub fn leaves(elements: &[&u32]) -> Option<Vec<u32>> {
    elements.iter().map(|e| leaf(e)).collect::<Option<Vec<u32>>>()
}

/// The mapper runs exactly up to the first None.
pub fn calls_until_none(xs: &[u32]) -> (Option<Vec<u32>>, usize) {
    let mut calls = 0;
    let collected = xs
        .iter()
        .map(|x| {
            calls += 1;
            if *x == 0 { None } else { Some(*x) }
        })
        .collect::<Option<Vec<u32>>>();
    (collected, calls)
}

/// The source is left just after the first None.
pub fn resume_after_none(xs: &[u32]) -> (Option<Vec<u32>>, Option<u32>) {
    let mut iter = xs.iter();
    let collected = iter
        .by_ref()
        .map(|x| if *x == 0 { None } else { Some(*x) })
        .collect::<Option<Vec<u32>>>();
    (collected, iter.next().copied())
}

pub fn option_set(xs: &[u32]) -> Option<BTreeSet<u32>> {
    xs.iter().map(|x| if *x == 0 { None } else { Some(*x) }).collect()
}

/// A collector that records the size_hint it sees before and after each of
/// two polls.
pub struct Hints(pub Vec<(usize, Option<usize>)>);

impl FromIterator<u32> for Hints {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let mut hints = vec![iter.size_hint()];
        let _ = iter.next();
        hints.push(iter.size_hint());
        let _ = iter.next();
        hints.push(iter.size_hint());
        Hints(hints)
    }
}

pub fn shunt_hints(xs: &[u32]) -> Option<Vec<(usize, Option<usize>)>> {
    xs.iter()
        .map(|x| if *x == 0 { None } else { Some(*x) })
        .collect::<Option<Hints>>()
        .map(|hints| hints.0)
}

/// A collector that polls three times, even after the shunt has stored a None:
/// the shunt then reports (0, Some(0)) and continues with the source.
pub struct AfterNone(pub Vec<(usize, Option<usize>)>, pub Vec<u32>);

impl FromIterator<u32> for AfterNone {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let mut hints = Vec::new();
        let mut items = Vec::new();
        if let Some(item) = iter.next() {
            items.push(item);
        }
        hints.push(iter.size_hint());
        if let Some(item) = iter.next() {
            items.push(item);
        }
        hints.push(iter.size_hint());
        if let Some(item) = iter.next() {
            items.push(item);
        }
        hints.push(iter.size_hint());
        AfterNone(hints, items)
    }
}

/// The mapper calls and the source left after an AfterNone collection.
pub fn after_none(xs: &[u32]) -> (Option<(Vec<(usize, Option<usize>)>, Vec<u32>)>, usize, Option<u32>) {
    let mut calls = 0;
    let mut iter = xs.iter();
    let collected = iter
        .by_ref()
        .map(|x| {
            calls += 1;
            if *x == 0 { None } else { Some(*x) }
        })
        .collect::<Option<AfterNone>>()
        .map(|after| (after.0, after.1));
    (collected, calls, iter.next().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collectors() {
        assert_eq!(doubled(&[1, 2, 3]), Some(vec![2, 4, 6]));
        assert_eq!(doubled(&[1, 200, 3]), None);
        assert_eq!(doubled(&[]), Some(vec![]));
        assert_eq!(leaves(&[&2, &4]), Some(vec![1, 2]));
        assert_eq!(leaves(&[&2, &3, &4]), None);
        assert_eq!(calls_until_none(&[1, 0, 2, 0]), (None, 2));
        assert_eq!(calls_until_none(&[1, 2]), (Some(vec![1, 2]), 2));
        assert_eq!(resume_after_none(&[1, 0, 2, 3]), (None, Some(2)));
        assert_eq!(resume_after_none(&[1, 2]), (Some(vec![1, 2]), None));
        assert_eq!(option_set(&[3, 1, 3]), Some([1, 3].into_iter().collect()));
        assert_eq!(option_set(&[3, 0]), None);
    }

    #[test]
    fn hints() {
        assert_eq!(shunt_hints(&[1, 2]), Some(vec![(0, Some(2)), (0, Some(1)), (0, Some(0))]));
        assert_eq!(shunt_hints(&[1, 0, 2]), None);
        assert_eq!(
            after_none(&[4, 5, 6]),
            (Some((vec![(0, Some(2)), (0, Some(1)), (0, Some(0))], vec![4, 5, 6])), 3, None)
        );
        // The shunt is not fused: after the None it polls the source again.
        assert_eq!(after_none(&[4, 0, 6, 7, 8]), (None, 3, Some(7)));
    }
}
