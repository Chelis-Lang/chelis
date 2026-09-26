#![feature(iter_advance_by, try_trait_v2)]

// Flatten<I> is FlattenCompat<I, U> (core flatten.rs): these native results pin
// its schedule (front inner, fused outer, inner overrides), the std size_hint
// (front inner, outer exhaustion, constant-size and OneShot inners), and the
// sources it is used with: option::IntoIter, &BTreeSet, &Vec, arrays.

use std::collections::{BTreeMap, BTreeSet};

/// chelis-ir ownership/last_use.rs: `Copied<Flatten<option::IntoIter<&BTreeSet>>>`
/// extending a Vec.
pub fn extend_successors(
    pending: &mut Vec<u32>,
    successors: &BTreeMap<u32, BTreeSet<u32>>,
    point: u32,
) {
    pending.extend(successors.get(&point).into_iter().flatten().copied());
}

pub fn collect_nested(groups: &[Vec<u32>]) -> Vec<u32> {
    groups.iter().flatten().copied().collect()
}

pub fn collect_options(items: Vec<Option<u32>>) -> Vec<u32> {
    items.into_iter().flatten().collect()
}

pub fn collect_arrays(items: Vec<[u32; 3]>) -> Vec<u32> {
    items.into_iter().flatten().collect()
}

pub fn option_into_iter(value: Option<u32>) -> (Option<u32>, Option<u32>, usize, Option<usize>) {
    let mut iter = value.into_iter();
    let (lower, upper) = iter.size_hint();
    (iter.next(), iter.next(), lower, upper)
}

pub type Hint = (usize, Option<usize>);

/// The hint before the first next and after each of the next two.
pub fn nested_hints(groups: &[Vec<u32>]) -> (Hint, Hint, Hint) {
    let mut iter = groups.iter().flatten();
    let first = iter.size_hint();
    let _ = iter.next();
    let second = iter.size_hint();
    let _ = iter.next();
    (first, second, iter.size_hint())
}

pub fn optional_hints(group: Option<&BTreeSet<u32>>) -> (Hint, Hint, Hint) {
    let mut iter = group.into_iter().flatten();
    let first = iter.size_hint();
    let _ = iter.next();
    let second = iter.size_hint();
    let _ = iter.next();
    (first, second, iter.size_hint())
}

pub fn oneshot_hints(items: Vec<Option<u32>>) -> (Hint, Hint, Hint) {
    let mut iter = items.into_iter().flatten();
    let first = iter.size_hint();
    let _ = iter.next();
    let second = iter.size_hint();
    let _ = iter.next();
    (first, second, iter.size_hint())
}

pub fn array_hints(items: Vec<[u32; 3]>) -> (Hint, Hint, Hint) {
    let mut iter = items.into_iter().flatten();
    let first = iter.size_hint();
    let _ = iter.next();
    let second = iter.size_hint();
    let _ = iter.next();
    (first, second, iter.size_hint())
}

// The optimized paths bypass a panicking next: they distinguish real outer and
// inner fold/try_fold/count/advance_by dispatch from repeated next.
pub struct Dispatch {
    pub cursor: usize,
    pub end: usize,
    pub panic_at: usize,
}

impl Iterator for Dispatch {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.cursor == self.panic_at {
            panic!("next was not the selected iterator operation");
        }
        if self.cursor < self.end {
            let value = self.cursor;
            self.cursor += 1;
            Some(value)
        } else {
            None
        }
    }

    fn count(self) -> usize {
        100 + self.end - self.cursor
    }

    fn fold<B, G: FnMut(B, usize) -> B>(mut self, mut acc: B, mut fold: G) -> B {
        while self.cursor < self.end {
            let value = self.cursor;
            self.cursor += 1;
            acc = fold(acc, value);
        }
        acc
    }

    fn try_fold<B, G, R>(&mut self, mut acc: B, mut fold: G) -> R
    where
        G: FnMut(B, usize) -> R,
        R: std::ops::Try<Output = B>,
    {
        while self.cursor < self.end {
            let value = self.cursor;
            self.cursor += 1;
            acc = fold(acc, value)?;
        }
        R::from_output(acc)
    }

    fn advance_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        let remaining = self.end - self.cursor;
        let advanced = if n < remaining { n } else { remaining };
        self.cursor += advanced;
        match std::num::NonZero::new(n - advanced) {
            None => Ok(()),
            Some(remaining) => Err(remaining),
        }
    }
}

fn batches(ends: &[usize]) -> Vec<Dispatch> {
    ends.iter()
        .enumerate()
        .map(|(index, end)| Dispatch { cursor: index * 10, end: index * 10 + end, panic_at: 99 })
        .collect()
}

/// Inner count overrides, from a partially consumed front inner.
pub fn count_override(partial: bool) -> usize {
    let mut iter = vec![
        Dispatch { cursor: 0, end: 2, panic_at: 9 },
        Dispatch { cursor: 0, end: 0, panic_at: 9 },
        Dispatch { cursor: 5, end: 6, panic_at: 9 },
    ]
    .into_iter()
    .flatten();
    if partial {
        let _ = iter.next();
    }
    iter.count()
}

pub fn fold_dispatch() -> (usize, usize) {
    let mut folds = 0;
    let inners = vec![
        Dispatch { cursor: 0, end: 2, panic_at: 0 },
        Dispatch { cursor: 10, end: 12, panic_at: 10 },
    ];
    let result = inners.into_iter().flatten().fold(0, |signature, item| {
        folds += 1;
        signature * 100 + item
    });
    (result, folds)
}

pub fn try_fold_dispatch() -> (Result<usize, usize>, Option<usize>, usize) {
    let mut folds = 0;
    let (result, next) = {
        let inners = vec![
            Dispatch { cursor: 0, end: 2, panic_at: 0 },
            Dispatch { cursor: 10, end: 13, panic_at: 10 },
        ];
        let mut iter = inners.into_iter().flatten();
        let result = iter.try_fold(0, |sum, item| {
            folds += 1;
            if item == 10 { Err(sum) } else { Ok(sum + item) }
        });
        (result, iter.next())
    };
    (result, next, folds)
}

pub fn nth_dispatch() -> (Option<usize>, Option<usize>) {
    let mut iter = batches(&[3, 3]).into_iter().flatten();
    (iter.nth(4), iter.next())
}

pub fn advance_remainder(n: usize) -> (usize, Option<usize>) {
    let mut iter = batches(&[1, 2]).into_iter().flatten();
    let remaining = match iter.advance_by(n) {
        Ok(()) => 0,
        Err(remaining) => remaining.get(),
    };
    (remaining, iter.next())
}

pub fn any_resumes() -> (bool, Option<usize>, usize) {
    let mut calls = 0;
    let (found, next) = {
        let mut iter = batches(&[3, 2]).into_iter().flatten();
        let found = iter.any(|value| {
            calls += 1;
            value == 1
        });
        (found, iter.next())
    };
    (found, next, calls)
}

/// The outer is fused after its first None (Fuse), even if it would resume.
pub struct Pulse {
    pub step: usize,
}

impl Iterator for Pulse {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let step = self.step;
        self.step += 1;
        if step == 0 { Some(vec![1]) } else if step == 2 { Some(vec![2]) } else { None }
    }
}

pub fn fused_outer() -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut iter = Pulse { step: 0 }.flatten();
    (iter.next(), iter.next(), iter.next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(u32, &[u32])]) -> BTreeMap<u32, BTreeSet<u32>> {
        entries.iter().map(|(k, vs)| (*k, vs.iter().copied().collect())).collect()
    }

    #[test]
    fn collectors() {
        let successors = map(&[(1, &[3, 2, 3]), (2, &[])]);
        let mut pending = vec![9];
        extend_successors(&mut pending, &successors, 1);
        assert_eq!(pending, vec![9, 2, 3]);
        extend_successors(&mut pending, &successors, 2);
        extend_successors(&mut pending, &successors, 5);
        assert_eq!(pending, vec![9, 2, 3]);
        assert_eq!(collect_nested(&[vec![1, 2], vec![], vec![3]]), vec![1, 2, 3]);
        assert_eq!(collect_nested(&[]), Vec::<u32>::new());
        assert_eq!(collect_options(vec![None, Some(1), None, Some(3)]), vec![1, 3]);
        assert_eq!(collect_arrays(vec![[1, 2, 3], [4, 5, 6]]), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(option_into_iter(Some(7)), (Some(7), None, 1, Some(1)));
        assert_eq!(option_into_iter(None), (None, None, 0, Some(0)));
    }

    #[test]
    fn hints() {
        assert_eq!(nested_hints(&[vec![1, 2, 3]]), ((0, None), (2, Some(2)), (1, Some(1))));
        assert_eq!(nested_hints(&[vec![1, 2], vec![3]]), ((0, None), (1, None), (0, None)));
        assert_eq!(nested_hints(&[vec![], vec![1]]), ((0, None), (0, Some(0)), (0, Some(0))));
        assert_eq!(nested_hints(&[]), ((0, Some(0)), (0, Some(0)), (0, Some(0))));
        let set: BTreeSet<u32> = [4, 5].into_iter().collect();
        assert_eq!(optional_hints(Some(&set)), ((0, None), (1, Some(1)), (0, Some(0))));
        assert_eq!(optional_hints(None), ((0, Some(0)), (0, Some(0)), (0, Some(0))));
        assert_eq!(
            oneshot_hints(vec![Some(1), None, Some(3), None]),
            ((0, Some(4)), (0, Some(3)), (0, Some(1)))
        );
        assert_eq!(
            array_hints(vec![[1, 2, 3], [4, 5, 6]]),
            ((6, Some(6)), (5, Some(5)), (4, Some(4)))
        );
    }

    #[test]
    fn dispatch() {
        assert_eq!(count_override(false), 303);
        assert_eq!(count_override(true), 101 + 100 + 101);
        assert_eq!(fold_dispatch(), (11011, 4));
        assert_eq!(try_fold_dispatch(), (Err(1), Some(11), 3));
        assert_eq!(nth_dispatch(), (Some(11), Some(12)));
        assert_eq!(advance_remainder(2), (0, Some(11)));
        assert_eq!(advance_remainder(5), (2, None));
        assert_eq!(any_resumes(), (true, Some(2), 2));
        assert_eq!(fused_outer(), (Some(1), None, None));
    }
}
