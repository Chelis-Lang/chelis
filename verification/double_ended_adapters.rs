#![feature(iter_advance_by, try_trait_v2)]

// DoubleEndedIterator of Enumerate, Skip, Map and Filter (core enumerate.rs,
// skip.rs, map.rs, filter.rs) and Index<&Q> for BTreeMap (alloc btree/map.rs):
// these native results pin the back indices (count + remaining len), the
// length bound of Skip's back, Filter's back as the inner rfind, the back
// folds' dispatch to the inner iterator's overrides, and the panic on a
// missing key.

use std::collections::BTreeMap;

/// chelis-ir ownership/last_use.rs verifier_plan_from: the last index after
/// `start` whose item satisfies the predicate, plus one.
pub fn last_matching(ops: &[u32], start: usize, owner: u32) -> Option<usize> {
    ops.iter()
        .enumerate()
        .skip(start)
        .filter(|(_, op)| **op == owner)
        .map(|(index, _)| index + 1)
        .next_back()
}

pub fn enumerate_back(xs: &[u32]) -> (Option<(usize, u32)>, Option<(usize, u32)>, Option<(usize, u32)>) {
    let mut iter = xs.iter().enumerate();
    let first = iter.next().map(|(i, x)| (i, *x));
    let last = iter.next_back().map(|(i, x)| (i, *x));
    (first, last, iter.nth_back(1).map(|(i, x)| (i, *x)))
}

pub fn enumerate_rfold(xs: &[u32]) -> usize {
    let mut iter = xs.iter().enumerate();
    let _ = iter.next();
    iter.rfold(0, |acc, (index, x)| acc * 10 + index + *x as usize)
}

pub fn enumerate_rfind(xs: &[u32], target: u32) -> Option<usize> {
    xs.iter().enumerate().rfind(|(_, x)| **x == target).map(|(index, _)| index)
}

pub fn enumerate_advance_back(xs: &[u32], n: usize) -> (usize, Option<(usize, u32)>) {
    let mut iter = xs.iter().enumerate();
    let remaining = match iter.advance_back_by(n) {
        Ok(()) => 0,
        Err(remaining) => remaining.get(),
    };
    (remaining, iter.next_back().map(|(i, x)| (i, *x)))
}

pub fn skip_back(xs: &[u32], n: usize) -> (Option<u32>, Option<u32>, usize) {
    let mut iter = xs.iter().skip(n);
    let last = iter.next_back().copied();
    let before = iter.next_back().copied();
    (last, before, iter.len())
}

pub fn skip_nth_back(xs: &[u32], n: usize, k: usize) -> (Option<u32>, Option<u32>) {
    let mut iter = xs.iter().skip(n);
    let item = iter.nth_back(k).copied();
    (item, iter.next().copied())
}

pub fn skip_rfold(xs: &[u32], n: usize) -> u32 {
    xs.iter().skip(n).rfold(0, |acc, x| acc * 10 + *x)
}

pub fn skip_rfind(xs: &[u32], n: usize, target: u32) -> (Option<u32>, Option<u32>) {
    let mut iter = xs.iter().skip(n);
    let found = iter.rfind(|x| **x <= target).copied();
    (found, iter.next_back().copied())
}

pub fn skip_advance_back(xs: &[u32], n: usize, k: usize) -> (usize, Option<u32>, Option<u32>) {
    let mut iter = xs.iter().skip(n);
    let remaining = match iter.advance_back_by(k) {
        Ok(()) => 0,
        Err(remaining) => remaining.get(),
    };
    (remaining, iter.next_back().copied(), iter.next().copied())
}

pub fn skip_enumerate_back(xs: &[u32], n: usize) -> Vec<(usize, u32)> {
    let mut out = Vec::new();
    let mut iter = xs.iter().enumerate().skip(n);
    while let Some((index, x)) = iter.next_back() {
        out.push((index, *x));
    }
    out
}

pub fn filter_map_rfold(xs: &[u32]) -> u32 {
    xs.iter().filter(|x| **x % 2 == 1).map(|x| *x * 10).rfold(0, |acc, x| acc * 100 + x)
}

pub fn map_rfind(xs: &[u32]) -> (Option<u32>, Option<u32>) {
    let mut iter = xs.iter().map(|x| *x + 1);
    let found = iter.rfind(|x| *x % 3 == 0);
    (found, iter.next_back())
}

pub fn filter_nth_back(xs: &[u32], k: usize) -> (Option<u32>, Option<u32>) {
    let mut iter = xs.iter().filter(|x| **x > 1);
    let item = iter.nth_back(k).copied();
    (item, iter.next().copied())
}

pub fn filter_try_rfold(xs: &[u32]) -> Result<u32, u32> {
    xs.iter()
        .filter(|x| **x != 0)
        .try_rfold(0, |acc, x| if *x > 5 { Err(acc) } else { Ok(acc * 10 + *x) })
}

/// A double-ended, exact-size iterator whose next_back panics at `panic_at`,
/// while its try_rfold/rfold/nth_back/advance_back_by overrides do not: the
/// back folds of Skip and Enumerate must dispatch to them.
pub struct Dispatch {
    pub front: usize,
    pub back: usize,
    pub panic_at: usize,
}

impl Iterator for Dispatch {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.front < self.back {
            let value = self.front;
            self.front += 1;
            Some(value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl ExactSizeIterator for Dispatch {}

impl DoubleEndedIterator for Dispatch {
    fn next_back(&mut self) -> Option<usize> {
        if self.back == self.panic_at {
            panic!("next_back was not the selected operation");
        }
        if self.front < self.back {
            self.back -= 1;
            Some(self.back)
        } else {
            None
        }
    }

    fn try_rfold<B, G, R>(&mut self, mut acc: B, mut fold: G) -> R
    where
        G: FnMut(B, usize) -> R,
        R: std::ops::Try<Output = B>,
    {
        while self.front < self.back {
            self.back -= 1;
            acc = fold(acc, self.back)?;
        }
        R::from_output(acc)
    }

    fn rfold<B, G: FnMut(B, usize) -> B>(mut self, mut acc: B, mut fold: G) -> B {
        while self.front < self.back {
            self.back -= 1;
            acc = fold(acc, self.back);
        }
        acc
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), std::num::NonZero<usize>> {
        let len = self.back - self.front;
        let step = if n < len { n } else { len };
        self.back -= step;
        match std::num::NonZero::new(n - step) {
            None => Ok(()),
            Some(rest) => Err(rest),
        }
    }
}

pub fn skip_try_rfold_dispatch(n: usize) -> (Result<usize, usize>, Option<usize>) {
    let mut iter = Dispatch { front: 0, back: 5, panic_at: 99 }.skip(n);
    let result = iter.try_rfold(0, |sum, x| if x == 1 { Err(sum) } else { Ok(sum * 10 + x) });
    (result, iter.next())
}

pub fn skip_rfold_dispatch(n: usize) -> usize {
    Dispatch { front: 0, back: 4, panic_at: 4 }.skip(n).rfold(0, |acc, x| acc * 10 + x)
}

pub fn enumerate_rfold_dispatch() -> usize {
    let mut iter = Dispatch { front: 0, back: 4, panic_at: 3 }.enumerate();
    let _ = iter.next();
    iter.rfold(0, |acc, (index, x)| acc * 100 + index * 10 + x)
}

pub fn enumerate_advance_back_dispatch(n: usize) -> (usize, Option<(usize, usize)>) {
    let mut iter = Dispatch { front: 0, back: 4, panic_at: 99 }.enumerate();
    let remaining = match iter.advance_back_by(n) {
        Ok(()) => 0,
        Err(remaining) => remaining.get(),
    };
    (remaining, iter.next_back())
}

pub fn filter_back_dispatch() -> (Option<usize>, Option<usize>) {
    let mut iter = Dispatch { front: 0, back: 4, panic_at: 4 }.filter(|x| *x % 2 == 0);
    let last = iter.next_back();
    (last, iter.next())
}

pub fn map_rfold_dispatch() -> usize {
    Dispatch { front: 0, back: 4, panic_at: 4 }.map(|x| x * 3).rfold(0, |acc, x| acc * 100 + x)
}

pub fn lookup(map: &BTreeMap<u32, u32>, key: u32) -> u32 {
    map[&key]
}

pub fn lookup_str(map: &BTreeMap<String, u32>, key: &str) -> u32 {
    map[key]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_and_skip() {
        assert_eq!(last_matching(&[7, 1, 7, 2, 7], 0, 7), Some(5));
        assert_eq!(last_matching(&[7, 1, 7, 2, 1], 1, 7), Some(3));
        assert_eq!(last_matching(&[7, 1, 7], 3, 7), None);
        assert_eq!(last_matching(&[7, 1, 7], 9, 7), None);
        assert_eq!(enumerate_back(&[5, 6, 7, 8]), (Some((0, 5)), Some((3, 8)), Some((1, 6))));
        assert_eq!(enumerate_back(&[5]), (Some((0, 5)), None, None));
        assert_eq!(enumerate_rfold(&[1, 2, 3]), 53);
        assert_eq!(enumerate_rfind(&[4, 9, 4, 9], 4), Some(2));
        assert_eq!(enumerate_rfind(&[4, 9], 3), None);
        assert_eq!(enumerate_advance_back(&[1, 2, 3], 1), (0, Some((1, 2))));
        assert_eq!(enumerate_advance_back(&[1, 2, 3], 5), (2, None));
        assert_eq!(skip_back(&[1, 2, 3, 4], 2), (Some(4), Some(3), 0));
        assert_eq!(skip_back(&[1, 2, 3, 4], 3), (Some(4), None, 0));
        assert_eq!(skip_back(&[1, 2, 3], 5), (None, None, 0));
        assert_eq!(skip_nth_back(&[1, 2, 3, 4, 5], 1, 1), (Some(4), Some(2)));
        assert_eq!(skip_nth_back(&[1, 2, 3, 4, 5], 1, 4), (None, None));
        assert_eq!(skip_nth_back(&[1, 2, 3], 3, 0), (None, None));
        assert_eq!(skip_rfold(&[1, 2, 3, 4], 1), 432);
        assert_eq!(skip_rfold(&[1, 2], 2), 0);
        assert_eq!(skip_rfind(&[1, 5, 2, 6], 1, 3), (Some(2), Some(5)));
        assert_eq!(skip_rfind(&[1, 5, 2, 6], 1, 1), (None, None));
        assert_eq!(skip_advance_back(&[1, 2, 3, 4], 1, 2), (0, Some(2), None));
        assert_eq!(skip_advance_back(&[1, 2, 3, 4], 1, 5), (2, None, None));
        assert_eq!(skip_enumerate_back(&[10, 11, 12], 1), vec![(2, 12), (1, 11)]);
        assert_eq!(filter_map_rfold(&[1, 2, 3, 5]), 503010);
        assert_eq!(filter_map_rfold(&[2, 4]), 0);
        assert_eq!(map_rfind(&[1, 2, 5, 7]), (Some(6), Some(3)));
        assert_eq!(map_rfind(&[1, 3]), (None, None));
        assert_eq!(filter_nth_back(&[1, 2, 3, 1, 4], 1), (Some(3), Some(2)));
        assert_eq!(filter_nth_back(&[1, 2, 3], 2), (None, None));
        assert_eq!(filter_try_rfold(&[1, 0, 2, 3]), Ok(321));
        assert_eq!(filter_try_rfold(&[1, 9, 0, 2]), Err(2));
    }

    #[test]
    fn dispatch() {
        assert_eq!(skip_try_rfold_dispatch(0), (Err(432), Some(0)));
        assert_eq!(skip_try_rfold_dispatch(2), (Ok(432), None));
        assert_eq!(skip_try_rfold_dispatch(5), (Ok(0), None));
        assert_eq!(skip_rfold_dispatch(1), 321);
        assert_eq!(enumerate_rfold_dispatch(), 332211);
        assert_eq!(enumerate_advance_back_dispatch(1), (0, Some((2, 2))));
        assert_eq!(enumerate_advance_back_dispatch(6), (2, None));
        assert_eq!(filter_back_dispatch(), (Some(2), Some(0)));
        assert_eq!(map_rfold_dispatch(), 9060300);
    }

    #[test]
    fn index() {
        let map: BTreeMap<u32, u32> = [(1, 10), (3, 30)].into_iter().collect();
        assert_eq!(lookup(&map, 3), 30);
        assert!(std::panic::catch_unwind(|| lookup(&map, 2)).is_err());
        let names: BTreeMap<String, u32> = [("b".to_string(), 2), ("a".to_string(), 1)].into_iter().collect();
        assert_eq!(lookup_str(&names, "b"), 2);
        assert!(std::panic::catch_unwind(|| lookup_str(&names, "c")).is_err());
    }
}
