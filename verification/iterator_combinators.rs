// A non-fused iterator with observable terminal calls and a conforming nth
// override. Count's deliberate bias distinguishes override dispatch from fold.
pub struct Probe {
    pub position: usize,
    pub gap: usize,
    pub end: usize,
    pub count_bias: usize,
    pub nth_calls: usize,
    pub nth_last: usize,
}

impl Probe {
    pub fn new(gap: usize, end: usize, count_bias: usize) -> Self {
        Self { position: 0, gap, end, count_bias, nth_calls: 0, nth_last: 0 }
    }
}

impl Iterator for Probe {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let position = self.position;
        self.position += 1;
        if position == self.gap || position >= self.end {
            None
        } else {
            Some(position)
        }
    }

    fn nth(&mut self, n: usize) -> Option<usize> {
        self.nth_calls += 1;
        self.nth_last = n;
        let mut remaining = n;
        while remaining != 0 {
            match self.next() {
                None => return None,
                Some(_) => remaining -= 1,
            }
        }
        self.next()
    }

    fn count(mut self) -> usize {
        let mut count = self.count_bias;
        while self.next().is_some() {
            count += 1;
        }
        count
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let end = if self.position <= self.gap && self.gap < self.end {
            self.gap
        } else {
            self.end
        };
        let remaining = end.saturating_sub(self.position);
        (remaining, Some(remaining))
    }
}

pub fn chain_nonfused() -> (Option<usize>, Option<usize>, Option<usize>, Option<usize>, usize, usize) {
    let mut a = Probe::new(0, 4, 100);
    let mut b = Probe::new(1, 4, 200);
    let (first, second, third, fourth) = {
        let mut chain = (&mut a).chain(&mut b);
        (chain.next(), chain.next(), chain.next(), chain.next())
    };
    (first, second, third, fourth, a.position, b.position)
}

pub fn chain_any_resume() -> (bool, usize, Option<usize>, usize, usize) {
    let mut a = Probe::new(1, 4, 100);
    let mut b = Probe::new(99, 3, 200);
    let mut calls = 0;
    let (found, next) = {
        let mut chain = (&mut a).chain(&mut b);
        let found = chain.any(|value| { calls += 1; value == 1 });
        (found, chain.next())
    };
    (found, calls, next, a.position, b.position)
}

pub fn chain_counts() -> (usize, usize) {
    let full = Probe::new(99, 2, 100).chain(Probe::new(99, 3, 200)).count();
    let mut fused = Probe::new(0, 2, 100).chain(Probe::new(99, 3, 200));
    let _ = fused.next();
    (full, fused.count())
}

pub fn chain_nth_crosses_left() -> (Option<usize>, Option<usize>, usize, usize, usize, usize) {
    let mut a = Probe::new(99, 2, 0);
    let mut b = Probe::new(99, 4, 0);
    let (item, next) = {
        let mut chain = (&mut a).chain(&mut b);
        (chain.nth(3), chain.next())
    };
    (item, next, a.position, b.position, b.nth_calls, b.nth_last)
}

pub fn zip_partial_consumption() -> (Option<(usize, usize)>, Option<(usize, usize)>, Option<usize>, usize, usize) {
    let mut a = Probe::new(99, 3, 100);
    let mut b = Probe::new(99, 1, 200);
    let (first, second) = {
        let mut zip = (&mut a).zip(&mut b);
        (zip.next(), zip.next())
    };
    let next = a.next();
    (first, second, next, a.position, b.position)
}

pub fn zip_left_before_right() -> (Option<(usize, usize)>, Option<(usize, usize)>, usize, usize) {
    let mut a = Probe::new(0, 3, 100);
    let mut b = Probe::new(99, 3, 200);
    let (first, second) = {
        let mut zip = (&mut a).zip(&mut b);
        (zip.next(), zip.next())
    };
    (first, second, a.position, b.position)
}

pub fn zip_count() -> usize {
    Probe::new(99, 3, 100).zip(Probe::new(99, 1, 200)).count()
}

pub fn zip_any_resume() -> (bool, usize, Option<(usize, usize)>, usize, usize) {
    let mut a = Probe::new(99, 4, 0);
    let mut b = Probe::new(99, 4, 0);
    let mut calls = 0;
    let (found, next) = {
        let mut zip = (&mut a).zip(&mut b);
        let found = zip.any(|(x, y)| { calls += 1; x + y == 2 });
        (found, zip.next())
    };
    (found, calls, next, a.position, b.position)
}

pub fn skip_nonfused() -> (Option<usize>, Option<usize>, usize, usize, usize) {
    let mut source = Probe::new(1, 5, 100);
    let (first, second) = {
        let mut skip = (&mut source).skip(3);
        (skip.next(), skip.next())
    };
    (first, second, source.position, source.nth_calls, source.nth_last)
}

pub fn skip_any_resume() -> (bool, usize, Option<usize>, usize, usize, usize) {
    let mut source = Probe::new(99, 6, 100);
    let mut calls = 0;
    let (found, next) = {
        let mut skip = (&mut source).skip(2);
        let found = skip.any(|value| { calls += 1; value == 3 });
        (found, skip.next())
    };
    (found, calls, next, source.position, source.nth_calls, source.nth_last)
}

pub fn skip_any_gap() -> (bool, usize, Option<usize>, usize, usize) {
    let mut source = Probe::new(1, 5, 100);
    let mut calls = 0;
    let (found, next) = {
        let mut skip = (&mut source).skip(3);
        let found = skip.any(|_| { calls += 1; true });
        (found, skip.next())
    };
    (found, calls, next, source.nth_calls, source.nth_last)
}

pub fn skip_counts() -> (usize, usize, usize) {
    let regular = Probe::new(99, 5, 100).skip(2).count();
    let early_gap = Probe::new(1, 5, 100).skip(3).count();
    let no_skip = Probe::new(99, 5, 100).skip(0).count();
    (regular, early_gap, no_skip)
}

pub fn skip_nth_overflow() -> (Option<usize>, usize, usize, usize) {
    let mut source = Probe::new(0, 3, 0);
    let item = (&mut source).skip(usize::MAX).nth(1);
    (item, source.position, source.nth_calls, source.nth_last)
}

pub fn chain_skip_try_fold() -> (bool, Option<usize>, usize, usize) {
    let mut source = Probe::new(99, 6, 0);
    let (found, next) = {
        let left = (&mut source).skip(2);
        let mut chain = left.chain(Probe::new(99, 0, 0));
        let found = chain.any(|value| value == 3);
        (found, chain.next())
    };
    (found, next, source.nth_calls, source.nth_last)
}

pub fn zip_trusted_count() -> (usize, usize, usize) {
    let mut left_calls = 0;
    let mut right_calls = 0;
    let count = (0usize..4).map(|x| { left_calls += 1; x })
        .zip((0usize..2).map(|x| { right_calls += 1; x })).count();
    (count, left_calls, right_calls)
}

pub fn zip_exact_untrusted_count() -> (usize, usize, usize) {
    let mut a = Probe::new(99, 4, 0);
    let mut b = Probe::new(99, 2, 0);
    let count = (&mut a).zip(&mut b).count();
    (count, a.position, b.position)
}

pub fn zip_random_access_resume() -> (Option<(usize, usize)>, Option<(usize, usize)>, Option<(usize, usize)>, usize, usize) {
    let mut left_calls = 0;
    let mut right_calls = 0;
    let (first, third, end) = {
        let left = (0usize..5).map(|x| { left_calls += 1; x });
        let right = (10usize..13).map(|x| { right_calls += 1; x });
        let mut zip = left.zip(right);
        (zip.next(), zip.nth(1), zip.next())
    };
    (first, third, end, left_calls, right_calls)
}

pub fn zip_skip_random_access() -> (Option<(usize, usize)>, Option<(usize, usize)>, Option<(usize, usize)>, usize, usize) {
    let mut left_calls = 0;
    let mut right_calls = 0;
    let (first, second, end) = {
        let left = (0usize..6).map(|x| { left_calls += 1; x }).skip(2);
        let right = (0usize..2).map(|x| { right_calls += 1; x });
        let mut zip = left.zip(right);
        (zip.next(), zip.next(), zip.next())
    };
    (first, second, end, left_calls, right_calls)
}

pub fn zip_count_after_next() -> (usize, usize, usize) {
    let mut left_calls = 0;
    let mut right_calls = 0;
    let count = {
        let left = (0usize..5).map(|x| { left_calls += 1; x });
        let right = (0usize..3).map(|x| { right_calls += 1; x });
        let mut zip = left.zip(right);
        let _ = zip.next();
        zip.count()
    };
    (count, left_calls, right_calls)
}

pub fn zip_borrowed_trusted_count() -> (usize, Option<usize>, usize, usize) {
    let mut left_calls = 0;
    let mut right_calls = 0;
    let (count, next) = {
        let mut left = (0usize..4).map(|x| { left_calls += 1; x });
        let mut right = (0usize..2).map(|x| { right_calls += 1; x });
        let count = (&mut left).zip(&mut right).count();
        (count, left.next())
    };
    (count, next, left_calls, right_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_fuses_only_left_and_restores_both_receivers() {
        assert_eq!(chain_nonfused(), (Some(0), None, Some(2), Some(3), 1, 4));
        assert_eq!(chain_any_resume(), (true, 3, Some(2), 2, 3));
        assert_eq!(chain_counts(), (305, 202));
        assert_eq!(chain_nth_crosses_left(), (Some(1), Some(2), 3, 3, 1, 1));
        assert_eq!(chain_skip_try_fold(), (true, Some(4), 1, 1));
    }

    #[test]
    fn zip_observes_order_partial_consumption_and_short_circuit() {
        assert_eq!(zip_partial_consumption(), (Some((0, 0)), None, Some(2), 3, 2));
        assert_eq!(zip_left_before_right(), (None, Some((1, 0)), 2, 1));
        assert_eq!(zip_count(), 1);
        assert_eq!(zip_any_resume(), (true, 2, Some((2, 2)), 3, 3));
        assert_eq!(zip_trusted_count(), (2, 2, 2));
        assert_eq!(zip_exact_untrusted_count(), (2, 3, 3));
        assert_eq!(zip_random_access_resume(), (Some((0, 10)), Some((2, 12)), None, 3, 3));
        assert_eq!(zip_skip_random_access(), (Some((2, 0)), Some((3, 1)), None, 4, 2));
        assert_eq!(zip_count_after_next(), (2, 3, 3));
        assert_eq!(zip_borrowed_trusted_count(), (2, Some(2), 3, 2));
    }

    #[test]
    fn skip_dispatches_nth_once_even_when_skipping_hits_none() {
        assert_eq!(skip_nonfused(), (None, Some(2), 3, 1, 3));
        assert_eq!(skip_any_resume(), (true, 2, Some(4), 5, 1, 1));
        assert_eq!(skip_any_gap(), (false, 0, Some(2), 1, 2));
        assert_eq!(skip_counts(), (103, 0, 105));
        assert_eq!(skip_nth_overflow(), (None, 1, 1, usize::MAX - 1));
    }
}
