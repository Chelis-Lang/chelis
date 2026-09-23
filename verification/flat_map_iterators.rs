#![feature(iter_advance_by, try_trait_v2)]

// A mapper borrowing its counter must return the final counter through the
// flat_map constructor's restoration lens, including advances over empty inners.
pub fn empty_inner_resume() -> (Option<usize>, Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let (first, second, third) = {
        let mut iter = (0usize..4).flat_map(|index| {
            calls += 1;
            let start = calls * 10;
            let len = if index == 1 { 2 } else if index == 3 { 1 } else { 0 };
            start..start + len
        });
        (iter.next(), iter.next(), iter.next())
    };
    (first, second, third, calls)
}

pub fn any_resume() -> (bool, Option<usize>, usize, usize) {
    let mut mappings = 0;
    let mut predicates = 0;
    let (found, next) = {
        let mut iter = (0..3).flat_map(|index| {
            mappings += 1;
            index * 10..index * 10 + 3
        });
        let _ = iter.next();
        let found = iter.any(|value| {
            predicates += 1;
            value == 11
        });
        (found, iter.next())
    };
    (found, next, mappings, predicates)
}

pub struct Inner {
    pub cursor: usize,
    pub end: usize,
    pub fail_next: bool,
    pub fail_count: bool,
}

impl Iterator for Inner {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.fail_next {
            panic!("inner next failed");
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
        if self.fail_count {
            panic!("inner count failed");
        }
        100 + self.end - self.cursor
    }

    fn any<F: FnMut(usize) -> bool>(&mut self, _predicate: F) -> bool {
        panic!("FlatMap must not select the inner any override");
    }
}

pub struct Batch {
    pub len: usize,
    pub fail_convert: bool,
    pub fail_next: bool,
    pub fail_count: bool,
}

impl IntoIterator for Batch {
    type Item = usize;
    type IntoIter = Inner;

    fn into_iter(self) -> Inner {
        if self.fail_convert {
            panic!("conversion failed");
        }
        Inner {
            cursor: 0,
            end: self.len,
            fail_next: self.fail_next,
            fail_count: self.fail_count,
        }
    }
}

pub fn count_override(partial: bool) -> (usize, usize) {
    let mut calls = 0;
    let count = {
        let mut iter = (0..3).flat_map(|index| {
            calls += 1;
            Batch { len: index, fail_convert: false, fail_next: false, fail_count: false }
        });
        if partial {
            let _ = iter.next();
        }
        iter.count()
    };
    (count, calls)
}

pub fn any_ignores_inner_override() -> (bool, Option<usize>) {
    let mut iter = (0usize..1).flat_map(|_| Batch {
        len: 3, fail_convert: false, fail_next: false, fail_count: false,
    });
    let found = iter.any(|value| value == 1);
    (found, iter.next())
}

// Stage 1 fails in the mapper, 2 in IntoIterator, 3 in next, and 4 in count.
// The first inner is always safe. No later stage may run before it is needed.
pub fn failure_boundary(stage: usize, demand: usize) -> Option<usize> {
    let mut iter = (0usize..2).flat_map(|index| {
        if index == 1 && stage == 1 {
            panic!("mapper failed");
        }
        Batch {
            len: 1,
            fail_convert: index == 1 && stage == 2,
            fail_next: index == 1 && stage == 3,
            fail_count: index == 1 && stage == 4,
        }
    });
    if demand == 0 {
        return None;
    }
    let first = iter.next();
    if demand == 1 {
        return first;
    }
    if demand == 2 {
        return iter.next();
    }
    Some(iter.count())
}

pub struct Pulse {
    pub step: usize,
}

impl Iterator for Pulse {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let step = self.step;
        self.step += 1;
        if step == 0 { Some(1) } else if step == 2 { Some(2) } else { None }
    }
}

pub fn fused_outer() -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut iter = Pulse { step: 0 }.flat_map(|end| 0..end);
    (iter.next(), iter.next(), iter.next())
}

pub fn discard_exhausted_inner() -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut iter = (0usize..2).flat_map(|_| Pulse { step: 0 });
    (iter.next(), iter.next(), iter.next())
}

// The optimized paths deliberately bypass a panicking next. This distinguishes
// real outer/inner fold and indexed dispatch from an observationally wrong
// implementation that repeatedly calls next.
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

pub fn fold_dispatch() -> (usize, usize, usize) {
    let mut mappings = 0;
    let mut folds = 0;
    let result = Dispatch { cursor: 0, end: 2, panic_at: 0 }
        .flat_map(|index| {
            mappings += 1;
            Dispatch { cursor: index * 10, end: index * 10 + 2, panic_at: index * 10 }
        })
        .fold(0, |signature, item| {
            folds += 1;
            signature * 100 + item
        });
    (result, mappings, folds)
}

pub fn try_fold_dispatch() -> (Result<usize, usize>, Option<usize>, usize, usize) {
    let mut mappings = 0;
    let mut folds = 0;
    let (result, next) = {
        let mut iter = Dispatch { cursor: 0, end: 2, panic_at: 0 }.flat_map(|index| {
            mappings += 1;
            Dispatch { cursor: index * 10, end: index * 10 + 2, panic_at: index * 10 }
        });
        let result = iter.try_fold(0, |sum, item| {
            folds += 1;
            if item == 10 { Err(sum) } else { Ok(sum + item) }
        });
        (result, iter.next())
    };
    (result, next, mappings, folds)
}

pub fn nth_dispatch() -> (Option<usize>, Option<usize>, usize) {
    let mut mappings = 0;
    let (selected, next) = {
        let mut iter = Dispatch { cursor: 0, end: 2, panic_at: 0 }.flat_map(|index| {
            mappings += 1;
            Dispatch { cursor: index * 10, end: index * 10 + 3, panic_at: index * 10 }
        });
        (iter.nth(1), iter.next())
    };
    (selected, next, mappings)
}

pub fn zero_advance_prefetch() -> (bool, usize) {
    let mut mappings = 0;
    let success = {
        let mut iter = (0usize..2).flat_map(|index| {
            mappings += 1;
            Dispatch { cursor: 0, end: index, panic_at: 0 }
        });
        match iter.advance_by(0) {
            Ok(()) => true,
            Err(_) => false,
        }
    };
    (success, mappings)
}

pub fn oneshot_zero_advance() -> (bool, usize) {
    let mut mappings = 0;
    let success = {
        let mut iter = (0usize..2).flat_map(|index| {
            mappings += 1;
            std::iter::once(index)
        });
        match iter.advance_by(0) {
            Ok(()) => true,
            Err(_) => false,
        }
    };
    (success, mappings)
}

pub fn advance_remainder(n: usize) -> (usize, Option<usize>, usize) {
    let mut mappings = 0;
    let (remaining, next) = {
        let mut iter = (0usize..2).flat_map(|index| {
            mappings += 1;
            Dispatch { cursor: 0, end: index + 1, panic_at: 9 }
        });
        let remaining = match iter.advance_by(n) {
            Ok(()) => 0,
            Err(remaining) => remaining.get(),
        };
        (remaining, iter.next())
    };
    (remaining, next, mappings)
}

pub fn trusted_array_zip() -> (bool, usize) {
    let mut mappings = 0;
    let exhausted = {
        let left = (0usize..2).flat_map(|index| {
            mappings += 1;
            [index, index + 10]
        });
        left.zip([0usize; 0]).count() == 0
    };
    (exhausted, mappings)
}

pub fn array_size_hint() -> (usize, Option<usize>, usize, Option<usize>) {
    let mut iter = (0usize..2).flat_map(|index| [index, index + 10]);
    let (before_lo, before_hi) = iter.size_hint();
    let _ = iter.next();
    let (after_lo, after_hi) = iter.size_hint();
    (before_lo, before_hi, after_lo, after_hi)
}

pub fn oneshot_size_hint() -> (usize, Option<usize>, usize, Option<usize>) {
    let mut iter = (0usize..2).flat_map(|index| [index]);
    let (before_lo, before_hi) = iter.size_hint();
    let _ = iter.next();
    let (after_lo, after_hi) = iter.size_hint();
    (before_lo, before_hi, after_lo, after_hi)
}

pub struct FusedProbe<'a> {
    pub polls: &'a mut usize,
}

impl Iterator for FusedProbe<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        *self.polls += 1;
        None
    }
}

impl std::iter::FusedIterator for FusedProbe<'_> {}

pub fn custom_fused_terminal() -> usize {
    let mut polls = 0;
    {
        let mut iter = FusedProbe { polls: &mut polls }.flat_map(|end| 0..end);
        let _ = iter.next();
        let _ = iter.next();
    }
    polls
}

pub struct FusedWrapper<I>(pub I);

impl<I: Iterator> Iterator for FusedWrapper<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        self.0.next()
    }
}

impl<I: std::iter::FusedIterator> std::iter::FusedIterator for FusedWrapper<I> {}

pub fn conditional_fused_terminal() -> usize {
    let mut polls = 0;
    {
        let outer = FusedWrapper(FusedProbe { polls: &mut polls });
        let mut iter = outer.flat_map(|end| 0..end);
        let _ = iter.next();
        let _ = iter.next();
    }
    polls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_and_resumption() {
        assert_eq!(empty_inner_resume(), (Some(20), Some(21), Some(40), 4));
        assert_eq!(any_resume(), (true, Some(12), 2, 4));
        assert_eq!(fused_outer(), (Some(0), None, None));
        assert_eq!(discard_exhausted_inner(), (Some(1), Some(1), None));
    }

    #[test]
    fn override_dispatch() {
        assert_eq!(count_override(false), (303, 3));
        assert_eq!(count_override(true), (202, 3));
        assert_eq!(any_ignores_inner_override(), (true, Some(2)));
        // Count selects the custom count and never invokes the failing next.
        assert_eq!(failure_boundary(3, 3), Some(201));
    }

    #[test]
    fn laziness_and_failures() {
        for stage in 1..5 {
            assert_eq!(failure_boundary(stage, 0), None);
            assert_eq!(failure_boundary(stage, 1), Some(0));
        }
        for stage in 1..4 {
            assert!(std::panic::catch_unwind(|| failure_boundary(stage, 2)).is_err());
        }
        for stage in [1, 2, 4] {
            assert!(std::panic::catch_unwind(|| failure_boundary(stage, 3)).is_err());
        }
        assert_eq!(failure_boundary(4, 2), Some(0));
    }

    #[test]
    fn optimized_dispatch_and_specialization() {
        assert_eq!(fold_dispatch(), (11011, 2, 4));
        assert_eq!(try_fold_dispatch(), (Err(1), Some(11), 2, 3));
        assert_eq!(nth_dispatch(), (Some(1), Some(2), 1));
        assert_eq!(zero_advance_prefetch(), (true, 1));
        assert_eq!(oneshot_zero_advance(), (true, 0));
        assert_eq!(advance_remainder(2), (0, Some(1), 2));
        assert_eq!(advance_remainder(5), (2, None, 2));
        assert_eq!(trusted_array_zip(), (true, 0));
        assert_eq!(array_size_hint(), (4, Some(4), 3, Some(3)));
        assert_eq!(oneshot_size_hint(), (2, Some(2), 1, Some(1)));
        assert_eq!(custom_fused_terminal(), 2);
        assert_eq!(conditional_fused_terminal(), 2);
    }
}
