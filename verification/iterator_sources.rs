#![feature(iter_advance_by)]

pub fn inclusive_endpoint(start: usize, end: usize) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut range = start..=end;
    (range.next(), range.next(), range.next())
}

pub fn inclusive_partial_count(start: usize, end: usize) -> (Option<usize>, usize) {
    let mut range = start..=end;
    let first = range.next();
    (first, range.count())
}

pub fn inclusive_any_resume(start: usize, end: usize, target: usize) -> (bool, Option<usize>, usize) {
    let mut range = start..=end;
    let mut calls = 0;
    let found = range.any(|value| {
        calls += 1;
        value == target
    });
    (found, range.next(), calls)
}

pub fn inclusive_max_singleton() -> (bool, Option<usize>, usize) {
    let mut range = usize::MAX..=usize::MAX;
    let exact = range.next() == Some(usize::MAX);
    let terminal = range.next();
    (exact, terminal, range.count())
}

pub fn inclusive_max_any() -> (bool, Option<usize>, usize) {
    inclusive_any_resume(usize::MAX, usize::MAX, usize::MAX)
}

pub fn inclusive_count_overflow() -> usize {
    (0..=usize::MAX).count()
}

pub fn inclusive_nth_resume(start: usize, end: usize, index: usize) -> (Option<usize>, Option<usize>) {
    let mut range = start..=end;
    (range.nth(index), range.next())
}

pub fn inclusive_predicate_failure() -> bool {
    (3..=3).any(|_| panic!("inclusive predicate failed"))
}

pub fn once_resume(value: usize) -> (Option<usize>, Option<usize>, usize) {
    let mut iter = std::iter::once(value);
    let first = iter.next();
    let second = iter.next();
    (first, second, iter.count())
}

pub fn once_count(value: usize) -> usize {
    std::iter::once(value).count()
}

pub fn once_any_resume(value: usize, target: usize) -> (bool, Option<usize>, usize) {
    let mut iter = std::iter::once(value);
    let mut calls = 0;
    let found = iter.any(|item| {
        calls += 1;
        item == target
    });
    (found, iter.next(), calls)
}

pub fn once_exhausted_any(value: usize) -> (bool, usize) {
    let mut iter = std::iter::once(value);
    let _ = iter.next();
    let mut calls = 0;
    let found = iter.any(|_| {
        calls += 1;
        true
    });
    (found, calls)
}

pub fn once_nth_resume(value: usize, index: usize) -> (Option<usize>, Option<usize>) {
    let mut iter = std::iter::once(value);
    (iter.nth(index), iter.next())
}

pub fn once_predicate_failure() -> bool {
    std::iter::once(3usize).any(|_| panic!("once predicate failed"))
}

pub fn inclusive_fold(start: usize, end: usize) -> (usize, usize) {
    let mut calls = 0;
    let count = (start..=end).fold(10, |count, _| {
        calls += 1;
        count + 1
    });
    (count, calls)
}

pub fn inclusive_try_fold(start: usize, end: usize, stop: usize) -> (Option<usize>, Option<usize>, usize) {
    let mut range = start..=end;
    let mut calls = 0;
    let result = range.try_fold(0, |count, item| {
        calls += 1;
        if item == stop { None } else { Some(count + 1) }
    });
    (result, range.next(), calls)
}

pub fn inclusive_advance(start: usize, end: usize, n: usize) -> (Option<usize>, Option<usize>) {
    let mut range = start..=end;
    let remaining = match range.advance_by(n) {
        Ok(()) => None,
        Err(remaining) => Some(remaining.get()),
    };
    (remaining, range.next())
}

pub fn inclusive_hint(start: usize, end: usize, consume: bool) -> (usize, Option<usize>) {
    let mut range = start..=end;
    if consume {
        let _ = range.next();
    }
    range.size_hint()
}

pub fn inclusive_max_hint() -> (bool, Option<usize>) {
    let (lower, upper) = (0..=usize::MAX).size_hint();
    (lower == usize::MAX, upper)
}

pub fn once_fold(value: usize) -> (usize, usize) {
    let mut calls = 0;
    let result = std::iter::once(value).fold(10, |acc, item| {
        calls += 1;
        acc + item
    });
    (result, calls)
}

pub fn once_try_fold(value: usize, stop: usize) -> (Option<usize>, Option<usize>, usize) {
    let mut iter = std::iter::once(value);
    let mut calls = 0;
    let result = iter.try_fold(10, |acc, item| {
        calls += 1;
        if item == stop { None } else { Some(acc + item) }
    });
    (result, iter.next(), calls)
}

pub fn once_advance(value: usize, n: usize) -> (Option<usize>, Option<usize>) {
    let mut iter = std::iter::once(value);
    let remaining = match iter.advance_by(n) {
        Ok(()) => None,
        Err(remaining) => Some(remaining.get()),
    };
    (remaining, iter.next())
}

pub fn once_hint(value: usize, consume: bool) -> (usize, Option<usize>) {
    let mut iter = std::iter::once(value);
    if consume {
        let _ = iter.next();
    }
    iter.size_hint()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusive_boundaries_and_partial_consumption() {
        assert_eq!(inclusive_endpoint(4, 5), (Some(4), Some(5), None));
        assert_eq!(inclusive_endpoint(5, 5), (Some(5), None, None));
        assert_eq!(inclusive_endpoint(6, 5), (None, None, None));
        assert_eq!(inclusive_max_singleton(), (true, None, 0));
        assert_eq!(inclusive_max_any(), (true, None, 1));
        assert_eq!(inclusive_partial_count(3, 5), (Some(3), 2));
        assert_eq!(inclusive_partial_count(6, 5), (None, 0));
        assert_eq!(inclusive_nth_resume(3, 5, 1), (Some(4), Some(5)));
        assert_eq!(inclusive_nth_resume(3, 5, 2), (Some(5), None));
        assert_eq!(inclusive_nth_resume(3, 5, usize::MAX), (None, None));
    }

    #[test]
    fn inclusive_short_circuit_preserves_callback_and_receiver() {
        assert_eq!(inclusive_any_resume(3, 5, 4), (true, Some(5), 2));
        assert_eq!(inclusive_any_resume(3, 5, 5), (true, None, 3));
        assert_eq!(inclusive_any_resume(3, 5, 9), (false, None, 3));
        assert_eq!(inclusive_any_resume(6, 5, 5), (false, None, 0));
    }

    #[test]
    fn once_consumption_and_short_circuit() {
        assert_eq!(once_resume(7), (Some(7), None, 0));
        assert_eq!(once_count(7), 1);
        assert_eq!(once_any_resume(7, 7), (true, None, 1));
        assert_eq!(once_any_resume(7, 9), (false, None, 1));
        assert_eq!(once_exhausted_any(7), (false, 0));
        assert_eq!(once_nth_resume(7, 0), (Some(7), None));
        assert_eq!(once_nth_resume(7, usize::MAX), (None, None));
    }

    #[test]
    fn fold_dispatch_and_recovery() {
        assert_eq!(inclusive_fold(3, 5), (13, 3));
        assert_eq!(inclusive_fold(6, 5), (10, 0));
        assert_eq!(inclusive_fold(usize::MAX, usize::MAX), (11, 1));
        assert_eq!(inclusive_try_fold(3, 5, 4), (None, Some(5), 2));
        assert_eq!(inclusive_try_fold(3, 5, 5), (None, None, 3));
        assert_eq!(inclusive_try_fold(3, 5, 9), (Some(3), None, 3));
        assert_eq!(once_fold(7), (17, 1));
        assert_eq!(once_try_fold(7, 7), (None, None, 1));
        assert_eq!(once_try_fold(7, 9), (Some(17), None, 1));
    }

    #[test]
    fn advancing_reports_real_remainder() {
        assert_eq!(inclusive_advance(3, 5, 0), (None, Some(3)));
        assert_eq!(inclusive_advance(3, 5, 2), (None, Some(5)));
        assert_eq!(inclusive_advance(3, 5, 3), (None, None));
        assert_eq!(inclusive_advance(3, 5, 5), (Some(2), None));
        assert_eq!(once_advance(7, 0), (None, Some(7)));
        assert_eq!(once_advance(7, 1), (None, None));
        assert_eq!(once_advance(7, 3), (Some(2), None));
    }

    #[test]
    fn trusted_hints_include_the_endpoint() {
        assert_eq!(inclusive_hint(3, 5, true), (2, Some(2)));
        assert_eq!(inclusive_hint(5, 5, true), (0, Some(0)));
        assert_eq!(inclusive_hint(6, 5, false), (0, Some(0)));
        assert_eq!(inclusive_max_hint(), (true, None));
        assert_eq!(once_hint(7, false), (1, Some(1)));
        assert_eq!(once_hint(7, true), (0, Some(0)));
    }

    #[test]
    #[should_panic]
    fn inclusive_count_checks_overflow() {
        inclusive_count_overflow();
    }

    #[test]
    #[should_panic]
    fn inclusive_propagates_predicate_failure() {
        inclusive_predicate_failure();
    }

    #[test]
    #[should_panic]
    fn once_propagates_predicate_failure() {
        once_predicate_failure();
    }
}
