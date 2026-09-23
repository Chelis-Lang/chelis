pub fn window_signature(values: &[usize], width: usize) -> usize {
    let mut signature = 0;
    for window in values.windows(width) {
        signature = signature * 100 + window[0] * 10 + window.len();
    }
    signature
}

pub fn chunk_signature(values: &[usize], width: usize) -> usize {
    let mut signature = 0;
    for chunk in values.chunks(width) {
        signature = signature * 100 + chunk[0] * 10 + chunk.len();
    }
    signature
}

pub fn window_count(values: &[usize], width: usize) -> usize {
    values.windows(width).count()
}

pub fn chunk_count(values: &[usize], width: usize) -> usize {
    values.chunks(width).count()
}

pub fn window_any_resumes(
    values: &[usize],
    width: usize,
    mut skip: usize,
    target: usize,
) -> (bool, Option<usize>) {
    let mut windows = values.windows(width);
    let found = windows.any(move |window| {
        if skip > 0 {
            skip -= 1;
            false
        } else {
            window[0] == target
        }
    });
    let next = match windows.next() {
        Some(window) => Some(window[0]),
        None => None,
    };
    (found, next)
}

pub fn chunk_any_resumes(values: &[usize], width: usize, target: usize) -> (bool, Option<usize>) {
    let mut chunks = values.chunks(width);
    let found = chunks.any(move |chunk| chunk[0] == target);
    let next = match chunks.next() {
        Some(chunk) => Some(chunk[0]),
        None => None,
    };
    (found, next)
}

pub fn windows_stay_exhausted(values: &[usize], width: usize) -> bool {
    let mut windows = values.windows(width);
    while let Some(_) = windows.next() {}
    match windows.next() {
        Some(_) => false,
        None => match windows.next() {
            Some(_) => false,
            None => true,
        },
    }
}

pub fn chunks_stay_exhausted(values: &[usize], width: usize) -> bool {
    let mut chunks = values.chunks(width);
    while let Some(_) = chunks.next() {}
    match chunks.next() {
        Some(_) => false,
        None => match chunks.next() {
            Some(_) => false,
            None => true,
        },
    }
}

pub fn mixed_end_signature(values: &[usize]) -> (usize, usize) {
    let mut iter = values.into_iter();
    let mut signature = 0;
    let mut step = 0;
    while step < 7 {
        let value = if step % 2 == 0 {
            iter.next()
        } else {
            iter.next_back()
        };
        signature = signature * 10
            + match value {
                Some(value) => *value,
                None => 0,
            };
        step += 1;
    }
    // Reuse the borrowed storage after both ends have exhausted the iterator.
    let mut total = 0;
    for value in values {
        total += *value;
    }
    (signature, total)
}

pub fn reversed_mixed_end_signature(values: &[usize]) -> usize {
    let mut forward = values.iter();
    let _ = forward.next();
    let mut iter = forward.rev();
    let mut signature = 0;
    let mut step = 0;
    while step < 5 {
        let value = if step % 2 == 0 {
            iter.next()
        } else {
            iter.next_back()
        };
        signature = signature * 10
            + match value {
                Some(value) => *value,
                None => 0,
            };
        step += 1;
    }
    signature
}

// Callers deliberately precede the implementation: inherited dispatch must add
// a dictionary dependency, rather than relying on source declaration order.
pub fn direct_inherited_count(remaining: usize) -> usize {
    Forward {
        inner: CountOverride { remaining },
    }
    .count()
}

pub fn generic_inherited_count(remaining: usize) -> usize {
    generic_count(Forward {
        inner: CountOverride { remaining },
    })
}

pub fn direct_inherited_any(remaining: usize, target: usize) -> (bool, usize) {
    let mut iter = Forward {
        inner: CountOverride { remaining },
    };
    let found = iter.any(move |value| value == target);
    (found, iter.inner.remaining)
}

pub struct Forward<I: Iterator> {
    pub inner: I,
}

impl<I: Iterator> Iterator for Forward<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub struct CountOverride {
    pub remaining: usize,
}

impl Iterator for CountOverride {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            Some(self.remaining)
        }
    }

    fn count(self) -> usize {
        // Deliberately distinguish dispatch to this override from the default.
        self.remaining + 100
    }
}

impl DoubleEndedIterator for CountOverride {
    fn next_back(&mut self) -> Option<usize> {
        self.next()
    }
}

pub fn reversed_count(remaining: usize) -> usize {
    CountOverride { remaining }.rev().count()
}

pub fn generic_count<I: Iterator>(iter: I) -> usize {
    iter.count()
}

pub fn count_override_dispatch(remaining: usize) -> usize {
    generic_count(CountOverride { remaining })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_ends_meet_once_and_preserve_borrowed_storage() {
        assert_eq!(mixed_end_signature(&[1, 2, 3, 4, 5]), (1524300, 15));
    }

    #[test]
    fn reversing_preserves_the_remaining_range_and_swaps_both_ends() {
        assert_eq!(reversed_mixed_end_signature(&[1, 2, 3, 4]), 42300);
    }

    #[test]
    fn reversed_count_does_not_select_the_forward_count_override() {
        assert_eq!(reversed_count(3), 3);
    }

    #[test]
    fn generic_count_preserves_the_selected_override() {
        assert_eq!(count_override_dispatch(3), 103);
    }

    #[test]
    fn inherited_defaults_use_next_and_preserve_method_binders() {
        assert_eq!(direct_inherited_count(3), 3);
        assert_eq!(generic_inherited_count(3), 3);
        assert_eq!(direct_inherited_any(5, 2), (true, 2));
    }

    #[test]
    fn overlapping_windows_and_partial_chunks() {
        let values = [1, 2, 3, 4];
        assert_eq!(window_signature(&values, 2), 122232);
        assert_eq!(chunk_signature(&values, 2), 1232);
        assert_eq!(chunk_signature(&values, 3), 1341);
        assert_eq!(window_signature(&values, 1), 11213141);
        assert_eq!(window_signature(&values, 5), 0);
        assert_eq!(chunk_signature(&values, usize::MAX), 14);
        assert_eq!(window_signature(&[], 1), 0);
        assert_eq!(chunk_signature(&[], 1), 0);
        assert_eq!(window_count(&values, 2), 3);
        assert_eq!(window_count(&values, usize::MAX), 0);
        assert_eq!(chunk_count(&values, 3), 2);
        assert_eq!(chunk_count(&values, usize::MAX), 1);
        assert_eq!(chunk_count(&[], 1), 0);
    }

    #[test]
    fn predicate_state_and_iterator_continuation() {
        let values = [1, 2, 3, 4, 5];
        assert_eq!(window_any_resumes(&values, 2, 0, 1), (true, Some(2)));
        assert_eq!(window_any_resumes(&values, 2, 2, 3), (true, Some(4)));
        assert_eq!(window_any_resumes(&values, 2, 0, 4), (true, None));
        assert_eq!(window_any_resumes(&values, 2, 0, 9), (false, None));
        assert_eq!(window_any_resumes(&[], 2, 0, 1), (false, None));
        assert_eq!(chunk_any_resumes(&values, 2, 3), (true, Some(5)));
        assert_eq!(chunk_any_resumes(&values, 2, 5), (true, None));
        assert_eq!(chunk_any_resumes(&values, 2, 9), (false, None));
        assert!(windows_stay_exhausted(&values, 2));
        assert!(windows_stay_exhausted(&values, 6));
        assert!(chunks_stay_exhausted(&values, 2));
    }

    #[test]
    #[should_panic]
    fn zero_window_width_panics_even_for_empty_input() {
        window_signature(&[], 0);
    }

    #[test]
    #[should_panic]
    fn zero_chunk_width_panics_even_for_empty_input() {
        chunk_signature(&[], 0);
    }
}
