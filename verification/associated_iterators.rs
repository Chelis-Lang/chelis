// Copy deliberately differs from Clone: copying must not call user Clone.
#[derive(Copy)]
pub struct DifferentClone {
    pub value: usize,
    pub panic_on_clone: bool,
}

impl Clone for DifferentClone {
    fn clone(&self) -> Self {
        if self.panic_on_clone {
            panic!("selected Clone failed");
        }
        Self {
            value: self.value + 10,
            panic_on_clone: false,
        }
    }
}

pub fn copied_signature(values: &[DifferentClone]) -> usize {
    let mut signature = 0;
    for value in values.iter().copied() {
        signature = signature * 100 + value.value;
    }
    signature
}

pub fn cloned_signature(values: &[DifferentClone]) -> usize {
    let mut signature = 0;
    for value in values.iter().cloned() {
        signature = signature * 100 + value.value;
    }
    signature
}

pub fn copied_resume(values: &[DifferentClone]) -> (Option<usize>, Option<usize>, usize) {
    let mut iter = values.iter().copied();
    let first = match iter.next() {
        Some(value) => Some(value.value),
        None => None,
    };
    let second = match iter.next() {
        Some(value) => Some(value.value),
        None => None,
    };
    (first, second, iter.count())
}

pub fn cloned_resume(values: &[DifferentClone]) -> (Option<usize>, Option<usize>, usize) {
    let mut iter = values.iter().cloned();
    let first = match iter.next() {
        Some(value) => Some(value.value),
        None => None,
    };
    let second = match iter.next() {
        Some(value) => Some(value.value),
        None => None,
    };
    (first, second, iter.count())
}

pub fn copied_any_resumes(values: &[DifferentClone], target: usize) -> (bool, Option<usize>) {
    let mut iter = values.iter().copied();
    let found = iter.any(|value| value.value == target);
    let next = match iter.next() {
        Some(value) => Some(value.value),
        None => None,
    };
    (found, next)
}

pub fn cloned_any_resumes(values: &[DifferentClone], target: usize) -> (bool, Option<usize>) {
    let mut iter = values.iter().cloned();
    let found = iter.any(|value| value.value == target);
    let next = match iter.next() {
        Some(value) => Some(value.value),
        None => None,
    };
    (found, next)
}

pub fn cloned_first(values: &[DifferentClone]) -> Option<usize> {
    match values.iter().cloned().next() {
        Some(value) => Some(value.value),
        None => None,
    }
}

pub fn copied_count(values: &[DifferentClone]) -> usize {
    values.iter().copied().count()
}

pub fn cloned_count(values: &[DifferentClone]) -> usize {
    values.iter().cloned().count()
}

pub struct CountOverride<'a> {
    pub values: &'a [usize],
    pub offset: usize,
}

impl<'a> Iterator for CountOverride<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset < self.values.len() {
            let offset = self.offset;
            self.offset += 1;
            Some(&self.values[offset])
        } else {
            None
        }
    }

    fn count(self) -> usize {
        100 + self.values.len() - self.offset
    }
}

pub fn copied_count_override(values: &[usize]) -> usize {
    CountOverride { values, offset: 0 }.copied().count()
}

pub fn cloned_count_override(values: &[usize]) -> usize {
    CountOverride { values, offset: 0 }.cloned().count()
}

// A stateful predicate accepts its second and fourth calls. Its state must
// survive both rejected values inside next and accepted values between nexts.
pub fn filter_resume(values: &[usize]) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut calls = 0;
    let mut iter = values.iter().copied().filter(move |_| {
        calls += 1;
        calls == 2 || calls == 4
    });
    let first = iter.next();
    let second = iter.next();
    let third = iter.next();
    (first, second, third)
}

pub fn filter_count(values: &[usize]) -> usize {
    let mut calls = 0;
    values
        .iter()
        .copied()
        .filter(move |_| {
            calls += 1;
            calls == 2 || calls == 4
        })
        .count()
}

pub fn filter_panics(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .copied()
        .filter(|_| panic!("predicate failed"))
        .next()
}

pub fn owned_array_resume<const N: usize>(
    values: [usize; N],
) -> (Option<usize>, Option<usize>, usize) {
    let mut iter = values.into_iter();
    let first = iter.next();
    let second = iter.next();
    (first, second, iter.count())
}

pub fn owned_array_any_resume<const N: usize>(
    values: [usize; N],
    needle: usize,
) -> (bool, Option<usize>) {
    let mut iter = values.into_iter();
    let found = iter.any(|value| value == needle);
    (found, iter.next())
}

pub struct DispatchOverride<'a> {
    pub values: &'a [usize],
    pub offset: usize,
}

impl<'a> Iterator for DispatchOverride<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset < self.values.len() {
            let index = self.offset;
            self.offset += 1;
            Some(&self.values[index])
        } else {
            None
        }
    }

    // Deliberately distinguish override dispatch from repeated next.
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.offset = self.values.len();
        if n < self.values.len() {
            Some(&self.values[self.values.len() - n - 1])
        } else {
            None
        }
    }

    fn fold<B, F>(self, mut init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut end = self.values.len();
        while self.offset < end {
            end -= 1;
            init = f(init, &self.values[end]);
        }
        init
    }

    // Search in the opposite order to next/try_fold. Adapters must dispatch
    // the selected method rather than rebuild it from an unrelated primitive.
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        while self.offset < self.values.len() {
            self.offset += 1;
            let item = &self.values[self.values.len() - self.offset];
            if predicate(&item) {
                return Some(item);
            }
        }
        None
    }

    fn find_map<B, F>(&mut self, mut f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        while self.offset < self.values.len() {
            self.offset += 1;
            match f(&self.values[self.values.len() - self.offset]) {
                Some(value) => return Some(value),
                None => {}
            }
        }
        None
    }
}

impl<'a> DoubleEndedIterator for DispatchOverride<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.offset < self.values.len() {
            self.offset += 1;
            Some(&self.values[self.values.len() - self.offset])
        } else {
            None
        }
    }

    // Opposite to next_back/try_rfold, distinguishing Rev.find's dispatch.
    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        while let Some(item) = self.next() {
            if predicate(&item) {
                return Some(item);
            }
        }
        None
    }
}

pub fn borrowed_fold_override(values: &[usize]) -> (usize, usize, usize) {
    let mut iter = DispatchOverride { values, offset: 0 };
    let mut calls = 0;
    let result = (&mut iter).fold(0, |acc, value| {
        calls += 1;
        acc * 100 + *value
    });
    (result, iter.offset, calls)
}

pub fn copied_nth_override(values: &[usize]) -> Option<usize> {
    DispatchOverride { values, offset: 0 }.copied().nth(0)
}

pub fn mapped_fold_override(values: &[usize]) -> usize {
    DispatchOverride { values, offset: 0 }
        .map(|value| *value + 10)
        .fold(0, |acc, value| acc * 100 + value)
}

pub fn cloned_fold_override(values: &[usize]) -> usize {
    DispatchOverride { values, offset: 0 }
        .cloned()
        .fold(0, |acc, value| acc * 100 + value)
}

pub fn map_nth_state(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = values.iter().map(|value| {
        calls += 1;
        *value + calls
    });
    let first = iter.nth(1);
    let second = iter.next();
    (first, second, calls)
}

pub fn filter_find_override(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = DispatchOverride { values, offset: 0 }.filter(|_| {
        calls += 1;
        calls == 2 || calls == 3
    });
    let first = match iter.next() {
        Some(value) => Some(*value),
        None => None,
    };
    let second = match iter.next() {
        Some(value) => Some(*value),
        None => None,
    };
    (first, second, calls)
}

pub fn filter_map_find_override(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = DispatchOverride { values, offset: 0 }.filter_map(|value| {
        calls += 1;
        if calls == 2 || calls == 3 {
            Some((*value + 10, calls == 2))
        } else {
            None
        }
    });
    let first = match iter.next() {
        Some((value, first)) => {
            if first {
                Some(value)
            } else {
                None
            }
        }
        None => None,
    };
    let second = match iter.next() {
        Some((value, first)) => {
            if first {
                None
            } else {
                Some(value)
            }
        }
        None => None,
    };
    (first, second, calls)
}

// The pinned Filter inherits find through try_fold; nested filters must not
// forward to DispatchOverride.find, nor repeatedly call the inner Filter.next.
pub fn nested_filter_find_dispatch(
    values: &[usize],
) -> (Option<usize>, Option<usize>, usize, usize) {
    let mut inner_calls = 0;
    let mut outer_calls = 0;
    let mut iter = DispatchOverride { values, offset: 0 }
        .filter(|value| {
            inner_calls += 1;
            **value != 1
        })
        .filter(|_| {
            outer_calls += 1;
            outer_calls >= 2
        });
    let first = match iter.next() {
        Some(value) => Some(*value),
        None => None,
    };
    let second = match iter.next() {
        Some(value) => Some(*value),
        None => None,
    };
    (first, second, inner_calls, outer_calls)
}

pub fn all_state(values: &[usize]) -> (bool, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = values.iter().copied();
    let result = iter.all(|value| {
        calls += 1;
        value != 2
    });
    (result, iter.next(), calls)
}

pub fn find_state(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = values.iter();
    let result = match iter.find(|value| {
        calls += 1;
        **value == 2
    }) {
        Some(value) => Some(*value),
        None => None,
    };
    let next = match iter.next() {
        Some(value) => Some(*value),
        None => None,
    };
    (result, next, calls)
}

pub fn reversed_find_override(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = DispatchOverride { values, offset: 0 }.rev();
    let result = match iter.find(|_| {
        calls += 1;
        calls == 2
    }) {
        Some(value) => Some(*value),
        None => None,
    };
    let next = match iter.next() {
        Some(value) => Some(*value),
        None => None,
    };
    (result, next, calls)
}

pub fn reversed_rfind_override(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = DispatchOverride { values, offset: 0 }.rev();
    let result = match iter.rfind(|_| {
        calls += 1;
        calls == 2
    }) {
        Some(value) => Some(*value),
        None => None,
    };
    let next = match iter.next_back() {
        Some(value) => Some(*value),
        None => None,
    };
    (result, next, calls)
}

pub fn find_map_state(values: &[usize]) -> (Option<bool>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = values.iter().copied();
    let result = iter.find_map(|value| {
        calls += 1;
        if calls == 2 {
            Some(value == 2)
        } else {
            None
        }
    });
    (result, iter.next(), calls)
}

pub fn position_state(values: &[usize]) -> (Option<usize>, Option<usize>, usize) {
    let mut calls = 0;
    let mut iter = values.iter().copied();
    let result = iter.position(|value| {
        calls += 1;
        value == 2
    });
    (result, iter.next(), calls)
}

// Match production's associated IntoIter dispatch rather than a concrete
// slice-adapter constructor. Rejected items still mutate the captured counter.
pub fn generic_filter_map_collect<'a>(values: impl IntoIterator<Item = &'a usize>) -> Vec<usize> {
    let mut calls = 0;
    values
        .into_iter()
        .filter_map(move |value| {
            calls += 1;
            if *value % 2 == 0 {
                Some(*value + calls)
            } else {
                None
            }
        })
        .collect()
}

pub fn filter_map_collect(values: &[usize]) -> Vec<usize> {
    generic_filter_map_collect(values)
}

pub fn borrowed_filter_map_collect(values: &[usize]) -> (Vec<usize>, usize) {
    let mut calls = 0;
    let selected = values
        .iter()
        .filter_map(|value| {
            calls += 1;
            if *value % 2 == 0 {
                Some(*value + calls)
            } else {
                None
            }
        })
        .collect();
    (selected, calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values() -> [DifferentClone; 3] {
        [
            DifferentClone {
                value: 1,
                panic_on_clone: false,
            },
            DifferentClone {
                value: 2,
                panic_on_clone: false,
            },
            DifferentClone {
                value: 3,
                panic_on_clone: false,
            },
        ]
    }

    #[test]
    fn copying_and_cloning_choose_different_semantics() {
        assert_eq!(copied_signature(&values()), 10203);
        assert_eq!(cloned_signature(&values()), 111213);
        assert_eq!(copied_resume(&values()), (Some(1), Some(2), 1));
        assert_eq!(cloned_resume(&values()), (Some(11), Some(12), 1));
        assert_eq!(copied_any_resumes(&values(), 2), (true, Some(3)));
        assert_eq!(cloned_any_resumes(&values(), 12), (true, Some(13)));
        assert_eq!(cloned_any_resumes(&values(), 13), (true, None));
        assert_eq!(copied_any_resumes(&values(), 99), (false, None));
    }

    #[test]
    fn empty_adapters_do_not_clone_and_remain_exhausted() {
        assert_eq!(copied_signature(&[]), 0);
        assert_eq!(cloned_signature(&[]), 0);
        assert_eq!(copied_resume(&[]), (None, None, 0));
        assert_eq!(cloned_resume(&[]), (None, None, 0));
        assert_eq!(cloned_first(&[]), None);
        assert_eq!(cloned_any_resumes(&[], 12), (false, None));
    }

    #[test]
    fn clone_panics_only_when_the_item_is_reached() {
        let values = [
            DifferentClone {
                value: 1,
                panic_on_clone: false,
            },
            DifferentClone {
                value: 2,
                panic_on_clone: true,
            },
        ];
        assert_eq!(copied_signature(&values), 102);
        assert_eq!(copied_count(&values), 2);
        assert_eq!(cloned_first(&values), Some(11));
        assert!(std::panic::catch_unwind(|| cloned_signature(&values)).is_err());
        assert!(std::panic::catch_unwind(|| cloned_count(&values)).is_err());
        assert!(std::panic::catch_unwind(|| cloned_first(&values[1..])).is_err());
    }

    #[test]
    fn count_preserves_actual_adapter_overrides() {
        assert_eq!(copied_count_override(&[1, 2, 3]), 103);
        assert_eq!(cloned_count_override(&[1, 2, 3]), 3);
        assert_eq!(copied_count_override(&[]), 100);
        assert_eq!(cloned_count_override(&[]), 0);
    }

    #[test]
    fn filtering_threads_predicate_and_iterator_state() {
        assert_eq!(filter_resume(&[1, 2, 3, 4, 5]), (Some(2), Some(4), None));
        assert_eq!(filter_resume(&[1]), (None, None, None));
        assert_eq!(filter_resume(&[]), (None, None, None));
        assert_eq!(filter_count(&[1, 2, 3, 4, 5]), 2);
        assert_eq!(filter_count(&[]), 0);
        assert_eq!(filter_panics(&[]), None);
        assert!(std::panic::catch_unwind(|| filter_panics(&[1])).is_err());
    }

    #[test]
    fn owned_arrays_preserve_the_unconsumed_suffix() {
        assert_eq!(owned_array_resume([3, 5, 29]), (Some(3), Some(5), 1));
        assert_eq!(owned_array_resume([]), (None, None, 0));
        assert_eq!(owned_array_any_resume([3, 5, 29], 5), (true, Some(29)));
        assert_eq!(owned_array_any_resume([3, 5, 29], 99), (false, None));
    }

    #[test]
    fn adapters_dispatch_native_indexed_and_fold_overrides() {
        assert_eq!(copied_nth_override(&[1, 2, 3]), Some(3));
        assert_eq!(mapped_fold_override(&[1, 2, 3]), 131211);
        assert_eq!(cloned_fold_override(&[1, 2, 3]), 30201);
        assert_eq!(borrowed_fold_override(&[1, 2, 3]), (10203, 3, 3));
        assert_eq!(map_nth_state(&[1, 2, 3]), (Some(4), Some(6), 3));
    }

    #[test]
    fn searching_selects_overrides_and_preserves_short_circuit_state() {
        assert_eq!(
            filter_find_override(&[1, 2, 3, 4, 5]),
            (Some(4), Some(3), 3)
        );
        assert_eq!(
            filter_map_find_override(&[1, 2, 3, 4, 5]),
            (Some(14), Some(13), 3)
        );
        assert_eq!(
            nested_filter_find_dispatch(&[1, 2, 3, 4, 5]),
            (Some(3), Some(4), 4, 3)
        );
        assert_eq!(
            reversed_find_override(&[1, 2, 3, 4, 5]),
            (Some(2), Some(3), 2)
        );
        assert_eq!(
            reversed_rfind_override(&[1, 2, 3, 4, 5]),
            (Some(4), Some(3), 2)
        );
        assert_eq!(all_state(&[1, 2, 3]), (false, Some(3), 2));
        assert_eq!(find_state(&[1, 2, 3]), (Some(2), Some(3), 2));
        assert_eq!(find_map_state(&[1, 2, 3]), (Some(true), Some(3), 2));
        assert_eq!(position_state(&[1, 2, 3]), (Some(1), Some(3), 2));
        assert_eq!(all_state(&[]), (true, None, 0));
        assert_eq!(find_state(&[1, 3]), (None, None, 2));
        assert_eq!(find_map_state(&[1]), (None, None, 1));
        assert_eq!(position_state(&[1, 3]), (None, None, 2));
    }

    #[test]
    fn filter_map_collection_preserves_owned_and_borrowed_callback_state() {
        assert_eq!(filter_map_collect(&[1, 2, 3, 4, 5]), vec![4, 8]);
        assert_eq!(
            borrowed_filter_map_collect(&[1, 2, 3, 4, 5]),
            (vec![4, 8], 5)
        );
        assert_eq!(borrowed_filter_map_collect(&[1]), (vec![], 1));
        assert_eq!(borrowed_filter_map_collect(&[]), (vec![], 0));
    }
}
