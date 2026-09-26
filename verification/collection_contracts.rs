use std::collections::{BTreeMap, BTreeSet};

pub struct One(pub bool);

impl Iterator for One {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.0 {
            None
        } else {
            self.0 = true;
            Some(7)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        panic!("observable size_hint callback")
    }
}

pub fn reject_vec_hint() -> Vec<u8> {
    One(false).collect()
}

pub fn reject_borrowed_map_hint() -> (Vec<u8>, u8) {
    let mut calls = 0;
    let values = {
        let mut iter = One(false).map(|item| {
            calls += 1;
            item
        });
        iter.by_ref().collect()
    };
    (values, calls)
}

// Admission is a dictionary contract, not a special case for a known input.
// Native Rust does not call size_hint here, but the same unmodeled dictionary
// must be rejected rather than conditionally published as an accepted model.
pub fn reject_empty_hint() -> Vec<u8> {
    One(true).collect()
}

pub fn reject_map_hint() -> bool {
    One(false)
        .map(|item| (item, item))
        .collect::<BTreeMap<_, _>>()
        .is_empty()
}

pub fn reject_set_hint() -> bool {
    One(false).collect::<BTreeSet<_>>().is_empty()
}

pub fn reject_extend_hint() -> Vec<u8> {
    let mut values = Vec::new();
    values.extend(One(false));
    values
}

pub struct WrappedOne;

impl IntoIterator for WrappedOne {
    type Item = u8;
    type IntoIter = One;

    fn into_iter(self) -> One {
        One(false)
    }
}

pub fn reject_wrapped_hint() -> Vec<u8> {
    Vec::from_iter(WrappedOne)
}

pub fn reject_result_hint() -> Result<Vec<u8>, u8> {
    One(false).map(Ok).collect()
}

pub fn reject_generic_iterator<I: Iterator<Item = u8>>(input: I) -> Vec<u8> {
    input.collect()
}

pub fn reject_generic_collector<C: FromIterator<u8>>(input: One) -> C {
    input.collect()
}

// Zip's trusted fold does not run an excess left mapping callback. Repeated
// next is not a valid replacement even though both source hints are pure.
pub fn reject_zip_consumer() -> (Vec<(u8, u8)>, u8) {
    let mut seen = 0;
    let values = [1, 2]
        .into_iter()
        .map(|item| {
            seen += item;
            item
        })
        .zip([7])
        .collect();
    (values, seen)
}

// A left side with a callback: the native collector folds this
// TrustedRandomAccessNoCoerce zip over the shorter length, while repeated
// Zip::next would map one surplus left element.
pub fn reject_mapped_vec_zip() -> (Vec<(i64, i64)>, u32) {
    let mut calls = 0;
    let values = vec![5i64, 7]
        .into_iter()
        .map(|l| {
            calls += 1;
            l
        })
        .zip(vec![1i64])
        .collect();
    (values, calls)
}

// vec::IntoIter of an element type with drop glue is not
// TrustedRandomAccessNoCoerce: its surplus next moves out and drops a value.
pub fn reject_drop_vec_zip() -> Vec<(String, i64)> {
    vec![String::from("a"), String::from("b")]
        .into_iter()
        .zip(vec![1i64])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_item_hint_panics_in_each_native_collector() {
        assert!(std::panic::catch_unwind(reject_vec_hint).is_err());
        assert!(std::panic::catch_unwind(reject_borrowed_map_hint).is_err());
        assert!(std::panic::catch_unwind(reject_map_hint).is_err());
        assert!(std::panic::catch_unwind(reject_set_hint).is_err());
        assert!(std::panic::catch_unwind(reject_extend_hint).is_err());
        assert!(std::panic::catch_unwind(reject_wrapped_hint).is_err());
        assert!(std::panic::catch_unwind(reject_result_hint).is_err());
        assert!(std::panic::catch_unwind(|| reject_generic_iterator(One(false))).is_err());
        assert!(
            std::panic::catch_unwind(|| reject_generic_collector::<Vec<_>>(One(false))).is_err()
        );
    }

    #[test]
    fn empty_source_skips_hint_and_trusted_zip_skips_excess_callback() {
        assert_eq!(reject_empty_hint(), Vec::<u8>::new());
        assert_eq!(reject_zip_consumer(), (vec![(1, 7)], 1));
        assert_eq!(reject_mapped_vec_zip(), (vec![(5, 1)], 1));
        assert_eq!(reject_drop_vec_zip(), vec![(String::from("a"), 1)]);
    }
}
