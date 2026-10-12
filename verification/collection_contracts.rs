use std::collections::{BTreeMap, BTreeSet, VecDeque};

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

// A collector in a trait method implementation, called through a generic
// dictionary, which does not name the implementation: on the method's own
// iterator parameter (rejected at the implementation), or on a clause of the
// implementation (rejected where the dictionary of `Gathered<One>` is built).
pub trait Gather {
    fn gather<I: IntoIterator<Item = u8>>(items: I) -> Vec<u8>;
}

pub struct Plain;

impl Gather for Plain {
    fn gather<I: IntoIterator<Item = u8>>(items: I) -> Vec<u8> {
        items.into_iter().collect()
    }
}

fn gather_through<G: Gather>(items: One) -> Vec<u8> {
    G::gather(items)
}

pub fn reject_dispatch_method() -> Vec<u8> {
    gather_through::<Plain>(One(false))
}

pub trait Run {
    fn run(self) -> Vec<u8>;
}

pub struct Gathered<I>(pub I);

impl<I: IntoIterator<Item = u8>> Run for Gathered<I> {
    fn run(self) -> Vec<u8> {
        self.0.into_iter().collect()
    }
}

fn run_through<R: Run>(r: R) -> Vec<u8> {
    r.run()
}

pub fn reject_dispatch_impl() -> Vec<u8> {
    run_through(Gathered(One(false)))
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

// A mapped left side under the inner zip: the native NoCoerce fold of the
// outer zip reads min(len) pairs by index and never maps the surplus left
// pair, while repeated next maps it once more.
pub fn reject_mapped_nested_zip() -> (Vec<(i64, i64)>, u32) {
    let mut calls = 0;
    let left = [1i64, 2];
    let values = left
        .iter()
        .map(|x| {
            calls += 1;
            *x
        })
        .zip(&[3i64, 4])
        .zip(vec![5i64])
        .map(|((x, _), z)| (x, z))
        .collect();
    (values, calls)
}

// A borrowed left side: &mut Zip is not TrustedRandomAccessNoCoerce, so the
// native TrustedLen fold takes no surplus left pair, which the caller then
// observes in the iterator it gets back.
pub fn reject_borrowed_zip_left() -> (Vec<i64>, Option<(i64, i64)>) {
    let mut left = [1i64, 2, 3].iter().zip(&[4i64, 5, 6]);
    let values = left.by_ref().zip(vec![7i64]).map(|((x, _), z)| *x + z).collect();
    (values, left.next().map(|(x, y)| (*x, *y)))
}

// A Cloned left side runs Clone::clone and yields values with drop glue.
pub fn reject_cloned_zip_left() -> Vec<(String, i64)> {
    let names = [String::from("a"), String::from("b")];
    names.iter().cloned().zip(vec![1i64]).collect()
}

// VecDeque's collector is Vec::from_iter(..).into(): the same observable
// size_hint callback as reject_vec_hint.
pub fn reject_deque_hint() -> VecDeque<u8> {
    One(false).collect()
}

// Native indexed collection invokes no skipped callback after exhaustion.
pub fn reject_take_mapped_skip() -> (Vec<i64>, u32) {
    let mut calls = 0;
    let values = [1i64, 2, 3]
        .iter()
        .map(|x| {
            calls += 1;
            *x
        })
        .skip(5)
        .take(2)
        .collect();
    (values, calls)
}

// Native in-place collection uses only indices that remain after the skip.
pub fn reject_skip_in_place() -> (Vec<i64>, u32) {
    let mut calls = 0;
    let values = vec![1i64, 2, 3]
        .into_iter()
        .map(|x| {
            calls += 1;
            x
        })
        .skip(5)
        .collect();
    (values, calls)
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
        assert!(std::panic::catch_unwind(reject_dispatch_method).is_err());
        assert!(std::panic::catch_unwind(reject_dispatch_impl).is_err());
        assert!(std::panic::catch_unwind(reject_deque_hint).is_err());
    }

    #[test]
    fn empty_source_skips_hint_and_trusted_zip_skips_excess_callback() {
        assert_eq!(reject_empty_hint(), Vec::<u8>::new());
        assert_eq!(reject_zip_consumer(), (vec![(1, 7)], 1));
        assert_eq!(reject_mapped_vec_zip(), (vec![(5, 1)], 1));
        assert_eq!(reject_drop_vec_zip(), vec![(String::from("a"), 1)]);
        assert_eq!(reject_mapped_nested_zip(), (vec![(1, 5)], 1));
        assert_eq!(reject_borrowed_zip_left(), (vec![8], Some((2, 5))));
        assert_eq!(reject_cloned_zip_left(), vec![(String::from("a"), 1)]);
    }

    #[test]
    fn indexed_skip_maps_no_skipped_item() {
        assert_eq!(reject_take_mapped_skip(), (vec![], 0));
        assert_eq!(reject_skip_in_place(), (vec![], 0));
    }
}
