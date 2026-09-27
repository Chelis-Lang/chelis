//! `<[T]>::sort_unstable` and `<[T]>::sort_unstable_by` (Aeneas.Std.Sort,
//! admission llbc/SortContracts.ml). Natively, the order of elements that
//! compare `Equal` is unspecified; the models (a stable sort) are admitted only
//! where `Equal` holds exactly for indistinguishable values, so that every
//! sorted permutation is the same list: `Ord` of integers, `bool`, `char`,
//! `str`, `String`, references, tuples and `#[derive(Ord)]` types of those, and
//! comparators `|a, b| a.cmp(b)` / `|a, b| b.cmp(a)` on such types. Custom
//! `Ord` impls, other comparators and key functions are rejected with
//! E_UNSUPPORTED_SORT.
//!
//! `<[T]>::sort_by` (stable) is admitted where the comparator is `Ord::cmp` of the
//! same key of both elements, projected without computation (fields, borrows,
//! copies, tuples), on such a key type: a pure total preorder, for which the
//! stable order is unique. Equal keys keep their input order.
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BindingId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slot {
    Free,
    Held(u32, bool),
}

pub fn sort_usizes(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable();
    values
}

pub fn sort_pairs(mut values: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    values.sort_unstable();
    values
}

pub fn sort_strings(mut values: Vec<String>) -> Vec<String> {
    values.sort_unstable();
    values
}

pub fn sort_strs(mut values: Vec<&str>) -> Vec<&str> {
    values.sort_unstable();
    values
}

pub fn sort_flags(mut values: Vec<(bool, char, i8)>) -> Vec<(bool, char, i8)> {
    values.sort_unstable();
    values
}

/// A derived `Ord` on a newtype, as `chelis_types::linearity::BindingId`.
pub fn sort_ids(raw: &[u32]) -> Vec<u32> {
    let mut ids: Vec<BindingId> = raw.iter().map(|&value| BindingId(value)).collect();
    ids.sort_unstable();
    ids.into_iter().map(|id| id.0).collect()
}

/// A derived `Ord` on an enum: the variant first, then the fields.
pub fn sort_slots(mut slots: Vec<Slot>) -> Vec<Slot> {
    slots.sort_unstable();
    slots
}

/// The comparator of `chelis_ir::lower` (lower.rs:14064): descending.
pub fn sort_descending(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable_by(|a, b| b.cmp(a));
    values
}

/// `Ord::cmp` of the arguments in order, on tuples: ascending.
pub fn sort_ascending_by(mut values: Vec<(u8, char)>) -> Vec<(u8, char)> {
    values.sort_unstable_by(|a, b| a.cmp(b));
    values
}

/// A descending comparator over a derived `Ord`.
pub fn sort_ids_descending(raw: &[u32]) -> Vec<u32> {
    let mut ids: Vec<BindingId> = raw.iter().map(|&value| BindingId(value)).collect();
    ids.sort_unstable_by(|a, b| b.cmp(a));
    ids.into_iter().map(|id| id.0).collect()
}

/// `Ord` compares the key only: equal keys hide distinct payloads.
#[derive(Clone, Copy, Debug)]
pub struct Keyed {
    pub key: u32,
    pub payload: u32,
}

impl PartialEq for Keyed {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Keyed {}

impl PartialOrd for Keyed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Keyed {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

pub fn reject_custom_ord(mut values: Vec<Keyed>) -> Vec<Keyed> {
    values.sort_unstable();
    values
}

/// Compares the first components only: `Equal` hides distinct second components.
pub fn reject_partial_comparator(mut values: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    values.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    values
}

/// Returns `Ord::cmp`, but also updates its state: only the plain bodies are admitted.
pub fn reject_stateful_comparator(mut values: Vec<usize>) -> (Vec<usize>, bool) {
    let mut called = false;
    values.sort_unstable_by(|a, b| {
        called = true;
        a.cmp(b)
    });
    (values, called)
}

/// `Ord::cmp` of a hand-written `Ord`.
pub fn reject_custom_ord_comparator(mut values: Vec<Keyed>) -> Vec<Keyed> {
    values.sort_unstable_by(|a, b| b.cmp(a));
    values
}

pub fn reject_key(mut values: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    values.sort_unstable_by_key(|pair| pair.0);
    values
}

/// `sort_by` on the first component of pairs (chelis-deep annotations_codec.rs):
/// equal keys keep their input order.
pub fn sort_by_first(mut entries: Vec<(u32, String)>) -> Vec<(u32, String)> {
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    entries
}

pub struct Definition {
    pub adt_name: String,
    pub ctor_name: String,
    pub arity: u32,
}

fn definitions(raw: Vec<(String, String, u32)>) -> Vec<Definition> {
    raw.into_iter()
        .map(|(adt_name, ctor_name, arity)| Definition { adt_name, ctor_name, arity })
        .collect()
}

fn flatten(definitions: Vec<Definition>) -> Vec<(String, String, u32)> {
    definitions.into_iter().map(|d| (d.adt_name, d.ctor_name, d.arity)).collect()
}

/// A tuple of borrowed fields as the key (chelis-ir host.rs `sort_by` of the
/// constructor definitions).
pub fn sort_by_names(raw: Vec<(String, String, u32)>) -> Vec<(String, String, u32)> {
    let mut definitions = definitions(raw);
    definitions.sort_by(|left, right| {
        (&left.adt_name, &left.ctor_name).cmp(&(&right.adt_name, &right.ctor_name))
    });
    flatten(definitions)
}

/// One field as the key (chelis-types builtins.rs), descending.
pub fn sort_by_name_descending(raw: Vec<(String, String, u32)>) -> Vec<(String, String, u32)> {
    let mut definitions = definitions(raw);
    definitions.sort_by(|a, b| b.adt_name.cmp(&a.adt_name));
    flatten(definitions)
}

/// A tuple of copied fields as the key.
pub fn sort_by_copied(mut values: Vec<(u32, u8, u8)>) -> Vec<(u32, u8, u8)> {
    values.sort_by(|a, b| (a.2, a.0).cmp(&(b.2, b.0)));
    values
}

/// Different keys of the two elements: not a preorder.
pub fn reject_by_crossed_keys(mut values: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    values.sort_by(|a, b| a.0.cmp(&b.1));
    values
}

/// A computed key.
pub fn reject_by_computed_key(mut values: Vec<u32>) -> Vec<u32> {
    values.sort_by(|a, b| (a % 3).cmp(&(b % 3)));
    values
}

/// Returns `Ord::cmp` of keys, but also updates its state.
pub fn reject_by_stateful(mut values: Vec<(u32, u32)>) -> (Vec<(u32, u32)>, u32) {
    let mut calls = 0;
    values.sort_by(|a, b| {
        calls += 1;
        a.0.cmp(&b.0)
    });
    (values, calls.min(1))
}

/// A float key (`partial_cmp`): no total order.
pub fn reject_by_float(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

/// `Ord::cmp` of a hand-written `Ord` (not checked to be a total order).
pub fn reject_by_custom_ord(mut values: Vec<Keyed>) -> Vec<Keyed> {
    values.sort_by(|a, b| a.cmp(b));
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted() {
        assert_eq!(sort_usizes(vec![3, 1, 2, 1]), vec![1, 1, 2, 3]);
        assert_eq!(sort_usizes(vec![5, 4, 3, 2, 1]), vec![1, 2, 3, 4, 5]);
        assert_eq!(sort_usizes(vec![]), vec![]);
        assert_eq!(sort_pairs(vec![(2, 1), (1, 9), (2, 0)]), vec![(1, 9), (2, 0), (2, 1)]);
        assert_eq!(
            sort_strings(vec!["b".to_string(), "ab".to_string(), "a".to_string()]),
            vec!["a".to_string(), "ab".to_string(), "b".to_string()]
        );
        assert_eq!(sort_strs(vec!["z", "", "é", "e"]), vec!["", "e", "z", "é"]);
        assert_eq!(
            sort_flags(vec![(true, 'a', 1), (false, 'b', -1), (true, 'a', -2)]),
            vec![(false, 'b', -1), (true, 'a', -2), (true, 'a', 1)]
        );
        assert_eq!(sort_ids(&[7, 3, 5, 3]), vec![3, 3, 5, 7]);
        assert_eq!(
            sort_slots(vec![Slot::Held(2, false), Slot::Free, Slot::Held(1, true), Slot::Held(2, true)]),
            vec![Slot::Free, Slot::Held(1, true), Slot::Held(2, false), Slot::Held(2, true)]
        );
        assert_eq!(sort_descending(vec![3, 1, 2, 1, 3]), vec![3, 3, 2, 1, 1]);
        assert_eq!(sort_descending(vec![1, 2, 3, 4]), vec![4, 3, 2, 1]);
        assert_eq!(sort_descending(vec![]), vec![]);
        assert_eq!(
            sort_ascending_by(vec![(2, 'b'), (1, 'z'), (2, 'a'), (1, 'z')]),
            vec![(1, 'z'), (1, 'z'), (2, 'a'), (2, 'b')]
        );
        assert_eq!(
            sort_ascending_by(vec![(3, 'c'), (2, 'b'), (1, 'a')]),
            vec![(1, 'a'), (2, 'b'), (3, 'c')]
        );
        assert_eq!(sort_ids_descending(&[3, 7, 3, 5]), vec![7, 5, 3, 3]);
    }

    fn owned(raw: &[(&str, &str, u32)]) -> Vec<(String, String, u32)> {
        raw.iter().map(|(a, c, n)| (a.to_string(), c.to_string(), *n)).collect()
    }

    #[test]
    fn stable_by_keys() {
        let entries = |raw: &[(u32, &str)]| -> Vec<(u32, String)> {
            raw.iter().map(|(k, v)| (*k, v.to_string())).collect()
        };
        assert_eq!(
            sort_by_first(entries(&[(2, "b"), (1, "z"), (2, "a"), (1, "y")])),
            entries(&[(1, "z"), (1, "y"), (2, "b"), (2, "a")])
        );
        assert_eq!(sort_by_first(vec![]), vec![]);
        assert_eq!(
            sort_by_names(owned(&[("B", "x", 1), ("A", "y", 2), ("A", "x", 3), ("B", "x", 4)])),
            owned(&[("A", "x", 3), ("A", "y", 2), ("B", "x", 1), ("B", "x", 4)])
        );
        assert_eq!(
            sort_by_name_descending(owned(&[("a", "", 1), ("c", "", 2), ("b", "", 3), ("c", "", 4)])),
            owned(&[("c", "", 2), ("c", "", 4), ("b", "", 3), ("a", "", 1)])
        );
        assert_eq!(
            sort_by_copied(vec![(1, 5, 2), (2, 0, 1), (3, 9, 2), (0, 0, 1)]),
            vec![(0, 0, 1), (2, 0, 1), (1, 5, 2), (3, 9, 2)]
        );
    }

    #[test]
    fn rejected_by_keys_are_native_behaviors() {
        assert_eq!(reject_by_crossed_keys(vec![(1, 0)]), vec![(1, 0)]);
        assert_eq!(reject_by_computed_key(vec![4, 3]), vec![3, 4]);
        assert_eq!(reject_by_stateful(vec![(2, 0), (1, 0)]), (vec![(1, 0), (2, 0)], 1));
        assert_eq!(reject_by_float(vec![2.0, 1.0]), vec![1.0, 2.0]);
        assert_eq!(reject_by_custom_ord(vec![Keyed { key: 2, payload: 0 }]).len(), 1);
    }

    #[test]
    fn rejected_are_native_behaviors() {
        assert_eq!(reject_partial_comparator(vec![(2, 0), (1, 5)]), vec![(1, 5), (2, 0)]);
        assert_eq!(reject_stateful_comparator(vec![2, 1]), (vec![1, 2], true));
        assert_eq!(reject_custom_ord_comparator(vec![Keyed { key: 2, payload: 0 }]).len(), 1);
        assert_eq!(reject_key(vec![(2, 0), (1, 5)]), vec![(1, 5), (2, 0)]);
        assert_eq!(reject_custom_ord(vec![Keyed { key: 2, payload: 0 }]).len(), 1);
    }
}
