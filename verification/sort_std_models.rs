//! `<[T]>::sort_unstable` and `<[T]>::sort_unstable_by` (Aeneas.Std.Sort,
//! admission llbc/SortContracts.ml). Natively, the order of elements that
//! compare `Equal` is unspecified; the models (a stable sort) are admitted only
//! where `Equal` holds exactly for indistinguishable values, so that every
//! sorted permutation is the same list: `Ord` of integers, `bool`, `char`,
//! `str`, `String`, references, tuples, `Option`, `Reverse` and
//! `#[derive(PartialOrd, Ord)]` types of those, and comparators
//! `|a, b| a.cmp(b)` / `|a, b| b.cmp(a)` on such types. Custom `Ord` or
//! `PartialOrd` impls, other comparators and key functions are rejected with
//! E_UNSUPPORTED_SORT.
//!
//! The stable sorts keep the input order of equal elements. `<[T]>::sort` is
//! admitted on the element types above. `<[T]>::sort_by` is admitted where the
//! comparator is `Ord::cmp` of the same key of both elements, projected without
//! computation (fields, borrows, copies, tuples), on such a key type: a pure
//! total preorder, for which the stable order is unique. `<[T]>::sort_by_key` is
//! admitted where the key function is an `Fn` closure (it does not mutate its
//! captures) or a function item, with such a key type.
use std::cmp::{Ordering, Reverse};

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

/// `sort` (stable) on a derived `Ord` newtype (chelis-ir capacity_key.rs
/// `obligations.sort()`).
pub fn sort_stable_ids(raw: &[u32]) -> Vec<u32> {
    let mut ids: Vec<BindingId> = raw.iter().map(|&n| BindingId(n)).collect();
    ids.sort();
    ids.into_iter().map(|id| id.0).collect()
}

/// `sort` on strings (chelis-ir span_merge.rs `merged_spans.sort()`).
pub fn sort_stable_strings(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values
}

/// `sort` on `Option` (`None` first) and `Reverse` (descending) components.
pub fn sort_stable_options(mut values: Vec<(Option<u32>, Reverse<u8>)>) -> Vec<(Option<u32>, Reverse<u8>)> {
    values.sort();
    values
}

/// A derived `Ord` through boxes and boxed slices (the field types of chelis-ir
/// capacity_key.rs `CanonicalCapacity`, `canonical_factors.sort()`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Capacity {
    Literal(u64),
    Product(Box<[u32]>),
    Quotient { dividend: Box<u32>, divisor: Box<u32> },
}

pub fn sort_capacities(mut values: Vec<Capacity>) -> Vec<Capacity> {
    values.sort();
    values
}

/// `sort` in place: the sorted slice is given back.
pub fn sort_stable_in_place(values: &mut [(u8, Slot)]) {
    values.sort();
}

/// `sort_by_key` on a copied field (chelis-types env.rs `vars.sort_by_key(|v| v.0)`):
/// equal keys keep their input order.
pub fn sort_by_key_field(mut values: Vec<(u32, char)>) -> Vec<(u32, char)> {
    values.sort_by_key(|v| v.0);
    values
}

/// A tuple of copied fields (chelis-ir ownership/last_use.rs
/// `seeds.sort_by_key(|seed| (seed.definition_order, seed.id))`).
pub fn sort_by_key_fields(mut values: Vec<(u8, u32, char)>) -> Vec<(u8, u32, char)> {
    values.sort_by_key(|v| (v.1, v.0));
    values
}

/// `Reverse` of a field (last_use.rs `sort_by_key(|placed| Reverse(placed.definition_order))`).
pub fn sort_by_key_reverse(mut values: Vec<(usize, char)>) -> Vec<(usize, char)> {
    values.sort_by_key(|v| Reverse(v.0));
    values
}

/// A borrowed `&str` key through a method call (chelis-deep printer.rs
/// `sorted.sort_by_key(|(key, _)| key.as_str())`).
pub fn sort_by_key_str(entries: Vec<(String, u32)>) -> Vec<u32> {
    let mut sorted: Vec<&(String, u32)> = entries.iter().collect();
    sorted.sort_by_key(|(key, _)| key.as_str());
    sorted.iter().map(|(_, value)| *value).collect()
}

/// A key read through a captured table (last_use.rs
/// `expected.sort_by_key(|owner| Reverse(definitions[owner]))`): an index out of
/// bounds panics in both sorts once there are two elements to compare.
pub fn sort_by_key_captured(mut owners: Vec<usize>, definitions: &[u32]) -> Vec<usize> {
    owners.sort_by_key(|owner| Reverse(definitions[*owner]));
    owners
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub slot: Option<usize>,
    pub node: u32,
}

impl Member {
    fn key(&self) -> (bool, Option<usize>, u32) {
        (self.slot.is_none(), self.slot, self.node)
    }
}

fn members(raw: &[(Option<usize>, u32)]) -> Vec<Member> {
    raw.iter().map(|&(slot, node)| Member { slot, node }).collect()
}

fn member_nodes(members: Vec<Member>) -> Vec<u32> {
    members.into_iter().map(|member| member.node).collect()
}

/// A method as the key function (chelis-ir axis_sources.rs
/// `members.sort_by_key(OrderedMember::key)`).
pub fn sort_by_key_fn_item(raw: &[(Option<usize>, u32)]) -> Vec<u32> {
    let mut members = members(raw);
    members.sort_by_key(Member::key);
    member_nodes(members)
}

/// A closure variable that reads its captures (axis_sources.rs `order_key`).
pub fn sort_by_key_local_closure(raw: &[(Option<usize>, u32)], by_node: bool) -> Vec<u32> {
    let mut members = members(raw);
    let order_key = |member: &Member| if by_node { (false, None, member.node) } else { member.key() };
    members.sort_by_key(order_key);
    member_nodes(members)
}

/// A cloned `String` in the key (chelis-types infer/declarations.rs
/// `(definition.item_index, definition.name.clone())`).
pub fn sort_by_key_cloned(mut values: Vec<(usize, String, u32)>) -> Vec<(usize, String, u32)> {
    values.sort_by_key(|v| (v.0, v.1.clone()));
    values
}

/// `sort_by_key` in place: the sorted slice is given back.
pub fn sort_by_key_in_place(values: &mut [(u32, u32)]) {
    values.sort_by_key(|v| v.1);
}

/// `sort` on a hand-written `Ord`.
pub fn reject_sort_custom_ord(mut values: Vec<Keyed>) -> Vec<Keyed> {
    values.sort();
    values
}

/// A derived `Ord` with a hand-written `PartialOrd`: the sorts call its `lt`.
#[allow(clippy::derive_ord_xor_partial_ord)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord)]
pub struct Backwards(u32);

impl PartialOrd for Backwards {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.0.cmp(&self.0))
    }
}

pub fn reject_sort_custom_lt(mut values: Vec<Backwards>) -> Vec<Backwards> {
    values.sort();
    values
}

pub fn reject_unstable_custom_lt(mut values: Vec<Backwards>) -> Vec<Backwards> {
    values.sort_unstable();
    values
}

/// A key function that updates its state (`FnMut`).
pub fn reject_by_key_stateful(mut values: Vec<u32>) -> (Vec<u32>, u32) {
    let mut calls = 0;
    values.sort_by_key(|v| {
        calls += 1;
        *v
    });
    (values, calls.min(1))
}

/// A key with a hand-written `Ord`.
pub fn reject_by_key_custom_ord(mut values: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    values.sort_by_key(|v| Keyed { key: v.0, payload: v.1 });
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

    #[test]
    fn stable_sorts() {
        assert_eq!(sort_stable_ids(&[7, 3, 5, 3]), vec![3, 3, 5, 7]);
        assert_eq!(sort_stable_ids(&[]), vec![]);
        assert_eq!(
            sort_stable_strings(vec!["b".to_string(), "ab".to_string(), "a".to_string(), "b".to_string()]),
            vec!["a".to_string(), "ab".to_string(), "b".to_string(), "b".to_string()]
        );
        assert_eq!(
            sort_stable_options(vec![
                (Some(2), Reverse(1)),
                (None, Reverse(0)),
                (Some(2), Reverse(5)),
                (None, Reverse(3)),
                (Some(1), Reverse(0)),
            ]),
            vec![
                (None, Reverse(3)),
                (None, Reverse(0)),
                (Some(1), Reverse(0)),
                (Some(2), Reverse(5)),
                (Some(2), Reverse(1)),
            ]
        );
        let product = |factors: &[u32]| Capacity::Product(factors.to_vec().into_boxed_slice());
        let quotient = Capacity::Quotient { dividend: Box::new(1), divisor: Box::new(2) };
        assert_eq!(
            sort_capacities(vec![product(&[2, 1]), Capacity::Literal(3), quotient.clone(), product(&[2]), Capacity::Literal(1)]),
            vec![Capacity::Literal(1), Capacity::Literal(3), product(&[2]), product(&[2, 1]), quotient]
        );
        let mut slots = [(2, Slot::Free), (1, Slot::Held(3, true)), (2, Slot::Held(0, false)), (1, Slot::Free)];
        sort_stable_in_place(&mut slots);
        assert_eq!(
            slots,
            [(1, Slot::Free), (1, Slot::Held(3, true)), (2, Slot::Free), (2, Slot::Held(0, false))]
        );
    }

    #[test]
    fn stable_by_key_functions() {
        assert_eq!(
            sort_by_key_field(vec![(2, 'b'), (1, 'z'), (2, 'a'), (1, 'y')]),
            vec![(1, 'z'), (1, 'y'), (2, 'b'), (2, 'a')]
        );
        assert_eq!(
            sort_by_key_fields(vec![(1, 5, 'a'), (0, 3, 'b'), (2, 3, 'c'), (0, 5, 'd'), (0, 3, 'e')]),
            vec![(0, 3, 'b'), (0, 3, 'e'), (2, 3, 'c'), (0, 5, 'd'), (1, 5, 'a')]
        );
        assert_eq!(
            sort_by_key_reverse(vec![(1, 'a'), (3, 'b'), (1, 'c'), (2, 'd'), (3, 'e')]),
            vec![(3, 'b'), (3, 'e'), (2, 'd'), (1, 'a'), (1, 'c')]
        );
        let entries = vec![("b".to_string(), 1), ("a".to_string(), 2), ("b".to_string(), 3), (String::new(), 4)];
        assert_eq!(sort_by_key_str(entries), vec![4, 2, 1, 3]);
        assert_eq!(sort_by_key_captured(vec![0, 1, 2, 3], &[5, 7, 5, 9]), vec![3, 1, 0, 2]);
        assert_eq!(sort_by_key_captured(vec![4], &[1]), vec![4]);
        assert!(std::panic::catch_unwind(|| sort_by_key_captured(vec![0, 4], &[1])).is_err());
        let raw = [(Some(2), 10), (None, 11), (Some(0), 12), (Some(2), 9), (None, 8)];
        assert_eq!(sort_by_key_fn_item(&raw), vec![12, 9, 10, 8, 11]);
        assert_eq!(sort_by_key_local_closure(&raw, true), vec![8, 9, 10, 11, 12]);
        assert_eq!(sort_by_key_local_closure(&raw, false), vec![12, 9, 10, 8, 11]);
        assert_eq!(
            sort_by_key_cloned(vec![
                (1, "b".to_string(), 0),
                (0, "z".to_string(), 1),
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 3),
            ]),
            vec![
                (0, "z".to_string(), 1),
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 0),
                (1, "b".to_string(), 3),
            ]
        );
        let mut pairs = [(1, 3), (2, 1), (3, 3), (4, 0)];
        sort_by_key_in_place(&mut pairs);
        assert_eq!(pairs, [(4, 0), (2, 1), (1, 3), (3, 3)]);
    }

    #[test]
    fn rejected_stable_are_native_behaviors() {
        assert_eq!(reject_sort_custom_ord(vec![Keyed { key: 2, payload: 0 }]).len(), 1);
        let backwards = vec![Backwards(1), Backwards(3), Backwards(2)];
        assert_eq!(reject_sort_custom_lt(backwards.clone()), vec![Backwards(3), Backwards(2), Backwards(1)]);
        assert_eq!(reject_unstable_custom_lt(backwards), vec![Backwards(3), Backwards(2), Backwards(1)]);
        assert_eq!(reject_by_key_stateful(vec![2, 1]), (vec![1, 2], 1));
        assert_eq!(reject_by_key_custom_ord(vec![(2, 0), (1, 5)]), vec![(1, 5), (2, 0)]);
    }
}
