//! Collection methods and conversions of the production crates (Aeneas.Std BTreeSet, BTreeMap,
//! Vec, Core/Borrow): `BTreeSet::retain` (chelis-types `infer::declarations`),
//! `BTreeSet::from([..])` (chelis-types `infer::vmap_extent`), `BTreeMap::into_values`
//! (chelis-types `builtins`), `<[T; N]>::try_from(Vec<T>)` (chelis-deep), and `ToOwned for [T]`
//! with `Borrow<[T]> for Vec<T>` (chelis-ir `host`).

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Ordered by `key` only: which of equal elements a set keeps is told by `tag`.
#[derive(Debug, Clone, Copy)]
pub struct Keyed {
    pub key: u32,
    pub tag: u32,
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

/// A value whose `clone` and `clone_from` are told apart by `clones` and `reuses`.
#[derive(Debug, PartialEq)]
pub struct Tracked {
    pub id: u32,
    pub clones: u32,
    pub reuses: u32,
}

impl Clone for Tracked {
    fn clone(&self) -> Self {
        Tracked { id: self.id, clones: self.clones + 1, reuses: self.reuses }
    }

    fn clone_from(&mut self, source: &Self) {
        self.id = source.id;
        self.clones = source.clones;
        self.reuses += 1;
    }
}

/// As chelis-types `infer::declarations`: drop the references to a vertex.
pub fn drop_vertex(references: &mut BTreeSet<u32>, vertex: u32) {
    references.retain(|reference| *reference != vertex);
}

/// `retain` with a closure counting its calls and summing what it sees through captured borrows.
pub fn keep_even(values: &mut BTreeSet<u32>, calls: &mut u32, seen: &mut Vec<u32>) {
    values.retain(|value| {
        *calls += 1;
        seen.push(*value);
        *value % 2 == 0
    });
}

/// As chelis-types `infer::vmap_extent`: a set from an array, as keys and tags in order.
pub fn set_of(items: [Keyed; 4]) -> Vec<(u32, u32)> {
    BTreeSet::from(items).iter().map(|k| (k.key, k.tag)).collect()
}

/// `BTreeSet::from` of an empty array.
pub fn empty_set() -> usize {
    let items: [u32; 0] = [];
    BTreeSet::from(items).len()
}

/// As chelis-types `builtins`: the values of a registry in key order.
pub fn registry_values(entries: &[(u32, u64)]) -> Vec<u64> {
    let mut registry = BTreeMap::new();
    for (key, value) in entries {
        registry.insert(*key, *value);
    }
    registry.into_values().collect()
}

/// `IntoValues::next` one value at a time.
pub fn first_two_values(entries: &[(u32, u64)]) -> (Option<u64>, Option<u64>) {
    let registry: BTreeMap<u32, u64> = entries.iter().copied().collect();
    let mut values = registry.into_values();
    let first = values.next();
    (first, values.next())
}

/// As chelis-deep: a vector of exactly three elements as an array, else the vector back.
pub fn triple(values: Vec<u32>) -> Result<[u32; 3], Vec<u32>> {
    <[u32; 3]>::try_from(values)
}

/// `ToOwned::to_owned` of a slice: each element cloned.
pub fn owned(items: &[Tracked]) -> Vec<Tracked> {
    items.to_owned()
}

/// `ToOwned::clone_into` of a slice into a caller's vector: `clone_from` on the kept elements,
/// clones of the rest.
pub fn refresh(items: &[Tracked], target: &mut Vec<Tracked>) {
    items.clone_into(target);
}

fn total<B: Borrow<[usize]>>(values: &B) -> usize {
    values.borrow().iter().sum()
}

/// `Borrow<[T]> for Vec<T>` through a generic bound.
pub fn borrowed_total(values: &Vec<usize>) -> usize {
    total(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracked(id: u32) -> Tracked {
        Tracked { id, clones: 0, reuses: 0 }
    }

    #[test]
    fn retains() {
        let mut references: BTreeSet<u32> = [4, 1, 9, 7].into_iter().collect();
        drop_vertex(&mut references, 9);
        assert_eq!(references.into_iter().collect::<Vec<_>>(), vec![1, 4, 7]);
        let mut values: BTreeSet<u32> = [5, 2, 8, 3, 6].into_iter().collect();
        let mut calls = 1;
        let mut seen = vec![0];
        keep_even(&mut values, &mut calls, &mut seen);
        assert_eq!(values.into_iter().collect::<Vec<_>>(), vec![2, 6, 8]);
        assert_eq!(calls, 6);
        assert_eq!(seen, vec![0, 2, 3, 5, 6, 8]);
    }

    #[test]
    fn from_arrays() {
        let items = [
            Keyed { key: 3, tag: 0 },
            Keyed { key: 1, tag: 1 },
            Keyed { key: 3, tag: 2 },
            Keyed { key: 1, tag: 3 },
        ];
        assert_eq!(set_of(items), vec![(1, 3), (3, 2)]);
        assert_eq!(empty_set(), 0);
    }

    #[test]
    fn values() {
        assert_eq!(registry_values(&[(3, 30), (1, 10), (2, 20), (1, 11)]), vec![11, 20, 30]);
        assert_eq!(registry_values(&[]), Vec::<u64>::new());
        assert_eq!(first_two_values(&[(5, 50), (4, 40)]), (Some(40), Some(50)));
        assert_eq!(first_two_values(&[(5, 50)]), (Some(50), None));
    }

    #[test]
    fn conversions() {
        assert_eq!(triple(vec![1, 2, 3]), Ok([1, 2, 3]));
        assert_eq!(triple(vec![1, 2]), Err(vec![1, 2]));
        assert_eq!(triple(vec![1, 2, 3, 4]), Err(vec![1, 2, 3, 4]));
        assert_eq!(owned(&[tracked(1), tracked(2)]), vec![
            Tracked { id: 1, clones: 1, reuses: 0 },
            Tracked { id: 2, clones: 1, reuses: 0 }
        ]);
        let mut target = vec![tracked(7), tracked(8), tracked(9)];
        refresh(&[tracked(1), tracked(2)], &mut target);
        assert_eq!(target, vec![
            Tracked { id: 1, clones: 0, reuses: 1 },
            Tracked { id: 2, clones: 0, reuses: 1 }
        ]);
        let mut target = vec![tracked(7)];
        refresh(&[tracked(1), tracked(2), tracked(3)], &mut target);
        assert_eq!(target, vec![
            Tracked { id: 1, clones: 0, reuses: 1 },
            Tracked { id: 2, clones: 1, reuses: 0 },
            Tracked { id: 3, clones: 1, reuses: 0 }
        ]);
        assert_eq!(borrowed_total(&vec![4, 5, 6]), 15);
    }
}
