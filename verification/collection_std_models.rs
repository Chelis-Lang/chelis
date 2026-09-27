//! `BTreeMap::get_key_value`, `BTreeMap::clear`, `Default for BTreeMap`,
//! `BTreeSet::len`, `BTreeSet::clear`, `Clone` and `PartialEq` for `BTreeSet`,
//! `Vec::retain`, `Vec::split_off` and `Default for Box` (Aeneas.Std BTreeMap.lean,
//! BTreeSet.lean, Vec.lean, Core/Default.lean), with the collections and closure
//! captures given back through mutable borrows.
use std::collections::{BTreeMap, BTreeSet};

/// `BTreeMap::get_key_value`: the stored key and its value.
pub fn entry_of(map: &BTreeMap<u32, String>, key: u32) -> Option<(u32, String)> {
    match map.get_key_value(&key) {
        Some((stored, value)) => Some((*stored, value.clone())),
        None => None,
    }
}

/// `BTreeMap::clear` on a borrowed map: the length before, and the given-back map.
pub fn cleared(map: &mut BTreeMap<u32, u32>) -> usize {
    let before = map.len();
    map.clear();
    before
}

/// `BTreeSet::len` then `BTreeSet::clear` on a borrowed set.
pub fn set_len_then_clear(set: &mut BTreeSet<u32>) -> usize {
    let len = set.len();
    set.clear();
    len
}

/// `Vec::retain` with a pure predicate (chelis-ir `retain(|x| ...)`).
pub fn keep_even(mut values: Vec<u32>) -> Vec<u32> {
    values.retain(|value| *value % 2 == 0);
    values
}

/// `Vec::split_off` on a borrowed vector: the tail, and the given-back head.
pub fn split_tail(values: &mut Vec<u32>, at: usize) -> Vec<u32> {
    values.split_off(at)
}

/// `Default for BTreeMap`, then an insert.
pub fn default_map(key: u32) -> BTreeMap<u32, u32> {
    let mut map: BTreeMap<u32, u32> = Default::default();
    map.insert(key, key + 1);
    map
}

/// `Clone` and `PartialEq` for `BTreeSet`.
pub fn cloned_set_equal(set: &BTreeSet<u32>, other: &BTreeSet<u32>) -> (BTreeSet<u32>, bool) {
    (set.clone(), set == other)
}

/// `Default for Box<T>`.
pub fn default_box() -> Box<Option<u32>> {
    Default::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_and_clears() {
        let map = BTreeMap::from([(1, "a".to_string()), (3, "c".to_string())]);
        assert_eq!(entry_of(&map, 3), Some((3, "c".to_string())));
        assert_eq!(entry_of(&map, 2), None);
        assert_eq!(entry_of(&BTreeMap::new(), 1), None);
        let mut numbers = BTreeMap::from([(1, 10), (2, 20)]);
        assert_eq!(cleared(&mut numbers), 2);
        assert!(numbers.is_empty());
        let mut empty = BTreeMap::new();
        assert_eq!(cleared(&mut empty), 0);
        assert!(empty.is_empty());
        let mut set = BTreeSet::from([4, 7]);
        assert_eq!(set_len_then_clear(&mut set), 2);
        assert!(set.is_empty());
    }

    #[test]
    fn retained() {
        assert_eq!(keep_even(vec![1, 2, 3, 4, 6, 7]), vec![2, 4, 6]);
        assert_eq!(keep_even(vec![]), vec![]);
    }

    #[test]
    fn splits_defaults_and_set_equality() {
        let mut values = vec![1, 2, 3, 4];
        assert_eq!(split_tail(&mut values, 1), vec![2, 3, 4]);
        assert_eq!(values, vec![1]);
        let mut whole = vec![1, 2];
        assert_eq!(split_tail(&mut whole, 2), vec![]);
        assert_eq!(whole, vec![1, 2]);
        assert!(std::panic::catch_unwind(|| split_tail(&mut vec![1], 2)).is_err());
        assert_eq!(default_map(3), BTreeMap::from([(3, 4)]));
        let set = BTreeSet::from([1, 5]);
        assert_eq!(cloned_set_equal(&set, &BTreeSet::from([1, 5])), (set.clone(), true));
        assert_eq!(cloned_set_equal(&set, &BTreeSet::from([1, 6])), (set.clone(), false));
        assert_eq!(cloned_set_equal(&BTreeSet::from([1]), &set), (BTreeSet::from([1]), false));
        assert_eq!(default_box(), Box::new(None));
    }
}
