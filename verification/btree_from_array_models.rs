//! `BTreeMap::from([(K, V); N])` (Aeneas.Std BTreeMap.lean) as chelis-ir `ownership::verify`
//! uses it (`BTreeMap::from([(unit.entry, initial)])`): `N == 0` gives the empty map, otherwise the
//! array is stably sorted by key and bulk-built, and of a run of equal keys only the last pair
//! (its key and its value) stays. The map is then read in key order.
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// As `verify_unit`: the entry block's initial owners, then a lookup and an insertion.
pub fn incoming(entry: BlockId, owners: Vec<u32>, next: BlockId) -> Vec<(u32, Vec<u32>)> {
    let initial: BTreeSet<u32> = owners.into_iter().collect();
    let mut incoming = BTreeMap::from([(entry, initial)]);
    let live = incoming[&entry].clone();
    incoming.insert(next, live);
    incoming.into_iter().map(|(id, set)| (id.0, set.into_iter().collect())).collect()
}

/// Three pairs: sorted by key; of equal keys the last pair stays.
pub fn three(a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<(u32, u32)> {
    BTreeMap::from([a, b, c]).into_iter().collect()
}

pub fn empty() -> usize {
    let map: BTreeMap<u32, u32> = BTreeMap::from([]);
    map.len()
}

/// A key compared by its first field only: the kept key is the last of its run too.
#[derive(Clone, Copy)]
pub struct Key(pub u32, pub u32);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

pub fn keys(pairs: [(Key, u32); 4]) -> Vec<(u32, u32, u32)> {
    BTreeMap::from(pairs).into_iter().map(|(k, v)| (k.0, k.1, v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_state() {
        assert_eq!(
            incoming(BlockId(4), vec![3, 1, 3], BlockId(2)),
            vec![(2, vec![1, 3]), (4, vec![1, 3])]
        );
        assert_eq!(incoming(BlockId(4), vec![], BlockId(4)), vec![(4, vec![])]);
    }

    #[test]
    fn pairs() {
        assert_eq!(three((3, 30), (1, 10), (2, 20)), vec![(1, 10), (2, 20), (3, 30)]);
        assert_eq!(three((2, 1), (1, 5), (2, 7)), vec![(1, 5), (2, 7)]);
        assert_eq!(three((5, 1), (5, 2), (5, 3)), vec![(5, 3)]);
        assert_eq!(empty(), 0);
    }

    #[test]
    fn equal_keys() {
        assert_eq!(
            keys([(Key(2, 0), 10), (Key(1, 1), 11), (Key(2, 2), 12), (Key(1, 3), 13)]),
            vec![(1, 3, 13), (2, 2, 12)]
        );
        assert_eq!(
            keys([(Key(9, 0), 1), (Key(8, 1), 2), (Key(7, 2), 3), (Key(6, 3), 4)]),
            vec![(6, 3, 4), (7, 2, 3), (8, 1, 2), (9, 0, 1)]
        );
    }
}
