//! `PersistentMap` and `PersistentSet` behave as ordinary ordered
//! collections, and a copy never observes a change made through another.
//!
//! The model is a `BTreeMap` that is cloned wherever the persistent map is.
//! A deterministic generator drives inserts, overwrites, removals of present
//! and absent keys, in-place updates and clears across many interleaved
//! snapshots, and every snapshot must still equal its model at the end.

use std::collections::{BTreeMap, BTreeSet};

use chelis_unord::{PersistentMap, PersistentSet};
#[cfg(feature = "serde")]
use chelis_unord::{UnordMap, UnordSet};

/// A fixed xorshift generator, so every run drives the same operations.
struct Generator(u64);

impl Generator {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

fn contents(map: &PersistentMap<u64, u64>) -> Vec<(u64, u64)> {
    map.to_sorted()
        .into_iter()
        .map(|(key, value)| (*key, *value))
        .collect()
}

fn model_contents(model: &BTreeMap<u64, u64>) -> Vec<(u64, u64)> {
    model.iter().map(|(key, value)| (*key, *value)).collect()
}

#[test]
fn every_snapshot_keeps_the_contents_it_was_taken_with() {
    for seed in 1..=8u64 {
        let mut generator = Generator(0x9e37_79b9_7f4a_7c15 ^ seed);
        let key_space = [4, 64, 1024][usize::try_from(seed % 3).unwrap()];
        let mut map = PersistentMap::new();
        let mut model = BTreeMap::new();
        let mut snapshots = Vec::new();
        for step in 0..4000u64 {
            let key = generator.below(key_space);
            match generator.below(16) {
                0..=6 => {
                    assert_eq!(map.insert(key, step), model.insert(key, step));
                }
                7..=9 => {
                    assert_eq!(map.remove(&key), model.remove(&key));
                }
                10 => {
                    if let Some(value) = map.get_mut(&key) {
                        *value += 1;
                    }
                    if let Some(value) = model.get_mut(&key) {
                        *value += 1;
                    }
                }
                11 => {
                    *map.get_or_insert_with(key, || step) += 7;
                    *model.entry(key).or_insert(step) += 7;
                }
                12 if generator.below(64) == 0 => {
                    map.clear();
                    model.clear();
                }
                _ => snapshots.push((map.clone(), model.clone())),
            }
            assert_eq!(map.get(&key), model.get(&key));
            assert_eq!(map.contains_key(&key), model.contains_key(&key));
            assert_eq!(map.len(), model.len());
            assert_eq!(map.is_empty(), model.is_empty());
        }
        assert_eq!(contents(&map), model_contents(&model));
        for (snapshot, model) in &snapshots {
            assert_eq!(contents(snapshot), model_contents(model));
            assert_eq!(snapshot.len(), model.len());
        }
    }
}

#[test]
fn a_set_tracks_an_ordered_set_across_snapshots() {
    let mut generator = Generator(0x2545_f491_4f6c_dd1d);
    let mut set = PersistentSet::new();
    let mut model = BTreeSet::new();
    let mut snapshots = Vec::new();
    for _ in 0..3000 {
        let value = generator.below(256);
        match generator.below(4) {
            0 | 1 => assert_eq!(set.insert(value), model.insert(value)),
            2 => assert_eq!(set.remove(&value), model.remove(&value)),
            _ => snapshots.push((set.clone(), model.clone())),
        }
        assert_eq!(set.contains(&value), model.contains(&value));
        assert_eq!(set.len(), model.len());
    }
    for (snapshot, model) in snapshots.iter().chain([(set, model)].iter()) {
        let sorted = snapshot
            .to_sorted()
            .into_iter()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(sorted, model.iter().copied().collect::<Vec<_>>());
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialization_matches_the_unordered_collections() {
    let pairs = [(3u64, "c"), (1, "a"), (2, "b")];
    let persistent: PersistentMap<u64, &str> = pairs.into_iter().collect();
    let unordered: UnordMap<u64, &str> = pairs.into_iter().collect();
    let bytes = serde_json::to_string(&persistent).unwrap();
    assert_eq!(bytes, serde_json::to_string(&unordered).unwrap());
    let decoded: PersistentMap<u64, String> = serde_json::from_str(&bytes).unwrap();
    assert_eq!(
        decoded
            .to_sorted()
            .into_iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect::<Vec<_>>(),
        vec![
            (1, "a".to_owned()),
            (2, "b".to_owned()),
            (3, "c".to_owned())
        ]
    );
    assert!(
        serde_json::from_str::<PersistentMap<u64, String>>(r#"[[1,"a"],[1,"b"]]"#).is_err(),
        "a duplicate key is rejected, as UnordMap rejects it"
    );
    let set: PersistentSet<u64> = [5, 1, 3].into_iter().collect();
    let unordered_set: UnordSet<u64> = [5, 1, 3].into_iter().collect();
    assert_eq!(set.to_sorted(), unordered_set.to_sorted());
}

#[test]
fn debug_reports_the_length_only() {
    let map: PersistentMap<&str, i32> = [("secret", 1)].into_iter().collect();
    assert_eq!(format!("{map:?}"), "PersistentMap { len: 1 }");
    let set: PersistentSet<&str> = ["secret"].into_iter().collect();
    assert_eq!(format!("{set:?}"), "PersistentSet { len: 1 }");
}

#[test]
fn last_where_finds_the_end_of_a_true_prefix() {
    // Every size up to 300 and every prefix length: the descent answers what a
    // scan of the sorted entries answers.
    for len in 0..300u64 {
        let map: PersistentMap<u64, u64> = (0..len).map(|key| (key * 3, key)).collect();
        for cut in 0..=len {
            let found = map
                .last_where(|key, _| *key < cut * 3)
                .map(|(key, value)| (*key, *value));
            let expected = cut.checked_sub(1).map(|last| (last * 3, last));
            assert_eq!(found, expected, "len {len}, prefix {cut}");
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn a_set_serializes_as_the_unordered_set_does() {
    // Every snapshot of a random insert and remove history writes the JSON
    // the `UnordSet` with the same contents writes, and reads back to the
    // same contents.
    let mut generator = Generator(0x1405_7b7e_f767_814f);
    let mut set = PersistentSet::new();
    let mut model = UnordSet::new();
    for _ in 0..2000 {
        let value = generator.below(128);
        if generator.below(3) == 0 {
            set.remove(&value);
            model.remove(&value);
        } else {
            set.insert(value);
            model.insert(value);
        }
        let bytes = serde_json::to_string(&set).unwrap();
        assert_eq!(bytes, serde_json::to_string(&model).unwrap());
        let decoded: PersistentSet<u64> = serde_json::from_str(&bytes).unwrap();
        assert_eq!(decoded.to_sorted(), set.to_sorted());
    }
    assert!(
        serde_json::from_str::<PersistentSet<u64>>("[1,1]").is_err(),
        "a duplicate element is rejected, as UnordSet rejects it"
    );
    let unsorted: PersistentSet<u64> = serde_json::from_str("[3,1,2]").unwrap();
    assert_eq!(unsorted.to_sorted(), vec![&1, &2, &3]);
    assert_eq!(serde_json::to_string(&unsorted).unwrap(), "[1,2,3]");
}
