use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

// Identity is observable but deliberately not part of ordering/equality.
pub struct Key {
    pub rank: u32,
    pub identity: u32,
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
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
        self.rank.cmp(&other.rank)
    }
}
impl Borrow<u32> for Key {
    fn borrow(&self) -> &u32 {
        &self.rank
    }
}

fn key(rank: u32, identity: u32) -> Key {
    Key { rank, identity }
}

fn populated() -> BTreeMap<Key, u32> {
    let mut map = BTreeMap::new();
    map.insert(key(3, 30), 300);
    map.insert(key(1, 10), 100);
    map.insert(key(2, 20), 200);
    map
}

fn snapshot(map: &BTreeMap<Key, u32>) -> Vec<(u32, u32, u32)> {
    let mut result = Vec::new();
    let mut iterator = map.iter();
    while let Some((key, value)) = iterator.next() {
        result.push((key.rank, key.identity, *value));
    }
    result
}

pub fn ordered_insert() -> Vec<(u32, u32, u32)> {
    snapshot(&populated())
}

pub fn equivalent_replacement() -> (Option<u32>, Vec<(u32, u32, u32)>) {
    let mut map = populated();
    let previous = map.insert(key(2, 99), 250);
    (previous, snapshot(&map))
}

pub fn occupied_restoration() -> Vec<(u32, u32, u32)> {
    let mut map = populated();
    match map.entry(key(2, 99)) {
        Entry::Occupied(entry) => {
            let value = entry.into_mut();
            *value += 7;
            *value += 4;
        }
        Entry::Vacant(entry) => {
            *entry.insert(0) = 999;
        }
    }
    snapshot(&map)
}

pub fn vacant_restoration() -> Vec<(u32, u32, u32)> {
    let mut map = populated();
    map.remove(&2u32);
    match map.entry(key(2, 40)) {
        Entry::Vacant(entry) => {
            let value = entry.insert(400);
            *value += 7;
            *value += 4;
        }
        Entry::Occupied(entry) => {
            *entry.into_mut() = 999;
        }
    }
    // Also exercise insertion past the last key.
    match map.entry(key(4, 50)) {
        Entry::Vacant(entry) => {
            *entry.insert(500) = 511;
        }
        Entry::Occupied(entry) => {
            *entry.into_mut() = 999;
        }
    }
    snapshot(&map)
}

pub fn dropped_entries() -> Vec<(u32, u32, u32)> {
    let mut map = populated();
    {
        let _entry = map.entry(key(2, 99));
    }
    {
        let _entry = map.entry(key(0, 40));
    }
    snapshot(&map)
}

pub fn remove_and_lookup() -> (
    Option<u32>,
    Option<u32>,
    Option<u32>,
    bool,
    bool,
    Vec<(u32, u32, u32)>,
) {
    let mut map = populated();
    let found = match map.get(&2u32) {
        Some(value) => Some(*value),
        None => None,
    };
    let removed = map.remove(&2u32);
    let absent = map.remove(&9u32);
    let missing = map.contains_key(&2u32);
    let present = map.contains_key(&3u32);
    (found, removed, absent, missing, present, snapshot(&map))
}

pub fn missing_lookup() -> Option<u32> {
    let map = populated();
    match map.get(&0u32) {
        Some(value) => Some(*value),
        None => None,
    }
}

pub fn shared_exhaustion() -> (bool, bool, bool, bool, bool) {
    let empty: BTreeMap<u32, u32> = BTreeMap::new();
    let empty_map = empty.is_empty();
    let mut iterator = empty.iter();
    let first = iterator.next().is_none();
    let second = iterator.next().is_none();
    let mut one = BTreeMap::new();
    one.insert(1u32, 10u32);
    let mut iterator = one.iter();
    let yielded = iterator.next().is_some();
    let exhausted = iterator.next().is_none() && iterator.next().is_none();
    (empty_map, first, second, yielded, exhausted)
}

pub fn owned_exhaustion() -> (bool, bool, Option<(u32, u32)>, bool, bool) {
    let empty: BTreeMap<u32, u32> = BTreeMap::new();
    let mut iterator = empty.into_iter();
    let first = iterator.next().is_none();
    let second = iterator.next().is_none();
    let mut one = BTreeMap::new();
    one.insert(1u32, 10u32);
    let mut iterator = one.into_iter();
    let yielded = iterator.next();
    let exhausted = iterator.next().is_none();
    let exhausted_again = iterator.next().is_none();
    (first, second, yielded, exhausted, exhausted_again)
}

pub fn owned_collect() -> Vec<(u32, u32)> {
    let mut map = BTreeMap::new();
    map.insert(3u32, 30u32);
    map.insert(1u32, 10u32);
    map.insert(2u32, 20u32);
    map.into_iter().collect()
}

pub fn shared_collect() -> Vec<(u32, u32)> {
    let mut map = BTreeMap::new();
    map.insert(3u32, 30u32);
    map.insert(1u32, 10u32);
    map.insert(2u32, 20u32);
    let references: Vec<(&u32, &u32)> = map.iter().collect();
    let mut output = Vec::new();
    for (key, value) in references {
        output.push((*key, *value));
    }
    output
}

pub fn duplicate_from_iterator() -> Vec<(u32, u32, u32)> {
    let mut items = Vec::new();
    items.push((key(2, 20), 200));
    items.push((key(1, 10), 100));
    items.push((key(2, 99), 250));
    items.push((key(1, 88), 150));
    items.push((key(3, 30), 300));
    let map: BTreeMap<Key, u32> = BTreeMap::from_iter(items);
    snapshot(&map)
}

pub fn empty_from_iterator() -> bool {
    let items: Vec<(u32, u32)> = Vec::new();
    let map: BTreeMap<u32, u32> = BTreeMap::from_iter(items);
    map.is_empty()
}

pub struct WrappedMap<K, V> {
    inner: BTreeMap<K, V>,
}

impl<K: Ord, V> FromIterator<(K, V)> for WrappedMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}

pub struct CountedSource<'a> {
    polls: &'a mut u32,
    length: u32,
}

pub struct CountedItems<'a> {
    polls: &'a mut u32,
    remaining: u32,
}

impl<'a> IntoIterator for CountedSource<'a> {
    type Item = (u32, u32);
    type IntoIter = CountedItems<'a>;

    fn into_iter(self) -> Self::IntoIter {
        *self.polls += 10;
        CountedItems {
            polls: self.polls,
            remaining: self.length,
        }
    }
}

impl Iterator for CountedItems<'_> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        *self.polls += 1;
        if self.remaining == 0 {
            None
        } else {
            let item = self.remaining;
            self.remaining -= 1;
            Some((item, item * 10))
        }
    }
}

pub fn delegated_collection_state(length: u32) -> (Vec<(u32, u32)>, u32) {
    let mut polls = 0;
    let source = CountedSource {
        polls: &mut polls,
        length,
    };
    let map: WrappedMap<u32, u32> = WrappedMap::from_iter(source);
    let entries = map.inner.into_iter().collect();
    (entries, polls)
}

pub enum WrappedEntry<'a, K, V> {
    Occupied(WrappedOccupied<'a, K, V>),
    Vacant(WrappedVacant<'a, K, V>),
}

pub struct WrappedOccupied<'a, K, V>(std::collections::btree_map::OccupiedEntry<'a, K, V>);
pub struct WrappedVacant<'a, K, V>(std::collections::btree_map::VacantEntry<'a, K, V>);

impl<K: Ord, V> WrappedMap<K, V> {
    pub fn entry(&mut self, key: K) -> WrappedEntry<'_, K, V> {
        match self.inner.entry(key) {
            Entry::Occupied(entry) => WrappedEntry::Occupied(WrappedOccupied(entry)),
            Entry::Vacant(entry) => WrappedEntry::Vacant(WrappedVacant(entry)),
        }
    }
}

impl<'a, K: Ord, V> WrappedOccupied<'a, K, V> {
    pub fn into_mut(self) -> &'a mut V {
        self.0.into_mut()
    }
}

impl<'a, K: Ord, V> WrappedVacant<'a, K, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        self.0.insert(value)
    }
}

impl<'a, K: Ord, V> WrappedEntry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(default()),
        }
    }
}

pub fn wrapped_entry_restoration() -> (u32, Vec<(u32, u32, u32)>) {
    let mut map = WrappedMap { inner: populated() };
    *map.entry(key(2, 99)).or_insert(900) += 7;
    *map.entry(key(4, 40)).or_insert(400) += 11;
    let mut calls = 0;
    *map.entry(key(2, 88)).or_insert_with(|| {
        calls += 1;
        800
    }) += 4;
    *map.entry(key(5, 50)).or_insert_with(|| {
        calls += 1;
        500
    }) += 13;
    (calls, snapshot(&map.inner))
}

pub fn zipped_iterator_callbacks() -> (usize, u32, usize, u32) {
    let mut map = BTreeMap::new();
    map.insert(1_u32, 10_u32);
    map.insert(2, 20);
    let mut shared_calls = 0;
    let shared_count = map
        .iter()
        .map(|(key, value)| {
            shared_calls += 1;
            key + value
        })
        .zip([0_u32])
        .count();
    let mut owned_calls = 0;
    let owned_count = map
        .into_iter()
        .map(|(key, value)| {
            owned_calls += 1;
            key + value
        })
        .zip([0_u32])
        .count();
    (shared_count, shared_calls, owned_count, owned_calls)
}

// `chelis-ir` ownership/last_use.rs: the keys of a map, mapped and collected
// into a map.
pub fn keys_collected() -> Vec<(u32, u32)> {
    let map = populated();
    let rekeyed = map
        .keys()
        .map(|key| (key.rank, key.identity))
        .collect::<BTreeMap<u32, u32>>();
    rekeyed.into_iter().collect()
}

/// `for (key, value) in &map`: IntoIterator for &BTreeMap is `self.iter()`,
/// in key order, and leaves the map intact.
pub fn borrowed_for_loop() -> (Vec<(u32, u32, u32)>, bool) {
    let map = populated();
    let mut visited = Vec::new();
    for (key, value) in &map {
        visited.push((key.rank, key.identity, *value));
    }
    (visited, map.is_empty())
}

/// The same impl as the IntoIterator dictionary of Vec's FromIterator.
pub fn borrowed_generic_total() -> u32 {
    let map = populated();
    let entries: Vec<(&Key, &u32)> = Vec::from_iter(&map);
    let mut total = 0;
    for (key, value) in entries {
        total += key.rank * *value;
    }
    total
}

fn weighted_total<'a, I: IntoIterator<Item = (&'a Key, &'a u32)>>(entries: I) -> u32 {
    let mut total = 0;
    for (key, value) in entries {
        total += key.rank * *value;
    }
    total
}

fn entry_count<'a, K: 'a, V: 'a, I: IntoIterator<Item = (&'a K, &'a V)>>(entries: I) -> u32 {
    let mut count = 0;
    for (_key, _value) in entries {
        count += 1;
    }
    count
}

/// `&map` as the IntoIterator dictionary of loops in generic functions.
pub fn generic_loop_totals() -> (u32, u32, bool) {
    let map = populated();
    (weighted_total(&map), entry_count(&map), map.is_empty())
}

pub fn keys_and_values() -> (Vec<u32>, Vec<u32>, (usize, Option<usize>), Option<u32>, usize) {
    let map = populated();
    let values: Vec<u32> = map.values().copied().collect();
    let mut keys = map.keys();
    let first = keys.next().map(|key| key.identity);
    let hint = keys.size_hint();
    let mut ranks = Vec::new();
    for key in keys {
        ranks.push(key.rank);
    }
    (ranks, values, hint, first, map.values().count())
}

// `chelis-ir` ownership/mod.rs compares owner maps. Keys compare through their
// own `PartialEq`, which ignores the identity.
pub fn maps_equal() -> (bool, bool, bool, bool) {
    let map = populated();
    let mut same = BTreeMap::new();
    same.insert(key(1, 77), 100);
    same.insert(key(2, 20), 200);
    same.insert(key(3, 30), 300);
    let mut changed = populated();
    changed.insert(key(2, 20), 201);
    let mut shorter = populated();
    shorter.remove(&3);
    (map == same, map != changed, map == shorter, map != same)
}

/// std's `Entry` methods, translated from their bodies: `or_default`,
/// `or_insert_with` (the closure, whose captured counter is given back, runs only
/// for a vacant entry), and `insert_entry`, which keeps an occupied entry's stored
/// key and gives back an entry whose `get_mut` writes reach the map; removing
/// through the returned entry takes out the inserted pair with its key. The
/// modeled `Entry::key` is the stored key of an occupied entry.
pub fn std_entry_methods() -> (u32, u32, (u32, u32), Vec<(u32, u32, u32)>) {
    let mut map = populated();
    *map.entry(key(2, 99)).or_default() += 5;
    *map.entry(key(4, 40)).or_default() += 4;
    let mut calls = 0;
    *map.entry(key(5, 50)).or_insert_with(|| {
        calls += 1;
        500
    }) += 1;
    map.entry(key(1, 77)).or_insert_with(|| {
        calls += 10;
        0
    });
    let stored = map.entry(key(3, 99)).key().identity;
    let mut inserted = map.entry(key(1, 11)).insert_entry(111);
    *inserted.get_mut() += 1;
    let (removed, value) = map.entry(key(6, 60)).insert_entry(600).remove_entry();
    (calls, stored, (removed.identity, value), snapshot(&map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_observations() {
        let original = vec![(1, 10, 100), (2, 20, 200), (3, 30, 300)];
        assert_eq!(ordered_insert(), original);
        assert_eq!(
            equivalent_replacement(),
            (Some(200), vec![(1, 10, 100), (2, 20, 250), (3, 30, 300)])
        );
        assert_eq!(
            occupied_restoration(),
            vec![(1, 10, 100), (2, 20, 211), (3, 30, 300)]
        );
        assert_eq!(
            vacant_restoration(),
            vec![(1, 10, 100), (2, 40, 411), (3, 30, 300), (4, 50, 511)]
        );
        assert_eq!(dropped_entries(), original);
        assert_eq!(
            remove_and_lookup(),
            (
                Some(200),
                Some(200),
                None,
                false,
                true,
                vec![(1, 10, 100), (3, 30, 300)]
            )
        );
        assert_eq!(missing_lookup(), None);
        assert_eq!(shared_exhaustion(), (true, true, true, true, true));
        assert_eq!(owned_exhaustion(), (true, true, Some((1, 10)), true, true));
        assert_eq!(owned_collect(), vec![(1, 10), (2, 20), (3, 30)]);
        assert_eq!(shared_collect(), vec![(1, 10), (2, 20), (3, 30)]);
        // February's stable bulk-build keeps the last KEY as well as its value.
        assert_eq!(
            duplicate_from_iterator(),
            vec![(1, 88, 150), (2, 99, 250), (3, 30, 300)]
        );
        assert!(empty_from_iterator());
        assert_eq!(
            wrapped_entry_restoration(),
            (
                1,
                vec![
                    (1, 10, 100),
                    (2, 20, 211),
                    (3, 30, 300),
                    (4, 40, 411),
                    (5, 50, 513)
                ]
            )
        );
        assert_eq!(keys_collected(), vec![(1, 10), (2, 20), (3, 30)]);
        assert_eq!(
            keys_and_values(),
            (vec![2, 3], vec![100, 200, 300], (2, Some(2)), Some(10), 3)
        );
        assert_eq!(maps_equal(), (true, true, false, false));
        assert_eq!(
            std_entry_methods(),
            (
                1,
                30,
                (60, 600),
                vec![
                    (1, 10, 112),
                    (2, 20, 205),
                    (3, 30, 300),
                    (4, 40, 4),
                    (5, 50, 501)
                ]
            )
        );
        assert_eq!(zipped_iterator_callbacks(), (1, 1, 1, 1));
        assert_eq!(delegated_collection_state(2), (vec![(1, 10), (2, 20)], 13));
        assert_eq!(delegated_collection_state(0), (vec![], 11));
        assert_eq!(
            borrowed_for_loop(),
            (vec![(1, 10, 100), (2, 20, 200), (3, 30, 300)], false)
        );
        assert_eq!(borrowed_generic_total(), 1400);
        assert_eq!(generic_loop_totals(), (1400, 3, false));
        let string = String::from("é");
        let borrowed: &str = Borrow::borrow(&string);
        assert_eq!(borrowed.as_bytes(), &[195, 169]);
    }
}
