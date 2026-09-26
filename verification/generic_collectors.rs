use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Equality and order observe only `key`: which of two Ord-equal elements
/// survives is visible through `tag`.
pub struct Tagged {
    pub key: u32,
    pub tag: u32,
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Tagged {}
impl PartialOrd for Tagged {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Tagged {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Generic collectors: the iterator dictionary is a trait clause, as in
/// `chelis_unord`'s `FromIterator`/`Extend` wrappers.
pub fn map_from<T: IntoIterator<Item = (Tagged, u32)>>(iter: T) -> BTreeMap<Tagged, u32> {
    iter.into_iter().collect()
}

pub fn set_from<T: IntoIterator<Item = Tagged>>(iter: T) -> BTreeSet<Tagged> {
    iter.into_iter().collect()
}

pub fn map_extend<T: IntoIterator<Item = (Tagged, u32)>>(map: &mut BTreeMap<Tagged, u32>, iter: T) {
    map.extend(iter);
}

pub fn set_extend<T: IntoIterator<Item = Tagged>>(set: &mut BTreeSet<Tagged>, iter: T) {
    set.extend(iter);
}

pub fn entries(pairs: &[(u32, u32, u32)]) -> Vec<(Tagged, u32)> {
    let mut items = Vec::new();
    for &(key, tag, value) in pairs {
        items.push((Tagged { key, tag }, value));
    }
    items
}

pub fn elements(pairs: &[(u32, u32)]) -> Vec<Tagged> {
    let mut items = Vec::new();
    for &(key, tag) in pairs {
        items.push(Tagged { key, tag });
    }
    items
}

const MAP_INPUT: [(u32, u32, u32); 5] = [(2, 20, 200), (1, 10, 100), (2, 21, 201), (3, 30, 300), (1, 11, 101)];
const SET_INPUT: [(u32, u32); 5] = [(2, 20), (1, 10), (2, 21), (3, 30), (1, 11)];

/// Concrete instantiations: vec::IntoIter discharges the deferred obligations.
pub fn map_collected() -> BTreeMap<Tagged, u32> {
    map_from(entries(&MAP_INPUT))
}

pub fn set_collected() -> BTreeSet<Tagged> {
    set_from(elements(&SET_INPUT))
}

pub fn map_extended() -> BTreeMap<Tagged, u32> {
    let mut map = BTreeMap::new();
    map_extend(&mut map, entries(&MAP_INPUT));
    map
}

pub fn set_extended() -> BTreeSet<Tagged> {
    let mut set = BTreeSet::new();
    set_extend(&mut set, elements(&SET_INPUT));
    set
}

/// A crate iterator that defines only `next`: every other method is the
/// trait's default, so its `next` loop is the reviewed behavior of `collect`.
pub struct Countdown(pub u32);

impl Iterator for Countdown {
    type Item = Tagged;
    fn next(&mut self) -> Option<Tagged> {
        if self.0 == 0 {
            return None;
        }
        self.0 -= 1;
        Some(Tagged { key: self.0 / 2, tag: self.0 })
    }
}

pub fn custom_source() -> BTreeSet<Tagged> {
    set_from(Countdown(4))
}

pub fn custom_vec() -> Vec<Tagged> {
    Countdown(4).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_view(map: &BTreeMap<Tagged, u32>) -> Vec<(u32, u32, u32)> {
        map.iter().map(|(k, v)| (k.key, k.tag, *v)).collect()
    }

    fn set_view(set: &BTreeSet<Tagged>) -> Vec<(u32, u32)> {
        set.iter().map(|k| (k.key, k.tag)).collect()
    }

    #[test]
    fn collect_keeps_the_last_equal_entry() {
        assert_eq!(map_view(&map_collected()), [(1, 11, 101), (2, 21, 201), (3, 30, 300)]);
        assert_eq!(set_view(&set_collected()), [(1, 11), (2, 21), (3, 30)]);
    }

    #[test]
    fn extend_keeps_the_first_key_and_the_last_value() {
        assert_eq!(map_view(&map_extended()), [(1, 10, 101), (2, 20, 201), (3, 30, 300)]);
        assert_eq!(set_view(&set_extended()), [(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn next_only_source_keeps_the_last_equal_element() {
        assert_eq!(set_view(&custom_source()), [(0, 0), (1, 2)]);
        let items: Vec<_> = custom_vec().iter().map(|k| (k.key, k.tag)).collect();
        assert_eq!(items, [(1, 3), (1, 2), (0, 1), (0, 0)]);
    }
}
