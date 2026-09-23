pub struct Bundle<T> {
    pub payload: T,
    pub stamp: u32,
}

pub struct State {
    pub left: u32,
    pub right: u32,
    pub untouched: u32,
    pub stamp: u32,
}

// One broad region owns both mutable positions in the returned symbolic value.
pub fn broad<'a>(state: &'a mut State) -> Bundle<(&'a mut u32, &'a mut u32)> {
    Bundle {
        payload: (&mut state.left, &mut state.right),
        stamp: state.stamp,
    }
}

// Calling this without expanding the Bundle splits that symbolic loan into two
// disjoint borrow projections. Each lifetime gives back its own updated value.
pub fn separate<'a, 'b>(bundle: Bundle<(&'a mut u32, &'b mut u32)>) {
    *bundle.payload.0 += 7;
    *bundle.payload.1 += 11;
    assert!(bundle.stamp == 41);
}

pub fn recovered() -> (u32, u32, u32, u32) {
    let mut state = State {
        left: 3,
        right: 5,
        untouched: 29,
        stamp: 41,
    };
    separate(broad(&mut state));
    (state.left, state.right, state.untouched, state.stamp)
}

// Forwarding a complete optional mutable projection must retain both variants.
pub fn optional_mut(values: &mut [u32], index: usize) -> Option<&mut u32> {
    values.get_mut(index)
}

pub fn optional_recovered(index: usize) -> (bool, u32, u32) {
    let mut values = [3, 29];
    let found = match optional_mut(&mut values, index) {
        Some(value) => {
            *value += 7;
            true
        }
        None => false,
    };
    (found, values[0], values[1])
}

// These opaque values have one mutable lifetime. Their complete current value
// must reach the map's backward function without trying to project private fields.
pub fn occupied_into_mut(
    entry: std::collections::btree_map::OccupiedEntry<'_, u32, u32>,
) -> &mut u32 {
    entry.into_mut()
}

pub fn vacant_insert(
    entry: std::collections::btree_map::VacantEntry<'_, u32, u32>,
    value: u32,
) -> &mut u32 {
    entry.insert(value)
}

pub fn opaque_recovered() -> (u32, u32, u32) {
    use std::collections::btree_map::Entry;
    let mut map = std::collections::BTreeMap::new();
    map.insert(1, 10);
    map.insert(3, 41);
    match map.entry(1) {
        Entry::Occupied(entry) => *occupied_into_mut(entry) += 7,
        Entry::Vacant(entry) => *vacant_insert(entry, 0) = 99,
    }
    match map.entry(2) {
        Entry::Vacant(entry) => *vacant_insert(entry, 20) += 3,
        Entry::Occupied(entry) => *occupied_into_mut(entry) = 99,
    }
    let first = *map.get(&1).unwrap();
    let second = *map.get(&2).unwrap();
    let untouched = *map.get(&3).unwrap();
    (first, second, untouched)
}

pub fn mutate_iterated_vec(mut values: Vec<u32>) -> Vec<u32> {
    for value in &mut values {
        *value += 7;
    }
    values
}

// Breaking after a prefix must retain its updates and the unconsumed suffix.
pub fn mutate_iterated_vec_prefix(mut values: Vec<u32>, mut count: usize) -> Vec<u32> {
    for value in &mut values {
        if count == 0 {
            break;
        }
        *value += 7;
        count -= 1;
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_updates_preserve_unrelated_fields() {
        assert_eq!(recovered(), (10, 16, 29, 41));
    }

    #[test]
    fn complete_projections_preserve_variants_and_opaque_state() {
        assert_eq!(optional_recovered(0), (true, 10, 29));
        assert_eq!(optional_recovered(2), (false, 3, 29));
        assert_eq!(opaque_recovered(), (17, 23, 41));
    }

    #[test]
    fn mutable_into_iterator_restores_every_vec_element() {
        assert_eq!(mutate_iterated_vec(vec![3, 5, 29]), vec![10, 12, 36]);
        assert_eq!(
            mutate_iterated_vec_prefix(vec![3, 5, 29], 2),
            vec![10, 12, 29]
        );
    }
}
