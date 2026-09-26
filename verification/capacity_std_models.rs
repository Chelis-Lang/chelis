use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Equality and order observe only `key`, so the models' choice of which of
/// two equal elements survives is visible through `tag`.
#[derive(Clone)]
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

pub fn tagged(pairs: &[(u32, u32)]) -> Vec<Tagged> {
    let mut items = Vec::new();
    for &(key, tag) in pairs {
        items.push(Tagged { key, tag });
    }
    items
}

pub fn dedup_runs(pairs: &[(u32, u32)]) -> Vec<Tagged> {
    let mut items = tagged(pairs);
    items.dedup();
    items
}

pub fn boxed_clone(pairs: &[(u32, u32)]) -> (Box<[Tagged]>, Box<[Tagged]>) {
    let boxed = tagged(pairs).into_boxed_slice();
    let copy = boxed.clone();
    (boxed, copy)
}

/// Consumes the boxed slice in order: digits of the result are the elements.
pub fn boxed_digits(digits: &[u32]) -> u32 {
    let mut items = Vec::new();
    for &digit in digits {
        items.push(digit);
    }
    let boxed = items.into_boxed_slice();
    let mut value = 0;
    for digit in boxed {
        value = value * 10 + digit;
    }
    value
}

pub fn to_unsigned(value: i64) -> Option<u64> {
    match u64::try_from(value) {
        Ok(unsigned) => Some(unsigned),
        Err(_) => None,
    }
}

pub fn lift_some(value: Result<u64, u8>) -> Result<Option<u64>, u8> {
    value.map(Some)
}

/// The insert/remove results of one set, and its final elements.
pub fn set_script() -> ([bool; 6], BTreeSet<Tagged>) {
    let mut set = BTreeSet::new();
    let first = set.insert(Tagged { key: 2, tag: 20 });
    let duplicate = set.insert(Tagged { key: 2, tag: 99 });
    let other = set.insert(Tagged { key: 1, tag: 10 });
    let removed = set.remove(&Tagged { key: 1, tag: 0 });
    let missing = set.remove(&Tagged { key: 1, tag: 10 });
    let last = set.insert(Tagged { key: 3, tag: 30 });
    ([first, duplicate, other, removed, missing, last], set)
}

/// The borrowed-tuple visiting set of `capacity_key`: insert, re-insert, remove.
pub fn visiting_script(node: u32, axis: usize) -> [bool; 4] {
    let mut visiting = BTreeSet::new();
    let entered = visiting.insert((node, axis));
    let reentered = visiting.insert((node, axis));
    let left = visiting.remove(&(node, axis));
    let left_again = visiting.remove(&(node, axis));
    [entered, reentered, left, left_again]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(items: &[Tagged]) -> Vec<(u32, u32)> {
        items.iter().map(|item| (item.key, item.tag)).collect()
    }

    #[test]
    fn dedup_keeps_the_first_of_each_run() {
        let input = [(1, 10), (1, 11), (2, 20), (1, 12), (1, 13), (3, 30), (3, 31)];
        assert_eq!(pairs(&dedup_runs(&input)), [(1, 10), (2, 20), (1, 12), (3, 30)]);
        assert_eq!(pairs(&dedup_runs(&[])), []);
        assert_eq!(pairs(&dedup_runs(&[(5, 50)])), [(5, 50)]);
        assert_eq!(pairs(&dedup_runs(&[(4, 1), (4, 2), (4, 3)])), [(4, 1)]);
    }

    #[test]
    fn boxed_slices_preserve_elements_and_order() {
        let (boxed, copy) = boxed_clone(&[(1, 10), (1, 11), (0, 5)]);
        assert_eq!(pairs(&boxed), [(1, 10), (1, 11), (0, 5)]);
        assert_eq!(pairs(&copy), [(1, 10), (1, 11), (0, 5)]);
        assert_eq!(boxed_digits(&[3, 1, 4, 1, 5]), 31415);
        assert_eq!(boxed_digits(&[]), 0);
    }

    #[test]
    fn signed_to_unsigned_rejects_exactly_the_negatives() {
        assert_eq!(to_unsigned(0), Some(0));
        assert_eq!(to_unsigned(i64::MAX), Some(9223372036854775807));
        assert_eq!(to_unsigned(-1), None);
        assert_eq!(to_unsigned(i64::MIN), None);
        assert_eq!(lift_some(Ok(7)), Ok(Some(7)));
        assert_eq!(lift_some(Err(3)), Err(3));
    }

    #[test]
    fn set_insert_keeps_the_stored_equal_element() {
        let (results, set) = set_script();
        assert_eq!(results, [true, false, true, true, false, true]);
        assert_eq!(set.iter().map(|item| (item.key, item.tag)).collect::<Vec<_>>(), [(2, 20), (3, 30)]);
        assert_eq!(visiting_script(7, 2), [true, false, true, false]);
    }
}
