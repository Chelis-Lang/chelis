//! `core::cmp::Reverse` (Aeneas.Std Core/Reverse.lean) as chelis-ir `ownership::last_use` uses
//! it: the key of a stable `sort_by_key`, alone and as the last component of a tuple key, and
//! from a map lookup; then the comparison operators of `Reverse` directly (`partial_cmp`, `lt`,
//! `le`, `gt`, `ge`, `cmp`, `max`, `min`, `clamp`, `==`, `!=`).
use std::cmp::{Ordering, Reverse};
use std::collections::BTreeMap;

/// As `last_use::Placement`: a derived total order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Placement {
    InBlock { block: u32, after: u32 },
    OnEdge { edge: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub point: Placement,
    pub definition_order: usize,
    pub owner: u32,
}

/// As `apply_placements`: the latest definition first, ties in input order.
pub fn latest_first(mut values: Vec<Placed>) -> Vec<Placed> {
    values.sort_by_key(|placed| Reverse(placed.definition_order));
    values
}

/// As `rebuild_host_sites`: by point, then the latest definition first.
pub fn by_point_then_latest(mut values: Vec<Placed>) -> Vec<Placed> {
    values.sort_by_key(|placed| (placed.point, Reverse(placed.definition_order)));
    values
}

/// As the canonical-order check: owners sorted by their definition, latest first.
pub fn owners_latest_first(definitions: &BTreeMap<u32, usize>, owners: &Vec<u32>) -> Vec<u32> {
    let mut expected = owners.clone();
    expected.sort_by_key(|owner| Reverse(definitions[owner]));
    expected
}

pub fn relations(a: u32, b: u32) -> (Option<Ordering>, bool, bool, bool, bool, Ordering) {
    let (x, y) = (Reverse(a), Reverse(b));
    (x.partial_cmp(&y), x < y, x <= y, x > y, x >= y, x.cmp(&y))
}

pub fn equalities(a: u32, b: u32) -> (bool, bool) {
    (Reverse(a) == Reverse(b), Reverse(a) != Reverse(b))
}

/// `max` and `min` return the argument whose field is the smaller and the larger one; on equal
/// fields `max` returns `other` and `min` returns `self` (the flag tells them apart).
pub fn extremes(a: (u32, bool), b: (u32, bool)) -> ((u32, bool), (u32, bool)) {
    let (x, y) = (Reverse(Key(a.0, a.1)), Reverse(Key(b.0, b.1)));
    let (max, min) = (x.max(y), x.min(y));
    ((max.0.0, max.0.1), (min.0.0, min.0.1))
}

/// A key compared by its first field only.
#[derive(Clone, Copy, Debug)]
pub struct Key(pub u32, pub bool);

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

/// `clamp` in the reversed order: `min` must not be greater than `max` in that order (a panic
/// otherwise).
pub fn clamped(value: u32, min: u32, max: u32) -> u32 {
    Reverse(value).clamp(Reverse(min), Reverse(max)).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placed(point: Placement, definition_order: usize, owner: u32) -> Placed {
        Placed { point, definition_order, owner }
    }

    fn owners(values: &[Placed]) -> Vec<u32> {
        values.iter().map(|p| p.owner).collect()
    }

    fn sample() -> Vec<Placed> {
        let e = |edge| Placement::OnEdge { edge };
        let b = |block, after| Placement::InBlock { block, after };
        vec![
            placed(e(2), 1, 10),
            placed(b(0, 3), 4, 11),
            placed(e(2), 4, 12),
            placed(b(0, 1), 1, 13),
            placed(b(0, 3), 7, 14),
            placed(e(1), 4, 15),
        ]
    }

    #[test]
    fn sorts() {
        assert_eq!(owners(&latest_first(sample())), vec![14, 11, 12, 15, 10, 13]);
        assert_eq!(owners(&by_point_then_latest(sample())), vec![13, 14, 11, 15, 12, 10]);
        assert_eq!(latest_first(Vec::new()), Vec::new());
        let definitions = BTreeMap::from([(1, 5), (2, 9), (3, 5), (4, 0)]);
        assert_eq!(owners_latest_first(&definitions, &vec![4, 1, 2, 3]), vec![2, 1, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn missing_definition() {
        let definitions = BTreeMap::from([(1, 5)]);
        owners_latest_first(&definitions, &vec![1, 7]);
    }

    #[test]
    fn comparisons() {
        assert_eq!(relations(3, 5), (Some(Ordering::Greater), false, false, true, true, Ordering::Greater));
        assert_eq!(relations(5, 3), (Some(Ordering::Less), true, true, false, false, Ordering::Less));
        assert_eq!(relations(4, 4), (Some(Ordering::Equal), false, true, false, true, Ordering::Equal));
        assert_eq!(equalities(4, 4), (true, false));
        assert_eq!(equalities(4, 5), (false, true));
        assert_eq!(extremes((3, true), (5, false)), ((3, true), (5, false)));
        assert_eq!(extremes((5, true), (3, false)), ((3, false), (5, true)));
        assert_eq!(extremes((4, true), (4, false)), ((4, false), (4, true)));
        assert_eq!(clamped(1, 9, 5), 5);
        assert_eq!(clamped(7, 9, 5), 7);
        assert_eq!(clamped(12, 9, 5), 9);
    }

    #[test]
    #[should_panic]
    fn clamp_bounds() {
        clamped(7, 5, 9);
    }
}
