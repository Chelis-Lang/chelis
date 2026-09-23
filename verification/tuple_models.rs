#![feature(partial_ord_chaining_methods)]

use std::cmp::Ordering;
use std::ops::ControlFlow;

struct Equality {
    eq: u8,
    ne: u8,
}

fn answer(mode: u8) -> bool {
    match mode {
        0 => false,
        1 => true,
        _ => panic!("comparison reached"),
    }
}

impl PartialEq for Equality {
    fn eq(&self, _other: &Self) -> bool {
        answer(self.eq)
    }

    fn ne(&self, _other: &Self) -> bool {
        answer(self.ne)
    }
}

pub fn pair_eq(first: u8, second: u8) -> bool {
    let left = (
        Equality { eq: first, ne: 2 },
        Equality { eq: second, ne: 2 },
    );
    left == (Equality { eq: 2, ne: 2 }, Equality { eq: 2, ne: 2 })
}

pub fn pair_ne(first: u8, second: u8) -> bool {
    let left = (
        Equality { eq: 2, ne: first },
        Equality { eq: 2, ne: second },
    );
    left != (Equality { eq: 2, ne: 2 }, Equality { eq: 2, ne: 2 })
}

struct Comparison {
    mode: u8,
}

impl PartialEq for Comparison {
    fn eq(&self, _other: &Self) -> bool {
        panic!("ordering must not call equality")
    }
}

impl Eq for Comparison {}

impl PartialOrd for Comparison {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        match self.mode {
            0 => Some(Ordering::Less),
            1 => Some(Ordering::Equal),
            2 => Some(Ordering::Greater),
            3 => None,
            _ => panic!("partial comparison reached"),
        }
    }

    fn lt(&self, _other: &Self) -> bool {
        // Deliberately distinct from partial_cmp: tuple lt must dispatch here
        // for its last component, but not for its equal prefix.
        self.mode == 4
    }

    fn __chaining_lt(&self, other: &Self) -> ControlFlow<bool> {
        match self.mode {
            5 => ControlFlow::Break(true),
            6 => ControlFlow::Break(false),
            _ => match self.partial_cmp(other) {
                Some(Ordering::Equal) => ControlFlow::Continue(()),
                Some(ordering) => ControlFlow::Break(ordering.is_lt()),
                None => ControlFlow::Break(false),
            },
        }
    }

    fn __chaining_le(&self, _other: &Self) -> ControlFlow<bool> {
        ControlFlow::Break(answer(self.mode - 5))
    }

    fn __chaining_gt(&self, _other: &Self) -> ControlFlow<bool> {
        ControlFlow::Break(answer(self.mode - 5))
    }

    fn __chaining_ge(&self, _other: &Self) -> ControlFlow<bool> {
        ControlFlow::Break(answer(self.mode - 5))
    }
}

impl Ord for Comparison {
    fn cmp(&self, _other: &Self) -> Ordering {
        match self.mode {
            0 => Ordering::Less,
            1 => Ordering::Equal,
            2 => Ordering::Greater,
            _ => panic!("total comparison reached"),
        }
    }
}

pub fn pair_partial_cmp(first: u8, second: u8) -> Option<Ordering> {
    (Comparison { mode: first }, Comparison { mode: second })
        .partial_cmp(&(Comparison { mode: 4 }, Comparison { mode: 4 }))
}

pub fn pair_cmp(first: u8, second: u8) -> Ordering {
    (Comparison { mode: first }, Comparison { mode: second })
        .cmp(&(Comparison { mode: 4 }, Comparison { mode: 4 }))
}

pub fn pair_lt_override() -> bool {
    (Comparison { mode: 1 }, Comparison { mode: 4 })
        < (Comparison { mode: 4 }, Comparison { mode: 4 })
}

pub fn pair_chaining_overrides(first: u8) -> (bool, bool, bool, bool) {
    let left = (Comparison { mode: first }, Comparison { mode: 4 });
    let right = (Comparison { mode: 4 }, Comparison { mode: 4 });
    (left < right, left <= right, left > right, left >= right)
}

pub fn ordering_discriminant(first: u8, second: u8) -> i8 {
    pair_cmp(first, second) as i8
}

struct Tagged {
    payload: u32,
}

impl PartialEq for Tagged {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for Tagged {}

impl PartialOrd for Tagged {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl Ord for Tagged {
    fn cmp(&self, _other: &Self) -> Ordering {
        panic!("extrema must dispatch PartialOrd.lt")
    }
}

pub fn pair_max_tie() -> u32 {
    let first = (Tagged { payload: 10 }, 0u8);
    let second = (Tagged { payload: 20 }, 0u8);
    first.max(second).0.payload
}

pub fn pair_min_tie() -> u32 {
    let first = (Tagged { payload: 10 }, 0u8);
    let second = (Tagged { payload: 20 }, 0u8);
    first.min(second).0.payload
}

#[derive(PartialEq)]
enum RecursiveEquality {
    Value(u8),
    Probe(Equality),
    Branch(Box<RecursiveEquality>, Box<RecursiveEquality>),
}

fn recursive_tree(first: u8, second: u8) -> RecursiveEquality {
    use RecursiveEquality::{Branch, Value};
    Branch(
        Box::new(Branch(Box::new(Value(7)), Box::new(Value(first)))),
        Box::new(Branch(Box::new(Value(second)), Box::new(Value(9)))),
    )
}

pub fn recursive_boxed_eq(first: u8, second: u8) -> bool {
    recursive_tree(first, second) == recursive_tree(3, 5)
}

pub fn recursive_boxed_ne(first: u8, second: u8) -> bool {
    recursive_tree(first, second) != recursive_tree(3, 5)
}

pub fn recursive_boxed_short_circuit(first: u8, second: u8) -> bool {
    use RecursiveEquality::{Branch, Probe, Value};
    let left = Branch(
        Box::new(recursive_tree(first, 5)),
        Box::new(Branch(
            Box::new(Value(9)),
            Box::new(Probe(Equality { eq: second, ne: 2 })),
        )),
    );
    let right = Branch(
        Box::new(recursive_tree(3, 5)),
        Box::new(Branch(
            Box::new(Value(9)),
            Box::new(Probe(Equality { eq: 2, ne: 2 })),
        )),
    );
    left == right
}

enum RecursiveOverride {
    Leaf(Equality),
    Branch(Box<RecursiveOverride>, Box<RecursiveOverride>),
}

impl PartialEq for RecursiveOverride {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(left), Self::Leaf(right)) => left == right,
            (Self::Branch(left_first, left_second), Self::Branch(right_first, right_second)) => {
                left_first == right_first && left_second == right_second
            }
            _ => false,
        }
    }

    fn ne(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(left), Self::Leaf(right)) => left != right,
            (Self::Branch(left_first, left_second), Self::Branch(right_first, right_second)) => {
                left_first != right_first || left_second != right_second
            }
            _ => true,
        }
    }
}

pub fn recursive_boxed_ne_override(first: u8, second: u8) -> bool {
    use RecursiveOverride::{Branch, Leaf};
    let left = Box::new(Branch(
        Box::new(Leaf(Equality { eq: 2, ne: first })),
        Box::new(Branch(
            Box::new(Leaf(Equality { eq: 2, ne: 0 })),
            Box::new(Leaf(Equality { eq: 2, ne: second })),
        )),
    ));
    let right = Box::new(Branch(
        Box::new(Leaf(Equality { eq: 2, ne: 2 })),
        Box::new(Branch(
            Box::new(Leaf(Equality { eq: 2, ne: 2 })),
            Box::new(Leaf(Equality { eq: 2, ne: 2 })),
        )),
    ));
    &left != &right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_boxed_comparison_preserves_nested_fields_and_defaults() {
        assert!(recursive_boxed_eq(3, 5));
        assert!(!recursive_boxed_eq(4, 5));
        assert!(!recursive_boxed_eq(3, 6));
        assert!(!recursive_boxed_ne(3, 5));
        assert!(recursive_boxed_ne(3, 6));
        assert!(!recursive_boxed_short_circuit(4, 2));
        assert!(recursive_boxed_short_circuit(3, 1));
        assert!(std::panic::catch_unwind(|| recursive_boxed_short_circuit(3, 2)).is_err());
    }

    #[test]
    fn recursive_boxed_ne_preserves_overrides_and_short_circuiting() {
        assert!(recursive_boxed_ne_override(1, 2));
        assert!(recursive_boxed_ne_override(0, 1));
        assert!(!recursive_boxed_ne_override(0, 0));
        assert!(std::panic::catch_unwind(|| recursive_boxed_ne_override(0, 2)).is_err());
        assert!(std::panic::catch_unwind(|| recursive_boxed_ne_override(2, 1)).is_err());
    }

    #[test]
    fn equality_short_circuits_and_ne_dispatches_overrides() {
        assert!(!pair_eq(0, 2));
        assert!(pair_ne(1, 2));
        assert!(pair_ne(0, 1));
        assert!(std::panic::catch_unwind(|| pair_eq(1, 2)).is_err());
        assert!(std::panic::catch_unwind(|| pair_ne(0, 2)).is_err());
        assert!(std::panic::catch_unwind(|| pair_eq(2, 0)).is_err());
        assert!(std::panic::catch_unwind(|| pair_ne(2, 1)).is_err());
    }

    #[test]
    fn lexicographic_partial_order_preserves_incomparability_and_failure() {
        assert_eq!(pair_partial_cmp(0, 4), Some(Ordering::Less));
        assert_eq!(pair_partial_cmp(3, 4), None);
        assert_eq!(pair_partial_cmp(1, 3), None);
        assert_eq!(pair_partial_cmp(1, 2), Some(Ordering::Greater));
        assert!(std::panic::catch_unwind(|| pair_partial_cmp(4, 0)).is_err());
        assert!(std::panic::catch_unwind(|| pair_partial_cmp(1, 4)).is_err());
        assert!(pair_lt_override());
        assert_eq!(pair_chaining_overrides(5), (true, false, false, false));
        assert_eq!(ordering_discriminant(0, 4), -1);
        assert_eq!(pair_chaining_overrides(6), (false, true, true, true));
    }

    #[test]
    fn total_order_short_circuits_and_extrema_preserve_tie_identity() {
        assert_eq!(pair_cmp(2, 4), Ordering::Greater);
        assert_eq!(pair_cmp(1, 0), Ordering::Less);
        assert!(std::panic::catch_unwind(|| pair_cmp(4, 0)).is_err());
        assert!(std::panic::catch_unwind(|| pair_cmp(1, 4)).is_err());
        assert_eq!(pair_max_tie(), 20);
        assert_eq!(pair_min_tie(), 10);
    }
}
