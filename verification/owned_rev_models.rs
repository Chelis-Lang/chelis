//! The back end of a consumed vector (Aeneas.Std VecIter.lean): `DoubleEndedIterator`
//! for `vec::IntoIter` (`next_back`, `advance_back_by`, and the provided `nth_back` and
//! `rfold`), reached through `Iterator::rev` as in chelis-ir `host.rs`
//! (`callable_scope.restore(undos.into_iter().rev())`).

use std::collections::BTreeMap;

pub struct Undo {
    pub name: String,
    pub previous: Option<Option<String>>,
}

/// As chelis-ir `CallableScope`: bindings that nested scopes update, then restore.
#[derive(Default)]
pub struct Scope {
    pub bindings: BTreeMap<String, Option<String>>,
}

impl Scope {
    pub fn bind(&mut self, name: String, target: Option<String>) -> Undo {
        let previous = self.bindings.insert(name.clone(), target);
        Undo { name, previous }
    }

    pub fn restore(&mut self, undos: impl IntoIterator<Item = Undo>) {
        for undo in undos.into_iter() {
            if let Some(previous) = undo.previous {
                self.bindings.insert(undo.name, previous);
            } else {
                self.bindings.remove(&undo.name);
            }
        }
    }
}

/// Bind every name (a later binding of the same name shadows an earlier one),
/// read `probe` inside the scope, then undo the bindings in reverse order.
pub fn with_bindings(scope: &mut Scope, names: Vec<String>, probe: &str) -> Option<Option<String>> {
    let mut undos = Vec::new();
    for (index, name) in names.into_iter().enumerate() {
        let target = if index % 2 == 0 { Some(name.clone()) } else { None };
        undos.push(scope.bind(name, target));
    }
    let seen = scope.bindings.get(probe).cloned();
    scope.restore(undos.into_iter().rev());
    seen
}

pub fn reversed(values: Vec<u32>) -> Vec<u32> {
    values.into_iter().rev().collect()
}

/// `next_back` and `next` on the same iterator, then the rest.
pub fn both_ends(values: Vec<u32>) -> (Option<u32>, Option<u32>, Vec<u32>) {
    let mut iter = values.into_iter();
    let back = iter.next_back();
    let front = iter.next();
    (back, front, iter.collect())
}

/// The provided `nth_back` through `advance_back_by`.
pub fn nth_from_back(values: Vec<u32>, n: usize) -> (Option<u32>, Vec<u32>) {
    let mut iter = values.into_iter();
    let found = iter.nth_back(n);
    (found, iter.collect())
}

/// `Rev::fold` is the inner iterator's provided `rfold`.
pub fn digits_reversed(values: Vec<u32>) -> u32 {
    values.into_iter().rev().fold(0, |acc, digit| acc * 10 + digit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn scoped_bindings_are_restored_in_reverse() {
        let mut scope = Scope::default();
        scope.bindings.insert("f".to_string(), Some("outer".to_string()));
        let seen = with_bindings(&mut scope, strings(&["f", "g", "f"]), "f");
        assert_eq!(seen, Some(Some("f".to_string())));
        assert_eq!(scope.bindings.len(), 1);
        assert_eq!(scope.bindings.get("f"), Some(&Some("outer".to_string())));
        let seen = with_bindings(&mut scope, strings(&["g", "h"]), "h");
        assert_eq!(seen, Some(None));
        assert_eq!(scope.bindings.len(), 1);
        let mut empty = Scope::default();
        assert_eq!(with_bindings(&mut empty, strings(&["a", "a"]), "a"), Some(None));
        assert!(empty.bindings.is_empty());
    }

    #[test]
    fn reversed_and_both_ends() {
        assert_eq!(reversed(vec![1, 2, 3]), vec![3, 2, 1]);
        assert_eq!(reversed(vec![]), Vec::<u32>::new());
        assert_eq!(both_ends(vec![4, 5, 6, 7]), (Some(7), Some(4), vec![5, 6]));
        assert_eq!(both_ends(vec![9]), (Some(9), None, vec![]));
        assert_eq!(both_ends(vec![]), (None, None, vec![]));
    }

    #[test]
    fn nth_back_and_rfold() {
        assert_eq!(nth_from_back(vec![1, 2, 3, 4, 5], 1), (Some(4), vec![1, 2, 3]));
        assert_eq!(nth_from_back(vec![1, 2, 3], 0), (Some(3), vec![1, 2]));
        assert_eq!(nth_from_back(vec![1, 2, 3], 3), (None, vec![]));
        assert_eq!(nth_from_back(vec![1, 2, 3], 7), (None, vec![]));
        assert_eq!(digits_reversed(vec![1, 2, 3]), 321);
        assert_eq!(digits_reversed(vec![]), 0);
    }
}
