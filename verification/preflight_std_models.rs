//! Models of the std items reached by the bridge's `chelis_ir::host::with_preflight`
//! root: `VecDeque` (`new`, `Default`, `From<[T; N]>`, `push_back`, `push_front`,
//! `pop_front`, `pop_back`, `front`, `back`, `len`, `is_empty`, `clear`),
//! `BTreeMap::get_mut`, `<[T]>::first_mut` and `last_mut`, `RangeFull` slice
//! indexing (`[..]`, `get(..)`, `get_mut(..)`), `String::with_capacity`, `push`,
//! `push_str` and `len`, `str::repeat` and `contains`, `bool::then`,
//! `Iterator::sum` (`Sum<usize> for usize`, through `Map`) and `AsRef for Box`.
//! The functions taking `&mut` check the values given back.
use std::collections::{BTreeMap, VecDeque};

/// A breadth-first worklist (host.rs `cached_dynamic_to_tensor_def_summaries`):
/// items leave in FIFO order, and items pushed while draining come after.
pub fn worklist(items: &[u32]) -> (Vec<u32>, Option<u32>, bool) {
    let mut queue = VecDeque::new();
    for item in items {
        queue.push_back(*item);
    }
    let mut order = Vec::new();
    while let Some(item) = queue.pop_front() {
        if item >= 10 {
            queue.push_back(item / 10);
        }
        order.push(item);
    }
    (order, queue.pop_front(), queue.is_empty())
}

/// Both ends of a deque built `From<[T; N]>`, and a cleared and a default one.
pub fn deque_ends(first: u32, second: u32) -> (Option<u32>, Option<u32>, Option<u32>, usize, usize, bool) {
    let mut queue = VecDeque::from([first, second]);
    queue.push_front(0);
    let back = queue.pop_back();
    let front = queue.front().copied();
    let last = queue.back().copied();
    let len = queue.len();
    queue.clear();
    let fresh: VecDeque<u32> = VecDeque::default();
    (back, front, last, len, queue.len() + fresh.len(), queue.is_empty())
}

/// `pop_back` and `pop_front` on a deque given by `&mut`, down to empty.
pub fn drain_ends(queue: &mut VecDeque<u32>) -> (Option<u32>, Option<u32>) {
    let back = queue.pop_back();
    let front = queue.pop_front();
    queue.push_front(7);
    (back, front)
}

/// `BTreeMap::get_mut`: the write through the found value is given back; a
/// missing key leaves the map unchanged.
pub fn bump(map: &mut BTreeMap<u32, u32>, key: u32) -> bool {
    match map.get_mut(&key) {
        Some(value) => {
            *value += 100;
            true
        }
        None => false,
    }
}

/// `get_mut` on a map built in place, then read back in key order.
pub fn bump_entries(entries: &[(u32, u32)], key: u32) -> (bool, Vec<(u32, u32)>) {
    let mut map: BTreeMap<u32, u32> = entries.iter().copied().collect();
    let found = bump(&mut map, key);
    (found, map.into_iter().collect())
}

/// `<[T]>::last_mut`: the write reaches the last element.
pub fn bump_last(items: &mut [u32]) -> bool {
    match items.last_mut() {
        Some(last) => {
            *last += 10;
            true
        }
        None => false,
    }
}

/// `<[T]>::first_mut`: the write reaches the first element.
pub fn bump_first(items: &mut [u32]) -> bool {
    match items.first_mut() {
        Some(first) => {
            *first += 1;
            true
        }
        None => false,
    }
}

/// The scope stack of host.rs (`if let Some(scope) = self.scopes.last_mut()`):
/// a push into the innermost scope.
pub fn push_innermost(scopes: &mut Vec<Vec<u32>>, value: u32) -> bool {
    if let Some(scope) = scopes.last_mut() {
        scope.push(value);
        true
    } else {
        false
    }
}

/// `[..]` shared and mutable, `get(..)` and `get_mut(..)`: the whole slice.
pub fn full_range(items: &mut Vec<u32>) -> (usize, Option<usize>) {
    let len = items[..].len();
    let got = items.get(..).map(|all| all.len());
    if let Some(all) = items.get_mut(..) {
        if let Some(first) = all.first_mut() {
            *first += 1;
        }
    }
    let whole = &mut items[..];
    if let Some(last) = whole.last_mut() {
        *last *= 2;
    }
    (len, got)
}

/// `String::with_capacity`, `push_str`, `push` and `len` (in UTF-8 bytes).
pub fn build(capacity: usize, word: &str, ch: char) -> (String, usize) {
    let mut out = String::with_capacity(capacity);
    out.push_str(word);
    out.push(ch);
    out.push_str("!");
    let len = out.len();
    (out, len)
}

/// `push_str` and `push` on a string given by `&mut`.
pub fn append(out: &mut String, word: &str) -> usize {
    out.push_str(word);
    out.push('.');
    out.len()
}

/// The indentation of chelis-deep printer.rs: `" ".repeat(indent)`.
pub fn indent(level: usize, text: &str) -> String {
    let mut line = " ".repeat(level);
    line.push_str(text);
    line
}

pub fn repeat_word(word: &str, n: usize) -> String {
    word.repeat(n)
}

/// `str::contains` with a `&str` and a `char` pattern.
pub fn contains_both(haystack: &str, needle: &str, ch: char) -> (bool, bool) {
    (haystack.contains(needle), haystack.contains(ch))
}

/// `bool::then` (host.rs `def_names.contains(name).then(|| name.to_string())`).
pub fn then_label(flag: bool, name: &str) -> Option<String> {
    flag.then(|| name.to_string())
}

/// The closure runs only when the condition holds: `then_count(false, u32::MAX)`
/// does not overflow.
pub fn then_count(flag: bool, base: u32) -> Option<u32> {
    flag.then(|| base + 1)
}

fn weight(value: &u32) -> usize {
    *value as usize * 2
}

/// `Iterator::sum::<usize>` of a `Map` (host.rs `deep_expr_nodes`).
pub fn total(items: &[u32]) -> usize {
    items.iter().map(weight).sum::<usize>()
}

/// `sum` of a `Map` with a closure; overflow panics (checked addition).
pub fn total_sizes(items: &[usize]) -> usize {
    items.iter().map(|size| *size).sum()
}

/// `AsRef<T> for Box<T>`.
pub fn unbox(boxed: &Box<(u32, String)>) -> (u32, usize) {
    let inner: &(u32, String) = boxed.as_ref();
    (inner.0 + 1, inner.1.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deques() {
        assert_eq!(worklist(&[3, 25, 4]), (vec![3, 25, 4, 2], None, true));
        assert_eq!(worklist(&[]), (vec![], None, true));
        assert_eq!(worklist(&[120]), (vec![120, 12, 1], None, true));
        assert_eq!(deque_ends(5, 6), (Some(6), Some(0), Some(5), 2, 0, true));
        let mut queue = VecDeque::from([1, 2, 3]);
        assert_eq!(drain_ends(&mut queue), (Some(3), Some(1)));
        assert_eq!(queue, VecDeque::from([7, 2]));
        let mut single = VecDeque::from([4]);
        assert_eq!(drain_ends(&mut single), (Some(4), None));
        assert_eq!(single, VecDeque::from([7]));
        let mut empty = VecDeque::new();
        assert_eq!(drain_ends(&mut empty), (None, None));
        assert_eq!(empty, VecDeque::from([7]));
    }

    #[test]
    fn map_get_mut() {
        let mut map = BTreeMap::from([(1, 10), (3, 30), (5, 50)]);
        assert!(bump(&mut map, 3));
        assert_eq!(map, BTreeMap::from([(1, 10), (3, 130), (5, 50)]));
        assert!(!bump(&mut map, 4));
        assert_eq!(map, BTreeMap::from([(1, 10), (3, 130), (5, 50)]));
        assert_eq!(bump_entries(&[(2, 1), (1, 7)], 1), (true, vec![(1, 107), (2, 1)]));
        assert_eq!(bump_entries(&[], 1), (false, vec![]));
        assert!(std::panic::catch_unwind(|| bump_entries(&[(1, u32::MAX)], 1)).is_err());
    }

    #[test]
    fn slice_ends() {
        let mut items = [1, 2, 3];
        assert!(bump_last(&mut items));
        assert_eq!(items, [1, 2, 13]);
        assert!(bump_first(&mut items));
        assert_eq!(items, [2, 2, 13]);
        let mut none: [u32; 0] = [];
        assert!(!bump_last(&mut none));
        assert!(!bump_first(&mut none));
        let mut scopes = vec![vec![1], vec![2]];
        assert!(push_innermost(&mut scopes, 9));
        assert_eq!(scopes, vec![vec![1], vec![2, 9]]);
        let mut no_scopes: Vec<Vec<u32>> = vec![];
        assert!(!push_innermost(&mut no_scopes, 9));
        assert_eq!(no_scopes, Vec::<Vec<u32>>::new());
    }

    #[test]
    fn full_ranges() {
        let mut items = vec![1, 2, 3];
        assert_eq!(full_range(&mut items), (3, Some(3)));
        assert_eq!(items, vec![2, 2, 6]);
        let mut one = vec![5];
        assert_eq!(full_range(&mut one), (1, Some(1)));
        assert_eq!(one, vec![12]);
        let mut empty: Vec<u32> = vec![];
        assert_eq!(full_range(&mut empty), (0, Some(0)));
        assert_eq!(empty, Vec::<u32>::new());
    }

    #[test]
    fn strings() {
        assert_eq!(build(4, "ab", 'c'), ("abc!".to_string(), 4));
        assert_eq!(build(0, "é", 'ß'), ("éß!".to_string(), 5));
        assert!(std::panic::catch_unwind(|| build(usize::MAX, "", 'a')).is_err());
        assert!(std::panic::catch_unwind(|| build(isize::MAX as usize + 1, "", 'a')).is_err());
        let mut out = "x".to_string();
        assert_eq!(append(&mut out, "yz"), 4);
        assert_eq!(out, "xyz.");
        assert_eq!(indent(3, "a"), "   a");
        assert_eq!(indent(0, "b"), "b");
        assert_eq!(repeat_word("ab", 3), "ababab");
        assert_eq!(repeat_word("é", 2), "éé");
        assert_eq!(repeat_word("ab", 0), "");
        assert_eq!(repeat_word("", usize::MAX), "");
        assert!(std::panic::catch_unwind(|| repeat_word("ab", usize::MAX)).is_err());
        assert!(std::panic::catch_unwind(|| repeat_word("ab", 1 << 62)).is_err());
        assert_eq!(contains_both("hello", "ll", 'z'), (true, false));
        assert_eq!(contains_both("", "", 'a'), (true, false));
        assert_eq!(contains_both("né", "é", 'é'), (true, true));
        assert_eq!(contains_both("abc", "abcd", 'c'), (false, true));
        assert_eq!(contains_both("abc", "b", 'a'), (true, true));
    }

    #[test]
    fn then() {
        assert_eq!(then_label(true, "f"), Some("f".to_string()));
        assert_eq!(then_label(false, "f"), None);
        assert_eq!(then_count(true, 1), Some(2));
        assert_eq!(then_count(false, u32::MAX), None);
        assert!(std::panic::catch_unwind(|| then_count(true, u32::MAX)).is_err());
    }

    #[test]
    fn sums() {
        assert_eq!(total(&[1, 2, 3]), 12);
        assert_eq!(total(&[]), 0);
        assert_eq!(total_sizes(&[usize::MAX, 0]), usize::MAX);
        assert!(std::panic::catch_unwind(|| total_sizes(&[usize::MAX, 1])).is_err());
    }

    #[test]
    fn boxes() {
        assert_eq!(unbox(&Box::new((4, "abc".to_string()))), (5, 3));
    }
}
