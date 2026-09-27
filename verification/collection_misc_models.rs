//! `Arc`, `BTreeSet`, `Option`, `Vec` and empty-reference defaults of the production crates
//! (Aeneas.Std Sync, BTreeSet, Core/CoreOption, Vec, Core/Default): `Arc::as_ref` and
//! `Arc::default`, `BTreeSet::default` and `BTreeSet::first`, `Option::clone_from`,
//! `option::IntoIter::next_back`, `<&[T]>::default`, `<&str>::default` and
//! `Vec::extend` from borrowed elements.
use std::collections::BTreeSet;
use std::sync::Arc;

/// As chelis-types `infer`: the value behind a shared `Arc`.
pub fn arc_value(shared: &Arc<Vec<u32>>) -> usize {
    let inner: &Vec<u32> = shared.as_ref();
    inner.len()
}

/// `Arc::default()` of a collection, then read back.
pub fn arc_default_len() -> usize {
    let shared: Arc<Vec<u32>> = Arc::default();
    shared.len()
}

/// As chelis-types `infer::declarations`: a default set, filled, and its least element.
pub fn least(values: &[u32]) -> Option<u32> {
    let mut set: BTreeSet<u32> = BTreeSet::default();
    for v in values {
        set.insert(*v);
    }
    set.first().copied()
}

/// As chelis-types `infer`: `Option::clone_from` into a caller's slot.
pub fn refresh(slot: &mut Option<String>, source: &Option<String>) {
    slot.clone_from(source);
}

/// `option::IntoIter::next_back`.
pub fn last_of(value: Option<u64>) -> (Option<u64>, Option<u64>) {
    let mut it = value.into_iter();
    let back = it.next_back();
    (back, it.next())
}

/// As chelis-types `infer::checked`: an empty default slice and string.
pub fn empty_defaults() -> (usize, usize) {
    let s: &[u32] = Default::default();
    let t: &str = Default::default();
    (s.len(), t.len())
}

/// As chelis-types `unify::recanonicalize_class_flags`: extend a caller's vector with copies
/// of borrowed elements.
pub fn extend_from(target: &mut Vec<u32>, source: &[u32]) {
    target.extend(source);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arcs() {
        assert_eq!(arc_value(&Arc::new(vec![1, 2, 3])), 3);
        assert_eq!(arc_default_len(), 0);
    }

    #[test]
    fn sets() {
        assert_eq!(least(&[5, 3, 9, 3]), Some(3));
        assert_eq!(least(&[]), None);
    }

    #[test]
    fn options() {
        let mut slot = Some("old".to_string());
        refresh(&mut slot, &Some("new".to_string()));
        assert_eq!(slot, Some("new".to_string()));
        refresh(&mut slot, &None);
        assert_eq!(slot, None);
        refresh(&mut slot, &Some("again".to_string()));
        assert_eq!(slot, Some("again".to_string()));
        assert_eq!(last_of(Some(7)), (Some(7), None));
        assert_eq!(last_of(None), (None, None));
    }

    #[test]
    fn defaults_and_extend() {
        assert_eq!(empty_defaults(), (0, 0));
        let mut v = vec![1];
        extend_from(&mut v, &[2, 3]);
        assert_eq!(v, vec![1, 2, 3]);
        extend_from(&mut v, &[]);
        assert_eq!(v, vec![1, 2, 3]);
    }
}
