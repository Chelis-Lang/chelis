pub fn remove_at(index: usize) -> (u32, Vec<u32>) {
    let mut values = Vec::from([10, 20, 30]);
    let removed = values.remove(index);
    (removed, values)
}

pub fn clear_then_extend() -> Vec<u32> {
    let mut values: Vec<u32> = Default::default();
    values.push(90);
    values.clear();
    assert!(values.is_empty());
    values.extend([7, 8]);
    values
}

// Deliberately non-fused: the terminal next call mutates externally retained
// state, and extension must stop at the first None without consuming 9.
pub struct Probe {
    pub calls: usize,
}

impl Iterator for Probe {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.calls += 1;
        match self.calls {
            1 => Some(4),
            2 => None,
            3 => Some(9),
            _ => None,
        }
    }
}

pub fn extend_resumes() -> (Vec<u32>, usize, Option<u32>) {
    let mut values = Vec::from([8]);
    let mut probe = Probe { calls: 0 };
    values.extend(&mut probe);
    let terminal_calls = probe.calls;
    (values, terminal_calls, probe.next())
}

pub fn extend_callback_state() -> (Vec<u32>, u32) {
    let mut sum = 0;
    let mut values = Vec::from([8]);
    values.extend([1, 2].into_iter().map(|item| {
        sum += item;
        item + 10
    }));
    (values, sum)
}

pub fn extend_failure() -> Vec<u32> {
    let mut values = Vec::new();
    values.extend([1, 2].into_iter().map(|item| {
        assert!(item != 2);
        item
    }));
    values
}

pub fn stable_duplicate_keys() -> (Vec<(u32, u32)>, bool) {
    let mut values = Vec::from([(2, 10), (1, 11), (2, 12), (1, 13)]);
    let mut called = false;
    values.sort_by_key(|item| {
        called = true;
        item.0
    });
    // No assertion about how often or in which order the key was evaluated.
    (values, called)
}

pub fn sort_values() -> Vec<u32> {
    let mut values = Vec::from([3, 1, 2, 1]);
    values.sort();
    values
}

pub fn sort_key_failure() -> Vec<u32> {
    let mut values = Vec::from([2, 1]);
    values.sort_by_key(|item| {
        assert!(*item != 2);
        *item
    });
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removal_and_clear_preserve_remaining_order() {
        assert_eq!(remove_at(1), (20, vec![10, 30]));
        assert!(std::panic::catch_unwind(|| remove_at(3)).is_err());
        assert_eq!(clear_then_extend(), vec![7, 8]);
    }

    #[test]
    fn extension_stops_at_none_and_preserves_callback_state() {
        assert_eq!(extend_resumes(), (vec![8, 4], 2, Some(9)));
        assert_eq!(extend_callback_state(), (vec![8, 11, 12], 3));
        assert!(std::panic::catch_unwind(extend_failure).is_err());
    }

    #[test]
    fn sorting_is_stable_and_observes_key_callbacks() {
        assert_eq!(
            stable_duplicate_keys(),
            (vec![(1, 11), (1, 13), (2, 10), (2, 12)], true)
        );
        assert_eq!(sort_values(), vec![1, 1, 2, 3]);
        assert!(std::panic::catch_unwind(sort_key_failure).is_err());
    }
}
