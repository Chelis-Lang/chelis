use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::BTreeSet;

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

pub fn duplicate_from_iterator() -> BTreeSet<Key> {
    let mut items = Vec::new();
    items.push(Key {
        rank: 2,
        identity: 20,
    });
    items.push(Key {
        rank: 1,
        identity: 10,
    });
    items.push(Key {
        rank: 2,
        identity: 99,
    });
    items.push(Key {
        rank: 3,
        identity: 30,
    });
    items.push(Key {
        rank: 1,
        identity: 88,
    });
    BTreeSet::from_iter(items)
}

pub fn borrowed_lookup() -> (bool, bool, bool, bool) {
    let set = duplicate_from_iterator();
    (
        set.contains(&0),
        set.contains(&2),
        set.contains(&4),
        set.contains(&1),
    )
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
    type Item = u32;
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
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        *self.polls += 1;
        if self.remaining == 0 {
            None
        } else {
            let value = self.remaining;
            self.remaining -= 1;
            Some(value)
        }
    }
}

fn delegated_collect<T: Ord, I: IntoIterator<Item = T>>(source: I) -> BTreeSet<T> {
    BTreeSet::from_iter(source)
}

pub fn delegated_collection_state(length: u32) -> (BTreeSet<u32>, u32) {
    let mut polls = 0;
    let set = delegated_collect(CountedSource {
        polls: &mut polls,
        length,
    });
    (set, polls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_sorts_and_retains_the_last_equal_key() {
        let keys: Vec<_> = duplicate_from_iterator()
            .into_iter()
            .map(|key| (key.rank, key.identity))
            .collect();
        assert_eq!(keys, [(1, 88), (2, 99), (3, 30)]);
        assert_eq!(borrowed_lookup(), (false, true, false, true));
    }

    #[test]
    fn collection_preserves_conversion_and_terminal_iterator_state() {
        let (set, polls) = delegated_collection_state(3);
        assert_eq!(set.into_iter().collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(polls, 14);
        let (empty, polls) = delegated_collection_state(0);
        assert!(empty.is_empty());
        assert_eq!(polls, 11);
    }
}
