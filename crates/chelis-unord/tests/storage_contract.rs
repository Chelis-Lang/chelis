use std::hash::{Hash, Hasher};

use chelis_unord::{Entry, UnordMap, UnordSet};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CollidingKey(String);

impl Hash for CollidingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        0_u8.hash(state);
    }
}

#[test]
fn colliding_keys_preserve_point_lookup_mutation_and_removal() {
    let mut map = UnordMap::new();
    assert_eq!(map.insert(CollidingKey("beta".into()), 2), None);
    assert_eq!(map.insert(CollidingKey("alpha".into()), 1), None);
    assert_eq!(map.insert(CollidingKey("gamma".into()), 3), None);
    assert_eq!(map.insert(CollidingKey("beta".into()), 20), Some(2));

    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&CollidingKey("alpha".into())), Some(&1));
    assert_eq!(
        map.get_key_value(&CollidingKey("beta".into())).unwrap().1,
        &20
    );
    assert!(map.contains_key(&CollidingKey("gamma".into())));
    *map.get_mut(&CollidingKey("gamma".into())).unwrap() = 30;
    assert_eq!(map[&CollidingKey("gamma".into())], 30);

    assert_eq!(map.remove(&CollidingKey("beta".into())), Some(20));
    assert_eq!(map.remove(&CollidingKey("missing".into())), None);
    assert_eq!(map.insert(CollidingKey("delta".into()), 4), None);
    assert_eq!(
        map.to_sorted()
            .into_iter()
            .map(|(key, value)| (key.0.as_str(), *value))
            .collect::<Vec<_>>(),
        [("alpha", 1), ("delta", 4), ("gamma", 30)]
    );
}

#[test]
fn collision_chain_head_middle_tail_removal_and_slot_reuse_stay_linked() {
    let key = |name: &str| CollidingKey(name.to_owned());
    let mut map = UnordMap::from([
        (key("alpha"), 1),
        (key("beta"), 2),
        (key("gamma"), 3),
        (key("delta"), 4),
        (key("epsilon"), 5),
    ]);

    // New entries become collision-chain heads, so this removes head,
    // middle, and tail in turn before reusing every freed arena slot.
    assert_eq!(map.remove(&key("epsilon")), Some(5));
    assert_eq!(map.remove(&key("gamma")), Some(3));
    assert_eq!(map.remove(&key("alpha")), Some(1));
    assert_eq!(map.insert(key("zeta"), 6), None);
    assert_eq!(map.entry(key("eta")).or_insert(7), &7);
    map.merge(UnordMap::from([(key("theta"), 8), (key("beta"), 20)]));

    assert_eq!(
        map.into_sorted()
            .into_iter()
            .map(|(key, value)| (key.0, value))
            .collect::<Vec<_>>(),
        [
            ("beta".to_owned(), 20),
            ("delta".to_owned(), 4),
            ("eta".to_owned(), 7),
            ("theta".to_owned(), 8),
            ("zeta".to_owned(), 6),
        ]
    );
}

#[test]
fn entry_and_merge_keep_every_storage_index_in_sync() {
    let mut map = UnordMap::from([("alpha".to_owned(), 1), ("gamma".to_owned(), 3)]);
    map.entry("beta".to_owned())
        .and_modify(|_| panic!("beta must be vacant"))
        .or_insert_with_key(|key| key.len() as i32);
    map.entry("alpha".to_owned())
        .and_modify(|value| *value += 10)
        .or_default();
    assert_eq!(map.get("alpha"), Some(&11));

    match map.entry("gamma".to_owned()) {
        Entry::Occupied(mut entry) => {
            assert_eq!(entry.key(), "gamma");
            assert_eq!(entry.get(), &3);
            assert_eq!(entry.insert(30), 3);
            assert_eq!(entry.remove_entry(), ("gamma".to_owned(), 30));
        }
        Entry::Vacant(_) => panic!("gamma must be occupied"),
    }
    match map.entry("delta".to_owned()) {
        Entry::Vacant(entry) => {
            assert_eq!(entry.key(), "delta");
            *entry.insert(4) += 1;
        }
        Entry::Occupied(_) => panic!("delta must be vacant"),
    }

    map.merge(UnordMap::from([
        ("alpha".to_owned(), 100),
        ("epsilon".to_owned(), 5),
    ]));
    assert_eq!(
        map.into_sorted(),
        [
            ("alpha".to_owned(), 100),
            ("beta".to_owned(), 4),
            ("delta".to_owned(), 5),
            ("epsilon".to_owned(), 5),
        ]
    );
}

#[test]
fn set_delegates_to_the_same_collision_safe_storage() {
    let mut set = UnordSet::new();
    assert!(set.insert(CollidingKey("beta".into())));
    assert!(set.insert(CollidingKey("alpha".into())));
    assert!(!set.insert(CollidingKey("alpha".into())));
    assert!(set.contains(&CollidingKey("beta".into())));
    assert!(set.remove(&CollidingKey("beta".into())));
    assert!(!set.remove(&CollidingKey("beta".into())));
    set.merge(UnordSet::from([CollidingKey("gamma".into())]));
    assert_eq!(
        set.into_sorted()
            .into_iter()
            .map(|key| key.0)
            .collect::<Vec<_>>(),
        ["alpha".to_owned(), "gamma".to_owned()]
    );
}
