use chelis_unord::ScopedMap;

fn snapshot(map: &ScopedMap<String, i32>, keys: &[&str]) -> Vec<Option<i32>> {
    keys.iter().map(|key| map.get(*key).copied()).collect()
}

#[test]
fn a_scope_restores_every_entry_it_displaced() {
    let mut map: ScopedMap<String, i32> = [("a".to_owned(), 1), ("b".to_owned(), 2)]
        .into_iter()
        .collect();
    let keys = ["a", "b", "c"];
    let before = snapshot(&map, &keys);
    let inside = map.scope(|map| {
        map.insert("a".to_owned(), 10);
        map.insert("c".to_owned(), 30);
        map.remove("b");
        map.insert("a".to_owned(), 11);
        snapshot(map, &keys)
    });
    assert_eq!(inside, vec![Some(11), None, Some(30)]);
    assert_eq!(snapshot(&map, &keys), before);
    assert_eq!(map.len(), 2);
}

#[test]
fn nested_scopes_unwind_innermost_first() {
    let mut map = ScopedMap::new();
    map.insert("x".to_owned(), 0);
    map.scope(|map| {
        map.insert("x".to_owned(), 1);
        map.scope(|map| {
            map.remove("x");
            map.insert("y".to_owned(), 2);
            assert_eq!(map.get("x"), None);
        });
        assert_eq!(map.get("x"), Some(&1));
        assert!(!map.contains_key("y"));
    });
    assert_eq!(map.get("x"), Some(&0));
}

#[test]
fn a_change_outside_every_scope_is_permanent() {
    let mut map = ScopedMap::new();
    map.insert("x".to_owned(), 0);
    map.remove("x");
    map.insert("y".to_owned(), 1);
    map.scope(|_| {});
    assert_eq!(map.get("x"), None);
    assert_eq!(map.get("y"), Some(&1));
}

#[test]
fn removing_an_absent_key_in_a_scope_restores_nothing() {
    let mut map = ScopedMap::new();
    map.insert("x".to_owned(), 0);
    map.scope(|map| map.remove("missing"));
    assert_eq!(map.get("x"), Some(&0));
    assert_eq!(map.len(), 1);
}
