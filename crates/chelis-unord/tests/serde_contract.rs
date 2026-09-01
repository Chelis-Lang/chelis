use chelis_unord::{UnordMap, UnordSet};

fn map_in_order(entries: &[(&str, i32)]) -> UnordMap<String, i32> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), *value))
        .collect()
}

fn set_in_order(entries: &[&str]) -> UnordSet<String> {
    entries.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn map_serialization_is_canonical_and_round_trips() {
    let canonical = map_in_order(&[("alpha", 1), ("beta", 2), ("gamma", 3)]);
    let reversed = map_in_order(&[("gamma", 3), ("beta", 2), ("alpha", 1)]);
    let shuffled = map_in_order(&[("beta", 2), ("alpha", 1), ("gamma", 3)]);

    let expected = br#"[["alpha",1],["beta",2],["gamma",3]]"#;
    for value in [&canonical, &reversed, &shuffled] {
        assert_eq!(serde_json::to_vec(value).unwrap(), expected);
    }
    let decoded: UnordMap<String, i32> = serde_json::from_slice(expected).unwrap();
    assert_eq!(decoded, canonical);
}

#[test]
fn set_serialization_is_canonical_and_round_trips() {
    let canonical = set_in_order(&["alpha", "beta", "gamma"]);
    let reversed = set_in_order(&["gamma", "beta", "alpha"]);
    let shuffled = set_in_order(&["beta", "alpha", "gamma"]);

    let expected = br#"["alpha","beta","gamma"]"#;
    for value in [&canonical, &reversed, &shuffled] {
        assert_eq!(serde_json::to_vec(value).unwrap(), expected);
    }
    let decoded: UnordSet<String> = serde_json::from_slice(expected).unwrap();
    assert_eq!(decoded, canonical);
}

#[test]
fn duplicate_map_key_is_rejected() {
    let error =
        serde_json::from_str::<UnordMap<String, i32>>(r#"[["duplicate",1],["duplicate",2]]"#)
            .unwrap_err();
    assert!(error.to_string().contains("duplicate key"), "{error}");
}

#[test]
fn duplicate_set_element_is_rejected() {
    let error =
        serde_json::from_str::<UnordSet<String>>(r#"["duplicate","duplicate"]"#).unwrap_err();
    assert!(error.to_string().contains("duplicate element"), "{error}");
}

#[test]
fn debug_is_deterministic_and_does_not_expose_contents() {
    let map = map_in_order(&[("alpha", 1), ("beta", 2)]);
    let set = set_in_order(&["alpha", "beta"]);
    assert_eq!(
        format!("{map:?}"),
        "UnordMap { len: 2 }",
        "Debug must not expose the private collection's contents"
    );
    assert_eq!(
        format!("{set:?}"),
        "UnordSet { len: 2 }",
        "Debug must not expose the private collection's contents"
    );
}

#[test]
fn ordered_exit_uses_key_order_without_a_projection_callback() {
    let map = map_in_order(&[("gamma", 3), ("alpha", 1), ("beta", 2)]);
    let set = set_in_order(&["gamma", "alpha", "beta"]);
    assert_eq!(
        map.to_sorted()
            .into_iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect::<Vec<_>>(),
        [("alpha", 1), ("beta", 2), ("gamma", 3)]
    );
    assert_eq!(
        set.into_sorted(),
        ["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()]
    );
}
