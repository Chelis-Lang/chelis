//! Canonical JSON ordering uses checked runtime scratch.
const HOST: &str = include_str!("../src/host_emit.rs");
fn checked_json(source: &str) -> bool {
    let start = source
        .find("static chelis_list *chelis_json_canonical_object_entries(")
        .unwrap();
    let body = &source[start
        ..source[start..]
            .find("\n        \"    return result;\"")
            .unwrap()
            + start];
    let steps = [
        "chelis_alloc(1, &len, CHELIS_DTYPE_I64)",
        "chelis_tensor_begin_write(order_storage)",
        "chelis_tensor_write_view(order_guard)",
        "order[index] = index;",
        "chelis_list_index(source, order[index])",
        "chelis_tensor_end_write(order_guard)",
        "chelis_tensor_release(order_storage)",
    ];
    let positions: Option<Vec<_>> = steps
        .iter()
        .map(|step| (body.matches(step).count() == 1).then(|| body.find(step).unwrap()))
        .collect();
    // The complete generated tail follows the ordering loop. Exact statements
    // reject both conditional cleanup and cleanup moved into either loop.
    let tail = "        \"    }\",\n        \"    chelis_tensor_end_write(order_guard);\",\n        \"    chelis_tensor_release(order_storage);\",\n        \"    chelis_list_release(source);\",";
    positions.is_some_and(|p| p.windows(2).all(|pair| pair[0] < pair[1]))
        && body.ends_with(tail)
        && !body.contains("malloc(")
        && !body.contains("sizeof(")
}
#[test]
fn json_order_scratch_has_checked_capacity_and_balanced_owner() {
    assert!(checked_json(HOST));
}
#[test]
fn json_scratch_bypass_and_lifetime_controls_fail() {
    for required in [
        "chelis_alloc(",
        "chelis_tensor_begin_write(",
        "chelis_tensor_write_view(",
        "chelis_tensor_end_write(",
        "chelis_tensor_release(",
    ] {
        assert!(
            !checked_json(&HOST.replace(required, "REMOVED(")),
            "{required}"
        );
    }
}

#[test]
fn json_scratch_early_or_duplicate_release_is_rejected() {
    let release = "chelis_tensor_release(order_storage)";
    let early = HOST.replace(release, "REMOVED").replace(
        "order[index] = index;",
        &format!("{release}; order[index] = index;"),
    );
    assert!(!checked_json(&early));
    assert!(!checked_json(
        &HOST.replace(release, &format!("{release}; {release}"))
    ));
}

#[test]
fn json_scratch_conditional_release_is_rejected() {
    let bad = HOST.replace(
        "chelis_tensor_release(order_storage);",
        "if (len == 0) chelis_tensor_release(order_storage);",
    );
    assert!(!checked_json(&bad));
}
