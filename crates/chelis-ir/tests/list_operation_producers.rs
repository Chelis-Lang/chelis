//! spec/04 section 4.7 names a category, not a list: every List operation of
//! [05-OP-54] and [05-OP-55] that returns an aggregate produces the tensors
//! in it, and the selections `index`, `take` and `skip` project. Both lanes
//! read `LIST_OPERATION_PRODUCERS`; this test derives the category from the
//! semantic-identity registry, so an operation added to either atom cannot
//! be missed.

use chelis_ir::host::LIST_OPERATION_PRODUCERS;
use std::collections::BTreeSet;

/// Operations of the two atoms that are not producers: the selections and
/// `len`, which returns an `i64`.
const NOT_PRODUCERS: &[&str] = &["index", "len", "skip", "take"];

#[test]
fn the_producer_set_is_the_registered_list_operation_category() {
    let registry = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/registry/builtin_semantic_identities.md"
    ))
    .expect("semantic-identity registry");
    let mut category = BTreeSet::new();
    for line in registry.lines() {
        let Some((identity, atom)) = line
            .strip_prefix("| `")
            .and_then(|rest| rest.split_once("` | "))
        else {
            continue;
        };
        if !matches!(
            atom.trim_end_matches(" |").trim(),
            "[05-OP-54]" | "[05-OP-55]"
        ) {
            continue;
        }
        let name = identity
            .split(':')
            .nth(1)
            .expect("identity is Domain:name:Case");
        category.insert(name.to_string());
    }
    assert!(
        category.contains("map") && category.contains("append"),
        "the registry parse found no [05-OP-54]/[05-OP-55] rows: {category:?}"
    );
    let expected: BTreeSet<String> = category
        .into_iter()
        .filter(|name| !NOT_PRODUCERS.contains(&name.as_str()))
        .collect();
    let actual: BTreeSet<String> = LIST_OPERATION_PRODUCERS
        .iter()
        .map(|name| name.to_string())
        .collect();
    assert_eq!(actual, expected);
}
