//! spec/04 section 4.7 keys a declared-result producer to a category: every
//! `Container`-domain operation of the semantic-identity registry produces the
//! tensors it returns, directly or in its aggregate result, except the
//! contiguous-selection projections `index`, `take` and `skip`. Both lanes ask
//! `produces_container_result`, which reads the builtin catalogue; this test
//! derives the same category from the registry so the two cannot drift.

use chelis_ir::host::{CONTAINER_PROJECTIONS, produces_container_result};
use std::collections::BTreeSet;

/// spec/04 section 4.7's contiguous-selection projections, stated here
/// independently of the implementation's constant.
const SPEC_PROJECTIONS: &[&str] = &["index", "skip", "take"];

fn registry_names(domain: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let registry = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../spec/registry/builtin_semantic_identities.md"
    ))
    .expect("semantic-identity registry");
    let mut inside = BTreeSet::new();
    let mut outside = BTreeSet::new();
    for line in registry.lines() {
        let Some(identity) = line
            .strip_prefix("| `")
            .and_then(|rest| rest.split_once('`'))
            .map(|(identity, _)| identity)
        else {
            continue;
        };
        let mut parts = identity.split(':');
        let (Some(row_domain), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        if row_domain == domain {
            inside.insert(name.to_string());
        } else {
            outside.insert(name.to_string());
        }
    }
    (inside, outside)
}

#[test]
fn every_registered_container_operation_but_the_projections_produces() {
    let (container, others) = registry_names("Container");
    assert!(
        container.contains("map") && container.contains("dict_values"),
        "the registry parse found no Container rows: {container:?}"
    );
    assert_eq!(CONTAINER_PROJECTIONS, SPEC_PROJECTIONS);
    for projection in SPEC_PROJECTIONS {
        assert!(
            container.contains(*projection),
            "{projection} is not a Container row"
        );
    }
    let mut wrong = Vec::new();
    for name in &container {
        let expected = !SPEC_PROJECTIONS.contains(&name.as_str());
        if produces_container_result(name) != expected {
            wrong.push(format!("{name}: expected {expected}"));
        }
    }
    for name in others.difference(&container) {
        if produces_container_result(name) {
            wrong.push(format!("{name}: not a Container row, expected false"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
