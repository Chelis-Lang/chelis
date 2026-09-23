//! Semantic identity discovery has unique values and canonical names.
use chelis_ir::dag::RiscAtomIdentity;
use std::collections::BTreeSet;

#[test]
fn semantic_identity_enumeration_has_unique_values_and_names() {
    let actual: BTreeSet<_> = RiscAtomIdentity::ALL
        .iter()
        .map(|v| format!("{v:?}"))
        .collect();
    assert_eq!(
        actual.len(),
        RiscAtomIdentity::ALL.len(),
        "duplicate identity"
    );
    let canonical: BTreeSet<_> = RiscAtomIdentity::ALL.iter().map(|v| v.as_str()).collect();
    assert_eq!(
        canonical.len(),
        actual.len(),
        "canonical identity collision"
    );
}
