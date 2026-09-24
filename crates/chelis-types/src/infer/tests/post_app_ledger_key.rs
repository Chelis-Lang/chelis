//! chelis#1512 round 2 P2-1: the `PostApp` ledger key names a node that
//! outlives its entry.
//!
//! The entry is keyed by the address of the call's `deep::List`. On the replay
//! path the caller holds the ledger's own CLONE of that list, which is dropped
//! when the replay iteration ends, so a route that re-registers during a
//! replay used to store an address freed moments later. Nothing observed a
//! collision, but `has_post_app_check_for` answering for an unrelated live
//! call is chelis#1512's own defect re-opened, and a documented invariant that
//! the code does not hold is worth closing on its own terms.
//!
//! This asserts the invariant where it is decided rather than through a
//! diagnostic: every key minted during a replay must be a key that was already
//! minted outside one, which is exactly "the original site, carried through".

use super::*;

fn check_and_take_keys(source: &str) -> Vec<(usize, bool)> {
    let _ = take_post_app_key_log();
    let exprs: Vec<deep::Expr> = serde_json::from_str(source).expect("lossless AST fixture");
    let _ = crate::check_typed_program(
        &exprs,
        crate::ProgramOrigin::Unlinked,
        &mut crate::NativeServices::new(None),
    );
    take_post_app_key_log()
}

/// REGRESSION TEST. The reviewer's own reproduction: `split`'s sizes operand is
/// `List[Var]`, so the readiness predicate passes at the top level while the
/// element variable is still free, the replay fires, and the arm registers
/// again from inside it.
///
/// Watched RED against the previous spelling: with the key taken from the list
/// the replay holds, the second mint is the clone's address, which appears in
/// no eager mint, and the `is_carried` assertion below fails.
#[test]
fn a_route_that_re_registers_during_a_replay_keeps_the_original_key() {
    let keys = check_and_take_keys(include_str!(
        "../../../tests/fixtures/core/src/post_app_ledger_key/a_route_that_re_registers_during_a_replay_keeps_the_original_key.json"
    ));
    let replayed: Vec<usize> = keys
        .iter()
        .filter(|(_, during_replay)| *during_replay)
        .map(|(key, _)| *key)
        .collect();
    assert!(
        !replayed.is_empty(),
        "the reproduction must re-register during a replay, or it proves nothing; log {keys:?}"
    );
    let eager: Vec<usize> = keys
        .iter()
        .filter(|(_, during_replay)| !*during_replay)
        .map(|(key, _)| *key)
        .collect();
    for key in &replayed {
        assert!(
            eager.contains(key),
            "a key minted during a replay must be the original site carried through, not the \
             address of the ledger's clone; {key:#x} is in no eager mint {eager:#x?}"
        );
    }
}

/// NEGATIVE TWIN. The translation must apply ONLY to the clone the replay
/// holds. An ordinary eager registration, with no replay in flight, still keys
/// by its own node, so two distinct calls keep two distinct keys and neither
/// is deduplicated into the other.
#[test]
fn two_distinct_eager_calls_keep_distinct_keys() {
    let keys = check_and_take_keys(include_str!(
        "../../../tests/fixtures/core/src/post_app_ledger_key/two_distinct_eager_calls_keep_distinct_keys.json"
    ));
    let eager: Vec<usize> = keys
        .iter()
        .filter(|(_, during_replay)| !*during_replay)
        .map(|(key, _)| *key)
        .collect();
    assert!(
        eager.len() >= 2,
        "two separate suspended calls are expected; log {keys:?}"
    );
    let mut unique = eager.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        eager.len(),
        "distinct calls must keep distinct keys, got {eager:#x?}"
    );
}
