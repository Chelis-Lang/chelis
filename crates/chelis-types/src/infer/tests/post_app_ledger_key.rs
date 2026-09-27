//! chelis#1512 round 2 P2-1: the `PostApp` ledger key names the call site,
//! including when the call is re-registered from the ledger's own clone.
//!
//! The entry is keyed by the node identity of the call's `app`. On the replay
//! path the caller holds the ledger's own CLONE of that call, and a route that
//! re-registers during the replay must key its new entry by the original
//! site: `has_post_app_check_for` answering for an unrelated call, or not
//! answering for this one, is chelis#1512's own defect re-opened.
//!
//! This asserts the invariant where it is decided rather than through a
//! diagnostic: every key minted during a replay must be a key that was already
//! minted outside one, which is exactly "the original site, carried through".

use super::*;
use chelis_deep::NodeIdentity;

fn check_and_take_keys(source: &str) -> Vec<(Option<NodeIdentity>, bool)> {
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
/// A key taken from anything but the call the clone copies (the clone's own
/// address, as before, or a clone that dropped the identity) is in no eager
/// mint, and the assertion below fails.
#[test]
fn a_route_that_re_registers_during_a_replay_keeps_the_original_key() {
    let keys = check_and_take_keys(include_str!(
        "../../../tests/fixtures/core/src/post_app_ledger_key/a_route_that_re_registers_during_a_replay_keeps_the_original_key.json"
    ));
    let replayed: Vec<Option<NodeIdentity>> = keys
        .iter()
        .filter(|(_, during_replay)| *during_replay)
        .map(|(key, _)| *key)
        .collect();
    assert!(
        !replayed.is_empty(),
        "the reproduction must re-register during a replay, or it proves nothing; log {keys:?}"
    );
    let eager: Vec<Option<NodeIdentity>> = keys
        .iter()
        .filter(|(_, during_replay)| !*during_replay)
        .map(|(key, _)| *key)
        .collect();
    for key in &replayed {
        assert!(
            key.is_some() && eager.contains(key),
            "a key minted during a replay must be the original site carried through, not a \
             key of the ledger's clone; {key:?} is in no eager mint {eager:?}"
        );
    }
}

/// NEGATIVE TWIN. An ordinary eager registration keys by its own call, so two
/// distinct calls keep two distinct keys and neither is deduplicated into the
/// other.
#[test]
fn two_distinct_eager_calls_keep_distinct_keys() {
    let keys = check_and_take_keys(include_str!(
        "../../../tests/fixtures/core/src/post_app_ledger_key/two_distinct_eager_calls_keep_distinct_keys.json"
    ));
    let eager: Vec<Option<NodeIdentity>> = keys
        .iter()
        .filter(|(_, during_replay)| !*during_replay)
        .map(|(key, _)| *key)
        .collect();
    assert!(
        eager.len() >= 2,
        "two separate suspended calls are expected; log {keys:?}"
    );
    assert!(
        eager.iter().all(Option::is_some),
        "every call of the checked program is numbered; log {keys:?}"
    );
    let mut unique = eager.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        eager.len(),
        "distinct calls must keep distinct keys, got {eager:?}"
    );
}
