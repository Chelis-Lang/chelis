//! Lock invariant: every name in `BUILTIN_NAMES` must either have a
//! dispatch arm in `eval_builtin` (in `crates/chelis-compiler-api/src/
//! runtime/eval.rs`) OR appear in the documented allowlist below.
//!
//! The allowlist is for builtins that exist only at type-check / IR-emit
//! time and never reach the host-runtime evaluator. Each entry MUST have
//! a `//` comment immediately above it stating the reason, drawn from
//! the closed vocabulary:
//!
//!   - `type-only` exists only for typing or signature lookups
//!   - `target=<backend>-only` only emitted on a specific backend target
//!   - `lowered-before-eval` IR lowering converts the call to a primitive
//!     RISC op before host eval runs
//!   - `pending-spec-stability` name reserved but semantics not yet
//!     pinned; do not implement on best-guess
//!
//! Issue Chelis-Lang/chelis#185 closed the original 12-builtin gap plus a
//! sibling-sweep handful; this invariant test catches future regressions
//! where someone adds a name to `BUILTIN_NAMES` without wiring the
//! host-runtime arm.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use chelis_types::BUILTIN_NAMES;

/// Builtins that legitimately do not need a host-runtime arm. Document
/// the reason per the closed vocabulary above.
const HOST_RUNTIME_ALLOWLIST: &[&str] = &[
    // `lowered-before-eval` — `scatter_replace` (Surf-facing sparse op)
    // always lowers to `RiscOp::Scatter` in
    // `crates/chelis-ir/src/lower.rs::lower_scatter_replace_uses_sparse_ir_node`.
    // The host runtime never sees the symbolic builtin call; the DAG
    // evaluator handles `RiscOp::Scatter` directly.
    "scatter_replace",
    // `pending-spec-stability` — `normalize` is marked unstable in
    // `crates/chelis-types/src/builtins.rs` and a separate follow-up
    // issue tracks its semantics. Do not implement on best-guess.
    "normalize",
];

fn runtime_source_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("src").join("runtime").join("eval.rs")
}

/// Extract every dispatch arm name from the `eval_builtin` match
/// statement. We scan the runtime source between the function signature
/// and the next `fn` definition (or end of file — `eval_builtin` is
/// currently the last item in `runtime/eval.rs`) for `"name" =>`
/// patterns.
fn extracted_dispatched_names() -> HashSet<String> {
    let source = fs::read_to_string(runtime_source_path())
        .expect("runtime/eval.rs must be readable from CARGO_MANIFEST_DIR/src");
    let start = source
        .find("fn eval_builtin")
        .expect("runtime/eval.rs must contain fn eval_builtin");
    let after_start = start + "fn eval_builtin".len();
    let end = [
        "\nfn ",
        "\n    fn ",
        "\n    pub(super) fn ",
        "\n    pub(crate) fn ",
    ]
    .iter()
    .filter_map(|marker| source[after_start..].find(marker))
    .min()
    .map(|offset| after_start + offset)
    .unwrap_or(source.len());
    let body = &source[start..end];

    let mut out = HashSet::new();
    for line in body.lines() {
        let trimmed = line.trim_start();
        // Match `"name" =>` at the start of a (possibly-indented) line.
        // Names are ASCII letters/digits/underscores; we don't need to
        // unify multi-byte UTF-8 here.
        if let Some(rest) = trimmed.strip_prefix('"')
            && let Some(end_q) = rest.find('"')
        {
            let name = &rest[..end_q];
            let after = rest[end_q + 1..].trim_start();
            if after.starts_with("=>")
                && name
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                out.insert(name.to_string());
            }
        }
    }
    out
}

#[test]
fn every_builtin_name_has_a_host_runtime_arm_or_an_allowlist_entry() {
    let dispatched = extracted_dispatched_names();
    let allowlisted: HashSet<&str> = HOST_RUNTIME_ALLOWLIST.iter().copied().collect();

    let mut missing: Vec<&str> = BUILTIN_NAMES
        .iter()
        .copied()
        .filter(|name| !dispatched.contains(*name) && !allowlisted.contains(name))
        .collect();
    missing.sort_unstable();

    assert!(
        missing.is_empty(),
        "the following BUILTIN_NAMES entries have no eval_builtin arm and \
         no allowlist entry: {missing:?}. Add a dispatch arm in \
         crates/chelis-compiler-api/src/runtime/eval.rs::eval_builtin OR add an \
         allowlist entry to HOST_RUNTIME_ALLOWLIST in this test with a \
         documented reason."
    );
}

#[test]
fn allowlist_does_not_shadow_any_dispatched_name() {
    // A name should not appear in BOTH the dispatch arms AND the
    // allowlist. If it does, the allowlist comment is stale and the
    // reason no longer applies. Catch the drift early.
    let dispatched = extracted_dispatched_names();
    let mut overlap: Vec<&str> = HOST_RUNTIME_ALLOWLIST
        .iter()
        .copied()
        .filter(|name| dispatched.contains(*name))
        .collect();
    overlap.sort_unstable();

    assert!(
        overlap.is_empty(),
        "the following names appear in BOTH the dispatch arms AND the \
         allowlist: {overlap:?}. Remove them from HOST_RUNTIME_ALLOWLIST \
         since they now have a real host-runtime arm."
    );
}

#[test]
fn allowlist_entries_are_in_builtin_names() {
    // Sanity: an allowlist entry that isn't in BUILTIN_NAMES is dead
    // configuration. The invariant the allowlist documents is "the
    // following BUILTIN_NAMES entries are intentionally not dispatched";
    // an entry that no longer appears in BUILTIN_NAMES has no
    // referent.
    let builtin_set: HashSet<&str> = BUILTIN_NAMES.iter().copied().collect();
    let mut stale: Vec<&str> = HOST_RUNTIME_ALLOWLIST
        .iter()
        .copied()
        .filter(|name| !builtin_set.contains(name))
        .collect();
    stale.sort_unstable();

    assert!(
        stale.is_empty(),
        "the following allowlist entries are not in BUILTIN_NAMES (likely \
         the name was removed without updating the allowlist): {stale:?}"
    );
}
