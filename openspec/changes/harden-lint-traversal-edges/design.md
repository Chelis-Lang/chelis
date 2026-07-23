# Design: harden-lint-traversal-edges

## Context

PR #839 introduced the deterministic traversal policy (`crates/chelis-lint/src/policy.rs`) and the policy-driven walker (`crates/chelis-lint/src/walker.rs`). Review found two edges where the shipped behavior is quieter than the fail-closed contract the PR itself established:

1. `walker::walk` returns `Ok(vec![])` when an explicitly named root exists but fails depth-zero admission (special file, or a symlink directory escaping the policy boundary). `chelis lint --check <root>` then exits 0 with no output. Tests currently lock this silence in (`explicit_source_shaped_special_entry_is_omitted`, `explicit_symlink_directory_root_must_resolve_inside_policy_boundary`). A *nonexistent* root, by contrast, fails loudly — an inconsistent pair.
2. `find_repository_policy` walks `target.ancestors()` on the path as given. `Path::ancestors` on a relative path cannot climb above the cwd, so `cd tests && chelis lint .` never discovers the repo-root `chelis-lint.toml`; repository exclusions silently do not apply, which contradicts §12.2's "local and CI scope is identical".

Two smaller review nits ride along as implementation-only cleanups: `TraversalPolicy::load_for` runs two to three times per `lint()` invocation (walker plus `doc-filename-convention`'s `prepare_run`), each re-reading the policy TOML and re-parsing the full spec markdown for cross-ref verification; and the walker's post-yield admission recheck runs for every entry although the `ignore`-crate gap it compensates for exists only at depth zero, costing a redundant canonicalize+stat per discovered symlink.

## Goals / Non-Goals

**Goals:**

- Make rejection of an existing explicit root loud and consistent with the nonexistent-root failure, end to end (lib error, CLI exit code and message).
- Make repository policy discovery independent of cwd and path spelling.
- Load the traversal policy once per `lint()` invocation and share it.
- Scope the walker's post-yield recheck to depth zero.
- Record one quiet-machine whole-repo lint timing as executable evidence for the #603 linearity claim.

**Non-Goals:**

- No change to discovered-entry semantics: non-explicit inadmissible entries remain silently omitted.
- No change to the policy schema, exclusion classes, or pattern semantics.
- No change to the style gate's fail-closed handling of malformed policies (already loud via the synthesized `lint-traversal-policy` violation).
- PR-hygiene items from the review (disclosing the `chelis-cli` drive-by fixes) are handled on PR #839 itself.

## Decisions

### D1: Loud rejection via a new `TraversalPolicyError` variant, raised in `walk()`

`walk()` currently short-circuits: `if symlink_metadata(root).is_ok() && !policy.is_admitted_explicit_entry(root, ...) { return Ok(Vec::new()) }`. Replace the empty return with `Err(TraversalPolicyError::InadmissibleExplicitRoot { path, reason })`, where `reason` distinguishes at least: non-regular entry kind (socket/FIFO/device), link resolving to a mismatched kind, link resolving outside the repository policy boundary, and unresolvable link. `is_admitted_explicit_entry` grows a variant-returning sibling (or is changed to return `Result<(), Reason>`) so the reason is computed where the checks already run, not re-derived.

*Alternative considered:* synthesizing a `Violation` (like the style gate does for policy errors). Rejected — a rejected root is not a per-file style finding; it is an invocation-level failure, and `LintError::Policy` already flows to a nonzero CLI exit with a message. The nonexistent-root case already errors through `LintError::Walk`; this makes the pair symmetric.

*Breaking surface:* `walker::walk` and `lint()` callers that relied on silent-empty. The only in-tree caller with that reliance is the test suite; the style gate already surfaces `LintError` loudly.

*Discovered during apply:* `cmd_lint` canonicalized targets at the CLI boundary (for path-substring rule dispatch), which resolved symlinks and erased a link root's identity before the walker's boundary check ran — an escaping link target linted its resolved external tree as a loose target and exited 0. The CLI now uses `std::path::absolute` instead of `fs::canonicalize`: same normalization benefit, no symlink resolution, so the depth-zero check sees the link.

### D2: cwd-resolve the discovery start, keep matcher roots lexical

In `TraversalPolicy::load_for`, compute `search_start = if target.is_absolute() { start } else { std::env::current_dir()?.join(start) }` and run `find_repository_policy` over that absolute path. Everything downstream (baseline root, matcher anchoring, `path_relative_to_scope`) keeps working on the lexical paths it already handles; only the *ancestor search* is absolutized. This preserves the existing behavior of relative walk roots producing relative entry paths (entries and matcher roots stay consistent), while fixing discovery.

*Alternative considered:* canonicalizing the target outright and walking canonical ancestors. Rejected — canonicalization changes entry-path presentation and symlink semantics that the current test corpus pins down (link-path classification, alias detection). Joining cwd is sufficient for discovery and touches nothing else.

*Consequence to verify in tests:* `path_relative_to_scope` already handles the absolute-scope-vs-relative-path mix via its `current_dir().join(scope)` branch; the new discovery path must not break the existing `relative_subdirectory_target_uses_workspace_anchoring` probe, and a new child-process probe covers cwd *below* the policy root.

### D3: Share one `Arc<TraversalPolicy>` per lint invocation

`lint()` loads the policy once, passes it to a new `walk_with_policy(root, &policy)` (thin wrapper; `walk()` remains as the compatibility entry that loads then delegates), and exposes it to rules through `prepare_run`. Rather than widening the `Rule::prepare_run` signature (churn for every rule), `doc-filename-convention`'s `prepare_run` receives the policy via a new optional parameter object — decision: extend `prepare_run(&self, root, entries)` to `prepare_run(&self, root, entries, policy: &TraversalPolicy)`. The trait is crate-internal-plus-tests; all in-tree implementors are updated in the same change set. Direct `Rule::check` fallback paths keep loading on demand (unchanged semantics, cold path only).

*Alternative considered:* a lazy `OnceLock` cache keyed by root inside `policy.rs`. Rejected — process-global caching breaks the locked invariant that repeated `lint()` invocations observe filesystem edits (`repeated_lint_invocations_rebuild_opaque_catalog_state`), and invalidation-by-mtime is complexity this does not need.

### D4: Depth-zero-only recheck in the walk loop

The post-yield `if !is_admitted(&policy, &entry) { continue; }` guard exists because `ignore` may yield an explicitly supplied root that `filter_entry` rejected. Scope it: `if entry.depth() == 0 && !is_admitted(...)`. Depth>0 entries were already vetted by `filter_entry`. This removes a second canonicalize+stat round per discovered symlink. Locked by the existing symlink corpus staying green plus a regression test asserting one admission evaluation per non-root entry if cheap to instrument; otherwise the existing behavioral tests suffice as the oracle.

### D5: Spec text lands in both governing documents

`spec/01-nomenclature.md` §12.2 gains: explicit-root rejection is loud with path and reason; discovery is cwd-resolved. The `openspec/specs/lint-traversal-policy/spec.md` delta in this change carries the same two requirement modifications, keeping the documentation hierarchy in agreement rather than adding a third explanation.

## Risks / Trade-offs

- [Loud rejection breaks a workflow that linted globs including special files] → Only *explicit* roots are affected; discovered entries stay silent. The CLI lints paths the user named — failing loudly on a named-but-unreadable root is the intended contract, and matches the nonexistent-path behavior users already see.
- [cwd-joining changes policy resolution for existing relative-target invocations from subdirectory cwds] → That is the fix, not a side effect: those invocations previously got shipped-baseline-only silently. The blast radius is exactly the buggy case. CI invokes lint from the repo root, so CI scope is unchanged.
- [`prepare_run` signature change ripples through every rule] → Mechanical; the default implementation keeps ignoring the parameter, so only `doc-filename-convention` meaningfully changes.
- [Depth-zero-only recheck could regress if a future `ignore` version changes filter semantics] → The migration-parity and symlink test corpus is the tripwire; the drift test pattern (`directory_only_fast_path_matches_ignore_parser_edges`) shows the precedent.

## Migration Plan

Land on top of PR #839's branch (or as an immediate follow-up PR if #839 merges first). No data or install migration. Rollback is a revert; no persisted state.

## Open Questions

- Whether `chelis lint` should print the rejection reason on stderr in the same one-issue-per-line diagnostic shape the style gate uses, or as a plain error line — resolve during implementation against existing CLI error formatting tests.
