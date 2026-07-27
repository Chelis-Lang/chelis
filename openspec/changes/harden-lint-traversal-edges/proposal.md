# Proposal: harden-lint-traversal-edges

## Why

Review of PR #839 found two edges where the new traversal policy is quieter than its own fail-closed philosophy: an explicitly named lint root that exists but is inadmissible (special file, or a symlink directory escaping the policy root) produces a silent empty walk and `chelis lint --check` exits 0 green, while a nonexistent root fails loudly; and repository policy discovery walks `target.ancestors()` on the path as given, so a relative target from a subdirectory cwd (`cd tests && chelis lint .`) never finds the repo-root `chelis-lint.toml` and repository exclusions silently do not apply — contradicting the §12.2 claim that local and CI scope are identical.

## What Changes

- **BREAKING** (lint library + CLI semantics): an explicitly named lint root that exists but is rejected by depth-zero admission (socket/FIFO/device, or a link resolving to a different kind or outside the repository policy boundary) SHALL fail the lint invocation loudly with the root path and rejection reason, instead of returning a successful empty entry set. Discovered (non-explicit) inadmissible entries remain silently omitted, unchanged.
- Repository policy discovery SHALL be cwd-insensitive: the ancestor search for `chelis-lint.toml` resolves the lint target against the current working directory before walking ancestors, so relative targets from any cwd inside a policy root discover the same policy as absolute targets.
- One `TraversalPolicy` load SHALL be shared across a single `lint()` invocation (walker plus rule `prepare_run` hooks) instead of re-reading and re-parsing the policy TOML and its spec document two to three times per invocation. Implementation-only; no requirement change.
- The walker's post-yield admission recheck is scoped to depth zero, where the `ignore` crate gap it compensates for actually exists, removing a redundant canonicalize+stat round per discovered symlink. Implementation-only; no requirement change.
- `spec/01-nomenclature.md` §12.2, `docs/book/src/cli.md`, and the CLI/lib test suites are updated in the same change set; tests currently locking in silent-empty explicit-root behavior flip to asserting the loud failure.
- A quiet-machine whole-repository `chelis lint --check .` timing run is recorded as executable evidence for the "structurally linear" claim from #603/#839.

Out of scope: PR-hygiene items from the same review (disclosing the `make_existing_copy_destination_writable` and `reef_setup.rs` drive-bys in the PR body) — those are addressed on the PR itself, not by spec work.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `lint-traversal-policy`:
  - "Explicit targets override only traversal exclusions at depth zero" — rejection of an existing explicit root becomes a loud lint failure naming the root and reason, not a silent empty result; the nonexistent-root and inadmissible-root outcomes become consistent (both loud).
  - "Traversal exclusions come from structured policy" — nearest-ancestor `chelis-lint.toml` discovery is defined over the cwd-resolved target, so relative and absolute spellings of the same target resolve the same policy.

## Impact

- `crates/chelis-lint/src/policy.rs` — cwd-resolved discovery start; new depth-zero rejection error variant(s) carrying path and reason.
- `crates/chelis-lint/src/walker.rs` — loud depth-zero rejection; recheck scoped to depth zero; shared policy handle.
- `crates/chelis-lint/src/lib.rs` — `lint()` threads one loaded policy into walk and `prepare_run`.
- `crates/chelis-lint/src/rules/doc_filename_convention.rs` — consumes the shared policy instead of reloading.
- `crates/chelis-cli` — standalone `lint` and style-gate integration tests for the new loud failure and cwd-insensitive discovery.
- `crates/chelis-lint/tests/traversal_policy.rs`, `crates/chelis-cli/tests/lint_traversal_policy.rs` — flip silent-empty assertions; add subdirectory-cwd discovery probes.
- `spec/01-nomenclature.md` §12.2, `docs/book/src/cli.md`, `openspec/specs/lint-traversal-policy/spec.md` (delta), `CHANGELOG.md`.
