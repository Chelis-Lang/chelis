# Lint exception path-root divergence — diagnosis

## Symptom

`chelis lint --check crates docs examples packages` from the workspace
root reports 6 false-positive `surf-def-arrow-form` errors against
`crates/chelis-surf/tests/fixtures/*.ch`:

```
crates/chelis-surf/tests/fixtures/block_binding_expr.ch:1: surf-def-arrow-form ...
crates/chelis-surf/tests/fixtures/lambda.ch:1: surf-def-arrow-form ...
crates/chelis-surf/tests/fixtures/match_expr.ch:5: surf-def-arrow-form ...
crates/chelis-surf/tests/fixtures/operators.ch:1: surf-def-arrow-form ...
crates/chelis-surf/tests/fixtures/pipe.ch:1: surf-def-arrow-form ...
crates/chelis-surf/tests/fixtures/simple_def.ch:1: surf-def-arrow-form ...
```

The same workspace under `chelis lint --check .` reports zero
`surf-def-arrow-form` errors. The Surf parser test corpus is
intentionally exempted via the
`crates/chelis-surf/tests/fixtures/*.ch` glob in
`crates/chelis-cli/src/style_gate.rs::exceptions`, with cross-ref
§3.5. The exemption applies under the CWD walk but not under the
subtree walk.

CI invokes `chelis lint --check .` (`.github/workflows/ci.yml`), so
the gate isn't broken. The bug surfaces when developers lint a subset
of the workspace.

## Divergence point

`crates/chelis-lint/src/exceptions.rs::is_excepted` (HEAD `073f038`,
lines 90 to 106):

```rust
fn is_excepted(violation: &Violation, exceptions: &[Exception], root: &std::path::Path) -> bool {
    let rel = violation
        .path
        .strip_prefix(root)
        .unwrap_or(&violation.path)
        .to_string_lossy()
        .to_string();
    for exc in exceptions {
        if exc.rule_id != violation.rule_id {
            continue;
        }
        if glob_match(&exc.pattern, &rel) {
            return true;
        }
    }
    false
}
```

`root` is the walk-target root (post-PR #93, the canonical absolute
path of the user-supplied target). For `chelis lint --check crates`,
`root` is `/abs/path/repo/crates`. Stripping that prefix from
`/abs/path/repo/crates/chelis-surf/tests/fixtures/block_binding_expr.ch`
produces `chelis-surf/tests/fixtures/block_binding_expr.ch`, missing
the leading `crates/` segment. The exception pattern
`crates/chelis-surf/tests/fixtures/*.ch` therefore never matches.

Under `chelis lint --check .`, `root` is the canonical path of the
workspace root, and the prefix-stripped path correctly begins with
`crates/`. The exception applies.

The CLI loop at `crates/chelis-cli/src/main.rs::cmd_lint` (HEAD
`073f038`, lines 5402 to 5403) passes the per-target walk root to
`apply_exceptions`:

```rust
let raw_violations = chelis_lint::lint(target, &rules)?;
let kept = chelis_lint::exceptions::apply_exceptions(&raw_violations, &exceptions, target);
```

The corresponding lint-fix loop at lines 5454 to 5455 does the same.
Both sites pass `target` (a single walk-target root), not a shared
workspace root.

PR #93 normalized walk-target paths via `fs::canonicalize` at the CLI
boundary, which fixed the parallel `doc_filename_convention`
substring-match divergence. The exception-list layer is a separate
code path that was flagged as a sibling sweep finding in
`docs/investigations/lint_cli_path_walk_diagnosis.md` (lines 104 to
122) and recommended as a follow-on §5 entry. This document closes
that follow-on.

## Canonical behavior

Both invocations must produce identical violation sets. Exception
patterns in `style_gate::exceptions` are written relative to the
workspace root (e.g., `crates/chelis-surf/tests/fixtures/*.ch`,
`docs/book/src/*.md`), so exception matching must strip the
workspace-root prefix, not the per-target walk root.

## Unification approach

Anchor exception matching against a detected workspace root, not
against the per-target walk root. Reuse the same CLI-boundary
canonicalization pattern PR #93 established for walk targets:
canonicalize the current working directory at the CLI boundary and
pass it through `apply_exceptions` as a separate
`workspace_root` argument.

The CWD is the workspace root by construction whenever `chelis lint`
is invoked from a workspace directory: the CI gate runs from the
workspace root, and developers running `chelis lint --check crates
docs` are doing so from the workspace root as well. Treating the CWD
as the workspace root mirrors PR #93's approach (canonicalize at the
CLI boundary, do not walk the filesystem) and matches the convention
established for the standalone `chelis lint` subcommand.

`apply_exceptions` and `is_excepted` gain a `workspace_root: &Path`
argument and use it as the prefix-strip target. Callers under the
build-time style gate
(`crates/chelis-cli/src/style_gate.rs::run_lint_for_single_file` and
`crates/chelis-cli/src/main.rs::emit_advisory_lint_warnings_for_file`)
also propagate the workspace root; for those single-file paths the
parent directory is a reasonable approximation (no exception pattern
ships today that would distinguish workspace root from file parent for
single-file builds, and those flows are not the target of the bug).

If the CLI cannot canonicalize the CWD (e.g., the user has invoked
`chelis lint` from a deleted or unreadable directory), the CLI emits a
clear stderr diagnostic and falls back to the un-canonicalized CWD —
the same fallback shape PR #93 established for individual targets.
This is not a walk-up filesystem traversal: no `Cargo.toml` or `.git`
search, no walk through ancestor directories.

## Rejected alternatives

- **Walk-up filesystem traversal looking for `Cargo.toml` or `.git`.**
  Rejected per `feedback_no_walkup_filesystem_detection.md`:
  walk-up is brittle under symlinks, mounts, and permission edges, and
  it would introduce semantics different from PR #93's existing
  canonicalize-at-CLI-boundary approach. Two detection paths with
  different semantics violate the "no silent deferrals" principle.

- **Per-rule allowlist rewrite to skip the offending fixtures.**
  Rejected because it doesn't close the bug class — any future
  exception pattern with a workspace-rooted glob would suffer the same
  silent miss under subtree walks.

- **Document the limitation in the style-gate exception list.**
  Rejected because it would convert a fixable bug into a permanent
  user-visible papercut.

## Sibling sweep

`rg "strip_prefix" crates/chelis-lint/src/` surfaces:

```
crates/chelis-lint/src/exceptions.rs:93:        .strip_prefix(root)
```

This is the only `strip_prefix` site in `chelis-lint/src/` that does
walk-root-anchored path matching. No other lint helper exhibits the
same bug shape; the fix is surgical and complete.

The companion search in `chelis-cli/src/` shows two additional
`apply_exceptions` callers
(`emit_advisory_lint_warnings_for_file` at line 1333 and
`style_gate::run_lint_for_single_file` at line 182), both of which
pass `file.parent()` as the root. Those are single-file build flows
where the exception list is largely vestigial (single-file builds
don't process the multi-file workspace corpus), but they propagate
the new `workspace_root` argument for consistency.

## Test coverage

Pinned in `crates/chelis-cli/tests/lint_path_walk_consistency.rs`:

1. `lint_cli_exception_pattern_matches_under_subtree_and_cwd_walks`
   (`#[ignore]` until fix lands) — asserts both `chelis lint --check
   .` and `chelis lint --check crates` produce zero
   `surf-def-arrow-form` violations on the fixture
   `crates/chelis-surf/tests/fixtures/block_binding_expr.ch`.

2. `lint_cli_exception_pattern_matches_under_multi_subtree_walk`
   (`#[ignore]` until fix lands) — mirrors the gap-synthesis
   reproducer verbatim: `chelis lint --check crates docs examples
   packages`.

After the fix, both fixtures unignore.

## References

- gap-synthesis entry: `Lint-ExceptionPathRoot-F1` in
  `docs/gap_synthesis.md`.
- PR #93 sibling-sweep note:
  `docs/investigations/lint_cli_path_walk_diagnosis.md`, lines 104 to
  122.
- Standing rule: no walk-up filesystem detection
  (`feedback_no_walkup_filesystem_detection.md`).
