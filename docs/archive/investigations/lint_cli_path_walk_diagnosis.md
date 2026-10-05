# Lint CLI path-walk divergence — diagnosis

## Symptom

`chelis lint --check src/ tests/ verify/ scripts/ docs/` reports 28 errors
against the hello-chelis corpus, but `chelis lint --check .` reports 32 errors
on the same tree. The 4-error discrepancy is `doc-filename-convention` (§8.3)
violations under `docs/`: the rule fires on the CWD walk but not the
explicit-path walk, even though `docs/` is explicitly named.

Reproducible in any tree containing a docs file whose name violates §8.3
(snake_case for narrative docs, SCREAMING_SNAKE_CASE for status reports).
A minimal fixture: `docs/0_leading_digit.md` (leading digit).

## Divergence point

`crates/chelis-lint/src/rules/doc_filename_convention.rs::classify_doc`,
specifically the path-string substring checks at lines 55, 58, 73, and 86:

```rust
let s = path.to_string_lossy();
if s.contains("/spec/design/") { return Slot::SpecDesign; }
if let Some(idx) = s.find("/spec/") { ... }
...
if s.contains("/docs/") { return Slot::Docs; }
```

and

```rust
fn is_inside_mdbook_tree(path: &Path) -> bool {
    let s = path.to_string_lossy();
    s.contains("/docs/src/") || s.contains("/docs/book/src/")
}
```

These substring patterns require the directory name to appear between
forward slashes. The walker (`crates/chelis-lint/src/walker.rs::walk`) uses
`walkdir::WalkDir::new(root)`, which prefixes every yielded path with the
root path it was given:

- `chelis lint --check .` invokes `walk(Path::new("."))`. WalkDir yields
  paths like `./docs/0_leading_digit.md`. The substring `/docs/` matches
  the literal `/docs/` between the leading `.` and the filename, so the
  rule classifies the file as `Slot::Docs` (§8.3) and fires the violation.

- `chelis lint --check docs/` invokes `walk(Path::new("docs/"))`. WalkDir
  yields paths like `docs/0_leading_digit.md`. The substring `/docs/`
  does NOT match anywhere in this path (the path starts with `docs/`, no
  leading slash). The rule falls through to `Slot::Other` and no
  violation fires.

The CLI loop at `crates/chelis-cli/src/main.rs::cmd_lint` (around line
5105) takes the `paths` argument list verbatim, defaulting to
`vec![PathBuf::from(".")]` when no paths are given, and passes each
target straight to `chelis_lint::lint(target, &rules)`. No
normalization or canonicalization happens at the CLI boundary, so the
exact textual form of each user-supplied path propagates into every
rule's path-classification logic.

`crates/chelis-lint/src/exceptions.rs::is_excepted` already strips the
`root` prefix before glob-matching, so exception application is robust to
the prefix mismatch. The asymmetry is exclusively in the rule's
self-classification, not in the exceptions layer.

## Canonical behavior

The CI gate runs `chelis lint --check .` (`.github/workflows/ci.yml:42`),
so the CWD-walk behavior is the version the workstream and hello-chelis
corpus baselines are calibrated against. The explicit-path walk must
behave identically: the user's choice of how to spell the path on the
command line must not affect which rules apply.

## Unification approach

Normalize at the CLI boundary: convert each user-supplied target to an
absolute path (via `std::fs::canonicalize`) before invoking
`chelis_lint::lint`. Absolute paths always include the canonical
ancestor directory segments, so the `/docs/`, `/spec/`, and
`/spec/design/` substring checks fire uniformly whether the user typed
`.`, `./docs`, `docs`, or an absolute path ending in `/repo/docs`.

This fixes the entire bug class for any current or future rule that does
path-substring classification, not just `doc-filename-convention`.

If `canonicalize` fails (path does not exist, permission denied,
non-Unicode component), fall back to the original path with a stderr
warning. Today's behavior on a non-existent path is to ask the walker to
yield zero entries (no error); the canonicalization fallback preserves
this so any existing scripts that rely on the silent behavior keep
working. The diagnostic to stderr signals the user that path resolution
soft-failed without changing exit codes.

An alternative (fix the substring patterns inside the rule) was rejected
because it leaves the bug class open for any future rule that needs
path-segment dispatch.

## Sibling sweep targets

- Other lint rules that do `path.to_string_lossy()` substring matching:
  ripgrep confirms `doc_filename_convention.rs` is the only rule using
  the pattern today.

- Exception-pattern matching in `crates/chelis-lint/src/exceptions.rs::is_excepted`:
  the function computes `violation.path.strip_prefix(root)` to obtain a
  path that exception globs match against. When the CLI passes a
  sub-directory as the walk root (e.g., `chelis lint --check crates`),
  the strip-prefix produces paths like `chelis-surf/tests/fixtures/x.ch`
  that no longer match workspace-rooted patterns like
  `crates/chelis-surf/tests/fixtures/*.ch`. This is a parallel
  bug-class to the rule-classification divergence: same root cause
  (CLI spelling of paths leaks into downstream matching), different
  consequence (exception patterns silently fail to apply for
  sub-directory walks instead of doc-filename-convention silently
  failing to fire). Confirmed in chelis-macro-fix's own corpus today:
  `chelis lint --check crates docs examples packages` from the
  workspace root reports 6 `surf-def-arrow-form` errors against
  `crates/chelis-surf/tests/fixtures/*.ch`, while `chelis lint --check .`
  excepts them. Out of scope for this PR (the brief targets Item 7's
  doc-filename-convention divergence); recommended as a follow-on §5
  entry. The clean fix is to root all exception matching against the
  detected workspace root rather than the CLI-supplied walk root.

- Walker root-skip filter at `crates/chelis-lint/src/walker.rs::is_skip_dir`:
  the `/.claude/worktrees/` substring filter was added defensively for
  the case where lint is run on the canonical repo and would otherwise
  descend into nested worktrees. After canonicalize-at-CLI-boundary,
  running lint from inside a worktree means the canonicalized walk
  root itself contains `/.claude/worktrees/`, and every descendant
  matches the substring filter. Fix included in this PR: also skip the
  filter for the walk-root entry itself (depth 0), and use the
  path-relative-to-root for the substring check so the filter only
  matches entries that are NESTED inside the walk, not the walk root.

- Other CLI subcommands with `paths: Vec<PathBuf>` arguments:
  - `chelis fmt PATH` — operates on a single user-supplied file/dir;
    rules see the exact path string in `chelis-surf::format`. Less
    likely to depend on substring-based dispatch, but the same path-arg
    spelling sensitivity exists in principle.
  - `chelis check`, `chelis build`, `chelis eval --file` — file-level,
    not directory-walk; the path-walk inconsistency does not apply.
  - `chelis surf`, `chelis deep` — file-level decompilers; same.

  Recommend a follow-on §5 entry if any of those exhibit the same
  bug-class. `chelis fmt` is the most likely candidate; will spot-check
  after the lint fix lands.

## Test coverage

Pinned in `crates/chelis-cli/tests/lint_path_walk_consistency.rs`:

1. `lint_cli_explicit_path_and_cwd_walk_produce_identical_violations`
   (gated `#[ignore]` until fix lands) — asserts both invocations
   produce the §8.3 violation on the same fixture.
2. `lint_cli_dot_prefix_explicit_path_already_fires_rule` (running) —
   positive control: `./docs/` does fire the rule today, isolating the
   bug class from a generic explicit-path failure.

After the fix, fixture 1 flips from `#[ignore]` to running.
