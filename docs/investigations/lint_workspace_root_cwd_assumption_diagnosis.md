# `detect_lint_workspace_root` cwd-assumption shortcut

Diagnosis for red-team finding LE-LEAK-A
(`docs/investigations/terminal_redteam_0_7_9.md`). Orchestrator-tracked
as `Lint-WorkspaceRootCwdAssumption-F1` (Wave 3.5).

## Symptom

PR #108 closed `Lint-ExceptionPathRoot-F1` by anchoring exception-glob
matching against a "workspace root" instead of the per-target lint walk
root. But `detect_lint_workspace_root`
(`crates/chelis-cli/src/main.rs`, around line 5539) was implemented as
`canonicalize(current_working_directory)` with no actual workspace
probe. The closure rests on the premise that `chelis lint` is always
invoked from the workspace root.

That premise is false. Running `chelis lint --check .` from a
subdirectory (e.g. `crates/`), or `chelis lint --check /abs/workspace`
from an unrelated directory, makes `detect_lint_workspace_root` return
the wrong directory. Exception globs like
`crates/chelis-surf/tests/fixtures/*.ch` are authored workspace-root
relative; when the "workspace root" is actually `<repo>/crates`, the
prefix strip drops the leading `crates/` segment and the exception
silently fails to match. The 6 false-positive `surf-def-arrow-form`
errors PR #108 claimed to close reappear.

Reproduction:

```
mkdir -p ws/crates/chelis-surf/tests/fixtures
printf '[workspace]\nmembers=[]\nresolver="2"\n' > ws/Cargo.toml
# fixture uses the legacy colon-form def, exempted by the §3.5 exception
printf 'def f(x: f32): f32 = {\n  y = mul(x, x)\n  add(y, x)\n}\n' \
  > ws/crates/chelis-surf/tests/fixtures/block_binding_expr.ch

(cd ws && chelis lint --check .)            # exception applies, exit 0
(cd ws/crates && chelis lint --check .)     # exception MISSES, surf-def-arrow-form fires
```

## Root cause

`detect_lint_workspace_root` does not detect the workspace root. It
returns the canonical CWD and names it the workspace root. There is no
`Cargo.toml` probe, no workspace marker check — the name promises a
property the implementation never establishes.

PR #108's own diagnosis
(`docs/investigations/lint_exception_path_root_diagnosis.md`) rejected
"walk-up filesystem traversal looking for `Cargo.toml` or `.git`" per
`feedback_no_walkup_filesystem_detection.md`, and substituted the
cwd-assumption. That feedback prohibits *hand-rolled* brittle walk-ups
(fragile under symlinks, mounts, permission edges). It does not
prohibit using Cargo's own canonical workspace-locating mechanism,
which is hardened against exactly those edges.

## Fix

Detect the workspace root with `cargo locate-project --workspace
--message-format plain`. Cargo reports the absolute path of the
workspace root `Cargo.toml`; the workspace root is its parent
directory. This is the canonical Cargo mechanism for workspace-root
discovery, not a hand-rolled traversal: it is the same probe `cargo`
itself uses, and it correctly handles invocation from any
subdirectory.

`detect_lint_workspace_root` lives in `style_gate.rs` (the shared home
of the exception list) and takes a `probe_dir: &Path` argument: the
`cargo locate-project` command runs with its working directory set to
`probe_dir` rather than the process CWD. `probe_dir` is the directory
the lint operates on — a walk target's directory, or a single file's
parent — so `chelis lint --check /abs/workspace/crates` issued from an
unrelated directory still resolves the correct workspace root. The
function returns `Result<PathBuf, String>`; it has exactly one
detection mechanism and fails clean with a diagnostic when `probe_dir`
is not inside any Cargo workspace.

### Caller policy: detection failure means "no workspace-rooted exception applies"

All three exception-anchor call sites — `cmd_lint`,
`emit_advisory_lint_warnings_for_file`, and the style gate's
`run_lint_for_single_file` — apply one uniform policy: on a detection
`Ok`, anchor exception matching against the detected root; on a
detection `Err`, pass the raw violations through unfiltered.

This is NOT a second detection mechanism with different semantics. The
exception globs are authored workspace-root relative, so they can only
match when there is a workspace root to anchor against. When the
targets are not inside any Cargo workspace, no workspace-rooted
exception can legitimately apply (a loose file under `/tmp` is not
`crates/chelis-surf/tests/fixtures/*.ch`), so the unfiltered raw
violations ARE the correct result.

The brief's literal wording was "fail with a clear diagnostic" on
detection failure. Hard-failing `cmd_lint` was rejected because
`chelis lint --check /tmp/loose_file.ch` outside any workspace is a
supported, pre-existing operation (covered by the LP linearity-call
fixtures in `red_team_0_7_9.rs`); aborting it would be a regression
unrelated to this bug. Degrading to unfiltered output is the
semantically-correct, non-regressing policy and keeps all three call
sites on one coherent contract. This deviation from the literal
wording is called out in the PR body for orchestrator review.

## Strategy decision and the red-team fixture conflict

The brief flagged workspace-root detection strategy as an orchestrator
decision and asked for escalation if `cargo locate-project` is not
clean. Investigation result: `cargo locate-project --workspace` is
clean for any real Cargo workspace — from the repo root and from
`crates/chelis-cli` it correctly reports the repo `Cargo.toml`.

The only friction is the red-team test harness: the LE fixtures
synthesize "workspace" trees in system tempdirs with no `Cargo.toml`.
A test that claims to exercise workspace-root detection but synthesizes
a tree with zero workspace markers is testing the cwd-assumption
shortcut, not the real mechanism. The honest fix is to give those
synthesized trees a real `[workspace]` `Cargo.toml` marker
(`write_workspace_marker` in `red_team_0_7_9.rs`). This is not fixture
engineering to hide a gap: it makes the fixture realistic for the
mechanism under test. The change is called out explicitly in the PR
body so the orchestrator can review it rather than have it buried.

`cmd_lint` still canonicalizes each user-supplied walk *target*
independently (PR #93 behavior, unchanged) — that is orthogonal to
workspace-root detection.

## Rejected alternatives

- **Hand-rolled walk-up looking for `Cargo.toml` / `.git`.** Rejected
  per `feedback_no_walkup_filesystem_detection.md`. `cargo
  locate-project` is Cargo's own hardened probe, not a hand-rolled
  traversal.
- **Keep `canonicalize(cwd)` with a fallback to a cargo probe.** Two
  detection paths with different semantics. Rejected.
- **Keep `canonicalize(cwd)` and document the limitation.** Converts a
  fixable bug into a permanent user-visible papercut, and the
  `Lint-ExceptionPathRoot-F1` closure claim would remain overstated.

## Sibling sweep

`rg "current_dir|canonicalize" crates/chelis-cli/src/` surfaces other
cwd / canonicalize sites:

- `crates/chelis-cli/src/style_gate.rs::run_lint_for_single_file`
  (around line 188): inline
  `current_dir().and_then(canonicalize)` with a `file.parent()`
  fallback — the SAME cwd-assumption bug shape, in a separate site.
  This PR routes it through `detect_lint_workspace_root` so all three
  exception-anchor sites share one detection mechanism. See the
  sibling-sweep section of the PR description.
- `cmd_lint` target canonicalization (around line 5423): canonicalizes
  each user-supplied walk *target*, not the workspace root. Correct as
  is — unchanged.
- Other `current_dir` uses in the CLI (file resolution for `eval`,
  `build` output paths) are not workspace-root detection and are out
  of scope.

## References

- gap-synthesis entry: `Lint-WorkspaceRootCwdAssumption-F1`.
- PR #108 diagnosis: `lint_exception_path_root_diagnosis.md`.
- PR #93 target canonicalization: `lint_cli_path_walk_diagnosis.md`.
- Standing rule: no hand-rolled walk-up filesystem detection
  (`feedback_no_walkup_filesystem_detection.md`).
