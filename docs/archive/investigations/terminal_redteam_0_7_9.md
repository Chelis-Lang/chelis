# 0.7.9 terminal red-team

Wave 3 fresh-context red team for the 0.7.9 lint precision + Shape A
broader return cleanup workstream. Adversarial fixtures and findings
against the four §5 closures shipped by PRs #107, #108, and #109.

Run against `origin/main` at commit `d7ec8b9` ("docs: §5 bookkeeping
for 0.7.9 closures (PRs #107 + #108 + #109) (#111)").

## Summary

Total findings: 4 material + 1 non-material observation.

| ID | Severity | Surface | Disposition |
| -- | -------- | ------- | ----------- |
| LP-LEAK-A | MEDIUM | `chelis check` advisory-emit path does not suppress unfixable warnings; `redundant-linearity-call` floods continue against the customer's primary workflow | Recommend new §5 entry + follow-on PR |
| LP-LEAK-B | MEDIUM | Same leak applies to `prefer-pipe-operator`; symptom is widespread (20+ false positives in `tokenizer.ch`, 34 in `decimal.ch`, 12 in `tests/decimal.ch`) | Folded into LP-LEAK-A fix |
| LE-LEAK-A | LOW–MEDIUM | "Workspace root" detection is just CWD; running `chelis lint --check .` from `crates/` re-surfaces the 6 false-positive `surf-def-arrow-form` errors | Recommend new §5 entry; documented scope in `Lint-ExceptionPathRoot-F1` should reflect the narrower customer-visible guarantee |
| SR-LEAK-A | HIGH (pre-existing, widened) | `types_structurally_equal` compares tensor rank only, not dim identity; a function declaring `tensor[n, f32]` returning a body of type `tensor[m, f32]` (different dim parameter names) silently passes type-check and runs with a runtime shape mismatch | Pre-existing in PR #91; recommend new §5 entry — separate from PR #109 closure |
| Non-material | LOW | Nested `copy(copy(borrow))` flags both copies with `[fix]` markers; convergence is correct but the per-call marker is ambiguous in isolation | Skip; downstream-invisible |

## §5 audit

| §5 ID | Anchor | Regression tests | Result |
| ----- | ------ | ---------------- | ------ |
| `Lint-RedundantLinearityCopyOnBorrowWarn-F1` (PR #107) | `crates/chelis-lint/src/rules/redundant_linearity_call.rs::check_mirrors_fix` | `crates/chelis-lint/tests/redundant_linearity_call_autofix.rs::f10*` + F11 | Pass (lint --check only; see LP-LEAK-A) |
| `Lint-PreferPipeRedundantLinearityPair-F1` (PR #107) | Same opt-in mechanism | `…autofix.rs::f12*` + F13 | Pass (lint --check only; see LP-LEAK-B) |
| `Lint-ExceptionPathRoot-F1` (PR #108) | `crates/chelis-lint/src/exceptions.rs::is_excepted` + `detect_lint_workspace_root` in `chelis-cli` | `crates/chelis-cli/tests/lint_path_walk_consistency.rs::lint_cli_exception_pattern_matches_under_subtree_and_cwd_walks` + multi-subtree variant + `apply_exceptions_anchors_on_workspace_root_not_walk_target` unit test | Pass (workspace-root-invocation only; see LE-LEAK-A) |
| `Linearity-ShapeABroadReturn-F1` (PR #109) | `crates/chelis-types/src/infer.rs::descend_to_tail_var` + `shape_a_relaxed_return` integration | `crates/chelis-ir/tests/implicit_copy_shape_a_broader_return.rs` (7 fixtures) | Pass (pre-existing dim-var-unification leak persists; see SR-LEAK-A) |
| `Lint-PreferPipeRecursive-F1` | Anchor at `crates/chelis-lint/src/rules/prefer_pipe_operator.rs` | Marked Verified-Not-Reproducible | No adversarial repro surfaced |

Every cited regression test passes on local main:

```
cargo test -p chelis-lint --test redundant_linearity_call_autofix
  -> 13 passed; 0 failed; 0 ignored
cargo test -p chelis-cli --test lint_path_walk_consistency
  -> 4 passed; 0 failed; 0 ignored
cargo test -p chelis-ir --test implicit_copy_shape_a_broader_return
  -> 7 passed; 0 failed; 0 ignored
cargo test -p chelis-cli --test red_team_0_7_8
  -> 7 passed; 0 failed; 0 ignored
cargo test -p chelis-ir --test implicit_copy_fanout_shape_a_adversarial
  -> 6 passed; 0 failed; 0 ignored
```

## CHANGELOG verification

`[Unreleased]` section in `CHANGELOG.md` contains four entries:

1. "redundant-linearity-call false-positive on copy(borrow)" — closes
   `Lint-RedundantLinearityCopyOnBorrowWarn-F1`. ✅ Wording accurate
   for the `chelis lint --check` workflow; INACCURATE for the
   customer-visible scope (Nautilus uses `chelis check`).
   See LP-LEAK-A.
2. "redundant-linearity-call false-positive on 2-arg list primitives
   in pipe form" — closes `Lint-PreferPipeRedundantLinearityPair-F1`.
   Same scope caveat. See LP-LEAK-B.
3. "lint exception path matching anchored at workspace root" —
   closes `Lint-ExceptionPathRoot-F1`. ✅ Wording accurate for
   the workspace-root invocation; INACCURATE for invocations from
   sub-directories. See LE-LEAK-A.
4. "implicit-copy Shape A covers let/if/match tail-position returns"
   — closes `Linearity-ShapeABroadReturn-F1`. ✅ Wording accurate
   for the body-shape coverage. Pre-existing dim-var-unification
   soundness gap is unrelated to PR #109's claim, but worth noting
   because PR #109 widens its reach. See SR-LEAK-A.

Recommendation: do not edit CHANGELOG entries before the underlying
leaks are addressed. When the follow-on PRs land, the entries should
either (a) remain as-is if the follow-on fixes ship before release-cut
or (b) be amended to scope the closure if the follow-ons defer.

## Per-finding detail

### LP-LEAK-A (MEDIUM) — `chelis check` advisory-emit path bypasses unfixable suppression

**Shape**

`chelis check <file>` calls `cmd_check_one` at
`crates/chelis-cli/src/main.rs:1191`. At line 1198 it invokes
`emit_advisory_lint_warnings_for_file(file)`. That function at
`crates/chelis-cli/src/main.rs:1311` runs the `non_blocking_rules`
list, applies the workspace-rooted exception filter, and emits every
remaining violation directly to stderr. It does NOT call
`should_suppress_unfixable_violation`.

PR #107 added the `check_mirrors_fix=true` opt-in on
`RedundantLinearityCall`. The check has effect only inside `cmd_lint`
at `crates/chelis-cli/src/main.rs:5474` where
`should_suppress_unfixable_violation` is consulted before emission.

**Customer impact**

`chelis check` is the canonical workflow developers and CI scripts
use. Nautilus's reported 319 false positives in `src/linalg.ch`
against 0.7.8 (the original symptom that filed
`Lint-RedundantLinearityCopyOnBorrowWarn-F1`) remain unaddressed for
`chelis check`. Verified by sweeping the repo's own corpus:

| File | `chelis check` warnings | `chelis lint --check` warnings |
| ---- | ---------------------- | ----------------------------- |
| `packages/chelis-std/src/decimal.ch` | 34 | 0 |
| `packages/chelis-std/src/tokenizer.ch` | 20 | 0 |
| `packages/chelis-std/tests/decimal.ch` | 12 | 0 |
| `packages/chelis-std/tests/optim.ch` | 18 | 0 |
| `packages/chelis-std/tests/schedule.ch` | 10 | 0 |
| `packages/chelis-std/tests/time.ch` | 23 | 0 |
| `examples/illustrative/linear_regression.ch` | 4 | 0 |
| `examples/illustrative/mha_two_heads_unrolled.ch` | 14 | 8 |

Both `redundant-linearity-call` and `prefer-pipe-operator` contribute.

**Repro** (LP-LEAK-A test):

```surf
def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)
def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(y))
input = to_tensor([1.0, 2.0])
result = caller(&input)
```

- `chelis lint --check` → no `redundant-linearity-call` warning. ✅
- `chelis check` stderr → emits the warning. ❌

**Recommended disposition**

New §5 entry, e.g. `Lint-CheckMirrorsFixAdvisoryEmitLeak-F1`. Follow-on
PR routes `emit_advisory_lint_warnings_for_file` through
`should_suppress_unfixable_violation` (or factors that helper out so
it can be called from both `cmd_lint` and the advisory emit path).
Pinned by `red_team_0_7_9::lp_leak_a_chelis_check_advisory_emit_does_not_suppress_unfixable_copy_borrow`
(`#[ignore]` until fixed). Flip the `#[ignore]` to active when fixed.

### LP-LEAK-B (MEDIUM) — same leak affects `prefer-pipe-operator`

**Shape**

`prefer-pipe-operator` also opts into `check_mirrors_fix=true`
(`crates/chelis-lint/src/rules/prefer_pipe_operator.rs:84`). Both
non-blocking rules leak unfixable warnings through the
`chelis check` path. Sweep evidence is in the LP-LEAK-A table above.

**Recommended disposition**

Folded into LP-LEAK-A's fix; same root cause. Pinned by
`red_team_0_7_9::lp_leak_b_chelis_check_advisory_emit_does_not_suppress_unfixable_prefer_pipe`.

### LE-LEAK-A (LOW–MEDIUM) — CWD-as-workspace-root assumption

**Shape**

`detect_lint_workspace_root` at
`crates/chelis-cli/src/main.rs:5525` is just
`fs::canonicalize(current_working_directory)` — no `Cargo.toml` /
`.git` probe, no walk-up. The PR #108 diagnosis explicitly says
"The CWD is the workspace root by construction whenever `chelis lint`
is invoked from a workspace directory" (`docs/archive/investigations/lint_exception_path_root_diagnosis.md:105-111`).
That assumption breaks for two natural workflows:

1. **Developer CDs into a subdirectory of the workspace.** Running
   `chelis lint --check .` from `<repo>/crates` re-surfaces all 6
   false-positive `surf-def-arrow-form` errors against the Surf
   fixture corpus.

2. **Developer runs from outside the workspace with absolute paths.**
   Running `chelis lint --check /abs/repo/crates/chelis-surf` from
   `/tmp` re-surfaces the same 6 errors.

Verified in this red team with the worktree's actual corpus, not
synthesized fixtures.

**Customer impact**

The `Lint-ExceptionPathRoot-F1` §5 entry claims "Both
`chelis lint --check .` and `chelis lint --check crates docs examples
packages` now produce identical output with zero false-positive
`surf-def-arrow-form` errors." That holds ONLY when invoked from the
workspace root. For developers who CD into `crates/` (a routine
workflow given the workspace size) the bug reproduces.

The `feedback_no_walkup_filesystem_detection.md` standing rule
applies — walk-up detection is rejected by repo policy. The
recommended path is to thread the workspace root explicitly: either
have the CLI accept an explicit `--workspace-root <path>` flag, or
detect the workspace root once from the binary's manifest at compile
time, or detect by walking upward from the *first walk-target* (not
the CWD) until a sentinel exists. Each path has trade-offs; the
follow-on §5 entry should surface them.

**Recommended disposition**

New §5 entry, e.g.
`Lint-WorkspaceRootCwdAssumption-F1`. Pinned by
`red_team_0_7_9::le_leak_a_cwd_not_workspace_root_breaks_workspace_rooted_exception`
(`#[ignore]` until a follow-on workstream decides the detection
strategy).

### SR-LEAK-A (HIGH, pre-existing, widened by PR #109)

**Shape**

`types_structurally_equal` at
`crates/chelis-types/src/infer.rs:8599-8632` checks tensor types
with:

```rust
(Type::Tensor(d1, p1), Type::Tensor(d2, p2)) => p1 == p2 && d1.len() == d2.len(),
```

This compares dim *count* only, not dim *identity*. Combined with
HM's free-dim-var unification at the relaxed-retry call site, a
function such as

```surf
def f[n, m](x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = {
  z = y
  z
}
```

passes type-check with `score=1`, lowers, and evaluates to a runtime
shape mismatch:

```
x_in = tensor(shape=[2], data=[1.0, 2.0])
y_in = tensor(shape=[3], data=[3.0, 4.0, 5.0])
result = tensor(shape=[3], data=[3.0, 4.0, 5.0])
```

The function is declared to return `tensor[n, f32]` (where `n` binds
to `x_in`'s rank-1 dim 2) but actually returns a `tensor[3, f32]`.

**Origin and scope**

PR #91 introduced both `shape_a_relaxed_return` and
`types_structurally_equal`. The dim-count-only check exists since
that PR. The same `f[n, m] -> tensor[n] = y` shape ALSO passes
without using Shape A at all (no borrow, plain owned tensor) — so the
root cause is in the HM inference unify step, not specifically in
`types_structurally_equal`. PR #109 widens the body-shape coverage
to `let`/`if`/`match`, which gives the leak more places to fire.

**Customer impact**

Soundness gap. Programs that should be rejected by type-check pass
silently and produce results with the wrong shape. Severity is HIGH
because the failure is silent (no warning, no runtime error, just
wrong data shape downstream). The leak does NOT depend on PR #109 —
it's pre-existing — but PR #109's CHANGELOG entry covers exactly
this body-shape class without mentioning the soundness gap.

**Recommended disposition**

New §5 entry, e.g.
`TypeCheck-FreeDimVarUnification-F1`. Scope: HM inference unify
treats distinct user-named dim parameters as freely unifiable when
both are free at the relaxation point. Follow-on PR should either
(a) treat user-named dim parameters as rigid (Skolem-style) during
return-position structural check, or (b) extend
`types_structurally_equal` to discriminate dim identities. Each
path has implications for legitimate inference paths; surface as
orchestrator decision per `feedback_escalate_structural_blockers.md`.

Pinned by `red_team_0_7_9::sr_leak_a_dim_var_mismatch_silently_passes`
(`#[ignore]` until the structural check tightens).

## Sweep / regression status

| Sweep | Result |
| ----- | ------ |
| `cargo test --workspace` from worktree branch | 100% green (no failures) |
| `chelis lint --check .` on worktree | exit 0; advisory warnings only |
| `chelis lint --check crates docs examples packages` from worktree | exit 0; identical to `chelis lint --check .` |
| `chelis lint --check crates/chelis-surf` from worktree | exit 0; 0 false-positive `surf-def-arrow-form` errors |
| `chelis lint --check .` from `<worktree>/crates` | exit 0 in stdout but emits 6 false-positive `surf-def-arrow-form` errors (LE-LEAK-A repro) |
| 0.7.8 red-team backward-compat (`red_team_0_7_8` in `chelis-cli`, `implicit_copy_fanout_shape_a_adversarial` in `chelis-ir`) | all pass |

## Full gate

| Gate | Result |
| ---- | ------ |
| `cargo build --workspace --all-targets` | Pass |
| `cargo test --workspace` | Pass (post-fix file additions; 11 active + 4 ignored tests in `red_team_0_7_9`) |
| `cargo clippy --workspace --all-targets -- -D warnings` | Pass |
| `cargo fmt --all -- --check` | Pass |
| `cargo run -p chelis-cli --bin chelis --quiet -- lint --check .` | Exit 0 |

## Files touched

- `crates/chelis-cli/tests/red_team_0_7_9.rs` (new) — 15 adversarial
  tests pinning expected behavior for the four §5 closures plus four
  `#[ignore]`d leak fixtures documenting the material findings.
- `docs/investigations/0_7_9_terminal_redteam.md` (this file) — the
  red-team report.

No other files modified.
