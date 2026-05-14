# `chelis check` advisory-emit path missing typed-pipeline suppression

Diagnosis for red-team findings LP-LEAK-A and LP-LEAK-B
(`docs/investigations/terminal_redteam_0_7_9.md`). Orchestrator-tracked
as `Lint-CheckMirrorsFixAdvisoryEmitLeak-F1` (Wave 3.5).

## Symptom

PR #107 opted `redundant-linearity-call` and `prefer-pipe-operator` into
`check_mirrors_fix = true`. The intent: suppress an advisory warning when
the rule's autofix would be rejected by the typed-pipeline gate (e.g.
stripping a `copy()` off a borrow makes the program fail type-check), so
the warning is not a non-actionable false positive.

That suppression only applies in the `chelis lint --check` flow.
`chelis check` (and the build-time style gate) flood with the same
warnings the lint flow suppresses.

Reproduction (copy-on-borrow program, post `chelis fmt`):

```
def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)
def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(y))
input = to_tensor([1.0, 2.0])
result = caller(&input)
```

- `chelis lint --check copy_borrow.ch`: emits only `prefer-pipe-operator`
  (the `redundant-linearity-call` warning is suppressed; its strip would
  fail type-check).
- `chelis check copy_borrow.ch`: emits BOTH `prefer-pipe-operator` AND
  `redundant-linearity-call`.

## Root cause

Two divergent emit paths in `crates/chelis-cli/src/main.rs`:

1. `cmd_lint` (around line 5453) iterates kept violations and calls
   `should_suppress_unfixable_violation(target, &rules, v)` before
   printing. That helper, for a rule with `check_mirrors_fix() == true`,
   runs `fix_would_apply_for_violation`, which applies the rule's fix to
   the source and runs `typed_pipeline_accepts_surf` on the result. If
   the rewrite would not type-check, the violation is suppressed.

2. `emit_advisory_lint_warnings_for_file` (around line 1311), which
   `cmd_check_one` invokes for `chelis check`, only applies the
   path-glob exception filter via `apply_exceptions`. It never calls
   `should_suppress_unfixable_violation`. Every advisory violation that
   survives the path filter is printed.

LP-LEAK-B is the same root cause: `prefer-pipe-operator` also opts into
`check_mirrors_fix`, so its warnings leak through the same path
whenever its autofix declines.

## Fix

Thread the typed-pipeline suppression into
`emit_advisory_lint_warnings_for_file` so both code paths apply the same
gate for rules with `check_mirrors_fix = true`. The advisory-emit path
already has the target file's parent directory (the lint walk root) and
the rule set in scope; it calls `should_suppress_unfixable_violation`
with the walk root as `target`, exactly as `cmd_lint` does. No new
suppression logic is introduced; the existing helper is reused so the
two paths cannot drift again.

## Positive control

A genuinely-redundant `copy()` on an OWNED tensor (`realize(copy(x))`
where `x: tensor[n, f32]`) has a safe autofix: the strip type-checks.
`should_suppress_unfixable_violation` returns `false` for it, so
`chelis check` must still emit `redundant-linearity-call`. The fix must
not over-suppress fixable violations.

## Sibling sweep

`chelis build`, `chelis eval --file`, `chelis check`, and
`chelis validate` all run the style gate
(`crates/chelis-cli/src/style_gate.rs`). The style gate's
`run_lint_for_single_file` also applies only `apply_exceptions`, with no
`should_suppress_unfixable_violation` call. However, it lints with
`chelis_lint::registry::all_rules()`, which does NOT include
`redundant-linearity-call` or `prefer-pipe-operator` — those two rules
live only in `non_blocking_rules()`. The style gate therefore never
emits either `check_mirrors_fix` rule, so it cannot leak them. No fix
is needed there.

The only path that emits the `check_mirrors_fix` advisory rules outside
`cmd_lint` is `emit_advisory_lint_warnings_for_file`, which lints with
`non_blocking_rules()`. That is the single leak surface and the single
site this PR changes.
