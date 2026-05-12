# `prefer-pipe-operator` trigger fires wider than autofix emits — diagnosis

Diagnoses V2-F3 from the 0.7.6 toolchain hygiene red-team v2 (PR #58).
This is a follow-on to PR #55, which fixed Finding 3b (autofix output
non-fmt-clean) by adding a defensive bail-out inside `fix()`. The
trigger that PR #55 left alone fires on a wider set of shapes than the
post-PR-55 `fix()` is willing to rewrite, so `chelis lint --fix` is
non-convergent for the remaining cases.

## Bug shape

`crates/chelis-lint/src/rules/prefer_pipe_operator.rs::check` walks
the source line by line and proposes a candidate pipe rewrite for
every nested first-argument call expression it finds. The walker is
purely syntactic: it does not consult the formatter's emit shape, the
typed/effect/linearity pipeline, or anything else that would tell it
whether a rewrite is safe and emittable.

The autofix path, by contrast, has two filters:

1. **`candidate_is_fmt_clean` (inside `fix()`).** Added by PR #55.
   Drops the rewrite if the predicted formatter output is multi-line.
2. **Path 1B typed-pipeline gate (inside `apply_lint_fixes`).** Added
   by PR #34 / PR #42. Drops the rewrite if the post-rewrite source
   fails parse, expand, type-check, effect-check, or linearity-check.

When either filter fires, the rewrite is silently dropped. The
warning is still printed by `cmd_lint` (without the `[fix]` marker,
because `fix_available_for_violation` mirrors both filters). The next
`chelis lint --check` or `chelis lint --fix` invocation walks the
same unchanged source, fires the same syntactic trigger, and prints
the same warning. There is no fixpoint.

Concrete reproducers (all observed on `origin/main` at `21c6386`):

| Shape                                            | Bail-out reason                                                  |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| `add(mul(x, x), x)`                              | Typed-pipeline gate: rewrite reuses `x` after pipe consumption.  |
| `sigmoid(relu(neg(x)))`                          | `candidate_is_fmt_clean`: 4-part pipe formats multi-line.        |
| `mul(relu(add(x, a)), a)`                        | Typed-pipeline gate: rewrite reuses `a` after pipe consumption.  |

For the linearity case, the original source is type-clean today
because auto-copy/auto-borrow handle in-call fan-out (see
`spec/design/implicit_linearity.md`). The pipe rewrite, however,
threads `x` through a seed-consuming stage and then re-uses it, which
the linearity checker correctly rejects. The rewrite is not equivalent
to the original under linearity; the lint should not be proposing it.

## Trigger-emit asymmetry as a contract failure

The PR #55 diagnosis explicitly preserved `check()`'s behavior:

> Anti-scope: modify the rule's `check()` method. The lint warning
> continues to fire for every nested first-argument call chain; only
> `fix()` is narrowed.

That decision was load-bearing for the Finding-3a-independent path
chosen there. It was also the source of V2-F3: a lint that fires
without offering an emittable rewrite is, for a `Warning`-severity
rule whose action is "rewrite to pipe form", non-actionable.

The contract `chelis lint --fix` should establish:

> For any rule `R` and any violation `v` that `R::check()` fires on
> source `S`, after one run of `lint --fix S`, the resulting source
> `S'` must satisfy `R::check(S')` not firing `v` again. Either the
> rewrite happened (warning is moot), or the trigger was correctly
> tightened (warning never fires on this shape).

This is the "lint --fix converges" invariant called out by V2-F3 and
pinned by the three fixtures in `crates/chelis-cli/tests/cli.rs`
(`lint_fix_prefer_pipe_operator_converges_*`).

## Two candidate fix shapes

The PR #58 brief named two paths.

### (a) Tighten the trigger so `check()` only fires when `fix()` would succeed

Two sub-shapes of bail-out to mirror:

- **Syntactic (`candidate_is_fmt_clean`).** Easy to mirror inside the
  rule: the predicate is local to the proposed `(seed, stages)` and
  already runs inside `fix()`. Moving the same call into `check()`
  is a few lines of code with no new dependencies.
- **Semantic (typed-pipeline).** Cannot run inside `chelis-lint`
  itself, because the lint crate is dependency-pure and does not link
  the type checker, effect checker, or linearity checker. The gate
  already lives in `crates/chelis-cli/src/main.rs::apply_lint_fixes`
  and `fix_available_for_violation`. Mirroring it in the warning-emit
  path means filtering `cmd_lint`'s final violations through
  `fix_available_for_violation` for rules that opt in.

Implementation plan for (a):

1. In `chelis-lint`, mirror the `candidate_is_fmt_clean` check inside
   `PreferPipeOperator::check`. Candidates predicted to format
   multi-line never become violations, so the unit tests for
   `check()` shrink with the change.
2. In `chelis-lint`, add a default-false `Rule::check_mirrors_fix()`
   trait method. `PreferPipeOperator` returns `true`.
3. In `chelis-cli`, when iterating final violations for printing in
   `cmd_lint`, suppress violations whose `rule.check_mirrors_fix()`
   is true and whose `fix_available_for_violation(...)` returns
   false. This filters out the typed-pipeline-rejected cases.

Side effect: fewer warnings overall. Source shapes the rule used to
flag as "valid-but-non-preferred" but where the rewrite is unsafe
(linearity-violating, or multi-line-emit-needed) no longer surface a
warning. The user receives no notice that the source could in
principle be rewritten as a pipe with a manual restructure of the
fan-out. That cost is bounded — the rule was previously flagging
these shapes and offering no path to fix them anyway.

### (b) Expand the autofix emit window

Restructure `fix()` to use a wider `Replacement` span (e.g., reach
the enclosing decl or binding) so the multi-line brace-wrapped form
can be emitted in place.

Cost analysis:

- The `candidate_is_fmt_clean` bail-out can be closed by (b): a wider
  span lets `fix()` produce `{\n  x\n  |> neg\n  |> relu\n  |> sigmoid\n}`
  inside the existing decl. The fmt round-trip is supported by PR #53.
- The typed-pipeline bail-out **cannot** be closed by (b). The
  rewrite `x |> mul(x) |> add(x)` is semantically distinct from
  `add(mul(x, x), x)` under implicit linearity: the pipe consumes
  `x` once and the trailing stages reuse it. No span widening fixes
  the linearity rejection. The lint is proposing an unsafe
  transformation.

So (b) closes one of the two bail-out classes but leaves the
larger class — the seed-reuse fan-out shapes that V2-F3 specifically
called out — unfixed. `--fix` would remain non-convergent on
`add(mul(x, x), x)` and `mul(relu(add(x, a)), a)`.

(b) is also strictly larger than (a) on the surface-area axis:
widening the replacement span requires updating the CLI driver's
overlap-detection logic (`apply_lint_fixes` rejects overlapping
replacements byte-range-wise; a decl-spanning replacement would
collide with other rules' line-local replacements) and the rule's
unit-test shape (replacement text no longer matches the call
expression's substring).

## Decision: option (a)

Rationale:

1. **Covers both bail-out classes.** (a) closes both the
   `candidate_is_fmt_clean` bail-out (via the rule-internal mirror)
   and the typed-pipeline bail-out (via the CLI driver's filter for
   `check_mirrors_fix` rules). (b) only closes the first; the
   linearity-rejected fan-out cases remain non-convergent.
2. **Semantic correctness.** When the rewrite would consume a
   fan-out variable, the lint is **proposing an incorrect
   transformation**. Suppressing the warning is the correct
   behavior: the user is not being prompted to make an unsafe edit.
3. **Surface-local change.** The CLI driver already computes
   `fix_available_for_violation` for the `[fix]` marker. Reusing that
   predicate for warning suppression is a small change with no new
   gates. The `check_mirrors_fix` opt-in keeps `redundant-linearity-call`'s
   advisory warning behavior intact (its warning is informational
   even when fix bails out; see `agent-skills/cli-surface/SKILL.md`
   and CLAUDE.md "redundant-linearity-call is advisory").
4. **Forward-compatible.** If a future enhancement teaches the
   autofix to handle fan-out cases (e.g., by introducing an explicit
   `copy(x)` stage in the rewrite, or by restructuring the lint to
   match only single-use seeds), the trigger filter naturally widens
   in response — no further plumbing change needed.

## Anti-scope

This change deliberately does **not**:

- modify the formatter (`crates/chelis-surf/src/format.rs`). The
  multi-line emit shape is correct as-is; PR #53 already locked
  formatter idempotency on multi-stage pipes.
- modify the typed-pipeline gate inside `apply_lint_fixes`. The gate
  is correct; the bug is that the warning fires when the gate
  rejects.
- widen the rule's `Replacement` span. (b) is rejected above.
- modify `redundant-linearity-call`. Its warnings remain advisory and
  fire independent of fix availability per the spec note in
  CLAUDE.md.
- expand the syntactic walker. The rule's syntactic check is correct;
  the issue is downstream of the walker.

## Files touched (fix commit)

- `crates/chelis-lint/src/rules/prefer_pipe_operator.rs` — mirror the
  `candidate_is_fmt_clean` check inside `check()`; declare
  `check_mirrors_fix() -> true`; update unit tests that assert
  `check()` flags multi-line-emit shapes (the assertion flips:
  no warning for those shapes).
- `crates/chelis-lint/src/lib.rs` — add `Rule::check_mirrors_fix`
  trait method with default `false`.
- `crates/chelis-cli/src/main.rs::cmd_lint` — filter final
  violations for rules where `check_mirrors_fix() == true` and
  `fix_available_for_violation(...)` returns `false`. Skip the
  print and the blocking-check bookkeeping for suppressed
  violations.
- `crates/chelis-cli/tests/cli.rs` — flip the three V2-F3 ignored
  fixtures (`lint_fix_prefer_pipe_operator_converges_*`) to running.

## References

- V2-F3 origin: PR #58 (0.7.6 toolchain hygiene red-team v2 report).
- PR #55 predecessor: Finding 3b (`prefer_pipe_autofix_output_diagnosis.md`).
- PR #42 re-enable: `pipe_autofix_and_bare_keyword_extras_diagnosis.md`.
- PR #34 Path 1B mechanism: `redundant_linearity_autofix_architecture.md`.
- Test pin commit: `82efae5` (`test: pin prefer-pipe-operator
  trigger-emit asymmetry`).
- Implicit linearity spec: `spec/design/implicit_linearity.md`.
- Rule spec: `spec/01-nomenclature.md` §3.6.
