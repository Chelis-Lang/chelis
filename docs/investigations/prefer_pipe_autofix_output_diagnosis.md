# `prefer-pipe-operator` autofix output not fmt-clean — diagnosis

Diagnoses Finding 3b from the 0.7.6 toolchain hygiene red-team (PR #51).
This is distinct from Finding 3a (formatter idempotency on multi-stage
pipes inside brace blocks); see "Coordination with Finding 3a" below.

## Bug shape

The `prefer-pipe-operator` autofix (re-enabled in PR #42 via Path 1B —
see `pipe_autofix_and_bare_keyword_extras_diagnosis.md`) emits a flat
single-line pipe expression for any candidate it accepts. For
candidates whose canonical pipe form has more than three pipe parts
(seed + ≥3 stages), the formatter would emit a multi-line, brace-
wrapped form — so `chelis lint --fix` produces text that immediately
fails `chelis fmt --check`.

The new fixtures in `crates/chelis-cli/tests/cli.rs`
(`lint_fix_prefer_pipe_operator_output_is_fmt_clean_*`, pinned by
`7c18df3`) reproduce this against the round-trip
`chelis lint --fix <file>` followed by `chelis fmt --check <file>`.

Concrete repro:

```ch
def f(x: tensor[3, f32]) -> tensor[3, f32] = sigmoid(relu(neg(x)))
```

After `chelis lint --fix`:

```ch
def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> neg |> relu |> sigmoid
```

`chelis fmt` on that file produces:

```ch
def f(x: tensor[3, f32]) -> tensor[3, f32] = {
  x
  |> neg
  |> relu
  |> sigmoid
}
```

`chelis fmt --check` therefore rejects the lint-fixed file.

## Root cause

`crates/chelis-lint/src/rules/prefer_pipe_operator.rs::pipe_candidate_at`
constructs the replacement text by string-joining stages with ` |> `:

```rust
let mut replacement = seed;
for stage in stages {
    replacement.push_str(" |> ");
    replacement.push_str(&stage);
}
```

This is always the flat single-line shape, regardless of stage count.

The formatter
(`crates/chelis-surf/src/format.rs::format_pipe_layout`) gates the
flat shape behind both stage count and line width:

```rust
if total_stages <= 3 && flat.chars().count() <= WIDTH {
    return flat;
}
```

Where `total_stages = 1 + stages.len()` (seed + stages) and `WIDTH = 80`.
When that gate fails, the formatter emits one element per line. When
the resulting expression contains a newline, `format_function_body`
additionally wraps it in `{ ... }` with 2-space indent.

The autofix's replacement covers only the call expression's byte span
(`call.start..call.end`), so it cannot extend itself to add brace
wrapping; the surrounding `def ... = <span>` is not part of the
replacement. Emitting the multi-line pipe form into the existing span
produces text like `def f(...) = x\n|> neg\n|> relu\n|> sigmoid`,
which the formatter still rejects (no braces, no indent).

## Two candidate fix shapes

### (a) Call `chelis_surf::format` on the rewritten AST

The rule would build an `Expr::Pipe(seed, stages)` AST node, format
the full enclosing decl (or binding) through `chelis_surf::format`,
and return the formatted text as the replacement. This is the most
robust option in principle — the formatter remains the single source
of truth for canonical layout.

Cost:

- The rule's replacement span would need to widen to cover the full
  enclosing decl (so brace wrapping is included). That is a meaningful
  change to the rule's API surface and to the CLI driver's conflict
  detection (which sorts and rejects overlapping replacements by
  byte-range overlap).
- The multi-line, brace-wrapped form depends on the formatter being
  idempotent for that shape. **As of `origin/main` at the time of
  this diagnosis, `fix/fmt-idempotency-multistage-pipe` (Finding 3a)
  has not landed**: re-formatting an already-formatted multi-line
  pipe inside a brace block produces non-canonical output. Until 3a
  lands, even a perfectly-formatted multi-line autofix output would
  fail the next `fmt --check` run after any non-pipe edit.

### (b) Align the autofix's text construction with the formatter's exact emit rules

The rule predicts what `format_pipe_layout(None, seed, stages)` would
emit for the proposed `(seed, stages)` and only returns a replacement
when that prediction is the flat single-line form.

When the prediction is the multi-line form, the rule returns `None`:
no fix is offered, the warning still fires (without a `[fix]` marker),
and the user can manually rewrite using their editor of choice
(which, post-3a, will be fmt-clean either way).

Cost:

- The rule must mirror two of the formatter's gate conditions:
  `total_stages <= 3` and `flat.chars().count() <= WIDTH`. Both are
  small, local checks; the rule already computes `stages` and the
  flat replacement string.
- The fix surface is conservative: candidates the formatter would
  emit multi-line are silently dropped from the autofix list rather
  than rewritten badly.
- No dependency on Finding 3a. The flat-only path round-trips
  through the formatter today.

## Decision: option (b)

Rationale:

1. **Independence from Finding 3a.** Finding 3a has not landed on
   `origin/main` as of this diagnosis. Option (a) would either need
   3a to ship first (sequencing the two fixes) or accept that the
   multi-line autofix output is itself non-idempotent (a separate
   bug). Option (b) sidesteps the dependency entirely by only
   emitting the flat shape, which round-trips today.
2. **Surface-local change.** Option (b) keeps the change inside the
   rule's `fix()` body. The rule's `Replacement` span semantics, the
   CLI driver's overlap detection, and the typed-pipeline Path 1B
   gate are all unchanged.
3. **Forward-compatible.** When Finding 3a lands, option (b)'s
   bail-out condition can be relaxed in a follow-on commit: the rule
   would then begin offering fixes for the multi-line case, with the
   replacement span widened to cover the enclosing decl. That work
   is strictly larger than what 3b needs today.
4. **Conservative under uncertainty.** Returning `None` for the
   multi-line case is a strictly-narrower fix surface. Users still
   see the warning and can rewrite manually. Option (a) under the
   current state of 3a would expand the bug surface (autofix writes
   text that survives one `fmt --check` but fails the next).

## Coordination with Finding 3a

Finding 3a tracks the formatter's non-idempotency on multi-stage
pipes inside brace blocks. The `fix/fmt-idempotency-multistage-pipe`
branch has not landed on `origin/main` as of this diagnosis. The
option-(b) fix for Finding 3b does **not** depend on Finding 3a:

- 3b's flat single-line autofix output is idempotent today (verified
  by the two-stage fixture, which passes pre-fix).
- 3b's bail-out path leaves the original nested call unchanged for
  cases where the formatter would emit multi-line. The original
  nested call is also fmt-clean today.

If Finding 3a lands after this fix, no rework of the option-(b) code
is required. A future follow-on can widen the rule's emit set to
cover the multi-line case once the formatter is idempotent there.

## Anti-scope

The fix deliberately does **not**:

- modify the formatter (`crates/chelis-surf/src/format.rs`). That is
  Finding 3a's surface.
- modify the CLI driver (`apply_lint_fixes`). The Path 1B typed-
  pipeline gate stays as-is; this fix only narrows what the rule
  proposes.
- modify the rule's `check()` method. The lint warning continues to
  fire for every nested first-argument call chain; only `fix()` is
  narrowed.
- touch `redundant-linearity-call`. That rule's fix output is
  text-substitution of an inner argument and is already fmt-clean
  in every observed case.
- expand the fixture corpus beyond the three round-trip cases. The
  three pin the invariant ("after `lint --fix`, the file passes
  `fmt --check`") across the two relevant emit shapes plus a mixed
  outer-call case.

## Files touched (fix commit)

- `crates/chelis-lint/src/rules/prefer_pipe_operator.rs` — narrow
  the `fix()` body to return `None` when the predicted formatter
  output is the multi-line form. Mirror `format_pipe_layout`'s flat-
  shape gate: `total_stages <= 3 && flat.chars().count() <= WIDTH`.
- `crates/chelis-cli/tests/cli.rs` — flip the two ignored fixtures
  (`*_three_stage` and `*_mixed_outer_args`) to running.

## References

- Finding 3b origin: PR #51 (0.7.6 toolchain hygiene red-team report).
- PR #42 re-enable: `pipe_autofix_and_bare_keyword_extras_diagnosis.md`.
- PR #34 Path 1B mechanism: `redundant_linearity_autofix_architecture.md`.
- Formatter emit logic: `crates/chelis-surf/src/format.rs::format_pipe_layout`.
- Test pin commit: `7c18df3` (`test: pin prefer-pipe-operator autofix
  produces fmt-clean output`).
- Coordinate with Finding 3a: `fix/fmt-idempotency-multistage-pipe`
  (not on `origin/main` as of this diagnosis).
