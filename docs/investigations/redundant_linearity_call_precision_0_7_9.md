# `redundant-linearity-call` precision fixes (0.7.9 cleanup)

Diagnoses two §5 entries closed in the same PR:

- **Lint-RedundantLinearityCopyOnBorrowWarn-F1** (Item 1): the rule emits
  its advisory warning unconditionally based on AST-level `copy() in arg
  position`, even when the typed pipeline rejects the stripped form.
- **Lint-PreferPipeRedundantLinearityPair-F1** (Item 3): the rule's
  syntactic `has_single_top_level_argument()` helper cannot distinguish
  the single-arg linearity-primitive `drop(x)` from the 2-arg library
  `drop(xs, n)` when one argument has been piped in. The pipe form `xs
  |> drop(n)` makes `drop(n)` appear single-arg in the source text.

Both bugs share a single closure mechanism: opt the rule into
`check_mirrors_fix=true` so the CLI driver suppresses warnings when the
autofix would silently decline at the typed-pipeline gate. This mirrors
the V2-F3 closure shape for `prefer-pipe-operator` (PR #58).

## Bug shape 1: copy(borrow) over-flag (Item 1)

`crates/chelis-lint/src/rules/redundant_linearity_call.rs::check` walks
the source line by line. For every `copy(` or `drop(` call with a
single top-level argument, it emits a `Warning`-severity violation
saying the call is "valid but redundant; implicit linearity inserts
the corresponding IR node."

The autofix proposes stripping `copy(x)` to `x`. The CLI driver's
typed-pipeline gate at
`crates/chelis-cli/src/main.rs::fix_would_apply_for_violation` (PR
#95 / 0.7.8 closure of `Lint-AutofixCopyOnBorrow-F1`) verifies that the
post-strip source still parses, type-checks, effect-checks, and
linearity-checks. When the program shape is

```
def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)
def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(y))
```

stripping `copy(y)` to `y` produces `consume_owned(y)`, which fails
type-checking because `y: &tensor[n, f32]` is a borrow and
`consume_owned` expects an owned tensor. The implicit-copy inserter
covers Shape A (return-position bare-var borrow-to-owned) and Shape B
(grad/vmap fan-out) per PR #91, but not "borrow as a non-return,
non-fan-out call argument." The gate correctly drops the `[fix]`
marker.

The warning still fires. The user sees a diagnostic that says "this
copy is redundant" when, in fact, the copy is structurally necessary.
There is no way to satisfy the lint without breaking the program.

Nautilus reports 137+ such false positives in `src/linalg.ch` against
0.7.8 (verified by running `chelis lint --check
/home/jeff/Documents/scratch/nautilus/src/linalg.ch | grep -c
redundant-linearity-call`).

## Bug shape 2: 2-arg list primitive in pipe form (Item 3)

`has_single_top_level_argument()` at
`crates/chelis-lint/src/rules/redundant_linearity_call.rs:116-155`
counts top-level commas inside the parenthesized arg list. The 2-arg
library primitive `drop` (registered at
`crates/chelis-types/src/builtins.rs:818` as a `generic_binop("drop",
...)`) has the surface form `drop(xs, n)` which the helper correctly
recognizes as 2-arg (test `ignores_list_drop_with_two_arguments` pins
this).

The bug surfaces when the call is in pipe form:

```
def f(xs: List[int64], n: int64) -> List[int64] = xs |> drop(n)
```

The source text contains the literal substring `drop(n)`, which the
regex matches and `has_single_top_level_argument` accepts. The
typed-pipeline gate correctly drops the strip (because `drop(n)`
without `xs` would parse but `drop(n)` is not the 1-arg form; even if
it were, the strip `drop(n) -> n` would replace the call with an
integer that fails type-checking). But the warning fires.

Coral observed this against 0.7.7 in `src/internal/hamt.ch` and
`src/internal/window.ch` and worked around it by writing
`drop(xs, one_i64())` directly instead of relying on
`prefer-pipe-operator`'s autofix to produce the pipe form.

## Symbol-resolution path investigation (Item 3 primary path)

Per the brief, the first investigation question for Item 3 is: "is the
resolved binding available to the lint rule at the point of emission?"

**Answer: no.** `crates/chelis-lint/Cargo.toml` declares only `regex`
and `walkdir` as dependencies. The architectural choice (documented in
`docs/investigations/redundant_linearity_autofix_architecture.md` Path
1B) is that `chelis-lint` stays dep-pure and the typed-pipeline gate
lives in the CLI driver. Adding `chelis-types` or `chelis-surf` as a
dep at the rule level would:

1. Reverse the Path 1B decision (cross-crate dep cycle risk: `chelis-types`
   currently has no `chelis-lint` dep, but multiple downstream crates
   depend on both; reversing the dependency direction adds an
   architectural surface the 0.7.9 cleanup scope does not include).
2. Require the lint rule to drive parse + type inference at every
   warning emission, which is exactly what the CLI driver's gate
   already does. Duplicating the gate at the rule level provides no
   precision gain.

**Fallback path chosen**: opt the rule into `check_mirrors_fix=true`.
This makes the CLI driver mirror the typed-pipeline gate at the
warning-emit path (using the existing `should_suppress_unfixable_violation`
helper at `crates/chelis-cli/src/main.rs:5629`). When the autofix
would silently decline (no fix proposed, fix proposed but bailed out,
or fix rejected by the gate), the warning is suppressed.

This closes both Items 1 and 3 with one mechanism:

- For Item 1 (`copy(borrow)`): the strip is rejected by the gate, so
  the warning is suppressed.
- For Item 3 (`xs |> drop(n)`): the strip `drop(n) -> n` is rejected
  by the gate, so the warning is suppressed.

No allowlist is required. A future workstream that wants stricter
precision (e.g., distinguishing migration-necessary from
migration-noise even when the strip would type-check) can revisit the
symbol-resolution path; the dep-pure lint architecture is the
constraint to relax.

## Anchor location: typed-pipeline-accepts gate

Today's gate lives at:

- `crates/chelis-cli/src/main.rs:5280-5298` --- `typed_pipeline_accepts_surf`:
  drives parse, desugar, expand, type-check, effect-check, linearity-check.
- `crates/chelis-cli/src/main.rs:5589-5616` --- `fix_would_apply_for_violation`:
  applies the rule's proposed replacement to a candidate `String` and
  runs `typed_pipeline_accepts_surf` on the result.
- `crates/chelis-cli/src/main.rs:5629-5655` --- `should_suppress_unfixable_violation`:
  the warning-emit-path gate that fires only for rules that opt in to
  `check_mirrors_fix`.
- `crates/chelis-cli/src/main.rs:5413` --- the per-violation suppression
  call in `cmd_lint`.

The fix lifts no code. The CLI driver's gate already runs the
typed-pipeline at the warning-emit path; the rule just has to declare
that it wants the suppression to apply.

## Doc-comment update needed at `chelis-lint/src/lib.rs::check_mirrors_fix`

The library doc-comment at
`crates/chelis-lint/src/lib.rs:166-178` calls out
`redundant-linearity-call` as the canonical example of a rule whose
default-`false` `check_mirrors_fix` is correct (because the warning
is informational). Commit 3 updates the doc-comment to reflect the
new opt-in.

## Sibling-sweep target

Per the brief: "grep for other lint rules that emit warnings without
consulting the typed pipeline. Same shape: `prefer-pipe-operator`."

`prefer-pipe-operator` already opts in to `check_mirrors_fix=true`
(verified at `crates/chelis-lint/src/rules/prefer_pipe_operator.rs:84`).
After commit 3 lands, the two rules that propose semantically-load
rewrites both opt in to the gate, closing the trigger-emit asymmetry
class for the rules currently in the registry.

Other rules in the registry are either:

- Naming/identifier rules (`module-compound-titlecase`, `surf-*-case`,
  etc.) whose `fix()` is a pure rename --- structural, not semantic;
  no typed-pipeline check needed.
- Filename/path rules (`doc-filename-convention`,
  `snapshot-filename-pattern`) where the fix is "move/rename" and the
  warning is independent of typed-pipeline acceptance.
- Source-rewrite rules with no autofix
  (`no-shell-scripts`, `no-em-dash-in-public-strings` is autofix but
  pure string replacement, etc.).

No other rule has the trigger-emit asymmetry shape. Sibling sweep
green for this PR.
