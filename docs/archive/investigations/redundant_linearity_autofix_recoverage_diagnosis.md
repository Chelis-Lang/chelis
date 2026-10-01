# `redundant-linearity-call` autofix re-coverage after implicit-copy v3

Diagnosis note for Wave 5 / Item 6 of the 0.7.8 compiler cleanup
workstream. Follows the implicit-copy inserter v3 fix (PR #91) that
shipped Shape A (borrow-to-owned at return position) and Shape B
(grad/vmap fan-out across observational higher-order calls).

## Background

The 0.7.6 release disabled the `redundant-linearity-call` autofix
because the source-level walker could not prove that stripping a
`copy()` preserved type, ownership, and linearity semantics. 0.7.7
(commit fbb6f1c, PR following 477bd0d) re-enabled the autofix with a
typed-pipeline safety gate: the CLI fix driver applies each candidate
replacement to the source, runs the full Surf parse + desugar +
typecheck + effect-check + linearity-check pipeline, and only writes
the result if every stage accepts.

The architectural decision is documented in
[`docs/archive/investigations/redundant_linearity_autofix_architecture.md`](redundant_linearity_autofix_architecture.md)
as Path 1B. The relevant CLI driver is
`crates/chelis-cli/src/main.rs::apply_lint_fixes` plus
`fix_would_apply_for_violation` and `typed_pipeline_accepts_surf`.

A consequence: any program whose pre-fix source did **not** type-check
or pass linearity could not have its post-fix candidate verified,
because both pre-fix and candidate would fail the same gate. Such
programs received the warning without a `[fix]` suffix and the fix
driver left them unchanged. Two well-known shapes from
`hello-chelis/linreg.ch` hit this gate:

- **Shape A** -- `def f[a](x: &tensor[a, f32]) -> tensor[a, f32] = copy(x)`.
  Pre-W4-A, the typecheck phase rejected the bare-`x` candidate
  (`def 'f' body doesn't match declared signature`). The
  stripped candidate failed for the same reason, so the safety gate
  dropped the rewrite.
- **Shape B** -- `dw = grad(my_loss, wrt=w)(copy(w), copy(b))` followed
  by a trailing borrow-read of `w` or `b`. Pre-W4-A, the linearity
  checker treated grad-app args as Structural consumes; the trailing
  borrow-read tripped `UseAfterConsume`. The stripped candidate failed
  for the same reason, so the safety gate dropped the rewrite.

## What W4-A changed

PR #91 (commit 80f6cf1) made both shapes typecheck and pass linearity:

- `chelis_types::infer::check_top_level` gained a relaxed-retry path
  that inserts an implicit copy at the return position when the body's
  inferred type is `Ref(T)` and the declared return is owned `T`.
- `chelis_types::linearity::arg_is_borrowed` was extended via
  `callee_is_observational_higher_order` to treat `grad(...)` and
  `vmap(...)` applications as borrowing-not-consuming.

Both fixes are upstream of the typed-pipeline gate the autofix uses.
The implication: any post-fix candidate that previously failed the
gate because of these two shapes now passes it, and the `[fix]`
marker is emitted automatically. No rule logic change required.

## Reproduction

On commit 51fae2e (parent of PR #91), with the workspace already at
0.7.7:

```text
$ chelis lint --check --rule redundant-linearity-call shape_a.ch
warning: shape_a.ch:1:61: redundant-linearity-call (implicit-linearity): \
  `copy()` is valid for migration compatibility but redundant; \
  implicit linearity inserts the corresponding IR node
$ chelis lint --check --rule redundant-linearity-call shape_b.ch
warning: shape_b.ch:7:29: ... (no [fix])
warning: shape_b.ch:7:38: ... (no [fix])
```

On commit 80f6cf1 (PR #91 merged) and later:

```text
$ chelis lint --check --rule redundant-linearity-call shape_a.ch
warning: shape_a.ch:1:61: ... [fix]
$ chelis lint --check --rule redundant-linearity-call shape_b.ch
warning: shape_b.ch:7:29: ... [fix]
warning: shape_b.ch:7:38: ... [fix]
```

`chelis lint --fix` then strips both forms cleanly and the stripped
source passes `chelis check` and `chelis eval`.

## Scope of new test coverage

Wave 5 adds five new fixtures to
`crates/chelis-lint/tests/redundant_linearity_call_autofix.rs`. Each
mirrors a Shape A or Shape B variant from
`crates/chelis-ir/tests/implicit_copy_fanout_v3.rs` with a redundant
`copy()` wrapper added. The fixtures pin that:

1. The pre-fix source type-checks and evaluates (so the typed pipeline
   gate has a baseline to reject against).
2. `chelis lint --check` flags the `copy()` with the `[fix]` marker.
3. `chelis lint --fix` strips the `copy()`.
4. The stripped source type-checks and evaluates identically.

Variants covered:

- `f5_shape_a_borrow_return_position` -- bare `copy(x)` body, return
  type owned.
- `f6_shape_a_with_use_site_driver` -- Shape A in a multi-decl program
  with a downstream consumer.
- `f7_shape_b_grad_fanout` -- two grad calls with redundant copies
  around args.
- `f8_shape_b_grad_fanout_four_arg_mse` -- four-arg mse-shape, every
  arg wrapped in `copy()`, two grad calls.
- `f9_shape_b_vmap_observational` -- vmap-app with redundant copy
  around the input.

## Resolution

Scenario A in the dispatch brief applies: the rule itself does not
need changes. The autofix coverage is unlocked entirely by W4-A's
upstream typed-pipeline fixes. The PR ships the new fixtures plus a
CHANGELOG note pinning that the W4-A implicit-copy v3 work also
extends `redundant-linearity-call` autofix coverage to Shapes A and
B.

## Remaining un-autofixable cases

None identified for the Shapes A / Shape B class after W4-A. The
existing `lint_fix_redundant_linearity_call_keeps_when_typed_pipeline_rejects`
test in `crates/chelis-cli/tests/cli.rs` still pins the correct
behavior for programs that reference undefined identifiers (the
safety gate correctly drops the rewrite). That is the intended safety
bar, not an unfixed coverage gap.

If a future hello-chelis corpus pass surfaces another shape that the
typed pipeline rejects, the right escalation is to recommend a §5
entry rather than relax the safety gate. The Path 1B architecture
preserves the invariant that every applied autofix yields a program
that passes the full typed pipeline.

## Sibling sweep findings

`prefer-pipe-operator` shares the same `fix_requires_typed_pipeline_check`
opt-in and routes through the same CLI driver
(`crates/chelis-lint/src/rules/prefer_pipe_operator.rs` and
`crates/chelis-cli/src/main.rs::apply_lint_fixes`). Its existing
tests already exercise both "rewrites when accepted" and "keeps when
rejected" paths. The W4-A fixes do not directly bear on pipe-operator
candidate acceptance (the rewrites are structural and don't involve
borrow-to-owned coercions or observational higher-order calls), so no
re-coverage delta is expected. A spot check on the existing
`lint_fix_prefer_pipe_operator_rewrites_when_typed_pipeline_accepts`
and `lint_fix_prefer_pipe_operator_keeps_when_typed_pipeline_rejects`
tests confirms that they still pass on the post-W4-A base. No further
sibling sweep work recommended.
