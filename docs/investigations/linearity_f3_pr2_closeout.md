# Linearity-F3 PR 2 closeout — severity flip

Workstream: Linearity-F3. Branch: `fix/linearity-f3-pr2-errors`.
This note closes the two-PR sequence opened by PR #65
(`fix/linearity-f3-pr1-warning-sweep`).
PR 1's diagnosis lives at
`docs/investigations/linearity_f3_module_skip_diagnosis.md`.

## PR 1 inventory result

PR 1 extended the pre-declare loop in `check_linearity` and
`check_linearity_with_context` to recurse through `(module {} name ...)`
wrappers, and routed any newly surfaced diagnostic into
`LinearityInfo::warnings` instead of `Checker::errors` for a
deprecation window. The CLI surfaced those warnings via
`warning: linearity: ...` on stderr but kept them out of the JSON
`errors` array and out of `report.score`.

PR 1's full corpus sweep (103 `.ch` files under `examples/`,
`examples/illustrative/`, `packages/`, and `crates/*/tests/`) found
**zero surfaced warnings**. Every in-tree program that uses
`module Foo` wrapping was already linearity-clean. As a result,
PR 2's originally-scoped "fix surfaced violations" step is a no-op,
and the remaining work is the severity flip plus removal of the
now-unused warning plumbing.

The in-repo corpus sweep deliberately excluded `hello-chelis`. That
repo's lint and check are re-run by its owning agent on a separate
cadence; any module-wrapped linearity violations there will surface
to its dispatcher when the next sweep runs against this binary, and
must be fixed before downstream consumers upgrade to a chelis
release that includes this PR.

## Scope of PR 2

Minimal severity-flip plus plumbing removal:

- Route module-wrapped violations through `Checker::errors` (the same
  error path as bare-top-level violations) instead of
  `LinearityInfo::warnings`.
- Remove the `LinearityInfo::warnings` field, the `warnings()`
  accessor, and the `push_warning` helper — the only consumers were
  PR 1's CLI stderr emit and the three fixtures, both of which
  flip to the error path in this PR.
- Remove the `Checker::in_module` field. It was used only to switch
  severity in `push_diagnostic`; with severity unified, no other
  callsite needs to distinguish module-wrapped from bare-top-level.
  The module-recursion case in `Checker::check_top_level` and the
  `pre_declare_top_level_defs` helper from PR 1 stay in place — they
  are the recursion that made linearity checking actually run on
  module-wrapped code in the first place, and removing them would
  re-open the original gap.
- Remove the `warning: linearity: ...` stderr emit in
  `chelis-cli/src/main.rs`. The check command's JSON `errors` array
  and `report.score` now carry the violation through the same path
  as bare-top-level errors.
- Drop the `PartialEq`/`Eq` custom impl on `LinearityInfo` that
  excluded `warnings` from comparison; the struct returns to deriving
  the standard impls because all fields again participate in
  identity.

## Fixture changes

The three PR 1 fixtures
(`module_wrapped_realize_then_borrow_emits_warning` etc.) are
renamed to `..._errors` and flip from
`LinearityInfo::warnings()` assertions to
`check_linearity` `Err` assertions. The three bare-top-level
controls stay unchanged — they remain the regression net for the
original error path.

## Verification

Reproducer (the program from PR 1's diagnosis):

```chelis
module Test

x = to_tensor([1.0, 2.0, 3.0])
y = realize(x)
b = add(x, y)
```

Before PR 2 (today, PR 1 baseline):

```
warning: linearity: variable `x` was already consumed by realize at offset 0; later use at offset 0 is invalid (UseAfterConsume)
{ "score": 1, "errors": [] }
```

After PR 2:

```
{ "score": 0.8, "errors": [{"kind":"UseAfterConsume", ...}] }
```

No stderr warning emission; the violation is in the JSON `errors`
array and `score` reflects the linearity penalty.

## Cross-references

- §5 entry: `docs/archive/reports/gap_synthesis.md` Linearity-F3. PR 2 marks this
  entry Closed.
- PR 1 diagnosis: `docs/investigations/linearity_f3_module_skip_diagnosis.md`.
- PR 1 (merged): GitHub PR #65.
- Spec: `spec/design/implicit_linearity.md` — canonical semantics
  treats linearity violations as errors; the warning-mode was a
  PR 1 deprecation window, not a spec change.
