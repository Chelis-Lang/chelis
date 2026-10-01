# Wave 2 cascade — tuple-destructure linearity corpus survey

Owning plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`,
"Wave 2 — Linearity cascade cleanup" (sections W2-cascade.1 / .2 / .3).

Phase 0 spec lock: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`
Contract 1 (typed `ConsumeKind`) and the F3 deprecation-window
reference at `docs/archive/investigations/linearity_f3_pr2_closeout.md`.

§5 entry closed by this cascade (text edits filed by the orchestrator
after merge): `Linearity-F2`.

## Background

W1 PR 1 (PR #83, merged) closed the tuple-destructure linearity
false-negative by resolving destructured tuple-component types at
linearity-check time via a local `tuple_get_element_type` helper in
`crates/chelis-types/src/linearity.rs`. To stage the rollout, W1
reintroduced the `LinearityInfo::warnings` channel (the F3 PR 2
closeout had removed it) and routed any newly surfaced violations on
destructure-marked bind chains through that channel rather than
`Checker::errors`.

The Wave 2 cascade closes the deprecation window: survey the corpus
for any warnings W1's lint emits, clean up true positives, then flip
the warning channel to errors and remove the plumbing.

## Survey method

The CLI's `chelis check` did not surface `LinearityInfo::warnings`;
that channel was internal to `chelis-types` and was only inspected
by the chelis-types test fixtures (`linearity_typed_consumekind.rs`,
`linearity_aliased_consume.rs`) and the W2-cascade survey driver.
The survey driver lived at
`crates/chelis-cli/tests/linearity_destructure_corpus_survey.rs`
during W2-cascade.1 and was removed alongside the warnings channel
in W2-cascade.3 (the same commit that flipped the surfaced
violations from warnings to errors).

The driver walked every `.ch` file under `examples/` and
`packages/`, ran the full front-end pipeline (reef import
resolution where applicable, parse, desugar, macro expansion, type
check, effect check, linearity check), and collected every entry
returned by `LinearityInfo::warnings()`.

`crates/*/tests/` was excluded because those `.ch` fixtures are
intentionally adversarial (linearity negative controls) and would
not represent a production-corpus signal.

Manual gate that produced the recorded result (run against the
W2-cascade.1 commit before the channel was removed):

```
cargo test --package chelis-cli --test linearity_destructure_corpus_survey \
  -- --ignored --nocapture
```

## Result

**Total destructure-warning emissions across the production corpus: 0.**

Files walked through the linearity check: every `.ch` under
`examples/` (10 root + 13 illustrative + nested phase3g tree) and
`packages/chelis-std/` (30 src files + 35 tests). 38 files were
skipped before the linearity check ran (parse, type-check, or
effect-check failures — normal for the illustrative negative-control
files such as `effects_handlers.ch` that exercise unimplemented
shapes, and for `packages/chelis-std/tests` runtime fixtures that
exercise reef context not present in this driver). Skipped files do
not represent a survey gap because they do not reach the linearity
layer regardless of the warning channel.

Static cross-check: every tuple-destructure binding site in the
production corpus binds non-tensor scalar tuples. Counts by file:

| File | Lines | Tuple element types |
|---|---|---|
| `packages/chelis-std/src/decimal.ch` | 1 | `(int64, int64, bool)` |
| `packages/chelis-std/src/tokenizer.ch` | 6 | `(string, int64)` shapes |
| `packages/chelis-std/src/io/json.ch` | 9 | `(Json, int64)` and `(string, int64)` shapes |

Total: 16 destructure sites, all on non-tensor tuples.
`expr_is_owned_linear` in `crates/chelis-types/src/linearity.rs`
returns false for these via `type_expr_is_owned_linear` (which
gates on `type_expr_contains_tensor`), so the linearity check
correctly skips the consume tracking and the
`destructure_warning_depth` gate that would route into the warning
channel never fires.

## Threshold-rule disposition

Total warnings = 0. Per the plan's W2-cascade.1 escalation rule:

> If `total_warnings < 10`: ratio is not statistically meaningful. Do
> NOT apply the 10% rule. Instead, escalate each false positive
> individually to the orchestrator.

Zero false positives to escalate. The escalation gate is satisfied
vacuously and the cascade proceeds to W2-cascade.3 (warning-to-error
flip + channel removal). W2-cascade.2 (clean up violations) is a
no-op because there are no true-positive violations in the corpus.

## Disposition summary

| Item | Status |
|---|---|
| W2-cascade.1 — Survey | This document. Zero warnings emitted. |
| W2-cascade.2 — Clean up violations | No-op. No true-positive violations in the corpus. |
| W2-cascade.3 — Flip warnings to errors, remove channel | Proceeds. Mirrors the F3 PR 2 closeout pattern (`docs/archive/investigations/linearity_f3_pr2_closeout.md`). |

The hello-chelis repository (a downstream consumer with its own
agent) is intentionally excluded from this in-tree survey, mirroring
F3 PR 1's exclusion. Any tuple-destructure linearity violations
there will surface to its dispatcher when the next sweep runs
against this binary, and must be fixed before downstream consumers
upgrade to a chelis release that includes this cascade.

## References

- Plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`
- Phase 0 spec lock: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`
- W1 diagnosis: `docs/archive/investigations/linearity_typed_consumekind_diagnosis.md`
- F3 closeout reference: `docs/archive/investigations/linearity_f3_pr2_closeout.md`
- §5 entry: `docs/archive/reports/gap_synthesis.md` row for `Linearity-F2`
- Survey driver: `crates/chelis-cli/tests/linearity_destructure_corpus_survey.rs`
  (lived during W2-cascade.1; removed in W2-cascade.3 alongside the
  warnings channel)
- W1 PR 1 fixture references: `crates/chelis-types/tests/linearity_typed_consumekind.rs`
  (Fixture 4 — `tuple_destructure_double_realize_warns_after_fix`)
  and `crates/chelis-types/tests/linearity_aliased_consume.rs`
  (Fixture 6 — `destructure_then_alias_consume_warns_after_fix`)
- Code anchors: `crates/chelis-types/src/linearity.rs`
  (`LinearityInfo::warnings`, `push_warning`, `Checker::push_diagnostic`,
  `destructure_warning_depth`, `bind_introduces_destructure_tmp`)
