# PR #2683 red-team round 1

Reviewed head: `5810744986c43aee6353673c6f8e21f2986cedb0`.
Review base: `af5dd14c7ac6187a63e435b73551bbe3376865ce`.
Scope: `crates/chelis-cli/tests/runtime_extent_claim_preparation.rs`.

## Verdict

Satisfied. No in-scope P0/P1/P2/P3 findings and no out-of-scope defect discovered. Round 1 is closed. I remain available for verification of subsequent changes.

## Executed evidence

In `/Users/robertronan/chelis-worktrees/ci-batch-extent-fixtures`:

```sh
cargo nextest run -p chelis-cli --test runtime_extent_claim_preparation --test-threads 1 --success-output final -E 'test(=claimed_extent_contract) | test(=host_produced_reshape_targets_preserve_declared_claims) | test(=staged_reshape_sources_preserve_captures_and_order) | test(=computed_claim_result_graph_contract) | test(=omitted_extent_claim_contract) | test(=acceptance_cannot_be_satisfied_by_equal_wrong_answers_or_missing_roots) | test(=claims_and_traps_require_their_own_evidence) | test(=trap_context_keeps_source_axis_and_signed_value_together)'
```

All eight passed, 211.183 seconds; 21 unrelated tests were not selected. Warm-target compilation took 6.31 seconds. Full output is `/Users/robertronan/chelis-worktrees/cli-fixture-redteam-round1.log`.

Cache receipts:

| Matrix | Compiled | Reused |
| --- | ---: | ---: |
| Claimed extent | 6 | 6 |
| Result graph | 17 | 6 |
| Host target | 10 | 10 |
| Staged source | 11 | 9 |
| Omitted extent | 20 | 19 |

These tests retain individual successful-result, checker-rejection, runtime-trap, signature and source-context assertions. The three oracle controls reject equal wrong answers, missing roots, and misleading trap evidence. Binding/root routes still execute Eval and compile their C independently; exported routes execute a fresh process for each case. Existing executable examples included by these matrices passed with them. No test identity or CI cadence changed in the reviewed diff.

Adversarial command: `.venv/bin/python target/review-probe/run.py` (temporary probe removed after execution). It extracted the actual dynamic-driver generator and output formatter, used minimal tensor API stubs to inspect input bits at the exported call, and compiled the generated callers with `cc -O2`. The GeneratedHeader parser had a stand-in solely for the unused fallback; this probe exercised the authored-f branch. It passed:

- exact transport of negative zero, the smallest subnormal, a float adjacent to one, both infinities, and a NaN payload;
- scalar and empty-vector inputs;
- missing and extra arguments returning 11;
- identical caller source despite changed vector extents and data.

This probe validates transport and caller generation; the matrix execution above validates the real runtime and generated export ABI.

## Static contract audit and limits

The cache keys exact appended C source, generated header, resolved compiler/flags and complete staged-runtime receipt. The receipt publisher verifies the archive and includes hashes for public headers. The linked toolchain and the key use matching requirements. Every case still runs checker and build before lookup, and cache lifetime is one matrix invocation. Reviewed the call-matrix routes, unchanged result/trap oracle, runtime staging publisher, shared C link helper, and runtime-extents design acceptance descriptions.

No full workspace suite, hosted CI, package expansion, or GPU execution was independently validated in this round. Runtime/toolchain changes during a matrix were audited in source rather than injected; the receipt consistency assertion fails closed on a changed receipt. No broad phase-completion claim is made.

## Restoration

No tracked source was mutated. Temporary probe files and binaries were removed; only the durable external report/log remain. `git status --short` and `git diff --check` were clean. Final worktree probe at 2026-09-27T18:57:38Z reported FREE, exact reviewed HEAD, zero modified/staged/untracked/unmerged files, and no processes scoped to the checkout. CPU handoff was sent to the author and census agent after execution stopped.
