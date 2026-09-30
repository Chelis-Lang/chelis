# Initializer C batching feasibility and timing

**Decision: reject the combined batch.** It preserves the tested trap ordering, but the two focused tests take about 17 seconds longer warm and quiet. The tracked edit was removed. No push, PR, gate, or agent was used.

## Scope and files

- Worktree `/Users/robertronan/chelis-worktrees/ci-initializer-batch`, branch `perf/ci-initializer-batch`, HEAD and `origin/main` both `bb1a5ed6638f84bc9e5efc092b8fbc5955c22d57`; own `.venv` is Python 3.11.14.
- Tracked changes: **none**; status and diff are clean. The only edited tracked path was `crates/chelis-cli/tests/key_std_initialisers_cli.rs`. Rejected candidate and exact baseline snapshots are `target/ci-initializer-batch-probe/candidate_key_std_initialisers_cli.rs` and `baseline_key_std_initialisers_cli.rs`.
- Task-owned scratch: `target/ci-initializer-batch-probe/` contains the Surf probes, generated C/header, compiled drivers, verdict JSON, and timing logs. The isolated Rust target is `target/agents/ci-initializer-batch/`.

## Feasibility verdict

The minimal definition-only Surf probes built and linked against the **real generated C header**. The driver resolves package-qualified exports from that header. The generated observation `main` would call every nullary definition, so generated C was compiled with `-Dmain=chelis_probe_unused_main` and linked with a case-selecting driver. Each case was a fresh process.

| Shape | Invalid case | Valid twin | Verdict |
|---|---|---|---|
| Plain | `numeric trap: domain in cast at bool` | `numeric trap: overflow in cast at i8` | Both passed |
| Grad | `numeric trap: domain in cast at bool` | `numeric trap: overflow in cast at i8` | Both passed |

All four returned nonzero (`-6` here). Invalid cases lacked `in cast at i8`; valid twins lacked the bool-domain trap. See `probe-results.json`. The exported grad wrapper and top-level grad control each called a generated tensor DAG entry before converting to `f32`. Exact equivalence of their generated functions was not proved.

The rejected candidate's `c_batch_source` at snapshot line 216 generated all six initializers with invalid/valid exports per shape. `CBatch::build` at line 332 built and linked one C unit per shape and validated its header. `CBatch::run` at line 449 launched a **new C process per case**. `expect_trap` at line 460 still created a fresh app and launched a **new Eval process per case**: 24 Eval and 24 C processes. Lines 485-533 kept both tests and each case's required/forbidden trap assertions. A C build error was reported while Eval cases continued.

The restored baseline's corresponding locations are `eval_lane` at line 230, `c_lane` at 243, `expect_trap` at 280, and the two tests at 331 and 338. The candidate retained the same required/forbidden text oracle. This is per-case trap-order equivalence, not byte-for-byte equality of Eval and C transcripts.

## Smallest meaningful warm comparison

Both measurements ran the two focused tests via `cargo nextest`, `--profile ci-fast`, `--test-threads 1`, the same case filter, locked dependencies, own cargo target, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, and the same warm `CHELIS_TEST_SHARED_REEF_HOME`. Process checks before timing showed no competing cargo/rustc/nextest work. Both sides had a prior warm run; the first candidate timed run followed a warm focused validation that may have overlapped scanner activity, so a further **uncontended candidate warm run** was done. The scanner-contended interrupted baseline log was excluded.

| Run | Plain | Grad | Two-test nextest | Wall |
|---|---:|---:|---:|---:|
| Baseline warm (`baseline-quiet-warm.log`) | 12.366s | 30.725s | 43.091s | — |
| Baseline quiet timed (`baseline-timed-quiet.log`) | 12.447s | 30.803s | 43.251s | 43.63s |
| Batch quiet timed (`candidate-timed-quiet.log`) | 7.728s | 52.284s | 60.012s | 60.41s |
| Batch additional quiet warm (`candidate-quiet-warm.log`) | 7.760s | 51.526s | 59.286s | — |

All listed runs passed both tests. Versus the timed baseline, batching saves **4.719s plain** and costs **21.481s grad**, for a **16.761s net regression** in nextest time (16.78s wall). The additional warm run confirms the slow grad result. The focused candidate validation also passed at 7.759s plain / 51.576s grad, but is not used for the quiet comparison.

The likely cost center is building or compiling a single C unit containing 12 grad exports; that stage was **not timed separately**, so this is a hypothesis. Both lanes are technically batchable; the grad batch is economically poor in this configuration. A plain-only batch could save roughly five seconds on this pair, but was not implemented or measured as an independent candidate. The full four-test file, fast gate, and CI were not run for the discarded edit. Final process check found no cargo/rustc/nextest processes.
