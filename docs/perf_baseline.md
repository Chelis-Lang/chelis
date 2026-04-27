# Phase J — Compiled Artifact Caching perf baseline

This document captures the wall-clock measurements taken at the close of Phase J of the
Compiled Artifact Caching plan (see
`/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md`).

The headline question for Phase J was: does the cache deliver on its core promise of
running Coral's 63 tests in under 30 seconds, down from the 15-minute pre-cache regime?

**Answer: no.** The post-cache build runs Coral's 63 tests in **9 m 17.8 s (557.8 s)**
on this workstation — not the < 30 s target. Per-file measurements show the cache is
*slower* than pre-cache for single-file workloads on the same workstation (post = ~2x
pre). The bottleneck is documented in §"Bottleneck analysis" below.

The numbers in this file are NOT a regression bound. They are a baseline for the next
round of work (Phase G/G' in-context-evaluator fixes + disk-cache wire-up). Once those
land, the same bench script (`scripts/bench_phase_j.py`) re-runs and the table updates.

## Environment

- Host OS: Fedora 43 (`Linux 6.18.16-200.fc43.x86_64`)
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
- GPU: Radeon 8060S Graphics (gfx1100; not exercised by these benches)
- Rust toolchain: workspace-pinned (`rust-toolchain.toml`)
- Build profile: `cargo build --release -p chelis-cli`
- Concurrency: every bench was run with no other chelis processes contending for CPU
  unless explicitly noted

## Binaries under test

| Label | Source | Commit | `chelis --version` |
|---|---|---|---|
| post-cache | `target/release/chelis` in this worktree | `e3b7ccc` (Phases A–I + RT-G/H/I) | `chelis 0.2.7` |
| pre-cache | separate worktree at `/home/jeff/Documents/scratch/chelis-phase-j-baseline` | `c13ea7a` ("test(cli): amortize chelis-std publish across integration tests (Layer 1)") | `chelis 0.2.7` |

The pre-cache commit is the parent of Wave 1 of the cache plan — the last commit before
any of the AST serde / `CompiledContext` / `compile_reef_context` machinery landed.

## Repo prep

Coral pinned `compiler = "=0.2.5"` and depended on registry-published `chelis-std@0.1.0` +
`nautilus@0.3.0`, both also pinned to `=0.2.5`. The current chelis worktree is `0.2.7`;
the registry-published shells of the deps are immutable, so they can never be loaded by a
`0.2.7` compiler without a re-publish.

To make Coral build at all on this branch, the bench changes Coral's `reef.toml` to:

- `package.compiler = "=0.2.7"`
- `[dependencies]` rewritten to **path** deps, pointing at the in-monorepo
  `packages/chelis-std/` (compiler `=0.2.7`) and the local Nautilus checkout
  `/home/jeff/Documents/scratch/nautilus/` (`v0.3.1`, compiler `=0.2.7`)

These edits are local only; they are not part of the bench harness's checked-in state.
Until Coral's registry-side deps are re-published against `0.2.7`, the bench has to use
path deps. The bench script (`scripts/bench_phase_j.py`) does **not** apply these
edits — they are documented here so a future runner can reproduce them.

## Bench harness

`scripts/bench_phase_j.py` runs each command, captures wall-clock + exit code +
stdout/stderr tails + parsed pass/fail counts, and emits a JSON summary. Invocation:

```sh
python3 scripts/bench_phase_j.py \
  --binary /home/jeff/Documents/scratch/chelis/.claude/worktrees/agent-ac1369283ad281583/target/release/chelis \
  --label post \
  --chelis-repo /home/jeff/Documents/scratch/chelis/.claude/worktrees/agent-ac1369283ad281583 \
  --out /tmp/chelis-bench/post.json
```

Most numbers in the tables below were collected manually with `time` rather than via
the harness (the harness exists as a regression-friendly entry point); both produce the
same wall-clock figure.

## Per-bench measurements

All wall-clock figures are real-time seconds. Pre/post are the two binaries above.

### 1. Coral `chelis test tests/` — the headline (63 tests across 8 files)

| Metric | pre-cache `c13ea7a` | post-cache `e3b7ccc` | Ratio |
|---|---|---|---|
| Wall clock | **6 m 40.1 s (400.08 s)** | **9 m 17.8 s (557.77 s)** | **post is 1.39x SLOWER than pre** |
| Tests passed / failed | 63 / 0 | 63 / 0 | parity |
| Hits the 30 s target? | n/a — target is post-cache | **NO — exceeds by ~18.6x** | — |

The plan cited the pre-cache wall-clock as "15+ minutes." The current measured figure
on this workstation is **6 m 40 s**. Two possibilities — (a) the original 15-minute
figure was on a different machine or with a different test corpus, or (b) some
intermediate landed work shrank the pre-cache cost between the original measurement and
`c13ea7a`. Either way, the relevant comparison for **this** branch is pre vs post on
the same machine on the same commit pair: pre 400 s, post 558 s.

Post-cache is **39 % slower** than pre-cache on the headline benchmark. The cache wiring
is a regression on `chelis test`, not a speedup. The < 30 s target is not even close —
the cache would need to be 18.6 x faster than today's post-cache figure to hit it, and
~13.3 x faster than today's pre-cache figure.

### 2. Coral `chelis test tests/internal.ch` — single-file apples-to-apples

A single-file run isolates per-file overhead from N-file cache amortization. This bench
is the cleanest test of "does the cache help?" because Coral's chelis-std/nautilus
context is the same for every file but the test file's compile is not.

| Metric | pre-cache | post-cache | Ratio |
|---|---|---|---|
| Wall clock (no contention) | **60.31 s** | **114.67 s** | **post is 1.90x SLOWER** |
| Tests passed / failed | 6 / 0 | 6 / 0 | parity |

A second post-cache solo measurement reproduced the slowdown. The single-test
(`--filter test_string_sort_lexicographic`) variant gave pre = 47.79 s, post = 111.81 s
(post 2.34x slower); the filter result strips per-test cost and isolates parent setup
overhead, which is the source of the regression (see §"Bottleneck analysis").

### 3. Coral `chelis check src/core.ch`

`cmd_check` was reverted to the legacy path during RT-H per the Phase J prompt; this
bench just confirms parity with pre-cache.

| Metric | pre-cache | post-cache | Ratio |
|---|---|---|---|
| Wall clock | **28.76 s** | **28.93 s** | parity (within noise) |

Disk cache for `chelis check` is **not yet wired**; once it is, this number should drop
to < 500 ms on warm runs.

### 4. Coral `chelis eval --file <small_program.ch>`

Tiny eval target: `def main() -> int64 = cast(42, int64)` (no Coral imports — Coral's
own modules' `main`-style functions otherwise dominate eval execution). Eval routes
through `compile_reef_context + eval_in_context` on post-cache for reef-in-scope sources.

| Metric | pre-cache (legacy) | post-cache (`eval_in_context`) | Ratio |
|---|---|---|---|
| Cold (first invocation) | **67.80 s** | **64.80 s** | parity |
| Warm (second invocation) | n/a (no caching) | **65.12 s** | n/a |

The post-cache eval surface is no faster than pre-cache and does not benefit from
warmth (Phase I disk cache exists in the API but is not wired into `cmd_eval`; the
plan's "Critical instructions" section explicitly notes this gap).

### 5. chelis-std self-test corpus (`packages/chelis-std/tests/decimal.ch`, 18 tests)

The cmd_test agent measured **276 s** on the full corpus (35 files / 205 tests) pre-fix.
Running the full corpus on both binaries here would have cost ~30+ minutes; one
representative file (decimal.ch, no chelis-std deps because chelis-std IS the package)
was used as a proxy.

| Metric | pre-cache | post-cache | Ratio |
|---|---|---|---|
| Wall clock | **2.20 s** | **4.76 s** | **post is 2.16x SLOWER** |
| Tests passed / failed | 18 / 0 | 18 / 0 | parity |

This bench has the cleanest interpretation: chelis-std has no library deps, so the cache
should be a pure overhead with zero benefit — and indeed it is. The 2x slowdown is
the cache wiring itself (24 MB serialization + worker decode + worker re-runs of
`prepare_eval` anyway).

### 6. Nautilus

Skipped. The Nautilus checkout at `/home/jeff/Documents/scratch/nautilus` does not have
a fast self-test entry point distinct from the `chelis test` flow already exercised
above; the prompt called this bench optional.

## Headline summary

| Bench | Pre-cache | Post-cache | Hit target? |
|---|---|---|---|
| Coral `chelis test tests/` (63 tests) | **6 m 40.1 s** | **9 m 17.8 s** | **NO** (target < 30 s; post is 39 % slower than pre) |
| Coral `chelis test tests/internal.ch` (6 tests) | 60.3 s | 114.7 s | n/a; cache 1.90 x slower |
| Coral `chelis check src/core.ch` | 28.8 s | 28.9 s | parity (legacy path) |
| Coral `chelis eval --file` cold | 67.8 s | 64.8 s | parity |
| Coral `chelis eval --file` warm | n/a | 65.1 s | NO (target < 1 s; disk cache unwired) |
| chelis-std `chelis test decimal.ch` (18 tests) | 2.2 s | 4.8 s | n/a; cache 2.16 x slower |

## Bottleneck analysis

The Phase J prompt warned that two pre-existing Phase G/G' bugs block the in-context
evaluator on chelis-std, so workers in `chelis test` still run the legacy `prepare_eval`
pipeline despite the cache being plumbed all the way to them:

- `unknown runtime name pkg__chelis__std__Std__Time__is_leap_year` (host-runtime gap)
- `check_linearity_with_context` over-strict on aliased tensor refs (`linspace ->
  assert_shape -> assert_close_tensor`)

Source of the bottleneck, in `crates/chelis-cli/src/main.rs`:

```rust
fn prepare_eval_in_exec_context(
    exec_context: &TestExecutionContext,
    synth_decls: &[Decl],
) -> Result<chelis_compiler_api::compiler::PreparedEval, String> {
    let prepared = chelis_reef::compile_with_reef_graph(exec_context.reef_graph(), synth_decls)
        .map_err(|e| e.to_string())?;
    let source_text = chelis_surf::format::format_program(&prepared.decls);
    chelis_compiler_api::compiler::prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source_text,
        bindings: BTreeMap::new(),
    })
    ...
```

The worker reformats the prepared decls to text and then runs `prepare_eval` from
scratch — full parse / desugar / Phase0e check / effects / linearity / lower — over the
entire chelis-std + Coral source. The cache plumbing did not change this; it only saved
the parent's reef-graph walk.

Add to that:

- The parent's `compile_reef_context` builds and serializes a 24 MB
  `CompiledContext` to a tempfile **even when no test will use it for skipping work**.
  Verified by `ls -la /tmp/chelis-compiled-context-*.bin` after a `chelis test` run.
- Each worker reads + decodes that 24 MB tempfile (on every test file) before doing
  anything else.
- Each worker still pays the per-file `prepare_eval` cost **on top of** the new
  decode-the-tempfile cost.

Net: parent overhead + per-worker overhead, no per-worker savings.

The "no-match filter" measurement isolates the parent overhead specifically:

| `chelis test tests/internal.ch --filter __no_match__` | wall clock |
|---|---|
| pre-cache | 0.07 s (filter short-circuits before worker spawn) |
| post-cache | 64.88 s |

The 65 s gap there is essentially pure cache-build overhead. (Pre-cache also short-
circuits earlier on a no-match filter; post-cache does the full setup before the no-
match check, so the comparison is not strictly apples-to-apples but it bounds the cost.)

### What needs to change for the cache to deliver

In rough priority order (matches the deferred items in the plan):

1. **Make workers actually use the cache.** Replace `prepare_eval_in_exec_context`'s
   text-reformat-and-reparse with an `eval_in_context` call so the worker uses the
   library decls already in the `CompiledContext` instead of re-checking and re-lowering
   them. Today this is blocked on the two Phase G/G' bugs above; fixing those is the
   load-bearing work for Phase J's headline target.
2. **Wire the disk cache into `cmd_eval` and `cmd_check`.** `CompiledContext::save /
   load_if_fresh` exists on `e3b7ccc`. Calling sites do not currently use it; second
   invocations therefore pay the full cold cost. The plan's Phase I acceptance bullet
   (`chelis eval --file` warm < 1 s) is unmet because of that gap, not because the API
   is missing.
3. **Skip cache build when no test file needs it.** A `--filter` that matches zero tests
   should never trigger `compile_reef_context` in the parent. Today it does, regardless.
4. **Avoid the 24 MB serialization round-trip on the same machine.** A shared-memory or
   in-process channel would skip the encode/decode entirely; that is a Phase I+ concern,
   not a Phase J fix.

Phase J's deliverable is the honest baseline above; the fixes are tracked separately.

## Reproducing the bench

Manual (full Coral suite):

```sh
cd /home/jeff/Documents/scratch/coral
# (apply the path-dep / =0.2.7 edits to reef.toml as documented above)
time /path/to/target/release/chelis test tests/
```

Automated harness (one binary at a time, captures everything to JSON):

```sh
python3 scripts/bench_phase_j.py \
  --binary /path/to/target/release/chelis \
  --label post \
  --chelis-repo /home/jeff/Documents/scratch/chelis/.claude/worktrees/agent-ac1369283ad281583 \
  --out /tmp/chelis-bench/post.json
```

The harness skips disk-cache warming because the disk cache is not wired into the CLI
yet; "warm" is just a second cold call.

## Manual gate

A new row, `phase4_perf_baseline`, has been added to `docs/manual_gates.md`. Its
expected outcome is **not** "the bench hits 30 s"; the gate's purpose is to keep the
bench reproducible so a future regression triggers visible noise. The success criterion
recorded there is "bench script exits 0 and JSON output contains numbers for every
named row." The 30 s contract will only be added once the bottleneck above is fixed.

## Phase G' re-bench (2026-04-26)

Phase G' addressed the four bugs documented above; the re-bench numbers
below replace the post-cache baseline ONLY in the sense that they sit
on a later commit. The original Phase J baseline above remains the
historical record of where the cache work landed empty-handed.

### Scope outcomes

| Scope | Status | Effect on benches |
|---|---|---|
| 1. Host-runtime registration gap | FIXED | `eval_in_context` resolves library `def`s; `unknown runtime name pkg__chelis__std__...` does not fire on snippets that call into chelis-std host-side functions. Locked in `crates/chelis-compiler-api/tests/phase_g_compiled_context.rs::eval_in_context_resolves_library_string_call_*`. |
| 2. Linearity over-strict on aliased tensor refs | INVESTIGATED, NOT FIXED | `check_linearity_with_context` empirically rejects `_ = lib_consume(t); other_lib_consume(t, ...)` patterns the monolithic `check_linearity` accepts (chelis-std `test_linspace_endpoints`). Direct experiment: monolithic-over-`library_checked.annotated_exprs() ++ new_checked.annotated_exprs()` ALSO rejects, indicating the divergence is structural in how cached library annotation differs from the format-then-reparse legacy path, NOT in `_with_context`'s walking strategy. Root cause not isolated within Phase G'. |
| 3. Wire `eval_in_context` into `cmd_test` worker | DEFERRED | Implementation drafted (`PreparedTestEval` enum, `prepare_eval_in_context` worker call); reverted to legacy `compile_with_reef_graph + prepare_eval` path because the linearity divergence (#2) breaks chelis-std's self-test corpus and Coral's tests when the worker uses the in-context path. The `chelis-compiler-api` API surface (`prepare_eval_in_context`, `eval_in_context`, `eval_many_in_context`, `check_in_context`) is correct on its own bench (`phase_g_compiled_context` + `phase_h_*` integration tests pass) and external CLI consumers (`chelis eval --file`, `chelis check`) DO use it. The cmd_test worker stays on legacy until the linearity divergence is root-caused. |
| 4. Skip cache build on no-match filter | FIXED | `chelis test --filter __no_match__` returns in 0.003 s instead of 64.88 s (21,000× speedup). Pre-flight surf-parse scan in `cmd_test` short-circuits before any reef-graph or context-build work. |

### Re-bench numbers (Coral, post-Phase-G')

Same workstation, same Coral checkout commit `53b2fd6` with the path-dep
edits documented above.

| Bench | Pre-cache `c13ea7a` | Post-cache `e3b7ccc` | Phase G' | Hit target? |
|---|---|---|---|---|
| Coral `chelis test tests/` (63 tests) | 6 m 40.1 s | 9 m 17.8 s | **6 m 38.0 s** | NO (target < 30 s; on par with pre-cache) |
| Coral `chelis test tests/ --filter __no_match__` | 0.07 s | 64.88 s | **0.003 s** | YES — 21,000× speedup |
| Coral `chelis check src/core.ch` | 28.76 s | 28.93 s | n/a — unchanged (legacy) | n/a |

The 6 m 38 s post-G' figure is parity with pre-cache (-29% vs post-cache,
matching pre-cache wall-clock). It does NOT hit the 30 s target. The
remaining bottleneck is the worker's per-file `prepare_eval` cost
(library re-compile every spawn) — exactly the load-bearing fix that
Scope 3 was supposed to deliver. With Scope 3 deferred, the cache wins
for `chelis eval --file` / `chelis check` (which consume the in-context
API directly) but NOT for `chelis test` (which runs through the
worker's legacy path until the linearity divergence is fixed).

### What's blocking the 30 s target

Root-causing the linearity divergence is the only remaining gate.
Reproduction:

```sh
cd /home/jeff/Documents/scratch/chelis/.../packages/chelis-std
target/release/chelis test tests/tensor/construct.ch --filter test_linspace_endpoints
# Expected (legacy worker path): PASS
# In-context worker path: FAIL with `variable 'actual' was already consumed by call to 'pkg__chelis__std__Std__Test__assert_shape' at offset 0`
```

The in-context path's `compile_new_source_in_context` calls
`check_linearity_with_context(library_checked, new_checked)`. Both args
are correctly built (`check_phase0e_with_context` annotates the new
exprs against the library's TypeEnv; library was already linearity-
checked at context build). Yet the two-call-on-`actual` pattern is
flagged as use-after-consume even though the format-then-reparse
legacy path's monolithic `check_linearity(library_text + new_text)`
accepts it. A direct comparison run from inside
`compile_new_source_in_context` — concatenating
`library_checked.annotated_exprs()` and `new_checked.annotated_exprs()`
into one `CheckedProgram` and running monolithic `check_linearity` over
the concatenation — ALSO rejects, so the bug is NOT in
`_with_context`'s walking strategy. It's structural in either:

1. how `library_checked.annotated_exprs()` (built once in
   `compile_reef_context`) differs from the same library re-annotated
   alongside new code in the legacy path; OR
2. how `chelis_reef::rewrite_entry_decls_with_reef_graph` produces deep
   AST with synthesized spans (the error reports both consume + reuse
   sites at offset 0, suggesting span loss); OR
3. some interaction between macro expansion of the library context vs
   the legacy single-pass macro expansion over the combined source.

Next investigation: dump
`library_checked.annotated_exprs()[that_def].canonical_print()` vs the
legacy path's deep-print of the same def from a format-reparse cycle,
diff the two. If they differ, fix
`compile_reef_context`'s annotation step. If they agree, the bug is in
the new-code annotation interacting with the cached library type-env.

## See also

- `/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md` — the full phase plan
- `crates/chelis-compiler-api/src/context.rs` — `CompiledContext` + `save / load_if_fresh`
- `crates/chelis-cli/src/main.rs` — `cmd_test`, `cmd_internal_test_file`,
  `prepare_eval_in_exec_context` (the line where workers re-enter the legacy pipeline)
- `crates/chelis-cli/tests/phase3t_compiled_context_bridge.rs` — env-var bridge regression

## Phase G' (final, post-linearity-fix + Json codegen fix) re-bench (2026-04-26)

With the linearity divergence root-caused (Scope 2 of the prior G'
deferral) and the worker re-wired through `prepare_eval_in_context`
(Scope 3), `cmd_test` now amortizes the library compile across every
file in a single run.

### Outcomes

| Scope | Status |
|---|---|
| 1. Host-runtime registration gap | FIXED (in prior G') |
| 2. Linearity divergence | FIXED (`annotate_phase0e_program` registers prelude ADTs; affected sites rewritten with `&t` / `copy(t)`; commits `04f1873` + `d26019d`) |
| 3. cmd_test worker through `eval_in_context` | FIXED (commit `bf0d568` + parent build re-enable here) |
| 4. Skip cache build on no-match filter | FIXED (in prior G') |

### Re-bench numbers (post-final)

Same workstation, same Coral checkout. Wall-clock from `time chelis test tests/`:

| Bench | Pre-cache `c13ea7a` | Post-cache `e3b7ccc` | Phase G' (parity) | Phase G' (final, this branch) | Hit < 30 s target? |
|---|---|---|---|---|---|
| Coral `chelis test tests/` (63 tests) | 6 m 40.1 s | 9 m 17.8 s | 6 m 38.0 s | **1 m 18 s** | NO (still > 30 s; 5.1× speedup vs prior G') |
| chelis-std self-test corpus (`chelis test packages/chelis-std/tests/`, 205 tests) | n/a measured | n/a measured | n/a | **7.9 s** | YES |
| Coral `chelis test tests/ --filter __no_match__` | 0.07 s | 64.88 s | 0.003 s | 0.003 s (unchanged) | YES |

### Where the remaining 1 m 18 s on Coral goes

The dominant cost is now the parent's single `compile_reef_context`
build itself. `time chelis eval --file tests/frame.ch` (which uses the
same context-build path) takes ~67 s on its own — i.e., the parent's
cost. Per-file worker overhead in the new path is ~10 s amortized, far
below the file-count × per-file cost that dominated the prior G' run.

For chelis-std (smaller library footprint), the parent build is fast
enough that 205 tests across 35 files run in 7.9 s end-to-end. So the
cache architecture is sound; the remaining > 30 s gap on Coral is a
single-call perf bottleneck in `compile_reef_context` itself, not in
the per-file test loop.

### Recommended next bottleneck

- Profile `compile_reef_context` for Coral specifically. The 67 s
  single-call cost is the new ceiling. Within that, the most likely
  culprits are: macro expansion of the library decls (~40 modules of
  Coral + ~50 modules of chelis-std), the monolithic
  `check_phase0e_with_context` library check, or the
  `lower_program_to_library` step.
- Disk cache (Phase I) once the in-memory cache is fast enough that
  the disk bridge is worth measuring.

The cache architecture is sound and we have a known surface (single
`compile_reef_context` call cost) to profile next.
