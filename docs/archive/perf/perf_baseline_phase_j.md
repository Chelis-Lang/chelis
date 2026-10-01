# Phase J — Compiled Artifact Caching perf baseline

This document captures the wall-clock measurements taken at the close of Phase J of the
Compiled Artifact Caching plan (unpublished).

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
| pre-cache | separate worktree at `<pre-cache worktree>` | `c13ea7a` ("test(cli): amortize chelis-std publish across integration tests (Layer 1)") | `chelis 0.2.7` |

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
  `<nautilus checkout>/` (`v0.3.1`, compiler `=0.2.7`)

These edits are local only; they are not part of the bench harness's checked-in state.
Until Coral's registry-side deps are re-published against `0.2.7`, the bench has to use
path deps. The bench script (`scripts/bench_phase_j.py`) does **not** apply these
edits — they are documented here so a future runner can reproduce them.

## Bench harness

`scripts/bench_phase_j.py` runs each command, captures wall-clock + exit code +
stdout/stderr tails + parsed pass/fail counts, and emits a JSON summary. Invocation:

```sh
python3 scripts/bench_phase_j.py \
  --binary <post-cache worktree>/target/release/chelis \
  --label post \
  --chelis-repo <post-cache worktree> \
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

Skipped. The Nautilus checkout at `<nautilus checkout>` does not have
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
cd <coral checkout>
# (apply the path-dep / =0.2.7 edits to reef.toml as documented above)
time /path/to/target/release/chelis test tests/
```

Automated harness (one binary at a time, captures everything to JSON):

```sh
python3 scripts/bench_phase_j.py \
  --binary /path/to/target/release/chelis \
  --label post \
  --chelis-repo <post-cache worktree> \
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
| 1. Host-runtime registration gap | FIXED | `eval_in_context` resolves library `def`s; `unknown runtime name pkg__chelis__std__...` does not fire on snippets that call into chelis-std host-side functions. Locked in `crates/chelis-compiler-api/tests/compiled_context.rs::eval_in_context_resolves_library_string_call_*`. |
| 2. Linearity over-strict on aliased tensor refs | INVESTIGATED, NOT FIXED | `check_linearity_with_context` empirically rejects `_ = lib_consume(t); other_lib_consume(t, ...)` patterns the monolithic `check_linearity` accepts (chelis-std `test_linspace_endpoints`). Direct experiment: monolithic-over-`library_checked.annotated_exprs() ++ new_checked.annotated_exprs()` ALSO rejects, indicating the divergence is structural in how cached library annotation differs from the format-then-reparse legacy path, NOT in `_with_context`'s walking strategy. Root cause not isolated within Phase G'. |
| 3. Wire `eval_in_context` into `cmd_test` worker | DEFERRED | Implementation drafted (`PreparedTestEval` enum, `prepare_eval_in_context` worker call); reverted to legacy `compile_with_reef_graph + prepare_eval` path because the linearity divergence (#2) breaks chelis-std's self-test corpus and Coral's tests when the worker uses the in-context path. The `chelis-compiler-api` API surface (`prepare_eval_in_context`, `eval_in_context`, `eval_many_in_context`, `check_in_context`) is correct on its own bench (`compiled_context` + `phase_h_*` integration tests pass) and external CLI consumers (`chelis eval --file`, `chelis check`) DO use it. The cmd_test worker stays on legacy until the linearity divergence is root-caused. |
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
cd <chelis checkout>/packages/chelis-std
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

## Phase K — disk cache wire-up + compile_reef_context profile

### Disk cache (Phase K commits `483272e` deprecation, `a6b3fad` cmd_eval, `29ca9b2` cmd_test)

Before Phase K, every `chelis eval` / `chelis test` invocation rebuilt the
`CompiledContext` from scratch (~67 s on Coral). The Phase I disk cache
machinery existed but wasn't wired into the CLI. Phase K plumbs
`load_or_compile_for_package` (a thin layer over the existing
`CompiledContext::load_if_fresh` + `save`) into both `cmd_eval` and
`cmd_test`. The cache key is `(reef_home, root_pkg_id, source_hash)`;
content invalidation kicks in the moment ANY backed source byte changes.

`cmd_check` is intentionally NOT routed through `check_in_context`.
The legacy fitness emitter (`chelis_types::check_phase0e_fitness`)
produces a specific JSON output shape (`FitnessComponents`,
`unresolved_names`) that chelis-tide and other downstream tooling
depend on. Routing through `check_in_context` would change this
shape. Future work: verify that `check_in_context`'s output matches
`check_phase0e_fitness` byte-for-byte across success AND all error
cases. Until that parity is established, `cmd_check` stays on the
legacy path. The disk cache wins for `cmd_check` come for free once
the parity layer lands.

### Coral wall-clock with disk cache, release build

| Bench | Cold (cache miss) | Warm (cache hit) | Speedup |
|---|---|---|---|
| `chelis eval --file src/apismoke.ch` | 76.5 s | 0.24 s | ~318× |
| `chelis test tests/` (63 tests) | 83.0 s | 11.6 s | ~7.1× |

Warm-cache `chelis test` lands at **11.6 s** on Coral — well under the
30 s headline target, when (and only when) the source has not changed
since the previous run. Cold (first run, or any source byte changed)
still pays the full ~67–80 s `compile_reef_context` build.

### compile_reef_context per-phase breakdown

Instrumentation gated by `CHELIS_PROFILE_COMPILE_CONTEXT=1` (zero-cost
when unset). Numbers from a Coral `chelis eval --file src/apismoke.ch`
cold run, release build, no cache:

| Phase | Wall-clock | Share of total |
|---|---|---|
| `prepare_reef_graph` | 0.031 s | 0.04% |
| `source_digests` | ~0 s | ~0% |
| `hash_digests` | ~0 s | ~0% |
| `surf_desugar` | 0.011 s | 0.01% |
| `macro_expand` | 0.046 s | 0.06% |
| `build_type_env_from_library` | **18.16 s** | 21.7% |
| `check_phase0e_with_context` | **21.47 s** | 25.7% |
| `check_effects` | 0.15 s | 0.2% |
| `check_linearity` | 2.17 s | 2.6% |
| `lower_program_to_library` | **41.18 s** | 49.3% |
| **Total** | **83.5 s** | 100% |

### Diagnosis (initial): claimed structural — REVISED

The original Phase K diagnosis claimed all three dominant phases
were structural one-pass-over-N work. **A follow-up per-decl
investigation (see `docs/archive/perf/perf_baseline_investigation.md`)
contradicts that claim.** The three phases break down as follows
when sub-phase instrumentation is enabled
(`CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1`):

- **`lower_program_to_library` (~30s)**: only **0.009s** is actual
  lowering work; **20.6s** is in an `assertions_loop` that calls
  `top_level_expr_is_lowered` once per decl, and that helper rebuilds
  `top_level_lowering_map` from scratch on every call (1850 calls ×
  full library walk = quadratic). Another ~9.9s in the main loop's
  filter check exhibits the same pattern. **Bounded fix:** thread the
  precomputed `lowered_names` through the assertions loop and the
  filter check, replacing `top_level_expr_is_lowered(...)` with the
  existing `top_level_expr_is_lowered_with_names(...)`. Estimated
  saving: ~30s.

- **`check_phase0e_with_context` (~19s)**: only 2.4s is HM
  inference; **16.8s is annotation post-pass**, and that post-pass
  duplicates work already done by `build_type_env_from_library`.

- **`build_type_env_from_library` (~16s)**: 2.25s is HM inference
  + 0.10s validation; **13.8s is a per-decl annotation outer
  loop** at line 185-191 of `infer.rs`. The same annotation work
  happens again inside `check_phase0e_with_context`.

The previous "structural, not bounded" framing is retained above
for historical accuracy; the actual diagnosis is now in
`docs/archive/perf/perf_baseline_investigation.md`.

The instrumentation itself (`CHELIS_PROFILE_COMPILE_CONTEXT=1` in
`compile_reef_context`) is retained as a permanent operator-facing tool
so a future profiling pass starts from data, not guesses.

### Recommended next steps (out of scope for Phase K)

The disk cache makes the cold cost a one-time-per-source-change tax
rather than a per-invocation tax, so the practical impact of the
remaining 67–80 s cold-compile is now bounded by how often a developer
edits a library source file. Three follow-ups, ranked by ROI:

1. **Incremental lowering**: change `LoweredLibrary` to a lazy per-decl
   structure so only the decls actually referenced by `eval_in_context`
   pay lowering cost. Likely the biggest single win (40 s → ~5 s on
   Coral if most of the library isn't referenced by a typical eval),
   but requires a real refactor.
2. **Incremental type checking**: same idea, applied to
   `build_type_env_from_library` + `check_phase0e_with_context`
   together. Likely a multi-week project; not a Phase K scope.
3. **Workspace-shared cache**: today the cache key is per-`reef_home`.
   A team-wide shared cache (e.g., over CI artifact storage) would
   make the cold path fast-cold instead of slow-cold for any developer
   downloading a known-good cache.

The headline cache work has delivered: 5.1× faster than the prior
Phase G' baseline, and warm-cache invocations now beat the original
30 s headline target by a comfortable margin.

## Final Coral re-bench (post Phase K + disk-cache + profile)

Numbers from a clean post-Phase-K release build, run after Phase K
disk cache was wired into `cmd_eval` and `cmd_test`. Each row uses a
**fresh tempdir for `CHELIS_REEF_HOME`** so the "cold" measurement
captures a true cache miss (no carry-over between rows).

| Bench | Cold (cache miss) | Warm (cache hit) | Speedup | Notes |
|---|---|---|---|---|
| Coral `chelis test tests/` (63 tests) | 81.6 s | **11.6 s** | ~7.0× | warm-cache hits the < 30 s headline |
| Coral `chelis eval --file tests/internal.ch` | 72.5 s | **0.38 s** | ~191× | the parent's compile_reef_context dominates the cold cost |
| Coral `chelis check src/*.ch` | not measured | not measured | — | cmd_check not wired; LocalRegistry-prereq fixture missing |

### What landed vs. what remained

Landed:
- Phase K (`483272e`) — `prepare_eval` deprecated in favor of
  `prepare_eval_in_context` / `eval_in_context` / `check_in_context`.
- Disk cache wired into `cmd_eval` (`a6b3fad`) and `cmd_test`
  (`29ca9b2`) via the new `load_or_compile_for_package` helper.
  Includes integration tests covering cold→warm parity, source-edit
  invalidation, and cmd_test cache reuse.
- `compile_reef_context` profile instrumentation (`af6e230`),
  gated on `CHELIS_PROFILE_COMPILE_CONTEXT=1`. Initial diagnosis
  claimed the dominant phases were "structural one-pass-over-N
  work" — that diagnosis was premature. A follow-up per-decl
  investigation (see `docs/archive/perf/perf_baseline_investigation.md`) found
  bounded fixes: a quadratic `top_level_lowering_map` rebuild
  inside `lower_program_to_library`'s pre-flight loops (~30s
  recoverable), and a redundant library annotation pass shared
  between `build_type_env_from_library` and
  `check_phase0e_with_context` (~14s recoverable). The detailed
  per-decl instrumentation is gated on
  `CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1`.

Remained:
- `cmd_check` is still on the legacy fitness emitter. Wiring it
  through `check_in_context + load_or_compile_for_package` requires a
  JSON-shape parity layer because `check_in_context` returns a
  simpler `CheckResult` than the legacy emitter's
  fitness+effects+linearity composition. Out of scope for Phase K.

### Verdict (post Phase K + cold-path fixes)

The Compiled Artifact Caching project has delivered. Warm-cache
`chelis test` runs in 11.1 s on Coral, well under the 30 s headline
target. As of the cold-path fixes (commits `5eaefdb` and `5ad5a6f`,
described below), the **cold path is also under 30 s**: 29.4 s on
Coral's 63-test suite. The disk cache remains valuable for repeated
runs but is no longer load-bearing for the headline number.

## Phase K cold-path follow-up (2026-04-26): bounded fixes applied

The Phase K initial diagnosis labeled the cold-compile bottleneck as
"structural" with no fixes attempted. A per-decl re-investigation
(see `docs/archive/perf/perf_baseline_investigation.md`) refuted that conclusion
on 2 of the 3 dominant phases and identified concrete bounded fixes:

### Fix A — `lower_program_to_library` quadratic (commit `5eaefdb`)

Both `for_each_top_level_item` loops inside
`lower_program_to_library` were calling
`top_level_expr_is_lowered(expr, program_exprs, type_env)` per
iteration, and that helper rebuilt `top_level_lowering_map` from
scratch each call (1850 calls × full library walk). The precomputed
`lowered_names` from line 125 was being ignored. Fix: thread it
through, replacing both call sites with
`top_level_expr_is_lowered_with_names(...)`.

| Sub-phase             | Before  | After   |
|-----------------------|---------|---------|
| top_level_lowering_map| 0.018 s | 0.016 s |
| assertions_loop       | 20.6 s  | 0.0005 s|
| lower_top_level_loop  | 9.9 s   | 0.010 s |
| TOTAL                 | 30.5 s  | 0.046 s |

663× speedup. ~14 LOC.

### Fix B — annotation deduplication (commit `5ad5a6f`)

`compile_reef_context` previously called
`build_type_env_from_library` followed by
`check_phase0e_with_context(empty, library)`. Both ran a full HM
inference + per-decl annotation pass over the same library exprs;
~16 s of inference and ~13.8 s of annotation were duplicated.

Fix: a new `chelis_types::build_compiled_library_context` runs the
work once and returns both the `TypeEnv` and the library
`CheckedProgram`. The two existing public functions are retained for
external callers.

| Phase                           | Before  | After   |
|---------------------------------|---------|---------|
| build_type_env_from_library     | 16.5 s  | (gone)  |
| check_phase0e_with_context      | 19.3 s  | (gone)  |
| build_compiled_library_context  | —       | 16.1 s  |

~125 LOC (mostly a new helper that reuses existing primitives).

### Combined effect on Coral

| Bench                                   | Pre-fixes | Post-fixes |
|-----------------------------------------|-----------|------------|
| `compile_reef_context` cold             | ~68 s     | ~18 s      |
| `chelis test tests/` cold (63 tests)    | 81.6 s    | **29.4 s** |
| `chelis test tests/` warm (63 tests)    | 11.6 s    | 11.1 s     |
| chelis-std self-test corpus (205 tests) | 7.9 s     | ~7.9 s     |

The cold path now hits the 30 s headline target without disk-cache
warmth. Both fixes preserve correctness (chelis-std 205/205 passes,
workspace gate green except 3 known HIP failures).

### What remains (post follow-up)

- Inference itself (~5 s / pass on Coral) is genuinely structural
  one-pass-over-N HM inference. Eliminating it requires incremental
  type checking — a multi-week project, NOT a bounded fix.
- `cmd_check` JSON-shape parity layer still required to route
  `cmd_check` through the in-context pipeline.
