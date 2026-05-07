# compile_reef_context per-decl investigation (Phase K follow-up)

This document reports the findings of a per-decl-granularity profile of
Coral's cold `compile_reef_context` build, instrumented inside the
three dominant phases (`build_type_env_from_library`,
`check_phase0e_with_context`, `lower_program_to_library`).

The previous agent's coarse-phase profile, now archived at
`docs/archive/perf/perf_baseline_phase_j.md`,
concluded the three phases were "structural, no fix attempted." This
investigation refutes that conclusion: **two of the three phases have
clearly bounded fixes** that explain the bulk of their cost.

## Step 1: what `compile_reef_context` actually calls

`compile_reef_context` calls these in order, each ONCE on the whole
linked library (no per-module loop):

| Step | Call | What it does |
|---|---|---|
| 1 | `prepare_reef_graph(package_dir)` | Resolve packages, link library decls into a flat list |
| 2 | `reef_state.source_digests()` | Hash sources for cache-key |
| 3 | `desugar_program(linked_library_decls)` | Surf-to-Deep |
| 4 | `expand_program(desugared)` | Macro expand |
| 5 | `build_type_env_from_library(deep_library_decls)` | Type-env snapshot for `_with_context` |
| 6 | `check_phase0e_with_context(TypeEnv::empty(), deep_library_decls)` | Phase 0e check (called WITH EMPTY CONTEXT — i.e., monolithic library check) |
| 7 | `chelis_effects::check_program(checked)` | Effects |
| 8 | `check_linearity(checked)` | Linearity |
| 9 | `lower_program_to_library(library_checked)` | IR lowering |

### Resolution of the user's structural suspicions

> **Suspicion #1**: `check_phase0e_with_context` taking 21.5s during
> context build is wrong by design.

**Partially right.** The function name is misleading: in this code
path it is called with `TypeEnv::empty()`, so it is not "checking new
code against a pre-built library context" — it IS the monolithic
library check. There is no per-module loop; one big call over 1850
flat decls. Not redundant work in the sense of "re-checking the
library" — it IS the only check the library gets. **But:** see Step 3,
where 16.8s of its 19.3s is in the annotation post-pass, which IS
redundant with similar work done by `build_type_env_from_library`.

> **Suspicion #2**: `build_type_env_from_library` taking 18.2s to
> build a name→type map is suspicious — likely doing more than
> map-building.

**Right.** Of the 16.2s phase, only 2.25s is type inference. **13.8s
is a per-decl annotation loop** (line 185-191) that walks every
library decl recursively to attach inferred type metadata. This loop
is duplicated by `check_phase0e_with_context`'s own annotation pass
(16.8s on the same exprs). Two full annotation passes over the same
library.

> **Suspicion #3**: `lower_program_to_library` at 41.2s for ~90 modules
> is ~450ms per module — likely hidden quadratic.

**Confirmed quadratic.** Of the 30.5s phase, only **0.009s** is
actual `lower_top_level` work. The rest is:
- `assertions_loop` (line 115-120): **20.6s** of repeated
  `top_level_expr_is_lowered` calls that each rebuild
  `top_level_lowering_map` from scratch.
- `lower_top_level_loop` wrapper (line 132-179): **9.9s** of the same
  pattern when `top_level_expr_name(expr).is_none()` triggers a
  fallback `top_level_expr_is_lowered` call.

## Step 2: per-decl breakdown for each phase

Coral's library has **1850 flat top-level decls** (no module wrappers
— `linked_library_decls` is already flattened in
`compile_with_reef_graph`). The instrumentation is gated on a new
`CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1` env var; default behavior
is unchanged.

### Phase totals (cold, fresh tempdir for `CHELIS_REEF_HOME`)

| Phase | Wall | Of total |
|---|---|---|
| `build_type_env_from_library` | 16.2s | 24% |
| `check_phase0e_with_context` | 19.3s | 28% |
| `lower_program_to_library` | 30.5s | 45% |
| (everything else) | ~2s | 3% |
| **Total** | **~68s** | 100% |

### `build_type_env_from_library` sub-phases

| Sub-phase | Wall |
|---|---|
| `build_phase0e_type_env_initial` | 0.0s |
| `infer_phase0e_program_with_state` (per-decl loop) | 2.25s |
| `validate_phase0e_program` | 0.10s |
| `validate_tensor_precisions` | 0.003s |
| `suppress_unbound_for_cycle` | 0.0s |
| `collect_library_def_names` | 0.0s |
| **`annotate_library_exprs_outer_loop`** | **13.8s** |
| `build_phase0e_type_env_from_annotated` | 0.002s |

### `check_phase0e_with_context` sub-phases

| Sub-phase | Wall |
|---|---|
| `build_phase0e_and_combine` | 0.0s |
| `infer_phase0e_program_with_state` (per-decl loop) | 2.4s |
| `validate_phase0e_program` | 0.11s |
| `validate_tensor_precisions` | 0.004s |
| `suppress_unbound_for_cycle` | 0.0s |
| **`annotate_phase0e_program_with_context`** | **16.8s** |
| `annotated_type_env_build` | 0.002s |

### `lower_program_to_library` sub-phases

| Sub-phase | Wall |
|---|---|
| `top_level_lowering_map` (initial) | 0.018s |
| **`assertions_loop`** (line 115-120, repeated map rebuilds) | **20.6s** |
| `program_types_build` | 0.0s |
| `collect_top_level_defs` | 0.005s |
| `lower_ctx_new` | 0.010s |
| **`lower_top_level_loop`** (incl. repeated map rebuilds in fallback branch) | **9.9s** |
| `flatten_bindings`, `dce`, `renumber_symbol_table` | 0.0s |

Within `lower_top_level_loop`, the actual `lower_top_level` work
summed per-decl is **0.009s** — meaning the wrapping
`top_level_expr_is_lowered` filter check accounts for the other ~9.9s.

### Top-5 slowest decls per phase (per-decl detail)

#### Inference pass A (build_type_env's `infer_phase0e_program_with_state`)

| Time | Decl |
|---|---|
| 26.9ms | pkg__nautilus__Nautilus__Roots__brent_rec |
| 26.8ms | pkg__nautilus__Nautilus__Optim__opt_brent_rec |
| 23.1ms | pkg__nautilus__Nautilus__ODE__rk45_dopri_step_vec_core |
| 22.1ms | pkg__nautilus__Nautilus__Interpolation__spline_eval |
| 21.8ms | pkg__nautilus__Nautilus__ODE__rk45_dopri_step |

Median per-decl: ~0ms. Mean: 1.13ms. Distribution is uniform-ish
across 1850 decls; pass total is 2.25s. **No pathological outlier
modules.**

#### Inference pass B (check_phase0e_with_context's `infer_phase0e_program_with_state`)

Same shape, same top-5 (different order; runtimes slightly higher
due to the larger initial state).

#### Annotation pass (per-decl in `check_phase0e_with_context`)

| Time | Decl |
|---|---|
| 264.8ms | pkg__nautilus__Nautilus__Optim__opt_brent_rec |
| 216.2ms | pkg__nautilus__Nautilus__Roots__brent_rec |
| 173.7ms | pkg__nautilus__Nautilus__Interpolation__spline_eval |
| 163.0ms | pkg__nautilus__Nautilus__ODE__rk45_dopri_step_vec_core |
| 148.4ms | pkg__nautilus__Nautilus__ODE__rk45_dopri_step |

Mean per-decl: 8.89ms. The annotation pass is uniformly more
expensive than inference (the same decls dominate both, but
annotation does more work per node — typed-meta-map insertion +
recursive walk). 1850 decls × 8.89ms = ~16.5s, matches phase total.

#### Top-5 slowest by module-prefix (annotation pass)

| Time | Decls | Module |
|---|---|---|
| 2.011s | 160 | pkg__nautilus__Nautilus__LinAlg |
| 1.953s | 114 | pkg__nautilus__Nautilus__Special |
| 1.477s | 122 | pkg__nautilus__Nautilus__Distributions |
| 1.072s | 198 | pkg__coral__Coral__Frame |
| 0.873s | 62  | pkg__nautilus__Nautilus__ODE |

Total modules: 71. Distribution roughly proportional to decl count
(~12.5ms per decl in heavy modules, ~5–10ms in lighter ones).
**Genuinely uniform within the inference/annotation passes.** The
quadratic cliff is only in the lowering pre-flight.

#### Lower-decl actual work

| Time | nodes_added | Decl |
|---|---|---|
| 0.30ms | 6 | pkg__nautilus__Nautilus__LinAlg__aat |
| 0.20ms | 6 | pkg__nautilus__Nautilus__LinAlg__gram |
| 0.20ms | 119 | pkg__chelis__std__Std__Nn__Conv__conv2d_small |
| 0.20ms | 41 | pkg__chelis__std__Std__Nn__Conv__conv1d |
| 0.10ms | 9 | pkg__chelis__std__Std__Nn__Linear__forward |

DAG is tiny — only ~400 nodes after all 1026 lowered decls are
folded in (most decls share node identity through CSE). Per-decl
lowering work is essentially free; the entire 9.9s wasted is in the
filter check.

## Step 3: classification — bounded vs structural

| Phase | Sub-phase | Bounded? | Why |
|---|---|---|---|
| `build_type_env_from_library` | `infer_phase0e_program_with_state` (2.25s) | **Structural** | One-pass HM inference over 1850 decls, ~1.2ms/decl, uniform distribution |
| `build_type_env_from_library` | `annotate_library_exprs_outer_loop` (13.8s) | **Bounded** | Annotation is duplicated by `check_phase0e_with_context`. Single annotation pass over the library suffices for both downstream uses (the annotated phase0e_types map; the checked-program annotated_exprs). Refactor the two callers to share. |
| `check_phase0e_with_context` | `infer_phase0e_program_with_state` (2.4s) | **Structural** | Same as above — one-pass HM. |
| `check_phase0e_with_context` | `annotate_phase0e_program_with_context` (16.8s) | **Bounded (jointly with build_type_env)** | One of the two annotation passes is redundant. Pick a single canonical annotation pass and have both callers consume its output. |
| `lower_program_to_library` | `assertions_loop` (20.6s) | **Bounded** | Quadratic: `top_level_expr_is_lowered(expr, program_exprs, type_env)` rebuilds `top_level_lowering_map` on every call (1850 calls × full walk). Replace the call with `top_level_expr_is_lowered_with_names(expr, type_env, &lowered_names)` reusing the precomputed map. |
| `lower_program_to_library` | `lower_top_level_loop` wrapper overhead (9.9s) | **Bounded** | Same root cause: line 137-139's `top_level_expr_name(expr).is_none() && top_level_expr_is_lowered(...)` rebuilds the map for each non-`def` decl. Pass `&lowered_names` through. |
| `lower_program_to_library` | `lower_top_level` actual work (0.009s) | n/a | Already negligible. |

### Estimated savings from bounded fixes

| Fix | Estimated saving |
|---|---|
| Lowering pre-flight quadratic (assertions_loop + filter check) | ~30s (the entire ~30.5s lowering phase drops to ~0.5s) |
| Redundant library annotation pass | ~13.8s (one of the two passes is removed) |
| **Combined** | **~44s saved** out of ~68s cold compile (~65% reduction) |

Cold `compile_reef_context` should drop from ~67–80s to **~25–35s**,
making warm-cache vs. cold less dramatic and putting cold `chelis
test`/`chelis eval` within a single-digit-multiple of the 30s
headline.

The user's explicit instruction is "stop early on Step 3 if structural;
fix if bounded." This investigation finds the work is **bounded.** Both
fixes are squarely within the ~50–150 LOC fix budget.

## Recommended fix order

1. **Lowering quadratic fix** (~30 LOC): `lower_program_to_library`
   already computes `lowered_names = top_level_lowering_map(...)`
   on line 114. Thread that map into the assertions loop and the
   filter check on lines 116 and 139, replacing
   `top_level_expr_is_lowered(...)` with
   `top_level_expr_is_lowered_with_names(...)`. The `_with_names`
   helper already exists (line 526 of `lower.rs`); just needs to be
   made `pub(crate)` or wrap-equivalent so the public callers can
   reuse it. Estimated saving: ~30s.

2. **Annotation deduplication** (~50–80 LOC): Restructure
   `compile_reef_context` so library annotation happens ONCE. The
   simplest path:
   - Run a single monolithic `check_phase0e_program(deep_library_decls)`
     to get the `library_checked: CheckedProgram` (which includes
     the annotated exprs).
   - Build the `TypeEnv` snapshot from `library_checked.annotated_exprs()`
     directly, without re-annotating.
   - Skip the `annotate_phase0e_program_with_context` pass inside
     `check_phase0e_with_context` when context is empty (the library
     case) — return the already-annotated exprs unchanged.
   This requires careful contract-preservation for downstream
   `_with_context` callers; the existing `library_def_count() == 0`
   branch in `annotate_phase0e_program_with_context` already
   distinguishes the empty-context case.

If only the lowering fix lands (the cleaner of the two), cold
compile drops by ~30s, putting Coral cold around ~38s. That's already
within the 30s headline ballpark for cold.

## Reproducing this report

```sh
# Fresh tempdir so the disk cache doesn't short-circuit
rm -rf /tmp/perf-investigate/reef_home && mkdir -p /tmp/perf-investigate/reef_home
cd /home/jeff/Documents/scratch/coral
CHELIS_REEF_HOME=/tmp/perf-investigate/reef_home \
CHELIS_PROFILE_COMPILE_CONTEXT=1 \
CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1 \
  /path/to/target/release/chelis eval --file src/apismoke.ch \
  2> /tmp/perf-investigate/full_detail.log

python3 /tmp/perf-investigate/aggregate.py /tmp/perf-investigate/full_detail.log
```

## Status

- Per-decl instrumentation landed (gated on
  `CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1`; default behavior
  unchanged, no overhead when unset).
- Diagnosis complete. Two bounded fixes identified.
- **Both fixes applied and pushed** after explicit ack:
  - Commit `5eaefdb`: lowering quadratic
    (`lower_program_to_library` 30.5 s → 0.046 s; 663× speedup).
  - Commit `5ad5a6f`: unified `build_compiled_library_context`
    (35.8 s → 16.1 s; eliminates the duplicated inference + annotation).
- Combined Coral cold `compile_reef_context`: **~68 s → ~18 s**.
- Coral cold `chelis test tests/` (63 tests): **81.6 s → 29.4 s**
  (under the 30 s headline on the cold path, no warm cache needed).
- chelis-std self-test corpus 205/205 still passes.
- Workspace gate green (3 known HIP failures only).
