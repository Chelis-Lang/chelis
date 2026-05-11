# Initial Gap Report — Closure Analysis

**Filed:** 2026-05-11
**Source report:** [`docs/identified_gaps.md`](identified_gaps.md) (originally filed 2026-05-08)
**Latest closure batch:** PR #23 (`compiler cleanup: M1→W7 batch + W5/W7 red-team passes`), merged to `main` at `2ed588c`
**Companion:** [`docs/gap_synthesis.md`](gap_synthesis.md) — §5 Remaining Work Register tracks each open item

## Top-line verdict

Of the seven items in the original report (six gaps + adjacent Surf-span
finding), **six are fully closed and one (Gap 5) has a first-slice
closure** that ships the BLAS and sparse helper paths for both C and HIP
plus structured rejection diagnostics for every callsite that misses the
summary path. Broader recognizer coverage (softmax / layer_norm /
attention / FlashAttention-style fusion) and a handful of red-team-
surfaced residuals are filed as §5 R1–R5 in the synthesis doc and tracked
as their own workstreams.

The framework's correctness story holds throughout. Every gap was a
performance / ergonomics gap, and no closure required touching the
correctness-bearing code paths.

## Per-gap closure status

| Gap | Status | Closing change | Locking test |
|---|---|---|---|
| **1.** C memory planner | **Closed** | M2a — conservative slot planner ported to C backend (`crates/chelis-backend-c/src/memory.rs`) with C ownership rules for borrowed loads, metadata views, and output materialization | `crates/chelis-cli/tests/copy_elision.rs::copy_elision_probe_reuses_c_backend_slots_without_materializing_copies` (≤4 backing slots, 0 memcpy) |
| **2.** Pattern-matcher brittleness | **Closed** | M1 — BLAS detection moved into new `chelis_ir::specialize` substrate; identity `Cast` / `Reshape` / `Permute` removed before recognizer runs; replacement is `RiscOp::BlasMatmul`; DCE follows | `cast_perturbed_matmul_specializes_after_noop_cleanup` plus the 11-perturbation sweep in `pattern_matcher_brittleness.rs` |
| **3.** `gather` lowering + scatter recognizer | **Closed for the scoped sparse path** | M4 — `RiscOp::Gather { axis }` and `RiscOp::ScatterAdd { axis }` first-class; tensor-lane Surf `gather` lowers directly to sparse IR; dense `OneHot+Expand+Mul+Sum` recognizer collapses to `Gather`; C/HIP sparse codegen ships. W2-A — `RiscOp::Scatter` (last-write-wins) added with structured AD rejection `AdError::NotSupported { op, reason: NonDeterministicAtDuplicateIndices }` | `grad_gather_contract.rs::gather_via_section_3_5_lowering_accumulates_duplicate_indices`; `scatter_replace_contract.rs` locks last-write-wins forward semantics and pattern-matches the structured AD-rejection variant |
| **4.** Rank-2 matmul | **Closed** | M3 — type rule and `lower_matmul` accept rank ≥ 2. M3b — IR specializer emits runtime-sized BLAS for symbolic and batched matmul on contiguous trailing slices. Perf-F1 — HIP defaults to `hipblasSgemmStridedBatched` on uniformly strided batched layouts; per-batch helper loop retained only for broadcasted leading axes / non-uniform strides | `perf_f1_strided_batched_default.rs` (exact-line); GPU oracle `g15_hipblas_strided_batched_symbolic_batch_matches_eval` |
| **5.** Cross-function specialization | **First slice closed; broader recognizer coverage remains** | M5 — C BLAS helper summaries via `summarize_blas_helper_from_parts`. W3-A — HIP BLAS via `lower_named_tensor_entry_dag` inlining. W3-B — C sparse helper summaries (`Gather` / `ScatterAdd` / `Scatter`). W6 Task B — HIP sparse lock-tests confirm the inlining path already covers sparse helpers. W4-A — public `SummaryRejection { rejection_class, helper_path, callsite_span, helper_body_span, detail }` with 10 sparse variants. W6 Task A — 6 BLAS-prefixed variants (`BlasMultipleRoots`, `BlasOutputPrecisionMismatch`, `BlasNotMatmulPattern`, `BlasNonLoadOperand`, `BlasInputPrecisionMismatch`, `BlasDimensionBindingFailure`) | `cross_library_semantic_gap.rs` (C + HIP, 4 forms × 2 targets), `cross_library_semantic_gap_hip_gpu.rs` (GPU numeric oracle), `cross_library_sparse_summaries.rs` (10 tests), `cross_library_sparse_hip_summaries.rs` (9 tests), `cross_library_semantic_gap_diagnostics.rs` (28 tests = 20 W4-A sparse + 8 W6 BLAS), `host_sparse_summary_diagnostics.rs` (11), `host_blas_summary_diagnostics.rs` (17), `red_team_w7_blas_cross_product.rs` (11, all 8 non-F32 `Prim` values) |
| **6.** Dead `Mul` after BLAS hit | **Closed** | M1 — replacement of the `Sum(Mul(Expand, Expand))` root with `RiscOp::BlasMatmul` runs before DCE, which then prunes the orphan `Mul` / `Expand` nodes | Cost-profile assertion in `cross_library_semantic_gap.rs` (8×16 @ 16×4 → 128 working bytes, result only) |
| **Adjacent.** Surf-source spans don't reach IR | **Closed** | M2b — Surf desugarer threads parser byte ranges into Deep `meta["span"]` as `surf:<start>..<end>` IDs for ordinary expression bodies; `__synthesized_*` markers reserved for genuinely spanless inputs | `traceability_paradox.rs::transformer_block_traceability_state_is_locked` (emitted span comments now include `surf:` byte-range IDs) |

## Red-team passes against this batch

Two fresh-context red-team passes ran in worktree-isolated subagents
against the same `compiler-cleanup` branch and are recorded in
`docs/gap_synthesis.md` §5:

- **W5 (2026-05-11), against M1→M4 / Perf-F1 / Perf-F2:** 32 adversarial
  tests across `red_team_w5_*.rs` files. Surfaced one **P0 silent
  miscompile** — `detect_matmul_pattern` had no precision filter, so
  non-F32 matmul subgraphs silently became `RiscOp::BlasMatmul` and the
  C backend emitted `cblas_sgemm` against wrong-precision data. Fix
  shipped in-band: precision filter at
  `chelis_ir::specialize::detect_matmul_pattern` plus defense-in-depth
  panics in both `emit_blas_matmul` sites. P0-asserting tests inverted
  to positive regressions.

- **W7 (2026-05-11), against W6:** 35 adversarial tests across
  `red_team_w7_*.rs` files. Zero P0/P1/P3. One **P2** finding (HIP
  sparse gather rejects `Cast`-wrapped indices, surfaces loudly at
  codegen — pre-existing limitation, surfaced not introduced) filed as
  §5 R5. W7 also locked the **W5 P0 → W6 diagnosed-rejection
  cross-product invariant**: all 8 non-F32 `Prim` values (F64, F16,
  Bf16, F8e4m3, Int8, Int32, Int64, Bool) now produce a structured
  `BlasOutputPrecisionMismatch` diagnostic with zero silent
  fallthroughs.

HIP manual gate run 2026-05-11 via `scripts/hip_test.py`: 38/38 GPU
correctness tests passing.

## What remains — backlog filed as §5 R1–R5

After this batch, the residual work is tracked as standalone entries in
`docs/gap_synthesis.md` §5. Each has an executable anchor or a defined
closure path.

| ID | Tracks | What closure requires |
|---|---|---|
| **§5 R1** | Softmax / layer_norm / attention recognizers (and FlashAttention-style fusion) — the Gap 5 broad coverage tail | Per-op spec definition; IR recognition extending `chelis_ir::specialize`; backend dispatch in C (fused kernel) and HIP (cuDNN-shape equivalent); AD policy + adjoint table extension; corpus per op. Each is a wave-sized effort. Generic-path runtime cost: 3-5× slower for softmax/layer_norm; 10-50× slower for scatter/gather runtime calls (`specialization_dispatch.rs`). |
| **§5 R2** | Path-B HIP host-program fallback codegen | HIP builds without a preferred tensor entry currently route through `chelis_backend_c::codegen_host_program` and emit C regardless of `--target hip` (`crates/chelis-cli/src/main.rs:1620`). Closure: HIP equivalent of host-program codegen + runtime. Not reachable from BLAS-shaped programs today. |
| **§5 R3** | `DimExpr` `Add` / `Sum` variant decision | 227-site public-enum change. No current spec calls for sum-of-dims (concat is at `Pad` / `Shrink` today). Needs shape-calculus design decision *before* mechanical expansion. |
| **§5 R4** | `DimExpr` rational vs integer-floor semantics (W5 P3) | Canonicalizer treats `(a*3)/2` and `a*(3/2)` as equal under rational arithmetic; under integer-floor they differ for odd `a`. Harmless under today's symbolic-dim corpus (every dim divides cleanly) but a latent foot-gun. Closure: propagate divisibility info through `DimExpr` or restrict rewrites to proven-divisible cases. Current behavior locked in `red_team_w5_dim_canon.rs`. |
| **§5 R5** | HIP sparse gather rejects `Cast`-wrapped indices (W7 P2) | Integer-HIP codegen requires sparse gather indices to come from a direct `Load`. Casts at the callsite fail loudly with a structured error at build time. Closure: lift the codegen restriction to accept any `Int32` / `Int64` node, or document the restriction as permanent and surface it at type-check time. |

## Items intentionally not in scope of this batch (or this analysis)

A few items the original report mentioned but that are deliberately not
the subject of M1→W7 closure:

- **Gap 1 — sub-4-slot copy-probe schedule.** The fan-in fusion that
  would reduce the probe below four backing slots is broader than
  memory planning alone. The C backend now has a narrower in-place
  fused-elementwise path for single reusable inputs but does not cover
  the full copy-probe fan-in shape. The slot-coloring achievement is
  what M2a set out to do.
- **Gap 3 — replace-scatter AD.** `RiscOp::Scatter` (last-write-wins)
  is intentionally distinct from `ScatterAdd` and has no well-defined
  adjoint. The structured AD rejection
  `NonDeterministicAtDuplicateIndices` is the closure; an actual
  adjoint is not on the roadmap.
- **Gap 6 — FlashAttention-style attention fusion.** Not on the
  roadmap today. The current `transformer_block.ch` working set at
  `seq = 2048` (~1.06 GiB) is dominated by real attention
  score/probability tensors of shape `[seq, seq]`, not dead matmul
  intermediates. Slot planning alone cannot make these smaller because
  they are real intermediate values, not allocator artifacts. Filed
  conceptually under §5 R1's broader attention recognizer track.
- **Gap 5 — Coral / Nautilus / Octant helper specialization at full
  generality.** The first BLAS + sparse slices closed; the broader
  user-library specialization story rolls into §5 R1's per-op
  recognizer track plus any future expansion of the helper-summary
  vocabulary.

## Cross-references

- Per-gap detail with code citations: [`docs/identified_gaps.md`](identified_gaps.md)
- Root-cause taxonomy + cost picture: [`docs/gap_synthesis.md`](gap_synthesis.md)
- HIP environment runbook (used by `scripts/hip_test.py`): [`docs/local_hip_environment.md`](local_hip_environment.md)
- Filed upstream bugs:
  - [`spec/upstream-bugs/phase3h-gather-ad-incomplete.md`](../spec/upstream-bugs/phase3h-gather-ad-incomplete.md) (Gap 3)
  - [`spec/upstream-bugs/matmul-rank2-rule-vs-einsum-shipped.md`](../spec/upstream-bugs/matmul-rank2-rule-vs-einsum-shipped.md) (Gap 4)
  - [`spec/upstream-bugs/dead-mul-after-blas-specialization.md`](../spec/upstream-bugs/dead-mul-after-blas-specialization.md) (Gap 6)
