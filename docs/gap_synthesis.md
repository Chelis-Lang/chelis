# Chelis Compiler Gap Synthesis

**Filed:** 2026-05-08
**Owning phase:** cross-phase (perf + ergonomics)
**Companion:** [`docs/identified_gaps.md`](identified_gaps.md) — per-gap
detail; this document is the synthesis across all six.

## What this is

A review across the six compiler gaps + adjacent finding documented in
`docs/identified_gaps.md`, asking the questions that the per-gap doc
doesn't answer:

- What kinds of issues are these?
- How fundamental are they?
- How feasible is closing all manifestations?
- How much memory and compute does the current state cost?
- What architectural changes would fix that?

Every cost number below is empirically measured by a regression test
checked into this branch — not analytical projection unless explicitly
labelled "linear projection."

## 1. Taxonomy: the six gaps cluster into four root causes

The gaps are not six independent issues. They cluster:

| Pattern | Gaps | Root cause |
|---|---|---|
| **A. Pipeline ordering** | 2, 6 | BLAS detection runs at codegen, AFTER DCE / algebraic-simp could have helped. Same architectural fix closes both. |
| **B. Recognizer coverage** | 3, 4 (partial) | Each specialised kernel needs its own detector. New primitives ship without their recognizer. |
| **C. Cross-function specialization boundary** | 5 | User-defined helper calls do not yet carry compiler-verified specialization summaries across function boundaries. |
| **D. Backend parity** | 1 | HIP shipped Phase 1c memory planner; C didn't. Pure replication work. |
| **E. Pipeline plumbing** | adjacent, closed by M2b | Surf desugaring now attaches parser byte-range `meta["span"]` IDs to ordinary Deep expression nodes; downstream propagation remains the consumer. |

Pattern A is the highest-leverage fix — one architectural change closes
two gaps and sets up the substrate for closing more. Pattern C is the
most architecturally-entrenched, but M5 now tracks it as a concrete
follow-up workstream rather than a permanent design exclusion. The rest
are bounded coverage / replication / plumbing work.

## 2. Difficulty × fundamentality

"Fundamental" here means "closing it requires adding cross-function
compiler summaries rather than only improving a local pass." Only Gap 5
qualifies; the rest are unfinished implementation work.

| Gap | Effort | Fundamental? | Notes |
|---|---|:---:|---|
| **1** — C memory planner | closed by M2a | No | C codegen now uses conservative backing-slot planning with C ownership rules. |
| **2** — Pattern matcher brittleness | closed by M1 | No | BLAS detection moved into `chelis_ir::specialize`; identity casts/reshapes/permutes are cleaned before replacement. |
| **3** — Gather lowering | ~1 month | Medium | Two coupled changes: §3.5 RISC lowering + scatter-recognition pattern matcher. Must ship together or arm the OOM trap. |
| **4** — Rank-2 matmul | closed by M3/M3b | Medium-high | Type rule + `lower_matmul` accept rank ≥ 2, and the IR specializer emits runtime-sized BLAS for symbolic and batched matmul where matrix slices are contiguous. |
| **5** — Cross-function specialization | separate workstream | **Yes** | M5 documents path (b): verified BLAS-equivalent helper summaries plus callsite emission. clang LTO is documented as a workaround, not the codegen story. |
| **6** — Dead Mul after BLAS hit | closed by M1 | No | `RiscOp::BlasMatmul` replacement plus DCE removes the orphan `Mul`/`Expand` subgraph. |
| Adjacent — Surf spans | closed by M2b | No | Surf parser token positions are now plumbed into Deep node metadata as `surf:<start>..<end>` IDs. |

## 3. Concrete cost picture (measured)

Numbers below come from regression tests on this branch. Each is
reproducible: `cargo test -p <crate> --test <name> -- --nocapture`
prints the cost-profile section.

### `copy_elision_probe.ch` — 5×copy(x) fanout

(`crates/chelis-cli/tests/copy_elision.rs`)

| Input size | Peak working set | What's allocated |
|---|---|---|
| 1024×1024 f32 (4 MiB) | **16 MiB helper-side** | 4 × 4 MiB backing slots; final output wrapper reuses the dead `neg(x)` intermediate slot |
| 2 GiB (linear projection) | **8 GiB peak helper-side**, ~10 GiB total RAM | 4 × 2 GiB backing slots + the 2 GiB caller input |

M2a closes the allocate-per-node gap by coloring non-overlapping
intervals in C codegen. The remaining gap between four slots and the
theoretical two/three-slot fan-in schedule is out of scope for memory
planning alone: it requires fan-in or in-place fusion while preserving
the current `restrict`-based C kernels.

### `transformer_block.ch` — 4-head MHA + FFN

(`crates/chelis-cli/tests/traceability_paradox.rs`)

The working set is a polynomial in `seq`:

```
const ............          0 bytes
linear * seq .....       16,140 bytes/seq
quadratic * seq² .          264 bytes/seq²
```

| seq | peak | dominant term |
|---|---|---|
| 128 | 6.1 MiB | quadratic attention buffers |
| 512 | 73.9 MiB | quadratic attention buffers |
| **2048** | **1.06 GiB** | quadratic attention buffers |
| 4096 | 4.19 GiB | quadratic attention buffers |

At seq=2048, M3b reduces the previous ~6.3 GiB projection to ~1.06 GiB
by making the symbolic and batched matmuls hit BLAS. The emitted C now
contains seven `cblas_sgemm` call sites and no dense generic matmul
product allocations such as `[seq, 256, 1024]` or `[seq, seq, 64]`.
The remaining dominant term is real vanilla-attention state:
score/probability tensors of shape `[seq, seq]` across the four heads.

If FlashAttention-style attention fusion ships (not on the roadmap
today), the quadratic score/probability materialization can collapse
further. Slot planning alone cannot make those tensors smaller because
they are real intermediate values, not allocator artifacts.

### Cross-function specialization (Gap 5)

(`crates/chelis-cli/tests/cross_library_semantic_gap.rs`)

Same logical 8×16 @ 16×4 matmul, four code paths:

| Form | BLAS hits? | Working bytes | Throughput estimate |
|---|:---:|---|---|
| `f(a, b) = matmul(a, b)` | ✅ | 128 | ~100% (sgemm) |
| `f(a, b) = { ae=expand(a,...); be=expand(b,...); sum(mul(ae,be), 1) }` | ✅ | 128 | same |
| `def my_mm = matmul; def f = my_mm` | ✅ | 128 | same C BLAS path for simple wrappers |
| `def my_mm = expand+mul+sum; def f = my_mm` | ✅ | 128 | same C BLAS path for simple wrappers |

M1 makes inline BLAS-hit matmul result-only in memory. The current branch also
closes the first Gap 5 executable slice for simple C user-`def` wrappers:
helper summaries and helper-body specialization recover the BLAS path without
clang LTO. W3-A then closed the HIP arm of the same four-form table on
`--target hip` (the helper subgraph is inlined into the HIP entry DAG before
the Tier 2 BLAS specializer runs, so the user-`def` and nested user-`def`
forms also emit `chelis_hipblas_sgemm_row_major`); the GPU numeric oracle
`g15_user_def_matmul_helper_hits_hipblas_numeric` proves agreement against a
hand-rolled row-major sgemm reference. The remaining Gap 5 work is broader
summary coverage, negative diagnostics, and non-BLAS recognizer summaries.

### Specialization dispatch reality

(`crates/chelis-cli/tests/specialization_dispatch.rs`)

Five common ML operations compiled to C:

| Op | Specialised? | Generic-path cost vs cuBLAS / cuDNN equivalent |
|---|:---:|---|
| rank-2/rank-N symbolic matmul with contiguous matrix slices | ✅ cblas_sgemm / hipBLAS helper | matches BLAS dispatch; C loops per batch slice for batched calls |
| softmax | ❌ | ~3-5× slower (no online-softmax, no SRAM tiling) |
| layer_norm | ❌ | ~3-5× slower (no fused mean+var pass) |
| scatter | ❌ runtime-call | ~10-50× slower (no parallel-radix-sort, no warp-aware) |
| gather | ❌ runtime-call | ~10-50× slower |

**Compile-time cost is fine** — every issue listed is a runtime cost
in the *emitted* code, not in the compiler itself.

## 4. Sequenced fix plan

A staged plan that closes the local compiler gaps while splitting Gap 5
into a dedicated cross-function specialization workstream.

**Phase α (closed by M1):** Architectural change A moved BLAS / pattern
detection out of codegen and into `chelis_ir::specialize`. It closes
Gaps 2 + 6 for rank-2 concrete matmul and sets up the substrate for
future recognizers (softmax, layer_norm, attention, gather→scatter).

**Phase β (closed by M2a/M2b):** The C backend now has conservative
slot-based memory planning, and desugared Deep now receives
`surf:<start>..<end>` metadata for parsed Surf expression bodies.

**Phase γ (closed by M3/M3b + Perf-F1):** `matmul` now accepts rank ≥ 2,
and runtime-sized BLAS specialization covers symbolic and batched matmul
when the operands have contiguous trailing matrix slices. The C backend
emits one `cblas_sgemm` per batch slice; the HIP backend defaults to
`hipblasSgemmStridedBatched` on uniformly strided batched layouts
(Perf-F1, shipped) and retains the per-batch hipBLAS helper loop only
for broadcasted leading axes or non-uniform leading strides. Symbolic
dimensions are read from the existing runtime shape bindings.

**Phase δ (~1 month, paired):** Ship the §3.5 gather lowering paired
with a scatter-recognition pattern in Phase α's new specialize pass
(Gap 3 closure). The recognizer goes into the substrate Phase α
already set up.

**Phase ε (continuing, opportunistic):** Add specialised recognizers
for softmax, layer_norm, and attention (FlashAttention-style fusion
is the largest of these). Each one is bounded; they accumulate in the
specialize pass as Tier-2-shape-recognition rules.

**Gap 5 — M5 workstream doc landed.** The active specs now frame
user-`def` specialization loss as a known limitation intended to close
through verified BLAS-equivalent helper summaries and callsite emission
rules, anchored by `cross_library_semantic_gap.rs`.

## 5. Remaining Work Register

These items are not optional cleanup. They are the explicit backlog left after
the M1/M2b/M3/M4/M5 batch, plus closed items retained here as completion
anchors for future audits.

| ID | Tracks | Required closure | Current executable anchor |
|---|---|---|---|
| **M4** | Gap 3, gather/scatter lowering | Closed for the scoped sparse path and the replace-scatter follow-up: tensor-lane Surf lowers directly to sparse IR, the internal dense §3.5 `OneHot + Expand + Mul + Sum` tree collapses to `Gather`, C/HIP emit bounded sparse code for supported dtypes, and W2-A added `RiscOp::Scatter` with last-write-wins semantics distinct from `ScatterAdd`. AD over `Scatter` raises the pinned structured error `AdError::NotSupported { op: "scatter_replace", reason: NonDeterministicAtDuplicateIndices }` (introduced as a public `chelis_ir::grad::AdError` / `AdRejectionReason` surface; existing free-text rejections for Argmax/Argmin/Floor/Ceil were migrated to structured variants too). | `crates/chelis-ir` specialization tests cover the dense recognizer and unmatched `OneHot` fallback; `grad_gather_contract.rs` locks duplicate-index AD; `crates/chelis-ir/tests/scatter_replace_contract.rs` locks last-write-wins forward semantics, the pinned AD-rejection variant (pattern-matched, not `contains()`), and unchanged `ScatterAdd` AD behavior; C/HIP emitted-code and GPU manual tests cover bounded sparse backend behavior. |
| **M5-follow-up** | Gap 5, cross-function specialization | **CLOSED by W3-A + W3-B + W4-A + W6.** All four originally-open sub-items shipped: (a) HIP consumption for BLAS summary kinds — closed by W3-A via `lower_named_tensor_entry_dag` inlining. (b) HIP consumption for sparse summary kinds — closed by W6 Task B via the same inlining path (lock-tests passed on first run; no HIP codegen changes needed). (c) Sparse-helper rejection diagnostics — closed by W4-A with the public `SummaryRejection { rejection_class, helper_path, callsite_span, helper_body_span, detail }` surface, 10 sparse variants. (d) BLAS-helper rejection diagnostics — closed by W6 Task A with 6 BLAS-prefixed variants (`BlasMultipleRoots`, `BlasOutputPrecisionMismatch`, `BlasNotMatmulPattern`, `BlasNonLoadOperand`, `BlasInputPrecisionMismatch`, `BlasDimensionBindingFailure`) and a parallel `BlasSummaryAttempt::{NotEligible, Rejected}` enum mirroring the sparse path. The W5 P0 fix's previously-silent precision rejection now surfaces as a structured `BlasOutputPrecisionMismatch` / `BlasInputPrecisionMismatch` diagnostic for every non-F32 precision, verified by W7 across all 8 non-F32 `Prim` values. | **BLAS helpers closed for C and HIP:** `crates/chelis-cli/tests/cross_library_semantic_gap.rs` (8 tests) on `--target c` + `--target hip`; `crates/chelis-cli/tests/cross_library_semantic_gap_hip_gpu.rs` (GPU manual gate). **Sparse helpers closed for C and HIP:** `crates/chelis-cli/tests/cross_library_sparse_summaries.rs` (10 tests, C path) + `crates/chelis-ir/tests/host_sparse_summary.rs` (9 tests) + `crates/chelis-cli/tests/cross_library_sparse_hip_summaries.rs` (9 tests, HIP wrapper lock-tests). **Sparse rejection diagnostics closed by W4-A:** `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs` (28 tests = 20 original W4-A + 8 W6 BLAS extension) + `crates/chelis-ir/tests/host_sparse_summary_diagnostics.rs` (11 tests). **BLAS rejection diagnostics closed by W6 Task A:** `crates/chelis-ir/tests/host_blas_summary_diagnostics.rs` (17 tests) pattern-matches all 6 BLAS variants. **W5→W6 cross-product invariant locked by W7:** `crates/chelis-ir/tests/red_team_w7_blas_cross_product.rs` (11 tests) enumerates all 8 non-F32 precisions and asserts each produces a structured `BlasOutputPrecisionMismatch` rejection with the observed precision — zero silent fallthroughs. |
| **Perf-F1** | HIP batched matmul implementation quality | **Closed.** `hipblasSgemmStridedBatched` is now the default on uniformly strided batched HIP layouts; the per-batch helper loop is retained only as a fallback for broadcasted leading axes (`Expand` on the batch dim → stride-0) or otherwise non-uniform leading strides. | `crates/chelis-backend-hip/tests/perf_f1_strided_batched_default.rs` locks the strided-batched default with exact-line matching on uniform layouts and exact-line fallback matching on broadcasted leading axes (default workspace pass). The HIP manual GPU gate `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` (with `HSA_OVERRIDE_GFX_VERSION=11.5.1` and `LD_LIBRARY_PATH` per `docs/local_hip_environment.md`) covers numerical agreement, including `g15_hipblas_strided_batched_symbolic_batch_matches_eval` and `g15_hipblas_batched_matmul_matches_eval`. |
| **Perf-F2** | Post-BLAS allocator/fusion compounding | Normalize equivalent symbolic shape expressions for slot reuse and broaden in-place elementwise/fan-in fusion where aliasing permits. | **F2(a) DimExpr normalization v2 closed by W1-C:** in addition to v1 product/identity rules, `normalized_key()` now does Div numerator/denominator atom cancellation, GCD reduction on concrete factors, nested-Div flattening, and Mul × Div cross-term distribution. Positive and negative tests in `crates/chelis-ir/tests/dim_canonicalization.rs`. **F2(c) C scoped same-property `forall` / binder-equivalent aliases closed by W1-B:** C fused-elementwise in-place emit now admits same-binder, literal-equal, and named-equal-to-lit aliases via `binder_equivalent_tensor_type`. Positive and negative tests in `crates/chelis-backend-c/tests/fused_in_place_forall_alias.rs`. **F2(b) HIP in-place fused-elementwise closed by W2-B:** HIP backend now ports the C-side alias proof via a new `chelis-backend-hip::fusion` module mirroring `binder_equivalent_tensor_type` and `fused_in_place_spec`. At runtime, the wrapper guards on `chelis_gpu_is_contiguous` and aliases the FusedElem output view onto the reusable input's device buffer; the kernel parameter list drops `__restrict__` on the aliased external + output. Structural tests in `crates/chelis-backend-hip/tests/fused_in_place_forall_alias.rs` cover 11 cases (5 positive + 6 negative). GPU manual gate `gf3_fused_in_place_fan_in_gpu_matches_cpu` locks end-to-end numeric agreement on the local ROCm/HIP workstation. DimExpr canonicalization explicitly does *not* alpha-rename symbolic dims — a future scoped path must take same-binder aliases as explicit input. The `DimExpr` enum vocabulary stays `Concrete / Sym / Mul / Div`; a sum/add variant remains an open question parked here (see footnote below). |

Fresh-context red-team status for M3/M3b: run 2026-05-10. The red-team
pass found one medium issue: the legacy C codegen-time BLAS detector could
still specialize non-contiguous rank-2 matrix slices by materializing
contiguous copies, bypassing the IR specializer's negative rule. The legacy
C detector path has been removed from reduction emission; BLAS now enters C
codegen through `RiscOp::BlasMatmul` produced by the IR specialization pass.

Fresh-context red-team status for the W1–W4 batch: run 2026-05-11 as **W5**
in a worktree-isolated subagent. The pass added 32 adversarial tests across
seven new files (`crates/chelis-{backend-c,backend-hip,cli,ir}/tests/red_team_w5_*.rs`)
and surfaced one **P0 silent miscompile**: the IR specializer's
`detect_matmul_pattern` had no precision filter, so an F64/Int32/Int64 matmul
subgraph silently became `RiscOp::BlasMatmul` and the C backend emitted
`cblas_sgemm` — single-precision BLAS — against the wrong-precision data.
The fix shipped in-band: precision filter at the canonical specializer site
plus defense-in-depth panics in both backend `emit_blas_matmul` sites. The
W5 P0-asserting tests were inverted to positive regressions and joined by
F32-still-hits-BLAS positive tests on both legs. The other red-team findings
(P2: BLAS-summary recognizer is silent on near-eligible rejections; P3:
DimExpr canonicalizer uses rational-not-floor semantics) are tracked: P2 in
the sibling-sweep follow-up named under the M5-follow-up entry above; P3 is
harmless under today's symbolic-dim corpus and parked.

HIP manual gate status for this batch: run 2026-05-11 on the local ROCm/HIP
workstation via `scripts/hip_test.py` (which sets the full hipBLAS env per
`docs/local_hip_environment.md`); 37/37 GPU correctness tests passed plus the
W5-added g15/g4 batched and symbolic batched cases. The earlier 2026-05-10 run
on the same suite also passed including `g16_sparse_*` for HIP gather and
duplicate-index scatter-add. The post-W6 GPU oracle re-run reports 38 tests
passing (one above the documented 37+ floor) under the same env.

Fresh-context red-team status for W6: run 2026-05-11 as **W7** in a
worktree-isolated subagent. The pass added 35 adversarial tests across four
new files (`crates/chelis-{ir,cli}/tests/red_team_w7_*.rs`). Zero P0, zero
P1, zero P3 findings. One **P2** finding: HIP sparse gather rejects indices
produced by a `Cast` IR node (e.g., from a Surf callsite that casts the
indices), failing loudly at codegen with a structured error message
("`chelis build --target hip sparse gather requires indices to be loaded
input tensors in this milestone; node N uses indices produced by Cast`")
rather than silently miscompiling. Pre-existing integer-HIP codegen
limitation, surfaced loudly, not introduced by W6. Filed as §5 R5 below.

The W7 pass also locked the **W5 P0 → W6 diagnosed-rejection cross-product
invariant**: all 8 non-F32 `Prim` values (F64, F16, Bf16, F8e4m3, Int8,
Int32, Int64, Bool) now produce a structured `BlasOutputPrecisionMismatch`
diagnostic when used in a user-`def` matmul helper, and the IR specializer's
precision filter from the W5 P0 fix remains locked (no `RiscOp::BlasMatmul`
replacement for any non-F32 case). Test: `red_team_w7_blas_cross_product.rs::
nonsilent_rejection_invariant_for_every_w5_rejected_precision`.

## Standalone follow-up entries (filed post-W7)

These were previously narrative tail-references inside the M5-follow-up
entry or held in caveats. After W6 + W7 closed the cross-product, they are
each their own named workstream:

### §5 R1 — Softmax / layer_norm / attention recognizers

| Tracks | Required closure | Current executable anchor |
|---|---|---|
| Recognizer coverage tail (Pattern B in §1) | Spec definition per op shape; IR recognition (likely extending `chelis_ir::specialize`); backend dispatch in C (fused kernel) and HIP (cuDNN-shape equivalent); AD policy and adjoint table extension; corpus per op. Each op is a wave-sized effort. | None today. `crates/chelis-cli/tests/specialization_dispatch.rs` lines 134-141 document the generic-path cost (3-5× slower for softmax/layer_norm). |

### §5 R2 — Path-B HIP host-program fallback codegen

| Tracks | Required closure | Current executable anchor |
|---|---|---|
| HIP target coverage for programs without preferred tensor entry | `crates/chelis-cli/src/main.rs:1620` routes HIP builds with no preferred tensor entry to `chelis_backend_c::codegen_host_program`, which emits C regardless of `--target hip`. Closing requires HIP equivalent of host-program codegen plus runtime support. | W3-A's report flagged this. Not reachable from BLAS-shaped programs today (any `def my_mm(a, b) = ...` with all-tensor parameters and tensor return is picked as the preferred tensor entry). |

### §5 R3 — DimExpr `Add`/`Sum` variant decision

| Tracks | Required closure | Current executable anchor |
|---|---|---|
| Whether IR vocabulary should grow `Add` for concat-shape inference or striped tile bookkeeping | 227-site public-enum change across IR / backends / consumers (W1-C's escalation report). Shape-calculus design decision BEFORE the mechanical change. No current spec calls for sum-of-dims; concat is at `Pad` / `Shrink` today. | §6 caveat 3 above already names this as parked. |

### §5 R4 — DimExpr rational vs integer-floor semantics (W5 P3)

| Tracks | Required closure | Current executable anchor |
|---|---|---|
| Canonicalizer treats `(a*3)/2` and `a*(3/2)` as equivalent under rational arithmetic; under integer-floor they differ for odd `a` | Propagate divisibility info through DimExpr to make canonicalization floor-aware, or restrict rewrite rules to proven-divisible cases. Harmless under today's symbolic-dim corpus (every dim divides cleanly) but a latent foot-gun if a future corpus violates that. | `crates/chelis-ir/tests/red_team_w5_dim_canon.rs::dim_canonicalizer_uses_rational_not_integer_floor_semantics` locks current behavior. |

### §5 R5 — HIP sparse gather rejects `Cast`-wrapped indices (W7 P2)

| Tracks | Required closure | Current executable anchor |
|---|---|---|
| Integer-HIP codegen path requires sparse gather indices to come from a direct `Load` IR node. When the indices flow through a `Cast` (e.g., user converts indices' dtype at the callsite), HIP codegen rejects with a structured error at build time. | Either lift the codegen restriction to accept any IR node whose precision is `Int32`/`Int64` (preferred — small HIP backend change), or document the restriction as a permanent constraint in the spec and surface it at type-check time so users don't hit it at the codegen layer. | `crates/chelis-cli/tests/red_team_w7_hip_sparse_wrapper.rs::hip_gather_wrapper_with_cast_callsite_indices_loud_failure_or_kernel` locks the current loud-failure behavior. Pre-existing limitation, not introduced by W6 or W7 — surfaced by the W7 adversarial pass. |

## 6. Verdict on structural feasibility

**Yes, structurally feasible.** The dominant cost (Gap 6 dead-Mul) is
20× of the transformer working-memory bloat and closes with a
one-month architectural change that also closes Gap 2 and lays the
substrate for everything in Pattern B. The remaining gaps are bounded
coverage / replication / plumbing work.

### Caveats worth pinning

1. **Gap 5 is a separate architecture workstream.** Coral / Nautilus /
   Octant helpers behind user-`def` boundaries still miss backend
   specialization today. The planned closure is not blind inlining or
   native-compiler LTO; it is a verified helper-summary mechanism that
   lets backend specialization treat selected user functions as
   compiler-visible abstractions.

2. **Gap 4 is closed, including the HIP strided-batched backend-quality
   follow-on.** M3 picked the PyTorch-ergonomic answer by lifting
   `matmul` to rank ≥ 2, and M3b lets symbolic/batched matmul specialize
   through runtime BLAS sizes. Perf-F1 then made
   `hipblasSgemmStridedBatched` the HIP default on uniformly strided
   batched layouts, with the per-batch helper loop retained only as the
   fallback for broadcasted leading axes or non-uniform leading strides.

3. **`DimExpr` sum/add support is parked.** The W1-C plan named "sum
   normalization (`a + b == b + a`)" and "broader identity folds
   (`0 + x == x`)" as candidate canonicalizations. The current
   `DimExpr` vocabulary is `Concrete / Sym / Mul / Div` — there is no
   `Add` or `Sub` variant. Adding one is a public-enum expansion that
   forces a match-arm addition at every existing site (227
   construction sites, plus match exhaustiveness across `eval`,
   `bind`, `as_concrete`, `symbolic_names`, `From<&DimInfo>`,
   `Display`, plus all backend `memory.rs` / `blas.rs` / `emit.rs`
   consumers). Per the agent contract that's an escalation trigger.
   No current spec calls for sum-of-dims in tensor shape arithmetic
   (axis sizes compose multiplicatively under reshape and product
   under flattening; the only sum-shaped dim that would matter is
   concatenation along an axis, which today is expressed at the
   `Pad` / `Shrink` op level rather than in `DimExpr`). If a future
   shape calculus needs `Add` — e.g. for explicit concat-shape
   inference, or for striped tile bookkeeping — that's a deliberate
   IR vocabulary change, scoped as its own work item rather than a
   silent expansion under F2(a).

4. **The dispatch-coverage tail is unbounded.** Softmax, layer_norm,
   attention, batched-attention, MoE routing, etc. each need their
   own recognizer. Phase α makes adding them cheap, but there's
   always one more shape that hasn't been recognised yet. The
   alternative architecture (treating high-level ops as first-class
   IR nodes rather than recognising them after Tier 2 decomposition)
   would close the tail by construction at the cost of a deeper IR
   refactor.

### What does NOT need to change

The framework's correctness story holds throughout. All six gaps are
performance gaps, not correctness gaps:

- Linearity prevents micro-fan-out at the variable level
  (`crates/chelis-types/src/linearity.rs`).
- AD multi-consumer accumulation is correct
  (`crates/chelis-ir/src/grad.rs:188-196`, with sum-merge of adjoint
  contributions).
- The §3.5 RISC composition would correctly accumulate
  duplicate-index gradients whenever the lowering ships — empirically
  verified in `crates/chelis-ir/tests/grad_gather_contract.rs`.

Closing the perf gaps does not require touching any of the
correctness-bearing code. The framework's headline claim ("AD is
correct by construction through linearity + the RISC adjoint table")
remains intact.

## 7. Cross-references

- Per-gap detail: [`docs/identified_gaps.md`](identified_gaps.md)
- Filed upstream bugs:
  - [`spec/upstream-bugs/phase3h-gather-ad-incomplete.md`](../spec/upstream-bugs/phase3h-gather-ad-incomplete.md) (Gap 3)
  - [`spec/upstream-bugs/matmul-rank2-rule-vs-einsum-shipped.md`](../spec/upstream-bugs/matmul-rank2-rule-vs-einsum-shipped.md) (Gap 4)
  - [`spec/upstream-bugs/dead-mul-after-blas-specialization.md`](../spec/upstream-bugs/dead-mul-after-blas-specialization.md) (Gap 6)
- Locked regression tests:
  - `crates/chelis-cli/tests/copy_elision.rs` (Gap 1)
  - `crates/chelis-backend-c/tests/pattern_matcher_brittleness.rs` (Gap 2)
  - `crates/chelis-ir/tests/grad_gather_contract.rs` (Gap 3 forward-looking AD contract)
  - `crates/chelis-cli/tests/cross_library_semantic_gap.rs` (Gap 5)
  - `crates/chelis-cli/tests/specialization_dispatch.rs` (cross-cutting dispatch reality)
  - `crates/chelis-cli/tests/traceability_paradox.rs` (adjacent span finding)
- Probe corpus: `examples/illustrative/`
