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
clang LTO. The remaining Gap 5 work is broader summary coverage, negative
diagnostics, HIP summary consumption, and non-BLAS recognizer summaries.

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

**Phase γ (closed by M3/M3b):** `matmul` now accepts rank ≥ 2, and
runtime-sized BLAS specialization covers symbolic and batched matmul
when the operands have contiguous trailing matrix slices. The C backend
emits one `cblas_sgemm` per batch slice; the HIP backend emits through a
batched hipBLAS helper loop. Symbolic dimensions are read from the
existing runtime shape bindings.

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

These items are not optional cleanup. They are the explicit backlog left
after the M1/M2b/M3/M5 documentation batch.

| ID | Tracks | Required closure | Current executable anchor |
|---|---|---|---|
| **M4** | Gap 3, gather/scatter lowering | Ship the dense §3.5 gather/scatter recognizer and HIP sparse backend so embedding/MoE-shaped programs do not allocate dense `[N, V, D]` intermediates on any backend. | `crates/chelis-ir/tests/grad_gather_contract.rs` locks duplicate-index AD; tensor-lane Surf `gather` now lowers to first-class sparse IR and C codegen has bounded emitted-code coverage, but the dense §3.5 recognizer and HIP sparse kernels remain open. |
| **M5-follow-up** | Gap 5, cross-function specialization | Broaden verified summaries beyond simple C BLAS helpers, add rejected-callsite diagnostics, and carry summary consumption into HIP. | `crates/chelis-cli/tests/cross_library_semantic_gap.rs` now proves direct, inline, user-def, and nested user-def C BLAS hits. Remaining work needs new negative and HIP tests. |
| **Perf-F1** | HIP batched matmul implementation quality | Benchmark and tune the `hipblasSgemmStridedBatched` path, retaining the helper loop for broadcasted/non-uniform leading strides. | Default structural coverage requires the strided-batched API on uniform layouts; the HIP manual GPU gate covers numerical agreement. |
| **Perf-F2** | Post-BLAS allocator/fusion compounding | Normalize equivalent symbolic shape expressions for slot reuse and broaden in-place elementwise/fan-in fusion where aliasing permits. | DimExpr normalization v1 now has target tests for product/identity canonicalization and negative tests that unrelated symbols are not alpha-renamed. C fused-elementwise in-place codegen now aliases a proven single reusable input and falls back for strided inputs; HIP in-place codegen and scoped same-property `forall` / binder-equivalent aliases remain future work. |

Fresh-context red-team status for M3/M3b: run 2026-05-10. The red-team
pass found one medium issue: the legacy C codegen-time BLAS detector could
still specialize non-contiguous rank-2 matrix slices by materializing
contiguous copies, bypassing the IR specializer's negative rule. The legacy
C detector path has been removed from reduction emission; BLAS now enters C
codegen through `RiscOp::BlasMatmul` produced by the IR specialization pass.

HIP manual gate status for this batch: run 2026-05-10 on the local ROCm/HIP
workstation with the documented `HSA_OVERRIDE_GFX_VERSION=11.5.1` environment;
`cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored
--test-threads=1` passed all 33 GPU correctness tests, including the new
batched hipBLAS helper test.

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

2. **Gap 4 is closed, with backend-quality follow-ons.** M3 picked the
   PyTorch-ergonomic answer by lifting `matmul` to rank ≥ 2, and M3b
   lets symbolic/batched matmul specialize through runtime BLAS sizes.
   The remaining work is quality of implementation: especially using
   `hipblasSgemmStridedBatched` on uniform HIP batch layouts instead of
   the current helper loop.

3. **The dispatch-coverage tail is unbounded.** Softmax, layer_norm,
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
