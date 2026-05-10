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
| **4** — Rank-2 matmul | partially closed by M3 | Medium-high | Type rule + generic `lower_matmul` now accept rank ≥ 2. Batched BLAS specialization remains open. |
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
linear * seq .....    2,244,364 bytes/seq
quadratic * seq² .          520 bytes/seq²
```

| seq | peak | dominant term |
|---|---|---|
| 128 | 282 MiB | linear (slot-planned generic matmul / activation intermediates) |
| 512 | 1.2 GiB | linear |
| **2048** | **6.3 GiB** | linear |
| 4096 | 16.7 GiB | linear, with quadratic catching up |

Decomposition at seq=2048:
- 4.4 GiB linear-in-seq — slot-planned Q/K/V/O, residual/LN, and FFN
  intermediates. Symbolic-dim matmuls still miss BLAS and use generic
  lowering, but non-overlapping buffers now share slots.
- 2.1 GiB quadratic-in-seq — attention-score and probs@V buffers that
  remain live under vanilla attention lowering.

**M1 closed Gap 6 for concrete rank-2 BLAS-hit matmul** by replacing
recognized matmul subgraphs before DCE/fusion. The transformer corpus
still shows the old symbolic-dim working set until Gap 4/M3b addresses
symbolic and batched matmul specialization, but M2a has reduced the
C helper-side allocation footprint by reusing non-overlapping slots.

If Gap 4 also closes (batched matmul) so that BLAS specialization
fires on the per-head matmuls, drops further. If FlashAttention-style
attention fusion also ships (not yet on the roadmap), the quadratic
term collapses entirely → ~200 MiB peak. That last figure is in the
same league as PyTorch.

### Cross-function specialization (Gap 5)

(`crates/chelis-cli/tests/cross_library_semantic_gap.rs`)

Same logical 8×16 @ 16×4 matmul, four code paths:

| Form | BLAS hits? | Working bytes | Throughput estimate |
|---|:---:|---|---|
| `f(a, b) = matmul(a, b)` | ✅ | 128 | ~100% (sgemm) |
| `f(a, b) = { ae=expand(a,...); be=expand(b,...); sum(mul(ae,be), 1) }` | ✅ | 128 | same |
| `def my_mm = matmul; def f = my_mm` | ❌ | 2176 (Mul allocated either way) | ~1-2% (scalar reduction) |
| `def my_mm = expand+mul+sum; def f = my_mm` | ❌ | 0 (host lane) | host-lane scalar |

M1 makes inline BLAS-hit matmul result-only in memory. The remaining Gap 5
cost is the function-boundary specialization miss: user-`def` wrappers still
lose BLAS dispatch and take the scalar-reduction fallback.

### Specialization dispatch reality

(`crates/chelis-cli/tests/specialization_dispatch.rs`)

Five common ML operations compiled to C:

| Op | Specialised? | Generic-path cost vs cuBLAS / cuDNN equivalent |
|---|:---:|---|
| concrete rank-2 matmul | ✅ cblas_sgemm | matches BLAS |
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

**Phase γ (partially closed by M3):** `matmul` now accepts rank ≥ 2
and `lower_matmul` emits the generic batched `expand + mul + sum`
decomposition. This closes the ergonomic story for canonical
heads-as-dim attention. The remaining Gap 4 performance work is batched
BLAS specialization for statically concrete shapes, with symbolic dims
falling through to generic lowering.

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
| **M3b** | Gap 4 performance follow-up | Add batched BLAS specialization for rank ≥ 3 matmul where every loop-bound/stride dimension is statically known. Symbolic dimensions must continue to fall through to generic lowering. | `crates/chelis-ir/src/specialize.rs::symbolic_matmul_stays_on_generic_path` locks the negative side; a positive batched-BLAS test still needs to be added with the implementation. |
| **M4** | Gap 3, gather/scatter lowering | Ship §3.5 gather lowering together with sparse gather/scatter recognition so embedding/MoE-shaped programs do not allocate dense `[N, V, D]` intermediates. | `crates/chelis-ir/tests/grad_gather_contract.rs` locks duplicate-index AD; emitted C/HIP structural tests for bounded sparse kernels still need to be added. |
| **M5-impl** | Gap 5, cross-function specialization | Implement verified BLAS-equivalent helper summaries and callsite emission rules from `spec/design/cross_function_specialization.md`. | `crates/chelis-cli/tests/cross_library_semantic_gap.rs::target_behavior_user_def_matmul_helpers_hit_blas` is ignored until this lands. |
| **M3-redteam** | Validation process | Run a contract-compliant fresh-context red-team pass for rank ≥ 2 matmul once local subagent execution is available. | The previous validation pass found no blockers but did not satisfy `redteam-exec`; do not count it as a formal red team. |

HIP manual gate status for this batch: run 2026-05-09 on the local ROCm/HIP
workstation with the documented `HSA_OVERRIDE_GFX_VERSION=11.5.1` environment;
`cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored
--test-threads=1` passed all 32 GPU correctness tests after fixing the
load-root host-output ownership bug exposed by `g7_host_device_roundtrip`.

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

2. **Gap 4 is split between ergonomics and performance.** M3 picked the
   PyTorch-ergonomic answer by lifting `matmul` to rank ≥ 2. Batched
   BLAS remains deliberately narrower and should only specialize shapes
   whose loop bounds and strides are statically known.

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
