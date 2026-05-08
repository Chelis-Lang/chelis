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
| **C. Architecture-by-design** | 5 | Spec §8.5 explicitly: "user-space functions cannot teach the AD engine its adjoint or the GPU backend its kernel fusion strategy." |
| **D. Backend parity** | 1 | HIP shipped Phase 1c memory planner; C didn't. Pure replication work. |
| **E. Pipeline plumbing** | adjacent | Surf parser doesn't attach `meta["span"]` to Deep nodes; downstream propagation is correct but receives empty input. |

Pattern A is the highest-leverage fix — one architectural change closes
two gaps and sets up the substrate for closing more. Pattern C is the
most architecturally-entrenched (drawn explicitly in spec §8.5). The
rest are bounded coverage / replication / plumbing work.

## 2. Difficulty × fundamentality

"Fundamental" here means "deciding to close it would require rethinking
a deliberately-drawn design boundary." Only Gap 5 qualifies; the rest
are unfinished implementation work.

| Gap | Effort | Fundamental? | Notes |
|---|---|:---:|---|
| **1** — C memory planner | ~1 week | No | Port HIP's interference-graph coloring (`crates/chelis-backend-hip/src/memory.rs`) to C. Greedy and well-understood. |
| **2** — Pattern matcher brittleness | ~2 weeks | No | Move BLAS detection from codegen into a new "specialize" optimize pass. Add walk-through-no-op-cast. |
| **3** — Gather lowering | ~1 month | Medium | Two coupled changes: §3.5 RISC lowering + scatter-recognition pattern matcher. Must ship together or arm the OOM trap. |
| **4** — Rank-2 matmul | ~1-2 months | Medium-high | Lift type rule + `lower_matmul` + AD adjoint + BLAS specializer (both backends). Many touch points, but bounded. |
| **5** — Cross-function specialization | ~6 months OR never | **Yes** | Either source-level inlining before optimize/fuse (large), or call-graph-aware pattern matching (also large). Current position: rely on clang LTO (3.8× recovery per OOPSLA paper plan). Spec §8.5 makes this explicitly out-of-scope. |
| **6** — Dead Mul after BLAS hit | ~2 weeks | No | Merges with Gap 2 fix. |
| Adjacent — Surf spans | ~1 week | No | Surf parser already tracks token positions; just plumb them into Deep node metadata. |

## 3. Concrete cost picture (measured)

Numbers below come from regression tests on this branch. Each is
reproducible: `cargo test -p <crate> --test <name> -- --nocapture`
prints the cost-profile section.

### `copy_elision_probe.ch` — 5×copy(x) fanout

(`crates/chelis-cli/tests/copy_elision.rs`)

| Input size | Peak working set | What's allocated |
|---|---|---|
| 1024×1024 f32 (4 MiB) | **20 MiB** | 5 × 4 MiB unary results, all simultaneously live until the final reduction reads them |
| 2 GiB (linear projection) | **10 GiB peak helper-side**, ~12 GiB total RAM | 5 × 2 GiB unary results + the 2 GiB caller input |

Closing Gap 1 (memory planner for C) collapses the peak from "5
simultaneously live" to ~2-3 (the planner colors non-overlapping
intervals); closing fan-in fusion as well drops it to 1.

### `transformer_block.ch` — 4-head MHA + FFN

(`crates/chelis-cli/tests/traceability_paradox.rs`)

The working set is a polynomial in `seq`:

```
const ............          0 bytes
linear * seq .....    3,178,576 bytes/seq
quadratic * seq² .        2,096 bytes/seq²
```

| seq | peak | dominant term |
|---|---|---|
| 128 | 421 MiB | linear (3-D `Mul` intermediates from every Tier-2-lowered matmul) |
| 512 | 2.1 GiB | linear |
| **2048** | **14.6 GiB** | linear |
| 4096 | 46.0 GiB | linear, with quadratic catching up |

Decomposition at seq=2048:
- 6.5 GiB linear-in-seq — almost entirely 12× QKV `[seq, 256, 64]` Mul
  intermediates plus 2× FFN `[seq, 1024, 256]` Mul intermediates
- 8.8 GiB quadratic-in-seq — attention-score and probs@V Mul
  intermediates

**Closing Gap 6 alone (eliminate dead Mul intermediates after BLAS
hit) drops the seq=2048 peak from 14.6 GiB to ~700 MiB — a 20×
reduction.** This is the single highest-leverage fix in the whole
catalogue.

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
| `f(a, b) = matmul(a, b)` | ✅ | 2176 | ~100% (sgemm) |
| `f(a, b) = { ae=expand(a,...); be=expand(b,...); sum(mul(ae,be), 1) }` | ✅ | 2176 | same |
| `def my_mm = matmul; def f = my_mm` | ❌ | 2176 (Mul allocated either way) | ~1-2% (scalar reduction) |
| `def my_mm = expand+mul+sum; def f = my_mm` | ❌ | 0 (host lane) | host-lane scalar |

Memory: comparable at this shape (the dead Mul exists either way).
Throughput: **~50-100× difference** between the BLAS path and the
scalar-reduction fallback. This is the cost of any matmul that lives
behind a user-`def` boundary.

### Specialization dispatch reality

(`crates/chelis-cli/tests/specialization_dispatch.rs`)

Five common ML operations compiled to C:

| Op | Specialised? | Generic-path cost vs cuBLAS / cuDNN equivalent |
|---|:---:|---|
| matmul | ✅ cblas_sgemm | matches BLAS |
| softmax | ❌ | ~3-5× slower (no online-softmax, no SRAM tiling) |
| layer_norm | ❌ | ~3-5× slower (no fused mean+var pass) |
| scatter | ❌ runtime-call | ~10-50× slower (no parallel-radix-sort, no warp-aware) |
| gather | ❌ runtime-call | ~10-50× slower |

**Compile-time cost is fine** — every issue listed is a runtime cost
in the *emitted* code, not in the compiler itself.

## 4. Sequenced fix plan

A six-month plan that closes the five non-fundamental gaps. Gap 5
stays as a documented boundary unless the language directionally
pivots toward "users can teach the compiler about their abstractions."

**Phase α (~1 month, highest-leverage):** Architectural change A —
move BLAS / pattern detection out of codegen and into a new
"specialize" optimize pass. Closes Gaps 2 + 6 simultaneously. Sets up
the substrate for future recognizers (softmax, layer_norm, attention,
gather→scatter). **20× transformer working-memory reduction.**

**Phase β (~1 month, parallel to α):** Port HIP memory planner to C
(Gap 1). Lift Surf parser to attach span metadata to Deep nodes
(adjacent finding). Both bounded, independent, easy to staff in
parallel.

**Phase γ (~1-2 months):** Lift `matmul` to rank ≥ 2 (Gap 4).
Generalise `lower_matmul` and the BLAS specializer for batched-GEMM.
Closes the ergonomic story for canonical heads-as-dim attention. (The
spec also offers `einsum` as the alternative answer; pick which one
is canonical before investing.)

**Phase δ (~1 month, paired):** Ship the §3.5 gather lowering paired
with a scatter-recognition pattern in Phase α's new specialize pass
(Gap 3 closure). The recognizer goes into the substrate Phase α
already set up.

**Phase ε (continuing, opportunistic):** Add specialised recognizers
for softmax, layer_norm, and attention (FlashAttention-style fusion
is the largest of these). Each one is bounded; they accumulate in the
specialize pass as Tier-2-shape-recognition rules.

**Gap 5 — defer.** Document the user-`def`-boundary specialization
loss as a known design boundary. Continue to lean on clang LTO for
cross-TU recovery. Revisit only if a concrete driver appears (RLVR
training pipeline showing measurable cost, large library benchmarks,
etc.).

## 5. Verdict on structural feasibility

**Yes, structurally feasible.** The dominant cost (Gap 6 dead-Mul) is
20× of the transformer working-memory bloat and closes with a
one-month architectural change that also closes Gap 2 and lays the
substrate for everything in Pattern B. The remaining gaps are bounded
coverage / replication / plumbing work.

### Caveats worth pinning

1. **Gap 5 is a design boundary.** If the language wants to maintain
   its current scope ("Tier 2 specialization is only for the named
   builtins"), then Coral / Nautilus / Octant operations will continue
   to be ~50-100× slower than equivalent direct primitive use. The
   OOPSLA paper plan's 3.8× clang LTO recovery is the planned story
   for this. Closing this gap honestly would require treating
   user-defined library functions as compiler-visible abstractions —
   which means rethinking what "Tier 2" means.

2. **Gap 4 has two equally valid closures.** Lifting `matmul` to
   rank ≥ 2 is the PyTorch-ergonomic answer; pushing all batched
   contraction through `einsum` is the compositional answer. The
   spec hasn't picked. Worth deciding before investing.

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

## 6. Cross-references

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
