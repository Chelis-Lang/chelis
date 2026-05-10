# Chelis Compiler Gaps — Empirical Findings

**Status:** partially closed — Gaps 2 and 6 are closed by M1; the adjacent
Surf-span finding is closed by M2b; Gap 4's rank ≥ 2 expressibility and
symbolic/batched BLAS performance paths are closed by M3/M3b. The remaining
work is explicitly tracked in `docs/gap_synthesis.md` §5 "Remaining Work
Register": Gap 3/M4, Gap 5 implementation, HIP strided-batched quality work,
post-BLAS slot/fusion compounding, and the formal red-team follow-up.
**Filed:** 2026-05-08
**Owning phase:** cross-phase (perf + ergonomics)

## Context

Five concrete compiler/codegen gaps were identified during a stress-test
review of Chelis's correctness-by-construction and cross-library AD
claims. Each gap is documented below with: the user-facing claim it
qualifies, the empirical observation, citations into the code where the
gap lives, and the test in this repo that locks today's behavior so
the gap re-surfaces if it re-opens or closes.

These are honest unfinished work, not blind spots. Every dangerous
lowering in `spec/05-risc-primitives.md` §3 / §4 is annotated with the
required pattern-match it depends on; the spec authors know the
heroics list. This document is the running summary of which heroics
have shipped and which haven't, as of 2026-05-08.

## Gap 1 — C-backend memory planning — CLOSED by M2a

**Claim qualified:** "linearity markers should not force avoidable peak memory."

**Current status:** M2a ports conservative slot planning to the C backend.
The copy-elision probe now emits four 4 MiB backing slots instead of five
4 MiB unary-result allocations; the final output wrapper reuses a dead
intermediate slot. For a 2 GB input this projects to ~8 GiB helper-side
slot footprint plus the borrowed 2 GiB input, not the prior ~12 GiB total.

**Original observation:** A function with five `copy(x)` calls feeding five
distinct unary results allocated five working buffers held alive
simultaneously. With a 2 GB `x`, peak working memory was ~12 GB even
though some buffers could share storage under a memory planner.

**Where it lived:** `crates/chelis-backend-c/src/memory.rs` previously
implemented only allocate-per-node cleanup. M2a replaces that with a
slot planner while keeping C-specific ownership rules for borrowed loads,
metadata views, standalone stores, and output materialization.

**Current copy-drop update:** `copy(x)` now lowers to an explicit
`RiscOp::Copy`, so it is visible to IR walkers, `chelis cost`, and the
memory-cost fitness signal. The C backend still emits zero raw `memcpy`
calls for the probe; explicit copies materialize through tensor
realization loops and participate in slot planning.

**What closed it:** the direct C-backend port. It intentionally remains
conservative for symbolic non-equality. Reducing the copy probe below four
slots still requires broader fan-in fusion; the C backend now has a narrower
in-place fused-elementwise path for single reusable inputs, but that does not
yet cover the full copy-probe fan-in shape.

**Spec coverage:**
- `spec/design/phase1c_memory_planning.md` ships the planner *as
  GPU-only by design* — the doc opens with "Phase 1c is a
  codegen/runtime-planning change inside the HIP backend."
- **Addressed by M2a:** the C backend now has its own slot planner. The
  planner is C-local rather than backend-agnostic.

**Locked test:** `crates/chelis-cli/tests/copy_elision.rs` —
`copy_elision_probe_reuses_c_backend_slots_without_materializing_copies`.
Asserts ≤4 backing slots, 0 memcpy, ≥2 fused parallel-for-simd, restrict
present, and explicit reuse of a dead intermediate slot by the final output
wrapper.

**Probe corpus:** `examples/illustrative/copy_elision_probe.ch`.

## Gap 2 — Pattern matchers are brittle to no-op interleaving — CLOSED by M1

**Claim qualified:** "Recognizable Tier 2 matmul patterns survive the
lowering pipeline back into BLAS specialization."

**Current status:** M1 added `chelis_ir::specialize`, an IR-level
specialization substrate that runs after AD and before DCE/fusion/codegen.
Its closed-list no-op cleanup removes identity `Cast`, identity `Reshape`,
and identity `Permute` nodes before replacing recognized matmul subgraphs
with `RiscOp::BlasMatmul`.

**Original observation:** The raw C backend BLAS detector keys off the
literal shape `Sum → Mul → (Expand, Expand)`. Inserting a no-op `Cast`
(e.g. `f32 → f32`) between an `Expand` and the `Mul` caused the detector
to miss. The raw detector remains strict, but the user-facing pipeline now
cleans the identity cast before specialization.

**Where it lives:** `crates/chelis-backend-c/src/blas.rs:42-64` —
`detect_matmul_pattern` walks `sum.inputs[0].op == Mul` then
`mul.inputs[*].op == Expand`. Any node interposed between them hides
the pattern. The HIP detector at `crates/chelis-backend-hip/src/blas.rs`
mirrors this brittleness.

**What's not at fault:** the spec acknowledges the brittleness — every
dangerous lowering in `spec/05-risc-primitives.md` §3.5 / §4.1 / §4.5
is annotated with "the compiler can recognize this pattern and emit
optimized BLAS / cuDNN / MKL calls" or "may special-case this pattern
for efficiency." The recognizer is an optimization, not a correctness
contract.

**What closed it:** option (a), implemented as a closed-list no-op cleanup
inside the IR specialization pass before BLAS replacement.

**Spec coverage:**
- `spec/05-risc-primitives.md` §3.5 / §4.1 / §4.5 acknowledge the
  pattern-match dependency in principle ("the compiler may
  special-case this pattern for efficiency").
- `spec/06-transformations.md` §5.4 ("Algebraic Simplification") and
  the broader optimize-pass design list constant-folding and
  algebraic identities, but no "no-op cast elimination before BLAS
  detection" pre-pass is concretely scoped.
- **Addressed by M1:** pass order is pinned as semantic lowering/AD first,
  then closed-list no-op cleanup, specialization replacement, DCE, fusion,
  and backend codegen.
— `canonical_matmul_pattern_is_detected` (positive),
`cast_perturbed_matmul_specializes_after_noop_cleanup` (positive),
`structural_no_gather_recognizer_today` (sentinel that there is no
analogous gather→scatter recognizer yet — see Gap 3).

### Mechanism walkthrough

How the matmul → cblas_sgemm dispatch actually happens, with file
citations, so future readers can audit the exact match conditions.

**1. The detector** — `crates/chelis-backend-c/src/blas.rs:27`
(`detect_matmul_pattern`) does a strict structural walk starting
from a `Sum` node:

| step | check | line |
|---|---|---|
| 1 | `sum_node.op == RiscOp::Sum { axis: _ }` | `blas.rs:31` |
| 2 | `sum.inputs.len() == 1` | `blas.rs:37` |
| 3 | `sum.inputs[0].op == RiscOp::Mul` | `blas.rs:41` |
| 4 | `mul.inputs.len() == 2` | `blas.rs:46` |
| 5 | `mul.inputs[0].op == RiscOp::Expand` | `blas.rs:52-55` |
| 6 | `mul.inputs[1].op == RiscOp::Expand` | `blas.rs:56-59` |
| 7 | each Expand has exactly one input | `blas.rs:62` |
| 8 | both Expand inputs have `dims.len() == 2` | `blas.rs:86` |
| 9 | every dim is `Lit(n)` or `Named(_, Some(n))` | `blas.rs:90-95` |

If all nine conditions hold, the detector returns
`Some(MatmulInfo { m, n, k })`. Otherwise `None`.

**2. The dispatch** — M1 moved user-facing dispatch to
`chelis_ir::specialize::specialize_for_blas`. The pass replaces the
recognized `Sum(Mul(Expand, Expand))` root with `RiscOp::BlasMatmul`,
then DCE removes the orphan `Mul` and `Expand` nodes before backend
emission. The C and HIP emitters now emit BLAS directly from the
specialized node.

**3. The link-flag side-channel** — C codegen now decides BLAS
requirements by scanning the post-specialization DAG for
`RiscOp::BlasMatmul`.

### Empirical brittleness sweep

Eleven perturbations of `f(a, b) = matmul(a, b)` (8×16 @ 16×4),
compiled to C and inspected for `cblas_sgemm` calls:

| Surf form | BLAS hits? | Why |
|---|:---:|---|
| `matmul(a, b)` (baseline) | ✅ | canonical Tier 2 lowering |
| `cast(matmul(a, b), f32)` | ✅ | cast wraps the Sum, doesn't intrude on the subgraph |
| `matmul(cast(a, f32), b)` | ✅ | cast wraps the Expand input; pattern below Sum still matches |
| `add(matmul(a, b), z)` | ✅ | add wraps after the Sum |
| `copy(matmul(a, b))` | ✅ | `copy` wraps the specialized node as an explicit `RiscOp::Copy` |
| `realize(matmul(a, b))` | ✅ | realize wraps after the Sum |
| `aa = a; bb = b; matmul(aa, bb)` | ✅ | let-bindings inlined in IR |
| `prod = mul(...); prod_again = prod; sum(prod_again, 1)` | ✅ | inner let-binding inlined |
| Hand-written `expand+mul+sum` inline | ✅ | produces the same canonical IR |
| `tensor[m, 16, f32]` (symbolic dim) | ✅ | IR-level specialization emits runtime-sized BLAS dimensions from shape bindings |
| `def my_mm(a, b) = matmul(a, b); def f(a, b) = my_mm(a, b)` | ❌ | detector runs on caller DAG, not helper (Gap 5) |

The structural fragility (`Sum → Mul → (Expand, Expand)` strict
walk) is real but **mostly invisible at the surface level today**
because the Tier 2 desugarer of `matmul` produces the canonical
subgraph with no user-reachable nodes between its steps. From Surf
source, the only easy ways to disable BLAS specialization are:

1. **Wrapping matmul in a user-level `def`** called from another
   `def`. The helper is emitted as a separate C function; the
   detector runs on the caller's DAG which only sees the call.
   Real-world impact: Coral / Nautilus / Octant abstractions over
   `matmul` lose specialization (Gap 5).
2. **Interposing a non-identity node between Expand and Mul (or Mul
   and Sum) at the IR level.** Identity casts, identity reshapes, and
   identity permutes are now removed by the M1 closed-list cleanup.
   Other interposed nodes still intentionally block specialization.

The original no-op brittleness is closed for the M1 no-op list. The
remaining miss cases are non-identity structural perturbations,
non-contiguous matrix slices, and user-`def` boundaries (Gap 5 workstream).

The structurally-better fix shipped in M1: BLAS detection moved into
the IR specialization substrate, paired with post-specialization DCE.
Gap 2 and Gap 6 close together under this approach.

## Gap 3 — Surf `gather` lowering / sparse recognizer not wired; OOM trap conditional

**Claim qualified:** "`gather` decomposes via `one_hot + expand + mul + sum`
(spec §3.5) so AD flows correctly through it."

**Current status:** the current branch ships the first sparse IR slice:
`RiscOp::Gather { axis }` and `RiscOp::ScatterAdd { axis }` have verifier,
evaluator, AD, C codegen, and compiler-API wire coverage. Duplicate-index
gradients accumulate through `ScatterAdd`, and generated C for first-class
sparse nodes has both bounded-memory structural coverage and a compile/run
numeric oracle.

**Remaining observation:** tensor-lane Surf `gather` now lowers directly to
the first-class sparse `RiscOp::Gather` node, so the old host/runtime path is
no longer the tensor-lane behavior. What remains open is the spec's §3.5
dense-decomposition route: the shared specialization substrate still does not
recognize a dense `one_hot + expand + mul + sum` tree and replace it with
sparse nodes before codegen. The OOM trap is still real if that dense lowering
ships without the recognizer, and HIP still rejects sparse gather/scatter with
an explicit diagnostic rather than emitting GPU kernels.

**Why this is a gap:** the spec promises §3.5 lowering. When that
lowering ships:
- AD will flow correctly through duplicate indices automatically
  (the `Sum → Expand` adjoint chain handles scatter-add by
  construction — verified by the contract test below).
- BUT the forward path will materialize a `[N, V, D]` intermediate
  for an embedding lookup of N tokens against a V-sized vocabulary.
  For typical LLM-scale shapes (V = 50K, D = 1K), that's ~200 GB —
  an OOM.

**What needs to ship together:** the §3.5 lowering AND a
scatter-recognition pattern matcher that turns the dense
`reshape+expand+mul+sum` shape back into a sparse scatter-add kernel.
Shipping the lowering without the recognizer arms the OOM trap.

**Where the recognizer would live:** the shared IR specialization substrate
(`crates/chelis-ir/src/specialize.rs`), reusing the pass order that already
handles BLAS before DCE/codegen. No dense gather recognizer exists today.

**Spec coverage:**
- `spec/05-risc-primitives.md` §3.5 documents the lowering shape
  (one_hot via const+eq+expand, then matmul; or via gather which
  itself lowers further). Spec §4.6 documents the embedding lowering
  via gather.
- `spec/design/chelis_phase3_plan.md` Phase 3h scope adds gather /
  scatter as core primitives and ships `Std.Nn.Embedding` as the
  named user-facing surface.
- **Roadmap status conflict partially reduced:** first-class sparse IR,
  tensor-lane Surf `gather` lowering, AD, and C codegen now exist, but Phase
  3h still should not be considered complete until the dense §3.5 recognizer
  and HIP sparse backend path ship.
- **Not fully addressed:** the joint requirement that any dense §3.5 lowering
  must ship paired with a scatter/gather recognizer.

**Locked test:** `crates/chelis-ir/tests/grad_gather_contract.rs` —
`gather_via_section_3_5_lowering_accumulates_duplicate_indices`.
Builds the post-§3.5 RISC DAG by hand and asserts the duplicate-index
gradient `[3, 3, 0, 0]` for an all-zero indices stress case. The test
proves the AD side will be correct by construction whenever §3.5
ships; it does not prove the OOM trap is closed.

**Required remaining M4 oracle:** closure must add the dense §3.5 recognizer
and HIP sparse codegen, then prove HIP build output for embedding/MoE-shaped
Surf lowering does not allocate the dense `[N, V, D]` one-hot materialization.
The C tensor-lane Surf path now lowers directly to first-class `Gather`, with
bounded emitted-code and small compile/run numerical coverage; the remaining
oracle is about the dense recognizer path and HIP parity.

**Probe corpus:** `examples/illustrative/moe_gather_duplicate_indices.ch`
(single MoE-style routing block with deliberately duplicated indices).

## Gap 4 — `matmul` is rank-2 only; canonical heads-as-dim MHA not expressible — CLOSED by M3/M3b

**Claim qualified:** "Multi-head attention expresses naturally as a
heads dimension, with batched matmul broadcasting over leading axes."

**Current status:** M3 lifted `matmul` to rank ≥ 2 at the type checker
and Tier 2 lowering layers. M3b extends the IR-level BLAS specializer to
symbolic and batched matmul when the operands have contiguous trailing
matrix slices. The C backend emits runtime-sized `cblas_sgemm` calls,
looping over batch slices for rank ≥ 3. The HIP backend emits through a
batched hipBLAS helper loop. A quality follow-up remains to use
`hipblasSgemmStridedBatched` directly for uniformly strided HIP batches.

**Original observation:** Chelis's `matmul` was hard rank-2 at the
type-checker level. PyTorch's canonical MHA form
(`qkv.reshape(batch, seq, num_heads, 3*head_dim).permute(0, 2, 1, 3)`
followed by batched scaled-dot-product attention over
`[batch, head, seq, dim]`) was not expressible.

**Where it lived:** `crates/chelis-types/src/infer.rs:7052` before M3:

```rust
if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
    errors.push(CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!("matmul expects rank-2 tensors, got rank {} and {}",
                lhs_dims.len(), rhs_dims.len()),
        vec![],
    ));
    return Type::Error;
}
```

**Former workaround in archived/illustrative corpus:** multi-head attention was
written as N per-head unrolled blocks, each with its own rank-2 `wq_i / wk_i /
wv_i / wo_i` and explicit copy fan-out. The executable
`examples/transformer_block.ch` now relies on implicit copy/drop insertion.

**What closed it:** generalizing `lower_matmul` and the matmul type rule
to accept rank ≥ 2, broadcasting over leading axes (the natural `... +
matrix-pair` shape rule), then extending the IR specializer and C/HIP
emitters to carry runtime `DimExpr` sizes into BLAS calls.

**Spec coverage:**
- `spec/design/chelis_canonical_reference.md:438-443` now documents
  runtime-sized symbolic/batched BLAS as shipped behavior, while naming
  the HIP strided-batched API as a quality follow-up.
- `spec/design/phase1d_flattening.md:39` now scopes the hipBLAS path to
  contiguous matrix slices rather than only rank-2 operands.
- `spec/design/chelis_phase2_plan.md:561` notes batched matmul
  remains correct via generic expand+mul+sum decomposition (i.e.,
  the rank-3+ case works numerically, just slowly).
- `spec/design/chelis_project_plan.md:457` and Phase 3h plan name
  **`einsum`** as the planned answer: "covers matmul, batched matmul,
  transpose, trace, outer products, and common contraction patterns
  in one primitive."
- **Addressed by M3:** `matmul` itself now accepts rank ≥ 2 and
  broadcasts leading axes for ergonomic parity with PyTorch-style
  batched matmul.
- **Addressed by M3b:** symbolic and batched matmul specialize to
  runtime-sized BLAS when trailing matrix slices are contiguous. The
  C backend loops over batch slices; HIP uses a helper loop over
  hipBLAS calls pending a strided-batched optimization.

**Probe corpus:**
- `examples/illustrative/mha_single_head.ch` — single-head reference,
  smallest viable MHA shape.
- `examples/illustrative/mha_two_heads_unrolled.ch` — multi-head via
  unrolling, the corpus-supported alternative to canonical
  heads-as-dim.
- `examples/illustrative/mha_heads_as_dim.ch` — canonical heads-as-dim
  batched matmul accepted by the M3 type rule and specialized by M3b
  when the emitted matrix slices are contiguous.
- `examples/illustrative/mha_slice_combined_qkv.ch` — combined-QKV
  with `shrink(&qkv)` borrows demonstrating linearity allows the
  zero-copy slicing pattern.

## Gap 5 — Cross-function pattern matching / inlining — first C BLAS slice shipped

**Claim qualified:** "AD flows through Coral / Nautilus / Octant /
Shoals because everything compiles to the same RISC primitive set."

**Current status:** simple pure C user-defined matmul helpers now recover the
BLAS path through compiler-derived host summaries and helper-body
specialization. The same logical matmul (8×16 @ 16×4) now compiles as:

| Form | `cblas_sgemm`? |
|---|---|
| `f(a, b) = matmul(a, b)` | ✅ |
| `f(a, b) = { ae = expand(a, ...); be = expand(b, ...); sum(mul(ae, be), 1) }` | ✅ |
| `def my_mm(a, b) = matmul(a, b); def f(a, b) = my_mm(a, b)` | ✅ |
| `def my_mm(a, b) = { expand+mul+sum }; def f(a, b) = my_mm(a, b)` | ✅ |

**The bridge that works:** the IR optimizer treats inline
hand-written `expand+mul+sum` identically to a Tier 2 desugared
matmul. The semantic gap is bridged for inline code.

**The bridge that now works for the first C BLAS slice:** wrapping the math
in a simple pure user `def` no longer loses BLAS in C. The generated helper
surface is still emitted for debugging/non-specialized paths, but eligible
calls can emit the specialized BLAS path without relying on clang/gcc LTO.

**The bridge that still doesn't:** this is not yet a general user-library
specialization system. HIP summary consumption, gather/scatter summaries,
summary-derived-but-callsite-rejected diagnostics, and broader helper shapes
remain follow-up work.

**Why this matters for the cross-library AD claim:** a Coral
`groupby + sum`, a Nautilus `simpsons_rule_integral`, or an Octant
LaTeX-translated equation written as a user-level `def` will not
inherit the codegen quality of builtin operations even though they
decompose into the same RISC primitives. The spec already concedes
this for `Nautilus.LinAlg` — those operators ship with hand-written
AD adjoints over opaque nalgebra calls
(`spec/design/chelis_canonical_reference.md:195` /
`spec/design/chelis_project_plan.md:557-562`) — but the user-level
broader concession isn't documented.

**What remains to close it fully:** broaden the M5 summary workstream beyond
the shipped C BLAS helper slice. Whole-program inlining remains an
implementation technique for small helpers, not the design contract.

**Spec coverage:**
- `spec/design/chelis_canonical_reference.md:463-464` now frames the
  boundary as current behavior, not a permanent design principle, and
  points to `spec/design/cross_function_specialization.md`.
- `spec/design/chelis_phase2_plan.md:326` confirms the same rule on
  the linearity side as current scope: no general cross-function helper
  specialization is part of the Phase 2 completion claim.
- `spec/design/chelis_oopsla_paper_plan.md:125` documents the perf
  cost and now frames clang LTO as a workaround rather than the codegen
  story for backend dispatch.
- `spec/design/chelis_span_survival.md:97` documents one *narrow*
  inlining mechanism — `inline_top_level_host_call` for HOF
  specialization (e.g. `grad(local_fn)(theta)`), which is how AD
  through user-defined wrapper functions stays correct.
- **Addressed by this branch for C BLAS helpers:** `cross_library_semantic_gap.rs`
  now proves direct, inline, user-def, and nested user-def matmul forms hit
  generated-C BLAS.
- **Not fully addressed:** HIP summary consumption, gather/scatter summaries,
  negative diagnostics for rejected summarized callsites, and broader helper
  compositions.

**Locked test:** `crates/chelis-cli/tests/cross_library_semantic_gap.rs`
asserts BLAS hits for the direct, inline, user-`def`, and nested user-`def`
forms.

## Gap 6 — BLAS-specialized matmul still allocates and computes the dead `Mul` intermediate — CLOSED by M1

**Claim qualified:** "Tier 2 BLAS specialization eliminates the
naïve `expand+mul+sum` cost when the pattern is recognized."

**Current status:** M1 eliminates both the compute and memory cost for
BLAS-hit rank-2 matmul. The IR specialization pass replaces the
recognized subgraph with `RiscOp::BlasMatmul` before DCE/fusion, so the
3-D `Mul` intermediate is not emitted.

**Original observation:** Codegen-time BLAS detection eliminated the
compute cost on the `Sum` step but left the 3-D `Mul` intermediate
allocated and computed independently.

**Where it lives:** `crates/chelis-ir/src/tier2.rs::lower_matmul`
emits the `Mul` node into the IR DAG. BLAS detection at
`crates/chelis-backend-c/src/emit.rs:1522` runs at codegen time
(after the optimize / DCE pass has already finished), so when the
detector replaces the `Sum` with sgemm, the now-orphan `Mul` is
still in the DAG and the emitter generates its allocation + fused
kernel anyway.

**Cost (cubic in matmul size):**

| Matmul shape | Dead `Mul` size | Useful result |
|---|---|---|
| 8 × 16 × 4 (f32) | 2 KiB | 128 B |
| 256 × 256 × 64 (f32) | 16 MiB | 64 KiB |
| 2048 × 2048 × 2048 (f32) | **32 GiB** | 16 MiB |
| 1024 × 4096 × 1024 (f32, FFN) | **16 GiB** | 4 MiB |

In `examples/transformer_block.ch`, M3b now lets symbolic `seq` matmuls
hit BLAS. The measured `seq = 2048` helper-side projection drops from
the prior M2a ~6.3 GiB state to ~1.06 GiB. The remaining dominant cost is
vanilla attention's real `[seq, seq]` score/probability buffers, not dead
generic matmul products.

**What closed it:** option (a). BLAS detection now runs as an IR
replacement pass followed by DCE, which prunes the orphan `Mul` and
`Expand` nodes.

**Spec coverage:**
- `spec/design/phase1d_flattening.md` ships the BLAS specializer
  as a codegen-time pattern match. No proposal addresses the
  dead-`Mul` follow-on.
- `spec/06-transformations.md` §5.2 (DCE) defines DCE but does
  not require it to run after codegen-time pattern matching.
- **Addressed by M1.** The cost-profile assertions now require the
  direct and inline BLAS-hit paths to allocate only the result buffer.

**Locked test:** the cost-profile assertions in
`crates/chelis-cli/tests/cross_library_semantic_gap.rs` already
encode the closed behavior (8×16 @ 16×4 → 128 working bytes, result
only) while keeping the user-`def` specialization miss locked for Gap 5.

**Probe corpus:** any matmul-heavy program. The
`mha_two_heads_unrolled.ch` and `transformer_block.ch` examples
amplify the cost dramatically when their matmuls miss specialization.
Rank-2, symbolic, and batched BLAS-hit matmuls no longer pay the
dead-`Mul` tax.

**Tracked in:**
`spec/upstream-bugs/dead-mul-after-blas-specialization.md`.

## Adjacent finding (not in the six) — closed by M2b

**Surf-source spans now reach the IR.** The Surf desugarer threads
parser byte ranges into Deep `meta["span"]` as opaque IDs of the form
`surf:<start>..<end>` for ordinary Surf expression bodies. Downstream
IR lowering, transformation passes, and backend emitters already
preserve `meta["span"]`, so generated source can now contain Surf
byte-range comments instead of bottoming out entirely at
`__synthesized_tier2__`.

Synthesized markers remain valid only for nodes whose source span input
is genuinely absent. Hand-constructed Surf ASTs that carry the legacy
zero-length sentinel still desugar without `meta["span"]`, preserving
the existing fallback behavior for tests and synthetic producers.

**Locked test:** `crates/chelis-cli/tests/traceability_paradox.rs` —
`transformer_block_traceability_state_is_locked`. Asserts that emitted
span comments include `surf:<start>..<end>` byte-range IDs and are no
longer all `__synthesized_*` markers. The same test also locks the
current symbolic/batched-BLAS transformer allocation profile so future
attention fusion or backend-quality changes must update the cost profile
deliberately.

**Spec coverage:**
- `spec/design/chelis_span_survival.md:64-72` defines the
  propagation rules for every pass (Lowering / Constant fold / DCE /
  CSE / Tier 2 / AD / Fusion / Vmap / Verify / Codegen). The rules
  are correct and the implementation follows them.
- The same doc accommodates "parent had no span" via the
  `__synthesized_tier2__` fallback. After M2b this fallback is reserved
  for genuinely spanless inputs, not ordinary parsed Surf bodies.
- **Addressed:** the upstream cause was the Surf desugarer emitting empty
  Deep metadata for parsed Surf expressions. It now writes `meta["span"]`
  when the Surf AST span has a real byte range.

## Honest scope

What this document does **not** claim:

- That these gaps are bugs. They are unfinished work; the spec
  consistently calls them out as required heroics.
- That closing them is straightforward. Each closure has follow-on
  costs (cross-function inlining changes the AD pipeline ordering;
  generalizing matmul to rank ≥ 2 forces the BLAS specializer to
  handle batched GEMM; the §3.5 gather lowering must ship paired with
  scatter recognition).
- That the gaps invalidate the framework. The correctness-by-
  construction story holds: linearity prevents micro-fan-out at the
  variable level (`crates/chelis-types/src/linearity.rs`), AD
  multi-consumer accumulation is correct
  (`crates/chelis-ir/src/grad.rs:188-196`), and the §3.5 RISC
  composition would correctly accumulate duplicate-index gradients
  whenever the lowering ships (verified empirically in the contract
  test).

What it does claim: each gap is empirically observable today, and a
test in this repo will fail if the gap closes (or, conversely, will
need updating if it doesn't).
