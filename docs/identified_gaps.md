# Chelis Compiler Gaps — Empirical Findings

**Status:** open — six gaps documented, each with a regression test.
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

## Gap 1 — C-backend memory planning is still allocate-per-node

**Claim qualified:** "`copy(x)` is free; the compiler will reuse buffers."

**Observation:** A function with five `copy(x)` calls feeding five
distinct unary results allocates five working buffers held alive
simultaneously. With a 2 GB `x`, peak working memory is ~12 GB even
though the five unaries could share buffers under a memory planner.

**Where it lives:** `crates/chelis-backend-c/src/memory.rs` — the
module docstring explicitly says: *"Phase 0: simple allocate-per-node,
free-all-at-end strategy."* The HIP backend has Phase 1c memory
planning (`crates/chelis-backend-hip/src/memory.rs`), but C codegen
has not adopted it.

**What's not at fault:** `copy(x)` markers themselves produce zero
buffers and zero `memcpy` calls. The cost is from the unary results
(`exp(x)`, `log(x)`, `sin(x)`, etc.), each of which allocates its own
output. Kernel fusion does eliminate the `add`-chain intermediates.

**What would close it:** lift the HIP memory planner's interference-
graph coloring into a backend-agnostic IR pass, or port it to the C
backend directly.

**Spec coverage:**
- `spec/design/phase1c_memory_planning.md` ships the planner *as
  GPU-only by design* — the doc opens with "Phase 1c is a
  codegen/runtime-planning change inside the HIP backend."
- `crates/chelis-backend-c/src/memory.rs` opening comment is honest:
  *"Phase 0: simple allocate-per-node, free-all-at-end strategy."*
- **Not addressed:** no proposal in `spec/design/` plans porting the
  planner to the C backend or lifting it into a backend-agnostic IR
  pass. Closure path is implicit (copy the HIP module) but unowned.

**Locked test:** `crates/chelis-cli/tests/copy_elision.rs` —
`copy_elision_probe_emits_no_memcpy_and_one_buffer_per_unary_result`.
Asserts 5 allocs, 0 memcpy, ≥2 fused parallel-for-simd, restrict
present.

**Probe corpus:** `examples/illustrative/copy_elision_probe.ch`.

## Gap 2 — Pattern matchers are brittle to no-op interleaving

**Claim qualified:** "Recognizable Tier 2 patterns survive the lowering
pipeline back into BLAS / cuDNN / MKL specializations."

**Observation:** The C backend's BLAS detector keys off the literal
shape `Sum → Mul → (Expand, Expand)`. Inserting a no-op `Cast` (e.g.
`f32 → f32`) between an `Expand` and the `Mul` causes the detector to
miss, falling through to the scalar `expand+mul+sum` codegen.

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

**What would close it:** either (a) a constant-fold / no-op-cast
elimination pass before the BLAS detector runs, or (b) walk-through
detectors that skip identity / no-op nodes.

**Spec coverage:**
- `spec/05-risc-primitives.md` §3.5 / §4.1 / §4.5 acknowledge the
  pattern-match dependency in principle ("the compiler may
  special-case this pattern for efficiency").
- `spec/06-transformations.md` §5.4 ("Algebraic Simplification") and
  the broader optimize-pass design list constant-folding and
  algebraic identities, but no "no-op cast elimination before BLAS
  detection" pre-pass is concretely scoped.
- **Not addressed:** the order between cast-eliding constant-fold
  and BLAS detection isn't pinned down; today the BLAS detector runs
  on a DAG that may still contain identity casts.
— `canonical_matmul_pattern_is_detected` (positive),
`cast_perturbed_matmul_pattern_misses` (negative),
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

**2. The dispatch** — `crates/chelis-backend-c/src/emit.rs:1525-1530`,
inside `emit_reduce_sum`:

```rust
if self.use_blas
    && let Some(matmul) = crate::blas::detect_matmul_pattern(dag, NodeId(id))
{
    self.emit_blas_matmul(id, &matmul, ty);
    return;
}
```

On hit, the emitter produces `cblas_sgemm(...)` reading directly
from the original `A` and `B` (not from any intermediate buffer)
and writes to a fresh result tensor, then `return`s — short-
circuiting the generic Sum-of-Mul reduction code that would
otherwise emit a scalar reduction loop. **The `Mul` predecessor is
still emitted by an independent visitor pass** (the dead-Mul
finding in Gap 6).

**3. The link-flag side-channel** —
`crates/chelis-backend-c/src/lib.rs:122` runs
`detect_matmul_pattern` across every `Sum` node before codegen so
the build can decide whether to require the `-lopenblas` link flag.

### Empirical brittleness sweep

Eleven perturbations of `f(a, b) = matmul(a, b)` (8×16 @ 16×4),
compiled to C and inspected for `cblas_sgemm` calls:

| Surf form | BLAS hits? | Why |
|---|:---:|---|
| `matmul(a, b)` (baseline) | ✅ | canonical Tier 2 lowering |
| `cast(matmul(a, b), f32)` | ✅ | cast wraps the Sum, doesn't intrude on the subgraph |
| `matmul(cast(a, f32), b)` | ✅ | cast wraps the Expand input; pattern below Sum still matches |
| `add(matmul(a, b), z)` | ✅ | add wraps after the Sum |
| `copy(matmul(a, b))` | ✅ | `copy` is a linearity marker, no IR node |
| `realize(matmul(a, b))` | ✅ | realize wraps after the Sum |
| `aa = a; bb = b; matmul(aa, bb)` | ✅ | let-bindings inlined in IR |
| `prod = mul(...); prod_again = prod; sum(prod_again, 1)` | ✅ | inner let-binding inlined |
| Hand-written `expand+mul+sum` inline | ✅ | produces the same canonical IR |
| `tensor[m, 16, f32]` (symbolic dim) | ❌ | step 9 above (`dim_size` requires concrete `n`) |
| `def my_mm(a, b) = matmul(a, b); def f(a, b) = my_mm(a, b)` | ❌ | detector runs on caller DAG, not helper (Gap 5) |

The structural fragility (`Sum → Mul → (Expand, Expand)` strict
walk) is real but **mostly invisible at the surface level today**
because the Tier 2 desugarer of `matmul` produces the canonical
subgraph with no user-reachable nodes between its steps. From Surf
source, the only easy ways to disable BLAS specialization are:

1. **Symbolic dim anywhere in the matmul shape.** Real-world
   impact: every matmul in `examples/transformer_block.ch` (the
   `seq` dim) — sequence-length-polymorphic transformer code
   currently never hits BLAS in the C backend.
2. **Wrapping matmul in a user-level `def`** called from another
   `def`. The helper is emitted as a separate C function; the
   detector runs on the caller's DAG which only sees the call.
   Real-world impact: Coral / Nautilus / Octant abstractions over
   `matmul` lose specialization (Gap 5).
3. **Interposing a node between Expand and Mul (or Mul and Sum) at
   the IR level.** Not easy from Surf today — the desugarer doesn't
   emit such nodes for any standard idiom — but trivial from Deep
   or programmatic DAG construction. Future DCE / CSE / fusion
   passes that interleave nodes will silently break BLAS detection
   for any matmul they touch.

The brittleness is **latent**: it doesn't bite typical Surf code
today, but it bites as soon as (a) symbolic-dim matmul becomes
common (transformer training), (b) library code gets layered, or
(c) future optimization passes start interleaving casts / no-ops
between Tier 2 nodes. The cast-perturbation regression test in
`pattern_matcher_brittleness.rs` is the early-warning system for
case (c).

The structurally-better fix is the same one outlined in Gap 6:
move BLAS detection out of codegen and into the optimize pass,
where (a) it runs alongside algebraic simplification so identity
casts and reshape no-ops can be skipped, and (b) it can rewrite
the DAG to remove the now-orphan Mul once the Sum has been
replaced. Both Gap 2 and Gap 6 close together under that approach.

## Gap 3 — `gather` lowering not wired; OOM trap conditional

**Claim qualified:** "`gather` decomposes via `one_hot + expand + mul + sum`
(spec §3.5) so AD flows correctly through it."

**Observation:** The decomposition is **not yet wired** in the
implementation. `gather` is a host-only Tier 2 builtin
(`crates/chelis-ir/src/host.rs:4640`) — programs that use `gather`
take the host runtime path entirely, and `grad` over a function
containing `gather` refuses (fail-closed, not silent-drop).

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

**Where the recognizer would live:** new `detect_gather_pattern` in
either backend `blas.rs` or in `crates/chelis-ir/src/optimize.rs`. No
such function exists today.

**Spec coverage:**
- `spec/05-risc-primitives.md` §3.5 documents the lowering shape
  (one_hot via const+eq+expand, then matmul; or via gather which
  itself lowers further). Spec §4.6 documents the embedding lowering
  via gather.
- `spec/design/chelis_phase3_plan.md` Phase 3h scope adds gather /
  scatter as core primitives and ships `Std.Nn.Embedding` as the
  named user-facing surface.
- **Roadmap status conflict:** `spec/12-roadmap.md` marks Phase 3h
  *shipped*, but empirically `gather` is host-only with no AD
  adjoint — meaning 3h shipped the *primitive* (forward-only host
  evaluation) without shipping the §3.5 RISC lowering OR the
  scatter-recognition pattern that would close the OOM trap. The
  Phase 3h doc says "extend the type/checking/lowering/backend docs
  and implementation for the new tensor primitives" but does not
  call out the recognizer requirement explicitly.
- **Not addressed:** the joint requirement that the §3.5 lowering
  must ship paired with a scatter recognizer.

**Locked test:** `crates/chelis-ir/tests/grad_gather_contract.rs` —
`gather_via_section_3_5_lowering_accumulates_duplicate_indices`.
Builds the post-§3.5 RISC DAG by hand and asserts the duplicate-index
gradient `[3, 3, 0, 0]` for an all-zero indices stress case. The test
proves the AD side will be correct by construction whenever §3.5
ships; it does not prove the OOM trap is closed.

**Probe corpus:** `examples/illustrative/moe_gather_duplicate_indices.ch`
(single MoE-style routing block with deliberately duplicated indices).

## Gap 4 — `matmul` is rank-2 only; canonical heads-as-dim MHA not expressible

**Claim qualified:** "Multi-head attention expresses naturally as a
heads dimension, with batched matmul broadcasting over leading axes."

**Observation:** Chelis's `matmul` is hard rank-2 at the type-checker
level. PyTorch's canonical MHA form
(`qkv.reshape(batch, seq, num_heads, 3*head_dim).permute(0, 2, 1, 3)`
followed by batched scaled-dot-product attention over
`[batch, head, seq, dim]`) is not expressible.

**Where it lives:** `crates/chelis-types/src/infer.rs:7052`:

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

**Workaround in current corpus:** multi-head attention is written as
N per-head unrolled blocks, each with its own rank-2 `wq_i / wk_i /
wv_i / wo_i` and an explicit `matmul(copy(x), wq_i)` — see
`examples/transformer_block.ch` (4 heads) and
`examples/illustrative/mha_two_heads_unrolled.ch` (2 heads).

**What would close it:** generalize `lower_matmul` and the matmul
type rule to accept rank ≥ 2, broadcasting over leading axes (the
natural `... + matrix-pair` shape rule). The Tier 2 lowering already
emits expand+mul+sum which would extend cleanly, but the BLAS
specializer would need to recognize batched-GEMM patterns too.

**Spec coverage:**
- `spec/design/chelis_canonical_reference.md:438-443` explicitly
  documents the perf gap: "the existing HIP rank-2 BLAS fast path
  does not yet upgrade vmapped rank-3 matmul into a batched BLAS
  call."
- `spec/design/phase1d_flattening.md:39` confirms the rank-2
  restriction is *intentional* in the hipBLAS specializer: "Only
  statically contiguous rank-2 f32 operands take the hipBLAS path."
- `spec/design/chelis_phase2_plan.md:561` notes batched matmul
  remains correct via generic expand+mul+sum decomposition (i.e.,
  the rank-3+ case works numerically, just slowly).
- `spec/design/chelis_project_plan.md:457` and Phase 3h plan name
  **`einsum`** as the planned answer: "covers matmul, batched matmul,
  transpose, trace, outer products, and common contraction patterns
  in one primitive."
- **Roadmap status conflict:** `einsum` is registered as a Tier 2
  builtin (`crates/chelis-types/src/builtins.rs:135`) and Phase 3h
  is marked shipped, but empirically the **type checker for
  `matmul` itself still rejects rank ≥ 2** (`infer.rs:7052`). The
  PyTorch-style heads-as-dim MHA form requires `matmul` (or einsum
  via reshape gymnastics) to broadcast cleanly; today users must
  unroll heads as separate rank-2 matmuls.
- **Not explicitly addressed:** whether `einsum` is meant to fully
  *replace* `matmul` for batched cases, or whether `matmul`'s type
  rule should be lifted to rank ≥ 2 for ergonomic parity with
  PyTorch.

**Probe corpus:**
- `examples/illustrative/mha_single_head.ch` — single-head reference,
  smallest viable MHA shape.
- `examples/illustrative/mha_two_heads_unrolled.ch` — multi-head via
  unrolling, the corpus-supported alternative to canonical
  heads-as-dim.
- `examples/illustrative/mha_slice_combined_qkv.ch` — combined-QKV
  with `shrink(&qkv)` borrows demonstrating linearity allows the
  zero-copy slicing pattern (single-head, since matmul is rank-2).

## Gap 5 — Cross-function pattern matching / inlining

**Claim qualified:** "AD flows through Coral / Nautilus / Octant /
Shoals because everything compiles to the same RISC primitive set."

**Observation:** The same logical matmul (8×16 @ 16×4) compiled four
ways:

| Form | `cblas_sgemm`? |
|---|---|
| `f(a, b) = matmul(a, b)` | ✅ |
| `f(a, b) = { ae = expand(a, ...); be = expand(b, ...); sum(mul(ae, be), 1) }` | ✅ |
| `def my_mm(a, b) = matmul(a, b); def f(a, b) = my_mm(a, b)` | ❌ |
| `def my_mm(a, b) = { expand+mul+sum }; def f(a, b) = my_mm(a, b)` | ❌ |

**The bridge that works:** the IR optimizer treats inline
hand-written `expand+mul+sum` identically to a Tier 2 desugared
matmul. The semantic gap is bridged for inline code.

**The bridge that doesn't:** wrapping the math in a separate user
`def` causes the compiler to emit a separate C function
(`my_mm__tensor_0(inputs, n_in, outputs, n_out)`), and the BLAS
detector keys off the *caller* DAG, not the helper. Library-level
abstractions sitting behind a function-call boundary lose BLAS /
cuDNN / scatter specialization.

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

**What would close it:** either (a) cross-function inlining before
optimize/fuse so helpers are visible to the BLAS detector, or (b)
call-graph-aware pattern matching that runs the detector on each
function's DAG independently and remembers a "this helper is a
matmul" annotation, then re-emits the helper as a BLAS call.

**Spec coverage:**
- `spec/design/chelis_canonical_reference.md:463-464` makes the
  scope boundary explicit by design: *"They cannot be defined as
  user-space library functions because a user-space function cannot
  teach the AD engine its adjoint or the GPU backend its kernel
  fusion strategy."* Tier 2 specialization is **only for the named
  builtins**; user defs are out of scope by intent.
- `spec/design/chelis_phase2_plan.md:326` confirms the same rule on
  the linearity side: "Linearity checking is intra-procedural — no
  cross-function lifetime analysis."
- `spec/design/chelis_oopsla_paper_plan.md:125` documents the perf
  cost they've already measured: *"LTO finding: 3.8x improvement
  from cross-TU inlining — a codegen insight, not just a benchmark
  result."* The current workaround is to lean on **clang LTO at link
  time** rather than source-level inlining inside Chelis.
- `spec/design/chelis_span_survival.md:97` documents one *narrow*
  inlining mechanism — `inline_top_level_host_call` for HOF
  specialization (e.g. `grad(local_fn)(theta)`), which is how AD
  through user-defined wrapper functions stays correct.
- **Not addressed:** general user-`def`-boundary inlining for
  tensor pattern matching. The OOPSLA paper plan documents the LTO
  workaround as the codegen story; no proposal exists for moving
  inlining earlier (pre-optimize/fuse) so BLAS detection sees user
  helpers. **This is the most architecturally entrenched gap of the
  five** — it's not "not yet shipped," it's "explicitly out of scope
  for source-level optimization, recovered partially via clang LTO."

**Locked test:** `crates/chelis-cli/tests/cross_library_semantic_gap.rs`
— `semantic_gap_inline_vs_user_def`. Asserts BLAS hits for the two
inline forms and misses for the two user-`def` forms.

## Gap 6 — BLAS-specialized matmul still allocates and computes the dead `Mul` intermediate

**Claim qualified:** "Tier 2 BLAS specialization eliminates the
naïve `expand+mul+sum` cost when the pattern is recognized."

**Observation:** It eliminates the *compute cost* on the `Sum`
step (replaced by `cblas_sgemm`), but **not** the memory cost on
the `Mul` step. The 3-D `Mul` intermediate `[m, k, n]` is still
allocated and computed in a fused parallel-for-simd loop, then
freed without ever being read by sgemm (which reads inputs
directly).

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

In `examples/transformer_block.ch`, this is the dominant
working-set cost — ~3 MiB per `seq` token, almost entirely from
dead `Mul` intermediates that BLAS hits would have eliminated in
any normal compiler pipeline.

**What would close it:** either (a) move BLAS detection into the
optimize pass so it can run DCE afterwards and prune the orphan
`Mul`, or (b) keep codegen-time detection but extend the emitter
to skip emission of the `Mul` (and its `Expand` operands) when
the consumer has been specialized to sgemm.

**Spec coverage:**
- `spec/design/phase1d_flattening.md` ships the BLAS specializer
  as a codegen-time pattern match. No proposal addresses the
  dead-`Mul` follow-on.
- `spec/06-transformations.md` §5.2 (DCE) defines DCE but does
  not require it to run after codegen-time pattern matching.
- **Not addressed.** The cost was discovered empirically when
  parsing emitted C for cost-profile assertions.

**Locked test:** the cost-profile assertions in
`crates/chelis-cli/tests/cross_library_semantic_gap.rs` already
encode today's reality (8×16 @ 16×4 → 2176 working bytes, of
which 2048 are the dead `Mul`). When this gap closes, the
assertion will flip to ~128 bytes (result only) and the test
docstring needs updating.

**Probe corpus:** any matmul-heavy program. The
`mha_two_heads_unrolled.ch` and `transformer_block.ch` examples
amplify the cost dramatically because every per-head matmul pays
the dead-`Mul` tax.

**Tracked in:**
`spec/upstream-bugs/dead-mul-after-blas-specialization.md`.

## Adjacent finding (not in the six)

**Surf-source spans don't reach the IR.** Compiling
`examples/transformer_block.ch` to C produces 158 `// span:` comments,
all of which are `__synthesized_tier2__`. None of them carries an
original Surf line number. The cause is upstream of fusion: the Surf
parser does not currently attach source spans to the Deep AST for
this corpus, so there is nothing for the IR pipeline to thread
through. The traceability machinery in
`spec/design/chelis_span_survival.md` is correct and runs; the bug is
that no spans enter the pipeline. This makes the audit chain
practically useless for back-tracing a generated C kernel to a Surf
line of business code.

**Locked test:** `crates/chelis-cli/tests/traceability_paradox.rs` —
`transformer_block_traceability_state_is_locked`. Asserts every span
is a `__synthesized_*` marker.

**Spec coverage:**
- `spec/design/chelis_span_survival.md:64-72` defines the
  propagation rules for every pass (Lowering / Constant fold / DCE /
  CSE / Tier 2 / AD / Fusion / Vmap / Verify / Codegen). The rules
  are correct and the implementation follows them.
- The same doc accommodates "parent had no span" via the
  `__synthesized_tier2__` fallback (line 68). This is exactly the
  case our test observes: parent Deep nodes carry no `meta["span"]`,
  so every Tier 2 sub-node falls through to the synthesized marker.
- **Not addressed:** the upstream cause — that the **Surf parser
  does not attach `meta["span"]` to Deep nodes** for typical
  function bodies — isn't tracked anywhere in `spec/upstream-bugs/`
  or in the span-survival doc. The audit chain machinery is
  designed correctly, but the user-source spans never enter the
  pipeline, so the chain bottoms out at "synthesized."

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
