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

If FlashAttention-style attention fusion ships, the quadratic
score/probability materialization can collapse further. Slot planning
alone cannot make those tensors smaller because they are real
intermediate values, not allocator artifacts. **Planned closure:**
Kerrent Phase K6 (`spec/design/kerrent.md` §Milestone 6) is the
committed-scope path — a FlashAttention-shaped fused attention kernel
authored in Kerrent and called from tensor-level Chelis, replacing the
current attention decomposition. See `spec/design/chelis_project_plan.md`
§Kerrent Track for the delivery sequence.

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
for softmax, layer_norm, and attention. Each one is bounded; they
accumulate in the specialize pass as Tier-2-shape-recognition rules.
**FlashAttention-style fusion is the largest of these and has graduated
to a separate committed-scope path:** Kerrent Phase K6
(`spec/design/kerrent.md` §Milestone 6) authors the kernel in Chelis
source rather than recognizing-and-fusing the decomposed pattern after
the fact. The remaining recognizers (softmax, layer_norm, and any
non-attention fusion targets) still belong in the specialize pass.

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
| **M5-follow-up** | Gap 5, cross-function specialization | **CLOSED by W3-A + W3-B + W4-A + W6.** All four originally-open sub-items shipped: (a) HIP consumption for BLAS summary kinds — closed by W3-A via `lower_named_tensor_entry_dag` inlining. (b) HIP consumption for sparse summary kinds — closed by W6 Task B via the same inlining path (lock-tests passed on first run; no HIP codegen changes needed). (c) Sparse-helper rejection diagnostics — closed by W4-A with the public `SummaryRejection { rejection_class, helper_path, callsite_span, helper_body_span, detail }` surface, 10 sparse variants. (d) BLAS-helper rejection diagnostics — closed by W6 Task A with 6 BLAS-prefixed variants (`BlasMultipleRoots`, `BlasOutputPrecisionMismatch`, `BlasNotMatmulPattern`, `BlasNonLoadOperand`, `BlasInputPrecisionMismatch`, `BlasDimensionBindingFailure`) and a parallel `BlasSummaryAttempt::{NotEligible, Rejected}` enum mirroring the sparse path. The W5 P0 fix's previously-silent precision rejection now surfaces as a structured `BlasOutputPrecisionMismatch` / `BlasInputPrecisionMismatch` diagnostic for every non-F32 precision, verified by W7 across all 8 non-F32 `Prim` values. | **BLAS helpers closed for C and HIP:** `crates/chelis-cli/tests/cross_library_semantic_gap.rs` (8 tests) on `--target c` + `--target hip`; `crates/chelis-cli/tests/cross_library_semantic_gap_hip_gpu.rs` (GPU manual gate). **Sparse helpers closed for C and HIP:** `crates/chelis-cli/tests/cross_library_sparse_summaries.rs` (10 tests, C path) + `crates/chelis-ir/tests/host_sparse_summary.rs` (9 tests) + `crates/chelis-cli/tests/cross_library_sparse_hip_summaries.rs` (9 tests, HIP wrapper lock-tests). **Sparse rejection diagnostics closed by W4-A:** `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs` (28 tests = 20 original W4-A + 8 W6 BLAS extension) + `crates/chelis-ir/tests/host_sparse_summary_diagnostics.rs` (11 tests). **BLAS rejection diagnostics closed by W6 Task A:** `crates/chelis-ir/tests/host_blas_summary_diagnostics.rs` (17 tests) pattern-matches all 6 BLAS variants. **W5→W6 cross-product invariant locked by W7:** `crates/chelis-ir/tests/blas_rejection_cross_product_adversarial.rs` (11 tests) enumerates all 8 non-F32 precisions and asserts each produces a structured `BlasOutputPrecisionMismatch` rejection with the observed precision — zero silent fallthroughs. |
| **Perf-F1** | HIP batched matmul implementation quality | **Closed.** `hipblasSgemmStridedBatched` is now the default on uniformly strided batched HIP layouts; the per-batch helper loop is retained only as a fallback for broadcasted leading axes (`Expand` on the batch dim → stride-0) or otherwise non-uniform leading strides. | `crates/chelis-backend-hip/tests/perf_f1_strided_batched_default.rs` locks the strided-batched default with exact-line matching on uniform layouts and exact-line fallback matching on broadcasted leading axes (default workspace pass). The HIP manual GPU gate `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` (with `HSA_OVERRIDE_GFX_VERSION=11.5.1` and `LD_LIBRARY_PATH` per `docs/local_hip_environment.md`) covers numerical agreement, including `g15_hipblas_strided_batched_symbolic_batch_matches_eval` and `g15_hipblas_batched_matmul_matches_eval`. |
| **Perf-F2** | Post-BLAS allocator/fusion compounding | Normalize equivalent symbolic shape expressions for slot reuse and broaden in-place elementwise/fan-in fusion where aliasing permits. | **F2(a) DimExpr normalization v2 closed by W1-C:** in addition to v1 product/identity rules, `normalized_key()` now does Div numerator/denominator atom cancellation, GCD reduction on concrete factors, nested-Div flattening, and Mul × Div cross-term distribution. Positive and negative tests in `crates/chelis-ir/tests/dim_canonicalization.rs`. **F2(c) C scoped same-property `forall` / binder-equivalent aliases closed by W1-B:** C fused-elementwise in-place emit now admits same-binder, literal-equal, and named-equal-to-lit aliases via `binder_equivalent_tensor_type`. Positive and negative tests in `crates/chelis-backend-c/tests/fused_in_place_forall_alias.rs`. **F2(b) HIP in-place fused-elementwise closed by W2-B:** HIP backend now ports the C-side alias proof via a new `chelis-backend-hip::fusion` module mirroring `binder_equivalent_tensor_type` and `fused_in_place_spec`. At runtime, the wrapper guards on `chelis_gpu_is_contiguous` and aliases the FusedElem output view onto the reusable input's device buffer; the kernel parameter list drops `__restrict__` on the aliased external + output. Structural tests in `crates/chelis-backend-hip/tests/fused_in_place_forall_alias.rs` cover 11 cases (5 positive + 6 negative). GPU manual gate `gf3_fused_in_place_fan_in_gpu_matches_cpu` locks end-to-end numeric agreement on the local ROCm/HIP workstation. DimExpr canonicalization explicitly does *not* alpha-rename symbolic dims — a future scoped path must take same-binder aliases as explicit input. The `DimExpr` enum vocabulary stays `Concrete / Sym / Mul / Div`; a sum/add variant remains an open question parked here (see footnote below). |
| **Linearity-F1** | Brittle consume-site discrimination in linearity helpers | **Closed by PR #83** (0.7.8 compiler cleanup W1). Added `enum ConsumeKind { Aliasing, Structural }` field on `ConsumeSite` per Phase 0 (PR #79) Contract 1. Migrated all eight producer sites in `crates/chelis-types/src/linearity.rs` to set `kind` explicitly per the design-note table (L330/L482 → `Aliasing`; L519/L575/L1200/L1211/L1217/L1223 → `Structural`). Replaced the `descriptor.starts_with("binding ")` check at L726 with `matches!(site.kind, ConsumeKind::Aliasing)`. Sibling-sweep finding: one residual string-discrimination remains at `consume_var_expr` ~L894 for closure-capture-vs-match-scrutinee (both `Structural`); `kind` alone cannot distinguish — see `Linearity-ConsumeKindDiscrim-F1` below. | Fix at `crates/chelis-types/src/linearity.rs` (ConsumeSite, BindingState, eight producer sites, read_or_error). Diagnosis at `docs/investigations/linearity_typed_consumekind_diagnosis.md`. Regression tests at `crates/chelis-types/tests/linearity_typed_consumekind.rs` (4 fixtures exercising producer-kind discrimination). |
| **Linearity-F2** | Linearity false-negative on tuple-destructure programs | **Closed by PRs #83 + #90** (0.7.8 W1 + Wave 2 cascade). Detection closed by PR #83 via the local `tuple_get_element_type` helper (smaller blast radius than Phase 0's suggested desugar-time `inject_type_metadata` widening; tracked as `Linearity-DestructureTypeMeta-F1` follow-on). PR #83 emitted surfaced violations through `LinearityInfo::warnings` mirroring the F3 deprecation-window pattern. Wave 2 cascade (PR #90) surveyed the production corpus (88 files in `examples/` + `packages/`): **zero warnings surfaced**. Per the threshold rule (sub-floor with N<10, escalate-each-individually vacuously satisfied), cleanup was a no-op. PR #90 then removed `LinearityInfo::warnings` field, the `push_warning` helper, and the `warnings()` accessor (plumbing PR #83 reintroduced is now gone); renamed `destructure_warning_depth` → `destructure_scope_depth`; routed destructured-component use-after-consume through `errors`. Fixture 4 (`tuple_destructure_double_realize`) and Fixture 6 (`destructure_then_alias_consume`) flipped from warning assertions to error assertions. | Fix at `crates/chelis-types/src/linearity.rs::tuple_get_element_type` (PR #83) + channel removal + scope-depth rename (PR #90). Diagnosis at `docs/investigations/linearity_typed_consumekind_diagnosis.md` (PR #83) and `docs/investigations/linearity_destructure_cleanup_survey.md` (PR #90). Regression tests at `crates/chelis-types/tests/linearity_typed_consumekind.rs` and `linearity_aliased_consume.rs`. |
| **Vocabulary-F1** | Hand-maintained `CLOSED_TAGS` allowlist drifts from canonical Deep tag vocabulary | The lint's `CLOSED_TAGS` allowlist at `crates/chelis-lint/src/rules/deep_user_symbol_charset.rs:31-92` is a hand-maintained mirror of `VALID_TAGS` in `crates/chelis-deep/src/validate.rs:3-75`. The two lists drift independently: `t-ref` was canonical and emitted but missing from the lint, causing `chelis lint --check` to reject `chelis deep` output on every borrow type. PR #25 patched the immediate symptom. Closure: either (a) promote `VALID_TAGS` to `pub` and re-export from `chelis-deep`, consume from `chelis-lint`; or (b) generate `CLOSED_TAGS` at build time from `spec/03-deep-syntax.md` §2 markdown tables. Eliminates the hand-maintained mirror so future emitted-tag additions can't drift. | Two consumers today: `crates/chelis-lint/src/rules/deep_user_symbol_charset.rs:31-92` (lint) and `crates/chelis-deep/src/validate.rs:3-75` (canonical). Diagnosis at `docs/investigations/deep_compound_tag_allowlist_diagnosis.md`. |
| **Vocabulary-F2** | Hand-maintained ecosystem-name allowlist drifts from canonical reference | Open; PR #92 (0.7.8 W4-C) extended the allowlist by 18 entries (4 from hello-chelis report — `Linearity`, `Hypothesis`, `Integration`, `Optimize` — plus a math/ML sweep adding 14 more single-word PascalCase module-component names). Drift fixed for the current corpus. Structural closure (centralize the vocabulary in one Rust constant generated from `spec/01-nomenclature.md` §2.6 at build time, or expose §2.6's `module_prefix` column as a parseable manifest) remains pending — surface-when-next-drift. PR #24 patched the lint; PR #36 patched the spec; PR #92 patched both. | Consumer at `crates/chelis-lint/src/rules/module_pascal_components.rs::KNOWN_SINGLE_WORDS`. Canonical source at `spec/01-nomenclature.md` §2.6. Diagnosis at `docs/investigations/module_pascal_allowlist_diagnosis.md`. |
| **IR-FirstClassFn-F1** | DAG cannot represent first-class function values; `grad`/`vmap` outside immediate application paths reject | The RISC DAG is value-level (`RiscOp::Add/Mul/Load/...`); there is no `RiscOp::Closure` or function-table indirection. Forms like `let g = grad(f); use_g_elsewhere(g)` work today only because `lower_let` routes the binding through `LowerCtx::local_callables` and inlines at each direct application. The moment `g` flows through non-application paths (tuple component, ADT field, return value, function argument to a non-callable position), lowering hits `lower_unrepresentable` because there's no value-level representation. Closure: add `RiscOp::Closure { func_id, captures }` (or function-table indirection), and define backend lowering for it in C and HIP. Substantial IR refactor with backend implications. Surfaced as **G1** by the Item 2 sibling-sweep (`docs/investigations/item2_sibling_sweep_findings.md`). The user-facing trigger from the sweep (`let g = grad(f); g(x)`) actually works today via the existing inline-on-apply path; the gap is the broader first-class-function story. **Downstream consumer:** Phase D2 of `spec/design/differentiable_language.md` (ADT and record gradients + higher-order function AD) depends on this closure. Committing to the differentiable-language track reclassifies this entry from surface-when-forced to required prerequisite. **Non-blocking consumer:** the Hydronnx ONNX shell (`spec/design/hydronnx.md`) extends operator coverage to dynamic-graph ONNX operators (If, Loop, Scan) when this entry closes; Hydronnx v0.1 explicitly excludes those operators, so this is additive coverage, not a hard prerequisite. | `crates/chelis-ir/src/lower.rs::lower_grad` (and parallel `lower_vmap` paths) — the rejection sites for non-application contexts. No regression test for the broader gap today. |
| **IR-SelectOp-F1** | `lower_if` rejects non-float branches; needs typed `Select`/`Where` IR op or widened arithmetic op contracts | `lower_if` at `crates/chelis-ir/src/lower.rs::lower_if` uses masked arithmetic (`cond * a + (1 - cond) * b`) via `RiscOp::Mul/Add/Neg`, which the C backend assumes float-precision. For `if cond then int_a else int_b` (or non-float tensor analogs), lowering rejects because the masking strategy would produce ill-typed IR. Closure: either (a) add a dedicated `RiscOp::Select { cond, true_val, false_val }` with backend lowering in C and HIP (preferred — explicit semantics, no precision contract change to existing ops), or (b) widen `RiscOp::Mul/Add/Neg` type contracts across backends to accept non-float and ship the masking strategy uniformly. Surfaced as **G5** by the Item 2 sibling-sweep; classified there as "spec-blessed Phase 0 implementation limit, no user-visible blocker today" because host-lane already covers `chelis build`. **Downstream consumer:** Phase D1 of `spec/design/differentiable_language.md` (control-flow AD) depends on this closure with the dedicated `RiscOp::Select` shape; the differentiable-language spec pins option (a) explicitly. Committing to the differentiable-language track reclassifies this entry from surface-when-forced to required prerequisite. **Non-blocking consumer:** the Hydronnx ONNX shell (`spec/design/hydronnx.md`) extends operator coverage to ONNX `If` when this entry closes; Hydronnx v0.1 explicitly excludes that operator, so this is additive coverage. | `crates/chelis-ir/src/lower.rs::lower_if` (rejection site). No regression test today; trigger is `if cond then int_a else int_b` at eval. |
| **IR-MatchLowering-F1** | `lower_match` unconditionally rejects; DAG has no ADT/tag representation | `lower_match` at `crates/chelis-ir/src/lower.rs::lower_match` hits `lower_unrepresentable` for any `match` form. The RISC DAG has no tag/variant representation and no destructuring primitives — there is no value-level encoding of "value of ADT type Foo, tagged Bar, with payload (x, y)". Closure: requires either (a) a tag-compare-and-branch lowering analogous to IR-SelectOp-F1's `if` extension plus structural-destructuring primitives, or (b) a richer IR node family for ADTs (tag query, variant constructor, payload accessor). Bigger workstream than IR-SelectOp-F1 because it crosses both type-level and value-level representation. Surfaced as **G7** by the Item 2 sibling-sweep. Separate from the related G7-CLI silent-no-output sub-bug, which is being closed by the orchestrator-decided option (a) — stderr warning + exit 0 — in a small CLI PR. **Downstream consumer:** Phase D1 of `spec/design/differentiable_language.md` (control-flow AD) depends on this closure for `match` differentiation; Phase D2 (ADT and record gradients) depends on the tag/variant/payload representation choice this closure picks. Committing to the differentiable-language track reclassifies this entry from surface-when-forced to required prerequisite. **Non-blocking consumer:** the Hydronnx ONNX shell (`spec/design/hydronnx.md`) extends operator coverage to dynamic-graph ONNX operators (`Loop`, `Scan`) when this entry closes — both operators have payload-carrying body subgraphs that benefit from the destructuring primitives this closure introduces; Hydronnx v0.1 explicitly excludes them. | `crates/chelis-ir/src/lower.rs::lower_match` (rejection site). The IR-eval canary in `crates/chelis-ir/src/lower.rs` (around L4898 per sweep) locks the current "loud rejection" behavior. |
| **TypeCheck-PipeCast-F1** | `cast(type)` as bare pipe stage rejected by type-checker even after parser accepts it | **Closed** by the same pipe-stage parameter pre-unification fix as Finding 4 (0.7.6 red team, PR #51). `infer_pipe` now detects the synthesized single-unannotated-param lambda shape and infers the body with the parameter bound to the upstream pipe value's type, so `infer_cast` (and `infer_copy`) see a concrete tensor type instead of a fresh variable. Both `x \|> copy` and `x \|> cast(f32)` now type-check on statically-typed tensor inputs. | Fix in `crates/chelis-types/src/infer.rs::infer_pipe` + `synthesized_unary_lambda_param` + `infer_pipe_stage_lambda`. Diagnosis at `docs/investigations/pipe_copy_typecheck_diagnosis.md`. Regression tests at `crates/chelis-types/tests/pipe_copy_typecheck.rs` (3 fixtures: realize control, copy, cast). |
| **HostEval-ScalarFn-F1** | `result = go()` returns `0.0` when `def go -> f32 = 7.5` (scalar zero-arg user-def call) | **Closed by PR #80** (0.7.8 compiler cleanup W3). Symptom: `chelis eval --file` reports `result = 0.0` instead of `7.5`. Root cause was NOT in the host evaluator as the original closure-path text speculated — it was upstream in IR lowering at `crates/chelis-ir/src/lower.rs:2712`. `LowerCtx::lower_app`'s arity guard `if elems.len() < 4` rejected zero-arg `(app {meta} (var fn-name))` forms (3 elements: tag + meta + fn), emitting `RiscOp::Const { value: 0.0 }` before any callable resolution. `top_level_lowering_map` classified `result = go()` as lowered, so the host evaluator was never invoked. Fix: relaxed the guard to `if elems.len() < 3`; `&elems[3..]` yielding an empty slice is already handled by every downstream arm (`lower_builtin_app`, `try_lower_callable_app`/`lower_plain_callable_app`, fallback). Single-line guard change. PR #80 sibling sweep: 6 `elems.len() < N` guards in `lower.rs`; only `lower_app`'s had the off-by-one. `lower_def`/`let`/`fn`/`cast` correctly require 4; `lower_pipe` correctly uses `< 3`. Bug isolated. | Fix at `crates/chelis-ir/src/lower.rs:2712-2713` (the guard, with explanatory comment). Diagnosis at `docs/investigations/host_eval_scalar_fn_call_diagnosis.md`. Regression tests at `crates/chelis-cli/tests/host_eval_scalar_fn_call.rs` (5 fixtures: f32/f64/i64/bool zero-arg + one-arg negative control). |
| **Linearity-F3** | Linearity check silently skipped on module-wrapped top-level defs — production code has the safety check disabled | **Closed** by the two-PR Linearity-F3 workstream. PR 1 (#65, `fix/linearity-f3-pr1-warning-sweep`) extended the pre-declare loop and main walk in `check_linearity` / `check_linearity_with_context` to recurse through `(module {} name ...)` wrappers, and routed surfaced violations to `LinearityInfo::warnings` for a deprecation window; the full corpus sweep (`examples/`, `examples/illustrative/`, `packages/`, `crates/*/tests/`) found zero surfaced warnings. PR 2 (`fix/linearity-f3-pr2-errors`) flipped the severity: module-wrapped violations now route through `Checker::errors` and the JSON `errors` array, the `LinearityInfo::warnings` accessor and `Checker::in_module` flag are removed, and the CLI's `warning: linearity: ...` stderr emit is gone. The control-vs-module disparity is closed: `module Test; x = to_tensor([...]); y = realize(x); b = add(x, y)` now reports `score=0.8` with `UseAfterConsume` in `errors`, matching the bare-top-level case. | Fixes in `crates/chelis-types/src/linearity.rs` (`pre_declare_top_level_defs`, module recursion in `Checker::check_top_level`) and `crates/chelis-cli/src/main.rs::cmd_check_one`. Regression tests in `crates/chelis-types/tests/linearity_module_wrapped.rs` (six fixtures: three bare-top-level controls + three module-wrapped error assertions). Diagnoses at `docs/investigations/linearity_f3_module_skip_diagnosis.md` (PR 1) and `docs/investigations/linearity_f3_pr2_closeout.md` (PR 2). The PR 2 closeout note explicitly excludes `hello-chelis` from the in-repo sweep; its owning agent must re-run lint against the post-PR-2 binary. |
| **CBackend-CastMemcpy** | C backend's `emit_cast` is a bit-preserving `memcpy`, not real precision conversion — silent data corruption in cast outputs | `crates/chelis-backend-c/src/emit.rs::emit_cast` (around L2858 per the V2-F2 diagnosis) emits a `memcpy` from input to output buffer regardless of the source/target precision pair. Result: `cast(tensor[f32], f64)` via `chelis build --target c` produces output bytes that are the bit-pattern of input f32s reinterpreted as f64s — garbage values, not converted ones. Pre-existing — predates the 0.7.6 hygiene workstream. The IR-eval path was rejecting cast(Tensor) entirely (closed by PR #59, V2-F2), so the C-backend's quietly-wrong cast was never visible via cross-validation. HIGH severity — silent data corruption. Closure: replace the `memcpy` with element-wise per-precision conversion. Mirror PR #59's `cast_tensor_value` / `convert_scalar_data` semantics in C (e.g., `(double)src[i]` for f32→f64, `(float)src[i]` for f64→f32, `(int32_t)src[i]` for float→int). Surgical fix, single-function scope, no cascade risk. **Closed by PR #64.** | `crates/chelis-backend-c/src/emit.rs::emit_cast` around L2858 — the memcpy site. Mirror site at `crates/chelis-compiler-api/src/runtime.rs::eval_cast` + helpers from PR #59. Diagnosis at `docs/investigations/cbackend_cast_memcpy_diagnosis.md`. Regression tests at `crates/chelis-cli/tests/cbackend_cast_memcpy.rs`. |
| **CBackend-ReshapeMemcpy** | C backend's `append_tensor_reshape_helper` hard-codes `sizeof(float)` on a dtype-preserving reshape memcpy — drops upper half of f64/i64 elements | `crates/chelis-backend-c/src/host_emit.rs::append_tensor_reshape_helper` (around L276) emits `memcpy(dst, src, n * sizeof(float))` regardless of the tensor's actual precision. For any precision > 4 bytes (f64, i64, anything wider than f32), only the lower 4 bytes of each element are copied; the upper half is uninitialized garbage in the destination buffer. Same bug class as CBackend-CastMemcpy (PR #64) and surfaced by that fix's sibling sweep. Silent data corruption for any program that uses `reshape` on f64/i64 tensors through `chelis build --target c`. HIGH severity. Closure: replace `sizeof(float)` with the source tensor's precision-derived element size; mirror PR #64's element-size lookup pattern. Surgical fix, single-helper scope. **Closed by PR #67.** | `crates/chelis-backend-c/src/host_emit.rs::append_tensor_reshape_helper` around L276 — the memcpy site. Sibling pattern: `crates/chelis-backend-c/src/emit.rs::emit_cast` post-PR-#64. Diagnosis at `docs/investigations/cbackend_reshape_memcpy_diagnosis.md`. Regression tests at `crates/chelis-cli/tests/cbackend_reshape_memcpy.rs`. |
| **CBackend-PrintTensorF64** | `chelis_print_tensor_stdout` reads f64 (and other >4-byte precision) tensor buffers as `float*` — silent data corruption on print output | The C backend's print routine in `crates/chelis-runtime/src/lib.rs::chelis_print_tensor_stdout` (or equivalent — exact site to be confirmed by the fix PR's diagnosis) reads tensor data as `float*` regardless of the tensor's actual precision. For any precision > 4 bytes (f64, i64), the read consumes the wrong number of bytes per element and reinterprets the bit pattern incorrectly. Surfaced during PR #67's fixture work for CBackend-ReshapeMemcpy: the agent had to **patch the emitted kernel** in test fixtures to bypass this bug so the reshape fix could be validated. The print bug was masking the V2 sibling-sweep storage bugs (PRs #64, #67) because both the storage and the print were silently wrong in compensating ways. HIGH severity — third instance of the C-backend hard-coded-float precision-lookup bug class (after #64 cast, #67 reshape). Closure: replace the `float*` read with a precision-dispatched read (switch on `tensor->dtype` selecting `float`/`double`/`int32_t`/`int64_t`/etc.). Surgical fix, single-function scope. **Closed by PR #72.** | `crates/chelis-backend-c/src/host_emit.rs:221` — the print loop's dtype-typed read after PR #72. Diagnosis at `docs/investigations/cbackend_print_tensor_f64_diagnosis.md`. Regression tests at `crates/chelis-cli/tests/cbackend_print_tensor_f64.rs` (3 fixtures: f32 control, f64, i64). PR #72 also surfaced a separate same-class bug: `cast(int32 → int64)` produces wrong values because the runtime stores int32 as f32 bit patterns; bundled with the foundational CRuntime-F32Coupling entry below. |
| **Linearity-AliasedConsume-F1** | Var-RHS aliasing tracks alias-name linearity state, not underlying value — silent linearity bypass on consume sites | **Closed by PR #83** (0.7.8 W1). Added `LinearScope.aliases: HashMap<String, Vec<Option<String>>>` parallel to existing `bindings` and `types` stacks. When `check_def_body` or `check_let` records an `Aliasing` consume on `let y = (var x)`, it also records `y → x` in the alias map. `consume_var_expr` forwards `Structural` consumes through `resolve_alias_chain` to the underlying source name's scope entry; aliasing consumes don't forward (they only update the alias's own entry). Multi-level chains (`let z = y; let y = x; consume(z) → x consumed`) work via chain walking with a cycle guard. The implicit-Copy IR pass still preserves runtime correctness; this closes the spec-level safety check. | Fix at `crates/chelis-types/src/linearity.rs` (LinearScope.aliases, record_alias, resolve_alias_chain, consume_var_expr forwarding). Diagnosis at `docs/investigations/linearity_typed_consumekind_diagnosis.md`. Regression tests at `crates/chelis-types/tests/linearity_aliased_consume.rs` (fixtures 3 multi-level bypass + 6 mixed alias-destructure). |
| **SurfDecompile-PascalLowercase-F1** | `chelis surf` decompiler loses compound PascalCase in module names; `HelloTensor` → `Hellotensor` after `deep → surf` round-trip | `module HelloTensor` desugars to Deep `hellotensor` (via `lower_module_path` lowercasing per the case-split rule §1.1), and `chelis surf` re-Title-cases only the leading character on the way back, producing `module Hellotensor`. The decompiled output fails the workstream-shipped `module-pascal-components` (§6.3) lint, directly violating the `CLAUDE.md` invariant: *"decompiler output must round-trip through the supported parser path"*. Surfaced by V3 final red-team (NEW-F2, PR #70). LOW severity — cosmetic on the surf layer, but breaks the round-trip claim. Closure: bundle with Vocabulary-F2 — preserve module-component compound case in Deep metadata (or stash it in a Deep `meta` field) so the decompiler can restore the original capitalization. | Diagnosis in `docs/investigations/hygiene_redteam_0_7_6_v3.md` (V3 NEW-F2 section). Source-line anchor at `crates/chelis-deep/src/decompile.rs::module_name_to_surf` (or wherever `chelis surf` re-Title-cases module path components). No regression test today. |
| **CRuntime-F32Coupling** | C runtime's `chelis_tensor.data` is typed `*mut f32` regardless of the tensor's actual precision — foundational mismatch behind every C-backend silent-data-corruption bug found in the 0.7.6 workstream | **Closed by PRs #84 + #86 + #87 + #88** (0.7.8 W2 series, four-PR migration). PR #84 (Agent A architectural piece): introduced `TensorElement` trait per Phase 0 (PR #79) Contract 2 with checked `data_ptr`, unchecked `data_ptr_unchecked` (debug_assert), and default `fill`. Impl blocks for `f32, f64, i32, i64` (bool excluded per orchestrator decision; routes through `f32::data_ptr` internally). Promoted `CHELIS_*` constants to `pub const`. Changed `chelis_tensor.data` from `*mut f32` to `*mut u8`. Migrated 2 anchor ops. PR #86 (Agent B PR 2): migrated 31 runtime call sites across 14 ops; matrix expanded 17 → 60. PR #87 (Agent B PR 3): migrated 6 host_emit code-generation sites; 11 host_emit dispatch fixtures added. PR #88 (Agent B PR 4): 22 multi-op composition fixtures (17 runtime-accessor chains + 5 cbackend cast+arithmetic compositions); workstream-wide sibling sweep audit; PR-3-surfaced f64-cast bug verified as no-repro on current main (it was PR 3's own in-flight fix observed through a `git stash`). **Total: 110 dtype-coupling fixtures locked across the workstream.** Sibling-sweep audit: 11 intentional `*mut f32` references remain in `crates/chelis-runtime/` (each enumerated with justification); 0 in `crates/chelis-backend-c/`; 0 in HIP/Metal/IR. PR #88 retained `data_as_f32` and `data_as_f32_const` routes for I32, Bool, and int32 indices. The native-int32 change removed all I32 routes. Current non-F32 helper consumers use the `BoolInBinary32` payload. `CRuntime-BoolStorage-F1` and `CRuntime-I8I16-F1` remain open. | Code: `crates/chelis-runtime/src/lib.rs`, `crates/chelis-backend-c/src/host_emit.rs`. Tests: `crates/chelis-e2e/tests/dtype_op_matrix.rs` (77 fixtures), `crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs` (11), `crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs` (5), plus the four 0.7.6 surface-fix regression locks (`cbackend_cast_memcpy.rs`, `cbackend_reshape_memcpy.rs`, `cbackend_print_tensor_f64.rs`). Diagnoses: `docs/investigations/c_runtime_dtype_accessors_diagnosis.md`, `docs/investigations/c_runtime_dtype_multiop_cast_diagnosis.md`, `docs/investigations/c_runtime_dtype_coupling_workstream_audit.md`. |
| **Linearity-ConsumeKindDiscrim-F1** | Residual string-discrimination at `consume_var_expr` for closure-capture-vs-match-scrutinee | Surfaced by PR #83's sibling sweep. Both closure-capture and match-scrutinee consumes are `ConsumeKind::Structural`; the typed `kind` field alone cannot distinguish them. One residual `description.contains(...)` lookup remains at `crates/chelis-types/src/linearity.rs::consume_var_expr` (~L894 per PR #83's diagnosis) discriminating these two flavors of Structural consume for a specific borrow-after-consume disposition. LOW severity — works correctly today, but the typed-`ConsumeKind`-refactor goal was to remove all string discrimination. Closure: add a finer discrimination axis (e.g., `enum ConsumeKind { Aliasing, Structural { source: StructuralSource } }` with `StructuralSource::{Realize, App, Pipe, ClosureCapture, MatchScrutinee, …}`), or accept the residual lookup as the discrimination axis for a niche case and document it. Dispatch when a future change has reason to differentiate other Structural sub-kinds. | Source-line anchor at `crates/chelis-types/src/linearity.rs::consume_var_expr` ~L894 (post-PR #83). Diagnosis in `docs/investigations/linearity_typed_consumekind_diagnosis.md` ("Sibling sweep findings" section). No regression test today; existing linearity tests don't exercise this discrimination axis directly. |
| **Linearity-DestructureTypeMeta-F1** | Tuple-destructure components have no desugar-time type metadata; W1 used a local checker-side helper instead | Surfaced as a structural choice during PR #83. The Phase 0 design note (PR #79 Contract 1) suggested propagating type metadata onto desugar-synthesized `__chelis_tmp_N` bindings via `inject_type_metadata`. W1 instead implemented a local `tuple_get_element_type` helper in `crates/chelis-types/src/linearity.rs` that indexes into the underlying tuple var's `t-tuple` scope-type at linearity-check time, avoiding the broader desugar-time widening into `annotate_let_children` / `should_attach_type_metadata`. This closes the linearity false-negative (Linearity-F2) with smaller blast radius, but other downstream consumers (type inference, IR lowering) may still see untyped destructure components. LOW-MEDIUM severity — no known user-visible bug, but a latent gap. Closure: when a downstream consumer hits an untyped destructure component, widen `inject_type_metadata` at desugar time or thread types through `destructure_pattern`'s signature. Dispatch when surface-when-bug-fires. | Source anchor at `crates/chelis-surf/src/desugar.rs::destructure_pattern` (L1135-1156, the synthesis site that doesn't call `inject_type_metadata`). Workaround at `crates/chelis-types/src/linearity.rs::tuple_get_element_type`. Diagnosis at `docs/investigations/linearity_typed_consumekind_diagnosis.md` ("Section 2: Linearity-F2 design choice"). |
| **CRuntime-I32Storage-F1** | **Closed.** `CHELIS_I32` storage uses four-byte native two's complement. | The base writers already stored native int32 values. Seven runtime consumers still used an f32 view. This change corrected `cmplt`, `where`, scatter-add, `cumsum`, `trace`, `clamp`, and `einsum`. The C host emitter now uses `int32_t` for generated int32 element access. The f32 boundary rejects int32 in debug builds. | Code: `crates/chelis-runtime/src/lib.rs` and `crates/chelis-backend-c/src/host_emit.rs`. Tests: `crates/chelis-runtime/tests/i32_native_decode.rs` and `crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs`. Mutation evidence: `openspec/changes/complete-int32-native-decode/mutation-evidence.md`. |
| **CRuntime-BoolStorage-F1** | bool stored as 4-byte f32-encoded today, not native 1-byte bool | Same shape as `CRuntime-I32Storage-F1` but for `CHELIS_BOOL`. `chelis_alloc` returns a 4-byte buffer for CHELIS_BOOL; the runtime writes `1.0f32`/`0.0f32`. Per the orchestrator decision on PR #79's Phase 0 (https://github.com/Chelis-Lang/chelis/pull/79#issuecomment-4434524160), the TensorElement trait deliberately omits a `bool` impl in 0.7.8; bool sites route through `f32::data_ptr` internally to match current runtime. LOW-MEDIUM severity — works correctly with the f32 detour, but propagates the f32-coupling pattern this workstream is trying to remove. Closure: migrate bool storage to native 1-byte u8 in `chelis_alloc`, every bool-write site (`chelis_tensor_cmplt` at L1897, `chelis_pad_sequences` at L1660-1664), and every bool-read site. Then add the `bool` TensorElement impl. Keep this work separate from the closed `CRuntime-I32Storage-F1` change. | Anchor at `crates/chelis-runtime/src/lib.rs::chelis_alloc` arm for CHELIS_BOOL (~L418-422 per Phase 0 design note). All bool-write/read sites enumerated by grep for `data` + `bool`/`cmplt`/`where`/`pad_sequences`. Diagnosis at `docs/investigations/c_runtime_dtype_accessors_diagnosis.md`. No regression test that exercises native-bool-storage today; PR #84's matrix tests bool via the f32 shim. |
| **CRuntime-I8I16-F1** | i8 and i16 precision storage is not represented in the runtime today | Surfaced by Phase 0 PR #79's discovery (orchestrator decision documented at https://github.com/Chelis-Lang/chelis/pull/79#issuecomment-4434524160). The 0.7.8 plan brief listed `f32, f64, i8, i16, i32, i64, bool` as supported precisions, but discovery showed `CHELIS_I8` and `CHELIS_I16` constants don't exist, `chelis_alloc` has no element-size arms for them, and no call site references them. Deferred per Phase 0 decision (option A: ship only f32/f64/i32/i64/bool in TensorElement trait; file §5 follow-on). LOW severity today — no program shape forces it. Closure: when a program shape requires i8 or i16 precision, add `CHELIS_I8` and `CHELIS_I16` constants, `chelis_alloc` element-size arms, TensorElement impls, and (per `CRuntime-I32Storage-F1` discovery) decide whether to use native byte/halfword storage from day one. | No anchor today; the constants and storage paths don't exist. The Phase 0 design note (`docs/design/compiler_cleanup_0_7_8_spec_lock.md` §"Open questions for orchestrator decision" → Q1) flags this. Dispatch when surface-when-bug-fires. |
| **Lint-ExceptionPathRoot-F1** | Workspace-rooted exception patterns in `chelis-lint` fail to match when CLI walks sub-directories | **Closed by PR #108** (0.7.9 Workstream LE). Added a `workspace_root` parameter to `apply_exceptions` and `is_excepted`, threaded through 4 call sites (`cmd_lint`, `apply_lint_fixes`, `emit_advisory_lint_warnings_for_file`, `style_gate::run_lint_for_single_file`). Anchored exception prefix-stripping against the workspace root instead of the per-target walk root. CLI computes the workspace root via `detect_lint_workspace_root` by canonicalizing CWD at the CLI boundary, reusing PR #93's pattern — NO walk-up filesystem traversal (per `feedback_no_walkup_filesystem_detection.md`). Both `chelis lint --check .` and `chelis lint --check crates docs examples packages` now produce identical output with zero false-positive `surf-def-arrow-form` errors. | Fix at `crates/chelis-lint/src/exceptions.rs::is_excepted` + 4 call sites in `chelis-cli`. Diagnosis at `docs/investigations/lint_exception_path_root_diagnosis.md`. Regression tests at `crates/chelis-cli/tests/lint_path_walk_consistency.rs`. |
| **Linearity-ShapeABroadReturn-F1** | Implicit-copy inserter Shape A v3 covers only bare-var return; `let`/`if`/`match` tail-position borrow-to-owned coercions still fail | **Closed by PR #109** (0.7.9 Workstream SR). Added a private helper `descend_to_tail_var(expr) -> Option<&str>` in `crates/chelis-types/src/infer.rs` (immediately after `shape_a_relaxed_return`) that walks `(var x)` (leaf), `(let bind body)`, `(if cond then_e else_e)`, and `(match scrutinee arm ...)`. Returns `Some(name)` only when every sibling branch resolves to the same bare-var name (preserves PR #91's lifetime justification). Integration: the bare-var-only `get_tag(inner_list) == Some("var")` gate becomes `descend_to_tail_var(inner).is_some()`. Structural type-equality check and relaxed-type construction unchanged — the relaxation is no more permissive on the *type* axis, only on the *body-shape* axis. Match-tail reaches today (no `#[ignore]` needed). Flipped 2 red-team fixtures from `expect_err` to `expect_ok`. | Fix at `crates/chelis-types/src/infer.rs` (`descend_to_tail_var` + integration in `check_top_level`). Diagnosis at `docs/investigations/implicit_copy_shape_a_broader_return.md`. Regression tests at `crates/chelis-ir/tests/implicit_copy_shape_a_broader_return.rs` (7 fixtures: let-tail, if-tail, match-tail, mixed nesting, positive bare-var control, negative type-mismatch control) + 2 flipped fixtures in `crates/chelis-ir/tests/implicit_copy_fanout_shape_a_adversarial.rs`. |
| **Lint-AutofixCopyOnBorrow-F1** | `redundant-linearity-call` autofix could strip `copy()` on a `&T` borrow → type mismatch (filed by Nautilus against 0.7.7) | **Verified Closed by 0.7.8** (PRs #83 + #91 + #95). The Path 1B typed-pipeline-accepts gate at `crates/chelis-cli/src/main.rs::fix_would_apply_for_violation` correctly drops the `[fix]` marker when stripping `copy()` would leave the program with a borrow→owned type mismatch the implicit-copy inserter can't bridge. Verified on current main (post-#95): `def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(y))` emits the `redundant-linearity-call` advisory WITHOUT a `[fix]` marker, and `chelis lint --fix` does NOT strip it. Coverage extended in PR #95 with 5 new fixtures (F5–F9) including Shape A bare/driver and Shape B grad/vmap. Closure verified by 0.7.8 terminal red-team adversarial probes. | Verification at `crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (9 fixtures, all strip cleanly post-Item-1-v3) and `crates/chelis-cli/tests/cli.rs::lint_fix_redundant_linearity_call_keeps_when_typed_pipeline_rejects` (negative case). Gate logic at `crates/chelis-cli/src/main.rs::typed_pipeline_accepts_surf`. |
| **Lint-AutofixPipeRewrite2ArgOuter-F1** | `prefer-pipe-operator` autofix could produce broken 5-segment pipe for `f(g(a), h(b))` (filed by Coral against 0.7.7) | **Verified Closed by 0.7.8** (PR #92's W4-C lint precision bundle preceded by the broader Path 1B typed-pipeline gate in PR #95). Coral's report against 0.7.7: `sub(hamt_size(next_child), hamt_size(child))` auto-rewrote to `next_child |> hamt_size |> child |> hamt_size |> sub` (treating `child` as a function-position value, broken). Verified on current main (post-#95): the same input rewrites to `next_child |> hamt_size |> sub(hamt_size(child))` — three segments with parens preserved around the inner call; `chelis check` on the autofix output returns `score=1, 0 errors`. The 2-arg outer case is correctly handled. | Verification: write the Coral repro shape to a fixture, run `chelis lint --fix`, run `chelis check` — passes clean. Path 1B gate at `crates/chelis-cli/src/main.rs::typed_pipeline_accepts_surf` is the safety net. |
| **Runtime-EmptyTensorNumel-F1** | `numel(to_tensor([]))` returns `1` instead of `0` (filed by Coral against 0.7.7) | Reproduces on current main: `chelis eval --file` on `result = numel(to_tensor([]))` outputs `1`. Upstream root cause of Coral red-team findings F2 (`filter` on empty result crash), F3 (`head`/`tail` empty crash), F4 (`slice` empty crash), and mixed-type empty-frame `from_pairs` length-check failures plus inner-join-no-overlap crashes. Probable root cause: `to_tensor` for an empty list literal produces a scalar shape `[]` instead of rank-1 shape `[0]` (numel of `[]` correctly returns `1`, numel of `[0]` correctly returns `0`). MEDIUM severity — silently wrong size, surfaces as downstream crashes. Closure: fix `to_tensor` shape inference for empty input. Targeted for 0.7.8 (PR pending). | Probable anchor at `crates/chelis-types/src/infer.rs` (list-literal type inference) or `crates/chelis-ir/src/host.rs::to_tensor` host evaluation. Coral workaround in downstream: switch `column_len`/`row_count` to `len(to_list(xs))` and short-circuit empty paths in `filter`/`slice`. Diagnosis pending. |
| **Lint-PreferPipeRecursive-F1** | `prefer-pipe-operator` rewrites just shift the warning to the inner call (filed by Nautilus against 0.7.7) | **Verified Not Reproducible on 0.7.8 or later** (0.7.9 Workstream LP survey). On current main (post-0.7.8 release), basic nested-call shapes do not exhibit the warn-on-inner-call recursion: e.g., `def outer(x: i32) -> i32 = f(g(x), 3)` fires the warning, applies the fix to `x |> g |> f(3)`, and re-running lint produces no warning. The rule's `find_pipe_candidates()` only flags direct first-argument nesting, and the `candidate_is_fmt_clean` gate at line 127 may suppress the recursive shape for typical line-length / stage-count combinations. The Nautilus 0.7.7 report described a deeper pattern (multi-level nesting where inner calls re-trigger), which may have been: (a) closed incidentally by 0.7.8 work (most likely), (b) require a specific nesting depth + line-length combination that avoids the fmt-clean filter, or (c) be a 0.7.7-only regression now resolved. Closure: awaiting downstream repro to determine if a specific-shape variant exists. If a concrete repro surfaces, reopen with the failing fixture cited. | Anchor at `crates/chelis-lint/src/rules/prefer_pipe_operator.rs` `find_pipe_candidates` + `candidate_is_fmt_clean`. 0.7.9 survey verified no-repro on basic shapes; survey result at `docs/investigations/redundant_linearity_call_precision_0_7_9.md` (sibling-sweep section). |
| **Lint-PreferPipeRedundantLinearityPair-F1** | `prefer-pipe-operator` + `redundant-linearity-call` interact to flag any rewrite shape as un-satisfiable (filed by Coral against 0.7.7) | **Closed by PR #107** (0.7.9 Workstream LP, bundled with `Lint-RedundantLinearityCopyOnBorrowWarn-F1`). Subsumed by the `check_mirrors_fix=true` opt-in that lifts the typed-pipeline-accepts gate from autofix-marker decision to warning emission. `… \|> drop(N) \|> f(…)` no longer triggers a false-positive `redundant-linearity-call` because the typed pipeline rejects the strip — the `drop(N)` is a 2-arg list primitive whose stripped form fails type-check, so the warning is suppressed structurally. Symbol resolution not needed; the typed pipeline discriminates structurally. Fixture pinned by F12 in `crates/chelis-lint/tests/redundant_linearity_call_autofix.rs`. | Fix at `crates/chelis-lint/src/rules/redundant_linearity_call.rs` (opt into `check_mirrors_fix=true`). Diagnosis at `docs/investigations/redundant_linearity_call_precision_0_7_9.md`. Regression tests at `crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (F12 pins the 2-arg primitive case). |
| **TypeCheck-NeqBorrowOverload-F1** | `neq` on `&tensor` ref doesn't resolve to a `tensor[n, bool]`-returning overload despite `copy()` being flagged as redundant (filed by Coral, possibly intentional) | Coral observation: a `&tensor` value cannot use `neq` to produce a bool-tensor result; the lint says `copy()` on the ref is redundant; net effect contradicts is-nan-shaped pattern code. Filed as "possibly intentional rather than a bug" — may be a documented type-overload constraint. LOW severity. Closure: clarify whether `neq` is intended to support `&tensor` operands, or document the workaround (force owned via `realize`). | Coral's downstream workaround: O(n) host-list. Spec reference: `spec/00-*` or `spec/01-*` for type-overload rules on `&T`. Not verified on current main. |
| **Lint-RedundantLinearityCopyOnBorrowWarn-F1** | `redundant-linearity-call` lint emits advisory warning on `copy(&T)` even though autofix gate correctly drops the `[fix]` marker (filed by Nautilus against 0.7.8) | **Closed by PR #107** (0.7.9 Workstream LP). Nautilus reported 319 false-positive warnings in `src/linalg.ch` after 0.7.8 release: the W5/Path 1B autofix gate correctly suppressed the `[fix]` marker on `copy(borrow)` but the warning itself continued to fire as advisory noise. Fix: opt the rule into `check_mirrors_fix=true` (mirroring `prefer-pipe-operator`'s V2-F3 closure in PR #58). The CLI driver's existing `should_suppress_unfixable_violation` at `crates/chelis-cli/src/main.rs` now runs the typed-pipeline gate at warning-emit time too. If the stripped form fails type-check, the warning is suppressed; the `copy()` is structurally necessary for the borrow→owned coercion, not migration-compat noise. No CLI driver code change needed — only the rule-level opt-in. Single unit test in the rule and 4 new integration fixtures (F10–F13) in `crates/chelis-lint/tests/redundant_linearity_call_autofix.rs`. **Note: the 0.7.9 terminal red-team (PR #113) found this fix only reached the `chelis lint --check` path, not `chelis check` — see `Lint-CheckMirrorsFixAdvisoryEmitLeak-F1` below, closed by PR #114.** | Fix at `crates/chelis-lint/src/rules/redundant_linearity_call.rs::check_mirrors_fix()` + `crates/chelis-lint/src/lib.rs` doc-comment. Diagnosis at `docs/investigations/redundant_linearity_call_precision_0_7_9.md`. Regression tests at `crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (F10 pins the borrow-arg shape from Nautilus's report). |
| **Lint-CheckMirrorsFixAdvisoryEmitLeak-F1** | `chelis check`'s advisory-warning emit path didn't run the typed-pipeline suppression gate, so the PR #107 `check_mirrors_fix` fix only reached `chelis lint --check` — `chelis check` (the customer's actual build path) still flooded | **Closed by PR #114** (0.7.9 fixup wave, FX-CLI). Surfaced by the 0.7.9 terminal red-team (PR #113, finding LP-LEAK-A/B). `emit_advisory_lint_warnings_for_file` (the path `chelis check` invokes) applied only the path-glob exception filter, never `should_suppress_unfixable_violation` — which `cmd_lint` (`chelis lint --check`) does apply. Corpus sweep: 34 false positives in `decimal.ch`, 20 in `tokenizer.ch` via `chelis check`, 0 via `chelis lint --check`. Since `chelis reef build` invokes `chelis check` on `.ch` files, Nautilus's flood persisted despite PR #107. LP-LEAK-B is the same root cause affecting `prefer-pipe-operator`; one fix closes both. Fix: threaded `should_suppress_unfixable_violation` into the advisory-emit path so both `chelis check` and `chelis lint --check` gate identically for `check_mirrors_fix=true` rules. Sibling sweep: style gate's `run_lint_for_single_file` uses `all_rules()` which excludes the two `check_mirrors_fix` rules — cannot leak; `emit_advisory_lint_warnings_for_file` was the single leak surface. | Fix at `crates/chelis-cli/src/main.rs::emit_advisory_lint_warnings_for_file`. Diagnosis at `docs/investigations/terminal_redteam_0_7_9.md` (finding LP-LEAK-A/B). Regression tests at `crates/chelis-cli/tests/red_team_0_7_9.rs` (`lp_leak_a`, `lp_leak_b` flipped from `#[ignore]` to active, plus a positive control). |
| **Lint-WorkspaceRootCwdAssumption-F1** | `detect_lint_workspace_root` was `canonicalize(cwd)` with no actual workspace probe — running `chelis lint --check` from a subdirectory or via absolute path from outside the workspace re-surfaced the false positives PR #108 claimed to close | **Closed by PR #114** (0.7.9 fixup wave, FX-CLI). Surfaced by the 0.7.9 terminal red-team (PR #113, finding LE-LEAK-A). PR #108's `detect_lint_workspace_root` (`crates/chelis-cli/src/main.rs`) just canonicalized the process CWD — so when CWD wasn't the workspace root, exception matching anchored against the wrong directory and the 6 `surf-def-arrow-form` false positives reappeared. Fix: replaced with `cargo locate-project --workspace --message-format plain` (Cargo's own canonical workspace probe — NOT a hand-rolled filesystem walk-up, per `feedback_no_walkup_filesystem_detection.md`), run with its working directory set to the lint target's directory rather than the process CWD so absolute-path invocations from outside the workspace resolve correctly. Moved the function to `style_gate.rs`, the shared home of the exception list; it now serves all three exception-anchor sites (`cmd_lint`, `emit_advisory_lint_warnings_for_file`, `style_gate::run_lint_for_single_file`). On detection failure the three callers degrade to unfiltered output (no workspace ⟹ no workspace-rooted exception can apply) rather than hard-failing — documented decision, avoids regressing `chelis lint --check /tmp/loose_file.ch`. | Fix at `crates/chelis-cli/src/style_gate.rs::detect_lint_workspace_root` + 3 call sites. Diagnosis at `docs/investigations/terminal_redteam_0_7_9.md` (finding LE-LEAK-A). Regression tests at `crates/chelis-cli/tests/lint_path_walk_consistency.rs` + `red_team_0_7_9.rs` (`le_leak_a` flipped active). |
| **TypeCheck-FreeDimVarUnification-F1** | Type checker accepts a def whose declared return type and body type differ in dimension identity but match in rank — passes `chelis check` with `score=1`, then evaluates with a runtime shape mismatch (HIGH-severity silent miscompilation) | **Closed by PR #115** (0.7.9 fixup wave, FX-SR). Surfaced by the 0.7.9 terminal red-team (PR #113, finding SR-LEAK-A) — pre-existing soundness gap, latent since PR #91 (0.7.8 Shape A bare-var path), reach widened by PR #109 (Shape A let/if/match tail-var path). The bug had **two independent paths**: (A) `types_structurally_equal` (`crates/chelis-types/src/infer.rs`) compared tensor dims by rank only (`d1.len() == d2.len()`), so the Shape A relaxed-retry guard accepted `tensor[n]` vs `tensor[m]`; (B) plain owned-tensor bodies (`def g[n,m](x: tensor[n], y: tensor[m]) -> tensor[n] = y`) bypass the relaxed-retry entirely — the post-body sig-unify freely collapsed `n := m` via `unify_dim`. Fix: (A) added `dims_identical` so `types_structurally_equal` compares tensor dims position-wise — two dims match iff same `Dim::Name`, same `Dim::Lit`, or the same `Dim::Var`; (B) added `check_declared_dvars_rigid` (folding in the existing `Var→Lit` self-pin check) that flags `Var→Var` collapse — two distinct declared dim params resolving equal — as `DimensionMismatch`, run from `infer_fn` and from the defsig site after post-body sig-unify. Spec completion: `spec/04-type-system.md` §4.4 now states declared dimension parameters are rigid within the def body. | Fix at `crates/chelis-types/src/infer.rs` (`dims_identical`, `types_structurally_equal`, `check_declared_dvars_rigid`). Diagnosis at `docs/investigations/typecheck_dim_identity_diagnosis.md`. Regression tests at `crates/chelis-cli/tests/red_team_0_7_9.rs` (`sr_leak_a_*` — 7 fixtures covering both paths, same-dim positives + divergent-dim negatives; the red-team's `#[ignore]`'d fixture flipped active). Spec: `spec/04-type-system.md` §4.4. |
| **Backend-C-Bf16F16-Admission** | C backend rejected bf16/f16 at every op surface even though HIP and Metal admitted them; the runtime reserved `CHELIS_BF16=5` / `CHELIS_F16=6` dtype tags for storage but had no arithmetic path | **Closed by PR #192** (WS-1, dtype + Metal cleanup cycle). C backend now admits bf16/f16 at every op surface: `elem_type` maps both to `uint16_t` storage; elementwise/unary kernels load via `chelis_bf16_to_f32` / `chelis_f16_to_f32`, compute in f32, store via the inverse helpers; reductions use a true f32 accumulator per spec §5.7.1 (the `accumulator` field on `RiscOp::Sum` is consumed, not destructured-defaulted); `emit_const` computes the bf16/f16 bit pattern at codegen time via `half::bf16::from_f64(v).to_bits()` and emits `chelis_fill_bf16(t{id}, 0xXXXXu)`; matmul follows the convert-then-cblas_sgemm path of spec §5.7.1 (allocate f32 scratch `A'`/`B'`/`C'`, populate via `chelis_bf16_buffer_to_f32`, dispatch `cblas_sgemm`, downcast `C'` to the result tensor); CLI gate `c_backend_supports_precision` admits both; spec §1.1.3 matrix C-row flipped from "rejected (deferred)" to "admitted". `sparse_elem_type`'s silent `_ => "float"` default-arm footgun and `emit_const`'s `_ => chelis_fill_f32(.., value as f32)` truncation default-arm are gone — both replaced with explicit per-dtype arms + panic on truly unsupported. | Behavior change at `crates/chelis-backend-c/src/emit.rs` (`elem_type`, `emit_const`, `scalar_zero_literal`, `fill_zero_call`, matmul wrapper at `validate_supported_precisions`), `crates/chelis-backend-c/src/host_emit.rs` (`sparse_elem_type`, dtype macro dispatch), `crates/chelis-cli/src/main.rs` (CLI gate). Runtime helpers at `crates/chelis-runtime/include/chelis_runtime.h` (`chelis_bf16_to_f32`/`chelis_f32_to_bf16`/`chelis_f16_to_f32`/`chelis_f32_to_f16`, `chelis_fill_bf16`/`chelis_fill_f16`, buffer conversion helpers, `chelis_dtype_size` arms for `CHELIS_BF16`/`CHELIS_F16`). Spec at `spec/04-type-system.md` §1.1.3 and `spec/08-backends.md` C-row. Acceptance oracle tests at `crates/chelis-backend-c/tests/dtype_matrix_bf16_f16.rs` (compile-and-run vs evaluator within `BF16_TOL=1e-2`/`F16_TOL=1e-3`; named `bf16_*_agrees_with_evaluator`, `bf16_reduce_sum_uses_f32_accumulator_per_spec_5_7_1`, `bf16_const_fill_produces_exact_bit_pattern`, `bf16_matmul_routes_through_convert_then_sgemm`) and CLI gate tests at `crates/chelis-cli/tests/cli_admits_bf16_f16_target_c.rs`. |
| **Backend-Metal-EmitConstBf16F16-HostFill** | Metal `emit_const` for f16/bf16 host-fill emitted host C++ that referenced the kernel-only MSL types `half` and `bfloat` in `sizeof(...)` and `(half*)` casts, so any f16/bf16 Const-rooted Metal program failed to compile under `clang++ -fobjc-arc` | **Closed by PR #191** (WS-2, dtype + Metal cleanup cycle). `emit_const` now factors host fills through the new `dtype::host_const_fill_body(prec, value, buf, n)` helper, which routes every active Metal dtype (F32, F16, Bf16, Int8, Int16, Int32, Int64, Bool) to a host-safe storage type. F16/Bf16 specifically use `uint16_t` storage with an IEEE-754 bit-pattern literal computed at codegen time via `half::f16::from_f64(value).to_bits()` / `half::bf16::from_f64(value).to_bits()`; `sizeof(msl_ty)` was replaced with `dtype::host_sizeof_expr(precision)`. F64 panics defensively (Metal hard-rejects f64 per spec §1.1.3 — IR validation already rejects upstream). | Fix at `crates/chelis-backend-metal/src/emit.rs::emit_const` and `crates/chelis-backend-metal/src/dtype.rs` (`host_const_fill_body`, `host_sizeof_expr`). Linux structural test at `crates/chelis-backend-metal/tests/codegen_structure.rs` asserts emitted host source uses `uint16_t` + bit-pattern literal and contains neither `(half*)` nor `sizeof(half)` nor `(bfloat*)` nor `sizeof(bfloat)`. macOS compile gate at `crates/chelis-backend-metal/tests/gpu_correctness.rs` invokes `xcrun clang++ -fobjc-arc` and asserts exact bit patterns (f16(2.5)=`0x4100`, bf16(2.5)=`0x4020`, plus 1.5 / -1.0 / 0.0 sweep). |
| **Test-Infra-RuntimeStaticLib-ColdCacheRace** | `ensure_runtime_static_lib` test helper across 6 test files used a shared `<canonical>.a.tmp` filename when materializing the chelis-runtime static archive; concurrent test binaries running in parallel under `cargo test` collided on the `fs::rename` step and one process saw ENOENT on cold-cache CI runs because the other had already moved the shared tmp away | **Closed by PR #195** (hotfix, dtype + Metal cleanup cycle). PID-suffix the tmp filename (`canonical.with_extension(format!("a.tmp.{}", std::process::id()))`) so each process writes and renames its own tmp into the shared canonical path; treat ENOENT-on-rename as benign (a parallel-process race that the other process already won, leaving the canonical archive in place). | Fix at `crates/chelis-backend-c/tests/dtype_matrix_bf16_f16.rs::ensure_runtime_static_lib` + 5 sibling test-file copies of the helper: `crates/chelis-backend-c/tests/exec_compile.rs`, `crates/chelis-backend-c/tests/fused_in_place_exec.rs`, `crates/chelis-backend-c/tests/span_comments.rs`, `crates/chelis-cli/tests/cbackend_cast_memcpy.rs`, `crates/chelis-cli/tests/cbackend_empty_tensor_numel.rs`. |
| **Backend-C-EmitCast-Bf16F16-SilentCorruption** | C backend's `emit_cast` cross-precision path used the C language cast `(dst_et)src` for every source/destination pair, silently corrupting whenever bf16/f16 (storage `uint16_t`) was on either side — `cast(1.5_f32, bf16)` integer-truncated `1.5` to `0x0001` instead of bf16(1.5)=`0x3FC0`; reverse direction read the bf16 bit pattern as a `uint16_t` integer value widened to float (HIGH-severity silent data corruption surfaced by RT-Cleanup) | **Closed by PR #198** (WS-Cleanup-Fixups, dtype + Metal cleanup cycle). `emit_cast` now routes bf16/f16 endpoints through the runtime helpers (`chelis_{bf16,f16}_to_f32` / `chelis_f32_to_{bf16,f16}`) so a cross-precision cast becomes a true value conversion: bf16/f16 ↔ f32 via the helpers; bf16/f16 ↔ bf16/f16 chains via f32; bf16/f16 ↔ f64 / integer endpoints route through an f32 intermediate that is then cast via the C language cast to the wider/integer endpoint. Same bug class as the V2 C-backend silent-data-corruption sweep (CBackend-CastMemcpy / CBackend-ReshapeMemcpy / CBackend-PrintTensorF64 / CRuntime-F32Coupling); this entry extends that class to the bf16/f16 precision pair. | Fix at `crates/chelis-backend-c/src/emit.rs::emit_cast`. Regression tests at `crates/chelis-backend-c/tests/rt_cleanup_redteam.rs` (3 cast tests pinning bit-exact bf16/f16 ↔ f32 round-trips); evaluator-agreement matrix at `crates/chelis-backend-c/tests/dtype_matrix_bf16_f16_extended.rs` (31 cast-and-arithmetic tests within `BF16_TOL`/`F16_TOL` of evaluator output). |
| **Backend-Metal-HostConstFill-NaNInfFormat** | `host_const_fill_body(Prim::F32, value, ..)` formatted the literal value via Rust's `{:?}` debug formatter, which renders `f64::NAN` as `NaN` and `f64::INFINITY` as `inf` and then appended the `f` suffix, producing invalid C++ literals `NaNf` / `inff` that fail under `clang++` (HIGH-severity Metal compile break for any f32 Const tensor holding NaN or infinity, surfaced by RT-Cleanup) | **Closed by PR #198** (WS-Cleanup-Fixups, dtype + Metal cleanup cycle). `host_const_fill_body(Prim::F32, ..)` now discriminates via `value.is_nan()` / `value.is_infinite()` / `value.is_sign_negative()` and emits the C99 macro forms `(float)NAN` / `(float)INFINITY` / `-(float)INFINITY` instead of relying on the `{:?}` formatter for non-finite values; finite values continue to use the `{value:?}f` form. | Fix at `crates/chelis-backend-metal/src/dtype.rs::host_const_fill_body`. Regression tests at `crates/chelis-backend-metal/tests/rt_cleanup_redteam.rs` (NaN + positive-infinity + negative-infinity negative tests asserting the emitted source no longer contains `NaNf`/`inff` literals and instead contains the C99 macro forms). |
| **TypeCheck-F8E4M3-Deferred** | `f8e4m3` (8-bit float, E4M3 format) is rejected at the type-checker with a diagnostic pointing at `spec/04-type-system.md` §1.1.1; no backend implements native E4M3 and the evaluator has no scalar representation for the type | Open per `spec/04-type-system.md` §1.1.1 (Deferred Numeric Primitives). The deferral is language-level: `(t-prim {} f8e4m3)` and the `f8e4m3` literal suffix are rejected at type-check / lex time. Closure requires (a) a concrete first-party backend (HIP, C, or Metal) implementing f8e4m3 natively with a documented compiler-builtin or library route, AND (b) a corresponding scalar representation in the IR evaluator so cross-backend agreement remains meaningful. Dispatch when an explicit downstream consumer (Nautilus / Coral / Octant / Shoals issue or PR comment) names f8e4m3 as load-bearing for a customer use case, OR a backend vendor toolchain ships first-class f8e4m3 support. | Type-checker rejection site at `crates/chelis-types/src/` (precision admission); CLI lex-time suffix rejection per `spec/04-type-system.md` §1.6 (literal grammar). Spec: `spec/04-type-system.md` §1.1.1, mirrored at `spec/02-surf-syntax.md` §"Deferred and out-of-scope suffixes" and `spec/03-deep-syntax.md` §2 (`t-prim` deferred-set note). No regression test for the deferral itself today; the existing rejection tests pin current behavior. |
| **TypeSystem-UnsignedInts-OutOfScope** | Unsigned integer types (`u8`, `u16`, `u32`, `u64`, or any `uint*` spelling) are not in the active or deferred numeric primitive set; documented workaround is to cast to signed `int32` / `int64` at the boundary where unsigned data enters | Out of scope per `spec/04-type-system.md` §1.1.2. **This entry exists to document the decision so future cycles do not accidentally pick it up as "missing parity."** No planned lift; no closure trigger in any foreseeable cycle. If a customer use case for unsigned integer semantics surfaces, that triggers a separate spec-level decision (separate signed/unsigned arithmetic, comparison, overflow, AD adjoints, and backend dispatch) before any implementation work would begin. | Spec: `spec/04-type-system.md` §1.1.2 (out-of-scope rationale), §1.6 (suffix rejection), §4.4 (excluded as type quantifier identifiers); mirrored at `spec/02-surf-syntax.md` §"Deferred and out-of-scope suffixes" and `spec/03-deep-syntax.md` §2. No code anchor (rejection sites exist at lex/parse/check time per spec); no regression test that exercises a closure path because the closure is intentionally undefined. |
| **Backend-Metal-PadShrink-Deferred** | Metal codegen has no `pad` / `shrink` implementations; the reject pass denies these ops at codegen with a clear diagnostic mirroring the HIP backend's earlier deferral | Open per `spec/08-backends.md` lines 80 (carried-forward limitation list) and 377 (Metal §-specific deferral list). Closure: implement `pad` / `shrink` in Metal codegen mirroring the C/HIP shape — parameterize over input shape + padding spec, emit a fill kernel for the padded region and a copy kernel for the inner region (shrink is the inverse: pure copy of the inner region into a smaller output). Dispatch when a customer Metal workload requires `pad` or `shrink`, OR when the HIP backend's own pad/shrink implementation lands and the Metal backend should mirror it for cross-backend parity. | Anchor at `crates/chelis-backend-metal/src/` reject pass (rejects `RiscOp::Pad` / `RiscOp::Shrink` at codegen). Spec: `spec/08-backends.md` lines 80 and 377. No regression test pinning the current rejection diagnostic today; should add one as part of the §5 anchor discipline. |
| **Backend-Metal-AsyncDispatch-Deferred** | M-phase Metal codegen uses synchronous `waitUntilCompleted` after every kernel launch; pipelining queued GPU work is not yet wired through the host wrapper | Open per `spec/08-backends.md` line 393. Closure: thread an async-completion handle through the Metal host wrapper; replace the per-launch `waitUntilCompleted` with either `addCompletedHandler` on the command buffer (callback-based completion) or `waitUntilScheduled` + `commit` (queued pipelining); surface an end-of-program join so the host code blocks once at the boundary instead of per-launch. Dispatch when M-phase latency becomes user-visible — e.g., multi-kernel programs where launch overhead dominates execution time, or downstream customer workloads call out per-launch synchronization as a measurable cost. | Anchor at `crates/chelis-backend-metal/src/emit.rs` (per-launch `waitUntilCompleted` emission site) and the Metal host wrapper. Spec: `spec/08-backends.md` line 393. No regression test today. |
| **Stdlib-Precision-Generalization-Deferred** | Stdlib's `packages/chelis-std/src/` exports a mix of precision-polymorphic ops (`linear`, `embedding`, `attention.sdpa`, etc.) and ops pinned to f32; only the precision-polymorphic subset inherits the numeric dtype + Metal cleanup cycle's expansion to 9 dtypes (8 on Metal). The f32-pinned ops do not currently accept bf16/f16/f64 operands | Deliberately out of scope from the numeric dtype + Metal cleanup cycles. The cycles generalized the precision-polymorphic op set following the `linear` / `embedding` / `attention.sdpa` pattern (precision-tvar in the sig, lowering that admits operand precision, accumulator promotion per spec §5.7.1); the f32-pinned ops stay f32-pinned until triggered. **Closure trigger (observable):** explicit request from a downstream consumer (Nautilus / Coral / Octant / Shoals issue or PR comment naming the specific op needing bf16 / f16 / f64 support) OR a documented use case in a paper or product surface that requires a specific stdlib op at non-f32 precision. When triggered, the specific op is generalized following the established pattern (precision-tvar in the sig, lowering that admits the operand precision, accumulator promotion per spec §5.7.1). Entry stays open until all stdlib ops are generalized OR explicitly closed with rationale via an orchestrator-led scope decision. | Anchor at `packages/chelis-std/src/` (stdlib source). Spec: `spec/04-type-system.md` §5.4 (active dtype list, "source of truth for stdlib generalization") and §"Stdlib polymorphism" (lines 805+). Reference pattern: the `linear` / `embedding` / `attention.sdpa` generalization shipped in the numeric dtype cycle. No regression test today because the entry tracks a deliberate non-expansion. |
| **Backend-C-EmitReduceSimple-Bf16F16-Deferred** | C backend's `emit_reduce_simple` is f32-hardcoded (`float acc`); bf16/f16 `MinReduce` / `ProdReduce` hit the f32 guard and panic at codegen time even though `Sum` and `MaxReduce` were generalized for bf16/f16 in the numeric dtype cycle | Open; surfaced and pinned by WS-Cleanup-Fixups (PR #198, dtype + Metal cleanup cycle) via `#[should_panic]` regression tests so the gap is locked and discoverable. Closure: extend `emit_reduce_simple` with the same convert-then-reduce pattern WS-1 (PR #192) shipped for `Sum` / `MaxReduce` — load via `chelis_bf16_to_f32` / `chelis_f16_to_f32`, accumulate in f32, store via `chelis_f32_to_bf16` / `chelis_f32_to_f16` if the destination is reduced-float, else direct. Deliberately out of scope for the cleanup cycle per the "no new architectural changes" constraint that scoped the cleanup to genuine BLOCKERs. Dispatch when any program needs bf16/f16 `MinReduce` or `ProdReduce`; the `#[should_panic]` tests un-ignore as the fix lands. | Anchor at `crates/chelis-backend-c/src/emit.rs::emit_reduce_simple` (around L3569, with the explicit "WS-A1 / F1: emit_reduce_simple path is f32-hardcoded" panic guard pinning the bug). Reference closure pattern: WS-1 (PR #192) bf16/f16 generalization for `Sum` / `MaxReduce` in the same file. `#[should_panic]` regression tests at `crates/chelis-backend-c/tests/dtype_matrix_bf16_f16_extended.rs` lock the current panic on bf16/f16 `MinReduce` / `ProdReduce` and will flip to positive assertions when the closure lands. |

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
replacement for any non-F32 case). Test: `blas_rejection_cross_product_adversarial.rs::
nonsilent_rejection_invariant_for_every_w5_rejected_precision`.

## Post-0.7.9 test-infrastructure workstream (shipped in v0.7.10)

A post-0.7.9 workstream that targeted CI wall-clock, test-suite quality, and
two pre-existing compiler bugs surfaced by the downstream cascade. Recorded
here as a completion anchor. Shipped in the `v0.7.10` release.

**Headline CI win:** the per-PR `Integration Tests` job dropped from a
~11.7min baseline to ~5.5min — the lever was splitting the heavyweight
end-to-end suite (real `chelis build` + gcc + run, real `reef build` /
`reef install`) off the per-PR gate onto a nightly + manual `heavy-e2e.yml`
workflow, so per-PR CI is no longer gated by tests that cannot parallelize
within a single invocation.

Closures:

- **Cross-process chelis-std typecheck cache** (PR #127). `chelis check` /
  `chelis build` no longer re-typecheck the entire chelis-std import graph
  per invocation; the typechecked + lowered chelis-std library sub-context
  is content-addressed and cached on disk (`StdLibContext`), reused across
  every process and every stdlib-importing fixture. Local `chelis check` of
  a stdlib-importing file is ~13x faster warm.
- **Heavyweight e2e split** (PR #126). New `.github/workflows/heavy-e2e.yml`
  runs the explicitly-named heavy suite nightly + on `workflow_dispatch`
  only; `.config/nextest.toml` `default`/`ci` profiles exclude it, the
  `nightly` profile includes exactly it. Partition invariant locked by
  `scripts/test_nextest_profile_partition.py`.
- **e2e test parsimony pass** (PRs #121-124, #136). Deduped and consolidated
  the chelis-cli / backend / ir / types test clusters; `production_stdlib_typechecks`
  returned to the per-PR gate once the typecheck cache made it cheap.
- **Toolchain footgun guards** (PRs #125, #133). A test-timing budget
  (`scripts/test_timing_check.py` + committed baseline), em-dash lint
  visibility (severity-bucketed `chelis lint` output — the recurring §8.6
  footgun was visibility, not a parser gap), and a `gate.py` CI-parity lock
  (`scripts/gate.py` is the single source of truth; the parity test catches
  both `cargo` and `chelis` invocations hand-inlined into CI gate steps).
- **CompiledContext cache package-identity + compiler-version keys**
  (PR #130). HIGH-severity: the Phase K compiled-context disk cache keyed
  only on `(package_name, package_version, source_hash)` — two packages at
  distinct on-disk roots with identical name+version+source collided on one
  cache file. Latent until the typecheck-cache work added an XDG cache-dir
  fallback that un-gated the cache for the no-`CHELIS_REEF_HOME` case. Fix
  folds a canonicalized `package_root` + `COMPILER_VERSION` `CacheIdentity`
  into the cache file name, the on-disk envelope, and the `load_if_fresh`
  freshness check; cache format version bumped 3 -> 4.
- **Negative-axis + rank-0 standalone-parameter IR lowering** (PR #132).
  Two pre-existing panics (present on the v0.7.9 tag) that shared one
  symptom — `chelis eval` / `chelis test` panicking with `softmax axis
  requires a statically known axis in IR lowering` for any package linking
  bundled chelis-std. (a) `extract_axis` cast a negative axis literal
  straight to `usize`; negative axes are now a uniform supported convention
  across spec, checker, and lowering. (b) A def whose parameter types come
  from a separate `sig` declaration lost those types under standalone
  library lowering, binding params to a rank-0 default; the checker now
  stamps declared sig param types onto the `(params ...)` node. This was
  the downstream-cascade blocker — it unblocks `chelis test` for every
  package that links chelis-std.
- **Atomic reef package-cache writes** (PR #137). Pre-existing concurrent-
  build race: `load_registry_package` extracted a package archive into the
  shared `$CHELIS_REEF_HOME/cache/<hash>/` directory in place, so a
  concurrent `chelis reef build` could read a half-written `reef.toml`. Fix
  unpacks into a unique sibling staging dir and `fs::rename`s it into place
  (same-directory rename is atomic).
- **Adversarial coverage + test naming cleanup** (PRs #129, #134, #138-140).
  A terminal red-team added adversarial coverage for the typecheck cache,
  the e2e split partition invariant, and the toolchain guards (it is what
  surfaced PR #130). 84 scaffolding-named test files (`phase3*`, `wsa*`,
  `wsc*`, `rt*`, `s*`, `red_team_*` stamps) were renamed to describe what
  each suite verifies; pure renames, every test count unchanged.

No tracked follow-ups remain open from this workstream.

## Standalone follow-up entries (filed post-W7)

These were previously narrative tail-references inside the M5-follow-up
entry or held in caveats. After W6 + W7 closed the cross-product, they are
each their own named workstream:

### §5 R1 — Softmax / layer_norm / attention recognizers

| Tracks | Required closure | Current executable anchor |
|---|---|---|
| Recognizer coverage tail (Pattern B in §1) | Spec definition per op shape; IR recognition (likely extending `chelis_ir::specialize`); backend dispatch in C (fused kernel) and HIP (cuDNN-shape equivalent); AD policy and adjoint table extension; corpus per op. Each op is a wave-sized effort. **Attention specifically** has graduated to Kerrent Phase K6 (`spec/design/kerrent.md` §Milestone 6) as a committed-scope path: the FlashAttention-shaped fused kernel is authored in Kerrent rather than recognized-and-fused after decomposition. Softmax and layer_norm remain on the recognizer track. | None today. `crates/chelis-cli/tests/specialization_dispatch.rs` lines 134-141 document the generic-path cost (3-5× slower for softmax/layer_norm). |

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
| Canonicalizer treats `(a*3)/2` and `a*(3/2)` as equivalent under rational arithmetic; under integer-floor they differ for odd `a` | Propagate divisibility info through DimExpr to make canonicalization floor-aware, or restrict rewrite rules to proven-divisible cases. Harmless under today's symbolic-dim corpus (every dim divides cleanly) but a latent foot-gun if a future corpus violates that. | `crates/chelis-ir/tests/dim_canon_adversarial.rs::dim_canonicalizer_uses_rational_not_integer_floor_semantics` locks current behavior. |

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
