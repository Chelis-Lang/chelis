# Futhark Research Synthesis for Chelis

Aggregation of findings from 106 Futhark publications and student projects, organized by theme. Each section synthesizes the state of knowledge and closes with implications for Chelis.

---

## 1. Type Systems for Safe Mutation and Shape Reasoning

Futhark's type system has evolved across three distinct axes: ownership, shape, and rank.

**Uniqueness types** provide the foundation for safe in-place array updates in a purely functional language. The system tracks ownership via intra-procedural alias analysis, using occurrence traces to prove that consumed arrays are never referenced again. This is deliberately simpler than full linear or affine type systems — the insight is that a conservative, lightweight analysis suffices for the data-parallel setting where arrays flow through combinators rather than being threaded through arbitrary control flow. The cost of an in-place update becomes O(element) rather than O(array), and the type system guarantees this statically.

**Size-dependent types** restrict dimension terms to variables and constants, avoiding the complexity of full dependent types. Existential sizes are handled via ANF transformation — when a function returns an array of unknown size, the compiler automatically binds the existential at the call site. A dynamic coercion serves as an explicit escape hatch when the type system cannot prove size equality. This design is pragmatic: purely syntactic size equality with a well-defined fallback, rather than attempting to solve arbitrary arithmetic constraints.

**Rank polymorphism** (AUTOMAP) uses integer linear programming to automatically insert minimal map/replicate operations, inferring how a scalar function should lift over arrays of different ranks. This preserves Hindley-Milner guarantees while approaching the conciseness of dynamic array languages like APL or NumPy broadcasting.

The **module system** uses static interpretation to eliminate all module structure at compile time, achieving zero-overhead abstraction. Higher-order functors are resolved statically, and module type abstraction unifies cleanly with core-language polymorphism.

**Refinement types** have been explored for compile-time bounds verification, potentially eliminating runtime checks entirely — though this remains a student-project-level result rather than production infrastructure.

### Chelis Implications

Chelis already has HM inference and named tensor dimensions. The Futhark research suggests several concrete next steps:

- **Uniqueness for buffer reuse (Phase 2):** Chelis plans linear types and borrowing. Futhark's experience shows that a *simple* uniqueness system with alias analysis is sufficient for the data-parallel case — full linear types may be over-engineering. The key question is whether Chelis's RISC DAG structure (where tensors flow through a fixed set of primitives) allows an even simpler ownership model than Futhark's.
- **Size-dependent types:** Chelis could benefit from Futhark's "syntactic equality + dynamic coercion" design for dimension checking. This avoids the complexity of SMT-based size solving while still catching most shape errors statically.
- **Rank polymorphism via ILP:** AUTOMAP's approach could inform how Chelis handles broadcasting. Instead of implicit broadcasting (which Chelis explicitly rejects), an ILP-based elaboration pass could insert explicit `expand` operations during type inference, preserving explicitness while reducing boilerplate.
- **Static module interpretation:** Relevant if Chelis adds a module system — the Futhark approach proves that higher-order modules can have zero runtime cost on GPU targets.

---

## 2. Array Combinators, Fusion, and the SOAC Algebra

Futhark's compiler is built around second-order array combinators (SOACs) — `map`, `reduce`, `scan`, `scatter`, and their streaming variants. The key insight developed across a decade of papers: **fusion is the central optimization**, and the right abstraction for fusion is an algebraic rewrite system over combinators rather than loop-level dependence analysis.

**The fusion algebra** derives from the "banana split theorem" and fold/unfold properties. Producer-consumer fusion (vertical: `map f . map g → map (f . g)`) and independent fusion (horizontal: adjacent maps over the same input merged into a single traversal) are realized as bottom-up T2 graph reductions over the SOAC dependency graph. Inhibitor propagation handles fusion-blocking operations like size changes and transposes.

**Streaming SOACs** (`stream_map`, `stream_red`, `stream_seq`) generalize the basic combinators by allowing chunked execution. The chunk is the unit of parallelism — `stream_red` processes chunks in parallel and reduces across them with an associative operator. This explicitly encodes strength-reduction invariants: expensive parallel initialization once per chunk, cheap sequential recurrence within. The optimal chunk size balances hardware occupancy against per-thread memory footprint.

**Redomap** is a compiler-synthesized construct that fuses a map and a reduce, returning both results from a single traversal. This avoids materializing intermediate arrays and maps naturally to GPU execution where threads compute local results and then reduce.

**Segmented operations** — reductions and scans over irregular segments — are compiled via runtime strategy selection. Three kernel variants (large segments, small segments, loop-in-map) cover the design space, with heuristics selecting the appropriate variant based on segment count and size. Single-pass scan algorithms using decoupled lookback achieve 85% of memcpy bandwidth.

**Scatter and histogram fusion** extend the algebra to irregular writes. The `gen_reduce` (generalized histogram) construct uses adaptive strategies balancing bin conflicts against memory footprint, with analytical models auto-selecting parameters.

### Chelis Implications

Chelis's RISC DAG IR and Phase 1 fusion plans are directly informed by this work:

- **DAG-level fusion rewriting:** Chelis's ~12 Tier 1 primitives (`add`, `mul`, `exp`, etc.) and ~10 Tier 2 derived operations (`relu`, `softmax`, `matmul`) form a closed vocabulary analogous to Futhark's SOACs. Fusion rules should operate at this level — rewriting DAG subgraphs rather than analyzing loop nests. The T2 graph-reduction approach is directly applicable.
- **Streaming as sequentialization control:** The streaming SOAC concept maps to Chelis's need to control GPU thread footprint. A `vmap`-with-chunk-size parameter could serve the same role.
- **Segmented reduction strategies:** Chelis will need to compile `reduce` over tensor slices (e.g., softmax's row-wise reduction). Futhark's three-strategy approach with runtime dispatch is the proven solution.
- **Histogram/scatter as first-class:** Embedding operations and gradient accumulation in ML training are fundamentally scatter-add operations. Making these first-class with dedicated fusion rules (as Futhark does with `gen_reduce`) would give Chelis a significant advantage over frameworks that treat these as opaque library calls.
- **Fusion search space:** Futhark uses hand-written heuristics for fusion decisions. Chelis's Phase 1 plan mentions `egg`-based equality saturation as an alternative — the Futhark experience suggests starting with simple greedy fusion (it works well in practice) and reserving equality saturation for cases where the greedy approach leaves performance on the table.

---

## 3. Nested Parallelism, Flattening, and Kernel Extraction

The central compilation challenge for a functional array language targeting GPUs is mapping nested parallelism to a flat thread model. Futhark's approach has evolved from aggressive flattening to a nuanced "moderate" strategy.

**Full flattening** (NESL-style) transforms all nested parallelism into flat data-parallel operations, preserving work-depth asymptotics. But it introduces irregular arrays, destroys locality, and prevents subsequent optimizations like tiling and coalescing. In practice, it is often slower than leaving some parallelism unexploited.

**Moderate/incremental flattening** is Futhark's production strategy. The algorithm distributes outer `map` operators across enclosed maps and loops using higher-order rewrite rules (map-loop interchange, map distribution). Critically, it *stops before introducing irregular arrays*, preserving the regular structure needed for memory coalescing and tiling. The result is a set of "perfect nests" where outer levels are parallel maps (trivially mapped to GPU thread blocks) and inner levels are sequential or use shared-memory reductions.

**Incremental flattening** takes this further by generating *multiple code versions* for each nest, covering all viable mappings of the application to hardware parallelism levels. Runtime predicates select the optimal version based on actual input sizes. This is paired with an autotuning framework that uses monotonicity properties to efficiently search the threshold space — bottom-up algorithms compute maximal threshold intervals per dataset, achieving 6-22x speedup over stochastic tuning.

**Flattening by expansion** offers a library-level alternative: an `expand` higher-order function that flattens irregular nested parallelism into regular segmented operations. This gives the programmer explicit control over the flattening process, which can be important when the compiler's heuristics make suboptimal choices.

**Irregular nested parallelism** remains a frontier. Flattening match-expressions (generalized if-flattening) and function lifting for array parameters have been explored in student work, achieving 1.7-5x speedups on specific patterns.

### Chelis Implications

This is the most directly relevant body of work for Chelis Phase 1:

- **Start moderate, not flat:** Chelis should not attempt full NESL-style flattening. The moderate flattening strategy — distribute outer maps, stop before irregularity — is the proven approach. The key rewrite rules (map-loop interchange, map distribution) should be implemented first.
- **Multi-versioning for kernel selection:** Generating multiple kernel variants with runtime dispatch is essential for production performance. This pairs naturally with Chelis's existing `jit` compilation boundary concept — the JIT compiler can select among pre-compiled kernel variants based on tensor dimensions.
- **Autotuning via monotonicity:** Rather than stochastic search, exploit the monotonic relationship between input sizes and optimal parallelism mappings. This is particularly relevant for ML workloads where batch sizes and sequence lengths vary but model dimensions are fixed.
- **Regularity preservation:** Chelis's tensor type system (no ragged arrays, explicit named dimensions) already guarantees regularity. This is a structural advantage over Futhark, which must prove regularity through analysis. Chelis kernels should always compile to perfect nests without needing the regularity checks that complicate Futhark's flattening.

---

## 4. Memory Management and Layout Optimization

GPU performance is dominated by memory access patterns. Futhark's research program has developed increasingly sophisticated memory analysis and optimization passes.

**Linear Memory Access Descriptors (LMADs)** are the foundational abstraction — structured index functions that describe how logical array indices map to physical memory offsets. LMADs capture transpositions, slicing, and reshaping without copying, enabling the compiler to reason about memory layout algebraically. The **FunMem IR** extends Futhark's functional core with non-semantic memory annotations, maintaining dual semantics (value-based for correctness, heap-based for optimization) that can be cross-validated.

**Array short-circuiting** eliminates redundant allocations by proving (via LMAD non-overlap analysis) that an output array can reuse the memory of a consumed input. This is the memory-level analog of fusion — avoiding materialization of intermediate buffers. Static analysis proves aliasing safety across complex control flow including loops and conditionals.

**Memory block merging** uses graph coloring to reuse memory blocks across non-overlapping lifetimes. Combined with allocation hoisting (moving allocations out of loops) and size hoisting, this achieves 0-70% memory footprint reduction. The runtime impact is mixed — sometimes the changed allocation pattern hurts cache behavior.

**Memory coalescing** — ensuring that consecutive GPU threads access consecutive memory addresses — is achieved by transposing non-parallel array dimensions to innermost position. The compiler records the symbolic composition of affine transformations needed to achieve coalesced access. For arrays used as `stream_seq` inputs that are invariant to a parallel dimension, **block tiling** into shared memory provides further speedup.

**GPU-host transfer minimization** uses graph-based dataflow optimization to minimize synchronous memory transfers between host and device. A specialized minimum vertex cut algorithm identifies which transfers can be eliminated or deferred.

### Chelis Implications

Memory optimization is where Chelis's Phase 1 GPU backend will face the most immediate engineering challenges:

- **LMADs for Chelis's RISC DAG:** Each Chelis primitive (`add`, `mul`, `matmul`, etc.) has well-defined access patterns. LMADs could describe these patterns algebraically, enabling the compiler to prove when transposes can be eliminated or when output buffers can alias inputs. This is more tractable than Futhark's general case because Chelis's primitive vocabulary is small and fixed.
- **Buffer reuse via lifetime analysis:** Chelis's DAG structure makes lifetime analysis straightforward — a tensor is live from its definition to its last use. Graph coloring for buffer reuse is directly applicable and should be a Phase 1 priority.
- **Coalescing as layout selection:** Rather than inserting transposes, Chelis could select memory layouts per-tensor based on how each tensor is consumed. Named tensor dimensions provide the metadata needed to make this decision at compile time.
- **Host-device transfer planning:** For the HIP backend, Chelis needs a transfer planner that minimizes data movement. The graph-based approach from Futhark's work is a good starting point — the RISC DAG is literally a dataflow graph.

---

## 5. Automatic Differentiation for Parallel Array Programs

Futhark's AD research addresses the fundamental tension between reverse-mode AD (which requires saving intermediate values) and parallel execution (which wants to minimize memory footprint).

**Recomputation-based reverse AD** eliminates the tape entirely by recomputing forward values during the backward pass. For nested parallel programs, this preserves work-span asymptotics — the backward pass has the same parallelism structure as the forward pass. The key insight: in a data-parallel setting, recomputation is often cheaper than the memory overhead of storing intermediates, because GPU compute is abundant relative to memory bandwidth.

**Combinator-level differentiation** provides rewrite rules for differentiating `map`, `reduce`, `scan`, `scatter`, and `reduce_by_index` as whole operations rather than expanding them to loops and differentiating element-wise. This preserves parallel structure through the AD transformation. For example, the adjoint of `map f xs` is `map (adjoint f) xs (adjoints)` — the parallelism is maintained.

**Specialized AD for common operators** yields significant speedups. Addition has a trivial adjoint (identity). Multiplication's adjoint involves the other operand. Min/max adjoints are sparse (only the extremal element gets a nonzero adjoint). Invertible operators allow recovering inputs from outputs, eliminating recomputation. Sparse Jacobian structures (common in ML) enable vectorized adjoint accumulation.

**High-level vs. low-level AD comparison** (Futhark's approach vs. Enzyme on LLVM IR) shows that high-level AD achieves up to 10x speedup on memory-bound workloads because it can perform semantic transformations (fusion, layout optimization) that are invisible at the LLVM level.

**AD for histograms/reduce_by_index** is critical for ML — embedding gradients, attention score accumulation, and sparse updates are all scatter-add operations. Futhark provides correct-by-construction rewrite rules with special cases for +, *, min/max operators.

### Chelis Implications

Chelis already has reverse-mode AD as a DAG-to-DAG rewrite (Phase 0). The Futhark research validates this approach and suggests refinements:

- **Recomputation over tape:** Chelis's `grad` transform should default to recomputation for GPU execution. The RISC DAG structure makes this natural — each primitive's forward computation can be replayed from inputs. Selective checkpointing (saving expensive-to-recompute intermediates like convolution results) can be added as an optimization.
- **Combinator-level adjoint rules:** Chelis's Tier 1 and Tier 2 operations should each have hand-written adjoint rules that preserve parallelism structure. `grad(matmul(A, B))` should produce `matmul(dout, B^T)` and `matmul(A^T, dout)` directly, not expand matmul into element-wise operations and differentiate those.
- **Sparse adjoint optimization:** For operations like `softmax`, `layer_norm`, and `reduce`, the Jacobian is structured (diagonal, block-diagonal, or sparse). Chelis's AD should exploit this structure rather than computing dense Jacobians.
- **AD-fusion interaction:** The Futhark work shows that AD and fusion must be co-designed. The adjoint of a fused kernel should itself be fusible. This means AD should operate on the pre-fusion DAG, and the result should be re-fused.
- **High-level AD is the right bet:** The empirical comparison validates Chelis's choice of DAG-level AD over LLVM-level approaches like Enzyme. The performance advantage comes from preserving semantic information that enables downstream optimization.

---

## 6. GPU Backend Engineering and Portability

Futhark has been compiled to OpenCL, CUDA, HIP, WebGPU, Vulkan, and WebAssembly. This breadth reveals both the power of a high-level IR and the practical challenges of GPU backend diversity.

**OpenCL vs. CUDA vs. HIP** performance comparison shows that equivalent source-level optimizations don't guarantee cross-platform performance. API-level defaults (occupancy, register usage), hardware introspection gaps, and missing features (like CUDA's warp-level primitives in OpenCL) create systematic differences. CUDA generally wins on NVIDIA hardware; HIP is competitive on AMD.

**Runtime compilation** (NVRTC for CUDA, hiprtc for HIP) enables kernel specialization at launch time, which is essential for the multi-versioning strategy. The compilation overhead is amortized across invocations.

**WebGPU/WGSL** targets face fundamental limitations: no 64-bit atomics (provably impossible under the WGSL memory model), limited atomic emulation for 8-bit and 16-bit types via CAS loops, and restricted control flow. Performance is roughly 4x slower than CUDA on equivalent benchmarks.

**Tensor Core integration** requires pattern-matching intragroup matrix multiplications in the compiler, then lowering to specialized WMMA instructions. A two-pass transformation with memory fixup achieves 1.9-60x speedups over standard Futhark, but remains 2-4x slower than hand-optimized code. The gap comes from register pressure and memory staging — areas where the compiler's general-purpose analysis doesn't match expert knowledge.

**Multi-GPU execution** uses a `husk` operator to distribute operations across GPUs with a worker-thread runtime. Scaling is good for large datasets but overhead limits small problems. This remains a research prototype.

**Bounds checking on GPU** uses a global failure variable rather than hardware exceptions (which GPUs don't support). The overhead is ~4% geometric mean — negligible for production use.

### Chelis Implications

Chelis Phase 1 targets HIP specifically. The Futhark experience provides direct guidance:

- **HIP via hiprtc:** Chelis's plan to use hiprtc for runtime compilation is validated by Futhark's experience. The multi-versioning strategy requires runtime compilation to specialize kernels for actual tensor dimensions.
- **Start with one backend, abstract later:** Futhark's multi-backend experience shows that portable performance is harder than it looks. Chelis should focus on HIP first and defer portability until the kernel generation strategy is mature.
- **Bounds checking strategy:** Chelis should adopt Futhark's global-failure-variable approach for debug builds. The 4% overhead is acceptable, and GPU debugging without it is painful.
- **Tensor Core / specialized hardware:** Relevant for Phase 2+. The Futhark experience shows that pattern-matching at the IR level can automatically exploit specialized hardware, but the 2-4x gap vs. hand-optimized code suggests that Chelis may need dedicated `matmul` kernel templates rather than relying purely on generic compilation.
- **WebGPU as a future target:** Interesting for inference deployment but the fundamental limitations (no 64-bit, limited atomics) mean it's a restricted subset. Chelis could target this for inference-only workloads where training-specific operations aren't needed.

---

## 7. Defunctionalization and Higher-Order Function Compilation

GPUs require statically-known function call targets — they cannot efficiently dispatch through function pointers. Futhark solves this with **branch-free defunctionalization**: a type-based transformation that eliminates higher-order functions at compile time without introducing runtime dispatch.

The key constraint is `orderZero` — functions cannot appear inside conditionals, arrays, or loop-variant bindings. This ensures that every function application can be statically resolved. The transformation is proven correct via logical relations and has zero runtime overhead.

This is critical for the deep learning use case: neural network layers are naturally expressed as higher-order functions (a layer is a function that takes parameters and returns a function from inputs to outputs), but GPU execution requires all function calls to be resolved at compile time.

### Chelis Implications

- **Chelis's Deep syntax already has `app` and `var`:** The explicit function application in Deep means defunctionalization is structurally simpler — the compiler already sees every application site. Surf's higher-order features would need defunctionalization during lowering to Deep/IR.
- **Layer combinators:** For ML, Chelis should support composing layers as higher-order functions in Surf (e.g., `sequential(linear(784, 128), relu, linear(128, 10))`) and defunctionalize them to a flat sequence of RISC DAG operations during compilation. The Futhark work proves this can be zero-cost.
- **Effect system interaction:** Chelis Phase 2 plans algebraic effects. Effects and defunctionalization interact — effect handlers are higher-order. The Futhark experience suggests keeping the function-in-function-out pattern but ensuring all handlers are statically resolved.

---

## 8. Autotuning and Adaptive Optimization

Futhark's autotuning research addresses a fundamental problem: the optimal parallelism mapping depends on input sizes, which are only known at runtime.

**Monotonicity-based tuning** exploits the observation that if a kernel variant is optimal for input size N, it remains optimal for all sizes ≥ N (or ≤ N). This turns the tuning problem from arbitrary search into a 1D threshold-finding problem. Bottom-up algorithms compute maximal threshold intervals per dataset, achieving deterministic tuning with 6-22x speedup over stochastic approaches.

**Reactive benchmarking** uses autocorrelation and relative standard error to automatically determine when enough samples have been collected, reducing benchmarking time from 23+ minutes to 4 minutes with comparable reliability.

**Domain-specific autotuners** that exploit compiler structure (comparison-based, branch-based, program instrumentation) achieve 2-25x speedup over general-purpose tuners like OpenTuner.

### Chelis Implications

- **Kernel variant selection for ML:** ML workloads have a fixed model architecture but variable batch sizes and sequence lengths. Monotonicity-based tuning is ideal — calibrate once per model architecture, then use thresholds to select kernel variants at runtime.
- **Integrate tuning into `chelis build`:** The tuning framework should be part of the build pipeline, not a separate offline tool. `chelis build --tune` could run a calibration pass with representative inputs.
- **JIT + autotuning:** Chelis's `jit` compilation boundary is the natural place to apply tuned thresholds — the JIT compiler selects pre-compiled kernel variants based on actual tensor dimensions.

---

## 9. Stencil, Convolution, and Structured Access Patterns

Several Futhark projects address structured access patterns that appear frequently in scientific computing and ML.

**Stencils** (1D, 2D, 3D) are compiled via a multi-write big-tile approach: shared memory staging with boundary-aware partitioning eliminates redundant global memory reads. This achieves up to 3x speedup over reference implementations. On CPU, tiling and partitioning with boundary-aware compilation eliminates runtime checks.

**Convolution tiling** using block-register tiling achieves 96.3% of A100 peak TFLOPs — but this requires hand-transformed Futhark IR, suggesting that automatic compilation still has a gap. The loop transformation formalization provides a template for compiler automation.

**Tiling generalization** to arbitrary tensor contractions uses LMAD-based copies for data staging between memory hierarchies, achieving 68-98% of reference implementations.

### Chelis Implications

- **`conv2d` as a Tier 2 primitive:** Chelis already has `conv2d` in its primitive set. The Futhark work shows that generic compilation reaches 68-98% of peak, but closing the last gap requires specialized tiling templates. Chelis should support both: generic compilation for uncommon configurations, hand-optimized templates for standard convolution shapes.
- **Stencil as a future primitive:** If Chelis extends to PDE or physics applications, a stencil primitive with shared-memory compilation would be valuable. The multi-write big-tile approach is the right compilation strategy.
- **Block-register tiling for matmul:** Chelis's `matmul` primitive should compile to tiled GPU kernels. The Futhark work on tensor contraction tiling provides the general framework — LMAD-based copies for staging, block + register tiling for compute.

---

## 10. Language-Level Safety and Correctness

Futhark's research program includes work on formal correctness and safety guarantees that go beyond performance.

**Bounds checking with hybrid analysis** separates safety checks into a standalone runtime predicate (inspector) that runs before the kernel. A cascade of increasingly complex sufficient conditions (constant bounds, symbolic bounds, runtime checks) handles everything from simple indexing to indirect array access with negligible overhead.

**Certified code extraction from Coq** demonstrates that formal verification can target Futhark. The FuCert framework extracts verified array programs, proving that the compilation target supports formal methods integration.

**Sum types and pattern exhaustivity** extend the type system with ADTs and exhaustive matching, enabling safe data modeling without runtime type tags on the GPU. The implementation transforms sum types to tagged tuples in the IR.

### Chelis Implications

- **Chelis already has ADTs and exhaustive matching.** The Futhark work on sum-type-to-tagged-tuple compilation is directly relevant for GPU lowering — Chelis will need the same transformation.
- **Bounds checking strategy:** The hybrid inspector approach is ideal for Chelis — run a lightweight check before kernel launch rather than checking inside every thread. This pairs with Chelis's explicit `expand` (no implicit broadcasting means fewer dynamic shape checks).
- **Formal methods as a Phase 3+ possibility:** The Coq extraction work shows that a functional array language is a viable formal methods target. Chelis's clean IR semantics could support verified compilation in the future.

---

## 11. Practical ML System Engineering

Several Futhark projects demonstrate end-to-end ML systems that reveal both the strengths and limitations of a functional array language for ML:

**Deep learning libraries** built in Futhark achieve competitive performance with TensorFlow on standard architectures (MLP, CNN). The functional approach — modeling layers as forward/backward function pairs with compile-time defunctionalization — is both type-safe and efficient. A complete DDPM (denoising diffusion) implementation with U-Net achieves ~90% classification accuracy on generated MNIST digits.

**SVM, GBDT, and k-NN** implementations show that classical ML algorithms parallelize well in the Futhark model, often exceeding hand-tuned libraries (2.4-9.6x faster SVM prediction than ThunderSVM).

**Financial computing** (trinomial trees, Monte Carlo, convex optimization) demonstrates 3-4 orders of magnitude speedup over sequential implementations, with Futhark code remaining readable and maintainable compared to hand-tuned CUDA.

**Sparse data structures** (CSR, COO, ELLPACK) are handled through explicit representation rather than built-in sparse types, achieving 1893x speedup for Gaussian mixture models. This validates the "explicit over implicit" design philosophy.

### Chelis Implications

- **Layer-as-function-pair is the right pattern:** Chelis should support this natively — a layer type that bundles forward, backward (derived from `grad`), and parameter shapes. The Futhark experience shows this is both ergonomic and compiles efficiently.
- **Classical ML matters:** Chelis shouldn't focus exclusively on deep learning. The same RISC DAG primitives that support neural networks also support SVMs, decision trees, and k-NN, often with large speedups.
- **Sparse as explicit representation:** Chelis's philosophy of "no implicit broadcasting, no implicit promotion" extends naturally to sparsity — represent sparse tensors as explicit CSR/COO structures rather than adding sparse type machinery. Operations over sparse representations compose from the same RISC primitives.

---

## Summary of Highest-Priority Chelis Implications

For **Phase 1** (GPU backend + fusion), ordered by impact:

1. **Moderate flattening with multi-versioning** — proven to outperform both full flattening and manual kernel writing. Implement map distribution and map-loop interchange first.
2. **SOAC-style fusion at the DAG level** — rewrite Chelis RISC DAG subgraphs using algebraic fusion rules. Start with greedy producer-consumer fusion; defer equality saturation.
3. **LMAD-based memory analysis** — algebraic index functions for buffer reuse, coalescing analysis, and transpose elimination. Chelis's fixed primitive vocabulary makes this more tractable than Futhark's general case.
4. **Buffer lifetime analysis and reuse** — graph coloring over the RISC DAG for memory block merging. Immediate memory savings.
5. **Segmented reduction strategies with runtime dispatch** — three kernel variants for different segment configurations. Critical for softmax, layer_norm, and attention.
6. **Host-device transfer planning** — minimize data movement using dataflow analysis over the RISC DAG.
7. **Bounds checking via global failure variable** — 4% overhead, essential for debugging GPU kernels.

For **Phase 2+** (type system, effects, AD refinement):

1. **Lightweight uniqueness over full linear types** — Futhark's experience suggests that simple ownership tracking suffices for the data-parallel case.
2. **Combinator-level AD with sparse Jacobian exploitation** — hand-written adjoint rules per RISC primitive, preserving parallelism structure.
3. **Recomputation-based AD for GPU** — eliminate tape, recompute from checkpoints. Validated by Futhark's 10x advantage over LLVM-level AD.
4. **Monotonicity-based autotuning** — deterministic, fast, and integrable into `chelis build`.
5. **Rank polymorphism via ILP elaboration** — explicit `expand` insertion during type inference, preserving Chelis's no-implicit-broadcasting guarantee.
