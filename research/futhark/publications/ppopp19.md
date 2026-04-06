# Incremental Flattening for Nested Data Parallelism

## Metadata
- **Authors:** Troels Henriksen, Frederik Thorøe, Martin Elsman, Cosmin Oancea
- **Venue/Year:** PPoPP '19 (Principles and Practice of Parallel Programming), 2019

## Summary
Modern parallel hardware, particularly GPUs, features multiple levels of parallelism and complex memory hierarchies, making it difficult to write high-level code that performs optimally across varying dataset sizes and architectures. Traditional compiler optimizations for nested data parallelism rely on static heuristics or offline autotuning to produce a single "best" code version, which often fails to adapt to dynamic input characteristics or different hardware targets. This paper introduces a compiler-driven technique that overcomes this limitation by generating multiple semantically equivalent code versions, each optimized for a different mapping of application parallelism to hardware parallelism levels.

The core innovation, called *incremental flattening*, systematically explores all viable ways to flatten nested parallel constructs and combines them into a single program guarded by runtime predicates. These predicates compare the actual size of parallel workloads against threshold values that are automatically tuned to the target hardware. By dynamically selecting the most appropriate code path at runtime, the compiler can adapt to both dataset characteristics and intermediate array sizes during execution.

Fully integrated into the Futhark functional data-parallel compiler, this approach bridges the gap between high-level, hardware-agnostic programming and low-level, architecture-specific optimization. It eliminates the need for manual kernel selection or brittle static heuristics, enabling developers to write clean, modular parallel code while still achieving performance competitive with hand-tuned libraries across diverse workloads and GPU architectures.

## Key Contributions
- **Incremental Flattening Algorithm:** A generic, top-down compiler pass that generates multiple code versions covering all possible utilizations of nested application parallelism and their mappings to hardware parallelism levels.
- **Runtime Multi-Version Dispatch:** A mechanism that combines generated code versions into a single executable using predicates that dynamically select the optimal version based on actual workload sizes.
- **Unsupervised Autotuning Framework:** A specialized training technique that efficiently clusters datasets to code versions and tunes predicate thresholds without requiring manual intervention or exhaustive search.
- **Empirical Validation:** Comprehensive evaluation on NVIDIA and AMD GPUs demonstrating significant speedups over baseline compiler heuristics and competitive performance against hand-optimized OpenCL/CUDA implementations across financial and Rodinia benchmarks.

## Technical Approach
The technique operates on a functional language with second-order array combinators (SOACs) like `map`, `reduce`, and `scan`. It extends traditional "moderate flattening" with a set of inference rules (G0–G9) that recursively transform nested parallelism. The core transformation (Rule G3) triggers when a nested parallel construct is encountered, producing three guarded branches:
1. **Top-level mapping (`etop`):** Maps the current parallelism to the target hardware level and sequentializes the inner body.
2. **Middle mapping (`emiddle`):** Maps the current level to hardware, then recursively attempts to map the inner parallelism to the next lower hardware level.
3. **Full flattening (`eflat`):** Continues flattening at the current hardware level without sequentializing inner work.

The target architecture is modeled with two parallelism levels: grid-level (`l=1`) and workgroup-level (`l=0`), where level 0 maps to fast local/shared memory for locality optimization. The generated branches are wrapped in `if-then-else` predicates comparing symbolic parallelism sizes (e.g., `numS * numX`) against tunable thresholds (`t_top`, `t_intra`). Thresholds are optimized using OpenTuner with a custom cost function that caches results for equivalent decision-tree paths, drastically reducing search space redundancy. The entire pipeline is integrated into the Futhark compiler, which handles defunctionalization, fusion, and OpenCL code generation.

## Results
- Evaluated on an NVIDIA K40 and AMD Vega 64 using three real-world financial benchmarks (FinPar suite) and six Rodinia benchmarks.
- Autotuned Incremental Flattening (AIF) consistently outperforms Moderate Flattening (MF), achieving **2× to 5× speedups** on average, and matches or exceeds hand-written OpenCL implementations in most cases.
- Demonstrates strong performance portability: AIF automatically adapts to architectural differences (e.g., favoring local memory on Vega 64 vs. global memory on K40) without code changes.
- The matrix multiplication case study shows AIF dynamically switching between fully parallel and locality-optimized versions based on matrix dimensions, closely tracking cuBLAS performance across varying shapes.
- Trade-offs are manageable: binaries are ~3× larger and compilation takes ~4× longer than MF, while autotuning typically converges in under one minute per benchmark.

## Relevance
- **Language Design:** Demonstrates how high-level functional/array languages can retain strong abstractions while achieving hardware-specific performance through compiler-driven multi-versioning, reducing the need for manual kernel specialization.
- **Compilers:** Introduces a novel alternative to static polyhedral scheduling and traditional autotuning by combining incremental transformation with runtime predicate dispatch. Highly relevant for GPU code generation, auto-vectorization, and performance-portable compiler design.
- **ML Systems:** Directly applicable to ML frameworks (e.g., JAX, PyTorch, TVM) that rely on nested parallelism for batched operations, attention mechanisms, and graph computations. The technique could automate dynamic kernel selection based on tensor shapes and batch sizes, improving performance portability across diverse accelerators without manual tuning.
