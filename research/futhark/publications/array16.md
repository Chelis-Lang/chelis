# Design and GPGPU Performance of Futhark’s Redomap Construct

## Metadata
- **Authors:** Troels Henriksen, Ken Friis Larsen, Cosmin E. Oancea
- **Venue/Year:** ARRAY '16 (ACM SIGPLAN International Workshop on Libraries, Languages and Compilers for Array Programming), 2016

## Summary
Programming modern GPGPUs efficiently remains challenging due to the complexity of low-level APIs (CUDA/OpenCL) and the difficulty of composing parallel primitives like `map` and `reduce` without incurring performance penalties from intermediate memory allocations or suboptimal memory access patterns. High-level functional array languages aim to solve this by allowing developers to write clear, composable code that compilers can automatically optimize for target hardware.

This paper introduces `redomap`, a novel second-order array combinator (SOAC) in the purely functional language Futhark. Rather than exposing `redomap` to users, the compiler automatically synthesizes it by aggressively fusing producer-consumer and horizontal `map`/`reduce` compositions. The construct supports returning both the final reduced value and the element-wise mapped results, and it correctly handles non-commutative reduction operators.

The authors detail a compilation strategy that translates `redomap` into highly efficient GPU kernels by chunking work to match hardware parallelism limits and strategically transposing arrays to guarantee coalesced global memory access. Empirical evaluation demonstrates that Futhark's automatically generated code consistently matches or significantly outperforms hand-tuned Thrust library implementations, validating the viability of high-level functional abstractions for high-performance GPGPU computing.

## Key Contributions
- **Extended `redomap` Operator:** A compiler-synthesized SOAC that unifies `map` and `reduce`, enabling both producer-consumer and horizontal fusion while allowing mapped results to be returned alongside the reduced accumulator.
- **GPU-Optimized Compilation Strategy:** A lowering technique that efficiently sequentializes excess parallelism and ensures coalesced global memory access, even for non-commutative operators, via strategic array transposition and chunked processing.
- **Comprehensive Performance Evaluation:** A reproducible mini-benchmark suite showing Futhark's auto-generated code achieves a geometric mean speedup of 1.75× over Thrust, with peak speedups reaching 8×.
- **Open Evaluation Infrastructure:** Public release of the benchmarking and compilation pipeline to encourage reproducibility and further research in functional GPGPU compilation.

## Technical Approach
- **Fusion Engine:** The compiler performs a bottom-up traversal of the program's dependency graph, fusing compatible SOACs. `redomap` extends prior work by relaxing type restrictions: it accepts an associative binary operator, a fold function, a neutral element, and input arrays, returning a tuple of the reduced accumulator and the mapped outputs. Horizontal fusion is enabled for independent SOACs operating on equally-sized arrays.
- **Chunked Execution & Parallelism Management:** Instead of naive tree reduction, the compiler partitions the input into chunks matching the GPU's optimal thread/workgroup count. Each thread sequentially processes its chunk using a `foldl`-style loop, eliminating excess parallelism and reducing synchronization overhead.
- **Coalesced Memory Access via Transposition:** To prevent non-coalesced memory accesses during sequential chunk processing, the compiler conceptually transposes the input array so that neighboring threads access contiguous memory locations. For non-commutative operators, an explicit in-memory transposition is performed before reduction to preserve evaluation order; the transposition is "undone" by the strided access pattern.
- **Kernel Generation:** `redomap` is lowered to a `reduceKernel` construct that processes chunks sequentially, storing intermediates in fast local memory. A final single-workgroup kernel reduces the per-workgroup partial results. Mapped outputs are generated in transposed form and transposed back if required by program semantics.

## Results
- Evaluated on a 9-program mini-benchmark suite covering simple reductions, tuple-based reductions, non-commutative operators (MSSP, 2×2 matrix multiplication), scans, and Black-Scholes option pricing.
- Tested on a GeForce GTX 780 Ti against both naive and hand-optimized Thrust implementations.
- Futhark consistently outperforms Thrust, particularly for tuple-based and non-commutative reductions: up to 14× faster for `IndexOfMax`, ~2.5× for `MSSP`, and ~8.8× for `Reduce2by2MM` at smaller array sizes.
- For `RedomapNT` (a complex fused workload), Futhark is ~2.3× faster than unfused Thrust and only ~1.2× slower than a manually split Thrust version.
- Disabling `redomap` fusion causes ~4× performance regressions in several benchmarks, confirming the critical role of fusion.
- Thrust's scan operator outperforms Futhark's in some cases due to transposition overhead in Futhark's implementation, highlighting a known trade-off in the current code generation strategy.

## Relevance
- **Language Design:** Demonstrates how high-level, purely functional abstractions can be designed to enable aggressive, semantics-preserving compiler optimizations without exposing complex intermediate constructs to end-users.
- **Compilers:** Provides a concrete case study in automatic parallelization, operator fusion, and memory layout transformations tailored for SIMD/GPU architectures. The techniques for handling non-commutative ops and ensuring coalesced access are directly applicable to modern polyhedral and ML compilers.
- **ML Systems:** Highly relevant to ML compiler stacks (e.g., XLA, TVM, Triton) that rely on operator fusion to minimize memory traffic and kernel launch overhead. The `redomap` concept closely mirrors fused operations like `reduce_sum` + `elementwise` in deep learning frameworks. The chunking and transposition strategies offer practical insights for optimizing custom GPU kernels for training/inference workloads where memory bandwidth is the primary bottleneck.
