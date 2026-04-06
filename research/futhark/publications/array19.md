# Data-Parallel Flattening by Expansion

## Metadata
- **Authors:** Martin Elsman, Troels Henriksen, Niels Gustav Westphal Serup
- **Venue/Year:** ARRAY '19 (6th ACM SIGPLAN International Workshop on Libraries, Languages and Compilers for Array Programming), 2019

## Summary
The paper addresses the longstanding challenge of efficiently compiling irregular and nested data-parallel programs to modern parallel hardware like GPUs. While regular parallelism maps cleanly to flat execution models, irregular workloads suffer from high overhead when using traditional compiler-based flattening techniques (e.g., NESL-style transformations). To bridge this gap, the authors introduce "flattening-by-expansion," a programmer-level design pattern implemented as a higher-order generic function (`expand`) in the Futhark language. This function transforms irregular nested parallel constructs into flat, work-efficient parallel code using a small, well-understood set of segmented primitives.

By providing explicit, library-level control over the flattening process, the technique avoids the performance pitfalls of automatic compiler transformations while sparing developers from writing low-level, error-prone flattened code. The approach is demonstrated across several domains, including geometric rasterization, prime number sieving, and sparse linear algebra. It achieves asymptotic work efficiency and competitive runtime performance on GPUs, proving that manual, composable flattening can be both practical and highly performant for real-world irregular parallel problems.

## Key Contributions
- Introduces `expand`, a generic higher-order function that flattens irregular nested parallelism into flat parallel code without requiring language-level nested parallelism support.
- Provides a concrete, work-efficient implementation in Futhark built on segmented scans, scatter, and gather operations.
- Demonstrates the technique's versatility across multiple irregular problems: line/triangle rasterization, Sieve of Eratosthenes, sparse matrix-vector multiplication, and z-buffering.
- Extends the pattern to `expand_reduce` for combined expansion and per-segment aggregation operations.
- Shows that programmer-directed flattening can significantly outperform both naive parallel implementations and hand-tuned sequential CPU code on GPUs, while maintaining optimal asymptotic complexity.

## Technical Approach
The core of the technique is the `expand` function, which takes a size function `sz: a → i32` (how many outputs each input element produces) and a generator `get: a → i32 → b` (how to compute each output), transforming an input array `[]a` into a flat output array `[]b`. Its implementation relies on three key parallel steps:
1. **Size Mapping:** Compute the number of outputs per input element via `map sz arr`.
2. **Index Generation:** Use `repl_iota` to generate destination indices for the flat output array. This is implemented via a segmented scan over the repetition counts.
3. **Local Index Generation:** Use `segm_iota` to generate intra-segment indices (0, 1, 2... per segment) by detecting segment boundaries via index rotation and another segmented scan.
4. **Parallel Generation:** Apply the generator in parallel using `map2` over the source indices and local indices.

The algorithm achieves `O(M)` work and `O(log M)` span, where `M` is the total number of output elements. The design composes cleanly with standard parallel primitives (`map`, `filter`) via algebraic fusion rules, can be nested (e.g., triangles → lines → points), and extends to `expand_reduce` for operations requiring per-segment reductions (e.g., sparse matrix-vector multiplication).

## Results
The authors evaluated the technique on an NVIDIA RTX 2080 Ti GPU and Intel Xeon CPU using Futhark's OpenCL and C backends:
- **Sieve of Eratosthenes:** The flattened version computes primes up to 100M in 11.3ms, drastically outperforming the non-flattened Futhark version (171ms) and a sequential C implementation (530ms).
- **Sparse Matrix-Vector Multiplication (SMVM):** Scales efficiently with matrix density. For a 10k×10k matrix at 1% density, Futhark GPU takes 0.52ms vs 27.1ms for C. Dense multiplication only surpasses the sparse version above ~10% density due to the fixed overhead of segmented scans.
- **Graphics/Rasterization:** Successfully renders 500k triangles at 15 FPS on a laptop GPU using `reduce_by_index` for z-buffering. While not matching dedicated graphics APIs, it demonstrates viable real-time performance for general-purpose parallel rendering.
Overall, the technique consistently delivers work-efficient, high-performance GPU code without requiring manual low-level kernel optimization.

## Relevance
- **Language Design:** Demonstrates how to expose flattening as a first-class, composable library abstraction rather than relying solely on opaque compiler passes. This "pay-as-you-go" approach gives programmers explicit control over parallelism granularity, memory layout, and work distribution.
- **Compilers:** Highlights the practical limits of automatic flattening on GPUs and underscores the importance of fusion, segmented operations, and work-efficient transformations. It provides a blueprint for hybrid compiler/runtime strategies where high-level patterns are lowered to optimized flat kernels.
- **ML Systems:** Highly relevant for handling irregular workloads ubiquitous in modern ML, such as graph neural networks (variable node degrees), sparse tensor operations, dynamic computation graphs, and ragged batch processing. The `expand`/`expand_reduce` patterns directly map to scatter/gather and segmented reduction primitives used in frameworks like PyTorch, JAX, and TVM, offering a principled approach to implementing efficient, high-level irregular parallel algorithms without dropping to custom CUDA kernels.
