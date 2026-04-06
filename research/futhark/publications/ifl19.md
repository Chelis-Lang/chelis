# A Functional Approach to Accelerating Monte Carlo based American Option Pricing

## Metadata
- **Authors:** Wojciech Michal Pawlak, Martin Elsman, Cosmin Eugen Oancea
- **Venue/Year:** IFL '19 (International Symposium on Implementation and Application of Functional Languages) / Published 2020

## Summary
Pricing American options is a computationally intensive task in quantitative finance due to the embedded optimal stopping problem, which lacks a closed-form solution. The industry standard relies on the Longstaff-Schwartz Monte Carlo (LSMC) algorithm, which uses backward induction and least-squares regression to estimate continuation values. Traditionally, high-performance implementations of LSMC are written in low-level, hardware-specific languages like CUDA. While performant, these implementations are difficult to maintain, require specialized GPU programming expertise, and create a barrier between financial domain experts and the underlying code.

This paper demonstrates that a high-level functional data-parallel language, Futhark, can express the LSMC algorithm cleanly while achieving performance competitive with hand-tuned CUDA code. The authors detail the algorithmic transformations required to make the regression step GPU-efficient, such as precomputing small matrix decompositions and trading redundant compute for reduced global memory traffic. By leveraging Futhark's optimizing compiler, the implementation automatically maps nested parallelism to the GPU without manual kernel tuning or explicit memory management.

The work matters because it bridges the gap between high-level domain-specific expressiveness and low-level hardware performance. It proves that functional array languages, combined with aggressive compiler optimizations, can deliver state-of-the-art execution times for complex financial simulations. This approach significantly improves code maintainability, portability across architectures, and accessibility for quantitative analysts who need to rapidly prototype and modify pricing models.

## Key Contributions
- A high-level, data-parallel Futhark implementation of the Longstaff-Schwartz (LSMC) algorithm for American option pricing, serving as a portable reference.
- Detailed algorithmic refinements to adapt LSMC for massive GPU parallelism, including QR decomposition, precomputation of SVD for small matrices, and on-the-fly computation to minimize memory bandwidth.
- Empirical demonstration that the Futhark implementation matches or exceeds a hand-optimized CUDA benchmark (up to 2.5× faster in specific configurations) while maintaining numerical accuracy.
- A case study showcasing Futhark's compiler capabilities (fusion, flattening, auto-tuning) for real-world, irregular parallel workloads in computational finance.

## Technical Approach
The core algorithm is the LSMC method, which simulates asset price paths forward in time and then works backward to estimate continuation values via least-squares regression. The naive approach computes the full pseudo-inverse $(A^T A)^{-1} A^T$ at each time step, which is prohibitively expensive on GPUs. The authors optimize this by:
1. **QR Decomposition & SVD Precomputation:** Decomposing the design matrix $A = QR$ reduces the problem to inverting a small $3 \times 3$ upper-triangular matrix $R$. The SVD of $R$ is computed in parallel for all time steps before the main backward loop.
2. **On-the-Fly Computation:** Instead of materializing the large orthogonal matrix $Q^T$, it is computed dynamically during the main loop using precomputed $R^{-1}$ and spot prices. This trades extra arithmetic for drastically reduced global memory traffic.
3. **Memory Layout & Fusion:** Path data is transposed to enable coalesced GPU memory accesses. Random number generation is fused with path simulation to eliminate intermediate memory writes.
4. **Compiler-Driven Parallelism:** The implementation uses Futhark's Second-Order Array Combinators (SOACs) like `map`, `reduce`, `scan`, and `scatter`. The compiler applies aggressive fusion, moderate/incremental flattening, and multi-version code generation to automatically map nested parallelism to GPU execution units without manual kernel configuration.

## Results
- **Hardware & Setup:** Evaluated on an NVIDIA Tesla V100 GPU, compared against a reference CUDA implementation optimized by NVIDIA engineers.
- **Accuracy:** Pricing results match established benchmarks (finite difference methods, binomial trees, and the original Longstaff-Schwartz paper) with negligible error, validating numerical correctness.
- **Performance:** Prices a put option with 1M paths and 100 time steps in ~17 ms. Overall execution time is comparable to the CUDA reference.
- **Scalability:** Outperforms the CUDA baseline by up to **2.5×** for small numbers of time steps (e.g., 10). Performance converges at ~100 time steps and remains within a small margin for larger step counts. The SVD preparation and main regression loop consistently match or beat the hand-tuned CUDA version, with minor overhead in path generation attributed to RNG implementation differences.
- **Portability:** Identical performance observed across Futhark's CUDA and OpenCL backends, demonstrating architecture-agnostic compilation.

## Relevance
- **Language & Compiler Design:** Validates the practicality of high-level functional array languages for complex, irregular workloads. Highlights how compiler-driven optimizations (fusion, flattening, auto-tuning) can replace manual low-level tuning, reducing the expertise barrier for HPC.
- **ML & Probabilistic Systems:** The techniques (precomputing small linear algebra decompositions, on-the-fly matrix operations to save memory bandwidth, fusing stochastic sampling with simulation) are directly applicable to Monte Carlo methods in ML, Bayesian inference, reinforcement learning, and probabilistic programming frameworks.
- **Performance Portability:** Demonstrates a viable path for domain experts (finance, science, ML) to write maintainable, high-level code that automatically compiles to efficient GPU kernels, aligning with modern trends in differentiable programming and tensor compiler stacks (e.g., TVM, JAX, Triton).
