# Efficient GPU Implementation of Multi-Precision Integer Division

## Metadata
- **Authors:** Aske N. Raahauge, Martin B. Marchioro & Marc I. Løvenskjold (Supervisor: Cosmin E. Oancea)
- **Venue/Year:** Master’s Thesis, University of Copenhagen, Faculty of Science / 2025

## Summary
Multi-precision integer arithmetic is a foundational requirement for cryptographic protocols, scientific computing, and exact numerical methods. While GPU-optimized addition and multiplication have been extensively studied, parallel division remains underexplored due to its inherent algorithmic complexity and sequential dependencies. This thesis bridges that gap by presenting a highly optimized CUDA implementation of exact multi-precision division capable of handling integers up to ~250,000 bits (2¹⁸ bits), constrained only by the shared memory limits of a single CUDA block. By keeping all intermediate computations within the integer domain, the approach avoids the precision loss and domain-conversion overhead typical of floating-point reciprocal methods.

The core algorithm adapts Watt's "whole shifted inverse" technique, which reformulates Newton's method to operate entirely on integers. The authors identify and correct several undocumented edge cases in the original formulation, including handling negative intermediates, fixing initial approximation overestimation, and bounding errors introduced by divisor prefix optimizations. Alongside the low-level CUDA prototype, the thesis implements the algorithm in Futhark, a high-level functional array language, to evaluate how well modern parallel compilers handle complex, irregular workloads.

This work matters because it pushes the boundaries of what is feasible for exact arithmetic on GPUs, demonstrating that division can scale efficiently to unprecedented bit-widths when carefully mapped to GPU memory hierarchies. Furthermore, the comparative analysis between hand-tuned CUDA and compiler-generated Futhark code provides concrete insights into the limitations of current high-level parallel programming models, offering valuable feedback for compiler designers and systems researchers.

## Key Contributions
- **Algorithmic Refinements:** Corrects and extends Watt's whole shifted inverse algorithm to handle negative intermediates, initial approximation overestimation, and divisor-prefix-induced errors, guaranteeing exact quotient computation with δ ∈ {-1, 0, 1}.
- **High-Performance CUDA Kernel:** Implements a division routine supporting integers up to 2¹⁸ bits (~262k bits), the largest known parallel division implementation, leveraging single-block shared memory for temporal data reuse and warp-level primitives for carry propagation.
- **High-Level Futhark Implementation:** Provides a semantically equivalent functional implementation that exposes critical compiler limitations regarding irregular nested parallelism and dynamic shared memory allocation.
- **Comprehensive Benchmarking:** Evaluates performance against state-of-the-art libraries (NVIDIA's CGBN and GMP), demonstrating strong scalability for large inputs and establishing a baseline for division-to-multiplication overhead (~5× for large operands).
- **GCD Extension:** Extends the division kernel to compute the Greatest Common Divisor via the Euclidean algorithm, validating correctness and highlighting performance bottlenecks in shrinking-operand scenarios.

## Technical Approach
The algorithm computes `u quo v` by first calculating the whole shifted inverse `shinv_h(v) = ⌊B^h / v⌋` using an integer-adapted Newton iteration: `w_{i+1} = w_i + ⌊w_i(B^ℓ - ⌊vB^{-s}⌋w_i)B^{-k+s}⌋`. This avoids floating-point arithmetic entirely. The quotient is then derived as `q = shift^{-h}(u · shinv_h(v)) + δ`, with a final correction step to handle δ ∈ {-1, 0, 1}.

**CUDA Optimizations:**
- **Memory Hierarchy Exploitation:** All operations are confined to a single CUDA block to maximize shared memory (scratchpad) reuse. Integers are partitioned across threads with a sequentialization factor `Q=4`, enabling register-level scalarization and minimizing shared memory traffic.
- **Dynamic Multiplication Sizing:** Implements `smallMult`, `smallMult2x`, and `completeMult` to match operand sizes during Newton iterations, avoiding unnecessary full-width computations.
- **Warp-Level SOACs:** Replaces block-level scans/reduces with warp-shuffle intrinsics for carry propagation and comparison operations, reducing synchronization overhead and improving addition/subtraction throughput by ~10%.
- **Close Products & Divisor Prefixes:** Optimizes `B^h - v·w` by computing only the lower `L` digits when the product is close to `B^h`, and truncates the divisor to its most significant digits during early iterations to reduce work.

**Futhark Implementation:**
Translates the algorithm into pure functional array combinators. However, Futhark's incremental flattening compiler cannot express the variable-sized "shorter iterates" optimization due to irregular nested parallelism. Consequently, the Futhark version performs full-size multiplications at every step, degrading asymptotic complexity from `O(M(n))` to `O(M(n) log n)`.

## Results
- **Hardware & Setup:** Benchmarked on an NVIDIA A100 GPU, comparing 32-bit and 64-bit CUDA variants, Futhark (16-bit), CGBN, and GMP.
- **Performance Scaling:** The 64-bit CUDA implementation scales steadily with input size, reaching ~2.35 Gu32ops/s at 2¹⁸ bits. At 2¹⁵ bits, it is ~3× slower than CGBN but supports up to 2¹⁸ bits, whereas CGBN fails to compile beyond 2¹⁵ bits due to shared memory constraints.
- **Futhark Limitations:** The Futhark implementation is orders of magnitude slower and crashes at runtime for inputs >2¹³ bits due to excessive dynamic shared memory allocation and compiler inability to flatten irregular parallelism.
- **Overhead Analysis:** Division runtime stabilizes at ~5× the cost of a single full multiplication for large inputs, aligning with theoretical predictions. The GCD extension is functionally correct but suffers from high overhead due to fixed-size operations on progressively smaller operands.
- **CPU Comparison:** GMP (sequential CPU) is ~10× slower than the optimized CUDA kernel, confirming the value of GPU parallelism for large-integer division.

## Relevance
- **Compiler & Language Design:** The thesis provides a concrete case study of how high-level parallel languages (like Futhark) struggle with irregular nested parallelism and dynamic memory patterns. It highlights the need for compiler passes that can safely sequentialize or flatten variable-sized parallel loops, offering actionable feedback for array language and GPU compiler developers.
- **ML Systems & Cryptography:** Multi-precision arithmetic underpins RSA, elliptic curve cryptography, post-quantum schemes, and exact numerical methods in symbolic AI/ML. Efficient GPU division enables faster key generation, modular inversion, and large-integer linear algebra, which are increasingly relevant for privacy-preserving ML and cryptographic workloads.
- **GPU Systems Programming:** Demonstrates advanced techniques for maximizing occupancy, managing register spilling, coalescing global memory accesses, and leveraging warp-level primitives. These patterns are directly applicable to developers building custom high-performance kernels for linear algebra, arbitrary-precision math, or domain-specific accelerators.
