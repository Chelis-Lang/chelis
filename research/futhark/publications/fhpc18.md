# Modular Acceleration: Tricky Cases of Functional High-Performance Computing

## Metadata
- **Authors:** Troels Henriksen, Martin Elsman, Cosmin Oancea
- **Venue/Year:** FHPC ’18 (7th ACM SIGPLAN International Workshop on Functional High-Performance Computing), 2018

## Summary
This paper investigates the practical challenges and performance potential of using high-level, data-parallel functional programming for GPU acceleration. Focusing on the Futhark language, the authors present three non-trivial case studies: an irregular graph algorithm (Breadth-First Search), a quasi-random number generator (Sobol sequences), and a financial parameter calibration routine (Heston model). The work demonstrates that while functional abstractions elegantly express parallelism, achieving competitive performance on GPUs requires careful interaction between algorithmic restructuring and compiler optimizations.

The core argument is that modular, hardware-independent code can match or exceed hand-tuned imperative GPU implementations (e.g., OpenCL) when supported by a compiler capable of aggressive fusion, flattening, and granularity control. The paper highlights that maximizing parallelism is not always optimal; instead, programmers and compilers must strategically balance parallel execution with sequentialization to avoid thread divergence, memory overhead, and synchronization costs.

This work matters because it bridges the gap between theoretical functional parallelism and real-world GPU programming constraints. It provides empirical evidence that high-level languages can preserve software modularity and developer productivity without sacrificing performance, provided the compiler exposes the right transformation primitives and the programming model allows fine-grained control over parallelism granularity.

## Key Contributions
- Demonstrates that high-level functional data-parallel code can achieve 1.1×–7.3× speedups over hand-optimized OpenCL baselines across diverse, irregular workloads.
- Introduces and analyzes compiler fusion rules for `scatter`/`scattermap` operations, enabling efficient in-place updates for irregular graph algorithms.
- Presents a modular, purely functional Sobol sequence library using a novel `stream_map` construct that combines independent and recurrent formulas for optimal performance.
- Validates the necessity of compiling *regular nested parallelism* across abstraction boundaries, showing that manual flattening destroys code reusability while naive parallelization leaves hardware underutilized.
- Provides practical guidelines on when to exploit full parallelism versus sequentializing inner loops to avoid GPU overhead, emphasizing that more parallelism does not always equal better performance.

## Technical Approach
The paper leverages **Futhark**, a purely functional, data-parallel language with a compiler that targets GPUs via OpenCL/CUDA. The technical approach centers on three compiler and language mechanisms:
1. **Aggressive Fusion & `scattermap`:** The compiler transforms high-level `scatter` operations into a fused `scattermap` IR construct. Vertical and horizontal fusion rules combine multiple parallel operations (maps, filters, scatters) into single GPU kernels, minimizing global memory traffic and intermediate allocations.
2. **`stream_map` for Controlled Granularity:** For Sobol sequences, the authors introduce `stream_map`, which partitions input arrays into chunks processed in parallel. Each chunk uses an independent formula for its first element and a recurrent formula for the rest. This allows the compiler to fuse downstream reductions (`stream_red`) and avoid materializing large intermediate arrays.
3. **Moderate Flattening & Regular Nested Parallelism:** For the Heston calibration benchmark, the compiler supports *regular nested parallelism* (where inner loop bounds are independent of outer loop indices). Instead of fully flattening all parallelism (which breaks modularity) or executing only the outermost loop (which underutilizes hardware), the compiler applies "moderate flattening" to exploit parallelism across abstraction boundaries while allowing manual sequentialization of small inner loops (e.g., a `reduce` over 20 elements) to avoid segmented reduction overhead.

## Results
- **Breadth-First Search (BFS):** Four Futhark implementations were tested against the Rodinia OpenCL baseline across six datasets. The iterative/one-time splitting strategy (FV4) achieved the best overall performance, delivering up to **7.3× speedup** on highly skewed graphs while maintaining competitive performance on regular graphs. Aggressive padding suffered on skewed data, while full flattening incurred excessive overhead on uniform datasets.
- **Sobol Sequences:** The `stream_map` chunked implementation running on a GPU was **11.13× faster** than a fully parallel GPU version and matched sequential CPU performance. This confirms that controlling parallelism granularity prevents thread divergence and unnecessary memory bandwidth consumption.
- **Heston Calibration:** On a real-world dataset (1,062 quotes), exploiting nested parallelism across the least-squares solver and objective function yielded a **97.87× speedup** over sequential OCaml, compared to only 8.84× when using only inner parallelism. On larger synthetic datasets (10k–100k quotes), speedups reached **183×–236×**. Sequentializing a small inner `reduce` further improved performance by eliminating unnecessary GPU synchronization overhead.

## Relevance
This paper is highly relevant to **language design** and **compiler engineering** for parallel systems, as it demonstrates how high-level abstractions (uniqueness types, size-dependent types, and bulk-parallel operators) can be safely compiled to efficient GPU code without sacrificing modularity. For **ML systems and scientific computing**, the insights directly apply to optimizing Monte Carlo simulations, graph neural networks, and probabilistic programming frameworks. The findings on fusion, controlled parallelism granularity, and regular nested parallelism inform the design of modern tensor compilers (e.g., TVM, JAX, Triton), where balancing kernel fusion, memory coalescing, and thread divergence is critical for achieving peak hardware utilization.
