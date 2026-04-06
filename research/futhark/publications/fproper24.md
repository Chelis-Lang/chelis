# A Comparison of OpenCL, CUDA, and HIP as Compilation Targets for a Functional Array Language

## Metadata
- **Authors:** Troels Henriksen
- **Venue/Year:** FProPer ’24 (1st ACM SIGPLAN International Workshop on Functional Programming for Productivity and Performance) / 2024

## Summary
This paper presents a systematic performance comparison of OpenCL, CUDA, and HIP as compilation targets for Futhark, a purely functional array language designed for data-parallel programming. Because GPUs are notoriously difficult to program directly, high-level languages like Futhark compile to vendor-specific APIs. The study investigates whether the choice of backend API significantly impacts runtime performance when the underlying generated GPU code is nearly identical across targets. By benchmarking 48 real-world applications on both NVIDIA (A100) and AMD (MI100) hardware, the author demonstrates that substantial performance variations exist despite equivalent source-level optimizations.

The research reveals that performance differences—ranging from 0.42× to 1.72×—are rarely caused by the quality of the generated kernel code itself. Instead, they stem from API-level defaults, hardware introspection limitations, missing language features, and runtime overhead. For example, OpenCL’s default floating-point behavior, restrictive thread block sizes on AMD, and lack of a memory model supporting single-pass scans lead to measurable slowdowns. Conversely, OpenCL occasionally outperforms CUDA/HIP due to looser numerical precision defaults or coincidental scheduling heuristics.

This work matters because it exposes the fragility of "performance portability" in GPU programming. It provides compiler developers and language designers with actionable insights into the hidden costs and inconsistencies of targeting multiple GPU APIs. The findings underscore that achieving consistent cross-platform performance requires more than a shared optimization pipeline; it demands careful handling of API defaults, dynamic hardware queries, and backend-specific algorithmic adaptations.

## Key Contributions
- A comprehensive empirical evaluation of OpenCL, CUDA, and HIP backends for the Futhark compiler across 48 benchmarks on modern NVIDIA and AMD GPUs.
- Identification and categorization of the primary root causes for cross-backend performance discrepancies, including numerical defaults, scan algorithm differences, thread block limits, missing atomic operations, and API overhead.
- Demonstration that identical high-level code and shared compiler optimizations do not guarantee performance portability due to API-specific runtime behaviors and hardware introspection gaps.
- Public release of the full experimental infrastructure and benchmark suite to ensure reproducibility and facilitate future cross-platform GPU compiler research.

## Technical Approach
The Futhark compiler employs a unified, aggressively optimizing ahead-of-time pipeline that diverges only at the final code generation stage. A minimal C host-side abstraction layer standardizes memory management and kernel launching across all three backends, with CUDA and HIP utilizing their lower-level driver APIs to match OpenCL’s verbosity and control. All backends rely on runtime compilation (via OpenCL’s native API, NVRTC, and HIPRTC), which enables dynamic tuning of parameters like thread block sizes and facilitates loop unrolling based on runtime constants.

Parallel constructs (maps, scans, reductions, histograms) are mapped to GPU kernels using a shared strategy, with one critical algorithmic divergence: OpenCL uses a two-pass scan due to memory model limitations, while CUDA and HIP leverage a single-pass decoupled lookback algorithm. The compiler also employs incremental flattening to map nested parallelism to flat GPU execution, and uses hardware queries to optimize thread counts and cache-aware multi-pass histograms. Performance is measured via wall-clock runtime across multiple workloads, excluding initialization and host-device data transfers to isolate kernel and API execution costs.

## Results
Across 48 benchmarks, OpenCL’s performance relative to CUDA (on A100) and HIP (on MI100) varied between 0.42× and 1.72×. The majority of discrepancies were traced to identifiable factors:
- **Numerical Defaults:** OpenCL’s faster but less precise single-precision division/sqrt defaults gave it an edge in compute-bound kernels (e.g., `mandelbrot`, `nbody`), but aligning precision removed this advantage.
- **Algorithmic Gaps:** OpenCL’s two-pass scan significantly slowed scan-heavy workloads (e.g., BFS, `convexhull`, `radix_sort`).
- **Hardware Limits & Queries:** AMD’s OpenCL implementation capped thread blocks at 256, hurting locality in nested parallelism. Inaccurate L2 cache and thread capacity queries in OpenCL led to suboptimal histogram and reduction configurations.
- **API Overhead:** OpenCL exhibited higher CPU-side overhead, disproportionately affecting short-running kernels or those with frequent host-device synchronization.
- **Unexplained Variance:** Some compute-bound kernels showed unexplained differences, likely due to backend compiler variations in register allocation and instruction scheduling.
Overall, CUDA and HIP provided more predictable performance, while OpenCL’s portability came at the cost of performance consistency without manual tuning or auto-tuning.

## Relevance
This paper is highly relevant to **language design and compiler engineering** because it highlights the practical limits of performance portability when targeting heterogeneous GPU ecosystems. It demonstrates that compiler writers must account for API-specific defaults, runtime compilation trade-offs, and hardware introspection capabilities, not just IR-level optimizations. For **ML systems and frameworks** (e.g., PyTorch, JAX, Triton), the findings underscore the importance of backend-aware tuning, consistent numerical precision handling, and the hidden costs of cross-platform abstraction layers. The study also reinforces the value of runtime compilation and auto-tuning in modern compiler stacks to dynamically adapt to hardware-specific constraints, offering a methodological blueprint for diagnosing and mitigating cross-backend performance gaps in high-performance computing.
