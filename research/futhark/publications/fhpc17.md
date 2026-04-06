# Strategies for Regular Segmented Reductions on GPU

## Metadata
- **Authors:** Rasmus Wriedt Larsen, Troels Henriksen
- **Venue/Year:** FHPC'17 (6th ACM SIGPLAN International Workshop on Functional High-Performance Computing), 2017

## Summary
This paper addresses the challenge of efficiently implementing regular segmented reductions on GPUs, a common parallel pattern where each inner array (segment) of a multidimensional array is reduced independently. Prior approaches either rely on a single strategy that performs poorly across diverse workloads, or map to irregular segmented reductions that introduce unnecessary overhead. The authors propose an adaptive approach that dynamically selects among three specialized kernel strategies based on runtime characteristics of the input (number of segments and segment size).

The technique is implemented in the Futhark compiler, a functional array language targeting GPUs. By automatically generating code for all three strategies and inserting a lightweight runtime dispatcher, the compiler achieves near-optimal performance across the entire design space of segment configurations. The approach eliminates the need for manual kernel tuning while maintaining high performance, bridging the gap between high-level functional abstractions and low-level GPU hardware efficiency.

This work matters because segmented reductions are foundational in data-parallel programming, yet their performance is highly sensitive to workload shape. By demonstrating that a compiler can automatically adapt to input characteristics without sacrificing performance, the paper provides a practical blueprint for building performance-portable parallel languages and libraries.

## Key Contributions
- **Three specialized GPU kernel strategies** for regular segmented reductions, each optimized for a distinct regime of segment count and segment size.
- **A runtime dispatch heuristic** that automatically selects the optimal strategy based on input dimensions, requiring no manual tuning.
- **Compiler integration in Futhark** that recognizes nested `map`/`reduce` patterns, fuses operations, and generates adaptive GPU code.
- **Comprehensive empirical evaluation** across synthetic microbenchmarks and real-world Rodinia applications, demonstrating consistent performance and 1.3×–1.7× speedups over prior scan-based implementations.

## Technical Approach
The core idea is to partition the workload space into three regimes and apply a tailored GPU execution strategy for each:
1. **Sequential Segments:** Launches one thread per segment, which sequentially reduces its assigned segment. Best when there are many segments. To avoid non-coalesced memory accesses, the compiler transposes the input array so that each thread reads contiguous memory.
2. **Large Segments:** Assigns multiple workgroups to a single segment, mimicking standard non-segmented reduction. Best for few, large segments. Threads chunk the input, reduce locally, and produce partial results that undergo a second-stage reduction. Supports efficient pipelining of memory transactions.
3. **Small Segments:** Packs multiple whole segments into a single workgroup. Best for few, small segments. Uses intra-workgroup segmented scans in fast shared memory, avoiding expensive inter-workgroup synchronization. Threads naturally access coalesced memory without transposition.

The Futhark compiler automatically detects perfectly nested reductions (via `map`/`reduce` fusion into a `redomap` construct) and generates all three kernels. At runtime, a simple heuristic selects the strategy:
- If segments > 2¹⁶ → Sequential Segments
- Else if segment size > ½ workgroup size → Large Segments
- Else → Small Segments

The compiler also handles non-commutative operators by applying index-space transformations to preserve reduction order while maintaining memory coalescing.

## Results
- **Microbenchmarks:** Evaluated on four workloads (Segmented Sum, Index of Max, Maximum Subarray Sum, Black-Scholes) across two total data sizes (2¹⁸ and 2²⁶ elements). Each strategy dominates its target regime, and the automatic selector tracks the optimal strategy closely across all segment/size ratios.
- **Comparison to CUB:** NVIDIA's CUB library performs well in its sweet spot but degrades significantly for edge cases (many small or few large segments) and fails entirely when segment count exceeds the GPU's maximum workgroup limit (2¹⁶). Futhark's adaptive approach remains robust.
- **Application Benchmarks:** On Rodinia's Backprop and K-means, replacing the previous scan-based segmented reduction with the adaptive approach reduced the time spent on segmented reductions from 19–35% down to 1–2%, yielding overall application speedups of 1.3× to 1.7×.
- **Overhead:** The runtime dispatch and strategy switching introduce negligible overhead, and the approach consistently matches or approaches the performance of hand-tuned non-segmented reductions.

## Relevance
This paper is highly relevant to **language design and compilers** because it demonstrates how high-level functional array languages can automatically lower nested parallel patterns to efficient, adaptive GPU code without exposing low-level hardware details to the programmer. The runtime dispatch mechanism offers a practical alternative to static autotuning, enabling performance portability across diverse input shapes.

For **ML systems**, regular segmented reductions appear frequently in batched operations, graph neural network pooling, ragged tensor processing (when padded to regular shapes), and attention mechanisms. The paper's strategies directly inform how ML compilers (e.g., TVM, XLA, Triton) can generate specialized kernels for batched reductions, handle memory coalescing for non-commutative ops, and dynamically adapt to varying batch/sequence lengths. The emphasis on avoiding scan-based overheads and leveraging workgroup-local fast memory aligns with modern GPU optimization practices critical for training and inference efficiency.
