# Regular Segmented Single-pass Scan in Futhark

## Metadata
- **Authors:** Morten Clausen
- **Venue/Year:** Master's Thesis, University of Copenhagen, 2021

## Summary
This thesis presents an implementation of a single-pass scan algorithm for regular segmented scans in the Futhark compiler. The work extends a prior non-segmented implementation by Persson & Nicolaisen to handle regular segments, where segments are of equal length. The implementation is based on Merrill & Garland's single-pass scan algorithm with decoupled lookback, which reduces memory movements compared to traditional two-pass approaches. The thesis makes two major contributions: an analytical model for determining optimal sequential work per thread based on type analysis, and optimizations for index calculations to minimize costly 64-bit modulo operations. The implementation is validated through benchmarking on various applications including longest satisfying streak problems, radix-sort, and KD-tree construction.

## Key Contributions
- Generalization of a single-pass scan implementation to handle regular segmented scans
- Analytical model for dynamically choosing optimal sequential work per thread based on operator type analysis
- Optimization of index calculations to reduce 64-bit modulo operations
- Implementation in the Futhark compiler with empirical validation
- Benchmarks showing significant performance improvements over two-pass approaches

## Technical Approach
The implementation follows Merrill & Garland's single-pass scan algorithm with decoupled lookback, adapted for regular segments. The algorithm divides work among threads, with each thread performing sequential work on a slice of elements. Key components include:

1. **Load & Map**: Elements are loaded from global memory with coalesced access patterns, applying any necessary mapping operations
2. **Transposition**: Elements are transposed in shared memory to group consecutive elements together
3. **Thread-Level Scan**: Each thread performs a sequential scan on its assigned elements, handling segment boundaries
4. **Block-Level Scan**: A scan is performed across threads within a block using shared memory
5. **Lookback Phase**: Blocks calculate prefix values by inspecting previous blocks' results, with optimizations for segment boundaries
6. **Result Distribution**: Final results are computed by combining prefix values with local scan results

The implementation includes optimizations for index calculations, transforming costly 64-bit modulo operations into more efficient 32-bit arithmetic where possible.

## Results
Benchmark results show significant performance improvements:

- **Segmented sums**: 4.0x-4.6x speedup on GTX 780 Ti, 2.9x-3.5x on RTX 2080 Ti compared to two-pass approach
- **Segmented radix-sort**: 1.2x-2.4x speedup depending on hardware and segment size
- **KD-tree construction**: 1.29x-1.91x speedup

The performance gains are more pronounced on hardware with greater memory resources, as the single-pass approach reduces inter-thread communication. The analytical model for determining sequential work per thread proves effective, with dynamically chosen values outperforming static configurations in most cases.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates how to efficiently implement fundamental parallel primitives that can serve as building blocks for more complex parallel algorithms. The techniques for optimizing memory access patterns, reducing inter-thread communication, and dynamically tuning thread behavior based on operator characteristics are broadly applicable to parallel programming systems. The implementation in Futhark, a functional array language with GPU compilation capabilities, shows how high-level language features can be compiled to efficient low-level code while maintaining correctness and performance.
