# Implementing Single-Pass Scan in the Futhark Compiler

## Metadata
- **Authors:** Andreas Nicolaisen, Marco Aslak Persson
- **Venue/Year:** Not specified

## Summary
This report presents the implementation of a single-pass parallel scan algorithm in the Futhark compiler, based on the work by Duane Merrill and Michael Garland. The algorithm achieves efficient scan operations by performing all computations within a single kernel, reducing global memory accesses from four to two per element. The implementation supports arbitrary scan operators and allows fusion with map operations. The authors provide a detailed explanation of the algorithm's phases, including dynamic block numbering, transposition, per-thread scanning, block-level scanning, and inter-block communication. Benchmarks show significant performance improvements over the previous Futhark implementation, achieving 84.8% of memcpy performance on a 1GiB input.

## Key Contributions
- Implementation of a single-pass scan algorithm in the Futhark compiler
- Support for arbitrary scan operators and fusion with map operations
- Detailed explanation of the algorithm's phases and implementation
- Performance evaluation showing significant speedup over previous implementation
- Discussion of limitations and potential improvements

## Technical Approach
The algorithm uses a single kernel with the following key phases:
1. **Dynamic Block Numbering**: Each block atomically acquires a unique ID to prevent deadlocks
2. **Global Load and Mapping**: Elements are read from global memory in coalesced fashion and mapped
3. **Transposition**: Elements are rearranged using shared memory to enable sequential processing
4. **Per-Thread Scan**: Each thread sequentially scans its assigned elements
5. **Block-Level Scan**: The last elements from each thread are scanned to compute block prefixes
6. **Lookback Phase**: Blocks communicate to compute inclusive prefixes of previous blocks
7. **Distribution**: Prefixes are distributed to elements
8. **Global Write**: Results are written back to global memory in coalesced fashion

The implementation uses shared memory for transposition and intermediate results, and register memory for per-thread computations. The number of elements processed per thread (M) is determined by available shared and register memory.

## Results
The implementation was benchmarked against the previous Futhark implementation and a reference OpenCL solution:
- **Simple-1GiB**: 15980µs (GTX 780Ti) vs 50150µs (Futhark), achieving 3.14x speedup
- **Advanced-100MiB**: 3109µs (GTX 780Ti) vs 7593µs (Futhark), achieving 2.44x speedup
- **Radix-100MiB**: 136568µs (GTX 780Ti) vs 230707µs (Futhark), achieving 1.69x speedup
- Performance reaches 84.8% of memcpy on GTX 780Ti for simple scan

## Relevance
This work is highly relevant to language design, compilers, and ML systems because:
1. **Language Design**: Demonstrates how to expose scan as a first-class primitive in a functional array language
2. **Compiler Implementation**: Shows how to generate efficient GPU code for complex parallel patterns
3. **ML Systems**: Scan operations are fundamental to many ML algorithms (e.g., prefix sums in sorting, cumulative operations)
4. **Performance Optimization**: Illustrates techniques for reducing global memory accesses and improving GPU utilization
5. **Fusion Support**: Demonstrates how to fuse scan with map operations while maintaining efficiency

The implementation provides a foundation for efficient scan operations in data-parallel programming languages and serves as a reference for compiler writers targeting GPU architectures.
