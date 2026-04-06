# Static and Dynamic Analyses for Efficient GPU Execution

## Metadata
- **Authors:** Philip Munksgaard
- **Venue/Year:** PhD Thesis, University of Copenhagen, 2023

## Summary
This thesis presents a comprehensive set of static and dynamic techniques for optimizing GPU program execution. The work focuses on two main areas: (1) static memory analyses for parallel languages, and (2) autotuning threshold parameters in multi-versioned code. The research introduces novel approaches to memory optimization in functional array languages, particularly through the use of linear memory access descriptors (lmads) and an intermediate representation called FunMem. Additionally, it presents an autotuning framework that leverages monotonicity assumptions to automatically find near-optimal threshold values for multi-versioned code, significantly improving performance across various benchmarks.

## Key Contributions
- Introduction of lmads as a slicing mechanism and index functions in compiler IRs
- Development of FunMem, an intermediate representation with non-semantic memory information
- Array short-circuiting optimization for in-place updates and concatenations
- Memory block merging using graph coloring techniques
- Autotuning framework for multi-versioned code based on monotonicity assumptions
- Implementation and evaluation of all techniques in the Futhark programming language

## Technical Approach
The thesis introduces several key technical innovations:

1. **Linear Memory Access Descriptors (lmads)**: These are used to represent array access patterns and can be interpreted as sets of points, index functions, or slicing mechanisms. The thesis presents operations on lmads including normalization, expansion, and non-overlap testing.

2. **FunMem IR**: An intermediate representation that extends a functional language with memory annotations and allocations. FunMem allows the compiler to reason about memory usage while preserving high-level language semantics.

3. **Array Short-Circuiting**: This optimization identifies cases where intermediate arrays in parallel operations can be eliminated by constructing them directly in the destination memory space. It uses lmads to prove non-overlap of memory accesses.

4. **Memory Block Merging**: Inspired by register allocation, this technique merges non-interfering memory blocks to reduce overall memory usage.

5. **Autotuning Framework**: Based on monotonicity assumptions, this framework automatically finds optimal threshold values for multi-versioned code by analyzing the structure of tuning trees and exploiting parallelism characteristics.

## Results
The thesis demonstrates significant performance improvements across multiple benchmarks:

- **Array Short-Circuiting**: Speedups of 1.1× to 2× on six public benchmarks
- **Autotuning**: Up to 22.6× reduction in tuning time and up to 10× performance improvement
- **Memory Block Merging**: 6% reduction in memory usage for the OptionPricing benchmark

The techniques were implemented in the Futhark programming language and evaluated on NVIDIA A100 and AMD MI100 GPUs.

## Relevance
This thesis is highly relevant to language design, compilers, and ML systems work for several reasons:

1. **Memory Optimization**: The FunMem IR and array short-circuiting techniques provide a framework for optimizing memory usage in functional array languages, which is crucial for GPU performance.

2. **Autotuning**: The monotonicity-based autotuning approach offers a systematic way to optimize multi-versioned code, which is increasingly important as hardware heterogeneity grows.

3. **Compiler Design**: The work demonstrates how to extend functional languages with memory optimizations while preserving their high-level semantics, which is valuable for designing efficient compilers for parallel languages.

4. **GPU Programming**: The techniques address key challenges in GPU programming, such as memory coalescing and efficient use of local memory, which are critical for achieving high performance on modern GPUs.

5. **Language Features**: The introduction of lmads as a slicing mechanism and index functions shows how new language features can enable powerful optimizations while maintaining expressiveness.

Overall, this thesis provides valuable insights and techniques for improving the performance of GPU programs through advanced compiler optimizations and autotuning strategies.
