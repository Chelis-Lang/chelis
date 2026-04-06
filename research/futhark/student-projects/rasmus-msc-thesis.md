# Generating Efficient Code for Futhark's Segmented Redomap

## Metadata
- **Authors:** Rasmus Wriedt Larsen
- **Venue/Year:** Master's Thesis, University of Copenhagen / 2017

## Summary
This thesis addresses the problem of generating efficient code for segmented reductions and segmented redomaps in the Futhark compiler. The current implementation uses a segmented scan approach, which is suboptimal for GPU execution due to excessive memory bandwidth usage. The author proposes a new approach using three specialized GPU kernels (large, small, and loop-in-map) that can handle different configurations of segment sizes and numbers of segments. The implementation is evaluated on benchmarks from Rodinia and Parboil, demonstrating significant speedups on Backprop and K-means benchmarks.

## Key Contributions
- Design and implementation of three specialized GPU kernels for segmented reductions and redomaps
- Development of a decision algorithm to select the optimal kernel based on segment size and number of segments
- Integration of the new implementation into the Futhark compiler
- Performance evaluation on real-world benchmarks showing speedups of up to 1.39× on Backprop and 1.36× on K-means
- Analysis of memory coalescing and occupancy optimization techniques for GPU programming

## Technical Approach
The thesis presents three GPU kernels for handling segmented reductions:
1. **Large kernel**: Uses multiple thread groups to cooperatively reduce large segments, with chunking to improve memory coalescing
2. **Small kernel**: Processes multiple small segments within a single thread group using segmented scan
3. **Loop-in-map kernel**: Sequentially processes each segment in a single thread, requiring array transposition for memory coalescing

The implementation uses Futhark's kernel extraction system to generate OpenCL code, with special handling for commutative vs non-commutative reductions and tuple-of-arrays transformations. The decision algorithm selects between kernels based on segment size, number of segments, and GPU occupancy considerations.

## Results
The implementation achieves significant performance improvements over the baseline segmented scan approach. For commutative reductions, the new implementation matches or exceeds the performance of one-dimensional reductions across all configurations. For non-commutative reductions, the implementation shows comparable performance with one-dimensional reductions when segments are large enough. The evaluation on four different GPUs demonstrates harmonic-mean speedups of 1.39× for Backprop and 1.36× for K-means benchmarks.

## Relevance
This work is highly relevant to language design and compiler optimization for parallel programming languages targeting GPUs. The techniques for handling segmented operations efficiently are applicable to other array programming languages and could inform the design of future GPU programming models. The analysis of memory coalescing, occupancy, and kernel selection strategies provides valuable insights for compiler writers targeting heterogeneous architectures.
