# Extending Futhark with a write construct

## Metadata
- **Authors:** Niels G. W. Serup
- **Venue/Year:** University of Copenhagen, 2016

## Summary
This paper presents the design and implementation of a new `write` construct for the Futhark programming language, a data-parallel, purely functional array programming language that compiles to efficient OpenCL code for GPUs. The `write` construct allows for bulk parallel writes to arbitrary array indexes, addressing a key limitation in Futhark's expressiveness. The author demonstrates the construct's utility through several applications including radix sort, filter, and a breadth-first search (BFS) benchmark ported from the Rodinia suite. The implementation is shown to be correct and performant, with the BFS port achieving speeds within a factor of three of the original CUDA code.

## Key Contributions
- Design and implementation of the `write` construct for Futhark
- Demonstration of `write`'s utility through applications like radix sort and filter
- Porting of the Rodinia BFS benchmark to Futhark using `write`
- Implementation of optimizations including map-write fusion, write-write fusion, and iota/replicate elimination
- Evaluation showing Futhark's BFS implementation achieves competitive performance

## Technical Approach
The `write` construct takes tuples of index arrays and value arrays, along with one or more target arrays, and performs parallel writes to the specified indexes. The implementation uses Futhark's uniqueness types to ensure safe in-place updates. Internally, `write` is represented using a Lambda structure to enable fusion optimizations. The author implements several fusion optimizations: map-write fusion (combining a map and write into a single operation), write-write fusion (combining multiple writes), and elimination optimizations for `iota` and `replicate` expressions. The BFS benchmark is ported using two main approaches: array padding to ensure equal-sized subarrays, and segmented operations using masks to track subarray boundaries.

## Results
The author evaluates the BFS implementation on several datasets, comparing three Futhark ports (simple padding-based, segmented operations, and an alternate segmented approach) against the original Rodinia CUDA implementation. The simple padding-based approach with all optimizations enabled performs best, achieving speeds within a factor of three of Rodinia on the largest dataset. The Futhark implementation outperforms Rodinia on two smaller, more irregular datasets designed to expose unparallelized code in the CUDA version. The author also demonstrates the correctness of the implementation through testing on multiple datasets.

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work as it addresses a fundamental limitation in data-parallel functional languages - the inability to express arbitrary parallel writes. The techniques for implementing and optimizing the `write` construct, particularly the fusion optimizations and use of uniqueness types for safe in-place updates, provide valuable insights for designing expressive and efficient data-parallel languages. The evaluation methodology and benchmark porting approach are also instructive for researchers working on GPU programming languages and compilers.
