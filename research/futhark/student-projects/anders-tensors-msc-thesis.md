# Optimizing Tensor Contractions for GPU Execution in Futhark

## Metadata
- **Authors:** Anders L. Holst
- **Venue/Year:** Master's thesis, University of Copenhagen, 2024-05-31

## Summary
This thesis explores the generalization of block/register tiling optimization for tensor contractions in the Futhark compiler. Tensor contractions are fundamental operations in computational sciences with high computational complexity, making them ideal candidates for GPU acceleration. While Futhark already implements efficient 2D block/register tiling for matrix multiplication-like expressions, this work extends the optimization to arbitrary tensor contractions. The implementation uses generic LMAD (Linear Memory Access Descriptor) copies to stage data between global and shared memory, along with several optimizations including shared memory padding to avoid bank conflicts and special case handling for regular matrix multiplication. The work documents both successful implementation strategies and encountered challenges, providing a foundation for future development.

## Key Contributions
- Implementation of block/register tiling for arbitrary tensor contractions in Futhark
- Use of generic LMAD copies for efficient data staging between global and shared memory
- Shared memory padding optimization to avoid bank conflicts
- Special case optimization for regular matrix multiplication
- Comprehensive benchmarking against reference implementations (COGENT and prototype kernel)
- Documentation of implementation challenges and limitations

## Technical Approach
The implementation generalizes the existing BlkRegTiling module to handle tensor contractions of arbitrary dimensionality. The approach involves:

1. **Pattern Matching**: Identifying tensor contraction expressions in the IR that match specific firing conditions, including having exactly two redomap input arrays and a single reduction dimension.

2. **Tile Parameterization**: Using T (parallel), R (register), and Q (reduction) tiles to partition the iteration space, with the number of tiles determined by the number of free indices in the contraction.

3. **Memory Staging**: Implementing LMAD-based copies from global to shared memory, with padding to avoid bank conflicts and virtualization loops to handle partial tiles.

4. **Register Tiling**: Mapping the innermost parallel dimensions to thread-private registers for optimal data reuse.

5. **Boundary Handling**: Implementing prologue/epilogue treatment for partial tiles in the reduction dimension, with an option to disable this optimization when unnecessary.

The implementation faces challenges in extracting array layout information from the IR, particularly when arrays have been rearranged using sequences of operations rather than single permutations.

## Results
Benchmarking results show the implementation performs well, reaching between 68% and 98% of reference programs (COGENT and prototype kernel). Key findings include:

- The implementation shows definite potential for the chosen strategy
- There is significant room for micro-optimization compared to the prototype
- The special case MM optimization substantially improves performance over the generic LMAD copy approach
- Performance gains are particularly notable for problem instances with large reduction dimensions
- The implementation reaches 90-98% of BlkRegTiling performance for matrix multiplication cases

## Relevance
This work is highly relevant to language design, compilers, and ML systems for several reasons:

1. **Compiler Optimization**: Demonstrates how to extend existing compiler optimizations (block/register tiling) to more general cases, providing insights for other compiler developers.

2. **GPU Programming**: Addresses key challenges in GPU programming including memory hierarchy optimization, coalesced access patterns, and bank conflict avoidance.

3. **Tensor Operations**: Tensors and tensor contractions are fundamental to machine learning and scientific computing, making this optimization directly applicable to ML systems.

4. **Code Generation**: The use of LMADs for generic memory operations provides a template for handling arbitrary-rank data structures in code generation.

5. **Performance Portability**: Shows how to balance between generic solutions and special-case optimizations to achieve good performance across different problem instances.

The work provides both practical implementation insights and theoretical foundations for extending compiler optimizations to more complex mathematical operations.
