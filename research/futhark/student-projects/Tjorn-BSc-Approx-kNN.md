# Data-parallel Implementation of Randomized Approximate Nearest Neighbours

## Metadata
- **Authors:** Tjørn Lynghus
- **Venue/Year:** Bachelor thesis, Datalogisk Institut, June 11, 2023

## Summary
This thesis presents a parallel implementation of the Randomized Approximate Nearest Neighbours (RANN) algorithm using the Futhark programming language. RANN is an approximate k-nearest neighbors (k-NN) algorithm that combines fast random rotations with k-d trees to achieve significant speedups compared to exact k-NN methods. The implementation leverages GPU parallelism to improve runtime performance. The thesis describes the algorithm, its parallel implementation, and evaluates its performance against a parallel brute-force exact k-NN implementation. Results show that the parallel RANN implementation with T=1 is 20-140 times faster than the parallel brute-force approach, while maintaining high accuracy (up to 100% with increased T values).

## Key Contributions
- A parallel implementation of RANN using Futhark
- Detailed description of the algorithm's steps and their parallel implementation
- Performance evaluation comparing RANN to parallel exact k-NN
- Analysis of the impact of various parameters (T, leaf size, dimensionality) on accuracy and runtime
- Demonstration that RANN achieves high accuracy (up to 100%) with appropriate parameter tuning

## Technical Approach
The implementation follows the RANN algorithm which consists of several key steps:
1. **Shifting points**: Subtract the center of mass from all points
2. **Random orthogonal transformation**: Apply a transformation consisting of permutations (P), rotations (Q), and fast Fourier transforms (F)
3. **Building k-d tree**: Construct a balanced k-d tree from the transformed reference points
4. **Searching k-d tree**: Find the "natural leaf" for each query and collect nearby leaves (Vi)
5. **Searching leaves**: Use brute-force search within Vi to find k-NN candidates
6. **Loop**: Repeat steps 2-5 T times to improve accuracy
7. **Supercharging**: Perform depth-one search on candidates to further improve accuracy

The implementation uses Futhark's parallel constructs including map, reduce, scan, and scatter operations. Key optimizations include sorting queries by their natural leaves for better memory access patterns and early exit in the brute-force search when distances exceed current worst candidate.

## Results
The evaluation shows that RANN with T=1 achieves 20-140x speedup over parallel exact k-NN, with the highest speedups at higher dimensions (d=50) and lower k values. Increasing T significantly improves accuracy, with the algorithm achieving near 100% accuracy at T=10. Supercharging further improves accuracy but at a substantial runtime cost. The default leaf size of 256 provides a good balance between accuracy and runtime. Sorting queries by their natural leaves provides significant performance improvements, especially at higher dimensions.

## Relevance
This work is relevant to language design, compilers, and ML systems because it demonstrates how a sophisticated machine learning algorithm can be effectively parallelized using a high-level data-parallel language (Futhark). The implementation showcases techniques for mapping complex algorithms to GPU architectures, including handling irregular data structures like k-d trees in a parallel context. The performance results validate the effectiveness of data-parallel approaches for approximate nearest neighbor search, which is a fundamental operation in many ML applications. The work also highlights the importance of algorithmic optimizations (like the random transformations in RANN) that enable efficient parallel implementations.
