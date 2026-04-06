# Efficient Histogram Computation on GPGPUs

## Metadata
- **Authors:** Sune Hellfritzsch
- **Venue/Year:** University of Copenhagen, Master's Thesis, 2018

## Summary
This thesis presents the design and implementation of a new language construct for efficient computation of generalized reductions in the Futhark programming language. Generalized reductions, also known as reduction by key or reduction by index, involve reducing a collection of data into k buckets where there is no pattern in the input data. The thesis focuses on histogram computation as a concrete instance of generalized reductions. The implementation leverages atomic functions and subhistogramming to achieve efficient GPU code generation. The thesis also includes a comprehensive experiment to determine the optimal cooperation level for subhistogramming and evaluates the new construct on adversarial datasets, demonstrating significant performance improvements over existing solutions.

## Key Contributions
- Design and implementation of a new language construct for generalized reductions in Futhark.
- Development of a heuristic for choosing the optimal cooperation level for subhistogramming.
- Comprehensive experiment to evaluate the impact of cooperation level on runtime performance.
- Implementation of three different code generations for atomic operations.
- Evaluation of the new construct on adversarial datasets, showing significant speedups compared to existing solutions.

## Technical Approach
The thesis proposes a combination of data-parallel chunking and atomic operations to compute generalized reductions efficiently on GPUs. The key idea is to let GPU threads cooperate on subhistograms, where each group of threads produces its own local histogram, which are ultimately combined into one final histogram. The implementation is based on the following strategies:

1. **Atomic Operations**: Three different atomic functions are used to implement arbitrary binary operators on one or multiple memory locations.
2. **Subhistogramming**: The input data is split between groups of threads such that each group produces its own local histogram. The number of cooperating threads per subhistogram is determined by a heuristic based on the histogram size.
3. **Code Generation**: The implementation generates efficient GPU code based on user-provided input, selecting the optimal strategy for implementing a combining function provided by the user.

## Results
The new construct is evaluated on 12 adversarial datasets, showing significant speedups compared to a sequential implementation of a traditional histogram, existing solutions in Futhark, and a single reference implementation in Thrust. The speedups range from 1.62× to 17.63×. The construct also performs well on two existing Futhark benchmarks, showing both a slowdown and a small speedup, which are known cases where existing solutions perform really well.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it presents a new language construct for efficient computation of generalized reductions, which is a common pattern in data-parallel computing. The implementation leverages atomic functions and subhistogramming to achieve efficient GPU code generation, which is important for high-performance computing applications. The thesis also includes a comprehensive experiment to determine the optimal cooperation level for subhistogramming, which is valuable for understanding the trade-offs involved in parallel computing.
