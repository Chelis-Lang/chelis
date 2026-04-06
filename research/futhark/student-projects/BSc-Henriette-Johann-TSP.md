# Solving TSP on the GPU based on heuristic algorithms

## Metadata
- **Authors:** Jóhann Utne, Henriette Naledi Winther Hansen
- **Venue/Year:** Department of Computer Science, June 10, 2023

## Summary
This project implements a parallel solution to the Traveling Salesman Problem (TSP) on GPU using the 2-Opt local search algorithm. The authors first survey three popular heuristic algorithms (2-Opt, Ant Colony Optimization, and Simulated Annealing) before selecting 2-Opt for GPU implementation due to its parallel structure and lack of parameter tuning requirements. The implementation maps each climber to a GPU block and uses threads within blocks to evaluate 2-opt moves in parallel. The solution is compared with O'Neil et al.'s GPU implementation, showing up to 27x speedup for smaller numbers of climbers while maintaining accuracy within 5% of optimal solutions. The authors also evaluate GPU hardware utilization by measuring throughput in GB/s, finding that larger datasets exceed the GPU's theoretical bandwidth due to distance matrix caching in L2 memory.

## Key Contributions
- Parallel implementation of 2-Opt algorithm on GPU using CUDA
- Comparison with O'Neil et al.'s GPU implementation showing significant speedup
- Evaluation of solution accuracy across different problem sizes
- Analysis of GPU hardware utilization through throughput measurements
- Three optimized versions of the CUDA implementation (original, 100 cities, calculated I and J)

## Technical Approach
The authors implement the 2-Opt algorithm with random restarts, where each climber starts with a random tour and iteratively improves it through edge swaps until reaching a local optimum. The GPU implementation maps each climber to a block and uses threads within blocks to evaluate 2-opt moves in parallel. Key optimizations include:
- Storing the distance matrix in global memory (fits in L2 cache for larger problems)
- Calculating I and J values on-the-fly instead of storing them
- Using shared memory for temporary storage of tours and changes
- Implementing efficient reduction operations for finding best changes

## Results
The implementation achieves up to 27x speedup compared to O'Neil et al.'s solution for 1000 climbers on 100-city problems. Accuracy tests show that solutions within 5% of optimal are found with relatively few climbers (20 for 52-city problems, 1000 for 100-city problems). Throughput measurements reveal that smaller problems (<100 cities) don't fully utilize the GPU, while larger problems (>100 cities) exceed the theoretical bandwidth due to L2 cache effects.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates:
- How to effectively parallelize a combinatorial optimization algorithm on GPU
- Trade-offs between memory usage and performance in GPU implementations
- Techniques for measuring and optimizing hardware utilization
- The importance of cache effects in GPU performance
- Practical considerations when implementing parallel algorithms in CUDA
