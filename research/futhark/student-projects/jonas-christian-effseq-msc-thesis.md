# Efficient Sequentialization of Parallelism

## Metadata
- **Authors:** Christian Marslev, Jonas Grønborg
- **Venue/Year:** University of Copenhagen, February 2024

## Summary
This thesis explores efficient sequentialization as an automatic compiler optimization for intra-block parallelism on GPUs. The authors implement transformation rules for four parallel operators (map, reduce, scan, and scatter) in the Futhark compiler, demonstrating that programs can achieve up to 2x speedup by introducing sequentialization to otherwise parallel programs. The work builds on mathematical foundations from list homomorphisms and Brent's theorem to justify the performance benefits of sequentialization.

## Key Contributions
- Developed transformation rules for efficient sequentialization of four parallel operators in Futhak
- Implemented the optimization as a compiler pass targeting intra-group kernels
- Demonstrated significant speedups (up to 2x) on benchmark programs
- Extended Futhark with a `seq_factor` attribute to control sequentialization
- Showed that sequentialization enables larger inner dimensions beyond CUDA's 1024 thread limit

## Technical Approach
The authors use list homomorphisms as a mathematical foundation, showing that parallel operations can be split into sequential chunks that are then combined. They implement this through tiling strategies that move data from global to shared memory, with each thread processing multiple elements sequentially. The transformation rules preserve work complexity while reducing depth complexity, particularly beneficial for reduce and scan operations. The implementation includes runtime discrimination between sequentialized and original versions based on thread count thresholds.

## Results
Experimental results show:
- Map: Slight improvements (1-2% speedup) or minor losses at very small block sizes
- Reduce: Significant speedups (2.5-2.6x) due to reduced reduction tree depth
- Scan: Good speedups (1.9-2.0x) comparable to reduce but with more intermediate steps
- Scatter: Mixed results - iota indices show 13% speedup, random indices show slowdown due to memory access patterns
- Big number addition: 75-85% speedup, peaking at inner dimension 512
- Intra-block radix sort: 1.3-2.1x speedup, best at block size 256 threads

## Relevance
This work is highly relevant to language design and compiler optimization for parallel systems. It demonstrates how compilers can automatically transform parallel programs to better utilize hardware resources when parallelism is saturated. The approach is particularly valuable for GPU programming where intra-block parallelism is limited and efficient use of shared memory is crucial. The techniques could be applied to other data-parallel languages and compilers targeting heterogeneous architectures.
