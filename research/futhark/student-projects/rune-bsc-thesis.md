# Implementation of Graph Algorithms in Futhark

## Metadata
- **Authors:** Rune Nielsen
- **Venue/Year:** Bachelor's Project, University of Copenhagen, 2023

## Summary
This bachelor's project implements four graph algorithms in Futhark - a data-parallel functional array language - and compares their performance against the Problem Based Benchmark Suite (PBBS). The algorithms implemented are Breadth First Search (BFS), Maximal Independent Set (MIS), Maximal Matching (MM), and Minimum Spanning Forest (MSF). The project explores how well these irregular graph algorithms can be expressed in Futhark's data-parallel paradigm and evaluates their performance on both CPU (multicore) and GPU (CUDA) targets compared to PBBS's optimized implementations.

## Key Contributions
- Implementation of four graph algorithms in Futhark: BFS, MIS, MM, and MSF
- Adaptation of PBBS algorithms to Futhark's data-parallel model
- Performance comparison between Futhark implementations and PBBS benchmarks
- Analysis of Futhark's suitability for irregular graph algorithms
- Investigation of the Work/Depth model for parallel algorithm analysis

## Technical Approach
The project uses Futhark's built-in second-order array combinators for parallelization, including map, filter, scan, scatter, and hist. The implementations follow the overall structure of PBBS algorithms but adapt them to Futhark's immutable array model and data-parallel approach. Key techniques include:

- Using hist for duplicate removal and filtering
- Implementing queue-like structures with arrays of pairs
- Adapting the Union-Find data structure for MSF with limited path compression
- Leveraging Futhark's segmented library for expand operations
- Converting graph data to adjacency array format for efficient parallel access

## Results
The Futhark CUDA implementations achieved significant speedups over PBBS's multicore implementations:
- BFS: 3.41x, 2.41x, and 17.58x speedup on different datasets
- MIS: 3.26x, 2.35x, and 1.69x speedup
- MM: 3.33x, 3.19x, and 3.84x speedup
- MSF: 1.07x to 2.02x speedup (smaller datasets due to memory constraints)

The multicore Futhark implementations were generally slower than PBBS, as expected given Futhark's focus on GPU code generation.

## Relevance
This work is highly relevant to language design and compiler research as it demonstrates both the strengths and limitations of data-parallel functional languages for irregular algorithms. The project shows that while Futhark can express complex graph algorithms, certain data structures (like Union-Find with path compression) are challenging to implement efficiently. The performance results provide valuable insights into when data-parallel approaches are effective for graph problems and highlight areas where language and compiler improvements could help. The work also contributes to the growing body of research on applying functional programming to high-performance computing tasks.
