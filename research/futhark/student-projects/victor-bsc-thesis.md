# Data-parallel Construction of Interval Trees and Range Trees

## Metadata
- **Authors:** Victor Alexander Schmidt
- **Venue/Year:** Master's Thesis, 2024

## Summary
This thesis explores the data-parallel construction of interval trees and k-dimensional range trees in the Futhark programming language. The author implements centered interval trees and 2D range trees using Futhark's array-oriented programming model, which lacks traditional recursion and pointer structures. The work focuses on achieving efficient parallel construction while maintaining correctness, and evaluates the performance against sequential implementations and brute-force approaches. The thesis demonstrates that data-parallel construction can achieve significant speedups, particularly on GPUs, though the benefits depend on the number of queries performed.

## Key Contributions
- Data-parallel implementation of centered interval trees in Futhark
- Data-parallel implementation of k-dimensional range trees (focusing on 2D)
- Performance evaluation comparing parallel construction against sequential approaches
- Analysis of when tree construction becomes beneficial compared to brute-force methods
- Validation testing ensuring correctness against brute-force implementations

## Technical Approach
The implementation leverages Futhark's SOACs (Second-Order Array Combinators) to achieve parallelism without recursion. For interval trees, the author uses a breadth-first construction approach with loops, partitioning intervals at each level based on a calculated midpoint. Range trees are constructed similarly, with associated structures built in parallel for higher dimensions. The author employs array flattening techniques to represent tree structures, using index arrays to simulate pointers. Key techniques include segmented scans for calculating midpoints and partition operations for dividing work across parallel threads.

## Results
Benchmark results show significant speedups for parallel construction:
- Range trees: ~100x speedup on GPU compared to sequential, ~2.4-2.8x on CPU
- Interval trees: ~15-100x speedup on GPU, ~2.3-3.8x on CPU

Query performance analysis reveals that range trees become beneficial for datasets with ≥215 points, while interval trees show ~2x speedup over brute force. The amortized cost analysis shows that for interval trees with 220 intervals, ~42 queries justify the construction cost.

## Relevance
This work is highly relevant to language design and compiler research as it demonstrates how to implement complex data structures in a purely functional, array-oriented language without traditional recursion or pointers. The techniques developed for representing trees using flat arrays and index-based references could inform future language features or compiler optimizations. For ML systems, the work shows how to leverage GPU parallelism for data structure construction, which could be valuable for preprocessing large datasets. The performance analysis provides insights into when parallel construction is worthwhile, which is crucial for practical deployment of such systems.
