# Optimizing Futhark's Type Checker

## Metadata
- **Authors:** Jacob Aleksandar Siegumfeldt, Laust Kjæp Dengsøe
- **Venue/Year:** Bachelor thesis, University of Copenhagen, 2025

## Summary
This thesis explores optimizing Futhark's type constraint solver by implementing a union-find data structure with path compression and union-by-weight heuristics. The authors propose a new implementation that theoretically should provide better asymptotic performance than Futhark's current approach. Through benchmarking and profiling, they evaluate their implementation against the existing one, finding that while their approach doesn't generally outperform the current implementation in practice, they have laid the groundwork for more efficient constraint solving in the future.

## Key Contributions
- Theoretical exploration of union-find data structures for type constraint solving
- Implementation of a new constraint solver in Haskell using the ST monad for efficient state management
- Comprehensive benchmarking comparing the new implementation against the existing one
- Profiling analysis identifying performance bottlenecks
- Discussion of potential improvements including hash tables and postponed occurs checks

## Technical Approach
The authors implement a union-find data structure in Haskell using the ST monad to enable in-place updates. The data structure supports path compression and union-by-weight heuristics to achieve near-constant time operations. The constraint solver processes type constraints by normalizing types, binding type variables, and unioning equivalence classes. The implementation handles practical complications like overloading, scope violations, and liftedness constraints specific to Futhark's type system.

## Results
Benchmarking results show that the new implementation is generally slower than the existing one, with the mean execution time being 16.4% higher across 598 benchmarks converted from real Futhark programs. However, in a synthetic scenario designed to benefit from path compression, the new implementation shows significantly better performance with linear scaling compared to the quadratic scaling of the old implementation. Profiling reveals that STRef operations and the solveEq.sub function are major performance bottlenecks.

## Relevance
This work is highly relevant to language design and compiler development, particularly for functional languages with Hindley-Milner type systems. The exploration of union-find data structures for type inference provides valuable insights into optimizing constraint solving algorithms. The findings about the trade-offs between theoretical efficiency and practical performance are particularly important for compiler implementers. The proposed improvements, including hash table lookups and postponed occurs checks, offer concrete directions for future optimization work in type checkers.
