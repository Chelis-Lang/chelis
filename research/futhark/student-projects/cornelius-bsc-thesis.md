# Flattening Irregular Nested Parallelism in Futhark

## Metadata
- **Authors:** Cornelius Sevald-Krause
- **Venue/Year:** BSc Thesis, University of Copenhagen, 2023

## Summary
This thesis presents work on implementing irregular nested parallelism flattening in the Futhark programming language. The author introduces a new flattening rule for match-expressions as a generalization of flattening if-expressions, and implements this transformation in the Futhark compiler. The work also details function lifting, a technique for flattening function calls, and implements it as part of achieving full irregular flattening in Futhark. The thesis includes benchmarking results showing that the dedicated match-transformation runs modestly faster than using nested if-expressions.

## Key Contributions
- New flattening rule for match-expressions that generalizes if-expression flattening
- Implementation of function lifting for flattening function calls
- Comprehensive testing framework for validating the flattening transformations
- Benchmark results demonstrating performance benefits of the match-expression flattening approach
- Integration of these features into the Futhark compiler

## Technical Approach
The thesis builds on the concept of irregular nested parallelism, where arrays can have varying sizes at different nesting levels. The author represents irregular arrays using data arrays and segment arrays, along with auxiliary structures like offset arrays and flag arrays. The flattening transformation converts nested parallel operations into flat parallel operations suitable for GPU execution.

The key technical contributions include:
1. A generalized flattening rule for match-expressions that partitions input arrays into k equivalence classes based on pattern matching, then flattens each branch separately before merging results
2. Function lifting that transforms functions with array parameters into versions that operate on the irregular array representation (segment arrays, data arrays, etc.)
3. Implementation of these transformations in the Futhark compiler's intermediate representation

## Results
The implementation was validated through extensive testing:
- 20 unit tests for function lifting covering various scenarios including scalar/array parameters and return types
- 21 unit tests for match-expression flattening covering different combinations of regular and irregular inputs/results
- All tests passed on both OpenCL and CUDA backends

Benchmark results compared nested if-expressions versus single match-expressions for different numbers of branches (2, 4, 8, 16, 32, 64). The match-expression version showed speedups up to 5x for smaller inputs, converging to approximately 1.7x for larger inputs.

## Relevance
This work is highly relevant to language design, compilers, and ML systems because:
1. It advances the state of nested data parallelism compilation, which is crucial for high-performance computing on GPUs
2. The techniques developed can be applied to other data-parallel languages facing similar challenges with irregular nested parallelism
3. The implementation demonstrates practical solutions to the "replication problem" when dealing with free variables in nested parallel contexts
4. The work contributes to making high-level parallel programming more accessible by automating complex flattening transformations that programmers previously had to perform manually
5. The research has implications for machine learning systems that often involve irregular nested operations on tensors and arrays
