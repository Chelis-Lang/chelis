# A multicore backend for Futhark

## Metadata
- **Authors:** Duc Minh Tran
- **Venue/Year:** MSc Thesis, University of Copenhagen / 2020

## Summary
This thesis presents the design and implementation of a new multicore backend for the data-parallel language Futhark. The work aims to enable efficient parallel execution on multicore CPUs while maintaining the simplicity of implicit parallelism. The key contribution is an oracle-guided scheduling approach that automatically performs granularity control without requiring manual tuning from the programmer. The implementation uses parallel for-loops and supports both regular (nested) parallelism and irregular parallelism through two online algorithms. The thesis includes a comprehensive evaluation comparing the new backend against Futhark's sequential backend and hand-optimized implementations from established benchmark suites.

## Key Contributions
- Design and implementation of a new code-generator in the Futhark compiler for generating parallel C code
- Design and implementation of a runtime system with workload balancing through work-stealing
- Two online algorithms for automatic granularity control of parallel for-loops for regular and irregular parallelism
- Extensive empirical evaluation using micro-benchmarks and established benchmarks
- Comparison against hand-optimized programs from FinPar, Accelerate, and Rodinia benchmark suites

## Technical Approach
The implementation uses parallel for-loops to express parallelism and employs an oracle-guided scheduling approach based on Acar et al.'s work. The runtime system maintains estimates of sequential execution time for parallel tasks and uses these to determine optimal chunk sizes for parallelization. For regular parallelism, the system uses a linear cost function assumption, while for irregular parallelism, it employs a dynamic chunk size adjustment algorithm with work-stealing. The compiler generates multiple semantically equivalent versions of parallel computations to exploit different levels of parallelism at runtime.

## Results
The evaluation shows significant speedups compared to the sequential backend, with average speedups of 15.9× across 13 benchmarks on the largest datasets. The implementation achieves speedups ranging from 7.8× to 23.8× on a 16-core machine with 2-way multithreading. The results demonstrate that the approach can eliminate the need for manual tuning in many cases, though the compiler does not consistently generate as efficient code as hand-optimized implementations, resulting in slower performance for some benchmarks.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it addresses the challenge of making parallel programming accessible while maintaining performance. The approach of using runtime systems for automatic granularity control without requiring programmer intervention represents a significant advancement in implicit parallelism. The techniques developed could be applied to other data-parallel languages and systems, and the evaluation provides valuable insights into the practical challenges of implementing efficient parallel execution on multicore CPUs.
