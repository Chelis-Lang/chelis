# Extending Futhark's multicore C backend to utilize SIMD using ISPC

## Metadata
- **Authors:** W. Pema N. H. Malling, Louis Marott Normann, Oliver B. K. Petersen, Kristoffer A. Kortbæk
- **Venue/Year:** Bachelor Project, University of Copenhagen, 2022

## Summary
This thesis details the design, implementation, and evaluation of a new compiler backend for the Futhark language that extends the existing multicore C backend to utilize SIMD instructions through Intel's ISPC language. The work aims to combine task-parallelism (already present in Futhark's multicore backend) with data-parallelism via SIMD to achieve better performance on multicore CPUs. The authors describe the challenges of using ISPC as a code generation target and evaluate the performance of their implementation against the existing backend using Futhark's extensive benchmark suite.

## Key Contributions
- Design and implementation of a new Futhark compiler backend that emits ISPC code
- Extension of Futhark's intermediate representation to support ISPC-specific constructs
- Implementation of variability analysis to correctly handle uniform/varying qualifiers in generated ISPC code
- Port of the AOBench benchmark to Futhark for comparison with handwritten ISPC
- Various improvements to Futhark's test suite
- Extension of the language-c-quote Haskell library to support ISPC parsing

## Technical Approach
The authors extend Futhark's compilation pipeline by adding ISPC code generation capabilities to the backend. They implement algorithms for Futhark's Second-Order Array Combinators (SOACs) that combine task-parallelism with SIMD vectorization. The approach involves:

1. Generating ImpCode representations of SOACs that can use ISPC-specific constructs
2. Implementing variability analysis to correctly handle uniform/varying qualifiers
3. Creating mechanisms for communication between C and ISPC code
4. Handling memory allocations and error handling in ISPC
5. Generating appropriate C and ISPC code from the intermediate representation

The authors implement different code generation strategies depending on the properties of the SOACs, such as whether reduction operators are commutative or whether mapped operators are used.

## Results
The evaluation shows mixed results:
- Significant speedups (up to 10x) on several benchmarks, particularly those with simple mapping operations
- Performance matching the existing multicore backend on many benchmarks
- Slowdowns on some benchmarks due to issues like non-optimal foreach placement causing unnecessary gather/scatter operations
- The performance gap between their backend and handwritten ISPC code (AOBench) was about 3x

The authors identify several reasons for the slowdowns, including poor foreach placement in the generated code and issues with memory access patterns.

## Relevance
This work is relevant to language design, compilers, and ML systems because it demonstrates how to extend a functional data-parallel language to better utilize modern CPU architectures through SIMD instructions. The challenges faced and solutions developed for using ISPC as a code generation target provide insights for other compiler writers looking to add SIMD support. The work also highlights the trade-offs between different approaches to vectorization (auto-vectorization vs. explicit vectorization) and the importance of careful code generation for performance.
