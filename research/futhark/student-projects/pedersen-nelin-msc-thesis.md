# General Array Locality Optimization by Permutation (GALOP)

## Metadata
- **Authors:** Bjarke Pedersen, Oscar Nelin
- **Venue/Year:** Master's Thesis, University of Copenhagen / 2023

## Summary
This thesis presents GALOP, a novel hardware-agnostic algorithm for optimizing array locality of reference in data-parallel programs through strategic permutation of array dimensions. The algorithm analyzes how arrays are accessed in programs and determines optimal memory layouts to improve performance on both CPU and GPU architectures. The authors implement GALOP in the Futhark compiler, replacing the existing GPU-specific solution and adding support for the multi-core backend. Evaluation shows GALOP generally performs better than the existing solution on GPU while matching performance in cases where the previous implementation succeeded.

## Key Contributions
- Novel hardware-agnostic algorithm for array locality optimization through dimension permutation
- Implementation in Futhark compiler supporting both GPU and multi-core backends
- Comprehensive validation suite including unit tests, integration tests, and black-box tests
- Performance evaluation comparing GALOP against existing solution across multiple hardware configurations

## Technical Approach
The algorithm operates in three stages: analysis, layout, and transformation. The analysis stage builds an index table mapping array accesses to their iteration variables and access patterns. The layout stage computes optimal dimension orderings based on iteration variable levels and access patterns, applying conditions to reject suboptimal layouts. The transformation stage inserts manifest statements to materialize arrays with the computed layouts. The approach relies on the insight that optimal locality can be achieved by ensuring iteration variables with the greatest access scope are mapped to the innermost dimensions.

## Results
Evaluation on two GPU architectures (NVIDIA A100 and RTX 2070 Super) shows GALOP generally outperforms the existing solution while matching performance in cases where the previous implementation succeeded. On the A100, doing nothing often outperforms both approaches, suggesting locality optimizations may be less critical on modern server-grade GPUs. The multi-core backend implementation shows promise but requires further refinement. Several benchmarks demonstrate cases where GALOP avoids suboptimal manifests that the previous implementation would insert.

## Relevance
This work is highly relevant to language design and compiler optimization for data-parallel languages. It addresses a fundamental challenge in optimizing array access patterns for different hardware architectures while maintaining a high-level programming model. The hardware-agnostic approach and modular design make it extensible to new architectures and backends, providing a foundation for future work in locality optimization across diverse computing platforms.
