# Reverse Automatic Differentiation in Futhark

## Metadata
- **Authors:** Lotte Bruun, Ulrik Larsen
- **Venue/Year:** Master's Thesis, University of Copenhagen, May 2022

## Summary
This thesis presents work on extending and optimizing reverse automatic differentiation (AD) in the Futhark programming language, specifically for reduce-by-index and scan operations. The authors develop new rewrite rules for these operations, implement them in the Futhark compiler, and evaluate their performance against both the primal programs and Futhark's existing forward AD implementation. The work demonstrates significant performance improvements, especially for special cases of reduce-by-index and scan operators with specific Jacobian patterns.

## Key Contributions
- A reverse mode AD rewrite rule for reduce-by-index with generic case operators that preserves the expected work-depth asymptotics of the primal program.
- Formal representation of rewrite rules for reduce-by-index with special case operators: min/max, addition, and multiplication.
- Systematic derivation of a rewrite rule for scan with an arbitrary operator, allowing for simplifications and optimizations.
- Analysis and implementation of optimizations based on statically-reasoned sparsity of intermediate results in scan operations.
- Implementation of reverse mode AD for reduce-by-index with generic and special cases in the Futhark compiler.
- Extension of reverse mode AD for scan with tuple operators and special case operators in the Futhark compiler.
- Experimental evaluation demonstrating reasonable overheads compared to primal programs for special cases, significant speedups on computation of the full Jacobian compared to forward mode AD, and identification of performance bottlenecks.

## Technical Approach
The authors use rewrite rules to transform programs into their differentiated counterparts. For reduce-by-index, they develop a generic case rule and special case rules for addition, multiplication, and min/max operations. For scan, they extend the existing implementation to work with tuple operators and optimize it for specific Jacobian patterns (ZeroQuad and MatrixMul). The implementation is done in the Futhark compiler, and the performance is evaluated using benchmarks.

## Results
The experimental evaluation shows that the reverse AD implementation has reasonable overheads compared to the primal programs for special cases of reduce-by-index. It also demonstrates significant speedups (up to 3x) on computation of the full Jacobian compared to forward mode AD in Futhark. The evaluation identifies that a significant performance bottleneck is due to Futhark not supporting a GPU-efficient Radix sort implementation. For scan, the implementation shows competitive performance with forward mode AD when using special case operators and up to 3x speedup when exploiting a Jacobian pattern.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it presents a practical implementation of reverse AD in a data-parallel programming language. The optimizations and techniques developed can be applied to other languages and systems that require efficient differentiation. The work also highlights the importance of considering GPU behavior and memory access patterns when implementing AD for data-parallel languages.
