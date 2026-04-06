# Data Parallel Programming B2-21/22: Multiple-precision Integer Arithmetic

## Metadata
- **Authors:** Amar Topalovic, Walter Restelli-Nielsen, Kristian Olesen
- **Venue/Year:** Academic Project, 2022

## Summary
This project implements a Futhark library for arbitrary-precision integer arithmetic, focusing on parallel algorithms for operations on very large numbers. The authors develop implementations for addition, multiplication, division, and other operations, comparing their performance against established libraries like GMP and CGBN. The work explores the trade-offs between parallel and sequential approaches, demonstrating that their parallel implementations can outperform existing libraries for certain operations on sufficiently large numbers.

## Key Contributions
- Implementation of big integer arithmetic in Futhark using arrays of 32-bit unsigned integers
- Parallel algorithms for addition, multiplication, and division with different complexity characteristics
- Comparison of performance against GMP and CGBN libraries
- Exploration of parallel vs sequential algorithm trade-offs for big integer operations

## Technical Approach
The authors represent big integers as arrays of 32-bit unsigned integers, where each element represents a digit in base 2^32. For addition, they use a parallel scan approach with carry propagation. Multiplication is implemented using long multiplication in base 2^32, calculating high and low parts of products in parallel. Division uses Knuth's "algorithm D" for arbitrary precision division and a parallel algorithm for division by single-precision divisors. The implementation includes various optimizations and explores different parallel strategies.

## Results
Benchmark results show that the parallel implementations outperform GMP and CGBN for addition and small division operations on very large numbers (up to 320k bits). However, for sequential operations like regular division, the implementation performs poorly compared to optimized sequential libraries. The performance varies significantly based on the operation and number size, with parallel algorithms showing advantages for large numbers but struggling when parallelism cannot be fully exploited.

## Relevance
This paper is relevant to language design and compiler research as it demonstrates the challenges and opportunities of implementing parallel algorithms for numerical computations in a functional data-parallel language like Futhark. The work provides insights into how different parallel strategies affect performance and highlights the importance of matching algorithm design to hardware capabilities. The comparison with established libraries also offers valuable lessons about the trade-offs between parallel and sequential approaches in numerical computing.
