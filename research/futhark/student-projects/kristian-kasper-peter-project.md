# Linear Algebra in Futhark

## Metadata
- **Authors:** Kasper Unn Weihe, Kristian Quirin Hansen, Peter Kanstrup Larsen
- **Venue/Year:** University of Copenhagen, Faculty of Science, Project Outside Course Scope, January 28, 2021

## Summary
This paper presents the implementation and design of various linear algebra algorithms in the data-parallel purely functional array language Futhark. The project aimed to explore how well Futhark handles commonly used linear algebra functions such as NMF, Cholesky Decomposition, QR Decomposition, and solving linear systems. The implementations were benchmarked against Python libraries NumPy and scikit-learn. While the Futhark implementations did not always match the performance of the highly optimized Python libraries, the authors found that Futhark and linear algebra are well-suited for each other, particularly for batched operations and certain algorithms.

## Key Contributions
- Implementation of basic linear algebra operations (vector and matrix operations) in Futhark
- Implementation of QR Decomposition using Gram-Schmidt process, Householder transformations, and blocked Householder transformations
- Implementation of Non-negative Matrix Factorization (NMF) with multiplicative update rules
- Implementation of Cholesky Decomposition using both standard and flattened approaches
- Implementation of matrix determinant calculation using Dodgson condensation, Doolittle LU decomposition, and Cholesky decomposition methods
- Implementation of linear system solvers using Gauss-Jordan elimination, LU decomposition, and Cholesky decomposition
- Comprehensive benchmarking comparing Futhark implementations to NumPy and scikit-learn

## Technical Approach
The authors leveraged Futhark's data-parallel capabilities and Second-Order Array Combinators (SOACs) to implement linear algebra algorithms. They used in-place array updates with uniqueness types to ensure memory safety. The implementations included:

- Basic operations: dot product, vector/matrix scalar multiplication, outer product, identity matrix
- QR Decomposition: Gram-Schmidt process, Householder transformations, and blocked Householder transformations
- NMF: Multiplicative update rules with Frobenius norm divergence checking
- Cholesky Decomposition: Standard column-by-column approach and flattened 1D array approach
- Matrix determinants: Dodgson condensation, Doolittle LU decomposition, and Cholesky decomposition methods
- Linear system solvers: Gauss-Jordan elimination, LU decomposition, and Cholesky decomposition

## Results
The benchmark results showed mixed performance compared to Python libraries:

- QR Decomposition: Futhark implementations were slower than NumPy/scikit-learn, but batched versions were significantly faster
- NMF: Futhark outperformed scikit-learn for larger matrices and batched operations
- Cholesky Decomposition: Futhark was slower than NumPy/scikit-learn, but batched versions showed substantial speedups
- Matrix determinants: Futhark implementations were slower, but batched Cholesky determinant was much faster
- Linear system solvers: Futhark was slower, but performance gap decreased with more right-hand side vectors

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work because it demonstrates the practical application of a data-parallel functional language to computationally intensive linear algebra operations. The results highlight both the potential and limitations of using functional programming languages for numerical computing. The work provides insights into how compiler optimizations and language features can impact performance in scientific computing contexts. The comparison with established libraries like NumPy and scikit-learn offers valuable benchmarks for evaluating the effectiveness of alternative approaches to linear algebra computation.
