# Convex Optimization and Parallel Computing for Portfolio Optimization

## Metadata
- **Authors:** Kasper Unn Weihe
- **Venue/Year:** Master's Thesis, University of Copenhagen, 2023

## Summary
This thesis explores the application of convex optimization techniques and the data-parallel programming language Futhark for solving convex optimization problems, with a focus on portfolio optimization as a case study. The research emphasizes the importance of parallel computing strategies, leveraging many-core general-purpose graphics processing units (GPGPUs) to optimize computational efficiency. The study develops three Futhark modules: one for solving linear systems of equations, one for resolving convex optimization problems, and one tailored for portfolio optimization. The findings demonstrate that the Futhark implementation can outperform established optimizers like CVXPY for solving many optimization problems in parallel or a single optimization problem with a large number of variables.

## Key Contributions
- Development of three Futhark modules for linear systems, convex optimization, and portfolio optimization
- Implementation of various algorithms including Gaussian elimination, LU decomposition, Cholesky decomposition, conjugate gradient method, gradient descent, Newton's method, barrier method, and ADMM
- Benchmarking and comparison of Futhark implementations against Python libraries like NumPy, SciPy, and CVXPY
- Application of the developed modules to portfolio optimization problems using S&P500 data

## Technical Approach
The thesis employs a data-parallel programming approach using Futhark, a functional programming language designed for efficient parallel code generation on GPUs. The core technical ideas include:

1. **Linear Systems Solvers**: Implementation of Gaussian elimination, LU decomposition, Cholesky decomposition, and conjugate gradient method for solving linear systems of equations.

2. **Convex Optimization Algorithms**: Development of gradient descent, Newton's method, barrier method, and ADMM for solving convex optimization problems with various constraints.

3. **Auto-differentiation**: Utilization of Futhark's built-in auto-differentiation capabilities to compute gradients and Hessians of functions automatically.

4. **Portfolio Optimization**: Application of the convex optimization module to solve portfolio optimization problems, incorporating environmental, social, and governance (ESG) factors.

## Results
The benchmarking results demonstrate that the Futhark implementation can outperform established optimizers like CVXPY in several scenarios:

- For solving many small optimization problems in parallel, the Futhark implementation is significantly faster than CVXPY.
- For problems with a large number of variables, the Futhark implementation can also outperform CVXPY.
- The Futhark implementation is particularly efficient when solving portfolio optimization problems with ESG constraints.

## Relevance
This paper is highly relevant to language design, compilers, and ML systems work for several reasons:

1. **Parallel Programming Languages**: The thesis demonstrates the effectiveness of Futhark as a high-performance parallel programming language for solving complex optimization problems.

2. **Compiler Optimizations**: The implementation showcases various compiler optimizations and techniques for efficient parallel code generation.

3. **ML Systems**: The developed modules and algorithms can be applied to various machine learning problems that involve convex optimization.

4. **Financial Applications**: The portfolio optimization case study highlights the practical applications of convex optimization in finance and investment strategies.

5. **Performance Comparison**: The benchmarking against established libraries like CVXPY provides valuable insights into the performance characteristics of different optimization approaches and implementations.
