# Accelerating Ocean Modelling

## Metadata
- **Authors:** Till Grenzdörffer
- **Venue/Year:** University of Copenhagen, Department of Computer Science, 2021

## Summary
This project addresses the performance bottlenecks in the ocean modelling framework Veros, which is written in Python and uses Jax for parallelization. The author investigates two algorithms and one routine to find performant solutions, implementing CUDA versions and integrating them into Jax through XLA custom calls. The CUDA implementations significantly outperform the Jax implementations, with the Futhark implementation showing even better performance.

## Key Contributions
- Investigation of tridiagonal solver algorithms and their performance optimizations
- Implementation of a tiled stencil computation for the Superbee scheme
- Integration of CUDA code into Jax through XLA custom calls
- Comparison of performance between Jax, CUDA, and Futhark implementations

## Technical Approach
The author explores different algorithms for solving tridiagonal systems, including the Thomas Algorithm and a flat version based on scan operations. For the Superbee scheme, a tiled stencil computation approach is used to optimize memory access patterns. The CUDA implementations are integrated into Jax through XLA custom calls, allowing for easy use through a Python interface.

## Results
The CUDA implementations significantly outperform the Jax implementations, with the turbulent kinetic energy routine being sped up by a factor of 1.7. The Futhark implementation performs even better, increasing the performance by a factor of 4.6. The benchmarks show that the choice of algorithm depends on the problem size and hardware capabilities.

## Relevance
This paper is relevant to language design, compilers, and ML systems work as it demonstrates the importance of choosing the right algorithm and optimization techniques for specific hardware and problem sizes. It also highlights the potential of integrating high-performance languages like CUDA and Futhark into user-friendly environments like Python for scientific computing.
