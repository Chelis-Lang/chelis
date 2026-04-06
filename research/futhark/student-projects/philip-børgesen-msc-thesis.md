# Reducing Synchronous GPU Memory Transfers

## Metadata
- **Authors:** Philip Jon Børgesen
- **Venue/Year:** MSc thesis, University of Copenhagen, 2022

## Summary
This thesis presents a series of dataflow-dependent program transformations that reduce synchronous memory transfers between a GPU and its host by migrating sequential CPU computations to the GPU. The work is implemented in the Futhark compiler and demonstrates significant performance improvements on benchmark programs. The core contribution is a graph-based optimization approach that models the problem of minimizing memory transfers as finding minimum vertex cuts in data dependency graphs, solved using a specialized algorithm based on the Ford-Fulkerson max-flow method.

## Key Contributions
- A graph-based model for minimizing synchronous GPU-host memory transfers
- A specialized algorithm for finding minimum vertex cuts in data dependency graphs
- Implementation of program transformations that migrate sequential work to the GPU
- Empirical evaluation showing 1.17-1.58x mean speedups across 27 benchmark programs
- Techniques for handling conditional execution and loops in a pure functional language

## Technical Approach
The thesis introduces a graph representation where vertices represent host variables and edges represent dependencies between them. The optimization problem becomes finding a partition of vertices into device and host sets that minimizes the number of device-to-host memory transfers. The algorithm uses depth-first search with edge exhaustion to find routes from source vertices (representing device values) to sink vertices (representing values needed on the host), reversing edges along successful paths to create routes.

The approach handles various language constructs including array reads/writes, conditionals, and loops. For loops, the thesis introduces a cyclic subgraph model and techniques for reducing reads across iterations. The implementation includes a two-phase transformation: first migrating statements to device, then merging GPU kernels to reduce overhead.

## Results
Experimental results on four different GPU architectures (NVIDIA A100, AMD Instinct MI100, AMD Radeon Pro 560, and Intel HD Graphics 630) show mean speedups of 1.17-1.58x across 27 benchmark programs. The most significant improvements came from programs with simple patterns of scalar operations interspersed between GPU kernels. Some programs saw manyfold improvements, while others saw modest gains or no change. The thesis also presents microbenchmarks demonstrating the benefits of migrating array literals with scalar variable operands.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it addresses a fundamental challenge in GPU programming: minimizing expensive host-device communication. The techniques presented could be applied to other data-parallel languages and compilers, and the graph-based optimization approach provides a general framework for reasoning about data movement in heterogeneous systems. The work demonstrates that aggressive optimization of memory transfers can yield significant performance improvements, even in a language with sophisticated automatic parallelization like Futhark.
