# MPI-Futhark: Distributed High-Performance Computing For People

## Metadata
- **Authors:** Baptiste Coudray
- **Venue/Year:** Bachelor thesis, August 2021

## Summary
This thesis presents the development of a library that enables distributed high-performance computing for cellular automata using the Futhark programming language and MPI (Message Passing Interface). The library allows programmers to write Futhark code that can be compiled and executed in four modes: distributed-sequential, distributed-multicore, distributed-OpenCL, and distributed-CUDA. The author implements and benchmarks three cellular automata (one-dimensional Simple Cellular Automaton, two-dimensional Game of Life, and three-dimensional Lattice-Boltzmann Method) to validate the library's scalability and performance across different dimensions and computing backends.

## Key Contributions
- Development of a library that automatically distributes cellular automata across multiple compute nodes using MPI
- Integration of Futhark's parallel computing capabilities with MPI's distributed computing framework
- Implementation of three cellular automata examples in one, two, and three dimensions
- Comprehensive benchmarking across four execution modes (sequential, multicore, OpenCL, CUDA)
- Creation of a Cartesian virtual topology for efficient neighbor communication in distributed cellular automata

## Technical Approach
The library uses MPI to distribute cellular automata across multiple compute nodes, with each node processing a chunk of the automaton. A Cartesian virtual topology is created to enable efficient neighbor communication, with special handling for boundary cells through an "envelope" mechanism. The Futhark language is used to implement the core update logic for each cellular automaton, with the library handling the MPI communication overhead. The library supports custom data types through MPI's derived datatype functionality, demonstrated with the three-dimensional Lattice-Boltzmann Method implementation.

## Results
The benchmarks show ideal speedup for one and two-dimensional cellular automata across all backends. For the three-dimensional Lattice-Boltzmann Method, the sequential and multicore backends achieved a maximum speedup of 41x with 128 tasks, while the GPU backends (OpenCL and CUDA) showed ideal speedup. Parallel computing consistently outperformed sequential and concurrent computing, with the Game of Life achieving up to 15x speedup compared to sequential execution.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates how to extend a functional array language (Futhark) with distributed computing capabilities. The approach of separating the parallel computation logic (in Futhark) from the distributed communication logic (in MPI) provides a clean abstraction that could be applied to other domains beyond cellular automata. The library serves as a practical example of how to bridge the gap between high-performance computing frameworks and modern functional programming languages, making distributed computing more accessible to developers.
