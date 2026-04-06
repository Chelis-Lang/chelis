# Optimisation and GPU Code Generation of Stencils for Futhark

## Metadata
- **Authors:** Christian Charlie Virt, Jonathan Wraa-Hansen
- **Venue/Year:** Master Thesis, University of Copenhagen, 2021

## Summary
This thesis focuses on extending the Futhark compiler to support efficient code generation for stencil computations on GPUs. Stencils are common in scientific computing, and optimizing them for parallel execution on GPUs is crucial for performance. The authors implement a stencil construct in Futhark, allowing users to express stencils in a high-level language while the compiler generates optimized CUDA and OpenCL code. They explore various design strategies for stencil execution on GPUs, prototype different approaches, and implement the most efficient one in the Futhark compiler. The evaluation shows significant speedups compared to existing Futhark implementations, especially for larger stencils and datasets.

## Key Contributions
- Prototyping of multiple GPU stencil execution designs (1D, 2D, 3D) with evaluation and documentation.
- Implementation of stencil construct handling in Futhark compiler modules (interpreter, type assignment, first-order transformation, GPU code generation).
- Description and documentation of the implemented parts and their connection to prototyping.
- Empirical evaluation of the implementation's correctness and performance compared to a reference implementation.

## Technical Approach
The authors extend Futhark with three new functions (`stencil_1d`, `stencil_2d`, `stencil_3d`) to handle stencils of different dimensions. They implement these functions in the Futhark interpreter, type checker, and code generators for sequential C, CUDA, and OpenCL. The core of their work is the GPU code generation, where they implement a "multi-write big-tile" approach. This approach divides the input array into tiles, loads them into shared memory, and processes multiple elements per thread to increase reuse and reduce memory transactions. They also implement a fallback "global read" approach for cases where the multi-write big-tile is not suitable.

## Results
The authors validate their implementation through unit tests and random data tests, ensuring correctness for both CPU and GPU code generation. They compare their implementation to a reference implementation using nested maps and show significant speedups, especially for larger stencils and datasets. The speedups vary depending on the stencil, dataset size, and GPU architecture, but can be up to three times faster than the reference implementation.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates how to extend a high-level array language with a new construct for a specific domain (stencils) and generate efficient GPU code for it. The techniques and design decisions explored in this thesis can be applied to other domain-specific languages and compilers targeting GPUs. The evaluation of different GPU execution strategies provides insights into optimizing stencil computations for parallel hardware.
