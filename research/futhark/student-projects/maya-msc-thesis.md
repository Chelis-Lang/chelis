# Code Generation for Stencils in Futhark

## Metadata
- **Authors:** Maya Saietz
- **Venue/Year:** Master's Thesis, December 2020

## Summary
This thesis investigates two optimizations - tiling and partitioning - for stencil computations on parallel architectures. The author prototypes these optimizations in CUDA and C, then implements three versions of code generation for stencils in the Futhark compiler's multicore backend. The work demonstrates that combining tiling and partitioning yields significant performance improvements for stencil computations, though the optimized Futhark implementations still don't outperform non-stencil-specific Futhark constructs.

## Key Contributions
- Performance analysis of tiling and partitioning optimizations through CUDA and C prototyping
- Investigation of tile size impact on performance
- Implementation of three stencil code generation versions in Futhark's multicore backend:
  - Unoptimized version
  - Tiled version
  - Tiled and partitioned version
- Performance benchmarking of Futhark stencil implementations against tabulate-based alternatives

## Technical Approach
The author employs two main optimizations:

**Tiling**: Combines loop strip-mining and loop interchange to create nested loops that iterate over tiles rather than individual elements. This improves cache utilization by reusing cached data more effectively.

**Partitioning**: Separates boundary regions from the main computation area, eliminating boundary checks for the central region where all neighborhood points are within bounds.

The implementation in Futhark involves modifying the compiler's internal representation to support stencil-specific constructs, then generating appropriate C code for the multicore backend. The code generation handles both static and dynamic stencil shapes, with the current implementation focusing on static shapes.

## Results
The benchmarking reveals several key findings:

- The unoptimized Futhark stencil implementation performs significantly worse than tabulate-based alternatives
- The tiled implementation shows substantial improvement, approaching the performance of tabulate-based code
- Adding partitioning provides additional speedup, with the 2D 9-point stencil actually outperforming tabulate-based implementations
- Performance gains from tiling alone were modest in the prototypes but significant in the Futhark implementation
- Tile size selection remains an open question, with no clear pattern emerging from the benchmarks

## Relevance
This work is highly relevant to language design and compiler optimization for parallel computing. Stencil computations are fundamental in scientific computing, image processing, and machine learning. The thesis demonstrates how domain-specific optimizations can significantly improve performance while maintaining high-level abstractions. The findings about cache utilization and the importance of boundary handling provide valuable insights for compiler writers targeting multicore architectures. The exploration of tile size optimization also highlights the challenges of autotuning in compiler design.
