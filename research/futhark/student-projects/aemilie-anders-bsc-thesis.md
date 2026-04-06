# Teaching the Futhark compiler block and register tiled matrix multiplication

## Metadata
- **Authors:** Æmilie Cholewa-Madsen, Anders Lietzen Holst, Rasmus Wulff Christensen
- **Venue/Year:** Bachelor's thesis, University of Copenhagen, 2020

## Summary
This thesis explores how to improve the performance of matrix multiplication programs in the Futhark compiler through block and register tiling optimizations. The authors analyze the theoretical benefits of these transformations, implement them in the Futhark compiler's loop tiling pass, and evaluate their effectiveness. They successfully implement block and register tiling for ordinary matrix multiplication, achieving promising speedups on modern GPU hardware (Nvidia RTX 2080ti), though their implementation falls short of handwritten optimized kernels. The work demonstrates the potential of automated compiler optimizations for improving GPU performance while highlighting the challenges in generalizing such optimizations to broader program patterns.

## Key Contributions
- Theoretical analysis of block and register tiling transformations for matrix multiplication
- Design and implementation of RegTileReturns, a new KernelResult constructor in the Futhark IR
- Implementation of block and register tiling in the Futhark compiler's loop tiling pass
- Validation testing showing correct implementation across various input sizes and tile configurations
- Benchmarking results demonstrating performance improvements over existing block tiling
- Discussion of limitations and future work for generalizing the optimization

## Technical Approach
The authors implement block and register tiling through a systematic code transformation process. They start with a naive matrix multiplication program and apply a series of transformations: loop stripmining to create tiles, loop interchange to improve data access patterns, array expansion of the accumulator variable, loop distribution to separate different operations, and sequentialization/unrolling of innermost loops for register tiling. The implementation handles residual input through boundary checks and an epilogue approach for the common dimension. They design RegTileReturns to convey necessary information for generating per-thread write indices and boundary guards in the final result array.

## Results
Validation tests show the implementation correctly handles various input sizes and tile configurations, including cases with partial tiles in different dimensions. Benchmarking on two GPUs (RTX 2080ti and GTX 780ti) demonstrates that the block/register tiled kernel outperforms the existing block-tiled kernel for most input sizes on the RTX 2080ti, with speedups ranging from 1.2x to 1.82x. On the GTX 780ti, the implementation performs roughly as well as block tiling. However, both implementations fall significantly short of handwritten optimized kernels, which achieve 2-4x better performance due to additional optimizations not implemented in the compiler.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates how automated compiler optimizations can significantly improve the performance of parallel programs on GPU hardware. The research addresses the challenge of generating efficient GPU code from high-level functional languages, showing that sophisticated transformations like block and register tiling can be implemented in a compiler to achieve performance close to hand-optimized code. The limitations identified and the discussion of generalization challenges provide valuable insights for future compiler development, particularly for handling more complex program patterns beyond simple matrix multiplication.
