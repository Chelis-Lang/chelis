# Optimizing Convolutions for GPU Execution in Futhark

## Metadata
- **Authors:** Emil Vilandt Rasmussen, Jóhann Utne
- **Venue/Year:** University of Copenhagen, Department of Computer Science (Master's Thesis) / 2025

## Summary
This thesis addresses the performance gap in compiling 2D discrete convolutions—a foundational operation in Convolutional Neural Networks (CNNs) responsible for up to 90% of inference/training runtime—for GPU execution in Futhark, a purely functional array programming language. While Futhark’s compiler already features sophisticated block-register tiling for General Matrix Multiplication (GEMM), it lacks equivalent, automated transformations for convolution operations. The authors aim to bridge this gap by developing and evaluating locality-optimizing transformations specifically tailored to convolution loop nests.

To establish a performance baseline and guide compiler design, the authors first implement a series of highly optimized CUDA kernels. They then manually apply a subset of these transformations to Futhark’s intermediate representation (IR) to prototype the intended compiler pass. Although full integration into the Futhark compiler was not completed within the project timeline, the work demonstrates a clear, validated pathway for automating convolution optimizations in functional array languages. The research highlights the critical role of memory hierarchy management, loop transformations, and hardware-aware tiling in achieving near-peak GPU performance for ML workloads.

## Key Contributions
- Development of a highly optimized direct convolution CUDA kernel achieving 89–96.3% of A100 peak TFLOPs through block-register tiling, shared memory caching, vectorized loads, and branchless execution.
- Formalization of loop transformation techniques (strip-mining, interchange, distribution, and register-tiling in y/z dimensions) specifically adapted for 2D convolution loop nests.
- A hand-transformed Futhark IR prototype demonstrating a ~3.2× speedup over naive Futhark implementations, validating the feasibility and performance potential of compiler-level convolution optimizations.
- Comprehensive empirical analysis of tiling parameter sensitivity, filter radius impact, and channel count effects on GPU arithmetic intensity and memory-bound vs. compute-bound behavior.

## Technical Approach
The core methodology revolves around classical loop nest optimization techniques adapted for GPU architectures and functional IRs:
1. **Block & Register Tiling:** The authors strip-mine outer parallel loops (output channels, height, width) and interchange them inward to create tiles that map to GPU thread blocks. Register tiling in the y and z dimensions enables temporal reuse of filter weights and input activations across multiple output elements.
2. **Memory Hierarchy Optimization:** Input tiles and filter weights are explicitly copied into shared memory to minimize global memory latency. Vectorized 128-bit loads (`float4`) maximize memory bandwidth, with careful tensor padding to guarantee natural alignment and avoid bank conflicts.
3. **Branch Elimination & Padding:** Out-of-bounds boundary checks are removed from the hot compute loop by pre-padding input and filter tensors. This eliminates warp divergence, maximizes instruction throughput, and allows full loop unrolling.
4. **Futhark IR Manipulation:** The authors manually modify Futhark’s `SegMap` IR to inject register-tiled loops, adjust array shapes, and preserve the convolution structure by disabling the existing `tile-loop` pass. This serves as a blueprint for a future pattern-matching compiler pass that would automatically recognize convolution redomap patterns and apply these transformations.

## Results
Benchmarks were conducted on an NVIDIA A100 GPU (19.5 TFLOPS peak FP32). The optimized CUDA kernels consistently outperformed baseline versions, with the final vector-load shared-memory version reaching 18.88 TFLOPS (96.3% of peak) for large inputs (4096×4096, 64 channels) and moderate filter radii. Performance degraded for very small workloads or single-channel inputs due to reduced parallelism and inability to tile in the z-dimension. The hand-transformed Futhark kernel achieved ~10.2 TFLOPS (~51% of peak), representing a ~3.2× speedup over the unoptimized Futhark baseline (~3.19 TFLOPS). The results confirm that block-register tiling and shared memory utilization are highly effective for convolutions, though optimal tiling parameters are workload-dependent and would benefit from runtime auto-tuning.

## Relevance
This work is highly relevant to compiler design for domain-specific languages (DSLs) and ML systems. It demonstrates how functional array languages like Futhark can be extended with hardware-aware optimizations traditionally reserved for imperative frameworks (e.g., cuDNN, TVM). The detailed breakdown of loop transformations, memory padding strategies, and IR manipulation provides a practical blueprint for implementing convolution-specific passes in polyhedral or tiling-based compilers. Furthermore, the emphasis on auto-tuning thresholds, handling boundary conditions via "epilogues," and mapping high-level functional combinators to low-level GPU memory hierarchies aligns closely with modern ML compiler research (e.g., MLIR, Triton), making it valuable for researchers building next-generation, high-performance, and memory-efficient deep learning runtimes.
