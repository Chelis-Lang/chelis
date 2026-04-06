# Compositional Deep Learning in Futhark

## Metadata
- **Authors:** Duc Minh Tran, Troels Henriksen, Martin Elsman
- **Venue/Year:** FHPNC '19 (8th ACM SIGPLAN International Workshop on Functional High-Performance and Numerical Computing), 2019

## Summary
This paper introduces a purely functional, statically typed design pattern for building and training deep neural networks in Futhark, a data-parallel functional language that compiles to efficient GPU code. Rather than relying on dynamic computation graphs or Python-embedded DSLs common in mainstream frameworks, the authors represent neural networks as composable pairs of forward and backward functions. This design enables type-safe layer composition, automatic gradient computation via backpropagation, and seamless integration of various layer types (dense, convolutional, pooling, etc.) within a single, extensible library.

The approach leverages Futhark's advanced compiler features, including higher-order functions, higher-order modules, and aggressive array fusion/tiling optimizations. Crucially, all high-level abstractions are eliminated at compile time through defunctionalization and static interpretation, ensuring zero runtime overhead. This allows the library to maintain the expressiveness and safety of functional programming while generating highly optimized, hardware-specific GPU kernels.

The work matters because it challenges the prevailing assumption that high-level functional abstractions inherently sacrifice performance in compute-intensive domains like deep learning. By demonstrating competitive or superior performance against established frameworks like TensorFlow on standard benchmarks, the paper provides a compelling blueprint for building type-safe, composable, and high-performance ML libraries using general-purpose parallel functional languages.

## Key Contributions
- A functional, type-safe design pattern for neural networks that models each layer/network as a record of forward, backward, and weight fields, enabling safe composition via function pairing.
- A complete, extensible deep learning library implemented in Futhark, supporting dense, 2D convolutional, max-pooling, and flatten layers, alongside SGD optimizers and standard loss/activation functions.
- Empirical demonstration that high-level language constructs (higher-order functions, polymorphism, modules) can be compiled to highly efficient GPU code without runtime overhead.
- Benchmark results showing that a non-domain-specific functional language can outperform TensorFlow on dense networks and remain within a factor of two on convolutional networks, highlighting the viability of compiler-driven optimization over hand-tuned kernels for certain workloads.

## Technical Approach
The core technical idea is to represent a neural network (or individual layer) as a record containing three components: a `forward` function, a `backward` function, and a `weights` structure. The forward function takes a training flag, weights, and input data, returning a cache of intermediate values and the layer output. The backward function consumes the cache, upstream errors, and a gradient application function, returning updated weights and propagated errors. This dual-function representation naturally aligns with the mathematical structure of forward propagation and backpropagation.

Layer composition is achieved through a `connect_layers` function that sequentially composes forward functions and reversely composes backward functions. Weights and caches are paired into nested tuples, preserving each layer's native data layout while avoiding 1D array flattening and manual indexing. Type parameters enforce dimensional compatibility between connected layers, catching mismatches early.

The implementation heavily utilizes Futhark's higher-order modules to abstract over scalar precision (`f32`/`f64`) and to define generic interfaces for layers and optimizers. During compilation, Futhark's frontend defunctionalizes higher-order constructs, flattens nested parallelism, and applies aggressive loop fusion and tiling. This transforms the high-level functional composition into flat, data-parallel GPU kernels. Convolutional layers use an explicit `im2col` + GEMM approach for generality, while max-pooling tracks indices during the forward pass to efficiently scatter gradients during the backward pass.

## Results
The authors evaluated the library on the MNIST dataset using an NVIDIA RTX 2080 Ti, comparing against TensorFlow 1.13.1 across batch sizes of 16, 32, 64, and 128.
- **Dense Network (MLP):** Futhark significantly outperformed TensorFlow, achieving speedups of 5.88× to 7.88× across all batch sizes. The authors attribute this to Futhark's ability to fuse and optimize simple layer operations more effectively than TensorFlow's Python-to-graph pipeline.
- **Convolutional Network:** Futhark was faster at smaller batch sizes (1.57× at 16, 1.26× at 32) but slower at larger batch sizes (0.87× at 64, 0.63× at 128). The performance gap at higher batch sizes stems from the explicit `im2col` memory transformation overhead and the lack of highly optimized, proprietary convolution algorithms like those in NVIDIA's cuDNN library.
Overall, the results demonstrate that compiler-driven optimization can match or exceed specialized frameworks for certain architectures, with remaining gaps tied to algorithmic choices rather than language overhead.

## Relevance
- **Language Design:** Demonstrates how to build type-safe, composable ML libraries using higher-order functions and modules without sacrificing performance. The forward/backward function pair pattern offers a clean, mathematically grounded alternative to dynamic computation graphs.
- **Compilers:** Highlights the effectiveness of defunctionalization, loop fusion, and moderate/incremental flattening in bridging the gap between high-level functional code and low-level GPU performance. Shows that general-purpose compilers can compete with domain-specific code generators when abstractions are properly eliminated.
- **ML Systems:** Provides insights into the trade-offs between framework flexibility and performance predictability. The work suggests that static typing and pure functional composition can improve correctness, extensibility, and optimization transparency in ML pipelines, while also identifying where specialized, hand-tuned libraries (e.g., cuDNN) remain necessary for peak performance.
