# Massively-Parallel Change Detection for Satellite Time Series Data with Missing Values

## Metadata
- **Authors:** Fabian Gieseke, Sabina Rosca, Troels Henriksen, Jan Verbesselt, Cosmin E. Oancea
- **Venue/Year:** Not explicitly stated in the provided excerpt (likely a recent HPC/Remote Sensing conference, ~2020)

## Summary
Satellite time-series data is critical for monitoring large-scale environmental changes such as deforestation, but existing change detection algorithms struggle to scale to billions of pixels. The primary bottleneck stems from the computational intensity of fitting per-pixel regression models and the irregularity of real-world data, where cloud cover frequently introduces missing values at different timestamps across pixels. Traditional implementations in high-level languages like R or Python cannot handle continental-scale analyses without prohibitive compute resources or weeks-to-years of runtime.

To address this, the authors present a massively-parallel GPU implementation of BFAST-Monitor, a state-of-the-art unsupervised change detection algorithm. The approach leverages a high-level functional specification in the Futhark language and introduces a novel parallelization strategy that efficiently maps nested, irregular parallelism to modern GPU architectures. By combining targeted compiler transformations with performance-critical kernel optimizations, the system handles variable-length time series with missing values without sacrificing hardware utilization.

The resulting implementation achieves speedups of up to three orders of magnitude over standard CPU/R baselines, enabling researchers to perform global-scale environmental monitoring on commodity desktop hardware. Beyond remote sensing, the algorithmic building blocks and parallelization techniques provide a reusable blueprint for accelerating other scientific and machine learning workloads that operate on large, irregular, or sparse datasets.

## Key Contributions
- **High-level parallel specification** of BFAST-Monitor that natively supports irregular time series with arbitrary missing values, expressed in the functional data-parallel language Futhark.
- **A hybrid GPU mapping strategy** that groups operations by inner-parallel size and pads them to maximum dimensions, striking a balance between full flattening and per-thread sequentialization.
- **Performance-critical kernel optimizations**, including register-tiling for batched masked matrix multiplication and shared-memory Gauss-Jordan elimination for batched matrix inversion.
- **Open-source, production-ready implementation** that achieves 24–48× speedup over optimized multi-threaded C and ~1000–5000× over standard R, enabling the first-ever continental-scale deforestation analysis on a single GPU.

## Technical Approach
The core algorithm, BFAST-Monitor, fits a linear regression model (capturing trend and seasonality) over a historical period, then applies a Moving Sums (MOSUM) statistical test over a monitoring period to detect structural breaks. Missing values (NaNs) are filtered per-pixel, resulting in highly irregular computation and memory access patterns across the dataset.

To parallelize this efficiently on GPUs, the authors avoid two extremes: (1) mapping each pixel to a single thread (which underutilizes hardware and suffers from thread divergence) and (2) fully flattening all nested parallelism (which destroys temporal locality and increases global memory traffic). Instead, they distribute the outer pixel-level loop across groups of inner-parallel operations of the same size, padding each group to its maximum dimension. Each group is compiled into a separate CUDA/OpenCL kernel.

Two key optimizations drive performance:
1. **Register-Tiled Masked Batch MatMul:** Computes $X^T X$ while dynamically ignoring NaN-masked rows. The transformation tiles loops, hoists invariant computations, and uses collective copies to move data into fast shared memory, reducing global memory accesses by ~30–45×.
2. **Shared-Memory Batch Matrix Inversion:** Implements Gauss-Jordan elimination for small matrices (e.g., $8 \times 8$) entirely within CUDA shared memory. Threads within a block synchronize to iteratively update the matrix, achieving a 5–6× speedup over global-memory baselines.

The entire pipeline is automatically compiled from Futhark to GPU code, with a Python/PyOpenCL wrapper handling data chunking, I/O, and host-device transfers for datasets exceeding GPU memory.

## Results
- **Kernel-Level Benchmarks:** Register-tiling yields 2–3× speedup for masked matrix multiplication; shared-memory inversion achieves 5–6× speedup. Performance remains stable across varying dataset sizes and NaN frequencies.
- **Application-Level Speedups:** The optimized GPU implementation runs 24–48× faster than a highly tuned 32-thread OpenMP C implementation, and ~1000–5000× faster than the standard R implementation.
- **Large-Scale Validation:** Processed a real-world dataset covering continental tropical Africa (~221k × 768 pixels across 38,234 images) in ~90 hours per monitoring period on a single GPU. The same task would take years using existing tools.
- **Overhead Analysis:** Data transfer and preprocessing account for a manageable fraction of runtime and can be overlapped with kernel execution, making the GPU compute phase the dominant and highly optimized component.

## Relevance
- **Compiler & Language Design:** Demonstrates how high-level functional/data-parallel languages (Futhark) can bridge the gap between algorithmic clarity and hardware efficiency. The paper provides a practical case study in compiler transformations for handling irregular parallelism, padding strategies, and automatic kernel generation.
- **GPU Systems & Architecture:** Offers a proven middle-ground parallelization strategy for workloads with ragged/variable-length inputs, directly applicable to ML systems processing sparse time-series, graph data, or dynamic sequences. The register-tiling and shared-memory inversion patterns are reusable templates for custom GPU kernels.
- **ML & Scientific Computing Pipelines:** The optimized batched linear algebra routines with dynamic masking are highly relevant to training/inference pipelines dealing with missing data, attention mechanisms with variable sequence lengths, and large-scale geospatial AI. The work also highlights how domain-specific scientific algorithms can be democratized through systematic hardware-aware optimization.
