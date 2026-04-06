# Approximate Nearest-Neighbour Fields via Massively-Parallel Propagation-Assisted K-D Trees

## Metadata
- **Authors:** Cosmin Eugen Oancea, Ties Robroek, Fabian Gieseke
- **Venue/Year:** Not explicitly stated in the provided text (formatted as an IEEE conference paper)

## Summary
Computing Approximate Nearest-Neighbour Fields (ANNFs) is a foundational operation in computer vision, enabling tasks such as image inpainting, super-resolution, and denoising by mapping similar patches between two images. However, exact nearest-neighbour search in high-dimensional patch spaces is computationally prohibitive, especially as image resolutions scale to 4K and beyond. While prior work has introduced approximations like PatchMatch, Coherency Sensitive Hashing (CSH), and propagation-assisted k-d trees, these methods still face significant runtime bottlenecks when processing large-scale, high-resolution image pairs on traditional multi-core CPUs.

To address this, the authors present a highly optimized, massively-parallel GPU implementation of propagation-assisted k-d trees. The approach combines spatial pruning via k-d trees with spatial coherence propagation, but fundamentally restructures the algorithm to align with modern GPU execution models. By leveraging block-level parallelism, shared memory tiling, and memory-efficient data layouts, the framework drastically reduces thread divergence and global memory traffic. The result is a system that maintains high approximation accuracy while delivering order-of-magnitude speedups over state-of-the-art CPU implementations.

This work matters because it demonstrates how algorithmic redesign tailored to parallel hardware can unlock practical, real-time ANNF computation for modern high-resolution vision pipelines. The framework's low memory footprint enables processing of 4K imagery on commodity GPUs, and its open-source release provides a reusable, high-performance building block for data-intensive computer vision and machine learning applications.

## Key Contributions
- **Massively-Parallel GPU Implementation:** A novel, highly optimized GPU version of propagation-assisted k-d trees tailored for ANNF computation.
- **Hardware-Aware Algorithmic Redesign:** Replaces naive thread-per-query mapping with a CUDA block-per-query strategy, lifting thread divergence to the block scheduler and maximizing shared memory utilization.
- **Memory-Efficient Pipeline:** Fuses patch creation and dimensionality reduction, reuses global memory buffers, and avoids materializing large intermediate arrays, enabling 4K image processing on 8GB GPUs.
- **Comprehensive Benchmarking:** Introduces the `VidPairs4K` dataset and demonstrates 5–10× speedups over optimized multi-threaded CPU baselines while maintaining or improving L2 accuracy.
- **Open-Source Release:** Provides a publicly available Python package and Futhark source code for reproducible research and integration into vision pipelines.

## Technical Approach
The core algorithm builds on He & Sun's propagation-assisted k-d trees, which combine spatial subdivision with row-wise propagation to approximate nearest neighbours efficiently. The pipeline proceeds as follows:
1. **Preprocessing & Dimensionality Reduction:** Patches are extracted from both images and projected into a lower-dimensional space using PCA fitted on a random subset, reducing the search space dimensionality.
2. **Balanced k-d Tree Construction:** The reference image's reduced patches are used to build a perfectly balanced k-d tree. Padding ensures uniform leaf sizes, enabling regular nested parallelism. Construction is performed level-by-level using parallel radix sort and map-reduce to select split dimensions.
3. **Initial Candidate Extraction:** Each query patch traverses the tree top-down to find its containing leaf. Queries are then sorted by leaf index to improve cache locality during brute-force leaf searches.
4. **Propagation & Refinement (`PROCESSROWS`):** Patches are processed row-by-row. For each patch, candidate leaves are inferred from the nearest neighbours of the patch directly above. Instead of assigning one thread per query, the implementation assigns one CUDA block per query. Distance computations and candidate updates are performed in shared memory using block-level reductions, and partially sorted candidate lists are merged efficiently.
5. **Re-ranking (`SELECTBEST`):** The top `k` candidates are evaluated against the original high-dimensional patches to select the final nearest neighbour.
The entire pipeline is implemented in **Futhark**, a functional data-parallel language that compiles to efficient CUDA/OpenCL code, leveraging its support for nested parallelism, incremental flattening, and dynamic compilation optimizations.

## Results
- **Speedup:** Achieves **5–10× speedup** over a highly optimized OpenMP multi-core CPU implementation across varying image sizes and patch dimensions. The GPU runtime remains relatively stable as resolution increases, while CPU time scales poorly.
- **Optimization Impact:** The block-level parallelism optimization alone yields a **3.8–5.2× speedup** over a naive thread-per-query GPU implementation.
- **Accuracy:** Maintains competitive or superior L2 error scores compared to CSH and the CPU baseline. Visual reconstructions exhibit sharper edges and better color fidelity, with more evenly distributed error maps.
- **Runtime Breakdown:** The propagation step (`PROCESSROWS`) dominates execution (~45–55%), followed by exact first-row search (~8–14%) and k-d tree construction (~7–16%). The overhead of Python bindings and PCA fitting is negligible (<6%).
- **Scalability:** Handles 4K resolution image pairs in under 2 seconds on an RTX 2070, demonstrating strong weak and strong scaling with respect to image size and patch dimensionality.

## Relevance
- **Language Design & Compilers:** Showcases the practical value of high-level functional data-parallel languages (Futhark) for generating production-grade GPU code. The paper highlights how compiler-driven transformations (e.g., incremental flattening, nested parallelism mapping, and automatic memory layout optimization) can outperform hand-tuned CUDA for irregular, data-dependent workloads.
- **ML Systems & Computer Vision:** ANNFs are critical for non-local means filtering, neural style transfer, self-supervised representation learning, and image editing. Efficient, low-memory GPU kernels for nearest-neighbour search directly accelerate these pipelines and enable scaling to higher resolutions without specialized hardware.
- **Hardware-Aware Algorithm Design:** Provides a clear case study in restructuring algorithms to match GPU execution models (SIMT, warp scheduling, memory hierarchy). Techniques like padding for regularity, block-level divergence management, and shared-memory tiling are broadly applicable to other spatial search and graph traversal problems in ML systems.
