# Accelerated Interest Rate Option Pricing using Trinomial Trees

## Metadata
- **Authors:** Marek Hlava, Martin Metaksov
- **Venue/Year:** Master Thesis, University of Copenhagen, August 2018

## Summary
This thesis explores parallelization strategies for the Hull-White Single-Factor Model, which prices financial derivatives using trinomial trees. The work focuses on finding the optimal balance between different levels of parallelism and trade-offs such as thread divergence vs. locality of reference. The authors present a sequential implementation, followed by three parallel approaches: one-option per thread, multiple options per thread block, and full flattening. They conduct experiments on various datasets to evaluate the performance of each implementation and identify the best-performing versions for different data distributions.

## Key Contributions
- Development of a sequential implementation of the Hull-White Single-Factor Model for validation purposes.
- Creation of a one-option per thread parallel implementation in CUDA and Futhark.
- Design of a multiple options per thread block parallel implementation in CUDA.
- Implementation of a fully-flattened parallel approach in Futhark.
- Systematic evaluation of all implementations on various datasets with different distributions of divergence.
- Identification of performance trade-offs and optimization techniques for each implementation.

## Technical Approach
The thesis presents three parallel approaches for pricing options using the Hull-White Single-Factor Model:

1. **One Option per Thread:** This approach exploits only outer parallelism, where each thread prices a single option. The authors implement four versions with different memory access patterns and padding strategies to optimize performance.

2. **Multiple Options per Thread Block:** This approach exploits both inner and outer parallelism by packing multiple options into a single CUDA block. The authors implement three versions with different memory access patterns and padding strategies.

3. **Full Flattening:** This approach fully flattens the nested parallelism of the algorithm, processing all input options simultaneously. The authors implement this version in Futhark, a functional data-parallel language.

## Results
The authors conduct extensive experiments on seven different datasets with various distributions of divergence. The results show that:

- CUDA-option (one option per thread) dominates the benchmarks on most datasets, achieving up to ~529× speed-up compared to the sequential implementation.
- CUDA-multi (multiple options per thread block) performs best on skewed datasets, providing up to ~2× speed-up over CUDA-option.
- CUDA-multi outperforms CUDA-option by up to ~13× for floats and ~11× for doubles when the dataset is small enough.
- The performance of each implementation is sensitive to the data distribution, highlighting the importance of dynamic analysis to choose the optimal strategy for each dataset.

## Relevance
This thesis is relevant to language design, compilers, and ML systems work because:

- It provides insights into the challenges of efficiently porting complex applications to GPGPU hardware.
- It demonstrates the importance of finding the right balance between parallelism and trade-offs such as thread divergence and locality of reference.
- It showcases the use of different programming models (CUDA and Futhark) for implementing parallel algorithms.
- It highlights the need for dynamic analysis to choose the optimal implementation based on the dataset characteristics.
- The findings and techniques presented in this thesis can be applied to other applications with similar characteristics, such as nested parallelism and irregular data access patterns.
