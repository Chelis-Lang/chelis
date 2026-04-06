# Efficient Big Integer Arithmetic Using GPGPU

## Metadata
- **Authors:** Thorbjørn Bülow Bringgaard
- **Venue/Year:** Master's Thesis, University of Copenhagen, 2024

## Summary
This thesis explores efficient parallel implementations of exact big integer arithmetic for GPGPUs, focusing on addition, multiplication, and division. The algorithms are implemented in CUDA C++ and Futhark, targeting medium-sized big integers. The addition algorithm uses a prefix sum approach, while the multiplication algorithm is the classical quadratic method parallelized by orchestrating convolutions. The division algorithm is based on finding multiplicative inverses without leaving the domain of big integers. The thesis presents the algorithms, optimizations, and implementations, along with a performance evaluation against a state-of-the-art CUDA library.

## Key Contributions
- Description of efficient parallel big integer addition and classical multiplication algorithms with optimizations.
- Benchmark-driven performance evaluation of the implementations against a state-of-the-art CUDA library.
- Presentation of Watt's algorithm for exact division by whole shifted inverse, including a revision for an unconsidered corner case.
- Inefficient sequential prototype of Watt's algorithm in C.
- Partially valid and semi-efficient Futhark implementation of Watt's algorithm, including efficient parallel operators for shifting, comparing, and subtracting big integers.

## Technical Approach
The thesis uses CUDA C++ and Futhark to implement the algorithms. The addition algorithm uses a prefix sum approach, while the multiplication algorithm is the classical quadratic method parallelized by orchestrating convolutions. The division algorithm is based on finding multiplicative inverses without leaving the domain of big integers. The implementations are optimized for block-level processing on GPGPUs.

## Results
The thesis presents benchmark results comparing the performance of the implementations against a state-of-the-art CUDA library. The results show that the implementations are competitive for medium-sized big integers, with the CUDA implementation generally outperforming the Futhark implementation.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it explores efficient parallel implementations of exact big integer arithmetic for GPGPUs. The algorithms and optimizations presented in the thesis can be applied to other domains that require efficient big integer arithmetic, such as cryptography and algebra. The thesis also provides insights into the challenges and opportunities of implementing parallel algorithms on GPGPUs.
