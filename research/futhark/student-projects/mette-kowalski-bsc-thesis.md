# Designing and Accelerating a Generic FFT Library in Futhark

## Metadata
- **Authors:** Mette Marie Kowalski
- **Venue/Year:** BSc Thesis, University of Copenhagen, 2018

## Summary
This thesis presents the design and implementation of a generic Fast Fourier Transform (FFT) library in the data-parallel programming language Futhark. The library is designed to be generic with respect to different data-sets and radix, while being transparent to the user. The study explores the extent to which FFT computations can be efficiently and generically expressed in a high-level, hardware-independent language. The results show that the radix has a significant effect on performance, with higher radix generally giving higher performance. Other optimizations, such as algorithmic improvements and compiler optimizations, also show varying increases in performance depending on the hardware. While the implementation is still slower than the industry standard cuFFT, it holds potential for future improvements as the Futhark compiler is optimized.

## Key Contributions
- Implementation of a generic FFT library in Futhark
- Exploration of efficient and generic expression of FFT computations in a high-level language
- Analysis of the effects of radix and other optimizations on performance
- Design of a modular and user-transparent library structure

## Technical Approach
The thesis describes the implementation of the FFT library in three main modules:
1. **FFT Iteration Module**: Implements the calculation of an FFT for a sequence with the length of the radix (2, 4, 8, etc.).
2. **Main FFT Module**: Implements the skeleton of the FFT algorithm, a loop with lg N iterations that updates and shuffles elements in each iteration.
3. **Planner-Executor Module**: Creates a plan by pre-computing information that depends on the input size and executes the plan.

The library supports both global and shared memory implementations, with optimizations such as pre-computation of twiddle factors. The implementation uses a generic representation of the radix to minimize code clones and allows the user to choose between different implementations based on the input data-set.

## Results
The performance of the implementation was evaluated on both AMD and NVIDIA GPUs. The results show that:
- Higher radix generally performs better than lower radix.
- Shared memory implementation performs better than global memory in most cases.
- Pre-computation of twiddle factors has varying effects on performance depending on the hardware.
- The implementation is still slower than the industry standard cuFFT but shows potential for future improvements.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates the potential of high-level, data-parallel programming languages like Futhark for implementing complex algorithms like FFTs. The study also highlights the importance of optimizing for specific hardware architectures and the potential benefits of using shared memory and other optimizations. The modular and user-transparent design of the library provides insights into how to structure and implement complex algorithms in a way that is both efficient and easy to use.
