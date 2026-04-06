# Parallel Implementation of the SPAI Algorithm

## Metadata
- **Authors:** Caroline Amalie Kierkegaard, Mikkel Willén
- **Venue/Year:** Master's Thesis, University of Copenhagen, 2023

## Summary
This thesis explores the Sparse Approximate Inverse (SPAI) preconditioner, which calculates an approximate inverse of a large and sparse matrix. The SPAI algorithm iteratively constructs a sparse approximation of an inverse matrix by minimizing the Frobenius norm, column by column, while preserving the sparsity. It is inherently parallel and thus the authors aim to execute it efficiently on the Graphics Processing Unit (GPU). The thesis elaborates on the original SPAI algorithm by Grote and Huckle by incorporating theoretical explanations of QR decomposition with Householder reflections and permutations. The authors implement sequential prototypes in Python and C, followed by a parallel version using CUDA kernels. Experimental results demonstrate the accuracy of the SPAI algorithm. They compare the sequential SPAI implementations to Scipy and cuSOLVERs functions for finding the exact inverse and find the run time of the library functions superior. They perform experiments on the CUDA kernels implemented in the parallel SPAI implementation. The results show that the kernels executed on the GPU exhibit superior run time to the corresponding sequential code running on the Central Processing Unit (CPU). This thesis shows that the SPAI preconditioner efficiently computes approximate inverses of e.g. large Hessian matrices in the Newton method. They show that the parallel implementation running on the GPU outperforms the sequential implementations running on the CPU.

## Key Contributions
- Elaborates on the original SPAI algorithm by Grote and Huckle by incorporating theoretical explanations of QR decomposition with Householder reflections and permutations.
- Implements sequential prototypes in Python and C, followed by a parallel version using CUDA kernels.
- Demonstrates the accuracy of the SPAI algorithm through experimental results.
- Compares the sequential SPAI implementations to Scipy and cuSOLVERs functions for finding the exact inverse and finds the run time of the library functions superior.
- Performs experiments on the CUDA kernels implemented in the parallel SPAI implementation and shows that the kernels executed on the GPU exhibit superior run time to the corresponding sequential code running on the CPU.

## Technical Approach
The authors implement the SPAI algorithm in three stages: a sequential prototype in Python, a sequential version in C, and a parallel version using CUDA kernels. The Python implementation serves as a prototype for understanding the algorithm in depth. The C implementation uses cuBLAS library functions for QR decomposition and inverse calculations of R. The CUDA implementation uses kernels to perform the computations in parallel on the GPU.

## Results
The authors conduct experiments on randomized matrices of varying sizes and sparsity levels to measure the accuracy and speed of their implementations. The results demonstrate the accuracy of the SPAI algorithm. They compare the runtime of their Python and C SPAI implementations with Scipy and cuSOLVERs functions for exact inverse computation. The results show that library functions exhibit superior runtimes to the sequential SPAI implementations. The authors also test individual CUDA kernels and show that the kernels executed on the GPU exhibit superior run time to the corresponding sequential code running on the CPU.

## Relevance
This paper is relevant to language design, compilers, and ML systems work because it demonstrates the effectiveness of parallelizing the SPAI algorithm on GPUs. The SPAI algorithm is inherently parallel, and the authors show that it can be efficiently implemented on GPUs using CUDA kernels. This work could inspire further research into parallelizing other algorithms for GPUs and developing new languages and compilers that are optimized for GPU execution.
