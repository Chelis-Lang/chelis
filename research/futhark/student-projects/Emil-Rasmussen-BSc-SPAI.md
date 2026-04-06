# Sparse Approximate Inverse - A Massively Parallel Implementation

## Metadata
- **Authors:** Emil Vilandt Rasmussen
- **Venue/Year:** Master's Thesis, 2023

## Summary
This thesis presents a detailed investigation into the implementation of the Sparse Approximate Inverse (SPAI) algorithm, a method for computing a preconditioner for solving large sparse linear systems. The SPAI algorithm is inherently parallel, making it well-suited for GPU acceleration. The author provides both a sequential Python implementation and a partially parallel CUDA implementation, evaluating their performance and correctness. The work focuses on the theoretical foundations of SPAI, including QR decomposition and sparsity pattern updates, and discusses the challenges and potential solutions for parallelizing the algorithm on GPU hardware.

## Key Contributions
- Detailed explanation of the SPAI algorithm, including QR decomposition and sparsity pattern updates.
- Sequential Python implementation of SPAI.
- Partially parallel CUDA implementation of SPAI, utilizing CUBLAS for dense matrix operations.
- Discussion of potential methods for further parallelization, including the use of Futhark.
- Evaluation of the implementations' correctness and performance, comparing results with existing literature.

## Technical Approach
The SPAI algorithm minimizes the Frobenius norm of the error between the approximate inverse and the true inverse of a matrix. This is achieved by solving a series of independent least squares problems for each column of the matrix. The author presents a sequential implementation in Python using NumPy and SciPy, and a partially parallel implementation in CUDA using CUBLAS for dense matrix operations. The CUDA implementation utilizes batched operations for efficiency and discusses potential methods for parallelizing the remaining sequential parts of the algorithm.

## Results
The sequential Python implementation produces results that match those reported in the literature, indicating its correctness. The partially parallel CUDA implementation also produces correct results but has some limitations, such as issues with matrices containing empty columns and a naive stopping criterion that leads to excessive fill-in. The author proposes improvements to address these limitations and discusses potential methods for further parallelization.

## Relevance
This work is relevant to language design, compilers, and ML systems as it explores the implementation of a parallel algorithm on GPU hardware. The author discusses the challenges of parallelizing the SPAI algorithm and proposes potential solutions, including the use of functional programming languages like Futhark. The work also highlights the importance of efficient memory management and coalesced access patterns in GPU programming.
