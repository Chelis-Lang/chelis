# Two optimizations to GPU code generation in the Futhark compiler

## Metadata
- **Authors:** Christian Påbøl and Anders Holst
- **Supervisor:** Cosmin E. Oancea
- **Venue/Year:** February 14, 2024

## Summary
This report documents two modifications made to GPU code generation in the Futhark compiler. The first optimization implements thread block virtualization in single-pass segmented scan kernels, while the second optimizes sequentialization in certain segmented reduction kernels with noncommutative, primitive, and non-vectorized operators. The authors detail the implementation of both changes, present validation testing results, and provide comprehensive benchmarking analysis comparing the new implementations against reference implementations on NVIDIA A100 and RTX 4090 GPUs.

## Key Contributions
- Implementation of thread block virtualization for single-pass scan kernels in Futhark
- Optimization of sequentialization in non-commutative reduction kernels by introducing a chunking factor
- Comprehensive validation testing across various kernel parameters
- Extensive benchmarking comparing new implementations against reference implementations on two different GPU architectures
- Analysis of performance patterns and optimal parameterization strategies

## Technical Approach
The first optimization extends the single-pass scan kernel with thread block virtualization by wrapping the kernel body in a virtualization loop that dynamically calculates the number of iterations based on the physical thread block index and the number of logical blocks. This allows the kernel to respect the NUM TBLOCKS parameter embedded in the Futhark intermediate representation.

The second optimization improves sequentialization in non-commutative reductions by introducing a chunking factor (CHUNK) that reduces the number of thread block reductions in the stage one virtualization loop. This is achieved by strip-mining the loop with factor CHUNK, where each thread reads CHUNK elements per iteration and performs per-thread reductions before the thread block reduction. The implementation includes firing conditions that ensure the optimization only applies to non-commutative, non-segmented or large-segmented reductions with primitive operators.

## Results
Validation testing confirmed that both implementations produce correct results across various parameterizations including different TBLOCK SIZE values (32, 448, 1024, and additional odd values for reduce), NUM TBLOCKS values (1, 31, 1024, 2^31-1), and CHUNK values (1-40). All validation tests in the Futhark CI test suite and additional test cases passed successfully.

Benchmarking results showed significant performance improvements on the A100 GPU, with speedups ranging from 2x to 4x across different test programs compared to the reference implementation. The RTX 4090 showed more mixed results, with the new implementation performing equally well or slightly better than the reference for most programs, though achieving significant speedups for mssp.fut and lssp.fut where the reference implementation performed poorly. The analysis revealed that optimal CHUNK values varied across programs and devices, with no universal pattern, and that lower TBLOCK SIZE values generally provided more stable performance.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it demonstrates practical optimizations for GPU code generation in a functional array programming language. The optimizations address fundamental challenges in parallel computing, including efficient memory access patterns and sequentialization in non-commutative operations. The comprehensive benchmarking across different GPU architectures provides valuable insights into performance portability and the importance of parameterization in GPU kernels. The findings have implications for compiler writers targeting GPUs and for developers of high-performance computing systems who need to understand the trade-offs between different optimization strategies.
