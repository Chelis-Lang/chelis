# Implementing a CUDA Backend for Futhark

## Metadata
- **Authors:** Jakob Stokholm Bertelsen
- **Venue/Year:** BSc Thesis, University of Copenhagen, January 2019

## Summary
This thesis describes the implementation of a CUDA backend for the Futhark compiler, which previously only supported generating GPGPU code through the OpenCL framework. The project aims to expand Futhark's device compatibility and enable performance comparisons with CUDA programs. The implementation leverages the similarities between CUDA and OpenCL programming models, particularly in their memory models and device code structures. The CUDA backend uses the driver API with NVRTC for runtime compilation of device code to PTX. The thesis includes a comprehensive evaluation of the backend using Futhark's test suite and benchmark suite, demonstrating that the CUDA backend passes all tests and performs similarly to the OpenCL backend for most benchmark programs, though some programs show significant performance differences.

## Key Contributions
- Implementation of a CUDA backend for the Futhark compiler
- Use of the driver API with NVRTC for runtime compilation of device code
- Creation of a CUDA prelude to translate OpenCL C code to CUDA device code
- Comprehensive testing and performance evaluation of the CUDA backend
- Analysis of performance differences between CUDA and OpenCL backends

## Technical Approach
The implementation builds upon the existing OpenCL backend design, reusing as much code as possible to maintain consistency and reduce code duplication. The key technical approaches include:

1. **Device Code Generation**: A CUDA prelude is used to translate OpenCL C code to CUDA device code. This prelude defines functions like `get_local_id` and `get_group_id` that evaluate to their corresponding CUDA expressions, as well as wrapper functions for atomic operations and address space specifiers.

2. **Host Code Generation**: The host code is generated using the driver API with NVRTC for runtime compilation of device code to PTX. Kernel launches are translated to calls to `cuLaunchKernel`, and memory operations are translated to appropriate CUDA API functions.

3. **Shared Memory Handling**: A key difference between OpenCL and CUDA is how shared memory is passed to kernels. In CUDA, dynamic shared memory is accessed through a single extern pointer, while in OpenCL, it's passed as pointer arguments. The implementation handles this by replacing OpenCL kernel pointers with integers specifying offsets into the shared memory buffer.

## Results
The CUDA backend was extensively tested using Futhark's test suite and benchmark suite. The results show that:

- The CUDA backend passes all 1972 test cases in the test suite and all 400 test cases in the benchmark suite, indicating correctness comparable to the OpenCL backend.
- Performance comparison with the OpenCL backend shows that about 60% of runtimes are within roughly a 10% range of their OpenCL equivalents.
- For some benchmark programs, the CUDA backend is significantly faster or slower than the OpenCL backend. The thesis analyzes one such case (accelerate/ray/trace.fut) and finds that the performance difference is likely due to issues in the device code translation or the CUDA prelude definitions, rather than the host code generation.

## Relevance
This work is highly relevant to language design, compilers, and ML systems for several reasons:

1. **Backend Implementation**: It demonstrates how to implement a new backend for a functional programming language compiler, leveraging similarities between different target frameworks.

2. **GPGPU Programming**: The thesis provides insights into the CUDA and OpenCL programming models, their similarities and differences, and how to translate between them.

3. **Performance Optimization**: The analysis of performance differences between CUDA and OpenCL backends highlights the importance of careful code generation and the potential impact of framework-specific optimizations.

4. **Language Interoperability**: The work shows how a functional language like Futhark can generate code for imperative frameworks like CUDA, bridging the gap between functional and imperative programming paradigms in the context of high-performance computing.
