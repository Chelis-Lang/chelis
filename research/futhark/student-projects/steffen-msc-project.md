# Futhark Vulkan Backend

## Metadata
- **Authors:** Steffen Holst Larsen
- **Venue/Year:** BSc Thesis, January 6, 2019

## Summary
This paper describes the implementation of a Vulkan backend for the Futhark compiler, which compiles Futhark programs to run on GPUs using the Vulkan API instead of the existing OpenCL backend. The work explores the challenges and limitations of using Vulkan, which is a more modern and explicit API compared to OpenCL. The Vulkan backend generates SPIR-V shaders and uses C host code with Vulkan API calls. While most Futhark programs compile and run correctly, there are still bugs and performance issues, with the Vulkan backend generally being slower than the OpenCL backend. The author suggests that with further development and Vulkan API maturation, the Vulkan backend could become a viable alternative.

## Key Contributions
- Implementation of a Vulkan backend for Futhark, extending the compiler pipeline without structural changes.
- Generation of SPIR-V shaders from Futhark's intermediate kernel representation.
- Handling of Vulkan-specific challenges such as descriptor sets, command buffer reuse, and ownership synchronization.
- Discussion of limitations, including limited 8-bit integer support and missing pointer conversion capabilities.

## Technical Approach
The Vulkan backend follows the same GPU computation model as OpenCL but requires more explicit setup. The implementation involves:
- Creating a Vulkan instance, selecting a physical device, and creating a logical device with necessary extensions and features.
- Managing descriptor sets and pools for kernel parameters, ensuring proper synchronization and reuse of command buffers.
- Generating SPIR-V shaders from Futhark's intermediate representation, handling control structures, expressions, and variable declarations.
- Using GLSL extensions for additional mathematical operations and enabling 8-bit storage extensions for limited 8-bit integer support.

## Results
- Compilation time for Futhark programs using the Vulkan backend is about 21% slower than the OpenCL backend.
- Execution time for most benchmarks is slower with the Vulkan backend, with some programs experiencing extreme slowdowns (e.g., 224x slower for FFT).
- The Vulkan backend successfully runs the entire Futhark test suite, but some benchmarks fail due to bugs or performance issues.

## Relevance
This paper is relevant to language design, compilers, and ML systems work as it explores the use of a modern GPU compute API (Vulkan) for a functional array language (Futhark). The implementation provides insights into the challenges of targeting GPUs with Vulkan and highlights the trade-offs between using Vulkan and OpenCL. The work also contributes to the broader goal of making Futhark more accessible on a wider range of hardware, including platforms where OpenCL is not supported.
