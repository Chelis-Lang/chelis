# A WebGPU backend for Futhark

## Metadata
- **Authors:** Sebastian Paarmann
- **Venue/Year:** MSc Thesis, University of Copenhagen, May 2024

## Summary
This thesis presents the development of a new backend for the Futhark compiler that targets WebGPU, enabling Futhark programs to run in web browsers while leveraging GPU compute capabilities. Futhark is a functional data-parallel array programming language designed for high-performance parallel computing. The new backend generates WebGPU Shading Language (WGSL) code for GPU kernels and implements a host-side runtime system for the WebGPU API. The work includes support for Futhark's built-in testing and benchmarking tools by interacting with a browser to run programs under test. The thesis discusses the challenges of translating Futhark's internal kernel representation to WGSL, given WGSL's restrictions compared to other GPU APIs like OpenCL and CUDA. While the backend can successfully run some Futhark programs, limitations remain due to WebGPU and WGSL constraints, such as missing primitive types, restricted pointer operations, and uniformity analysis requirements. The thesis evaluates the backend's usability, performance, and suitability of WebGPU as a target, and presents a demonstration web page showcasing a Mandelbrot set calculation.

## Key Contributions
- Implementation of a WebGPU backend for the Futhark compiler.
- Generation of WGSL shader code from Futhark's internal kernel representation.
- Development of a host-side runtime system for the WebGPU API.
- Support for Futhark's testing and benchmarking tools in a web environment.
- Investigation and discussion of limitations and potential future solutions.

## Technical Approach
The backend translates Futhark's internal kernel representation (ImpCode) into WGSL shaders. This involves handling WGSL's restrictions on pointer types, array sizes, and uniformity analysis. The host-side code is generated using Emscripten to compile C code to WebAssembly, with a custom implementation of the WebGPU API abstraction layer. The backend also includes a JavaScript interface to simplify interaction with compiled Futhark programs. Testing support is provided through a Python wrapper that controls a browser using Selenium to run Futhark programs and communicate results back to the test framework.

## Results
The backend successfully runs some Futhark programs, particularly those with simple parallel structures like maps and reductions. However, limitations in WebGPU and WGSL prevent support for more complex kernels, 64-bit floating-point numbers, and certain atomic operations. Performance benchmarks show significant overhead compared to native CUDA backends, likely due to the JavaScript and WebAssembly host code. The Mandelbrot set demo demonstrates the feasibility of embedding Futhark programs in web pages for GPU-accelerated computations.

## Relevance
This work is relevant to language design, compilers, and ML systems as it extends Futhark's portability to web environments, enabling high-performance functional programming in browsers. It also contributes to the understanding of WebGPU and WGSL as compilation targets, highlighting their current limitations and potential for future development. The techniques developed for handling WGSL's restrictions may inform future compiler designs targeting similar APIs.
