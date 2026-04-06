# WebAssembly Backends for Futhark

## Metadata
- **Authors:** Philip Lassen
- **Venue/Year:** MSc Thesis, University of Copenhagen, 2021

## Summary
This thesis develops WebAssembly backends for the Futhark programming language, enabling efficient execution of Futhark programs in web browsers. Futhark is a high-performance purely functional data-parallel array programming language targeting parallel compute hardware. The thesis adds two new backends: one targeting WebAssembly for sequential execution and another targeting threaded WebAssembly for parallel execution on multicore CPUs. These backends allow Futhark programs to run efficiently in browsers, leveraging the performance benefits of WebAssembly while maintaining the language's high-level abstractions for parallel programming. The thesis also designs a JavaScript API for convenient interoperation between Futhark and browser applications.

## Key Contributions
- Design and implementation of WebAssembly and threaded WebAssembly backends for Futhark
- Development of a JavaScript API for calling Futhark WebAssembly libraries from browser applications
- Benchmarking of the backends against native C implementations, showing near-native performance for sequential WebAssembly and significant speedups for threaded WebAssembly on multicore CPUs
- Demonstration of practical applications, including Mandelbrot set visualization and ray tracing, running efficiently in web browsers

## Technical Approach
The thesis leverages Emscripten, a toolchain for compiling C/C++ to WebAssembly, to generate WebAssembly code from Futhark's C backend output. The implementation involves:
1. Modifying Futhark's C code generation to be compatible with Emscripten
2. Generating JavaScript glue code and API wrappers around Emscripten's JavaScript API
3. Implementing memory management and error handling in the JavaScript API
4. Extending the backend to support threaded WebAssembly using Web Workers and SharedArrayBuffers

The JavaScript API provides a convenient interface for calling Futhark functions from browser applications, supporting both scalar and array inputs/outputs, and handling memory management automatically.

## Results
The benchmarks show that the sequential WebAssembly backend performs close to the native C backend, with a performance penalty of 13-67% depending on the benchmark. The threaded WebAssembly backend achieves significant speedups on multicore CPUs, with up to 6x improvement over the sequential backend for perfectly parallelizable tasks. The performance is bounded by the number of physical CPU cores, not logical cores, due to the underlying V8 engine implementation.

## Relevance
This work is highly relevant to language design, compilers, and ML systems as it:
- Extends high-level parallel programming to web browsers, enabling efficient execution of data-parallel algorithms in client-side applications
- Demonstrates the feasibility of using WebAssembly as a compilation target for functional programming languages
- Provides insights into the challenges and opportunities of parallel programming in web environments
- Contributes to the growing ecosystem of WebAssembly backends for high-performance languages, potentially influencing future language design and compiler implementation strategies
