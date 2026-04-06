# Pushing the Boulder: Addressing key limitations of the Futhark WebGPU backend

## Metadata
- **Authors:** Caleb Andreasen (Supervised by Troels Henriksen)
- **Venue/Year:** BSc Thesis, University of Copenhagen / 2025

## Summary
Futhark is a purely functional, data-parallel array programming language with an optimizing compiler that traditionally targets native GPGPU APIs like CUDA, OpenCL, and HIP. Prior work introduced an experimental WebGPU backend to enable browser-based execution, but it suffered from significant functional gaps that prevented it from compiling or correctly running non-trivial programs. This thesis systematically addresses those gaps, extending the backend's capabilities to support a broader subset of Futhark's language features and compiler transformations.

Using a test-driven development methodology, the author implements missing core functionalities including built-in transpose and copy kernels, in-kernel function calls, in-kernel memory copies, and proper handling of IEEE-754 special floating-point values (NaNs and infinities). A major focus is placed on atomic operations: the work demonstrates how to emulate 8-bit, 16-bit, and floating-point atomics using compare-and-swap loops and bit-packing, while rigorously proving that 64-bit atomics cannot be safely implemented under WebGPU's current relaxed memory model and lack of memory fences.

The resulting backend successfully compiles and executes a wide range of representative and benchmark programs in web browsers, marking a substantial step toward feature parity with native backends. However, fundamental WebGPU/WGSL restrictions—such as the inability to reassign pointers (`SetMem`), strict storage buffer limits, and missing memory fences—continue to constrain the backend's practicality for large-scale, high-performance data-parallel workloads. The thesis provides both practical compiler engineering solutions and a clear roadmap for future development.

## Key Contributions
- Implemented atomic operations for 8-bit, 16-bit, and floating-point types via compare-and-swap (CAS) loops and strategic bit-packing.
- Added WGSL implementations of Futhark's built-in transpose and copy kernels, adapting them to WebGPU's explicit resource binding model.
- Implemented in-kernel function calls (handling multi-return values via pointer arguments) and in-kernel memory copy operations.
- Added compiler support for special floating-point values (NaNs, infinities) using bitcast workarounds and custom `isinf`/`isnan` helpers.
- Proved the impossibility of emulating 64-bit atomics in WGSL due to its strictly relaxed memory ordering and absence of acquire/release semantics or memory fences.
- Resolved numerous compiler/runtime bugs (64-bit literal truncation, incorrect negation operators, multiline comment formatting, WebSocket/OOM test harness crashes).
- Updated the shared C runtime and JavaScript/WASM test infrastructure to align with WebGPU's API constraints and enable larger-scale testing.

## Technical Approach
The core technical strategy involves extending Futhark's `ImpCode`-to-WGSL code generation pipeline while working around WGSL's restrictive type system, memory model, and lack of C-like pointer semantics. Key techniques include:
- **Atomic Emulation:** Since WGSL only natively supports 32-bit integer atomics, smaller types and floats are packed into 32-bit slots. Atomic operations are implemented using `atomicCompareExchangeWeak` CAS loops that extract, modify, and repack the target bits. Global memory uses tight packing for host-GPU coherence, while shared memory uses padded 32-bit slots for simplicity.
- **WGSL Prelude & Type Emulation:** A generated prelude provides helper functions to emulate missing WGSL types (e.g., `i16` as `i32`), handle signedness mismatches in atomic operations, and generate special floats via bitcasted integer literals.
- **Code Generation Adaptations:** In-kernel copies are translated into dynamically generated nested loops that compute linearized addresses using offsets and strides. Multi-return function calls are lowered to pass destination pointers as arguments, matching WGSL's single-return constraint.
- **Memory Model Analysis:** The author formally analyzes WGSL's relaxed memory model to demonstrate that spinlocks and 64-bit atomics require acquire/release ordering or memory fences, neither of which are exposed in WGSL, making correct emulation theoretically impossible under the current spec.

## Results
- **Testing:** The backend passes the majority of test cases across primitive, SOACs, histogram, intragroup, and noinline suites. Remaining failures stem from unimplemented features (64-bit division, `SetMem`, buffer type reuse, JS BigInt conversion errors) rather than regressions.
- **Benchmarking:** On an NVIDIA 3050ti, WebGPU generally runs ~4x slower than CUDA across standard benchmarks (e.g., `kmeans`, `mandelbrot`, `hashcat`). However, it surprisingly outperforms CUDA on large datasets for specific workloads (`fluid`, `LocVolCalib`, and 8-byte transpose kernels), though the exact cause remains uninvestigated.
- **Overhead & Limits:** Small kernels exhibit high launch overhead. Larger programs frequently hit WebGPU API limits, specifically `maxStorageBuffersPerShaderStage` (capped at 10 in Chromium/Dawn) and `maxComputeWorkgroupsPerDimension`, causing runtime crashes. 8/16-bit emulation also incurs significant padding overhead in shared memory.
- **Conclusion:** The backend is now robust enough for non-trivial browser-based execution but requires further architectural changes (e.g., buffer coalescing, `SetMem` emulation) to match native backend performance and scalability.

## Relevance
This work is highly relevant to **language design and compiler engineering** as a practical case study in targeting restrictive, portable shading languages (WGSL) from a high-level data-parallel IR. It demonstrates effective compiler-level workarounds for missing hardware/language features (atomics, special floats, multi-return functions) and highlights the friction between traditional GPGPU programming models and modern web graphics APIs. For **ML systems and web deployment**, the thesis provides critical insights into the performance trade-offs, API limits (buffer counts, workgroup sizes), and memory model constraints that developers face when porting compute-heavy workloads to WebGPU. Finally, the rigorous analysis of WGSL's relaxed memory model offers valuable guidance for anyone designing concurrent algorithms or synchronization primitives for browser-based GPU computing.
