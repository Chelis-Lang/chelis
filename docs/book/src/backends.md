# Backends

Chelis lowers a typed program to a RISC DAG and then emits source for one of three
backends. The build step is a code generator: it writes source and runtime artifacts, and
you compile that source with your own platform toolchain. Chelis does not call `gcc`,
`hipcc`, or `clang++` for you. The authoritative source is `spec/08-backends.md`.

## The pipeline

```text
Surf  ->  Deep  ->  IR (RISC DAG)  ->  backend source
```

Parsing, type checking, and lowering are decoupled from backend emission. Shared IR passes
(elementwise fusion, autodiff expansion, memory planning) run once over the DAG, and each
backend is a string emitter over the result. The three emitters mirror each other in
structure and differ only in the kernel language they print.

## Targets

| Target | Hardware | Emits |
|---|---|---|
| `c` | CPU | Portable C source |
| `hip` | AMD GPU | HIP host source with embedded kernel strings |
| `metal` | Apple GPU | Objective-C++ `.mm` host files with Metal Shading Language kernel strings |

Select a target with `--target`. The C backend is the default and the reference path: every
other backend is checked for numerical agreement against it on the shared test suite.

```sh
chelis build app.ch --target c --output out/
chelis build app.ch --target hip --output out/
chelis build app.ch --target metal --output out/
```

## What build emits

For each target, `chelis build` writes the generated source plus the runtime support it
needs:

- The C backend writes portable C and ships `chelis_runtime.h` alongside a static runtime
  library. Elementwise and reduction loops emit with OpenMP parallel pragmas, and
  matmul-shaped subgraphs are pattern-matched to a BLAS fast path.
- The HIP backend writes `*_hip.cpp` host code with HIP kernel source compiled at runtime by
  `hiprtc`. Contiguous f32 matmul of rank two or higher specializes to hipBLAS, and the
  build surfaces the `-lhipblas` link flag you need.
- The Metal backend writes `.mm` host code and Metal Shading Language kernel strings, with a
  `chelis_metal_runtime.h` header. Kernels compile at runtime through
  `newLibraryWithSource`. You compile the `.mm` output with `clang++ -fobjc-arc -framework
  Metal -framework Foundation`.

All three backends share one runtime ABI: a generated function takes input and output
`chelis_tensor` arrays. Tuple-returning exports use a stable tuple ABI with `chelis_tuple`
helpers documented in the runtime header. The build reports a peak-memory formula so you can
size allocations, and the emitted source is what you hand to your compiler.

## Numerics and determinism

The C backend is the numerical oracle. Every backend must preserve numerical correctness
within documented tolerances, preserve the named-dimension and precision semantics
established before lowering, and agree with the C backend on the shared test suite.

A few platform constraints are worth knowing:

- `f64` is not available on the Metal target, because Apple Silicon GPUs have no
  double-precision ALUs. Run f64 workloads on the C or HIP target.
- `bf16` on Metal requires an Apple7 or later GPU; pipeline creation reports a clean
  diagnostic on earlier devices.
- Metal kernels heavy in `exp`, `log`, and `sqrt` may need a wider f32 tolerance than HIP
  because of Metal Shading Language fast-math semantics.

## Validating the front end

`chelis validate` runs the conformance grammar over Surf, Deep, or desugared output before
any backend work:

```sh
chelis validate --surf app.ch
chelis validate --deep app.dp
chelis validate --desugar app.ch
```

Interactive execution through `chelis eval` and the Tide tooling surface uses the IR
evaluator directly rather than a compiled backend, so you can run a program without
generating and compiling source.
