# Backends

`chelis build` invokes the selected native toolchain and produces an executable
for a program with observable roots, or a static library for a module of callable
definitions. C is the default target:

```sh
chelis build app.ch --output out/
./out/app
```

For a program supported by a GPU target, use `--target hip` or `--target metal`
instead. The selected target determines which operations and dtypes can be built;
a successful `chelis check` alone does not guarantee that every target admits the
program. See [Backend selection and requirements](https://github.com/Chelis-Lang/chelis/blob/main/spec/08-backends.md)
and the [dtype support matrix](https://github.com/Chelis-Lang/chelis/blob/main/spec/04-type-system.md).

## From source to artifacts

Chelis parses Surf or Deep, checks types and effects, and lowers tensor work to a
RISC graph. Programs using host values or effects also have a host execution plan.
Each target prepares that checked work for its own emitter: the CPU target emits
loops and host calls, while the GPU targets emit host code that launches device
kernels. Fusion, storage planning, and supported operations depend on the target.

| Target | Main generated source | Native toolchain |
|---|---|---|
| `c` | `<stem>.c` | A C compiler on the host CPU |
| `hip` | `<stem>_hip.cpp` with embedded HIP kernel source | `hipcc` and an AMD GPU |
| `metal` | `<stem>_metal.mm` with embedded Metal Shading Language source | `clang++` with Apple's Metal frameworks and an Apple GPU |

The output also includes a generated header, the bundled native runtime archive
`libchelis_runtime.a`, `chelis_runtime.h`, a runtime receipt, and target-specific
support files such as `chelis_hip_runtime.h` or `chelis_metal_runtime.h`.
The native artifact uses the generated source stem: `app`, `app_hip`, or
`app_metal` for an executable, and `libapp.a`, `libapp_hip.a`, or
`libapp_metal.a` for a static library. The output directory defaults to the
current directory. An explicit source filename passed to `--output` determines
the artifact's filename, while the module's generated symbols retain their
source identity. Generated sources stay in the output directory for inspection.

To emit sources without native compilation, use:

```sh
chelis build app.ch --emit-c --output out/
chelis build app.ch --target metal --emit-c --output out/
```

`--emit-c` applies to all C-family outputs, including HIP C++ and Metal
Objective-C++. It stages the runtime and prints native compile guidance; no
native compiler or archiver is required. This also permits source generation
on a host without the target SDK.

Normal builds own their required flags and support sources. Eligible HIP matrix
multiplication adds hipBLAS; Metal matrix multiplication uses
MetalPerformanceShaders. HIP kernels still compile at runtime through `hiprtc`,
and Metal kernels through `newLibraryWithSource`. A native build does not execute
the program or require a GPU to be present at build time.

C compiler selection honors `CHELIS_CC`, then available platform compilers.
HIP honors `CHELIS_HIPCC` and otherwise uses `hipcc`; Metal honors
`CHELIS_METAL_CXX` and otherwise uses `clang++`. Static libraries use
`CHELIS_AR` or `ar`. Each override is one executable name or path, without
embedded arguments. An invalid explicit override fails rather than choosing
another tool. Host compilation uses one pinned profile, `-O2 -ffp-contract=off
-fno-fast-math` with no `-march` (the instruction set is the compiler's
configured default, never the build machine's CPU), plus the target's other
required flags. Native tools run with an environment cleared down to `PATH` and
`TMPDIR`, so variables such as `CCC_OVERRIDE_OPTIONS`, `NIX_CFLAGS_COMPILE`,
`CPATH`, or `SDKROOT` cannot change a compile; `hipcc` is the exception and
keeps its environment. On macOS the build sets `SDKROOT` to the SDK of the
`xcode-select` default, so a compiler named in `CHELIS_CC`, such as another
Xcode's `clang`, finds the system headers. The C compiler is a declared input:
the build prints `Compiler: <path> (<version>)`, and refuses a compiler, for
example a wrapper script, that predefines `__FAST_MATH__`, a nonzero
`__FINITE_MATH_ONLY__`, or no `__OPTIMIZE__` under the profile.

Transcendentals (`exp`, `log`, `sin`, `cos`, `tan`, `atan`, `tanh`) are
correctly rounded, so every lane returns the same bits for them. Generated C
does not call the platform math library, Accelerate vForce, or Sleef: each unit
defines the kernels it uses as `static` functions, taken byte for byte from the
compiler's vendored CORE-MATH kernels that `chelis eval` also runs. A static
library therefore exports no extra math symbol, and `chelis_math.h` declares
nothing. Activations such as `sigmoid`, `silu`, `gelu`, and `softmax` are
graphs over these kernels and IEEE arithmetic, so they agree bit for bit too.

Static libraries contain module and support objects, with no process entry.
Consumers include the generated header, link the module archive followed by the
staged `libchelis_runtime.a`, and add the native dependencies reported by the
build. For example, an ordinary scalar C library on macOS can be used with:

```sh
clang driver.c out/libfunctions.a out/libchelis_runtime.a -lm -framework Accelerate -o driver
```

Use the build's reported dependencies for the actual target and operations.
With gcc the module's parallel loops are compiled with OpenMP, so the reported
link line includes `-fopenmp`; a link without it fails with undefined
`omp_*` and `GOMP_*` symbols.
Native tool failures fail the build and preserve the previous native artifact;
generated sources and runtime files may already have been refreshed.

For C, the generated header declares each authored export using its actual C
parameter and result types. Tensor-only entries can use the four-argument
`chelis_tensor` input/output ABI; a host export returning a scalar or tuple has a
different declaration. Read the generated header before calling an export from
C. Tuple values use the `chelis_tuple` helpers in `chelis_runtime.h`.

HIP and Metal tensor-graph builds print a peak *device* memory formula, with a
concrete estimate when sizes are known. The C build and GPU builds that emit
host wrappers do not promise that report.

## Numerical behavior and availability

See [C support and exclusions](c-support.md) for the source-derived build gates,
their rejection scopes, and the separate builtin and stdlib I/O routes.

CPU is the primary delivery and acceptance lane. HIP and Metal are prerelease
and have known imperfections; native compilation does not imply full backend
coverage. The native-build acceptance suite covers CPU execution and local
Metal cases. HIP command construction is tested, while HIP execution validation
is deferred to a host with ROCm and a compatible AMD GPU.

Host wrappers containing C tensor helpers can fail HIP/Metal C++ compilation
at the existing [restrict qualifier gap](https://github.com/Chelis-Lang/chelis/issues/2595).
The CLI reports that native compiler failure instead of reporting a successful
source-only build. Scalar host programs and the sampled Metal tensor-library
path have separate native-build coverage.

The [numeric rules](https://github.com/Chelis-Lang/chelis/blob/main/spec/04-type-system.md) define results for every
target. The C backend is a practical reference for comparing implementations;
agreement tests cover selected programs, with GPU execution checks requiring
suitable hardware. A successful build is not a claim that every operation has
been compared across targets.

- Metal rejects `f64` before kernel emission. Use `c` for an `f64` program, or
  confirm that its operations are supported on HIP.
- Metal `bf16` kernels require an Apple7 GPU family device.
- HIP support for `bf16` and `f16` depends on the operation. A target limit
  produces a diagnostic rather than silently changing the calculation.
- HIP and Metal reject the transcendentals and `sqrt`, and anything built from
  them, such as `sigmoid` or `softmax`, at build time: the device lanes have no
  correctly rounded kernels for them yet, and the build reports the operation rather than
  computing it with a vendor library. Use `c` for such a program.
- Every CPU target computes the same bits for an admitted operation; there is no
  per-operation or per-target tolerance. Metal is not yet held to this: its f32
  division and reciprocal can differ under the device compiler's fast math
  ([#2968](https://github.com/Chelis-Lang/chelis/issues/2968)).

Use `chelis eval --file app.ch` to execute locally without generating native
source. It evaluates host code and tensor operations through the compiler's
evaluation paths. For syntax validation, see the [CLI workflow](cli.md).

## Native build validation

The CPU acceptance oracle is `cargo test -p chelis-cli --test native_build`.
`cargo test -p chelis-cli --test parity` also checks the executable example corpus
against eval using the actual native products. On a Mac with a real Metal device,
run the ignored `native_build` suite as documented in [Manual gates](https://github.com/Chelis-Lang/chelis/blob/main/docs/manual_gates.md).
The selected Metal library kernel and host executable have local execution coverage;
this is not a claim that the prerelease Metal backend is complete.

HIP runtime validation for this change is deferred. On the ROCm host, a first
smoke check is to build `examples/native_program.ch` and run its host result:

```sh
chelis build examples/native_program.ch --target hip --output target/native-hip/
./target/native-hip/native_program_hip
```

That scalar program checks the native HIP toolchain path without exercising a GPU
kernel. Follow it with the existing HIP hardware suites through
`scripts/hip_test.py`, which supplies the reconciled ROCm environment. The native
`hip_and_metal_build_commands_include_support_and_ordered_flags` test checks HIP
compiler argv and static-library staging without claiming GPU execution.
