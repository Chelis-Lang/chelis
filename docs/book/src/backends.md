# Backends

`chelis build` generates native source and the files needed to compile it. It does not
run a C, HIP, or Objective-C++ compiler. Choose a target with `--target`; `c` is the
default:

```sh
chelis build app.ch --target c --output out/
```

For a program supported by a GPU target, use `--target hip` or `--target metal`
instead. The selected target determines which operations and dtypes can be built;
a successful `chelis check` alone does not guarantee that every target admits the
program. See [Backend selection and requirements](../../../spec/08-backends.md)
and the [dtype support matrix](../../../spec/04-type-system.md).

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
The build prints the files it wrote and a `Compile:`, `Compile object:`, or
`Compile objects:` command. Use that command for the generated program:
required flags and extra sources vary by target and by the operations selected.
For example, eligible HIP
matrix multiplication uses hipBLAS and adds `-lhipblas`; Metal matrix
multiplication may need `MetalPerformanceShaders`. HIP kernels compile at runtime
through `hiprtc`, and Metal kernels through `newLibraryWithSource`.

For C, the generated header declares each authored export using its actual C
parameter and result types. Tensor-only entries can use the four-argument
`chelis_tensor` input/output ABI; a host export returning a scalar or tuple has a
different declaration. Read the generated header before calling an export from
C. Tuple values use the `chelis_tuple` helpers in `chelis_runtime.h`.

HIP and Metal tensor-graph builds print a peak *device* memory formula, with a
concrete estimate when sizes are known. The C build and GPU builds that emit
host wrappers do not promise that report.

## Numerical behavior and availability

The [numeric rules](../../../spec/04-type-system.md) define results for every
target. The C backend is a practical reference for comparing implementations;
agreement tests cover selected programs, with GPU execution checks requiring
suitable hardware. A successful build is not a claim that every operation has
been compared across targets.

- Metal rejects `f64` before kernel emission. Use `c` for an `f64` program, or
  confirm that its operations are supported on HIP.
- Metal `bf16` kernels require an Apple7 GPU family device (M3 or later).
- HIP support for `bf16` and `f16` depends on the operation. A target limit
  produces a diagnostic rather than silently changing the calculation.
- Floating-point comparisons across platforms use operation-appropriate
  tolerances; Metal transcendental kernels can require wider `f32` tolerance.

Use `chelis eval --file app.ch` to execute locally without generating native
source. It evaluates host code and tensor operations through the compiler's
evaluation paths. For syntax validation, see the [CLI workflow](cli.md).
