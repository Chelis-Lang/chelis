# GPU backends and native build validation

The user book documents C, the build target of the public release. This page
covers the prerelease HIP and Metal targets and how native builds are
validated. CPU is the primary delivery and acceptance lane.

## Building with a GPU target

Both emitters are included in a normal CLI build; no `hip` or `metal` Cargo
feature is required. With the [contributor setup](contributor_setup.md) in
place:

```sh
cargo build -p chelis-cli
./target/debug/chelis build app.ch --target hip --output out/hip/
./target/debug/chelis build app.ch --target metal --output out/metal/
```

HIP native builds require a ROCm/HIP toolchain with `hipcc` and `hiprtc`, and
matching official hipBLAS 3+ headers and libraries. Metal native builds
require macOS and Apple's compiler and framework toolchain. Executing device
kernels requires a compatible AMD or Apple GPU. `build` compiles but does not
run the program, and does not need a GPU at build time. `--emit-c` applies to
HIP C++ and Metal Objective-C++ too, so sources can be generated on a host
without the target SDK. `chelis eval --target hip` and `--target metal`
select target capabilities for host evaluation; they do not execute GPU
kernels.

| Target | Main generated source | Native toolchain |
|---|---|---|
| `c` | `<stem>.c` | A C compiler on the host CPU |
| `hip` | `<stem>_hip.cpp` with embedded HIP kernel source | `hipcc` and an AMD GPU |
| `metal` | `<stem>_metal.mm` with embedded Metal Shading Language source | `clang++` with Apple's Metal frameworks and an Apple GPU |

GPU outputs add target support headers such as `chelis_hip_runtime.h` or
`chelis_metal_runtime.h`, and the native artifact takes the generated source
stem: `app_hip` or `app_metal` for an executable, `libapp_hip.a` or
`libapp_metal.a` for a static library. HIP honors `CHELIS_HIPCC` and
otherwise uses `hipcc`; Metal honors `CHELIS_METAL_CXX` and otherwise uses
`clang++`. Unlike the C compiler, `hipcc` keeps its environment. Eligible HIP
matrix multiplication adds hipBLAS; Metal matrix multiplication uses
MetalPerformanceShaders. HIP kernels compile at run time through `hiprtc`,
Metal kernels through `newLibraryWithSource`. HIP and Metal tensor-graph
builds print a peak device memory formula, with a concrete estimate when
sizes are known.

## Target limits

- Metal rejects `f64` before kernel emission.
- Metal `bf16` kernels require an Apple7 GPU family device.
- HIP support for `bf16` and `f16` depends on the operation; a target limit
  produces a diagnostic rather than a changed calculation.
- HIP and Metal reject the transcendentals and `sqrt`, and anything built
  from them such as `sigmoid` or `softmax`, at build time: the device lanes
  have no correctly rounded kernels for them.
- Metal f32 division and reciprocal can differ from the CPU lanes under the
  device compiler's fast math
  ([#2968](https://github.com/Chelis-Lang/chelis/issues/2968)).
- Host wrappers containing C tensor helpers can fail HIP and Metal C++
  compilation at the restrict qualifier gap
  ([#2595](https://github.com/Chelis-Lang/chelis/issues/2595)); the CLI
  reports the native compiler failure.

## Native build validation

The CPU acceptance oracle is `cargo test -p chelis-cli --test native_build`.
`cargo test -p chelis-cli --test parity` checks the executable example corpus
against eval using the actual native products. On a Mac with a Metal device,
run the ignored `native_build` suite as documented in the
[manual gates](manual_gates.md). The
`hip_and_metal_build_commands_include_support_and_ordered_flags` test checks
HIP compiler argv and static-library staging without GPU execution.

On a ROCm host, a first smoke check builds a scalar program and runs its host
result, without exercising a GPU kernel:

```sh
chelis build examples/native_program.ch --target hip --output target/native-hip/
./target/native-hip/native_program_hip
```

Follow it with the HIP hardware suites through `scripts/hip_test.py`, which
supplies the reconciled ROCm environment; the
[HIP runbook](local_hip_environment.md) has the details.
