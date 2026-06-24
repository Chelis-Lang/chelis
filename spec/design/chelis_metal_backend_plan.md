# Metal Backend Implementation Plan

**Goal:** Add `--target metal` to the Chelis compiler, enabling GPU-accelerated tensor operations on Apple Silicon via Metal Shading Language (MSL). The Metal backend is the macOS-native GPU peer of the existing HIP (AMD) backend.

**Prerequisite:** macOS ARM64 CPU support shipped (toolchain resolver, Accelerate BLAS, dual-tarball release). This plan assumes that work is complete.

---

## 1. Architecture Overview

The Metal backend occupies the same slot in the compiler pipeline as HIP:

```
Typed AST
    ↓ lower to RISC tensor IR (shared, unchanged)
Tensor DAG + Host IR
    ↓ fusion pass (shared, unchanged)
Fused DAG
    ↓ emit (backend-specific — this is the new work)
  ┌─────────────────────────────────────────┐
  │  --target c     → C + OpenMP            │  existing
  │  --target hip   → C host + HIP kernels  │  existing
  │  --target metal → C host + MSL kernels  │  NEW
  └─────────────────────────────────────────┘
    ↓ compile
  Binary
```

Everything upstream of the emit step is shared across all backends. The Metal backend implements:

1. **MSL kernel emission** — walk the fused tensor DAG and produce MSL kernel source strings
2. **Metal runtime** — device/queue/buffer/pipeline management in Rust, exposed via C ABI
3. **Host-side dispatch code** — C code that calls the Metal runtime to allocate buffers, dispatch kernels, and read back results
4. **CLI integration** — `--target metal` flag routing

---

## 2. Reference: How the HIP Backend Works

Before building Metal, understand the HIP backend's structure. The Metal backend mirrors it.

Locate these files in the repo (exact paths may vary — `grep` to confirm):

```bash
# Find the HIP backend entry points
grep -rn "hip\|HIP\|Hip" crates/chelis-backend-c/src/ --include="*.rs" | head -30
grep -rn "hip\|HIP" crates/chelis-cli/src/main.rs | head -10
grep -rn "hip\|HIP" crates/chelis-ir/src/ --include="*.rs" | head -10
```

The HIP backend likely does these things:

1. **Kernel emission:** For each fused DAG node (or group), emit a `__global__` HIP C function. Elementwise ops become per-element kernels indexed by `blockIdx.x * blockDim.x + threadIdx.x`. Reductions use shared memory.

2. **Host dispatch:** Emit C code that calls `hipMalloc`, `hipMemcpy` (or `hipMemcpyHostToDevice`), launches the kernel via `hipLaunchKernelGGL`, and calls `hipMemcpy` back (or `hipDeviceSynchronize`).

3. **Compilation:** The emitted code is compiled with `hipcc` (AMD's compiler wrapper around clang).

The Metal backend replaces each of these with Metal equivalents, but the DAG walking logic, the fusion decisions, and the kernel structure are the same.

---

## 3. Implementation Components

### 3.1 New Crate: `crates/chelis-backend-metal/`

Create a new crate. It only compiles on macOS:

```toml
# crates/chelis-backend-metal/Cargo.toml
[package]
name = "chelis-backend-metal"
version = "0.1.0"
edition = "2021"

[dependencies]
chelis-ir = { path = "../chelis-ir" }

[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.29"
objc = "0.2"
```

On non-macOS platforms, the crate compiles as an empty module with a stub `emit_metal()` that returns an error. The CLI's `--target metal` flag is only accepted on macOS.

### 3.2 MSL Kernel Emission (`emit.rs`)

The core of the backend. Walk the fused tensor DAG and produce MSL kernel source strings.

**Elementwise operations (add, mul, sub, div, neg, exp, log, sqrt, sin, cast, clamp, where):**

These are the simplest and the most numerous. Each fused elementwise kernel has this shape:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void fused_elementwise_0(
    device const float* input_0 [[buffer(0)]],
    device const float* input_1 [[buffer(1)]],
    device float* output_0 [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float v0 = input_0[tid];
    float v1 = input_1[tid];
    // Fused body — emitted per-op from the DAG:
    float t0 = v0 + v1;           // add
    float t1 = exp(t0);           // exp
    output_0[tid] = t1;
}
```

The fusion pass has already determined which ops are fused into this kernel. The emitter's job is to walk the fused group and emit the per-element body. This is the same logic as the HIP emitter — only the boilerplate changes.

**MSL vs HIP C syntax mapping for elementwise kernels:**

| Concept | HIP C | MSL |
|---|---|---|
| Kernel declaration | `__global__ void name(...)` | `kernel void name(...)` |
| Thread index | `int tid = blockIdx.x * blockDim.x + threadIdx.x;` | `uint tid [[thread_position_in_grid]]` (kernel parameter) |
| Buffer parameter | `float* data` | `device float* data [[buffer(N)]]` |
| Scalar uniform | `int n` (passed by value or pointer) | `constant uint& n [[buffer(N)]]` |
| Bounds check | `if (tid >= n) return;` | `if (tid >= n) return;` (identical) |
| Math functions | `expf()`, `logf()`, `sqrtf()`, `sinf()` | `exp()`, `log()`, `sqrt()`, `sin()` (Metal uses overloaded names) |
| Header | `#include <hip/hip_runtime.h>` | `#include <metal_stdlib>` + `using namespace metal;` |

The kernel body (the actual computation) is identical. Only the declaration boilerplate differs.

**Reduction operations (sum, max, min, prod, mean, argmax, argmin):**

Reductions are the most complex kernels. The standard pattern for a parallel reduction on Metal:

```metal
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // 1. Each thread loads and accumulates its portion
    threadgroup float shared_data[256];  // threadgroup size
    float acc = 0.0f;
    for (uint i = tid; i < n; i += tg_size * /* num threadgroups */) {
        acc += input[i];
    }
    shared_data[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Tree reduction in threadgroup memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. First thread writes result
    if (lid == 0) {
        output[tgid] = shared_data[0];
    }
}
```

For large arrays, this is a two-pass reduction: first pass reduces to one value per threadgroup, second pass reduces the threadgroup results to a single scalar. The second pass is either another kernel dispatch or a CPU-side sum of a small array.

**Alternative: use SIMD group intrinsics for small reductions:**

```metal
float val = input[tid];
val = simd_sum(val);  // reduces across the SIMD group (32 threads on Apple Silicon)
if (simd_is_first()) {
    // write to threadgroup memory or output
}
```

`simd_sum`, `simd_max`, `simd_min` are Metal built-ins that do warp-level reduction without explicit shared memory. Use these for the inner loop and threadgroup memory for the outer loop.

**Start simple:** For the initial implementation, use the threadgroup-memory approach (it works for all reduction types). Optimize with SIMD intrinsics later.

**Matmul:**

Two options:

**(a) MPS (Metal Performance Shaders):** Apple ships optimized matrix multiplication via `MPSMatrixMultiplication`. This is the Metal equivalent of calling rocBLAS on HIP. The dispatch is through Objective-C/Swift API:

```rust
// In the Rust runtime:
use metal::*;

let mat_mul = MPSMatrixMultiplication::init(
    &device,
    false,  // transpose A
    false,  // transpose B
    m, n, k,
    1.0,    // alpha
    0.0,    // beta
);
mat_mul.encode_to_command_buffer(&cmd_buffer, &matrix_a, &matrix_b, &matrix_c);
```

MPS is reachable from Objective-C++ directly via `<MetalPerformanceShaders/MetalPerformanceShaders.h>`. Since the Metal backend emits `.mm` host source (not Rust) and the user links the resulting binary themselves, MPS integration is just an additional `#import` plus an `MPSMatrixMultiplication` setup block in the emitted Objective-C++ — no Rust crate involvement is required.

**(b) Custom tiled MSL kernel:** Write a tiled matmul kernel using threadgroup memory:

```metal
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint TILE = 16;
    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];

    float acc = 0.0f;
    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tiles
        uint aRow = gid.y, aCol = t * TILE + lid.x;
        uint bRow = t * TILE + lid.y, bCol = gid.x;
        tileA[lid.y][lid.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        tileB[lid.y][lid.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += tileA[lid.y][i] * tileB[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid.y < M && gid.x < N) {
        C[gid.y * N + gid.x] = acc;
    }
}
```

**Recommendation:** Start with option (b) (custom tiled kernel). It is self-contained, has no extra framework dependency beyond Metal itself, and is adequate for the matrix sizes Chelis targets. Switch to MPS later if profiling shows the tiled kernel is the bottleneck on large matrices.

**Gather / Scatter / Where:**

These are index-based operations. Each is a simple per-element kernel:

```metal
// gather: out[i] = input[indices[i]]
kernel void gather_1d(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    output[tid] = input[indices[tid]];
}

// scatter_add: output[indices[i]] += input[i]  (atomic)
kernel void scatter_add(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device atomic_float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    atomic_fetch_add_explicit(&output[indices[tid]], input[tid], memory_order_relaxed);
}

// where: out[i] = cond[i] ? a[i] : b[i]
kernel void where_select(
    device const bool* cond [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    output[tid] = cond[tid] ? a[tid] : b[tid];
}
```

These are straightforward. scatter with atomic add requires `atomic_float` (available in Metal 3.0 / Apple Silicon).

**Sort / Argsort:**

Parallel sorting on GPU is a real algorithm (bitonic sort, radix sort). For the initial implementation, skip GPU sort — leave sort/argsort on the CPU path. The host-lane already handles sort via the C backend. This is the same approach many GPU frameworks take (PyTorch's CUDA sort is a relatively recent addition).

If GPU sort is needed later, Metal has `MPSSortDescriptor` or you can implement bitonic sort in MSL.

**Einsum:**

Einsum lowers to matmul + reshape + permute in the RISC DAG. The Metal backend doesn't need a separate einsum kernel — it handles the lowered matmul and elementwise operations.

### 3.3 Metal Runtime (`chelis_metal_runtime.h`)

**Architectural decision (2026-04-29).** This section was originally specified as a Rust-side runtime in `crates/chelis-runtime/` using the `metal-rs` crate, exposed to generated C via `#[no_mangle] pub extern "C"` functions. After comparing against how the HIP backend actually works, that approach was rejected and replaced with **string-emission-only** to mirror HIP exactly. The HIP backend has no Rust intermediary: it emits `.cpp` host source with embedded HIP kernel strings, and `hiprtc` performs runtime compilation. The user runs `hipcc` themselves; no `hip-rs` Rust crate sits between the generated code and the GPU runtime.

The Metal backend now follows the same shape: emit `.mm` host source with embedded MSL kernel strings; `[MTLDevice newLibraryWithSource:options:error:]` plays the `hiprtc` role at runtime; the user runs `clang++ -framework Metal -framework Foundation` themselves. This keeps `chelis-runtime` platform-portable (no `cfg(target_os = "macos")` branches), avoids `metal-rs` ABI churn tracking Apple SDK changes, and gives both GPU backends the same architecture so contributors only have one model to learn.

**Runtime location:** `crates/chelis-backend-metal/runtime/chelis_metal_runtime.h` — peer of `chelis_hip_runtime.h`. Pure Objective-C++ header, `static inline` helpers, no Rust. Emitted host `.mm` files `#import` it.

**Core helpers exposed to generated host code** (declarations; full bodies in the runtime header):

```objc
// Singleton device + queue, lazily initialized on first use.
id<MTLDevice>       chelis_metal_device(void);
id<MTLCommandQueue> chelis_metal_queue(void);

// Pipeline-state cache keyed by (library-source-hash, function-name).
// First call compiles MSL via [device newLibraryWithSource:options:error:]
// (the hiprtc analog) and caches the resulting MTLComputePipelineState.
id<MTLComputePipelineState>
chelis_metal_get_pipeline(NSString *src, NSString *fn);

// Buffer alloc over MTLResourceStorageModeShared (unified memory; CPU
// and GPU share the same physical RAM on Apple Silicon). Buffers are
// owned by ARC at the call site — when the local `id<MTLBuffer>` strong
// reference goes out of scope the buffer is released automatically.
// There is no `chelis_metal_free`: an ARC-owned buffer is freed by ARC,
// and a no-op wrapper would mislead contributors. If explicit lifetime
// control is needed later (e.g., for slot reuse), bridge across ARC at
// the buffer-handle boundary rather than resurrecting a free function.
id<MTLBuffer> chelis_metal_alloc(size_t bytes);

// memcpy over [buf contents]; no DMA on Apple Silicon. Exists for API
// symmetry with HIP, where these are real host<->device transfers.
void chelis_metal_host_to_device(id<MTLBuffer> dst, const void *src, size_t bytes);
void chelis_metal_device_to_host(void *dst, id<MTLBuffer> src, size_t bytes);

// Encode and dispatch one kernel. Buffers bound at indices 0..n_buffers-1;
// scalar uniforms packed contiguously at index n_buffers via setBytes:.
void chelis_metal_launch(
    id<MTLComputePipelineState> pso,
    NSUInteger grid_x, NSUInteger tg_x,
    id<MTLBuffer> *buffers, NSUInteger n_buffers,
    const void *uniforms, NSUInteger uniforms_bytes);

// Tensor wrapper analogous to chelis_gpu_tensor in chelis_hip_runtime.h.
typedef struct {
    id<MTLBuffer> data;
    int shape[8];
    int strides[8];
    int ndim;
    int dtype;
    int size;
    int storage_size;
} chelis_metal_tensor;
```

**Generated host code calls these helpers directly.** A typical emitted dispatch site looks like:

```objc
// Compile-once / cache-by-name pipeline lookup
id<MTLComputePipelineState> pso_fused_0 =
    chelis_metal_get_pipeline(@(kernel_fused_0_src), @"fused_0");

uint32_t n_val = (uint32_t)n;
id<MTLBuffer> buffers[3] = { in_a->data, in_b->data, out->data };
chelis_metal_launch(pso_fused_0,
                    /*grid*/ n, /*tg*/ MIN(n, 256u),
                    buffers, 3,
                    &n_val, sizeof(n_val));
```

**ARC discipline.** The runtime header and emitted `.mm` files are compiled with `-fobjc-arc`. Every `id<...>` is owned by ARC; emitted code must contain no manual `[obj retain]` / `[obj release]`. Helper functions return autoreleased objects safely.

**Out-of-bounds detection (deferred).** The HIP backend uses a `chelis_gpu_failure` device symbol checked after every launch in debug builds. The Metal-side analog would be a length-4 `MTLBuffer` with `StorageModeShared`, written via `device atomic_int*` and read by the host after `waitUntilCompleted`. **Not implemented in the M-phase first cut** — current kernels rely on the dispatched grid + `if (tid >= n) return;` guards to stay in bounds, and the M6 oracle catches obvious miscompiles by comparing against the evaluator. A future phase can add `chelis_metal_failure_flag` / `chelis_metal_check_failure` if a specific class of regression motivates it.

**Key simplification from unified memory.** On Apple Silicon, `MTLResourceStorageModeShared` means the buffer is accessible from both CPU and GPU without explicit copies. `chelis_metal_host_to_device` and `chelis_metal_device_to_host` are just `memcpy` — they exist for API symmetry with HIP (which needs real device↔host copies) but on Apple Silicon the data never actually moves. The "upload" and "download" overhead is essentially zero.

**One consequence to document for users:** the `peak_device_bytes_formula` reported by `chelis build --target metal` is also peak **system RAM** consumed for tensor storage on Apple Silicon — there is no separate VRAM. Users comparing memory budgets across HIP and Metal builds are comparing different physical resources.

### 3.4 Host-Side C Dispatch Code

The emitter produces C code for the host program that calls the Metal runtime. For each Metal kernel, the host code:

1. Allocates Metal buffers for inputs/outputs
2. Uploads input data (memcpy on unified memory)
3. Compiles the kernel (first call only — cached thereafter)
4. Dispatches the kernel
5. Downloads results (memcpy on unified memory)

Example generated `.mm` (Objective-C++) for `add(a, b) -> c` on Metal:

```objc
// Generated by chelis --target metal
#import "chelis_metal_runtime.h"
#include "chelis_runtime.h"

// MSL kernel source (embedded as a C++11 raw string literal)
static NSString *const kernel_fused_0_src = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void fused_0(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    out[tid] = a[tid] + b[tid];
}
)MSL";

extern "C" void compute_add(chelis_tensor **inputs, int n_in,
                            chelis_tensor **outputs, int n_out) {
    int n = inputs[0]->shape[0];

    // Compile-once / cache pipeline state by (source, name)
    id<MTLComputePipelineState> pso =
        chelis_metal_get_pipeline(kernel_fused_0_src, @"fused_0");

    // Allocate Metal buffers (zero-copy on Apple Silicon)
    id<MTLBuffer> buf_a   = chelis_metal_alloc((size_t)n * sizeof(float));
    id<MTLBuffer> buf_b   = chelis_metal_alloc((size_t)n * sizeof(float));
    id<MTLBuffer> buf_out = chelis_metal_alloc((size_t)n * sizeof(float));

    chelis_metal_host_to_device(buf_a, inputs[0]->data, (size_t)n * sizeof(float));
    chelis_metal_host_to_device(buf_b, inputs[1]->data, (size_t)n * sizeof(float));

    uint32_t n_val = (uint32_t)n;
    id<MTLBuffer> buffers[3] = { buf_a, buf_b, buf_out };
    chelis_metal_launch(pso, /*grid*/ (NSUInteger)n, /*tg*/ MIN((NSUInteger)n, 256u),
                        buffers, 3, &n_val, sizeof(n_val));

    int out_shape[1] = { n };
    outputs[0] = chelis_alloc(1, out_shape, CHELIS_F32);
    chelis_metal_device_to_host(outputs[0]->data, buf_out, (size_t)n * sizeof(float));

    // No explicit free: ARC owns buf_a / buf_b / buf_out and releases
    // them when the function returns and the strong refs go out of scope.
}
```

**Optimization: avoid alloc/free per kernel call.** The generated code should allocate buffers once for the program's lifetime and reuse them. The DAG already knows the buffer sizes — emit the allocations at program startup and frees at program exit. This mirrors what the HIP backend already does via the slot-reuse memory plan.

### 3.5 CLI Integration

In `crates/chelis-cli/src/main.rs`, add `"metal"` to the existing `target: String` match (alongside `"c"` and `"hip"`). The dispatch path mirrors `cmd_build_hip`:

```rust
"metal" => {
    reject_unsupported_metal_ops(&dag)?;
    reject_unsupported_metal_precisions(&dag)?;
    let dag = dead_code_eliminate(dag);
    let dag = fuse(dag);
    let result = chelis_backend_metal::codegen_metal(&dag, &func_name);
    cmd_build_metal(result, &func_name, output, &symbolic_dims)
}
```

`cmd_build_metal` writes `<func_name>_metal.mm` plus the header, copies `chelis_metal_runtime.h` from `chelis_backend_metal::runtime_dir()` into the output, prints the symbolic dim list, the unified-memory-aware `peak_device_bytes` formula, and the recommended `clang++` recipe.

**No `cfg(target_os = "macos")` gate on the CLI side.** The Metal backend crate is pure Rust string emission — it builds and runs identically on Linux, macOS, and Windows. The platform-specific work happens later when the user invokes `clang++` against the emitted `.mm`. This keeps `--target metal` available as a cross-compilation target (you can produce Metal source on a Linux build server for later compilation on a Mac).

The compile command printed by `cmd_build_metal`:

```sh
clang++ -std=c++17 -fobjc-arc -O2 <func>_metal.mm \
  -L<runtime_dir> -lchelis_runtime \
  -framework Metal -framework Foundation \
  -o <func>
```

### 3.6 Workspace Build Invariants

Unlike the original draft of this section, the Metal backend uses **no `cfg(target_os)` gates and no Apple-SDK Rust dependencies**. This is enforced as a build invariant:

- `crates/chelis-backend-metal/Cargo.toml` declares only `chelis-ir` and `chelis-types` (same as `chelis-backend-hip`).
- The crate compiles, tests, and lints clean on Linux, macOS, and Windows.
- An M1 oracle step runs `cargo tree -p chelis-cli --prefix none --no-dedupe` and fails closed if any of `metal`, `objc`, `objc-foundation`, `objc_id`, `objc_exception`, `cocoa`, `core-graphics`, `core-foundation`, or `block` appears in the dep graph. (The published crate name for `metal-rs` is `metal`; the alternation also catches its common transitive deps so that a sneaky addition produces an informative failure rather than a silent dep bloat.)
- `chelis-runtime` stays platform-portable as it is today; no new `metal_runtime.rs` lives there.
- The Metal Objective-C++ runtime header lives at `crates/chelis-backend-metal/runtime/chelis_metal_runtime.h` — only the user's `clang++` ever consumes it, never `cargo`.

---

## 4. RISC Op Coverage Matrix

Which ops get Metal kernels and which fall back to CPU:

| RISC Op | Metal kernel | Notes |
|---|---|---|
| add, sub, mul, div | Yes (elementwise) | Fused with adjacent elementwise ops |
| neg, exp, log, sqrt, sin | Yes (elementwise) | MSL has all of these as built-ins |
| cast | Yes (elementwise) | Type conversion kernel |
| clamp | Yes (elementwise) | `clamp()` is an MSL built-in |
| where | Yes (elementwise) | Ternary select kernel |
| sum, max, min, prod | Yes (reduction) | Parallel reduction with threadgroup memory |
| mean | Yes (reduction + elementwise) | sum then divide |
| argmax, argmin | Yes (reduction) | Returns index, needs careful handling |
| matmul | Yes (tiled kernel) | Custom tiled MSL or MPS |
| gather | Yes (index kernel) | Simple index lookup |
| scatter | Yes (atomic kernel) | Atomic add/replace |
| concat | Yes (copy kernel) | Copy slices to offsets |
| split | Yes (copy kernel) | Copy from offsets |
| reshape | No kernel needed | Metadata change only (pointer reinterpretation) |
| permute | Yes (strided copy) | Reindex with stride computation |
| expand | Yes (broadcast kernel) | Read with modular indexing |
| pad | Yes (conditional copy) | Bounds-check + fill |
| cumsum, cumprod | CPU fallback | Prefix scan is complex on GPU — defer |
| sort, argsort | CPU fallback | Parallel sort is complex — defer |
| diagonal, trace | CPU fallback | Small data, not worth GPU dispatch |
| einsum | Lowered to matmul + reshape + permute | No dedicated kernel |
| uniform_like | Yes (RNG kernel) | Philox or SplitMix counter-based RNG on GPU |

**Initial implementation:** Cover the "Yes" rows. Fall back to CPU for the "CPU fallback" rows. The fallback path is: download tensor from GPU → run the CPU implementation → upload result back to GPU. This is correct but slow for cumsum/sort-heavy workloads. Optimize later.

---

## 5. Kernel Compilation and Caching

Metal compiles MSL source strings to GPU binaries at runtime (unlike CUDA/HIP which can pre-compile). The compilation is fast (~1-10ms per kernel) but should happen once per kernel, not per dispatch.

The runtime maintains a `HashMap<String, ComputePipelineState>` cache. The generated C code calls `chelis_metal_compile_kernel("fused_0", source)` on the first use of each kernel. Subsequent dispatches hit the cache.

Alternatively, compile all kernels at program startup in a single batch. The generated C includes a `chelis_metal_init()` call that compiles all kernel strings. This moves the compilation cost to program start rather than first-use latency.

---

## 6. Apple Silicon Unified Memory

This is the single biggest simplification vs HIP/CUDA. On Apple Silicon:

- CPU and GPU share the same physical RAM
- `MTLResourceOptions::StorageModeShared` buffers are accessible from both CPU and GPU without copies
- `buffer.contents()` returns a pointer the CPU can read/write directly
- GPU reads from the same pointer after the CPU writes to it (no explicit sync for `StorageModeShared`)
- After a GPU kernel completes (`waitUntilCompleted()`), the CPU can read the results directly from the same pointer

This means:
- `chelis_metal_host_to_device()` = `memcpy` (no DMA transfer)
- `chelis_metal_device_to_host()` = `memcpy` (no DMA transfer)
- No `hipMemcpyHostToDevice` / `hipMemcpyDeviceToHost` equivalent needed
- Buffer management is simpler — allocate shared buffers and pass them around

The only synchronization point is `waitUntilCompleted()` after kernel dispatch, which ensures the GPU has finished writing before the CPU reads.

**Implication for `peak_device_bytes` reporting:** because there is no separate VRAM, the formula reported by `chelis build --target metal` is also peak system RAM consumed for tensor storage. The CLI prefixes the formula with a one-line note so users do not double-count against host-side allocations.

---

## 7. File Manifest

**New files:**

```
crates/chelis-backend-metal/
    Cargo.toml                  -- deps: chelis-ir, chelis-types. ZERO macOS-only deps.
    src/
        lib.rs                  -- public API: codegen_metal(dag, func_name) -> MetalCodegenResult
        emit.rs                 -- MetalEmitter, mirrors HipEmitter::emit_dag
        kernels.rs              -- MSL kernel templates (elementwise, reduction, matmul, fill)
        launch.rs               -- threadgroup/grid sizing (analog of HIP's launch.rs)
        memory.rs               -- slot reuse plan (initial copy of HIP's; hoist later)
        blas.rs                 -- tiled matmul detection + emission
    runtime/
        chelis_metal_runtime.h  -- Objective-C++ helpers, peer of chelis_hip_runtime.h
    tests/
        codegen_structure.rs    -- default-gate: structural assertions on emitted strings
        codegen_adversarial.rs  -- default-gate: edge cases
        gpu_correctness.rs      -- #[ignore] manual gate: real device dispatch on M-series Mac

.github/scripts/
    smoke_macos_metal.py        -- macOS CI compile-and-link smoke (Python; Never-shell rule)
```

**Modified files:**

```
Cargo.toml (workspace)              -- add crates/chelis-backend-metal to members + workspace deps
crates/chelis-cli/Cargo.toml        -- add chelis-backend-metal dependency (NOT cfg-gated)
crates/chelis-cli/src/main.rs       -- add --target metal arm + cmd_build_metal +
                                       reject_unsupported_metal_ops +
                                       reject_unsupported_metal_precisions +
                                       extend copy_runtime_artifacts
crates/chelis-effects/src/lib.rs    -- validate_build_target arm for "metal"

spec/08-backends.md                 -- new "Phase M: Metal Backend" top-level section
spec/12-roadmap.md                  -- add Phase M row
docs/manual_gates.md                -- register M6 manual oracle
docs/phase_oracles.md               -- register M1-M7 phase oracles
.github/workflows/ci.yml            -- append metal compile/link step to macos-smoke job
```

**Notably NOT modified** (deliberate, per architectural decision in §3.3):

```
crates/chelis-runtime/Cargo.toml    -- no metal-rs dep
crates/chelis-runtime/src/lib.rs    -- no metal_runtime module
crates/chelis-runtime/build.rs      -- no Metal/Foundation framework links
                                       (the user's clang++ links those, not Cargo)
```

---

## 8. Test Plan

### Unit tests (in `crates/chelis-backend-metal/`)

- **Elementwise emission:** Emit MSL for `add(a, b)`, verify the string contains correct kernel signature, thread index, and body.
- **Fused emission:** Emit MSL for `exp(add(a, b))`, verify single fused kernel with both operations in the body.
- **Reduction emission:** Emit MSL for `sum(a, axis=0)`, verify threadgroup memory and barrier usage.
- **Matmul emission:** Emit MSL for `matmul(a, b)`, verify tiled kernel structure.

### Integration tests (in `crates/chelis-cli/tests/`)

- **Elementwise correctness:** `chelis build --target metal` on `def f(a: tensor[n, f32], b: tensor[n, f32]) -> tensor[n, f32] = add(a, b)`, compile, run, verify output matches evaluator. Per-dtype matrix coverage (f32 plus the WS-M1 admit set: f16, bf16, int8, int16, int32, int64, bool — see `spec/04-type-system.md` §1.1.3) extends the elementwise test family with the same shape per dtype.
- **Reduction correctness:** Same for `sum(a, 0)`, verify against evaluator. Per-dtype coverage applies the §5.7.1 accumulator rule (f16/bf16 reduce_sum → f32 accumulator → operand-precision result; integer reduce_sum widens per spec) for the dtypes the Metal backend admits.
- **Matmul correctness:** Same for `matmul(a, b)`. The f32 and f16 paths route through the MPS wrapper helpers (`chelis_metal_mps_gemm_f32` / `chelis_metal_mps_gemm_f16`) per the ARC ownership model in `spec/04-type-system.md` §1.1.3; bf16 routes through the parameterized 16x16 tiled MSL kernel; integer matmul is rejected at type-check per §5.7.2 and never reaches the backend. Verify f32/f16 against an MPS-reference baseline; verify bf16 against an f32 reference within bf16 tolerance.
- **Fused correctness:** Same for `exp(add(mul(a, b), c))`, verify the output is from a single fused kernel (grep generated C for kernel count) and numerically correct.
- **Fallback correctness:** Same for a program using `cumsum` (CPU fallback), verify it still produces correct results via the fallback path.
- **Evaluator agreement:** For every test, the Metal-compiled binary must produce results matching `chelis eval` within the dtype's documented tolerance (f32 ~1e-6 relative; f16 ~1e-3; bf16 ~1e-2; integer dtypes byte-identical). This is the silent-corruption gate: byte-identical agreement on integer dtypes catches the silent-downgrade class the C-backend WS-A0 footgun fix surfaced.

### CI considerations

GitHub's `macos-latest` runners are Apple Silicon with Metal support but limited GPU capability. Metal compute kernels should work on CI runners. Verify by running a small test. If CI runners don't support Metal compute (some VM environments disable GPU access), mark Metal tests as `#[ignore]` in CI and run them locally.

---

## 9. Execution Order

Phases are numbered M0–M7 to avoid colliding with the existing Phase 1a–1f HIP work, and match the active implementation plan. Each phase has **one acceptance oracle** per the repo's "One Acceptance Oracle Per Phase" rule.

```
M0: Spec sync (no code)
├── Copy upstream plan into spec/design/chelis_metal_backend_plan.md (this file)
├── Rewrite §3.3 to drop the metal-rs Rust runtime and use the
│   [MTLDevice newLibraryWithSource:] runtime-compilation path instead
├── Add "Phase M" section to spec/08-backends.md
├── Add roadmap row, manual-gate entry, phase-oracle registration
└── Oracle: grep -F "[MTLDevice newLibraryWithSource:]" finds at least one hit

M1: Scaffolding + CLI dispatch (default-gate)
├── crates/chelis-backend-metal/ skeleton (lib.rs stub, empty src/, runtime header)
├── --target metal arm in chelis-cli/src/main.rs (stub codegen + cmd_build_metal)
├── reject_unsupported_metal_ops (pad/shrink now implemented as MSL movement kernels, WS-8A; sort/argsort/cumsum/cumprod are not yet IR variants)
├── reject_unsupported_metal_precisions per the Metal column of
│   `spec/04-type-system.md` §1.1.3: admit f32, f16, bf16, int8, int16,
│   int32, int64, bool; hard-reject f64 with the FP64-ALU diagnostic
│   ("Apple Silicon GPUs lack FP64 ALUs; use `--target c` or
│   `--target hip` for f64 workloads"). bf16 admits at codegen but the
│   runtime surfaces the Apple7+ requirement on M1/M2 devices per
│   §1.1.3. Pre-WS-M1 the gate was f32+bool only; WS-M1 lifts the
│   matrix to the §1.1.3 admit set.
├── chelis-effects validate_build_target arm
└── Oracle: cargo build --workspace + cli target_metal tests +
    no Apple-SDK Rust deps via `cargo tree | grep` guard

M2: Elementwise emission + structural test surface (default-gate)
├── M2 prelude: verify MSL spec for `device const bool*` on Apple Silicon
│   (almost certainly OK on family 7+; drop the uchar workaround if so)
├── MetalEmitter mirrors HipEmitter (same passes, MSL kernel syntax)
├── kernels.rs templates: add/sub/mul/div/neg/exp/log/sqrt/sin/cast/clamp/where + fill
├── launch.rs emits chelis_metal_launch dispatch sites
├── memory.rs initial copy of HIP's plan (track hoist as follow-up)
└── Oracle: cargo test -p chelis-backend-metal --test codegen_structure

M3: Compile-and-link smoke on macOS CI (default-gate, macOS only)
├── .github/scripts/smoke_macos_metal.py (Python, not shell)
├── Drives chelis build --target metal then clang++ -framework Metal/Foundation
├── Compile-and-link only, NO execution — MTLCreateSystemDefaultDevice may
│   return null on macos-latest VMs (Apple's CI GPU policy oscillates)
├── Append step to .github/workflows/ci.yml macos-smoke job
└── Oracle: the new CI step passes on macos-latest

M4: Reductions + fused-elementwise-into-reduction (default-gate)
├── kernels.rs reduction templates (sum/max/min) — threadgroup memory + tree reduce
├── Two-pass reduction for arrays > one threadgroup
├── Inline staged-reduction scratch tracked via peak_device_bytes_static_extra
├── Reuses chelis-ir reduction_inlined_fused_elems for fused-elem-into-reduction
└── Oracle: cargo test -p chelis-backend-metal --test codegen_structure -- reduction

M5: Tiled matmul (default-gate structural)
├── blas.rs: 16x16 tiled MSL matmul (custom kernel, not MPS)
├── Specialization mirrors HIP's chelis_hipblas_sgemm_row_major detection
└── Oracle: cargo test -p chelis-backend-metal --test codegen_structure -- matmul_tiled

M6: GPU correctness oracle (manual gate, real device dispatch)
├── tests/gpu_correctness.rs with #[ignore] on every test
├── Each test: codegen → write .mm → clang++ link → run → assert evaluator agreement
├── Coverage: every M2 elementwise, every M4 reduction, M5 matmul, fused chains,
│   symbolic dims, bool round-trips, mixed-dtype cast paths
├── Expect Risk #7 (fpfast tolerance) to surface; widen Metal-specific tolerance
│   first, switch to `precise::*` per-call only where required
└── Oracle (manual):
    cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1

M7: Adversarial / red-team test surface (default-gate)
├── tests/codegen_adversarial.rs mirroring HIP's
├── Cases: zero-element tensors; rank-0 scalars; symbolic dims of 0/1;
│   bool through where; cast f32→bool→f32; very large grids (>2^16 threadgroups);
│   single-element reductions; matmul degenerate dim
└── Oracle: cargo test -p chelis-backend-metal --test codegen_adversarial
```

After each phase lands, invoke `/red-team` (which routes to `redteam-exec`) — fresh local subagent in fresh context. Main-thread validation does not satisfy the red-team requirement.

Deferred (not in M0–M7, tracked as follow-ups):
- Hoist `memory.rs` from per-backend copy into a shared crate once both GPU backends are stable
- ~~Convert `smoke_macos_accelerate.sh` to Python ("Never shell" cleanup)~~ Done; the runner is now `.github/scripts/smoke_macos_accelerate.py`.
- MPS integration for matmul, GPU sort/cumsum, RNG kernel, async dispatch (see §10)

---

## 10. What's NOT in This Plan

- **MPS integration for matmul:** WS-M1 lands MPS for f32 and f16 matmul via
  the wrapper helpers (`chelis_metal_mps_gemm_f32` / `chelis_metal_mps_gemm_f16`)
  exposed from `chelis_metal_runtime.h` under the ARC ownership model pinned in
  `spec/04-type-system.md` §1.1.3. bf16 matmul stays on the parameterized 16x16
  tiled MSL kernel; integer matmul is rejected at type-check per §5.7.2 and
  never reaches the backend.
- **GPU sort / argsort / cumsum:** Deferred. CPU fallback for now.
- **f64 (double precision) kernels:** Hard-rejected on Metal per
  `spec/04-type-system.md` §1.1.3. Apple Silicon GPUs (M1, M2, M3, M4, all
  announced successors) have **no FP64 ALUs** in their GPU compute units;
  software emulation (e.g., double-double over two f32s) is explicitly out of
  scope for this cycle and any foreseeable cycle. f64 workloads must use
  `--target c` or `--target hip`. The diagnostic surfaces at the CLI gate, the
  IR validation pass, and the codegen entry point with the same wording.
- **Multi-GPU:** Apple Silicon has one GPU. No multi-device dispatch.
- **Async dispatch:** All dispatches use `waitUntilCompleted()` (synchronous). Async command buffer pipelining is an optimization for later.
- **Integration with the Resource effect:** The type system tracks `Resource` for device placement. Wiring this to Metal buffer allocation (so `Resource(GPU)` maps to Metal buffers and `Resource(CPU)` maps to CPU memory) is a type-system integration, not a backend task. Defer to when the effect system is fully wired for device management.
