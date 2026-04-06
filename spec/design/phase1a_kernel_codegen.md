## Phase 1a: Kernel Code Generation

**Goal:** A single RISC op compiles to a HIP kernel, runs on GPU, produces correct output.

### What the Agent Builds

**HIP runtime (`chelis_hip_runtime.h`):**

```c
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

typedef struct {
    float *data;          // device pointer
    int shape[CHELIS_MAX_DIM];
    int strides[CHELIS_MAX_DIM];
    int ndim;
    int dtype;
    int size;
} chelis_gpu_tensor;

chelis_gpu_tensor* chelis_gpu_alloc(int ndim, int *shape, int dtype);
void chelis_gpu_free(chelis_gpu_tensor *t);
void chelis_host_to_device(chelis_gpu_tensor *dst, chelis_tensor *src);
void chelis_device_to_host(chelis_tensor *dst, chelis_gpu_tensor *src);

// JIT compilation
hipModule_t chelis_compile_kernel(const char *source, const char *name);
void chelis_launch_kernel(hipModule_t module, const char *name,
                          dim3 grid, dim3 block, void **args);
```

Key differences from CPU runtime:
- `data` is a device pointer (hipMalloc), not host pointer (malloc)
- Explicit host↔device transfer functions
- JIT compilation via `hiprtc` — takes kernel source string, returns compiled module
- Kernel launch wrapper that handles grid/block configuration

**Kernel string emission (`kernels.rs`):**

For each RISC op type, emit a HIP C++ kernel as a string literal embedded in the generated host `*.cpp` code.

Elementwise binary (add, mul, max_elem, cmplt):
```c
const char *kernel_add_src =
    "extern \"C\" __global__ void kernel_add(\n"
    "    const float *a, int a_s0, ..., int a_s7, int a_ndim,\n"
    "    const float *b, int b_s0, ..., int b_s7, int b_ndim,\n"
    "    float *out, int out_sh0, ..., int out_sh7, int out_ndim, int size) {\n"
    "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (i >= size) return;\n"
    "  int indices[CHELIS_MAX_DIM];\n"
    "  int a_s[] = { a_s0, ..., a_s7 };\n"
    "  int b_s[] = { b_s0, ..., b_s7 };\n"
    "  int out_sh[] = { out_sh0, ..., out_sh7 };\n"
    "  chelis_flat_to_indices(i, out_sh, out_ndim, indices);\n"
    "  int idx_a = chelis_indices_to_flat(indices, a_s, a_ndim);\n"
    "  int idx_b = chelis_indices_to_flat(indices, b_s, b_ndim);\n"
    "  out[i] = a[idx_a] + b[idx_b];\n"
    "}\n";
```

Note: stride-aware indexing in the kernel, identical logic to the CPU backend. The `chelis_flat_to_indices` and `chelis_indices_to_flat` helpers are device-side functions included in every kernel string. Phase 1a passes metadata as individual integers rather than device-side arrays so kernel launches do not need extra small-buffer allocations.

Elementwise unary (neg, exp, log, sin, sqrt):
```c
// Same pattern, one input instead of two
"  out[i] = expf(a[idx_a]);\n"
```

Reductions (sum, max_reduce):
```c
// Parallel reduction pattern. Two-phase:
// Phase 1: each block reduces a chunk to a partial sum
// Phase 2: a second kernel reduces the partial sums
// This is the standard GPU reduction pattern.
```

Reductions are significantly more complex than elementwise ops. The standard approach: thread-block-level shared memory reduction, then a second pass to combine block results. For Phase 1a, implement a naive "one thread per output element, inner loop over reduction axis" approach (matches CPU backend logic but runs on GPU). Optimize in 1d.

Movement ops:
- `reshape`: metadata change only (same as CPU)
- `permute`: stride reorder (same as CPU — the kernel reads via strided indexing)
- `expand`: set stride to 0 (same as CPU — kernel handles stride-0 via indexing helpers)
- `stride`: multiply inherited strides on the host side
- These are NOT kernels — they're metadata operations on the host side

**Host code emission (`emit.rs`):**

The generated host `*.cpp` file contains:

1. `#include "chelis_hip_runtime.h"`
2. Kernel source strings as `const char*` literals
3. A `model_forward_hip()` function that:
   - Allocates GPU tensors
   - Transfers inputs from host to device
   - For each DAG node in topological order:
     - Compiles the kernel (first call) or reuses cached module
     - Configures grid/block dimensions
     - Launches the kernel
   - Transfers outputs from device to host
   - Frees GPU tensors

```c
void model_forward_hip(chelis_tensor **inputs, int n_inputs,
                       chelis_tensor **outputs, int n_outputs) {
    // Compile kernels (cached after first call)
    static hipModule_t mod_add = NULL;
    if (!mod_add) mod_add = chelis_compile_kernel(kernel_add_src, "kernel_add");

    // Allocate device buffers
    chelis_gpu_tensor *d_x = chelis_gpu_alloc(inputs[0]->ndim, inputs[0]->shape, CHELIS_F32);
    chelis_host_to_device(d_x, inputs[0]);

    // ... launch kernels in DAG order ...

    // Transfer result back
    chelis_device_to_host(outputs[0], d_result);

    // Free device buffers
    chelis_gpu_free(d_x);
    // ...
}
```

**Grid/block configuration (`launch.rs`):**

For elementwise ops: `grid = ceil(size / block_size)`, `block_size = 256` (standard default). Size is total output elements.

For reductions: more complex — depends on reduction axis size and output shape. Phase 1a uses the naive approach (one thread per output element). Phase 1d optimizes.

### Test Strategy (~15 tests)

- [ ] `add(const(1), const(2))` on GPU produces 3.0
- [ ] All elementwise unary ops: neg, exp, log, sin, sqrt — GPU matches CPU (within 1e-5 for f32)
- [ ] All elementwise binary ops: add, mul, max_elem, cmplt — GPU matches CPU
- [ ] `cmplt` produces 1.0f/0.0f on GPU (not integer bool)
- [ ] Reduction: `sum(x, axis=0)` on GPU matches CPU
- [ ] Stride-aware: `expand` then `add` — GPU handles stride-0 correctly
- [ ] Stride-aware: `permute` then elementwise — GPU handles reordered strides
- [ ] Generated C compiles with `hipcc` (or gcc + HIP runtime link)
- [ ] Kernel JIT compilation succeeds at runtime
- [ ] Host↔device transfer round-trips correctly (transfer to device, transfer back, compare)
- [ ] Multiple kernels in sequence (add then mul) — correct chaining
- [ ] Name-based Load mapping works (same contract as CPU backend)

### Acceptance Oracle

Phase 1a is complete when this manual oracle passes on a HIP-capable machine:

```sh
cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
```

Supporting evidence:

- `cargo test -p chelis-backend-hip`
- CLI build output via `chelis build examples/mnist.ch --target hip`

### Execution Strategy

```
Commit 1: HIP runtime (alloc, free, transfer, JIT compile wrapper)
Commit 2: Elementwise kernel templates + host emission for elementwise ops
Commit 3: Reduction kernel (naive) + movement op host-side metadata handling
Commit 4: Top-level codegen_hip API + CLI --target hip flag
Commit 5: Correctness tests — every op GPU vs CPU
Commit 6: Red team
```

### Verification

- `cargo test -p chelis-backend-hip` — structural tests plus HIP compile-smoke pass
- `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` — manual GPU oracle
- `chelis build examples/mnist.ch --target hip` emits `mnist_hip.cpp`, `mnist_hip.h`, `chelis_runtime.{h,c}`, and `chelis_hip_runtime.h`
- GPU add(const(1), const(2)) = 3.0
- Every supported Phase 1a RISC op path matches the evaluator within 1e-4

### Prerequisites / Environment

The coding agent needs:
- AMD GPU with ROCm installed (or NVIDIA GPU with HIP installed via ROCm)
- `hipcc` compiler available on PATH
- `hiprtc` library available for JIT compilation
- The GPU correctness tests are `#[ignore]` by default and are run via the manual oracle above so default CI stays hardware-independent
