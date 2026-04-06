## Phase 1e: Benchmarking and Real Models

**Goal:** Prove the GPU backend works on real models and characterize performance.

### Benchmark Suite

| Model | Surf file | Purpose | Key ops exercised |
|---|---|---|---|
| MNIST MLP | `examples/mnist.ch` | Already exists, baseline | matmul, relu, softmax, cross-entropy, grad |
| CNN (LeNet-5) | `examples/lenet.ch` | Tests conv2d on GPU | conv2d, max_pool, relu, matmul |
| Transformer block | `examples/transformer_block.ch` | Tests attention, layer norm | matmul, softmax, layer_norm, add, mul |
| Linear regression | `examples/linreg.ch` | Simplest possible, sanity check | matmul, add, sum, grad |

Each model exists as a `.ch` file in `examples/`. Each has:
- A reference PyTorch implementation in `benchmarks/pytorch/` for numerical comparison
- A training script that runs on CPU (via C backend) and GPU (via HIP backend) and reports: accuracy/loss, wall-clock time, peak memory

### Performance Targets

| Metric | Target | Rationale |
|---|---|---|
| Correctness | GPU output matches CPU within 1e-5 | Non-negotiable. If they don't match, the backend has a bug. |
| MNIST training | Completes, >90% accuracy | Same milestone as Phase 0h but on GPU |
| Wall-clock vs CPU | GPU faster than CPU for batch ≥ 32 | If GPU is slower than CPU, something is fundamentally wrong |
| Wall-clock vs PyTorch | Within 2-5x | Proving the architecture works, not winning benchmarks |
| Peak VRAM | Fits in 8GB for MNIST, 16GB for transformer block | Consumer GPU target |

### Performance is NOT the Phase 1 priority

Correctness is. If the GPU backend produces wrong results fast, that's a failure. If it produces correct results slowly, that's Phase 1 success with optimization work remaining. Do not optimize at the expense of correctness testing time.

### Profiling

Use `rocprof` (AMD) or equivalent to identify:
- Kernel launch overhead (are we launching too many small kernels? → fusion helps)
- Memory transfer overhead (are we transferring too often? → memory planning helps)
- Kernel execution time (are individual kernels slow? → thread block sizing, memory coalescing)
- hipBLAS utilization (is matmul actually using the hardware GEMM units?)

Document findings in a `benchmarks/RESULTS.md` with reproducible commands.

### Test Strategy (~10 tests)

- [ ] MNIST trains on GPU to >90% accuracy
- [ ] MNIST GPU results match CPU results (numerical comparison per batch)
- [ ] LeNet trains on GPU (if conv2d is implemented)
- [ ] Transformer block forward pass matches CPU
- [ ] Linear regression trains correctly on GPU
- [ ] GPU is faster than CPU for MNIST with batch=64 (basic sanity)
- [ ] Peak VRAM for MNIST fits in 8GB
- [ ] No numerical divergence over multiple training epochs (accumulated precision errors)

### Execution Strategy

```
Commit 1: Benchmark scaffold — examples/*.ch files, PyTorch references, timing harness
Commit 2: MNIST on GPU — train, verify accuracy, compare to CPU
Commit 3: Additional models (LeNet, transformer block, linreg)
Commit 4: Performance profiling with rocprof
Commit 5: RESULTS.md with findings and bottleneck analysis
```
