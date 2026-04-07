## Phase 1b: Fusion

**Goal:** Adjacent RISC DAG nodes that can share a GPU kernel are fused into single kernel launches, reducing memory round-trips.

### Why Fusion Matters

Without fusion, a chain like `add(x, y) |> relu |> mul(z)` launches three separate GPU kernels. Each kernel reads from global memory and writes to global memory. The intermediate results (`add` output, `relu` output) are written and then immediately read — pure waste. Fused, the entire chain becomes one kernel that reads inputs once, computes all three operations in registers, and writes the final output once.

For a transformer block, the unfused version might launch 30+ kernels with 30+ memory round-trips. Fused, it might be 5-8 kernels. This is typically a 3-10x speedup.

### What the Agent Builds

**Fusion pass (`chelis-ir/src/fuse.rs`):**

A DAG-to-DAG rewrite that identifies fusible subgraphs and merges them into `FusedOp` nodes. The fused DAG has fewer nodes, each representing a compound kernel.

**AD ordering requirement:**

Fusion runs after `grad`, not before it. The intended pipeline is:

```text
lower -> optimize -> grad -> optimize again -> fuse -> codegen
```

This keeps adjoint rules defined only on the ordinary RISC DAG. The backward pass produced
by `grad` is then fused using the same pass as the forward DAG. The adjoint of a fused
kernel should itself be fusible; it should not require special `FusedOp`-aware gradient
rules.

**Fusibility rules:**

| Pattern | Fusible? | Reason |
|---|---|---|
| Elementwise → Elementwise | Yes | Same iteration space, no data dependency between elements |
| Elementwise → Reduction | Yes | The elementwise output feeds directly into the reduction |
| Reduction → Elementwise | Maybe | Only if the elementwise op broadcasts the reduction result back |
| Reduction → Reduction | No | Different iteration spaces |
| Any → Movement op | N/A | Movement ops are metadata-only, not kernels |
| Multi-consumer node → Fuse | Split | If a node has multiple consumers, it can be fused into one consumer's chain but the other consumer needs its own copy |

**Invariant: fusion must never duplicate computation or reduce available parallelism.** A fusion candidate that would duplicate a shared subexpression (multi-consumer node) must be penalized or rejected. The `egg` cost model should encode this directly. The greedy fallback must check: "does this fusion duplicate any node's computation?" If yes, skip. This is the difference between a fusion pass that's always safe to run and one that sometimes makes things worse.

**Fusion does NOT cross:**
- Different reduction axes (iteration spaces don't match)
- Scatter/gather boundaries
- Nodes explicitly marked as materialization points (`realize`)

**`egg` evaluation protocol:**

Before writing hand-crafted fusion heuristics:

1. Define a minimal `egg::Language` for the RISC ops (just the fusibility-relevant subset)
2. Write fusion rules as `egg` rewrite rules: `(add (relu x)) → (fused_elem [add, relu] x)`
3. Define a cost model: minimize kernel launches, tie-break by minimizing memory traffic
4. Run `egg::Runner` on the MNIST DAG and a transformer-block DAG
5. Compare the `egg` result against what a greedy "fuse all adjacent elementwise" heuristic produces

If `egg` and greedy produce the same result on both test cases → use greedy (simpler, no dependency). If `egg` finds a better fusion strategy → adopt `egg` for the fusion pass.

If the `egg` prototype is taking too long or fighting the library's API, abandon it and write greedy fusion directly. The prototype is time-boxed — do not spend more than 2 days on the `egg` evaluation.

**Fused kernel emission:**

A fused node `FusedOp { ops: [Add, ReLU, Mul], inputs: [...] }` emits a single kernel string where the loop body contains all three operations chained:

```c
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i >= size) return;\n"
"  float v = a[idx_a] + b[idx_b];\n"   // Add
"  v = fmaxf(v, 0.0f);\n"              // ReLU
"  v = v * c[idx_c];\n"                // Mul
"  out[i] = v;\n"
```

No intermediate memory allocation. All operations happen in registers.

### Test Strategy (~12 tests)

- [x] `add → relu` fusion produces identical output to unfused
- [x] `add → relu → mul` three-way fusion: identical output, one kernel launch
- [x] Reduction following elementwise: fuses correctly
- [x] Multi-consumer split: node used by two downstream ops — one fuses, other gets its own path
- [x] Fusion does not cross reduction axis boundaries
- [x] `realize()` prevents fusion across the boundary
- [x] MNIST model: count kernel launches with and without fusion, verify reduction
- `egg` was not adopted in the shipped 1b scope; the greedy fusion path is the implemented boundary.

### Acceptance Oracle

Phase 1b is complete when this oracle passes:

```sh
cargo test --workspace
```

Supporting manual evidence:

- rerun the Phase 1a HIP manual oracle:
  `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1`
- verify the real CLI build path still emits fused HIP kernels for MNIST:
  `cargo test -p chelis-cli build_hip_mnist_emits_fused_kernels_and_launches -- --exact`

### Execution Strategy

```
Commit 1: Fusibility analysis — identify fusible pairs in a DAG
Commit 2: egg prototype (time-boxed to 2 days) — or skip to commit 3
Commit 3: Greedy fusion pass — merge fusible chains into FusedOp nodes
Commit 4: Fused kernel emission — FusedOp → single kernel string
Commit 5: Correctness tests — fused vs unfused, all patterns
Commit 6: MNIST fusion — count kernel reduction, verify correctness
```
