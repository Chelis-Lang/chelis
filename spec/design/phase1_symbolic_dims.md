## Phase 1 Symbolic Dimensions

**Goal:** Support dimension-polymorphic compilation for the GPU backend.

Phase 0 required concrete dimensions at IR lowering time. The GPU backend needs runtime shape parameters — you don't recompile a kernel for each batch size. This means the IR must support symbolic dimensions that become concrete at runtime.

### What Changes

**IR (`chelis-ir`):**
- Dimension values in the DAG can be `Concrete(usize)` or `Symbolic(DimName)`
- Kernel code emission uses symbolic dimensions as function parameters: `void kernel(int batch, int seq, ...)`
- Allocation, indexing, and launch configuration compute sizes from symbolic dims at runtime

**Type checker (`chelis-types`):**
- Already supports named dimensions at the type level
- The change: when lowering to IR, named dims that aren't bound to concrete values stay symbolic instead of erroring

**C backend (`chelis-backend-c`):**
- Also benefits: generated C functions accept shape parameters instead of hardcoding sizes
- Same MNIST program works with batch=32 or batch=128 without recompilation

**This is NOT full dependent types or symbolic arithmetic.** The dimensions are runtime integers, not type-level expressions. `tensor[batch, hidden, f32]` means "a tensor whose first dimension is called batch and has some runtime size." The compiler emits `int batch` as a function parameter.

### Test Strategy (~6 tests)

- [ ] A function compiled with symbolic `batch` dim works with batch=32 and batch=128
- [ ] Kernel launch grid size is computed from runtime dim values
- [ ] Matmul with symbolic dims: `[batch, in] × [in, out]` → correct shape
- [ ] Reduction with symbolic axis size: `sum([batch, seq], axis=1)` works for varying seq
- [ ] CPU and GPU backends agree with symbolic dims

### Execution Strategy

```
Commit 1: IR dimension representation (Concrete | Symbolic)
Commit 2: Kernel emission with symbolic parameters
Commit 3: Host code emission with runtime dim binding
Commit 4: CPU backend symbolic dims (same change, simpler)
Commit 5: Tests
```

### Note on Sequencing

Symbolic dimensions could be tackled before or after the GPU backend. Arguments for before: it simplifies the GPU backend design from the start (no hardcoded sizes in kernels). Arguments for after: it's an IR change that touches the CPU backend too and could introduce regressions. **Recommendation: implement symbolic dims first (1-2 weeks), then build the GPU backend on top.** This avoids retrofitting symbolic dims into an already-working GPU backend.
