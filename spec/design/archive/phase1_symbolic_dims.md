## Phase 1 Symbolic Dimensions

**Goal:** Support dimension-polymorphic compilation for the C and HIP backends.

Phase 0 required concrete dimensions at IR lowering time. The GPU backend needs runtime shape parameters — you don't recompile a kernel for each batch size. This means the IR must support symbolic dimensions that become concrete at runtime.

### What Changes

**IR (`chelis-ir`):**
- Dimension values in the DAG can remain symbolic through lowering and backend emission
- runtime shape calculations are expressed with `DimExpr` (`Concrete`, `Sym`, `Mul`, `Div`)
- allocation, indexing, and launch configuration compute sizes from symbolic dims at runtime

**Type checker (`chelis-types`):**
- Already supports named dimensions at the type level
- The change: when lowering to IR, named dims that aren't bound to concrete values stay symbolic instead of erroring

**Backends (`chelis-backend-c`, `chelis-backend-hip`):**
- generated functions keep the stable tensor ABI and bind symbolic names from input tensor metadata at runtime
- repeated occurrences of the same symbolic dim are validated across all inputs before execution
- the same compiled artifact works for batch=32 or batch=128 without recompilation on the supported Phase 1 surface

**This is NOT full dependent types or symbolic arithmetic.** The dimensions are runtime integers discovered from input metadata, not type-level expressions. `tensor[batch, hidden, f32]` means "a tensor whose first dimension is called batch and has some runtime size."

### Test Strategy (~6 tests)

- [ ] A function compiled with symbolic `batch` dim works with batch=32 and batch=128
- [ ] Kernel launch grid size is computed from runtime dim values
- [ ] Matmul with symbolic dims: `[batch, in] × [in, out]` → correct shape
- [ ] Reduction with symbolic axis size: `sum([batch, seq], axis=1)` works for varying seq
- [ ] `softmax([batch, seq], axis=1)` preserves the symbolic axis size through lowering/codegen
- [ ] CPU and GPU backends agree with symbolic dims

### Current shipped boundary

- implemented: symbolic dims on the stable tensor ABI, repeated symbolic-occurrence validation, symbolic `sum`/`max_reduce`/`softmax`, symbolic matmul/expand/reshape/permute paths, HIP memory formulas
- checker-admitted but not yet implemented on every backend: symbolic normalized-axis execution for `mean`/`layer_norm`
- not yet implemented: HIP `pad`/`shrink`

### Execution Strategy

```
Commit 1: IR dimension representation (Concrete | Symbolic)
Commit 2: Kernel emission with symbolic parameters
Commit 3: Host code emission with runtime dim binding
Commit 4: CPU backend symbolic dims (same change, simpler)
Commit 5: Tests
```

### Note on Sequencing

Symbolic dimensions could be tackled before or after the GPU backend. Arguments for before: it simplifies the GPU backend design from the start (no hardcoded sizes in kernels). Arguments for after: it's an IR change that touches the CPU backend too and could introduce regressions. **Recommendation: implement symbolic dims first, then build the GPU backend on top.** This avoids retrofitting symbolic dims into an already-working GPU backend.
