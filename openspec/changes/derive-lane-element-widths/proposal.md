# Derive every lane's element width from the vocabulary

## Why

`RuntimeDType::byte_width()` is the single source of truth for how many bytes a dtype
occupies. At least six other places restate it by hand:

| site | shape |
| --- | --- |
| `chelis-backend-hip/src/memory.rs` | `bytes_per_element(Prim) -> usize` |
| `chelis-backend-hip/src/emit.rs` | `bytes_per_element(Prim) -> usize` |
| `chelis-backend-hip/tests/bf16_f16_matmul.rs` | `dtype_bytes(Prim) -> usize` |
| `chelis-backend-c/tests/exec_compile.rs` | `c_sizeof_width(&str) -> usize` |
| `chelis-backend-metal` | `metal_elem_size`, `host_sizeof_expr` |
| generated C header | derived — the one that is correct by construction |

Each is a copy of a table that already exists, and each drifts independently.

**This is not hypothetical.** `CRuntime-BoolStorage-F1` changed bool from four bytes to
one. The generated C header followed automatically, so the HIP runtime's `hipMalloc` and
its host/device transfers all moved to one byte. The two hand-written HIP tables did not,
and neither did the emitter's C type name — so kernels kept writing `float` into a buffer
sized for one byte per element. **A four-times device-side heap overflow, introduced by a
change that was correct everywhere the width was derived.**

The int32 defect is the same lesson from the other direction. Native int32 was decoded
through an f32 view in nine places for months, and no width table caught it, because
`4 == 4`. A duplicated width table is invisible exactly when the widths agree and the
encodings do not.

## What changes

Delete the duplicated tables. Where a lane needs a width, it asks:

```rust
prim.runtime_dtype()?.byte_width()
```

You cannot drift from a table that does not exist. This is the cheapest possible form of
prevention and it covers the width half of the defect completely.

## Impact

- **Affected specs:** new capability `lane-width-derivation`.
- **Affected code:** `chelis-backend-hip` (two tables), `chelis-backend-metal`,
  `chelis-backend-c` test helpers, HIP test harness.
- **Not affected:** behaviour, when the tables currently agree. Where they disagree, the
  derived value is by definition the correct one — the disagreement is the bug.

## Non-goals

- **The C type *name*.** `Prim::Bool => "float"` is a spelling, not a width, so deleting
  width tables does not reach it. That is the live half of the HIP defect and belongs to
  [`bind-lane-element-types`](../bind-lane-element-types/proposal.md).
- **Cross-lane encoding agreement.** Equal widths do not imply interchangeable buffers;
  that is [`probe-lane-dtype-abi`](../probe-lane-dtype-abi/proposal.md).
