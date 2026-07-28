# Migrate the HIP lane to one-byte bool storage

## Why

`CRuntime-BoolStorage-F1` moved `CHELIS_BOOL` from a 4-byte f32 payload to one native
byte. The HIP lane went half-way on its own and the half that moved is the dangerous one.

`chelis_gpu_dtype_size` calls `chelis_runtime_dtype_size_checked`, which is generated from
the vocabulary. So `hipMalloc`, `chelis_host_to_device`, and `chelis_device_to_host` all
followed bool to one byte **automatically**. The emitter did not:

```rust
Prim::Bool => "float",                             // dtype_c_type, emit.rs:3710
Prim::F32 | Prim::Bool => kernels::ElemKind::F32,  // elem_kind,    emit.rs:3678
```

Kernels therefore write `float` — **four bytes per element into a buffer allocated at one.
A four-times device-side heap overflow.**

Nothing caught it. The 6,392-test workspace run was green: HIP's GPU tests are `#[ignore]`d
manual gates, and its codegen tests inspect emitted text without relating the element type
to the allocation width. **Metal has `dtype_abi_width_parity.rs`, which is the only reason
the analogous Metal mismatch was ever found (chelis#892). HIP has no such probe.**

This is the same failure the runtime migration was built to avoid — a width flip with
writers still on the old width — and it demonstrates that "atomic" was scoped to one crate
when it needed to be scoped to every lane.

## What changes

Seven sites, together — and the spelling becomes a **compile-time** property rather than a
convention, so this instance is fixed and the class is closed for HIP in the same change.

### The element table is width-checked at compile time

`dtype_c_type` returns a `&'static str` with nothing relating it to the dtype. Replacing
it with a macro-generated table that also emits a `const` assertion per arm makes the
mismatch unwriteable:

```rust
hip_elements! {
    F32   => (f32, "float",         RuntimeDType::F32),
    Bool  => (u8,  "unsigned char", RuntimeDType::Bool),
    Int32 => (i32, "int32_t",       RuntimeDType::I32),
    // ...
}
```

The Rust type is a **width witness** for the C type — `u8` models `unsigned char`, `f32`
models `float` — and the macro emits
`assert!(size_of::<witness>() as u32 == dtype.byte_width())` alongside each arm.

**Verified.** Planting the actual bug (`Bool => (f32, "float", RuntimeDType::Bool)`)
produces:

```
error[E0080]: evaluation panicked: HIP element type `float` for Bool
              does not have that dtype's declared width
```

That is a build failure, not a test failure. `RuntimeDType::byte_width` is already a
`const fn`, so no vocabulary change is needed. `Prim::runtime_dtype` is **not** const, so
the table names the `RuntimeDType` directly rather than deriving it.

The macro is the mechanism: because there is no hand-written arm path, a lane cannot add a
spelling and omit its assertion. Agreement stops being maintained by hand, which is what
produced this overflow.

### Why this does not need the cross-crate refactor

`bind-lane-element-types` argues the element markers must move to `chelis-vocab`, because
`Bool8` and friends live in `chelis-runtime` and no backend depends on it.

**That reasoning does not apply here, and the design doc overstated it.** The witness does
not need to be `Bool8`; it only needs a Rust type whose size models the C type's width, and
those are primitives. Nothing moves between crates and this stays local to
`chelis-backend-hip`. `bind-lane-element-types` should be re-derived on this basis — it is
likely much smaller than its design currently assumes.

### The seven sites

| site | now | becomes |
| --- | --- | --- |
| `dtype_c_type` | `Prim::Bool => "float"` | `"unsigned char"` |
| `elem_kind` | `Prim::Bool => ElemKind::F32` | not an `ElemKind`; bool routes through the typed templates |
| `emit_typed_scalar_local` | bool shares the f32 bit-pattern arm | a byte-valued local |
| `bytes_per_element` (emit) | `Bool => 4` | derived from `byte_width` |
| `bytes_per_element` (memory) | `Bool => 4` | derived from `byte_width` |
| `ElemKind::{one,zero}_lit_bool` | `"1.0f"` / `"0.0f"` | `1` / `0` |
| `dtype_bytes` (test harness) | `Bool => 4` | derived |

### Bool joins the typed-template path, not `ElemKind`

`ElemKind` has exactly two variants, `F32` and `F64` — there is no integer family. i8, i16,
i32 and i64 already avoid it by going through `dtype_c_type` and the `*_typed` kernel
builders (`binary_elementwise_typed`, `pad_typed`, `shrink_typed`).

Bool belongs on that path for the same reason they do: it is not a float. Giving it an
`ElemKind` variant would mean a third kernel family for a type that needs no arithmetic.

The `one_lit_bool` / `zero_lit_bool` helpers exist because `cmplt` writes boolean results
into a float-typed buffer. Once the output is a byte, they emit `1` / `0` — the same change
the C backend's `cmplt` ternary needed, and one that no pointer-typing would have forced.

## Impact

- **Affected specs:** new capability `hip-bool-storage`.
- **Affected code:** `chelis-backend-hip` emitter, memory planner, kernel constants, test
  harness; a new parity probe.
- **Not affected:** the HIP runtime header. It already derives from the vocabulary, which
  is why the allocation side moved on its own.
- **Stacked on:** `harden-vocabulary-kernel` (chelis#894), which is what breaks HIP. This
  change targets that branch rather than `main`, so #894 only ever reaches `main`
  complete.

## Non-goals

- **Generalising the width derivation.** Two of these tables are HIP's, and they are part
  of this fix. The other four repo-wide belong to `derive-lane-element-widths`.
- **Generalising the binding to the C and Metal lanes.** `bind-lane-element-types` does
  that, and should be re-derived against the witness pattern proven here rather than the
  cross-crate marker relocation its design currently assumes. This change closes the class
  for HIP only.
- **Probes for the other lanes.** `probe-lane-dtype-abi` generalises the pattern. HIP's
  lands here because a fix without its oracle is how this got here.
