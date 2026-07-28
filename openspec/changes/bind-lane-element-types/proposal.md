# Bind a lane's element spelling to a type, not to a string

## Why

Deleting duplicated width tables ([`derive-lane-element-widths`](../derive-lane-element-widths/proposal.md))
does not reach the half that actually corrupts memory.

A backend does not only need to know that bool is one byte. It needs to emit the *name* of
a C or MSL type to read that buffer through, and that name is a `&'static str`:

```rust
// chelis-backend-hip/src/emit.rs
Prim::Bool => "float",   /* bool tensors store as float on the GPU lane */
```

Nothing relates that string to the dtype's width. When `CRuntime-BoolStorage-F1` moved
bool to one byte, `chelis_gpu_dtype_size` followed automatically — it derives from the
vocabulary — so HIP began allocating device buffers at `n * 1`. The emitted kernels kept
their `float*` spelling and wrote `n * 4`. **A four-times device-side heap overflow, from
a string that no longer matched a number.**

The C backend had the identical defect one line wide: `elem_type` mapping
`Prim::Bool => "float"` gave every op that typed a pointer through it a four-byte stride
over a one-byte buffer. Correcting that single line cleared six gradient tests at once.

This is the same shape as the eight int32 accessors — a *name* asserting an encoding the
storage does not have — but in emitted text rather than in Rust, and it is the reason a
runtime-only fix is not enough.

## What changes

Make the spelling a property of a type that already knows its width, and assert the two
agree at compile time:

```rust
trait LaneElement {
    const DTYPE: RuntimeDType;
    const C_NAME: &'static str;
}

const _: () = assert!(size_of::<Bool8>() as u32 == <Bool8 as LaneElement>::DTYPE.byte_width());
```

Emitters then ask the type rather than restating a name:

```rust
Prim::Bool => <Bool8 as LaneElement>::C_NAME,   // "unsigned char"
```

`Prim::Bool => "float"` becomes unwriteable, because the only way to obtain a spelling is
through a type whose size the vocabulary already constrains.

**The property this buys is atomicity.** Changing `Repr::Bool8`'s width without changing
`Bool8` becomes a build failure in every lane simultaneously — which is exactly the
guarantee the bool migration needed and did not have. That migration was correct in the
runtime, in the C backend, and in Metal, and silently wrong in HIP, because HIP's
agreement was maintained by hand.

## Impact

- **Affected specs:** new capability `lane-element-binding`.
- **Affected code:** `chelis-backend-c`, `chelis-backend-hip`, `chelis-backend-metal`
  element-type helpers; the element newtypes in `chelis-runtime`.
- **Depends on:** `derive-lane-element-widths` for the width half. This change is about
  names; that one is about numbers.

## Relationship to the runtime proposals

[`type-tensor-element-access`](../type-tensor-element-access/proposal.md) explicitly
excludes the backends, on the grounds that emitted C is text and no Rust type prevents
returning the wrong string.

**That exclusion was too narrow, and HIP is the evidence.** A Rust type cannot check the
*body* of an emitted kernel, but it can absolutely constrain the *element spelling* — the
thing the backend chooses from a fixed set, and the thing that was wrong. This change
takes that half; the kernel body remains uncoverable and belongs to
[`probe-lane-dtype-abi`](../probe-lane-dtype-abi/proposal.md).

## Non-goals

- **Type-checking emitted kernel bodies.** A kernel that writes `1.0f` into a correctly
  typed buffer is still wrong and still compiles. Phantom-typed `CExpr<T>` is the move
  there and is a much larger argument.
- **Unifying the three backends' emitters.** They differ for real reasons. This binds one
  narrow decision they each make, not their structure.
