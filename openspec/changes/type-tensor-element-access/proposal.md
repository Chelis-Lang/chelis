# Give tensor handles an element type

## Why

`Bool8` is a newtype on the **element**. Every defect in this class has lived in the
**container**.

A tensor is `*mut chelis_tensor` regardless of what it stores. That one type is handed
to every accessor, so nothing in the signature of

```rust
unsafe fn data_as_f32(tensor: *mut chelis_tensor) -> *mut f32
```

distinguishes a legal call from one that reads a bool buffer at a four-byte stride. The
dtype is carried in a runtime field, checked — when checked at all — by a
`debug_assert`.

`TensorElement::data_ptr` already does the right check and returns
`Err(DtypeMismatch)`. All eight int32 decode arms simply did not call it. **A check the
caller may decline is not a guarantee**, and every mechanism this runtime has tried so
far has been declinable:

| mechanism | how it was bypassed |
| --- | --- |
| `TensorElement::data_ptr` | wrong arms called `data_as_f32` instead |
| `data_as_f32`'s dtype assert | a direct `(*t).data as *const f32` skips it |
| `dtype_op_matrix.rs` fixtures | the fixture helper shared the code's wrong assumption |
| sealing `.data` (chelis#893) | forces an accessor, but not the *right* accessor |

Sealing the pointer is necessary and does not finish the job. It makes the accessor the
only path; it does not make the path typed. `f32::data_ptr_unchecked(bool_tensor)` still
compiles after the seal.

## What changes

Make the runtime check a **constructor** whose result carries the proof, so downstream
access cannot be wrong — parse, don't validate.

```rust
#[repr(transparent)]
pub struct Tensor<T: TensorElement> {
    raw: *mut chelis_tensor,
    _elem: PhantomData<T>,
}
```

`Tensor<f32>` and `Tensor<Bool8>` become distinct types. `cmp_loop::<f32>` cannot receive
a bool tensor. The line that caused the bool heap overflow —

```rust
let out_buf = data_as_f32(out);   // out is the CHELIS_BOOL output
```

— stops compiling, because `out: Tensor<Bool8>` has no `data_as_f32`. That is the same
line shape as five of the eight int32 arms.

The runtime tag still has to become a static type somewhere. The point is that it
happens in **one** place, an exhaustive typed dispatch, rather than ambiently at sixty
call sites.

## Impact

- **Affected specs:** new capability `typed-tensor-access`.
- **Affected code:** `chelis-runtime` internals. The `extern "C"` surface keeps raw
  pointers — that boundary is irreducible — but each exported function converts once, on
  entry, instead of leaving the tensor untyped throughout its body.
- **Not affected:** the C ABI. `Tensor<T>` is `#[repr(transparent)]` over the same
  pointer, and no struct layout, symbol, or header changes.
- **Depends on:** `seal-tensor-data-pointer` (chelis#893). Sealing first is what forces
  every site through an accessor; typing the accessor is what makes the forced path
  correct. In the other order the seal has to be redone.

## Non-goals

- **Kernel bodies.** A kernel that writes `1.0f` into a correctly typed one-byte buffer
  still compiles. The C backend's `cmplt` ternary had to change from `1.0f : 0.0f` to
  `1 : 0` independently of its pointer type. Phantom-typed `CExpr<T>` is the analogous
  move and is a separate, larger argument.

  **Correction.** This non-goal originally excluded the backends wholesale, on the
  grounds that emitted C is text and no Rust type could have caught `elem_type` returning
  `"float"` for bool. That was too broad, and HIP is the evidence: when
  `CRuntime-BoolStorage-F1` moved bool to one byte, HIP's device allocation followed
  automatically while `dtype_c_type` kept its `float` spelling, producing a four-times
  device-side heap overflow. A Rust type cannot check a kernel's *body*, but it can
  constrain the *element spelling* — a choice from a fixed set, and the thing that was
  wrong. That half is [`bind-lane-element-types`](../bind-lane-element-types/proposal.md).
- **The FFI boundary itself.** `#[repr(C)]` structs crossing a C ABI cannot be typed by
  Rust generics. The goal is to make the untyped region one line per exported function,
  not to eliminate it.
- **Removing runtime checks.** The check moves to the constructor; it does not disappear.
  A tensor arriving from C still has to be validated once.
