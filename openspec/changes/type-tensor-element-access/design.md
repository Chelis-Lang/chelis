# Design

## Why the container rather than the element

`Bool8` was worth adding and is not sufficient, and the reason is worth being precise
about.

An element newtype answers "are these two *values* interchangeable". `Bool8` and `i8`
share a width and are now distinct types, so bool storage and int8 storage cannot be
cross-wired at the point where a value is read or written. That is real.

It does not answer "is this *buffer* the one you think it is". The buffer is the tensor,
and the tensor's type is `*mut chelis_tensor` for every dtype. So the element newtype
protects the last inch and leaves the preceding mile untyped — which is where all
twenty-three known sites in this class went wrong.

## Parse, don't validate

The check exists today and is in the wrong position. `data_ptr` validates and returns a
pointer that has forgotten it was validated, so the next site must validate again, and
the site after that, and any one of them may decline.

Moving the check into a constructor changes what a successful check *produces*: not a
pointer, but a value of a type that only exists when the check passed. Downstream code
does not re-check because it cannot be reached with an unchecked tensor.

This is why the accessor on `Tensor<T>` can be free of asserts without weakening
anything. `debug_assert` in `data_ptr_unchecked` is doing the work of a type, at runtime,
in debug builds only.

## The one dynamic-to-static conversion

A visitor is the honest shape, because the tag's domain is closed and the compiler can
force exhaustiveness:

```rust
trait DtypeVisitor<R> {
    fn visit<T: TensorElement>(self, tensor: Tensor<T>) -> R;
}

unsafe fn dispatch<R>(raw: *mut chelis_tensor, v: impl DtypeVisitor<R>) -> R;
```

`dispatch` contains the only `match` on `RuntimeDType` that yields typed access, and it
is one `match` that the compiler checks for completeness against the closed vocabulary.

**This is the `dispatch_dtype!` macro idea, and it fixes the objection to it.** A macro
was rejected earlier in this workstream for sharing the weakness that let the defect
through: it would have been opt-in, exactly as `TensorElement::data_ptr` was opt-in, so a
hand-written `match dtype { … }` beside it bypasses it. Under this design a hand-written
match cannot produce a `Tensor<T>`, so it cannot reach a typed accessor. The dispatch is
not enforced by convention; it is enforced by being the only constructor.

## What this cannot do

**Emitted C is text.** The single highest-impact bug in the bool migration was
`elem_type` mapping `Prim::Bool` to `"float"` — one line, six failing gradient tests. No
Rust type prevents returning the wrong string. The analogous move is a phantom-typed
expression (`CExpr<T>`) so an f32 expression cannot be assigned into a bool slot, and it
is deliberately out of scope: it is a larger change to a different crate and should be
argued on its own evidence.

**The FFI boundary.** Every `#[no_mangle] extern "C"` function receives
`*mut chelis_tensor` and must. The win is that the untyped region shrinks to the entry
line of each exported function rather than spanning its body.

**Dtype-polymorphic ops.** Some operations are genuinely generic over element type and
some are not. `where` copies raw bytes and does not care; `cmplt` needs `PartialOrd`.
Phase 1 should classify the surface before migrating it, because an op that is generic in
its body but not in its bounds is where the design will push back.

## Why this migration is safe to land incrementally

The same property that made the pointer-privacy migration safe applies here, and it is
worth stating because the bool migration did **not** have it and produced a heap
overflow from a half-finished state.

`Tensor<T>` is `#[repr(transparent)]` over the pointer it replaces, and every accessor it
introduces is a typed spelling of a cast that already exists. **No byte moves.** A
half-migrated tree is correct-but-inconsistent, not wrong. Sites can convert in batches
and the suite stays meaningful throughout.

The contrast is instructive: bool had to be atomic because flipping `Repr` changed
allocation sizes, so any unconverted writer overran its buffer. Nothing here changes a
size.

## Alternatives considered

**Keep runtime asserts.** Status quo. Demonstrably insufficient: eight int32 arms, two
e2e bool sites, and five direct casts all bypassed an assert that was present and
correct.

**The `dispatch_dtype!` macro alone.** Rejected earlier in this workstream for being
opt-in. It becomes viable *as an implementation detail* of `dispatch` once the only way
to obtain typed access is through it.

**Encode the dtype id as a const generic (`Tensor<const D: i32>`).** Ties the handle to
the tag rather than to the Rust element type, so `Tensor<3>` still needs a mapping to
`Bool8` to produce a pointer, and the mapping is where the error would live. It moves the
problem rather than removing it.

**Make `chelis_tensor` itself generic.** Impossible: it is `#[repr(C)]` and crosses the
ABI. This is why the phantom type belongs on a wrapper, not on the struct.

## Open questions for Phase 0

- How many runtime functions are genuinely element-generic versus dtype-dispatched? The
  ratio decides whether `dispatch` is called dozens of times or a handful.
- Do `bf16`/`f16` need element newtypes before this lands? They cast to `*mut u16` today
  and have no `TensorElement` impl, for the same reason bool had none: the logical type is
  not the storage type. `Bool8` is the precedent. This is the same gap
  `seal-tensor-data-pointer` Phase 0 must settle, and the two should settle it once.
- Does the visitor need a fallible variant for ops that support a subset of dtypes
  (`cmplt` rejects bf16/f16/i8/i16), or does each op keep its own guard before
  dispatching?
