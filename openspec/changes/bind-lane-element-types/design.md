# Design

Four decisions this change cannot be built without. Each is settled here rather than left
to implementation, because three of them constrain where the code can live at all.

## 1. The element marker types have to move to `chelis-vocab`

This looks like a free choice and is not.

The element types that would carry the binding — `Bool8`, and whatever eventually stands
for `bf16` / `f16` — live in `chelis-runtime`. **No backend depends on `chelis-runtime`:**

| crate | depends on |
| --- | --- |
| `chelis-backend-c` | `chelis-ir`, `chelis-types`, `chelis-vocab`, `half` |
| `chelis-backend-hip` | `chelis-ir`, `chelis-types`, `chelis-vocab` |
| `chelis-backend-metal` | `chelis-ir`, `chelis-types`, `chelis-vocab`, `half` |

Backends emit *text* that calls the runtime; they do not link it. So a trait over
`chelis-runtime`'s types is unreachable from the lanes without inverting that layering,
which would make every backend depend on the artifact it generates calls into. That is a
much larger change than this one and a worse architecture.

The alternative — each lane defining its own marker types — reintroduces exactly the
duplication this change exists to remove.

So the markers belong in `chelis-vocab`, the shared dependency bottom. That is coherent
rather than a workaround: vocab already owns `RuntimeDType` and `Repr` and describes
itself as the single authority for identifiers crossing compiler, runtime, and
generated-code boundaries. **A type whose size pins a dtype's width is exactly such an
identifier.** `chelis-runtime` then implements `TensorElement` over vocab's markers rather
than over its own.

## 2. Vocab must not learn a C type name

Given (1), the obvious next step is to put `C_NAME` on the marker in vocab. Don't.

There is no single C name. Metal emits MSL (`bool`), the C backend emits C99
(`unsigned char`), HIP emits HIP C++. A vocab-owned name would have to enumerate lanes,
which makes the dependency bottom know how many backends exist — backwards, and it means
adding a lane edits vocab.

The split that works:

- **vocab owns the constraint** — the marker type, its size, and its `RuntimeDType`
- **each lane owns its spelling** — a per-lane trait, implemented in that backend

Note vocab already names C identifiers via `c_macro()` (`CHELIS_F32`), so the boundary is
not "vocab knows nothing about C". It is that `CHELIS_F32` is *one* identifier shared by
every lane, whereas an element type name is per-lane by construction.

## 3. The assertion must be unskippable, so the impl is macro-generated

A per-lane trait means each lane writes impls. If the width assertion is a separate
`const _: () = assert!(...)` next to each impl, a lane can write the impl and forget the
assertion — and the whole point is that agreement stops being maintained by hand. That is
the failure mode that produced the HIP overflow.

So the impl and the assertion are emitted together and there is no way to have one
without the other:

```rust
macro_rules! lane_element {
    ($trait:ident, $ty:ty, $dtype:expr, $name:literal) => {
        impl $trait for $ty {
            const DTYPE: RuntimeDType = $dtype;
            const NAME: &'static str = $name;
        }
        const _: () = assert!(
            core::mem::size_of::<$ty>() as u32 == $dtype.byte_width(),
            concat!(stringify!($ty), " does not have its dtype's declared width")
        );
    };
}
```

A lane that wants an element spelling invokes the macro. There is no hand-written impl
path, so there is no path that skips the check.

## 4. bf16 / f16 get homegrown newtypes, and the reason is a test

Three proposals now need this decision, so settle it here.

`half::bf16` and `half::f16` are `#[repr(transparent)]` over `u16` with **no validity
invariant** — every bit pattern is a valid value. That is a real difference from `bool`,
which is why `impl TensorElement for half::bf16` would be sound where
`impl TensorElement for bool` was not. So the `Bool8` precedent does not automatically
carry: the newtype is not needed for *soundness* here.

It is needed anyway, because of (1). The markers must live in `chelis-vocab`, and vocab
has **zero dependencies** — a property now enforced by
`crates/chelis-vocab/tests/crate_purity.rs::manifest_declares_no_dependencies`, which
fails if any dependency section is non-empty. Taking `half` into vocab would trip that
test, and the test is worth more than the convenience.

`chelis-backend-hip` also does not depend on `half` today, so a `half`-based marker would
add a dependency to a crate that has done without one.

So: `Bf16Bits(u16)` and `F16Bits(u16)` in vocab, named for the storage rather than the
logical type, exactly as `Bool8` is. Conversion to `half::bf16` stays in the crates that
already depend on `half` and actually do arithmetic. **Storage identity and numeric
behaviour are different concerns and this is the seam between them.**

## What the const assertion does not prove

Worth stating plainly, because the assertion is persuasive and narrow.

`size_of::<Bool8>() == RuntimeDType::Bool.byte_width()` is a claim about the **host**.
It says the marker agrees with the vocabulary. It says nothing about whether the device
type the lane names is that width *on the device*: MSL `bool` being one byte on Apple
silicon is a fact about Metal's ABI that no Rust assertion reaches.

That gap is `probe-lane-dtype-abi`'s, and it is the reason these are separate proposals
rather than one. The compile-time binding makes the host side impossible to get wrong;
the probe is what checks the device agrees.

Similarly out of reach: a kernel *body* writing a wrong-width value into a correctly typed
buffer. The C backend's `cmplt` ternary had to change from `1.0f : 0.0f` to `1 : 0`
independently of its pointer type, and no binding would have forced that.

## `elem_kind` is a different thing and probably in scope anyway

HIP's `elem_kind` selects a *kernel family*, not a type name — `Prim::F32 | Prim::Bool =>
ElemKind::F32`. It is not a spelling, so it falls outside this change's stated shape.

But it is how bool kernels came to write four-byte values, so excluding it on a
definitional technicality would leave the actual defect half-fixed. Task 2.4 carries the
decision; the honest options are to widen this change to cover "any per-dtype choice that
implies a width" or to hand `elem_kind` to `probe-lane-dtype-abi` and accept that it is
caught rather than prevented.
