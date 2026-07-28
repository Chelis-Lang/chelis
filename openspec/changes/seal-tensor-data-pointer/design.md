# Design: seal-tensor-data-pointer

## The risk that shapes everything else

A partial pointer migration compiles, passes the test suite, and corrupts memory. This is
not a hypothetical; it happened twice in the week before this proposal was written, in
the same file.

**First:** flipping `Repr` for bool to a 1-byte width without migrating the access sites.
`chelis_tensor_cmplt` allocates `CHELIS_BOOL` and writes through `data_as_f32(out)`, so
it wrote `size * 4` bytes into a `size * 1` buffer — a 4× heap overflow.
`cargo test -p chelis-runtime` reported green.

**Second:** the `data_as_f32` dtype assert looked like a complete tripwire, and five
bool-reachable sites bypassed it with direct casts. Under the same width flip each would
have read four bytes per element out of a one-byte buffer. Found by counting the direct
casts rather than by any test.

Both were caught by measurement, not by the suite. So the sequencing below is built to
make a partial state *unshippable* rather than merely discouraged.

### How a partial state is prevented here

The migration is safe in a way the bool one is not, and the difference is worth stating:
**every accessor introduced here is a typed spelling of a cast that already exists.** No
byte moves. A half-migrated tree is therefore *correct*, just inconsistent — the failure
mode is a missed site that still compiles, not a buffer whose stride changed underneath
its readers.

That means the field seal is the only irreversible step, and it is last. Before it, the
compiler cannot help; after it, every remaining direct access is a compile error. There
is no window in which the code is wrong.

The ordering obligation is therefore narrow but absolute: **the seal lands only when the
site count is zero, and the seal is what proves it.** Do not seal and then chase errors
in the same commit — seal in a commit that changes nothing else, so the diff is the proof.

## Why module privacy and not the alternatives

| mechanism | prevents the eight bugs? | cost |
| --- | --- | --- |
| `TensorElement::data_ptr` | no — optional, and the wrong arms did not call it | shipped |
| `data_as_f32` dtype assert | no — direct casts bypass it | shipped |
| `pub(crate)` on `data` | **no — all eight sites were inside the crate** | trivial |
| Barnacle `restricted_field_access` | yes | a nightly Dylint toolchain |
| **module privacy** | **yes** | **63 call sites** |

`pub(crate)` is the tempting answer because the external count is 4. It is also useless
here, and recording why matters: the external boundary was never the leak. Every instance
of this defect class has been runtime-internal.

Barnacle's `boundary_safety_restricted_field_access` is genuinely the right shape — it
warns when a configured field is accessed outside allowed modules, which is this rule
exactly. It is the fallback rather than the primary because it costs a second toolchain
and because privacy is enforced by the compiler every contributor already runs.

## The accessor surface

Three tiers, deliberately unequal in ergonomics so the safe one is the easy one.

**Typed element access.** `TensorElement::data_ptr` / `data_ptr_unchecked`, already
shipped, covering `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, and `Bool8` once bool migrates.
This should be the only comfortable way to read elements.

**Raw bytes.** For the memcpy, allocation, and free paths that are legitimately untyped —
8 sites cast to `*const u8` today. This accessor must be *awkward enough to notice*:
`unsafe`, named for what it is, and documented as not for element access. If it is
pleasant to use it will be used to rebuild the hole this change closes.

**Pointer hygiene.** Null checks and the free path, including `chelis-python`'s four
sites. These never dereference and should not need `unsafe` element access at all.

### The u16 / u8 / u32 gap

Three cast targets have no `TensorElement` impl:

- **`u16`** carries `bf16` and `f16`. The logical type is a 16-bit float; `u16` is its
  storage. This is structurally identical to bool-as-f32, and the `Bool8` newtype is the
  precedent — a `#[repr(transparent)]` element type that every bit pattern inhabits, so
  reads are defined and the narrowing to a semantic value is explicit. Introducing
  `Bf16Bits` / `F16Bits` would let `bf16`/`f16` join the typed tier instead of falling to
  raw bytes, which is where they belong.
- **`u8`** is raw byte access. It belongs in the bytes tier by definition.
- **`u32`** is a single site and is uncharacterised. It may be an f32 bit-pattern read, in
  which case it is a latent instance of the same defect class and should be treated as a
  finding rather than migrated as-is.

Phase 1 settles all three before any call site moves, because the answer determines how
many sites the typed tier can absorb.

## What this does not fix

**Stale documentation.** Nothing here stops a comment rotting, and a stale rustdoc is
what authorised all eight int32 sites — it asserted int32 was f32-encoded long after
RT-4 F1 made it native. The defence is not better comments; it is that after this change
a comment cannot authorise a direct cast, because the cast will not compile. The doc stops
being load-bearing.

**Divergent match arms.** Every one of the eight bugs was a hand-written arm beside a
generic loop the other arms used. Privacy forces those arms to use an accessor; it does
not force them to use the *right* one. A `dispatch_dtype!` macro generated from the
dtype↔element mapping is the complementary fix and is deliberately out of scope here, so
that neither change waits on the other.

**Fixture conventions that encode the bug.** `alloc_vec_with_values` writing I32 as
`value as f32` is why 77 fixtures caught nothing. That is a test-design problem, not an
encapsulation one, and is tracked at its site on `harden-vocabulary-kernel`.

## Rollback

Every batch before the seal is an independent refactor with no behavior change, revertable
on its own. The seal is one line and reverts to `pub`. There is no state in which a
partial revert produces incorrect behavior, which is the property the bool migration
notably lacks.
