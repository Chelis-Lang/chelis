# Give every lane a dtype ABI parity probe

## Why

`chelis-backend-metal/tests/dtype_abi_width_parity.rs` exists. `chelis-backend-hip` has no
equivalent. That asymmetry is the whole issue.

The Metal probe is the reason anyone knew bool disagreed across the host/device boundary
(chelis#892). It compared the emitted device encoding against the runtime's and failed,
loudly, on a mismatch that had existed since the Metal lane was written.

HIP had the same class of mismatch and nobody found it, because nothing was looking. When
`CRuntime-BoolStorage-F1` moved bool to a native byte, HIP's device allocation followed
automatically while its kernel element type stayed `float` — a four-times device-side heap
overflow. The 6,392-test workspace run was green, because HIP's GPU tests are `#[ignore]`d
manual gates and the codegen tests only inspect emitted text.

**The lane with a probe is the lane whose defect was found. The lane without one is the
lane that broke.**

## What changes

Every backend that moves tensor data across a lane boundary gets a parity probe with the
same shape as Metal's:

1. map the emitted device type to a `Repr`
2. compare against the runtime `Repr` for that dtype
3. fail with a message naming the *direction* of the correct fix

## Compare representations, never widths

This is the part worth stating precisely, because the cheaper check looks equivalent and
is not.

A width is a projection of a representation, and it is **not injective**:

- `Ieee754Binary32` and `TwosComplement32` are both four bytes and are not interchangeable
- `TwosComplement8` and `Bool8` are both one byte and are not interchangeable

The evidence is unusually clean. The width probe caught **bool** because `1 != 4`, and was
completely blind to **int32**, where both lanes reported four bytes while one held two's
complement and the other read IEEE floats — nine sites decoding wrongly for months, in the
same file the probe was passing against.

**Width found the defect it happened to be sensitive to and was blind to the one beside
it.** A parity probe that compares `byte_width` provides false assurance in exactly the
case that is hardest to detect by other means.

## The failure message names a direction

Metal's probe carries an instruction not to close a mismatch by widening Metal to f32,
because that would satisfy the equality while moving toward the f32 coupling the runtime
was removing. Without it, the cheapest way to make the test green is the wrong fix.

Every probe should carry the same: which lane owes the migration, and why.

## Impact

- **Affected specs:** new capability `lane-abi-parity`.
- **Affected code:** new test files under `chelis-backend-hip` and any future lane; the
  existing Metal probe becomes the reference implementation.
- **Depends on:** nothing. This is the cheapest item in the family and the only one that
  covers what the compile-time work cannot.

## Why this cannot be replaced by the compile-time proposals

[`bind-lane-element-types`](../bind-lane-element-types/proposal.md) makes a wrong element
*spelling* unwriteable. It does not constrain a kernel *body*: a kernel that writes `1.0f`
into a correctly typed one-byte buffer still compiles. The C backend's `cmplt` kernel did
exactly that, and its emitted ternary had to be changed from `1.0f : 0.0f` to `1 : 0`
independently of its pointer type.

Emitted kernels are text. A probe that inspects or executes them is the only thing that
reaches them.

## Non-goals

- **Running GPU work in CI.** These probes are text-and-metadata comparisons and belong on
  the per-PR gate. Executing kernels stays a documented manual gate.
- **Unifying the lanes' emitters.** The probe compares outputs; it does not require the
  backends to share structure.
