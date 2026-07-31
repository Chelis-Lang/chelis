## Why

PR #964 defines `RuntimeDType::I32` as `Repr::TwosComplement32`. However, several runtime consumers still read the four-byte storage through an `f32` view.

Equal widths hide this error. Negative values, `i32::MIN`, and values outside the float denormal range expose different wrong results at different sites.

PR #894 contains an int32-only correction with regression evidence. This change ports the remaining correction onto PR #964 without the bool-storage or environment changes from PR #894.

## What Changes

- Decode native int32 storage through `i32` access in comparison, scatter-add, `where`, `cumsum`, `trace`, `clamp`, and `einsum`.
- Emit `int32_t` for the C backend `DtypeArm::I32` element type.
- Add a debug assertion that rejects `RuntimeDType::I32` at the `data_as_f32` compatibility boundary.
- Add regression tests with values that fail under an `f32` reinterpretation.
- Update the runtime dtype consumer contract and stale storage documentation.
- Preserve the int32 ABI tag, byte width, two's-complement storage, and public C signatures.
- Exclude the completed tensor-formatting correction from this change.
- Exclude bool-storage migration, arithmetic-width work, tensor-pointer sealing, and development-environment changes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `representation-vocabulary-kernel`: Require runtime and generated C consumers to access elements through a type compatible with `RuntimeDType::repr()`.

## Impact

The change affects `chelis-runtime`, the C backend host emitter, runtime and emitter tests, and the runtime dtype consumer documentation.

Runtime and generated C results change for affected int32 operations. The change adds no dependency and changes no C ABI value or storage width.
