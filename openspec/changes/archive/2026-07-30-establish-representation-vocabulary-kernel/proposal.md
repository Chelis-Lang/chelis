## Why

`RuntimeDType` currently models physical storage only through byte width. Equal widths cannot identify different physical encodings, such as IEEE 754 binary32 and two's-complement int32.

`chelis-vocab` also emits C source with allocation and `std`, although the crate is the dependency-bottom vocabulary. PR #896 needs a closed representation model before it adds arithmetic width.

## What Changes

- Add a closed `Repr` vocabulary for each physical element encoding.
- Preserve all current ABI behavior, including the current four-byte encoded bool representation.
- Derive each runtime dtype byte width from its `Repr` value.
- Make `chelis-vocab` use `no_std`, no allocation, no unsafe code, and no dependencies.
- **BREAKING Rust API**: Borrow unknown effect symbols instead of allocating owned error strings.
- **BREAKING Rust API**: Move C-header rendering from `chelis-vocab` to `chelis-runtime` without an output change.
- Add structural tests for crate purity and closed enum coverage.
- Add executable C tests for every runtime dtype tag and the invalid-tag failure path.
- Exclude arithmetic-width rules, int32 decoder fixes, bool storage migration, tensor-pointer sealing, and toolchain changes.

## Capabilities

### New Capabilities

- `representation-vocabulary-kernel`: Defines the closed runtime representation vocabulary, derived widths, purity boundary, and cross-language evidence.

### Modified Capabilities

None.

## Impact

The change affects `chelis-vocab`, the C-header generator in `chelis-runtime`, and their tests. Workspace callers must use the borrowed decode error and the new generator path.

The change adds no dependency and changes no public dtype ID, macro, width, or generated header byte.
