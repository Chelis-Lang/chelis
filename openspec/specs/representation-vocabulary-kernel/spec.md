# Representation Vocabulary Kernel Specification

## Purpose

Define the closed physical representation vocabulary, derived width rules, purity boundary, total decoders, and cross-language runtime evidence.

## Requirements

`spec/design/loud_unsupported.md` §C4 controls vocabulary ownership, purity,
and decoder totality. `spec/design/dtype_semantics.md` §C3 controls storage
decisions. This capability records the representation mechanism. It does not
replace either authority. `Repr` describes each current encoding. It does
not select a future storage format.

### Requirement: Closed physical representation vocabulary
`chelis-vocab` SHALL define one closed `Repr` variant for each active runtime element encoding. The vocabulary SHALL contain no unknown, custom, or fallback variant.

#### Scenario: Runtime dtype maps to one representation
- **WHEN** a caller requests the representation of any value in `RuntimeDType::ALL`
- **THEN** the caller receives one concrete `Repr` value

#### Scenario: Equal widths keep distinct identities
- **WHEN** two encodings have equal byte widths but different bit meanings
- **THEN** the encodings have different `Repr` values

#### Scenario: Current bool encoding remains explicit
- **WHEN** a caller requests the current runtime bool representation
- **THEN** the result names the four-byte binary32 payload encoding

### Requirement: Width derives from representation
`RuntimeDType::byte_width()` SHALL derive its result from `RuntimeDType::repr()`. No per-dtype width table SHALL exist beside the representation table.

#### Scenario: Existing widths remain stable
- **WHEN** a caller requests the width of each value in `RuntimeDType::ALL`
- **THEN** every width equals the value on `main` before this change

#### Scenario: Distinct representations can share a width
- **WHEN** a caller compares IEEE binary32 with two's-complement int32
- **THEN** both widths equal four bytes and both representation values remain distinct

### Requirement: Dependency-bottom purity
`chelis-vocab` SHALL use `no_std`, forbid unsafe code, declare no dependency, and use no allocation. The crate SHALL contain only closed data and total vocabulary functions.

#### Scenario: Dependency addition fails the purity test
- **WHEN** the manifest declares a normal, development, or build dependency
- **THEN** the crate purity test fails

#### Scenario: Allocation opt-in fails the purity test
- **WHEN** the crate imports or references `alloc`
- **THEN** the crate purity test fails

#### Scenario: Unsafe opt-in fails the purity test
- **WHEN** the crate removes its unsafe-code prohibition
- **THEN** the crate purity test fails

### Requirement: Vocabulary decoders remain total
Each public vocabulary decoder SHALL return a typed result for every input. An unknown effect symbol SHALL remain borrowed from the caller input.

#### Scenario: Known effect symbol decodes
- **WHEN** the decoder receives `random` or `resource`
- **THEN** it returns the matching `EffectKind`

#### Scenario: Unknown effect symbol is rejected without allocation
- **WHEN** the decoder receives an unknown borrowed symbol
- **THEN** it returns a typed error that borrows the same symbol

#### Scenario: Runtime dtype ID round-trips
- **WHEN** a caller decodes the ID of any value in `RuntimeDType::ALL`
- **THEN** the decoder returns that original value

#### Scenario: Invalid runtime dtype ID is rejected
- **WHEN** a caller decodes an ID outside the closed runtime dtype domain
- **THEN** the decoder returns `RuntimeDTypeDecodeError::InvalidId`

### Requirement: C-header rendering belongs to the runtime shell
`chelis-runtime` SHALL own C-header rendering. The renderer SHALL use the closed vocabulary and SHALL preserve the checked-in header bytes.

#### Scenario: Checked-in header has no drift
- **WHEN** the runtime test renders the dtype header
- **THEN** the result equals `include/chelis_runtime_dtype.h` byte for byte

#### Scenario: Vocabulary contains no source renderer
- **WHEN** the purity test inspects `chelis-vocab`
- **THEN** it finds no source generator and no allocation support

### Requirement: Generated C behavior agrees with the Rust vocabulary
An executable C test SHALL compare the generated helper behavior with `RuntimeDType::byte_width()` for the complete valid tag domain. The test SHALL also execute the invalid-tag path.

#### Scenario: Valid C tag returns the Rust width
- **WHEN** the C helper receives any ID from `RuntimeDType::ALL`
- **THEN** it returns the width from the matching Rust vocabulary value

#### Scenario: Invalid C tag fails loudly
- **WHEN** the C helper receives an ID outside the closed runtime dtype domain
- **THEN** its process exits unsuccessfully instead of returning a fallback width

### Requirement: Element access is compatible with physical representation
Runtime and generated-code element consumers SHALL select a pointer or value type that is compatible with `RuntimeDType::repr()`.

This rule supplies implementation evidence for `spec/design/loud_unsupported.md` section C6.2. Equal byte widths SHALL NOT permit one shared element view.

`Repr::TwosComplement32` SHALL use `i32` in Rust and `int32_t` in generated C.

#### Scenario: Runtime int32 access uses the native element type
- **WHEN** a corrected runtime consumer dispatches `RuntimeDType::I32`
- **THEN** it reads and writes elements through `i32` access instead of an `f32` compatibility view

#### Scenario: Equal widths keep separate views
- **WHEN** a consumer supports both `Repr::Ieee754Binary32` and `Repr::TwosComplement32`
- **THEN** it uses separate float and signed-integer element views although both representations use four bytes

#### Scenario: Current bool payload keeps its compatibility view
- **WHEN** a consumer dispatches the current `Repr::BoolInBinary32`
- **THEN** it can use the `f32` compatibility view until a separate bool-storage migration changes the representation

#### Scenario: Int32 use of the f32 boundary fails in debug builds
- **WHEN** debug code passes an int32 tensor to `data_as_f32` or `data_as_f32_const`
- **THEN** the compatibility boundary fails with an assertion before element access

### Requirement: Affected runtime operations decode native int32 values
The seven affected runtime operations SHALL decode each int32 operand and result as a signed two's-complement 32-bit value.

For in-range inputs, each result SHALL obey [04-NUM-8] and the operation rules in `spec/05-risc-primitives.md`.

This requirement does not close the overflow trap contract in [04-NUM-3].

#### Scenario: Comparison orders negative int32 values
- **WHEN** `cmplt` compares native int32 values `[-1, 0, 1]` with `[0, 0, 0]`
- **THEN** it returns `[true, false, false]` instead of results from NaN bit reinterpretation

#### Scenario: Where treats int32 minimum as nonzero
- **WHEN** `where` receives int32 conditions `[i32::MIN, 0]`
- **THEN** it selects the first then-value and the second else-value

#### Scenario: Scatter-add accumulates native int32 updates
- **WHEN** in-range int32 updates `1_000_000_000` and `-999_999_999` target one destination
- **THEN** scatter-add stores the exact int32 result `1`

#### Scenario: Cumsum preserves exact in-range prefixes
- **WHEN** `cumsum` receives int32 values `[1_000_000_000, -999_999_999]`
- **THEN** it returns `[1_000_000_000, 1]`

#### Scenario: Trace accumulates native int32 diagonal values
- **WHEN** an int32 diagonal contains `1_000_000_000` and `-999_999_999`
- **THEN** `trace` returns the exact int32 result `1`

#### Scenario: Clamp compares signed int32 values
- **WHEN** `clamp` receives `[-1_000_000_000, 7, 1_000_000_000]` with bounds `-8` and `8`
- **THEN** it returns `[-8, 7, 8]`

#### Scenario: Einsum multiplies native int32 values
- **WHEN** an in-range int32 einsum multiplies `2` by `3`
- **THEN** it returns the exact int32 result `6` instead of a float-underflow result

### Requirement: Generated C uses the int32 element type
The C host emitter SHALL map `DtypeArm::I32` to `int32_t`. It SHALL keep the int32 arm separate from the f32 and bool arms.

#### Scenario: Int32 arm emits signed integer pointers
- **WHEN** the host emitter generates an elementwise `CHELIS_I32` arm
- **THEN** the target and input pointers in that arm use `int32_t`

#### Scenario: Int32 arm contains no float pointer
- **WHEN** a test inspects the generated `CHELIS_I32` arm
- **THEN** that arm contains no `float` pointer for int32 element access

#### Scenario: F32 arm remains a float arm
- **WHEN** the host emitter generates the sibling `CHELIS_F32` arm
- **THEN** its element pointers remain `float` pointers

#### Scenario: Int32 max and min remain exact above the binary32 range
- **WHEN** generated max or min code receives an int32 value above 2²⁴
- **THEN** it compares int32 values without calling `fmaxf` or `fminf`

#### Scenario: F32-only unary helpers reject int32
- **WHEN** generated dispatch reaches an int32 arm for an f32-only unary helper
- **THEN** it aborts instead of converting the int32 value through binary32
