# Representation Vocabulary Kernel Specification

## Purpose

Define the closed physical representation vocabulary, derived width rules, purity boundary, total decoders, and cross-language runtime evidence.

## Requirements

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
