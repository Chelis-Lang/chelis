## ADDED Requirements

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
