# hip-bool-storage

## ADDED Requirements

### Requirement: The HIP lane reads and writes bool at one byte per element
Every HIP emission decision that implies an element width for `CHELIS_BOOL` SHALL agree
with the runtime's declared width.

#### Scenario: The emitted element type is one byte
- **WHEN** the emitter names the C type for a bool tensor's buffer
- **THEN** it names a one-byte type, matching what `chelis_gpu_alloc` reserved

#### Scenario: Device buffers are not overrun
- **WHEN** a kernel writes every element of a bool tensor
- **THEN** it writes exactly the number of bytes the device allocation holds

#### Scenario: Comparison results are byte-valued
- **WHEN** a comparison kernel writes its boolean result
- **THEN** it stores `1` or `0` as a byte, not a four-byte float bit pattern

#### Scenario: A scalar bool fill is byte-valued
- **WHEN** a pad or const fill materialises a bool scalar
- **THEN** it emits a byte-valued local rather than reconstructing an f32 bit pattern

### Requirement: The HIP lane derives element widths rather than restating them
HIP SHALL obtain element widths from the vocabulary.

#### Scenario: No hand-written width table remains in HIP
- **WHEN** the emitter or memory planner needs an element width
- **THEN** it derives it, so a vocabulary change moves it without edit

#### Scenario: The memory estimate matches the allocation
- **WHEN** the memory planner estimates device bytes for a dtype
- **THEN** the estimate uses the same width the runtime allocates with

### Requirement: A HIP parity probe compares encodings against the runtime
The HIP backend SHALL carry a test comparing its element encoding to the runtime's for
every dtype it emits, and that test SHALL run on the default suite.

#### Scenario: The probe runs without a GPU
- **WHEN** the default test suite runs on a machine with no AMD device
- **THEN** the probe runs, because it compares emitted text and metadata rather than
  executing kernels

#### Scenario: A disagreement fails with both encodings named
- **WHEN** HIP's element encoding for a dtype differs from the runtime's
- **THEN** the probe fails naming the dtype, both encodings, and which lane owes the fix

#### Scenario: Representations are compared, not widths
- **WHEN** two encodings share a width but differ in meaning
- **THEN** the probe still fails, because equal widths do not make buffers interchangeable

#### Scenario: Agreement is pinned to a value
- **WHEN** HIP and the runtime agree on bool
- **THEN** the probe pins that the agreed width is one byte, so both drifting together is
  still a failure
