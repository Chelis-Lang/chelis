# lane-abi-parity

## ADDED Requirements

### Requirement: Every lane that transfers tensor data has a parity probe
Each backend that moves tensor buffers across a host/device or host/foreign boundary SHALL
carry a test comparing its element encoding against the runtime's, for every dtype it
supports.

#### Scenario: A lane without a probe is a gap
- **WHEN** a backend gains the ability to transfer a dtype's buffers
- **THEN** that dtype appears in that lane's parity probe

#### Scenario: The probe runs on the per-PR gate
- **WHEN** the default test suite runs
- **THEN** the probe runs, because it compares emitted text and metadata rather than
  executing device work

#### Scenario: A disagreeing lane fails
- **WHEN** a lane's element encoding differs from the runtime's for any dtype
- **THEN** the probe fails and names the dtype, both encodings, and the lane

### Requirement: Parity is checked by representation, not by width
A probe SHALL compare representations. Comparing byte widths alone does not satisfy this
requirement.

#### Scenario: Equal widths with different encodings fail
- **WHEN** two lanes agree on a dtype's width but disagree on its encoding
- **THEN** the probe fails, because equal widths do not make buffers interchangeable

#### Scenario: The mapping follows the emitter
- **WHEN** the emitted device type for a dtype changes
- **THEN** the probe's mapping follows the emitter rather than restating a fixed table,
  so the comparison is what fails and not the mapping

#### Scenario: An unmapped device type is loud
- **WHEN** a lane emits a device type the probe has no representation for
- **THEN** the probe fails naming that type, rather than skipping the dtype

### Requirement: A parity failure names which lane owes the fix
A probe's failure message SHALL state the direction of the correct resolution, not only
that a mismatch exists.

#### Scenario: The cheap wrong fix is called out
- **WHEN** a mismatch could be resolved by changing either lane, and only one direction is
  correct
- **THEN** the message says which lane owes the migration and why the other direction is
  wrong

#### Scenario: Agreement is pinned to a value, not to equality alone
- **WHEN** two lanes agree
- **THEN** the probe also pins the agreed encoding, so that both lanes drifting together
  is still a failure
