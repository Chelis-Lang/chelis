# Compiler Pipeline Architecture Delta: repair-pipeline-core-review-findings

## MODIFIED Requirements

### Requirement: Reef uses canonical semantic transitions
The Reef package artifact and schema paths SHALL use the core type, effect, and linearity transitions. Reef SHALL NOT call those stages in sequence.

The Reef adapter SHALL preserve its linked-program guard. It SHALL preserve exact accepted artifacts and exact rejected error text.

A parity leg deferred behind an ignore attribute SHALL execute its assertions when explicitly invoked through its documented manual command. A harness that re-executes itself in a child process SHALL propagate the flags the deferred leg needs to run, and an invocation whose child process executes zero tests SHALL fail rather than report success.

#### Scenario: Valid linked package keeps its artifact
- **WHEN** a valid linked package passes through the core transitions
- **THEN** Reef produces the same checked package, archive content, schema content, and hashes as the baseline

#### Scenario: Invalid linked package keeps its rejection
- **WHEN** a linked package fails type, effect, or linearity checks
- **THEN** Reef returns the same error text and order as the baseline

#### Scenario: Linked internal names remain accepted
- **WHEN** linked Deep contains valid internal mangled names
- **THEN** the linked-program guard remains active for the complete core semantic check

#### Scenario: Reef restores direct stage calls
- **WHEN** a negative source fixture restores the former Reef helper sequence
- **THEN** the source guard rejects the fixture and identifies the repeated stages

#### Scenario: A deferred parity leg is explicitly invoked
- **WHEN** the ignored accepted parity leg runs through its documented manual command
- **THEN** the harness child process executes the leg's assertions and the invocation reports that outcome, not the skip

#### Scenario: A harness child executes zero tests
- **WHEN** the re-executed child process selects or runs zero tests for any reason
- **THEN** the parent invocation fails instead of reporting a vacuous success

### Requirement: Context and cache parity

Monolithic and contextual modes SHALL use the same semantic state transitions. The layered Reef path SHALL preserve its clean cache path and monolithic error fallback.

Semantic rejections on the layered path SHALL fold into the monolithic fallback so the user receives the byte-identical monolithic rejection. A library proof-bind mismatch is an internal invariant failure, not a user-program rejection: it SHALL surface as a loud check-stage error and SHALL NOT fold into the monolithic fallback.

Fitness formulas SHALL have one implementation owner. Layered code SHALL NOT copy fitness weights or clean-report formulas.

#### Scenario: Warm contextual check matches cold check

- **WHEN** a valid Reef program runs through cold, warm, and cache-disabled checks
- **THEN** all three modes produce byte-identical public output

#### Scenario: Contextual check rejects new code

- **WHEN** non-stdlib code fails a semantic stage against a valid cached library context
- **THEN** the documented fallback produces the same rejection as the monolithic path

#### Scenario: Layered proof binding fails

- **WHEN** the layered path cannot bind its composed type environment to its checked program
- **THEN** compilation fails with a check-stage error naming the mismatch instead of silently recompiling monolithically
