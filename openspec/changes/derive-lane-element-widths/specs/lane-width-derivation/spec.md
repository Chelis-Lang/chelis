# lane-width-derivation

## ADDED Requirements

### Requirement: No lane restates a dtype's element width
Backend crates, their tests, and their harnesses SHALL obtain a dtype's element width
from the vocabulary rather than from a locally written table.

#### Scenario: A lane needing a width derives it
- **WHEN** a backend sizes a buffer, a transfer, or a memory estimate
- **THEN** the width comes from `RuntimeDType::byte_width`, not from a match written in
  that crate

#### Scenario: Changing a width moves every lane at once
- **WHEN** a dtype's representation changes its width in the vocabulary
- **THEN** every derived site follows without edit, and no site retains the old value

#### Scenario: A reintroduced table is caught
- **WHEN** a new hand-written mapping from a dtype to a byte count is added to a backend
- **THEN** a check reports it, so the duplication is a deliberate reviewable act rather
  than an unnoticed one

### Requirement: A dtype without a width is a loud failure
Deriving a width for a dtype the vocabulary does not describe SHALL fail with the dtype
named, rather than falling back to a default.

#### Scenario: An unmapped precision names itself
- **WHEN** a backend derives a width for a precision that has no runtime dtype
- **THEN** the failure names the precision and the backend, rather than silently choosing
  four bytes

#### Scenario: No silent default arm
- **WHEN** the derivation is written
- **THEN** it contains no wildcard arm that yields a width, because such an arm makes a
  new dtype inherit whichever width happened to be the default
