## ADDED Requirements

### Requirement: Authority transfers one chapter at a time
Normative authority SHALL move from a numbered `spec/` chapter to its captured capability one chapter at a time, through a change that records the transfer. A chapter SHALL remain controlling for its subject until its transfer is recorded.

#### Scenario: Untransferred subject keeps numbered authority
- **WHEN** a capability and its source chapter describe the same subject and no transfer is recorded for that chapter
- **THEN** the `spec/` chapter SHALL be controlling and the capability SHALL be reference material

#### Scenario: Transferred subject moves authority
- **WHEN** a chapter's transfer has been recorded
- **THEN** the capability SHALL be controlling for that subject

#### Scenario: Bulk transfer is rejected
- **WHEN** a change records transfer for a chapter whose capability it does not also present for review
- **THEN** the transfer SHALL NOT be accepted

### Requirement: A capability names the chapter it was captured from
Every capability captured from `spec/` SHALL name and link its source chapter. A capability without that citation SHALL NOT be eligible for authority transfer.

#### Scenario: Cited capability is auditable
- **WHEN** a capability records its source chapter
- **THEN** a reviewer SHALL be able to compare it against that chapter, and the chapter SHALL be discoverable as captured

#### Scenario: Uncited capability cannot transfer
- **WHEN** a transfer is proposed for a capability carrying no source citation
- **THEN** the transfer SHALL be rejected

### Requirement: Known divergence is recorded before transfer
A capture SHALL NOT be required to be lossless. Divergence known at capture time between the
chapter and capability SHALL be recorded in the owning change before that chapter transfers.
Implementation divergence SHALL be tracked in the owning change, design document, or issue and
SHALL NOT be copied into the normative capability or used to narrow it.

#### Scenario: Recorded gap does not block transfer
- **WHEN** a capture omits or contradicts part of its chapter and the divergence is recorded
- **THEN** the transfer SHALL remain eligible to proceed

#### Scenario: Unrecorded known divergence blocks transfer
- **WHEN** a divergence is known at capture time and is not recorded
- **THEN** the transfer SHALL be rejected until it is

### Requirement: Normative capabilities are timeless contracts
Every controlling capability and normative capability delta SHALL describe the decided
architecture, syntax, or semantics independent of delivery state. It SHALL NOT contain phase or
milestone labels, completion claims, delivery history, release or PR inventories, acceptance
results, temporary workarounds, or descriptions of implementation behavior. A short
non-normative parenthetical MAY say that a requirement is not fully implemented and link its
owning issue; it SHALL NOT describe the gap.

#### Scenario: Implementation behavior cannot narrow a capability
- **WHEN** an evaluator or backend supports only a subset of the intended dtype or operation contract
- **THEN** the capability retains the intended contract and the divergence is tracked outside the normative text

#### Scenario: Status belongs in the owning change
- **WHEN** work is sequenced or implementation progress is reported
- **THEN** that information belongs in proposal, design, tasks, project design documents, or issues rather than a capability spec

### Requirement: A transferred chapter is marked superseded
When a chapter's authority transfers, that chapter SHALL be marked superseded and SHALL name the capability that supersedes it. A superseded chapter SHALL NOT receive new normative content.

#### Scenario: Reader of a superseded chapter is redirected
- **WHEN** a reader opens a chapter whose authority has transferred
- **THEN** the chapter SHALL state that it is superseded and name the controlling capability

#### Scenario: Normative edit to a superseded chapter is rejected
- **WHEN** a change adds or alters normative content in a superseded chapter
- **THEN** the change SHALL be rejected and directed at the controlling capability

### Requirement: A controlling document that denies OpenSpec authority blocks transfer
Authority SHALL NOT transfer for a subject while a document still in force denies OpenSpec authority over that subject. Such a document SHALL be amended before the transfer it blocks.

#### Scenario: Boundary is amended before transfer
- **WHEN** every in-force document permits OpenSpec authority over a subject
- **THEN** that subject's chapter SHALL be eligible for transfer

#### Scenario: Transfer against an unamended boundary is rejected
- **WHEN** a transfer is proposed while an in-force document denies OpenSpec authority over that subject
- **THEN** the transfer SHALL be rejected until that document is amended
