# Proof Obligation Routing: route-proof-obligations

## ADDED Requirements

### Requirement: A cheaper oracle establishing the same guarantee is preferred to a proof
Before an obligation is discharged by mechanized proof, the available cheaper oracles SHALL be considered, and the comparison SHALL be recorded. Where an exhaustive check over a finite domain, a differential test against an independent implementation, or a property test establishes the same guarantee at materially lower cost, that oracle SHALL be the default and a proof SHALL require a separate justification. A proof SHALL NOT be justified on the grounds that a property is unobtainable otherwise when it is obtainable otherwise.

#### Scenario: A finite domain is checked exhaustively rather than proved
- **WHEN** a property quantifies over a domain small enough to enumerate within the test budget
- **THEN** the exhaustive check is the default oracle, and any proof of the same property carries its own separate justification

#### Scenario: The comparison is recorded
- **WHEN** an obligation is routed to mechanized proof
- **THEN** the cheaper oracles considered and the reason they were insufficient are recorded

#### Scenario: A justification that a cheaper oracle refutes is corrected
- **WHEN** a proof was justified as the only way to establish a property, and a cheaper oracle is found to establish it
- **THEN** the justification is corrected rather than retained, and the proof is re-justified on other grounds or dropped

#### Scenario: Establishing a lane is a permitted justification, stated as such
- **WHEN** a proof is undertaken to measure or establish the prover lane rather than because the property is otherwise unobtainable
- **THEN** that is recorded as the reason, and the property is not described as requiring a proof

### Requirement: The repository maintains one prover lane
The repository SHALL maintain a single mechanized-proof lane. An obligation chosen for proof SHALL be discharged in that lane. Routing SHALL NOT be decided by which prover a contributor is more familiar with.

#### Scenario: An obligation is discharged in the single lane
- **WHEN** an obligation is routed to mechanized proof
- **THEN** it is discharged in the repository's one prover lane

#### Scenario: An obligation the lane cannot express is recorded, not forced
- **WHEN** an obligation is outside the lane's supported subset
- **THEN** it is recorded as unverified with the reason, and working code is not rewritten solely to fit a prover

#### Scenario: Familiarity is not a routing reason
- **WHEN** a contributor proposes a different prover for an obligation the existing lane can express
- **THEN** the proposal is declined, because convenience of expression is not the criterion

### Requirement: A second prover lane requires an obligation the first cannot express
Adding a prover lane SHALL require a named obligation that is wanted, and that the existing lane cannot express. Inconvenience SHALL NOT be treated as inexpressibility. The count of lanes SHALL NOT grow on the strength of a tool's general merits, a preference, or the availability of an unused capability.

#### Scenario: A named inexpressible obligation opens a lane
- **WHEN** an obligation someone wants discharged is outside the existing lane's supported subset, and the candidate prover expresses it
- **THEN** the entry criterion is met and adding the lane may be proposed

#### Scenario: Inconvenience does not open a lane
- **WHEN** the existing lane can express an obligation but another prover would express it more conveniently
- **THEN** the criterion is not met and no lane is added

#### Scenario: General merit does not open a lane
- **WHEN** a prover is proposed on the strength of its capabilities rather than a named obligation
- **THEN** the criterion is not met

#### Scenario: A deferred prover is recorded as deferred, not rejected
- **WHEN** a prover is not adopted for want of a qualifying obligation
- **THEN** the record states that it was deferred for that reason, and names the criterion that would bring it back

### Requirement: A prover lane may exist only if the repository builds without it
A prover SHALL be installable through the repository's declared development environment without altering what the pinned toolchain installs for gate jobs. Code carrying proof annotations SHALL remain compilable and testable on the pinned toolchain by a contributor who has not installed that prover. A lane that cannot satisfy this SHALL be declined rather than adopted with an exception.

#### Scenario: A contributor without the prover is unaffected
- **WHEN** a contributor who has not installed a prover builds and tests the repository
- **THEN** the build and the test suite succeed, and no proof-related step is required of them

#### Scenario: The pinned toolchain is unchanged
- **WHEN** a prover lane is added
- **THEN** what the gate jobs install from the pinned toolchain file is unchanged

#### Scenario: A lane that cannot be isolated is declined
- **WHEN** isolating a prover's verification step from ordinary compilation cannot be arranged cleanly
- **THEN** that lane is declined and the negative result is recorded, rather than adopted with a carve-out

#### Scenario: Declining is recorded as an outcome
- **WHEN** a lane or an individual obligation is declined
- **THEN** the decision and its evidence are recorded, so a later reader does not read the absence as an oversight

### Requirement: Overlapping proofs are not reported as compounded assurance
Where two provers establish the same or overlapping properties about the same code, the result SHALL NOT be reported as stronger than the stronger of the two individually. A count of provers SHALL NOT appear as an assurance figure. Where an overlap exists, the reason it was permitted SHALL be recorded.

#### Scenario: Two proofs of one property are reported once
- **WHEN** both lanes establish the same property about the same code
- **THEN** the reported guarantee is that property, established once, and not a claim strengthened by the number of tools

#### Scenario: Tool count is not an assurance metric
- **WHEN** verification status is summarized
- **THEN** no figure counting provers, lanes, or proof artifacts is presented as a measure of assurance

#### Scenario: A permitted overlap is justified
- **WHEN** the same obligation is deliberately discharged in both lanes
- **THEN** the purpose is recorded, and the overlap is not presented as redundancy-derived confidence

### Requirement: The lane's cost is measured, not estimated
The first obligation discharged in a prover lane SHALL record the effort spent, the toolchain friction encountered, the proof-artifact length, and whether the resulting artifact is reusable by a later proof about the same code. That record SHALL be the basis for sizing any subsequent proof work, in place of estimation.

#### Scenario: The first obligation produces measured figures
- **WHEN** the lane's first obligation is discharged
- **THEN** effort, friction, artifact length, and reusability are recorded

#### Scenario: Later sizing cites the measurement
- **WHEN** further proof work is proposed
- **THEN** its cost estimate cites the recorded measurement rather than a fresh guess

#### Scenario: A cost measurement is not an assurance claim
- **WHEN** the measurement is published
- **THEN** it is presented as a cost figure, and does not describe the measured obligation as more strongly established than any other
