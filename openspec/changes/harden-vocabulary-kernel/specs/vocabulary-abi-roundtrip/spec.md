# Vocabulary ABI Round-Trip: harden-vocabulary-kernel

## ADDED Requirements

### Requirement: The runtime dtype tag encoding is total, injective, and exhaustively rejecting
The mapping between a runtime dtype and its ABI tag value SHALL be total and injective, and the decoder SHALL reject every tag value outside the closed set. These SHALL hold for the entire domain of the tag's representation, not only for sampled values.

#### Scenario: Every dtype round-trips
- **WHEN** any runtime dtype is encoded to its ABI tag and decoded back
- **THEN** the result is that same dtype

#### Scenario: No two dtypes share a tag
- **WHEN** the tag values of the closed dtype set are compared pairwise
- **THEN** no two are equal

#### Scenario: Every non-tag value is rejected
- **WHEN** the decoder receives any value of the tag's representation that is not the tag of some dtype
- **THEN** it reports an invalid-tag rejection carrying that value, for every such value rather than for a sampled one

#### Scenario: Rejection is not substitution
- **WHEN** the decoder rejects a value
- **THEN** it returns no dtype, and does not fall back to a default dtype

### Requirement: The tag properties are established by exhaustive check over the whole domain
The round-trip, injectivity, and exhaustive-rejection properties SHALL be established over the entire domain of the tag's representation by enumeration, not by sampling. The check SHALL cover every value the representation admits, and SHALL be part of the repository's ordinary test surface.

#### Scenario: Every representable tag value is checked
- **WHEN** the exhaustive check runs
- **THEN** it evaluates the decoder at every value of the tag's representation, and no value is sampled or skipped

#### Scenario: A sampled probe does not satisfy the requirement
- **WHEN** a test establishes rejection by probing a chosen unknown value
- **THEN** it does not by itself satisfy this requirement, because the property is universally quantified

#### Scenario: A new variant reopens the check
- **WHEN** a variant is added to the closed dtype set
- **THEN** the exhaustive check must pass for the extended set rather than continuing to pass unchanged

### Requirement: A mechanized proof of an already-established property is justified as such
Where a machine-checked proof establishes a property that a cheaper oracle already establishes, the proof's purpose SHALL be recorded as establishing or measuring the prover lane, and SHALL NOT be recorded as making the property available. Any proof SHALL operate on a model extracted mechanically from the shipped source, so that the proved model and the executed code cannot diverge silently.

#### Scenario: The stated purpose matches the situation
- **WHEN** a proof covers a property the exhaustive check also covers
- **THEN** the documentation states that the proof establishes the lane, and does not present the property as obtainable only by proof

#### Scenario: The model is extracted, not restated
- **WHEN** the proof is checked
- **THEN** the model it reasons about was produced mechanically from the crate's source, and no hand-written transcription of the implementation stands between them

#### Scenario: Drift fails the gate
- **WHEN** the implementation changes such that a proved property no longer holds
- **THEN** the gate that runs the extraction and the proof fails

#### Scenario: Declining the lane does not remove the guarantee
- **WHEN** the prover lane is declined
- **THEN** the tag properties remain established by the exhaustive check, and no guarantee is lost

### Requirement: The generated decoder is checked against the Rust decoder
The decoder emitted into generated code SHALL be checked for agreement with the implementation it is generated from, by executing both rather than by comparing their source text. The check SHALL cover every valid tag and a set of invalid ones, and SHALL require that the two agree on acceptance, on the decoded result, and on rejection.

#### Scenario: Both decoders are executed
- **WHEN** agreement is checked
- **THEN** both decoders are executed over the inputs, and agreement is not inferred from the generated text matching a checked-in artifact

#### Scenario: Every valid tag agrees
- **WHEN** each valid tag is decoded by both
- **THEN** they accept it and agree on the result, including the associated byte width

#### Scenario: Invalid tags are rejected by both
- **WHEN** an invalid tag is decoded by both
- **THEN** both reject it, and neither silently substitutes a default

#### Scenario: A divergence fails the check
- **WHEN** the generated decoder is altered so that it accepts a value the implementation rejects
- **THEN** the check fails

### Requirement: The verified subset is stated explicitly
The change SHALL state which items of the crate are covered by mechanized proof and which are not. A claim that the crate is verified SHALL NOT be made when only part of it is. Items outside the verified subset SHALL remain covered by their existing tests, and the reason for their exclusion SHALL be recorded.

#### Scenario: The boundary is documented
- **WHEN** a reader asks which parts of the vocabulary crate are proved
- **THEN** the covered items are listed, and the excluded items are listed with the reason for exclusion

#### Scenario: Excluded items keep their coverage
- **WHEN** an item is outside the verified subset
- **THEN** its existing test coverage is retained rather than removed in favor of the proof

#### Scenario: No overclaim in status text
- **WHEN** documentation or a phase summary describes this work
- **THEN** it describes the proved scope as the numeric ABI core, and does not describe the crate as verified

### Requirement: The honesty requirements are mechanically enforced
The requirements governing what may be claimed SHALL be checked by an automated gate, not by review alone. A documentation statement asserting a guarantee stronger than what is established SHALL fail that gate. A statement the requirements oblige the documentation to make SHALL fail the gate by its absence. The gate SHALL itself be tested.

#### Scenario: An overclaiming sentence fails the gate
- **WHEN** documentation asserts a guarantee broader than the established scope, such as describing the whole crate as verified
- **THEN** the gate fails and names the offending text

#### Scenario: A missing required statement fails the gate
- **WHEN** a statement of the verified-subset boundary or of a named residual is removed from the documentation
- **THEN** the gate fails

#### Scenario: An unfinished proof cannot report success
- **WHEN** a proof source contains an admission token, or a required theorem is absent, renamed, or weakened to a vacuous statement
- **THEN** the gate fails rather than reporting the property established

#### Scenario: Escape hatches are enumerated, not open
- **WHEN** a proof source uses a construct the gate bans, such as an assumed declaration
- **THEN** the gate fails unless that exact declaration appears in a reviewed allowlist with a recorded reason

#### Scenario: The gate is itself tested
- **WHEN** the enforcing checker is exercised
- **THEN** it has tests establishing that each planted violation is detected, so a silently inert checker cannot pass as enforcement

### Requirement: Each evidence mechanism is reported at its own strength
Where more than one mechanism produces evidence about the vocabulary crate, each SHALL be reported with the question it answers and SHALL NOT be aggregated with another into a single score or badge. A mechanism that establishes a property for every input SHALL NOT be averaged with one that establishes only that code executed, or that a test would notice a change. Where one mechanism's guarantee is strictly weaker than another's over the same items, the weaker one SHALL NOT be reported for those items.

#### Scenario: Results are not combined into one number
- **WHEN** proof, coverage, and mutation results are published for the crate
- **THEN** each is reported separately and labelled with what it establishes, and no combined score is presented

#### Scenario: Coverage is not reported over proved items
- **WHEN** a property has been proved for every input of an item
- **THEN** an execution-coverage figure for that same item is not published alongside it as if it added assurance

#### Scenario: Mutation outcomes keep their categories
- **WHEN** mutation results are reported
- **THEN** caught, missed, unviable, timeout, and baseline-failure counts are distinct, and no category is folded into a success rate that obscures it

#### Scenario: An absent mechanism is explained
- **WHEN** a mechanism a reader might expect is not applied
- **THEN** the reason is recorded, so its absence is not read as an oversight

### Requirement: Residual unproved boundaries are named
Where a proved property stops short of a boundary a reader would reasonably assume it covers, the change SHALL name that residual explicitly rather than leaving the stronger reading available.

#### Scenario: The generated decoder is named as unproved
- **WHEN** the proof establishes that the Rust decoder rejects every non-tag value
- **THEN** the documentation records that the separately generated decoder in emitted C is not covered by that proof, and states what evidence does cover it
