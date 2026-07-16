## ADDED Requirements

### Requirement: Verdict-authoritative semantic inputs are explicit
Every value capable of changing proof engine selection, observation mapping, soundness, qualifiers, degradation, or composite verdict SHALL be supplied through the proof goal, policy, timeout/fuzz configuration, ordered engine specifications, trusted authorization/mapping-policy version, or typed observation sequence. Verdict code MUST NOT infer such values from process environment, filesystem state, executable discovery, wall clock, or mutable global state. Adapter-provided descriptors MUST NOT grant or strengthen their own trust authorization.

#### Scenario: Equal semantic inputs and observations have equal decisions
- **WHEN** two proof mappings receive equal goals, policies, budgets, engine specifications, mapping versions, and observation sequences
- **THEN** they produce equal semantic decision records

#### Scenario: Different engine is an explicit difference
- **WHEN** a caller replaces one engine specification or configuration fingerprint
- **THEN** the changed semantic identity and any selection difference are attributable to that explicit input

### Requirement: Engine execution uses requests and observations
Pure proof dispatch SHALL NOT invoke solver, FFI, worker, filesystem, clock, or subprocess APIs. It SHALL select a typed `EngineRequest` with a deterministic invocation-local sequence identity and resume a non-cloneable, consumed-by-value suspension with a correlated typed `EngineObservation` returned by an imperative adapter. Observation validation SHALL occur before continuation state changes.

#### Scenario: Supported goal yields a request
- **WHEN** policy selects a configured engine for a supported goal
- **THEN** dispatch yields an engine request rather than executing that engine

#### Scenario: Observation resumes dispatch
- **WHEN** the matching adapter observation is supplied
- **THEN** dispatch validates and maps it before completing or selecting the next fitting engine

#### Scenario: Mismatched observation fails closed
- **WHEN** an observation names a different engine or request identity
- **THEN** dispatch returns a structured protocol failure and no green decision

#### Scenario: Oversized observation fails before mapping
- **WHEN** an observation exceeds the explicit payload bound
- **THEN** dispatch returns a structured protocol failure without changing continuation state

### Requirement: Engine specifications are stable semantic data
Each configured engine SHALL have a serializable and comparable specification containing stable identity, implementation version or digest, supported goal shapes, configuration fingerprint, and transport class. Ordered engine specifications SHALL participate in the pre-execution proof query key. An engine specification MUST NOT contain self-asserted maximum soundness or qualifier authorization.

#### Scenario: Equal specifications select deterministically
- **WHEN** two fitting engine specifications occur in the same explicit order
- **THEN** selection chooses the same first fitting specification

#### Scenario: Engine version changes identity
- **WHEN** an engine implementation version or digest changes
- **THEN** the proof query key changes even if its display name is unchanged

### Requirement: Trusted policy authorizes proof claims
A separately trusted, versioned `EngineAuthorizationPolicy` SHALL map recognized engine fingerprints to maximum soundness, permitted qualifiers, and required evidence validation. Resolvers and adapters MAY provide engine specifications but MUST NOT create or strengthen authorization. Unknown fingerprints SHALL be untrusted and MUST NOT produce a green composite verdict.

#### Scenario: Recognized engine receives bounded authorization
- **WHEN** a descriptor fingerprint is recognized by the active authorization policy and its evidence passes the required validator
- **THEN** common mapping may grant no more than that policy entry's maximum soundness and qualifiers

#### Scenario: Self-authorized or unknown engine fails closed
- **WHEN** an adapter supplies an unknown descriptor or claims a stronger trust ceiling than the trusted policy grants
- **THEN** mapping ignores the self-assertion and produces no unauthorized green verdict

### Requirement: Environment variables cannot fabricate verdicts
Production proof paths SHALL NOT honor an environment variable that substitutes a solver result, including `CHELIS_PROVE_TEST_FORCE_SMT_RESULT`. Tests SHALL obtain outcomes through explicit scripted adapters returning raw observations.

#### Scenario: Hostile forced-result environment is ignored
- **WHEN** forced-result or unrecognized proof-result variables request `Proved`
- **THEN** request selection and observation mapping are identical to the run with those variables absent

#### Scenario: Scripted proved observation is mapped normally
- **WHEN** a test adapter returns a valid proved observation for an authorized engine specification
- **THEN** the common mapper applies the same soundness and composite-verdict rules used for production observations

### Requirement: Solver discovery occurs outside semantic dispatch
Binary paths, solver availability, worker transport, Beacon configuration, and platform discovery SHALL be resolved by an outer adapter into engine specifications and executable adapters. Semantic dispatch MUST NOT probe the host.

#### Scenario: Beacon path is resolved explicitly
- **WHEN** a caller wants a Beacon executable from `CHELIS_BEACON_BIN`
- **THEN** an outer resolver reads it, fingerprints the adapter, and explicitly supplies the resulting engine specification before dispatch

#### Scenario: No configured engine fails closed
- **WHEN** no supplied engine specification supports a required goal
- **THEN** dispatch returns the established non-green no-fit decision without searching the host

### Requirement: Engine selection is deterministic over specifications
Dispatch SHALL evaluate declared support and select engines according to explicit proof policy and specification order. Report timing and process scheduling MUST NOT influence selection.

#### Scenario: First supported engine is selected
- **WHEN** two specifications support the same goal and policy uses registration order
- **THEN** dispatch requests the first one

#### Scenario: Non-verdict selects the next engine
- **WHEN** the first fitting engine's validated observation is timeout, unknown, or another non-verdict and policy allows fallback
- **THEN** dispatch requests the next fitting engine without laundering prior trust metadata

### Requirement: Common mapping owns trust-bearing verdicts
Adapters SHALL return raw observations and MUST NOT directly construct or transmit final soundness, qualifiers, degradation, or composite verdicts. A common pure mapper SHALL validate each observation against the selected engine descriptor and separately trusted authorization policy.

#### Scenario: Authorized observation maps to a verdict
- **WHEN** an engine observation is internally consistent and within its declared trust authorization
- **THEN** the mapper constructs the corresponding decision record

#### Scenario: Invalid evidence fails closed
- **WHEN** an observation's raw status or evidence is inconsistent or fails the validator selected by the trusted authorization policy
- **THEN** common mapping rejects or degrades it and does not construct an unauthorized green verdict

### Requirement: Transport failures fail closed
Unavailable solver, timeout, unknown result, malformed or oversized output, worker crash, subprocess failure, worker hang, correlation mismatch, and replay SHALL map to typed non-verdicts, protocol failures, or host failures before composite status is computed. Semantic rejection, protocol failure, and host execution failure SHALL remain distinct algebraic states. None MAY produce a green verdict.

#### Scenario: Malformed solver output is non-green
- **WHEN** an adapter returns malformed output
- **THEN** mapping records a typed malformed observation and the composite verdict is not green

#### Scenario: Timeout is non-green
- **WHEN** an adapter reports expiration of the explicit timeout budget
- **THEN** mapping records timeout and the composite verdict is not green

#### Scenario: Replayed observation is non-green
- **WHEN** a previously consumed observation is supplied again
- **THEN** mapping rejects it as replayed and does not change the completed decision

### Requirement: Proof decisions are replayable
Chelis SHALL be able to start a fresh dispatcher and replay a recorded normalized observation sequence through pure selection/mapping to reproduce its semantic decision without executing a solver. Replay MUST NOT reuse a consumed live suspension.

#### Scenario: Recorded sequence reproduces decision
- **WHEN** replay receives the original goal, policy, engine specifications, mapping version, and normalized observations
- **THEN** it produces the same decision record

#### Scenario: Tampered replay fails
- **WHEN** a recorded observation's request identity, engine fingerprint, or payload integrity is changed
- **THEN** replay fails closed with a structured mismatch

### Requirement: Query keys, decision digests, and report metadata are distinct
The pre-execution `ProofQueryKey` SHALL contain every authoritative initial input required for cache lookup and SHALL exclude observations. The `ProofDecisionDigest` SHALL include that query key plus the normalized observations and validated evidence used for the decision. Pure mapping SHALL return a decision without reading elapsed time. A measured adapter MAY add `duration_ms` to the existing artifact, but report duration SHALL NOT influence selection, observation mapping, soundness, qualifiers, degradation, either semantic identity, or equality.

#### Scenario: Cache lookup precedes observations
- **WHEN** proof orchestration checks a cache before executing an engine
- **THEN** it computes the query key without requiring an observation that does not yet exist

#### Scenario: Observation changes decision digest, not query key
- **WHEN** equal proof queries receive different normalized observation sequences
- **THEN** their query keys remain equal while their decision digests or decisions may differ

#### Scenario: Different report duration preserves identity
- **WHEN** two artifacts contain equal decisions but different measured durations
- **THEN** they have the same proof query key and decision digest

#### Scenario: Timeout remains an explicit observation
- **WHEN** adapter execution reaches its explicit timeout budget
- **THEN** the resulting timeout observation affects the decision while unrelated report-duration metadata does not

### Requirement: Public proof artifacts remain compatible
The measured adapter SHALL preserve established serialized fields and fail-closed meanings, including nonnegative `duration_ms`, unless a separately versioned compatibility change is approved.

#### Scenario: Existing consumer decodes output
- **WHEN** a refactored proof artifact is serialized
- **THEN** a prior-schema compatibility fixture decodes its established fields

#### Scenario: Duration is present but non-authoritative
- **WHEN** a measured artifact is serialized
- **THEN** `duration_ms` is nonnegative while semantic identity excludes it

### Requirement: Scripted adapters use production mapping rules
Scripted adapters SHALL implement the same request/observation contract as production adapters and SHALL NOT directly construct final verdicts. Any authorization for a scripted engine SHALL come from an explicit test-only policy that production constructors cannot select. Fixtures SHALL cover proved, disproved, timeout, unknown, malformed, oversized, crash, hang, unavailable, mismatched, unauthorized, and replayed observations.

#### Scenario: Complete observation matrix runs
- **WHEN** the adapter contract suite executes
- **THEN** every required success and failure observation passes through common validation and mapping

#### Scenario: Script cannot bypass trust policy
- **WHEN** a scripted adapter supplies a proved observation for an engine descriptor not authorized by the active trusted policy
- **THEN** common mapping rejects or degrades it rather than producing an unauthorized green verdict
