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
Pure proof dispatch SHALL NOT invoke solver, FFI, worker, filesystem, network, clock, terminal, entropy, thread-scheduling, mutable-global, or subprocess APIs. It SHALL select a typed `EngineRequest` with a deterministic invocation-local sequence identity and resume a non-cloneable, consumed-by-value suspension with a correlated typed `EngineObservation` returned by an imperative adapter. The FCIS contract manifest SHALL pin exact v1 request, observation, evidence, transcript, engine-count, and lane-specific transport bounds before protocol implementation. Observation validation SHALL occur before continuation state changes.

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
Each configured engine SHALL have a serializable and comparable specification containing stable identity, implementation version or digest, supported goal shapes, configuration fingerprint, and transport class. Ordered engine specifications SHALL be authoritative semantic inputs to selection and decision-record equality. An engine specification MUST NOT contain self-asserted maximum soundness or qualifier authorization.

#### Scenario: Equal specifications select deterministically
- **WHEN** two fitting engine specifications occur in the same explicit order
- **THEN** selection chooses the same first fitting specification

#### Scenario: Engine version changes identity
- **WHEN** an engine implementation version or digest changes
- **THEN** descriptor equality, authorization lookup, and any resulting selection difference reflect the changed implementation even if its display name is unchanged

### Requirement: Trusted policy authorizes proof claims
A separately trusted, versioned `EngineAuthorizationPolicy` SHALL map an exact typed structural `EngineAuthorizationKey` covering engine family, implementation digest, canonically ordered supported goal shapes, configuration fingerprint, and transport class to maximum soundness, permitted qualifiers, required evidence validation, and any freshness/revocation rule. Authorization MUST compare the full structural key; an adapter-provided or display-oriented hash MUST NOT grant authority. If serialized, the key/policy schema and canonical field encoding SHALL be versioned and golden-tested without implying proof-decision caching. Resolvers and adapters MAY provide engine specifications but MUST NOT create or strengthen authorization. Any unrecognized authority-bearing descriptor field SHALL make the descriptor untrusted and MUST NOT produce a green composite verdict.

#### Scenario: Recognized engine receives bounded authorization
- **WHEN** a descriptor fingerprint is recognized by the active authorization policy and its evidence passes the required validator
- **THEN** common mapping may grant no more than that policy entry's maximum soundness and qualifiers

#### Scenario: Self-authorized or unknown engine fails closed
- **WHEN** an adapter supplies an unknown descriptor, changes an authorized descriptor's configuration/transport/support field, or claims a stronger trust ceiling than the trusted policy grants
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

### Requirement: Replay, measurement, and persistence scope are explicit
Pure mapping SHALL return a decision without reading elapsed time. A measured adapter MAY add `duration_ms` to the existing artifact, but report duration SHALL NOT influence selection, observation mapping, soundness, qualifiers, degradation, or decision-record equality. Fresh-machine replay SHALL accept explicitly supplied normalized non-secret observations and validate them through current selection, correlation, evidence, and authorization rules. This change SHALL NOT introduce `ProofQueryKey`, `ProofDecisionDigest`, a persistent observation store, or proof-result cache semantics.

#### Scenario: Different report duration preserves the decision
- **WHEN** two artifacts contain equal decisions but different measured durations
- **THEN** their semantic decision records remain equal

#### Scenario: Timeout remains an explicit observation
- **WHEN** adapter execution reaches its explicit timeout budget
- **THEN** the resulting timeout observation may affect fallback or the decision while unrelated report-duration metadata does not

#### Scenario: Explicit audit replay validates current policy
- **WHEN** a caller supplies the original non-secret normalized observations to a fresh dispatcher under the current authorization policy
- **THEN** replay validates and reproduces the decision without executing a solver

#### Scenario: Persistent proof cache is not implied
- **WHEN** callers use the proof protocol introduced by this change
- **THEN** no stable hash, persistent evidence-retention, or final-result reuse contract exists; adding one requires a separate proposal

### Requirement: Public proof artifacts remain compatible
The measured adapter SHALL preserve established serialized fields and fail-closed meanings, including nonnegative `duration_ms`, unless a separately versioned compatibility change is approved.

#### Scenario: Existing consumer decodes output
- **WHEN** a refactored proof artifact is serialized
- **THEN** a prior-schema compatibility fixture decodes its established fields

#### Scenario: Duration is present but non-authoritative
- **WHEN** a measured artifact is serialized
- **THEN** `duration_ms` is nonnegative while semantic identity excludes it

### Requirement: Scripted adapters use production mapping rules
Scripted adapters SHALL implement the same request/observation contract as production adapters and SHALL NOT directly construct final verdicts. Any authorization for a scripted engine SHALL come from an explicit test-only policy compiled only into test-support targets and unavailable from production constructors or production features. Fixtures SHALL cover proved, disproved, timeout, unknown, malformed, oversized, crash, hang, unavailable, mismatched, unauthorized, stale-evidence, and replayed observations.

#### Scenario: Complete observation matrix runs
- **WHEN** the adapter contract suite executes
- **THEN** every required success and failure observation passes through common validation and mapping

#### Scenario: Script cannot bypass trust policy
- **WHEN** a scripted adapter supplies a proved observation for an engine descriptor not authorized by the active trusted policy
- **THEN** common mapping rejects or degrades it rather than producing an unauthorized green verdict
