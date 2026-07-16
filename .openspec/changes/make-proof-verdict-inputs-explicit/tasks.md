## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, forced-result-removal prerequisite edge, exact core/adapter/test-support boundaries, stable requirement/scenario IDs, explicit polarity, and complete fixture-to-slice-to-final-oracle traces in the FCIS contract manifest
- [ ] 0.2 Pin exact v1 request, observation, evidence, transcript, engine-count, and lane-specific transport payload bounds; absent bounds block protocol implementation
- [ ] 0.3 Register one typed engine order/spec/adapter-binding/authorization owner using full structural `EngineAuthorizationKey`, with optional-feature and public-compatibility tripwires
- [ ] 0.4 Register the exact raw-status/policy/evidence/protocol/host failure projection table and private-construction constraints for authorized engines and decision records
- [ ] 0.5 Make `.venv/bin/python scripts/fcis_gate.py proof-execution --slice protocol` exist and fail on current effectful-dispatch, unauthorized, malformed, replay, and ambient fixtures
- [ ] 0.6 Pass the owning FCIS contract registration/traceability slice before proof protocol implementation

## 1. Lock Trust-Boundary Behavior With Tests

- [ ] 1.1 Add hostile-environment tests proving forced and unrecognized proof-result variables cannot change request selection or decision mapping
- [ ] 1.2 Add scripted-observation fixtures for proved, disproved, timeout, unknown, malformed, crash, hang, unavailable, mismatched, stale, and replayed outcomes
- [ ] 1.3 Add positive and negative engine-spec ordering, support, version/digest, configuration-fingerprint, transport, full-descriptor recognized-authorization, changed-field, and unknown-fingerprint tests
- [ ] 1.4 Add fail-closed tests asserting every transport, protocol, oversized-payload, and unauthorized-descriptor failure remains non-green
- [ ] 1.5 Add invalid-observation tests for inconsistent raw status, failed evidence validation, attempted self-authorization, and authorization-policy tampering
- [ ] 1.6 Add replay positive tests and tampered identity/fingerprint/payload negative tests
- [ ] 1.7 Add fresh-machine audit replay, tampered-observation denial, duration-normalization, explicit no-persistent-cache surface, and prior-artifact-schema compatibility tests
- [ ] 1.8 Add `chelis-prove-core` dependency-boundary tests and specifically claimed alias, re-export, callback, macro, trait, entropy, scheduling, unsafe-FFI, mutable-global, and `cfg(test)` fixtures with documented completeness limits
- [ ] 1.9 Commit all stubs and verify intended hostile, malformed, unauthorized, and tampered cases fail before refactoring
- [ ] 1.10 Require recorded green evidence from `.venv/bin/python scripts/fcis_gate.py proof-forced-result-removal` in the independent `remove-ambient-proof-result-override` change
- [ ] 1.11 Retain its hostile-environment and production-source absence cases as regression inputs without reimplementing or re-claiming that security correction

## 2. Define Proof Requests, Observations, And Identity

- [ ] 2.1 Create dependency-minimal `chelis-prove-core` and define serializable/comparable `EngineSpec` with identity, version/digest, supported shapes, configuration fingerprint, and transport class but no self-asserted trust
- [ ] 2.2 Define the trusted versioned `EngineAuthorizationPolicy` mapping full canonical descriptor fingerprints to maximum soundness, qualifiers, evidence validators, and freshness/revocation rules
- [ ] 2.3 Define correlated `EngineRequest`, raw `EngineObservation`, protocol/host failure, and non-cloneable single-use continuation types with deterministic sequence identities
- [ ] 2.4 Define explicit proof policy for goal, tier, timeout, fuzz seed/count, authorization/mapping-policy version, and ordered engine specs
- [ ] 2.5 Define a verdict-bearing decision record separate from transport and measurement metadata
- [ ] 2.6 Define decision-record structural equality and explicit audit replay while exposing no `ProofQueryKey`, `ProofDecisionDigest`, persistent observation store, or proof-result cache contract
- [ ] 2.7 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py proof-execution --slice protocol`

## 3. Build Pure Selection And Mapping

- [ ] 3.1 Implement deterministic engine-spec support selection and try-until-verdict continuation
- [ ] 3.2 Validate deterministic request identity, engine fingerprint, observation kind, stale/replay state, integrity, and payload bounds before mapping or state transition
- [ ] 3.3 Centralize mapping from raw observations to status, soundness, evidence, assumptions, qualifiers, degradation, and composite verdict
- [ ] 3.4 Enforce the separately trusted authorization policy and make unknown/self-authorized descriptors non-green
- [ ] 3.5 Enforce the `chelis-prove-core` Cargo dependency allowlist and architecture threat model; ensure semantic dispatch imports no solver, FFI, worker, process, environment, filesystem, network, clock, terminal, entropy, scheduling, mutable-global, or adapter capability
- [ ] 3.6 Implement observation replay by starting a fresh dispatcher through the same selection and mapping path
- [ ] 3.7 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py proof-execution --slice dispatch`

## 4. Make Engine Execution Imperative Adapters

- [ ] 4.1 Split cvc5, Z3, Clarabel, Beacon, worker, and solver-free implementations into stable specifications plus executable adapters
- [ ] 4.2 Add outer resolver code for solver binaries, native libraries, worker transports, and implementation fingerprints without granting trust authorization
- [ ] 4.3 Move `CHELIS_BEACON_BIN` resolution to CLI/service boundaries and explicitly register the Beacon spec/adapter pair
- [ ] 4.4 Preserve deterministic specification order across default and optional-feature configurations
- [ ] 4.5 Return the established non-green result when no explicit specification supports the goal
- [ ] 4.6 Add one-minor-version compatibility adaptation for out-of-tree `DischargeEngine` users without using it in canonical semantic dispatch or granting unknown engines production green authorization
- [ ] 4.7 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py proof-execution --slice adapters` in default and optional-feature lanes

## 5. Extend Explicit Test Outcomes Through The Protocol

- [ ] 5.1 Extend the prerequisite's raw scripted test support through the production request/observation protocol and a test-target-only authorization policy unavailable from production constructors/features
- [ ] 5.2 Migrate protocol-specific crate tests from direct raw fixtures to scripted engine observations
- [ ] 5.3 Migrate CLI, worker-isolation, and Beacon protocol fixtures to explicit adapter configuration
- [ ] 5.4 Verify architecture and hostile-environment fixtures keep `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` and equivalent ambient substitutions absent from every production path
- [ ] 5.5 Make hostile-environment regression and complete observation-matrix tests pass

## 6. Separate Measurement And Artifact Assembly

- [ ] 6.1 Make semantic mapping return a complete decision record without reading a clock
- [ ] 6.2 Add a measured outer adapter that records elapsed report time and constructs the existing artifact
- [ ] 6.3 Preserve public JSON fields and verify prior consumers decode refactored artifacts
- [ ] 6.4 Implement explicit non-secret audit replay through a fresh dispatcher and verify that the production surface introduces no persistent proof-result cache or stable digest contract
- [ ] 6.5 Verify different report durations never alter selection, mapping, soundness, qualifiers, degradation, or verdict
- [ ] 6.6 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py proof-execution --slice artifacts`

## 7. Harden Failure Mapping Across Lanes

- [ ] 7.1 Route subprocess unavailable, nonzero exit, malformed output, and timeout to typed observations
- [ ] 7.2 Route worker crash and hang to typed observations before common mapping
- [ ] 7.3 Route Beacon and optional in-process solver failures through the same fail-closed mapper
- [ ] 7.4 Make the complete non-green failure matrix pass in default and SMT-enabled lanes

## 8. Documentation And Acceptance

- [ ] 8.1 Update proof architecture, trusted authorization, environment, Beacon, worker, explicit audit replay, no-persistent-cache scope, engine identity, compatibility-window, and artifact documentation
- [ ] 8.2 Document scripted adapter usage and remove obsolete environment-based fixture guidance
- [ ] 8.3 Consolidate the already landed focused commands and implement the final `.venv/bin/python scripts/fcis_gate.py proof-execution` acceptance runner
- [ ] 8.4 Run each focused command before landing its slice; do not claim completion until the final oracle passes default and SMT-enabled configurations with exit 0, an empty error list, fresh-machine replay validation, no persistent proof cache/digest surface, and no green result from invalid protocol/host outcomes
- [ ] 8.5 Run the repository local gate and record any CI-owned solver/workspace evidence required for completion
