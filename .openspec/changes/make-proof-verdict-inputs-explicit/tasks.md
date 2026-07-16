## 1. Lock Trust-Boundary Behavior With Tests

- [ ] 1.1 Add hostile-environment tests proving forced and unrecognized proof-result variables cannot change request selection or decision mapping
- [ ] 1.2 Add scripted-observation fixtures for proved, disproved, timeout, unknown, malformed, crash, hang, unavailable, mismatched, stale, and replayed outcomes
- [ ] 1.3 Add positive and negative engine-spec ordering, support, version/digest, configuration-fingerprint, recognized-authorization, and unknown-fingerprint tests
- [ ] 1.4 Add fail-closed tests asserting every transport, protocol, oversized-payload, and unauthorized-descriptor failure remains non-green
- [ ] 1.5 Add invalid-observation tests for inconsistent raw status, failed evidence validation, attempted self-authorization, and authorization-policy tampering
- [ ] 1.6 Add replay positive tests and tampered identity/fingerprint/payload negative tests
- [ ] 1.7 Add query-key, decision-digest, cache-lookup-before-observation, duration-normalization, and prior-schema compatibility tests
- [ ] 1.8 Add architecture fixtures for aliases, re-exports, callbacks, trait-hidden capabilities, and `cfg(test)` production-file handling
- [ ] 1.9 Commit all stubs and verify intended hostile, malformed, unauthorized, and tampered cases fail before refactoring

## 2. Define Proof Requests, Observations, And Identity

- [ ] 2.1 Define serializable/comparable `EngineSpec` with identity, version/digest, supported shapes, configuration fingerprint, and transport class but no self-asserted trust
- [ ] 2.2 Define the trusted versioned `EngineAuthorizationPolicy` mapping recognized fingerprints to maximum soundness, qualifiers, and evidence validators
- [ ] 2.3 Define correlated `EngineRequest`, raw `EngineObservation`, protocol/host failure, and non-cloneable single-use continuation types with deterministic sequence identities
- [ ] 2.4 Define explicit proof policy for goal, tier, timeout, fuzz seed/count, authorization/mapping-policy version, and ordered engine specs
- [ ] 2.5 Define a verdict-bearing decision record separate from transport and measurement metadata
- [ ] 2.6 Define `ProofQueryKey` over pre-execution authoritative inputs and `ProofDecisionDigest` over the query key plus normalized observations/evidence

## 3. Build Pure Selection And Mapping

- [ ] 3.1 Implement deterministic engine-spec support selection and try-until-verdict continuation
- [ ] 3.2 Validate deterministic request identity, engine fingerprint, observation kind, stale/replay state, integrity, and payload bounds before mapping or state transition
- [ ] 3.3 Centralize mapping from raw observations to status, soundness, evidence, assumptions, qualifiers, degradation, and composite verdict
- [ ] 3.4 Enforce the separately trusted authorization policy and make unknown/self-authorized descriptors non-green
- [ ] 3.5 Create the exact designated-module manifest and ensure semantic dispatch imports no solver, FFI, worker, process, environment, filesystem, clock, or adapter capability
- [ ] 3.6 Implement observation replay by starting a fresh dispatcher through the same selection and mapping path

## 4. Make Engine Execution Imperative Adapters

- [ ] 4.1 Split cvc5, Z3, Clarabel, Beacon, worker, and solver-free implementations into stable specifications plus executable adapters
- [ ] 4.2 Add outer resolver code for solver binaries, native libraries, worker transports, and implementation fingerprints without granting trust authorization
- [ ] 4.3 Move `CHELIS_BEACON_BIN` resolution to CLI/service boundaries and explicitly register the Beacon spec/adapter pair
- [ ] 4.4 Preserve deterministic specification order across default and optional-feature configurations
- [ ] 4.5 Return the established non-green result when no explicit specification supports the goal
- [ ] 4.6 Add one-minor-version compatibility adaptation for out-of-tree `DischargeEngine` users without using it in canonical semantic dispatch or granting unknown engines production green authorization

## 5. Replace Ambient Test Outcomes

- [ ] 5.1 Implement scripted test adapters through the production request/observation protocol and a production-inaccessible test authorization policy
- [ ] 5.2 Migrate crate tests from forced-result environment mutation to scripted observations
- [ ] 5.3 Migrate CLI, worker-isolation, and Beacon integration fixtures to explicit adapter configuration
- [ ] 5.4 Delete `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` handling from `solve_property` and all production paths
- [ ] 5.5 Make hostile-environment and complete observation-matrix tests pass

## 6. Separate Measurement And Artifact Assembly

- [ ] 6.1 Make semantic mapping return a complete decision record without reading a clock
- [ ] 6.2 Add a measured outer adapter that records elapsed report time and constructs the existing artifact
- [ ] 6.3 Preserve public JSON fields and verify prior consumers decode refactored artifacts
- [ ] 6.4 Update cache lookup to use `ProofQueryKey`; store and compare `ProofDecisionDigest` with normalized observations/evidence while excluding `duration_ms`
- [ ] 6.5 Verify different report durations never alter selection, mapping, soundness, qualifiers, degradation, or verdict

## 7. Harden Failure Mapping Across Lanes

- [ ] 7.1 Route subprocess unavailable, nonzero exit, malformed output, and timeout to typed observations
- [ ] 7.2 Route worker crash and hang to typed observations before common mapping
- [ ] 7.3 Route Beacon and optional in-process solver failures through the same fail-closed mapper
- [ ] 7.4 Make the complete non-green failure matrix pass in default and SMT-enabled lanes

## 8. Documentation And Acceptance

- [ ] 8.1 Update proof architecture, trusted-authorization, environment, Beacon, worker, replay, query-key/decision-digest, engine identity, compatibility-window, and artifact documentation
- [ ] 8.2 Document scripted adapter usage and remove obsolete environment-based fixture guidance
- [ ] 8.3 Implement `.venv/bin/python scripts/fcis_gate.py proof-execution` as the named acceptance runner
- [ ] 8.4 Run `.venv/bin/python scripts/fcis_gate.py proof-execution` in default and SMT-enabled configurations and require exit 0 with an empty error list
- [ ] 8.5 Run the repository local gate and record any CI-owned solver/workspace evidence required for completion
