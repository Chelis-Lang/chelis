## Why

Proof verdicts are trust-critical outputs, but current proof code can change a solver result through an ambient test environment variable and includes wall-clock measurements in the same orchestration path. Solver discovery and subprocess transport are also coupled to verdict computation, obscuring which inputs are authoritative.

## What Changes

- Remove `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` from production proof-verdict paths.
- Replace environment-driven forced results with an explicitly injected test solver or scripted engine available through test support APIs.
- Separate pure goal classification, engine-request selection, observation validation, verdict mapping, and artifact construction from solver transport execution.
- Replace effectful engine objects in semantic dispatch with stable engine descriptors, typed execution requests, and typed observations returned by imperative engine adapters.
- Separate adapter-provided engine descriptors from a trusted, versioned authorization policy that alone grants maximum soundness, qualifiers, and evidence-validation rules; unknown descriptors cannot self-authorize a green verdict.
- Make solver availability, binary paths, versions/digests, authorization policy, timeouts, and transport choices explicit orchestration inputs.
- Move elapsed-time measurement to an outer adapter and keep it outside semantic proof decisions. This change introduces no proof-decision cache or persistent query/digest contract; deterministic fresh-machine replay remains test and audit functionality over explicitly supplied non-secret observations.
- Preserve fail-closed timeout, unknown, malformed-output, crash, and unavailable-engine behavior.
- Add negative tests proving ambient environment variables cannot manufacture a `Proved` result.

## Capabilities

### New Capabilities

- `proof-execution-purity`: Defines authoritative proof-verdict inputs, explicit solver adapters, and separation of verdict semantics from environment, clock, and process transport.

### Modified Capabilities

None.

## Impact

This adds dependency-minimal `chelis-prove-core` and affects the `chelis-prove` engine-adapter facade, Tier B solving, engine registration, Beacon integration, worker isolation, proof artifacts, and CLI proof orchestration. Test fixtures that currently force solver outcomes through environment variables will migrate to explicit scripted engine adapters that return raw observations through the same mapping path as production engines under a test-only authorization policy. The independent prerequisite `remove-ambient-proof-result-override` owns the hostile-environment correction and `.venv/bin/python scripts/fcis_gate.py proof-forced-result-removal` oracle. This change retains those cases as regression coverage but does not delay or re-claim that prerequisite's completion. After the standalone `establish-dylint-tooling` oracle passes, `establish-fcis-contract-mechanics` registers stable coverage IDs, exact protocol bounds/result projection, the typed engine/authorization registry, prerequisite edge, boundary/threat model, and fail-closed slice/final oracles.
