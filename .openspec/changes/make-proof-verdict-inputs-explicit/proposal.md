## Why

Proof verdicts are trust-critical outputs, but current proof code can change a solver result through an ambient test environment variable and includes wall-clock measurements in the same orchestration path. Solver discovery and subprocess transport are also coupled to verdict computation, obscuring which inputs are authoritative.

## What Changes

- Remove `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` from production proof-verdict paths.
- Replace environment-driven forced results with an explicitly injected test solver or scripted engine available through test support APIs.
- Separate pure goal classification, engine-request selection, observation validation, verdict mapping, and artifact construction from solver transport execution.
- Replace effectful engine objects in semantic dispatch with stable engine descriptors, typed execution requests, and typed observations returned by imperative engine adapters.
- Separate adapter-provided engine descriptors from a trusted, versioned authorization policy that alone grants maximum soundness, qualifiers, and evidence-validation rules; unknown descriptors cannot self-authorize a green verdict.
- Make solver availability, binary paths, versions/digests, authorization policy, timeouts, and transport choices explicit orchestration inputs.
- Move elapsed-time measurement to an outer adapter and distinguish the pre-execution proof query key, the observation-bearing decision digest, and non-semantic report metadata.
- Preserve fail-closed timeout, unknown, malformed-output, crash, and unavailable-engine behavior.
- Add negative tests proving ambient environment variables cannot manufacture a `Proved` result.

## Capabilities

### New Capabilities

- `proof-execution-purity`: Defines authoritative proof-verdict inputs, explicit solver adapters, and separation of verdict semantics from environment, clock, and process transport.

### Modified Capabilities

None.

## Impact

This affects `chelis-prove` dispatch, Tier B solving, engine registration, Beacon integration, worker isolation, proof artifacts, and CLI proof orchestration. Test fixtures that currently force solver outcomes through environment variables will migrate to explicit scripted engine adapters that return raw observations through the same mapping path as production engines under a test-only authorization policy. The hostile environment test and deletion of the forced-result branch are an immediate security slice and do not wait for the rest of the protocol migration.
