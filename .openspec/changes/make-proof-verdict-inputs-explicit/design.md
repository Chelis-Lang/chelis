## Context

`chelis-prove` has deterministic goal-shape classification and registration-order routing, but its `DischargeEngine` seam combines engine identity, fitness, transport execution, and final trust-bearing `Discharge` construction. `solve_property` also honors `CHELIS_PROVE_TEST_FORCE_SMT_RESULT`, Beacon registration discovers a binary from the environment, and dispatch mixes decision construction with elapsed measurement.

A timeout or external solver result is an observation of an imperative execution, not a deterministic function of a goal and timeout budget. The FCIS boundary must therefore separate pure request selection and verdict mapping from engine execution rather than hiding execution behind a trait object.

The change keeps existing proof tiers, soundness lattice, qualifier rules, fail-closed behavior, optional-feature lanes, and public JSON fields. It follows the decision, request, identity, and trust laws in `.openspec/FCIS_ARCHITECTURE.md`.

## Goals / Non-Goals

**Goals:**

- Make every input capable of changing engine selection or verdict mapping explicit.
- Prevent environment variables from fabricating proof outcomes.
- Represent engine execution as typed requests and observations at one auditable boundary.
- Give each engine a stable descriptor and semantic fingerprint while keeping trust authorization in a separately trusted, versioned policy.
- Keep selection and mapping deterministic for equal semantic inputs and equal normalized observations.
- Separate the pre-execution query key, observation-bearing decision digest, and measured report identity.
- Separate report measurement from semantic proof identity while preserving fail-closed timeout behavior.

**Non-Goals:**

- Changing mathematical encodings, proof tiers, or soundness classifications.
- Replacing cvc5, Z3, Clarabel, Beacon, or worker isolation.
- Claiming solver execution itself is pure or deterministic.
- Removing elapsed-time reporting from user-facing output.
- Making subprocesses part of the functional core.

## Decisions

### 1. Engine descriptors are data; trusted policy grants authority; adapters execute

Semantic routing receives an ordered list of `EngineSpec` values rather than `Box<dyn DischargeEngine>`. An engine specification contains stable identity, implementation version or digest, supported goal shapes, configuration fingerprint, and transport class. It is comparable and serializable, but it contains no self-asserted soundness or qualifier authorization.

A separately supplied, versioned `EngineAuthorizationPolicy` is part of the trusted proof-mapping implementation. It maps recognized engine family and implementation fingerprints to maximum soundness, permitted qualifiers, required evidence validators, and fallback behavior. Outer resolvers and adapters may construct or discover engine specifications, but cannot add or strengthen authorization. An unknown descriptor is untrusted and cannot produce a green composite verdict. Test-only scripted engines use an explicit test authorization policy unavailable from production constructors.

Pure selection returns an `EngineRequest` naming the selected specification, normalized goal input, explicit timeout budget, and deterministic invocation-local request identity. An imperative adapter registry maps the specification identity to a concrete in-process, subprocess, worker, or scripted executor and returns an `EngineObservation`.

Alternative considered: retain effectful `DischargeEngine::discharge` calls inside semantic dispatch. This was rejected because dependency injection would hide process, timeout, mutable, and FFI effects rather than establishing an FCIS boundary. Allowing a descriptor to declare its own trust ceiling was also rejected because it would let an untrusted adapter authorize itself.

### 2. Dispatch is a resumable request/observation state machine

Proof dispatch advances from goal and policy to one of:

- an accepted `DecisionRecord`;
- a structured semantic rejection;
- an `EngineRequest` plus single-use continuation state.

Host execution and protocol failures remain distinct from semantic rejection. A suspension is consumed by value, is not cloneable, and has exactly one outstanding request. Request identities are deterministic invocation-local sequence numbers starting at zero. When an observation is supplied, dispatch validates request correlation, engine fingerprint, kind, payload bounds, and integrity before changing continuation state, then maps it through common trust rules and either completes or selects the next fitting engine after a non-verdict. Duplicate, stale, mismatched, malformed, or oversized observations fail closed.

Registration-order selection remains explicit and deterministic. Wall-clock scheduling may change which observation an imperative adapter obtains under a timeout, so the determinism guarantee applies to equal semantic inputs plus equal normalized observation sequences, not merely equal solver configurations. Replay starts a new dispatcher and feeds that transcript; it never reuses a consumed suspension.

### 3. Test outcomes use explicit scripted adapters

The environment override is removed from `solve_property`. Tests use scripted adapters that implement the same request/observation transport contract as production adapters. They return raw proved, disproved, timeout, unknown, malformed, crash, hang, and unavailable observations; they do not construct final composite verdicts.

Integration harnesses select scripted adapters and their test-only authorization policy through explicit test configuration, not a production environment variable interpreted by verdict code. The forced-result branch is removed in the first security slice, before the larger engine migration.

### 4. Discovery and transport are imperative adapters

Binary paths, solver availability, worker transport, Beacon configuration, timeout enforcement, and platform discovery occur at CLI/service boundaries. `CHELIS_BEACON_BIN` may be read only by a resolver that constructs an `EngineSpec` and matching adapter explicitly. Resolution can identify a descriptor but cannot grant it trust; authorization remains in `EngineAuthorizationPolicy`.

In-process cvc5, Z3, or Clarabel calls are also engine adapter execution: although computational, they may depend on native libraries, timeouts, crashes, or mutable solver state and therefore do not execute inside pure decision mapping.

### 5. Common mapping owns trust-bearing decisions

Engine observations contain raw status, evidence payload, protocol metadata, and transport classification. They do not contain adapter-selected soundness, qualifiers, degradation, or composite verdict fields. A common pure mapper validates the raw observation against the selected descriptor and the separately trusted authorization policy before constructing those trust-bearing fields.

An engine adapter cannot directly construct a green composite verdict or strengthen its own authority. Evidence that contradicts its raw status, fails the policy-selected validator, or exceeds the authorized engine fingerprint is rejected or degraded fail-closed.

### 6. Query, decision, and report identities are separate

The pre-execution `ProofQueryKey` includes goal, tier policy, timeout/fuzz configuration, ordered engine specifications, implementation versions or digests, authorization/mapping-policy version, and every other initial semantic input. It excludes observations and is the only key used for cache lookup.

The `ProofDecisionDigest` includes the query key plus the normalized observation sequence and validated evidence used to reach the decision. A replay API can feed that recorded sequence through the same pure selector and mapper and reproduce the same `DecisionRecord` without executing a solver. Adapter logging, cache location, terminal rendering, and elapsed duration belong only to report metadata and affect neither identity.

### 7. Decision records are separate from measured artifacts

Pure mapping returns an accepted or rejected decision record containing tier, status, solver status, evidence, assumptions, qualifiers, degradation, selected engine identity, and observation classification. Protocol or host execution failure is represented separately and cannot masquerade as semantic rejection or a completed green record. A measured outer adapter records elapsed time and builds the existing `ProofArtifact`, preserving `duration_ms` for compatibility.

The explicit timeout budget affects engine execution and may produce a timeout observation. Report duration is metadata only and does not affect selection, mapping, soundness, query keys, decision digests, or proof equivalence.

### 8. Exact boundary and trust-boundary tests are mandatory

The designated proof core is the planned `crates/chelis-prove/src/protocol.rs`, `selection.rs`, `mapping.rs`, `identity.rs`, and the existing pure goal, composition, transformation, and artifact-data modules they import. Engine implementations, `solver.rs`, `tier_b.rs` solver calls, `worker.rs`, `beacon_shim.rs`, optional solver/FFI modules, measured artifact assembly, and CLI/service resolution are adapters and may not be imported by those designated modules. The checked-in manifest records the exact transitive production module set and treats `cfg(test)` bodies separately rather than exempting a production file wholesale.

Architecture fixtures reject direct, aliased, re-exported, qualified, callback-hidden, and trait-hidden solver, FFI, worker, process, environment, filesystem, or clock capabilities. Tests run proof orchestration under hostile values for known proof-related environment variables and assert that semantic selection and mapping do not change unless an outer resolver explicitly produces a different descriptor; even then, trust changes only through the trusted authorization policy. Negative fixtures cover unavailable, timeout, unknown, malformed, oversized, crash, hang, observation replay, engine mismatch, unauthorized descriptor, tampered policy reference, and invalid evidence.

## Risks / Trade-offs

- **Risk: The existing public `DischargeEngine` seam is used out of tree.** Mitigation: retain an adapter compatibility layer while making request/observation routing canonical; document deprecation separately.
- **Risk: Engine fingerprints complicate configuration.** Mitigation: provide constructors for built-in engines and require explicit opaque fingerprints from out-of-tree adapters; unknown fingerprints remain untrusted until a separately reviewed authorization-policy change recognizes them.
- **Risk: Integration tests lose convenient environment control.** Mitigation: provide compact scripted adapters and worker fixtures.
- **Risk: Moving measurement changes serialized durations.** Mitigation: preserve the field and compare semantic identity with duration normalized.
- **Risk: Optional feature lanes drift.** Mitigation: retain default, SMT, and configured optional-engine matrices over the same mapper.

## Migration Plan

1. Immediately add hostile-environment negative tests, add explicit scripted fixtures, remove the forced-result branch, and make that focused security gate green.
2. Introduce `EngineSpec`, trusted `EngineAuthorizationPolicy`, `EngineRequest`, `EngineObservation`, deterministic request identity, `ProofQueryKey`, and `ProofDecisionDigest`.
3. Split current engine implementations into descriptors plus imperative adapters without letting descriptors self-authorize trust.
4. Migrate registration-order selection to the resumable request/observation dispatcher.
5. Move Beacon, worker, and solver discovery to outer resolver code.
6. Separate decision records, replay, measurement, cache lookup, and artifact construction while preserving JSON shape.
7. Remove compatibility dispatch paths after default and optional-feature acceptance matrices pass.

Rollback may retain compatibility adapters, but must not restore ambient result substitution or allow adapters to construct final green verdicts directly.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py proof-execution
```

The runner must execute hostile-environment, complete scripted-observation, deterministic identity/correlation/replay, deterministic selection, recognized/unknown authorization, evidence validation, query-key/decision-digest, fail-closed transport, duration-neutrality, and artifact-compatibility matrices in default and SMT-enabled configurations. Success means exit status 0, an empty error list, and no green verdict from an unavailable, malformed, oversized, mismatched, unauthorized, replayed, timed-out, crashed, or hanging observation.

## Resolved Compatibility Decisions

- `duration_ms` remains in its current wire location; moving it requires a separately versioned change.
- The public effectful `DischargeEngine` trait remains for one documented minor-version compatibility window, is adapted only in the imperative shell, and receives no production green authorization unless its descriptor fingerprint is recognized by `EngineAuthorizationPolicy`.
