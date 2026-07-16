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
- Keep measured report metadata separate from semantic proof decisions without introducing a speculative persistent decision-cache identity.
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

A separately supplied, versioned `EngineAuthorizationPolicy` is part of the trusted proof-mapping implementation. It maps an exact typed structural `EngineAuthorizationKey`—covering engine family, implementation digest, canonically ordered supported shapes, configuration fingerprint, and transport class—to maximum soundness, permitted qualifiers, required evidence validators, and fallback behavior. Authorization lookup compares the full structural key; an adapter-provided or display-oriented hash cannot authorize a descriptor. If the policy/key is serialized, its schema and canonical field encoding are versioned and golden-tested, but this does not introduce a proof-decision cache identity. Outer resolvers and adapters may construct or discover engine specifications, but cannot add or strengthen authorization. A descriptor with any unrecognized or changed authority-bearing field is untrusted and cannot produce a green composite verdict. Test-only scripted engines use an explicit test authorization policy compiled only into test-support targets and unavailable from production constructors or production features.

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

Integration harnesses select scripted adapters and their test-only authorization policy through explicit test configuration, not a production environment variable interpreted by verdict code. The independent prerequisite `remove-ambient-proof-result-override` removes the forced-result branch and must pass `.venv/bin/python scripts/fcis_gate.py proof-forced-result-removal` before this protocol change begins.

### 4. Discovery and transport are imperative adapters

Binary paths, solver availability, worker transport, Beacon configuration, timeout enforcement, and platform discovery occur at CLI/service boundaries. `CHELIS_BEACON_BIN` may be read only by a resolver that constructs an `EngineSpec` and matching adapter explicitly. Resolution can identify a descriptor but cannot grant it trust; authorization remains in `EngineAuthorizationPolicy`.

In-process cvc5, Z3, or Clarabel calls are also engine adapter execution: although computational, they may depend on native libraries, timeouts, crashes, or mutable solver state and therefore do not execute inside pure decision mapping.

### 5. Common mapping owns trust-bearing decisions

Engine observations contain raw status, evidence payload, protocol metadata, and transport classification. They do not contain adapter-selected soundness, qualifiers, degradation, or composite verdict fields. A common pure mapper validates the raw observation against the selected descriptor and the separately trusted authorization policy before constructing those trust-bearing fields.

An engine adapter cannot directly construct a green composite verdict or strengthen its own authority. Evidence that contradicts its raw status, fails the policy-selected validator, or exceeds the authorized engine fingerprint is rejected or degraded fail-closed.

### 6. Replay is explicit, but persistent proof caching is deferred

A fresh dispatcher can consume an explicitly supplied normalized, non-secret observation sequence for deterministic tests and audit. Replay validates request identity, descriptor fingerprint, payload bounds, integrity, evidence, and the current authorization/mapping policy through the same path as live dispatch. It never reuses a consumed suspension.

This change introduces no `ProofQueryKey`, `ProofDecisionDigest`, persistent observation store, or proof-result cache. Engine descriptors and decision records remain comparable semantic data, but no stable cross-version hash encoding or result-reuse authorization is implied. Adding persistent proof caching requires a separate proposal covering evidence retention, freshness, revocation, confidentiality, and compatibility.

### 7. Decision records are separate from measured artifacts

Pure mapping returns an accepted or rejected decision record containing tier, status, solver status, evidence, assumptions, qualifiers, degradation, selected engine identity, and observation classification. Protocol or host execution failure is represented separately and cannot masquerade as semantic rejection or a completed green record. A measured outer adapter records elapsed time and builds the existing `ProofArtifact`, preserving `duration_ms` for compatibility.

The explicit timeout budget affects engine execution and may produce a timeout observation. Report duration is metadata only and does not affect selection, mapping, soundness, or decision-record equality.

### 8. Exact boundary and trust-boundary tests are mandatory

The designated proof core is a planned dependency-minimal `crates/chelis-prove-core` crate containing protocol, selection, mapping, goal, composition, transformation, replay, and decision-record data. `chelis-prove` remains the engine-adapter facade containing solver implementations, `solver.rs`, `tier_b.rs` solver calls, `worker.rs`, `beacon_shim.rs`, optional solver/FFI modules, measured artifact assembly, and CLI/service resolution. `chelis-prove-core` cannot depend on `chelis-prove`, solver/FFI/process crates, or capability-bearing callbacks/traits; the facade may re-export compatibility data types during the migration window.

Cargo dependency allowlists are the primary architecture boundary. Through contract mechanics, the accepted `establish-dylint-tooling` prerequisite's pinned `chelis-fcis-boundaries` library consumes the manifest's proof-core/adapter boundary and default/optional package/library/test/feature lanes. It rejects resolved direct, alias, re-export, qualified, function-item, callback, trait, declarative-macro-expanded, and adapter-module references for solver, FFI, worker, process, environment, filesystem, network, clock, terminal, entropy, thread-scheduling, unsafe-FFI, or mutable-global capabilities, plus public callback/function-pointer/unsealed-trait or capability-handle inputs. Unresolved configured entries, zero matched production-core items, or omitted declared lanes fail closed. UI fixtures classify production-in-test-build code separately from true `cfg(test)` owners.

The gate records exact pins, compilation lanes, resolved entries, matched production counts, typed detector contracts, fix policies, and blind spots. Every claimed detector form has allowed-positive and violating-negative evidence before use; each domain registration is `NoFix` and may reference a machine-applicable fix only after a separately accepted Dylint-tooling change supports that fix ID and `FCIS-EVIDENCE-011` validates disposable exact-rewrite/no-machine-fix, compile/lint, proof-trust/failure-parity, idempotence, and checkout-immutability plans. It does not claim coverage of unexecuted cfg/target/feature code, procedural-macro/build-script implementation effects, arbitrary dynamic dispatch, precompiled dependency behavior, or future Rust syntax. Tests run proof orchestration under hostile values for known proof-related environment variables and assert that semantic selection and mapping do not change unless an outer resolver explicitly produces a different descriptor; even then, trust changes only through the trusted authorization policy. Negative behavioral fixtures cover unavailable, timeout, unknown, malformed, oversized, crash, hang, observation replay, engine mismatch, unauthorized descriptor, tampered policy reference, and invalid evidence.

### 9. Engine authorization, payload bounds, and result projection have typed owners

One typed engine catalog owns built-in engine order, exact `EngineSpec`, adapter binding identity, optional-feature availability, and the trusted authorization entry keyed by `EngineAuthorizationKey`. Registration order and feature projections are deterministic. Public compatibility `DischargeEngine` values are adapted only in the shell and cannot insert an authorization entry; unknown structural keys remain non-green.

`EngineObservation` is a raw closed enum that cannot contain soundness, qualifiers, degradation, or composite verdict fields. `AuthorizedEngine` and accepted `DecisionRecord` have private constructors available only to common policy validation/mapping. Suspensions are non-cloneable and consumed by value. Test scripted adapters and their authorization policy live in dev-only targets with no production dependency or constructor path.

The FCIS manifest pins exact v1 request, observation, evidence, transcript, engine-count, and per-engine transport payload bounds for each default/optional lane and owns the exact projection table from raw statuses, policy denial, malformed/correlation failure, timeout/unavailable/crash/hang, evidence failure, protocol failure, and host failure to decision/failure variants. Protocol implementation is blocked while a bound or projection is absent.

Before implementation, stable requirement/scenario/fixture IDs, the engine/authorization registry, exact core/adapter boundary, prerequisite regression edge, and failing `proof-execution --slice protocol` runner are registered in the FCIS manifest.

## Risks / Trade-offs

- **Risk: The existing public `DischargeEngine` seam is used out of tree.** Mitigation: retain an adapter compatibility layer while making request/observation routing canonical; document deprecation separately.
- **Risk: Engine fingerprints complicate configuration.** Mitigation: provide constructors for built-in engines and require explicit opaque fingerprints from out-of-tree adapters; unknown fingerprints remain untrusted until a separately reviewed authorization-policy change recognizes them.
- **Risk: Integration tests lose convenient environment control.** Mitigation: provide compact scripted adapters and worker fixtures.
- **Risk: Moving measurement changes serialized durations.** Mitigation: preserve the field and compare semantic identity with duration normalized.
- **Risk: Optional feature lanes drift.** Mitigation: retain default, SMT, and configured optional-engine matrices over the same mapper.

## Migration Plan

1. Require the independently completed `remove-ambient-proof-result-override` oracle and retain its hostile-environment cases as regression inputs.
2. Introduce `EngineSpec`, trusted `EngineAuthorizationPolicy`, `EngineRequest`, `EngineObservation`, deterministic request identity, and fresh-machine replay without persistent proof caching; pass `--slice protocol`.
3. Split current engine implementations into descriptors plus imperative adapters without letting descriptors self-authorize trust; pass `--slice adapters` in default and optional-feature lanes.
4. Migrate registration-order selection to the resumable request/observation dispatcher and pass `--slice dispatch`.
5. Move Beacon, worker, and solver discovery to outer resolver code.
6. Separate decision records, explicit audit replay, measurement, and artifact construction while preserving JSON shape; pass `--slice artifacts`.
7. Remove compatibility dispatch paths only after focused gates and the default/optional-feature final acceptance matrices pass.

Rollback may retain compatibility adapters, but must not restore ambient result substitution or allow adapters to construct final green verdicts directly. Focused slice commands are supporting evidence, not completion oracles.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py proof-execution
```

The runner must execute hostile-environment, manifest-configured pinned Dylint plans, positive/negative detector fixtures, and registered disposable fix/no-fix/proof-parity/idempotence fixtures for every declared default/optional proof-core lane, complete scripted-observation, deterministic request identity/correlation/fresh-machine replay, deterministic selection, full-descriptor recognized/unknown authorization, evidence validation, fail-closed transport, duration-neutrality, no-persistent-cache surface checks, and artifact-compatibility matrices in default and SMT-enabled configurations. Success means exit status 0, an empty error list, no persistent proof-result identity introduced, and no green verdict from an unavailable, malformed, oversized, mismatched, unauthorized, invalid-evidence, replayed, timed-out, crashed, or hanging observation.

## Resolved Compatibility Decisions

- `duration_ms` remains in its current wire location; moving it requires a separately versioned change.
- The public effectful `DischargeEngine` trait remains for one documented minor-version compatibility window, is adapted only in the imperative shell, and receives no production green authorization unless its descriptor fingerprint is recognized by `EngineAuthorizationPolicy`.
