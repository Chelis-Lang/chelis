## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, explicit-credential prerequisite edge, exact Reef/conformance core/adapter boundaries, stable requirement/scenario IDs, explicit polarity, and complete fixture-to-slice-to-final-oracle traces in the FCIS contract manifest
- [ ] 0.2 Register the command-family capability matrix and the shared no-policy-I/O-bypass cross-slice invariant; every focused family remains non-final evidence
- [ ] 0.3 Pin exact v1 path/component, request, observation, metadata, inline/chunk/blob inspection, archive, action, lock-set, transcript, journal, and total-workflow bounds; absent bounds block protocol implementation
- [ ] 0.4 Register one typed request/action/lock-rank/transaction/remote/report-state owner plus the exact semantic/protocol/host/partial/indeterminate projection table
- [ ] 0.5 Register private-construction/typestate fixtures for candidate/inspected/verified blobs and candidate/validated/locked/revalidated/staged plans
- [ ] 0.6 Make `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice protocol` exist and fail on current hidden-I/O, self-verified-blob, invalid-plan, lock-order, partial, and indeterminate fixtures
- [ ] 0.7 Pass the owning FCIS contract registration/traceability slice before Reef protocol implementation

## 1. Lock Current Behavior And Failure Cases With Tests

- [ ] 1.1 Add request/observation traces for Reef install, fetch, publish, source-sync, wiring, audit, and doctor plus conformance audit/init/sync/bump
- [ ] 1.2 Add deterministic request-id, matching, stale, replay, mismatch, malformed, oversized, and fresh-workflow replay fixtures
- [ ] 1.3 Add positive/negative fresh-workflow replay and no-general-cache-surface tests plus remote-plan body-digest/idempotency golden vectors proving the key is excluded from its source digest and report metadata is excluded
- [ ] 1.4 Add blob-candidate/core-inspection fixtures for claimed/actual digest, size, media type, deterministic chunking, archive/manifest/origin/signature inspection, missing/substituted blob, lifetime, and cleanup
- [ ] 1.5 Add whole-local/remote-plan validation fixtures with malformed first, middle, and final actions and assert zero lock/executor events
- [ ] 1.6 Add ordered-lock-set fixtures for scope ranking, same-rank logical ordering, partial-acquisition release, prerequisite/blob drift, and zero pre-revalidation mutations
- [ ] 1.7 Add local failure injection for lock contention, network/process denial, verification, staging, each atomicity-unit commit, journal recovery, compensation, and cleanup
- [ ] 1.8 Add remote publish fixtures for deterministic idempotency keys, accepted/rejected/indeterminate results, lost responses, reconciliation-before-retry, and no false rollback
- [ ] 1.9 Add explicit-credential and secret-exclusion fixtures proving missing credentials never invoke `gh auth token` and secret bytes never enter persistent records or logs
- [ ] 1.10 Add structured-notice/report fixtures for terminal and machine callers, including partial local recovery and indeterminate remote state
- [ ] 1.11 Add conformance snapshot host-change and hidden-I/O fixtures for every policy-bearing audit class
- [ ] 1.12 Add path fixtures for normalized UTF-8 `/` destinations and rejected non-UTF-8, `.`, `..`, absolute, and escaping components
- [ ] 1.13 Add architecture fixtures for aliases, re-exports, qualified paths, callbacks, trait-hidden capabilities, unsafe FFI, entropy/scheduling, and `cfg(test)` handling
- [ ] 1.14 Commit all stubs and verify intended hidden-I/O, denial, drift, corruption, replay, adapter-self-verification, blob substitution, lock-order, remote-indeterminate, and partial-state cases fail before implementation

## 2. Create The Reef Functional Core And Protocol

- [ ] 2.1 Create dependency-minimal `crates/chelis-reef-core` without reqwest, filesystem, process, environment, credential, lock, clock, terminal, entropy, thread-scheduling, or unsafe-FFI dependencies
- [ ] 2.2 Define immutable local/project/package/conformance snapshots and resolved non-secret host configuration
- [ ] 2.3 Define correlated closed workflow requests, observations, semantic rejection, protocol failure, host failure, non-cloneable continuation state, integrity metadata, payload bounds, and normalized replay records
- [ ] 2.4 Define explicit non-secret fresh-workflow replay without `ReefQueryKey` or general `ReefPlanDigest`, plus domain-separated canonical `RemotePlanBodyDigest` and derived idempotency-key encodings with no cycle
- [ ] 2.5 Define structured notices and reports without contradictory complete-success, partial, indeterminate, or error states
- [ ] 2.6 Move version, pin, origin, dependency, linking, managed-block, repair, remote retry, and indeterminate-state policy into the core
- [ ] 2.7 Define separate `LocalMutationPlan` and `RemoteActionPlan` algebras
- [ ] 2.8 Implement `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice protocol` and make it green before command migration

## 3. Define Content-Addressed Observation Ownership

- [ ] 3.1 Define bounded inline bytes, untrusted `BlobCandidateRef`, opaque core-issued `InspectedBlob`, and `VerifiedBlobRef { digest, size, media_type }`
- [ ] 3.2 Implement explicit blob-store issue/open/retain/release operations without placing ephemeral paths or process-local handles in protocol data
- [ ] 3.3 Implement pure direct/chunked inspection that computes actual size/digest and validates archive, manifest, origin, signature evidence, and package policy before constructing `InspectedBlob`
- [ ] 3.4 Make workflow planning reject bare adapter references and accept only core-issued inspected facts
- [ ] 3.5 Reopen and reverify blob size/digest after ordered lock acquisition and before staging
- [ ] 3.6 Make missing, changed, oversized, substituted, uninspected, expired, and cleanup-failed blob fixtures fail for the specified reason

## 4. Preserve The Explicit-Credential Prerequisite

- [ ] 4.1 Require recorded green evidence from `.venv/bin/python scripts/fcis_gate.py reef-explicit-credentials` in the independent `require-explicit-reef-credentials` change
- [ ] 4.2 Accept only the prerequisite's resolved opaque credential references in workflow requests without secret bytes in observations, plans, digests, diagnostics, notices, or logs
- [ ] 4.3 Retain its no-`gh`, hostile-`PATH`, configured-authentication, anonymous-public, and secret-exclusion corpus as workflow regression inputs
- [ ] 4.4 Do not reintroduce helper/environment fallback or re-claim the prerequisite's security completion under a focused workflow slice

## 5. Migrate Read-Only Reef And Conformance Analysis

- [ ] 5.1 Migrate manifest, lockfile, pin, origin, graph, and linking analysis to supplied snapshots
- [ ] 5.2 Migrate Reef audit and doctor to pure analysis or resumable requests
- [ ] 5.3 Add `ConformanceSnapshot` and migrate managed-block, pin, narrowing, skill, workflow, registry, and source-link checks away from direct repository reads
- [ ] 5.4 Keep temporary path adapters clearly outside pure-core evidence and remove them from normal call paths after parity
- [ ] 5.5 Run `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice read-only`

## 6. Define Local Plans, Ordered Locks, And Recovery

- [ ] 6.1 Define the closed local vocabulary for normalized UTF-8 relative destinations, verified blob/inline content, links, removals, expected prior-state fingerprints, lock sets, atomicity units, journals, and recovery obligations
- [ ] 6.2 Implement whole-plan validation of every path, digest, inspected blob reference, action, transition, containment rule, lock scope/order, atomicity unit, and invariant before any executor event
- [ ] 6.3 Implement dry-run rendering from the same validated local plan with no lock or transaction event
- [ ] 6.4 Implement real execution order: validate, acquire the complete canonical lock set, revalidate prerequisites/blobs, stage, verify, commit by declared atomicity unit, cleanup/recovery report
- [ ] 6.5 Implement package/store-entry atomic directory rename, managed-file atomic replacement, and durable journal/recovery for multi-destination plans without claiming global atomicity
- [ ] 6.6 Implement filesystem, blob, process, credential, lock, and local-mutation adapters that return observations/reports without choosing policy or rendering
- [ ] 6.7 Migrate fetch/install discovery and execution to workflows plus local plans
- [ ] 6.8 Run `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice local-install`

## 7. Migrate Conformance And Source Wiring Incrementally

- [ ] 7.1 Migrate conformance init, sync, bump, bump-check reconciliation, and setup integration one operation at a time through local plans
- [ ] 7.2 Make multi-artifact conformance repair use explicit atomicity units and journal recovery, with no false whole-repository atomicity claim
- [ ] 7.3 Run `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice conformance`
- [ ] 7.4 Migrate source sync and wiring one operation at a time through local plans
- [ ] 7.5 Run `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice source-wiring`
- [ ] 7.6 Remove superseded policy-bearing direct environment, filesystem, process, lock, mutation, and reporting calls only after each family gate passes

## 8. Model Remote Publication Separately

- [ ] 8.1 Define `RemoteActionPlanBody` with canonical remote identities, expected preconditions, content digests, and ordered irreversible operations; hash the body excluding the key, then derive the idempotency key under a separate domain
- [ ] 8.2 Implement a remote executor that returns accepted, rejected, or `IndeterminateRemoteState` without claiming rollback
- [ ] 8.3 Require read-only remote reconciliation before retry after an indeterminate response and reuse the same idempotency key for an equal plan
- [ ] 8.4 Migrate publish discovery/upload operations to request/observation workflows followed by remote plans
- [ ] 8.5 Run `.venv/bin/python scripts/fcis_gate.py reef-workflows --slice remote-publish`

## 9. Preserve Security, Scoped Atomicity, And Idempotency

- [ ] 9.1 Enforce complete ordered lock acquisition before prerequisite revalidation and every local mutating action; reject unlocked or lock-order fallback
- [ ] 9.2 Enforce digest, signature, archive, origin, inspected-blob, and path-containment verification before staging/commit
- [ ] 9.3 Preserve prior committed state before first commit and deterministic journal recovery after partial multi-unit commit
- [ ] 9.4 Ensure failed local execution never reports complete success and indeterminate remote execution never reports success or rollback
- [ ] 9.5 Make repeated conform sync/setup reconciliation produce no state-changing actions after desired state is reached
- [ ] 9.6 Make single-artifact drift repair only that artifact
- [ ] 9.7 Make malicious, corrupt, conflicting, incomplete, drifted, contended, denied, uninspected/substituted-blob, partial-local, and indeterminate-remote fixtures fail for the specified reason

## 10. Lock Architecture And Public Parity

- [ ] 10.1 Enforce the `chelis-reef-core` dependency allowlist and check in the Reef-adapter/conformance decision/adapter manifest plus the architecture-gate threat model, specifically proven fixture classes, and known blind spots
- [ ] 10.2 Reject the documented direct, alias, re-export, qualified, callback, trait, macro, and `cfg(test)` host-capability fixtures and adapter imports without claiming arbitrary macro/dynamic-dispatch completeness
- [ ] 10.3 Compare exit status, machine output shape, notices, on-disk formats, dependency/origin/pin decisions, declared atomicity/recovery behavior, remote state, and conformance outcomes with established fixtures
- [ ] 10.4 Verify every secret is absent from persistent requests/observations, plans, identities, notices, diagnostics, logs, and replay records

## 11. Documentation And Acceptance

- [ ] 11.1 Update Reef/conformance architecture, explicit replay/no-general-cache scope, remote-body digest/idempotency encoding, blob inspection/ownership, package security, dry-run, explicit credentials, lock ordering, scoped atomicity, journal recovery, remote indeterminate state, and embedding docs
- [ ] 11.2 Update examples and machine-output fixtures for structured notices, partial recovery, missing explicit credentials, and indeterminate publication
- [ ] 11.3 Implement every focused command and the final `.venv/bin/python scripts/fcis_gate.py reef-workflows` acceptance oracle
- [ ] 11.4 Run each focused command before landing its family; do not claim change completion until the final oracle exits 0 with an empty error list, no denied/replay host actions, no general workflow-cache surface, no remote digest/key cycle, no dry-run lock/mutation events, no adapter-self-verified blob planning, no lock-order violations, and no false complete-success/rollback report
- [ ] 11.5 Run the repository local gate when implementation begins and record any CI-owned integration evidence required for completion
