## 1. Lock Current Behavior And Failure Cases With Tests

- [ ] 1.1 Add request/observation traces for Reef install, fetch, publish, source-sync, wiring, audit, and doctor plus conformance audit/init/sync/bump
- [ ] 1.2 Add deterministic request-id, matching, stale, replay, mismatch, malformed, oversized, and fresh-workflow replay fixtures
- [ ] 1.3 Add positive/negative query-key and plan-digest tests proving observations are not required for cache lookup and report metadata is excluded
- [ ] 1.4 Add bounded-inline and verified-blob fixtures for digest, size, media type, missing blob, substituted blob, lifetime, and cleanup
- [ ] 1.5 Add whole-plan validation fixtures with malformed first, middle, and final actions and assert zero lock/executor events
- [ ] 1.6 Add lock-then-revalidate fixtures for prerequisite/blob drift and assert zero state-changing mutations
- [ ] 1.7 Add failure injection for lock contention, network/process denial, verification, staging, commit, rollback, and cleanup
- [ ] 1.8 Add credential/secret exclusion and structured-notice/report fixtures for terminal and machine callers
- [ ] 1.9 Add conformance snapshot host-change and hidden-I/O fixtures for every policy-bearing audit class
- [ ] 1.10 Add architecture fixtures for aliases, re-exports, qualified paths, callbacks, trait-hidden capabilities, and `cfg(test)` handling
- [ ] 1.11 Commit all stubs and verify intended hidden-I/O, denial, drift, corruption, replay, blob substitution, and partial-state cases fail before implementation

## 2. Create The Reef Functional Core

- [ ] 2.1 Create dependency-minimal `crates/chelis-reef-core` without reqwest, filesystem, process, environment, credential, lock, clock, or terminal dependencies
- [ ] 2.2 Define immutable local/project/package/conformance snapshots and resolved non-secret host configuration
- [ ] 2.3 Define correlated closed workflow requests, observations, non-cloneable continuation state, integrity metadata, payload bounds, and normalized replay records
- [ ] 2.4 Define `ReefQueryKey` over initial inputs and `ReefPlanDigest` over normalized observations/blobs/final decision
- [ ] 2.5 Define structured semantic rejection, protocol failure, host failure, notices, and reports without contradictory success states
- [ ] 2.6 Move version, pin, origin, dependency, linking, managed-block, and repair policy into the core

## 3. Define Content-Addressed Observation Ownership

- [ ] 3.1 Define bounded inline bytes and `VerifiedBlobRef { digest, size, media_type }`
- [ ] 3.2 Implement explicit blob-store adapter issue/open/retain/release operations without placing ephemeral paths in protocol data
- [ ] 3.3 Validate observed manifest/archive/origin/digest/signature evidence before planning
- [ ] 3.4 Reopen and reverify blob size/digest after lock acquisition and before staging
- [ ] 3.5 Make missing, changed, oversized, substituted, expired, and cleanup-failed blob fixtures fail for the specified reason

## 4. Define Final Plans And Transaction Executors

- [ ] 4.1 Define the closed plan vocabulary for traversal-safe destinations, verified blob/inline content, links, removals, expected prior-state fingerprints, and transaction obligations
- [ ] 4.2 Implement whole-plan validation of every path, digest, blob reference, action, transition, containment rule, and invariant before any executor event
- [ ] 4.3 Implement dry-run rendering from the same validated final plan with no mutation lock or transaction event
- [ ] 4.4 Implement real execution order: validate, acquire lock, revalidate prerequisites/blobs, stage, verify, atomic commit, cleanup/report
- [ ] 4.5 Implement transaction guards for lock ownership, same-filesystem staging/commit, rollback, cleanup, and deterministic recovery reporting
- [ ] 4.6 Implement filesystem, blob, network, process, credential, lock, and mutation adapters that return observations/reports without choosing policy or rendering

## 5. Migrate Read-Only Reef And Conformance Analysis

- [ ] 5.1 Migrate manifest, lockfile, pin, origin, graph, and linking analysis to supplied snapshots
- [ ] 5.2 Migrate Reef audit and doctor to pure analysis or resumable requests
- [ ] 5.3 Add `ConformanceSnapshot` and migrate managed-block, pin, narrowing, skill, workflow, registry, and source-link checks away from direct repository reads
- [ ] 5.4 Keep temporary path adapters clearly outside pure-core evidence and remove them from normal call paths after parity

## 6. Migrate Mutating Command Families Incrementally

- [ ] 6.1 Migrate fetch/install discovery to request/observation workflows followed by final plans
- [ ] 6.2 Migrate publish discovery/upload/process operations to request/observation workflows followed by final plans
- [ ] 6.3 Migrate source sync and wiring one operation at a time
- [ ] 6.4 Migrate conformance init, sync, bump, bump-check reconciliation, and setup integration one operation at a time
- [ ] 6.5 Run focused parity, replay, denial, blob, and transaction gates after each command-family migration
- [ ] 6.6 Remove superseded policy-bearing direct environment, filesystem, network, process, lock, mutation, and reporting calls only after family parity passes

## 7. Preserve Security, Atomicity, And Idempotency

- [ ] 7.1 Enforce advisory locking before prerequisite revalidation and every mutating action; reject unlocked fallback
- [ ] 7.2 Enforce digest, signature, archive, origin, blob, and path-containment verification before staging/commit
- [ ] 7.3 Preserve prior committed state, staging cleanup, rollback reporting, and same-filesystem atomic commit under injected failures
- [ ] 7.4 Ensure failed execution never reports success or leaves a successful partial package installation
- [ ] 7.5 Make repeated conform sync/setup reconciliation produce no state-changing actions after desired state is reached
- [ ] 7.6 Make single-artifact drift repair only that artifact
- [ ] 7.7 Make malicious, corrupt, conflicting, incomplete, drifted, contended, denied, and substituted-blob fixtures fail for the specified reason

## 8. Lock Architecture And Public Parity

- [ ] 8.1 Check in the exact Reef-core, Reef-adapter, conformance-decision, and conformance-adapter manifest
- [ ] 8.2 Reject direct/indirect host capabilities and adapter imports from designated decision modules
- [ ] 8.3 Compare exit status, machine output shape, notices, on-disk formats, dependency/origin/pin decisions, atomicity, and conformance outcomes with established fixtures
- [ ] 8.4 Verify every secret is absent from persistent requests, observations, plans, identities, notices, diagnostics, and replay records

## 9. Documentation And Acceptance

- [ ] 9.1 Update Reef/conformance architecture, replay, identity, blob ownership, package security, dry-run, credentials, lock/revalidation, transaction, recovery, and embedding docs
- [ ] 9.2 Update examples and machine-output fixtures for structured notices without changing established semantics
- [ ] 9.3 Implement `.venv/bin/python scripts/fcis_gate.py reef-workflows`
- [ ] 9.4 Run `.venv/bin/python scripts/fcis_gate.py reef-workflows` and require exit 0, an empty error list, no denied/replay host actions, no dry-run lock/mutation events, and no successful partial state
- [ ] 9.5 Run the repository local gate when implementation begins and record any CI-owned integration evidence required for completion
