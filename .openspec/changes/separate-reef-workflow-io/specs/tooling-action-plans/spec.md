## ADDED Requirements

### Requirement: Host-dependent tooling decisions use a request protocol
Reef and conformance workflows whose next decision depends on filesystem, registry, package-byte, credential-reference, process, or lock observations SHALL advance through invocation-owned state that returns accepted analysis, semantic rejection, protocol failure, host failure, one correlated host request plus non-cloneable continuation, or a final plan candidate. The FCIS contract manifest SHALL pin exact v1 path/component, request, observation, metadata, inline/chunk/blob-inspection, archive, plan-action, lock-set, transcript, journal, and total-workflow bounds before protocol implementation. Expected not-found, denied-capability, unavailable-registry, child-exit, timeout, and lock-state outcomes SHALL be typed observations when policy interprets them; malformed correlation/kind/bounds/integrity SHALL be protocol failures; unrepresentable adapter breakage SHALL be host failure. Workflow decision code MUST NOT execute requests directly.

#### Scenario: Registry metadata is requested explicitly
- **WHEN** dependency resolution needs registry metadata absent from supplied snapshots
- **THEN** the workflow yields a typed metadata request instead of contacting the network

#### Scenario: Observation changes the next decision
- **WHEN** a correlated metadata response reveals an additional package dependency
- **THEN** the core deterministically yields the next required request or diagnostic from that observation

#### Scenario: Adapter cannot choose package policy
- **WHEN** a network adapter returns package metadata
- **THEN** it does not select versions, origins, pins, or follow-up downloads; those decisions remain in the workflow core

### Requirement: Workflow requests and observations are correlated and replayable
Every host request and observation SHALL carry a deterministic invocation-local sequence identity starting at zero, operation kind, bounded structured data, and integrity context needed for replay. A continuation SHALL be consumed by value. Duplicate, stale, mismatched, malformed, and oversized observations MUST fail closed without advancing state.

#### Scenario: Matching observation resumes workflow
- **WHEN** a workflow receives the matching response to its package-metadata request
- **THEN** it consumes that continuation exactly once and advances deterministically

#### Scenario: Replayed observation is rejected
- **WHEN** an already consumed download observation is submitted again
- **THEN** the workflow rejects it as replayed and performs no mutation

#### Scenario: Recorded workflow replays
- **WHEN** equal initial inputs start a fresh workflow and the original normalized observation sequence is replayed
- **THEN** the workflow produces the same diagnostics or final mutation plan without host access or reuse of a consumed suspension

### Requirement: Host facts and capabilities are explicit
Logical home/cache roots, platform facts, API endpoints, drift policy, external binary descriptors, available capabilities, limits, and opaque credential identifiers SHALL be resolved by outer adapters and passed explicitly. Workflow and plan code MUST NOT inspect process environment. Secret bytes and ephemeral host paths MUST NOT enter persistent requests, observations, plans, identities, notices, diagnostics, or replay records.

#### Scenario: Explicit credential reference enables a request
- **WHEN** resolved configuration contains the opaque credential identifier required for an authenticated operation
- **THEN** the request references that identifier without embedding secret bytes

#### Scenario: Missing credential fails without fallback
- **WHEN** required credentials are absent from resolved configuration
- **THEN** the workflow returns a structured missing-capability diagnostic without consulting environment variables or invoking `gh`

### Requirement: Decision-bearing bytes have explicit content ownership
A package or generated-content adapter SHALL provide bounded inline bytes or an untrusted `BlobCandidateRef` containing claimed digest, size, and media type from an explicit blob store. Adapter issuance MUST NOT make the reference verified. The shell SHALL borrow immutable bytes or deterministic ordered chunks and invoke the core's pure inspection operation, which validates actual size/digest, archive, manifest, origin, signature evidence, and package policy before constructing an opaque `InspectedBlob` and `VerifiedBlobRef`. Workflow decisions MUST accept the core-issued inspected value rather than bare adapter metadata. Persistent workflows and plans MUST NOT refer to ephemeral paths or process-local handles. Executors SHALL reopen and reverify referenced blob size and digest after the ordered lock set is acquired and before staging.

#### Scenario: Candidate bytes are inspected by the core
- **WHEN** a blob-store candidate's borrowed bytes match the claimed identity and pass archive, manifest, origin, signature, and package validation
- **THEN** the pure inspector returns an opaque core-issued `InspectedBlob` that workflow planning may use

#### Scenario: Adapter metadata cannot self-verify
- **WHEN** an adapter returns a candidate reference without supplying bytes or deterministic chunks to the pure inspector
- **THEN** workflow planning rejects it as uninspected and produces no mutation or remote action plan

#### Scenario: Verified blob is staged unchanged
- **WHEN** a plan references a core-issued retained blob whose reopened bytes match its size and digest
- **THEN** the executor may stage those bytes without consulting an ambient source path

#### Scenario: Substituted or missing blob fails closed
- **WHEN** a referenced blob is missing or its reopened size or digest differs
- **THEN** execution performs no state-changing mutation and returns a structured blob failure

### Requirement: Discovery precedes final action planning
A final `LocalMutationPlan` or `RemoteActionPlan` SHALL be produced only after all observations capable of changing dependency, origin, verification, content, path, local transaction, or remote action decisions have been supplied and validated. Reads and downloads with decision-bearing results MUST NOT be represented as blind actions in an upfront plan.

#### Scenario: Downloaded manifest adds a dependency
- **WHEN** package bytes reveal a manifest with another dependency
- **THEN** the workflow requests and resolves that dependency before producing the final mutation plan

#### Scenario: Invalid observed bytes produce no plan
- **WHEN** downloaded bytes fail digest, signature, archive, origin, or manifest validation
- **THEN** the workflow returns structured diagnostics and no mutation plan

### Requirement: Final local plans use a closed mutation vocabulary
A final `LocalMutationPlan` SHALL contain ordered typed mutations and transaction obligations for normalized traversal-safe UTF-8 `/`-separated destinations, verified blob or bounded inline staging content, declared atomicity units, atomic renames/replacements, links, removals, commit, recovery/cleanup, lock scopes, and returned notices as applicable. One typed registry SHALL own request/observation kinds, local actions, lock ranks, transaction states, remote states, and report states. Non-UTF-8, `.`, and `..` destination components SHALL fail before planning. Lock acquisition and prerequisite revalidation SHALL be transaction obligations rather than unconditional file mutations. Unknown actions, ambient source paths, secret bytes, or malformed arguments MUST be rejected.

#### Scenario: Install plan exposes transaction order
- **WHEN** all install observations validate
- **THEN** the final plan makes ordered lock acquisition, stage, per-unit commit, and cleanup/recovery obligations explicit

#### Scenario: Unknown action is rejected
- **WHEN** a plan contains an action kind outside the closed vocabulary
- **THEN** validation rejects the entire plan before action one

### Requirement: Entire plans are validated before execution
Plan validation SHALL check all actions, paths, digests, inspected blob and opaque credential references, containment rules, transitions, prerequisite fingerprints, lock scopes/order, atomicity units or remote preconditions, and local/remote execution invariants before any lock or executor event. A plan failing validation MUST produce zero executor events.

#### Scenario: Late malformed action causes no partial execution
- **WHEN** the final action in a plan has an invalid path or transition
- **THEN** validation rejects the plan before acquiring a lock, writing staging data, or executing any earlier action

#### Scenario: Valid plan passes whole-plan validation
- **WHEN** every action and transition satisfies the closed vocabulary and transaction rules
- **THEN** validation returns an execution-ready plan without performing host effects

### Requirement: Plans carry prerequisite fingerprints
Final local plans SHALL identify the local snapshot, blob references, and other host observations whose continued validity is required for execution plus a complete `LockSet`. Executors SHALL acquire the whole set ordered by `(scope_rank, logical_identity)`, where `toolchain_home < reef_store < project_root < conformance_root`, releasing already acquired locks before retry/failure if the set cannot be completed. They SHALL then revalidate prerequisites and blobs before the first state-changing mutation, fail closed on drift, never acquire a lower-ranked lock while holding a higher-ranked lock, and never proceed unlocked.

#### Scenario: Workspace drifts after dry run
- **WHEN** a relevant manifest or lockfile changes after planning but before execution
- **THEN** execution acquires the ordered lock set, reports prerequisite drift during revalidation, and performs no state-changing mutation

#### Scenario: Lock set cannot be completed
- **WHEN** an executor acquires an earlier-ranked lock but cannot acquire a later required lock under policy
- **THEN** it releases the acquired lock set before waiting, retrying, or failing and performs no mutation

#### Scenario: Unchanged prerequisites permit execution
- **WHEN** all required fingerprints still match
- **THEN** execution may proceed with the validated plan

### Requirement: Dry-run and real execution share the final plan
Dry-run output and real mutation SHALL derive from the same validated final plan. Dry-run MAY perform explicitly authorized observation requests needed to reach the plan, but SHALL NOT acquire mutation locks, enter transaction execution, or mutate state. A later real execution MUST still lock and revalidate prerequisites/blobs.

#### Scenario: Dry-run performs no mutation
- **WHEN** a mutating command completes workflow discovery in dry-run mode
- **THEN** it renders the validated mutation plan without executing lock or mutation actions

#### Scenario: Execution trace matches plan
- **WHEN** the same plan executes against unchanged prerequisites and blobs
- **THEN** the executor trace follows validate, lock, revalidate, stage, verify, commit, and cleanup obligations plus the plan's ordered mutations

### Requirement: Executors preserve transaction and security invariants
Local mutation executors SHALL preserve ordered advisory locking, lock-held prerequisite/blob revalidation, digest/signature checks, archive validation, path containment, staging, and no-silent-fallback behavior. Package/store entry directory renames SHALL be atomic for that entry; individual managed files SHALL use atomic replacement. Plans spanning independent destinations SHALL use a durable journal and deterministic forward/compensating recovery and MUST NOT claim global atomicity. Private-construction typestates or equivalent closed transitions SHALL prevent staging before validation, mutation before complete lock acquisition/revalidation, and complete-success construction from partial/recovery state. A failed verification or mutation MUST NOT leave an operation reported completely successful.

#### Scenario: Verification failure prevents commit
- **WHEN** a required verification fails before commit
- **THEN** no invalid package is committed and execution returns structured failure

#### Scenario: Mid-transaction failure invokes recovery semantics
- **WHEN** execution fails after staging begins but before the first atomicity-unit commit
- **THEN** prior committed state remains valid and staging is cleaned or reported for deterministic recovery

#### Scenario: Failure after one unit commits is not reported atomic
- **WHEN** a multi-destination plan fails after one declared atomicity unit commits
- **THEN** execution returns structured partial/recovery state, follows the journal's recovery policy, and never reports complete success or global rollback

#### Scenario: Lock contention is explicit
- **WHEN** the advisory lock cannot be acquired under configured policy
- **THEN** execution fails or waits according to that policy and never proceeds unlocked

### Requirement: Notices and reports are returned as data
Analysis, workflow, plan validation, and execution SHALL return algebraic accepted, semantic-rejected, protocol-failed, and host-failed values plus notices, diagnostics, observations, and reports with structured codes and context. Successful values MUST NOT contain error diagnostics, rejected values MUST NOT claim completed mutation, protocol/host failures MUST remain distinct, and partial or indeterminate execution MUST NOT appear as complete success. Library code and executors MUST NOT write directly to stdout or stderr. Rendering belongs to caller adapters.

#### Scenario: CLI renders a returned notice
- **WHEN** planning returns a warning notice
- **THEN** the CLI renders it according to output mode while machine callers consume it structurally

#### Scenario: Quiet machine caller emits no terminal output
- **WHEN** a machine caller runs workflow and execution without a renderer
- **THEN** it receives structured data and no terminal output occurs

### Requirement: Executors perform only authorized capabilities
Each request or mutation executor SHALL receive explicit filesystem, network, process, credential, and lock capabilities. A denied or unavailable capability SHALL return a structured observation/error and MUST NOT bypass its adapter through a direct host API.

#### Scenario: Network denial rejects fetch
- **WHEN** a workflow yields a download request and its adapter lacks network capability
- **THEN** it returns capability denial and no package commit occurs

#### Scenario: Process-denied executor handles local plan
- **WHEN** a final plan contains only authorized local filesystem mutations
- **THEN** an executor without process capability can execute it successfully

### Requirement: Conformance policy uses supplied snapshots and plans
Conformance audit, init, sync, bump, bump-check, and setup reconciliation SHALL move policy-bearing repository analysis into designated decision modules over supplied `ConformanceSnapshot` values and SHALL express mutations through the workflow/final-plan boundary. Normal conformance paths MUST NOT retain hidden repository traversal inside policy checks.

#### Scenario: Host change after conformance collection is invisible
- **WHEN** a managed file changes after `ConformanceSnapshot` collection but before pure audit analysis
- **THEN** that analysis reports the collected state and does not reopen the live file

#### Scenario: Conformance repair is adapter-mediated
- **WHEN** sync or bump decides that a managed artifact needs repair
- **THEN** decision code returns a plan action and does not write the artifact directly

### Requirement: Workflow-based commands preserve public behavior
For established Reef and conformance fixtures, request/observation and plan-based commands SHALL preserve exit status, machine output shape, on-disk formats, dependency and pin decisions, documented atomicity units/recovery semantics, and fail-closed behavior except for the approved credential correction: authenticated operations SHALL use explicitly resolved credential references and MUST NOT invoke `gh auth token` or another ambient subprocess fallback.

#### Scenario: Existing install is equivalent
- **WHEN** an established package fixture is processed through the workflow and final-plan executor
- **THEN** its package tree, lock state, notices, and exit status match established behavior

#### Scenario: Existing malicious input remains rejected
- **WHEN** a malicious, corrupt, conflicting, or incomplete fixture is processed
- **THEN** it retains its specified rejection and leaves no successful partial installation

#### Scenario: Missing explicit credential has no gh fallback
- **WHEN** an authenticated operation has no resolved credential reference
- **THEN** it returns the structured missing-capability diagnostic, launches no `gh` process, and performs no remote action

### Requirement: Workflow replay and persistence scope are explicit
A fresh workflow SHALL be able to consume explicitly supplied normalized non-secret observations and core-issued inspected facts for deterministic test/audit replay through the normal validation path without host access. Secret material and ephemeral paths MUST NOT enter replay inputs. This change SHALL NOT introduce `ReefQueryKey`, a general `ReefPlanDigest`, persistent workflow-result caching, or local-plan reuse authorization. Mutable local prerequisites and blobs SHALL always be revalidated under the ordered lock set before mutation. Execution timing, logs, staging paths, cleanup details, and rendering SHALL remain report metadata.

#### Scenario: Explicit workflow replay reproduces a plan
- **WHEN** equal initial inputs and the original normalized non-secret observations are supplied to a fresh workflow
- **THEN** it reproduces the same diagnostics or plan without host access

#### Scenario: Mutable prerequisites are never reused blindly
- **WHEN** an equal local plan is considered for execution after host state may have changed
- **THEN** the executor still acquires the ordered lock set and revalidates every mutable prerequisite and blob

#### Scenario: General workflow cache is not implied
- **WHEN** callers use the workflow protocol introduced by this change
- **THEN** no stable query key, local-plan digest, or final-result reuse contract exists; adding one requires a separate proposal

### Requirement: Remote actions use explicit idempotent plans
A `RemoteActionPlanBody` SHALL contain canonical remote identities, expected remote preconditions, content digests, and ordered irreversible operations. `RemotePlanBodyDigest` SHALL hash only that body with SHA-256 under `chelis-fcis/reef-remote-body/v1\0` using tagged length-prefixed canonical fields. The execution envelope SHALL contain a deterministic idempotency key derived from the body digest under `chelis-fcis/reef-idempotency/v1\0`; neither the key nor execution/report metadata participates in the body digest. Remote execution SHALL return accepted, rejected, or `IndeterminateRemoteState`. It MUST NOT claim local-style rollback after the remote may have accepted an operation, and an indeterminate result SHALL require read-only reconciliation before retry.

#### Scenario: Publish retry uses the same idempotency key
- **WHEN** equal publish inputs and validated observations produce equal remote plans
- **THEN** their remote-plan body digests and derived idempotency keys are equal, retries reuse that key, and no digest/key cycle exists

#### Scenario: Lost publish response is indeterminate
- **WHEN** transport fails after the remote may have accepted publication
- **THEN** execution returns `IndeterminateRemoteState`, reports no complete success or rollback, and requires remote reconciliation before retry

### Requirement: Specified reconciliation operations are idempotent
Operations documented as idempotent, including conform sync and setup reconciliation, SHALL produce no state-changing mutations after desired state is reached.

#### Scenario: Repeated reconciliation is a no-op
- **WHEN** reconciliation executes successfully and workflow/planning runs again over the resulting snapshot
- **THEN** the second final plan contains no state-changing actions

#### Scenario: Drift produces only necessary repair
- **WHEN** one managed artifact drifts after successful reconciliation
- **THEN** replanning repairs that artifact without rewriting unrelated state
