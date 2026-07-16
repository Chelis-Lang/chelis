## ADDED Requirements

### Requirement: External host effects use a closed request protocol
The evaluator SHALL represent every filesystem, directory, mapped-file-open, and subprocess effect as a typed `HostEffectRequest`. Each request SHALL have a deterministic invocation-local sequence identity starting at zero, identify the originating builtin, carry bounded normalized structured arguments, and include non-secret source and policy context needed for diagnostics. Persistent requests MUST NOT contain secret bytes, renderer configuration, process-local handles, or ephemeral host paths. A live observation MAY transiently carry sensitive payload required by evaluation only when it is marked non-persistable before core consumption.

#### Scenario: File read suspends with a request
- **WHEN** evaluation reaches `read_file`
- **THEN** it yields a correlated file-read request instead of reading the host filesystem

#### Scenario: Unknown effect kind is rejected
- **WHEN** a driver or continuation receives a request kind outside the closed protocol vocabulary
- **THEN** evaluation fails with a structured protocol error rather than silently executing or ignoring it

### Requirement: Evaluation is resumable with single-use explicit state
Semantic evaluation SHALL advance through an invocation-owned machine that returns progress, accepted completion, semantic rejection, protocol failure, host execution failure, or one suspension with a host-effect request and continuation. A suspension SHALL be non-cloneable, carry exactly one outstanding request, and be consumed by value. Resumption SHALL validate a matching response before state transition.

#### Scenario: Matching response resumes evaluation
- **WHEN** the machine is suspended on a file-read request and consumes its matching successful response
- **THEN** evaluation resumes from the saved continuation and uses the returned content

#### Scenario: Mismatched response is rejected
- **WHEN** the machine is suspended on a directory-list request and receives a subprocess response
- **THEN** it returns a structured protocol error without corrupting evaluator state

#### Scenario: Replayed response is rejected
- **WHEN** a response for an already consumed suspension is submitted again
- **THEN** it is rejected as stale or replayed and causes no additional evaluator or host action

#### Scenario: Recorded transcript replays on a fresh machine
- **WHEN** equal initial inputs are supplied to a new evaluator machine with the original normalized observations in order
- **THEN** it reproduces the same outcome without reusing a consumed suspension or performing a host action

#### Scenario: Failed run does not contaminate another run
- **WHEN** one evaluator machine fails after effect suspensions
- **THEN** a separate machine produces the same result it would have produced in isolation

### Requirement: Every host-capable language builtin has one evaluator policy
One typed executable registry SHALL classify every evaluator builtin carrying a current or planned host-capable language effect—including `IO` today and `Filesystem` or `Network` when introduced—as exactly one of: external host request, evaluator-local captured event, or explicit unsupported policy, and SHALL own request/response kind, persistability, and target support. The gate SHALL compare builtin registration, declared language effect, evaluator classification, protocol mapping, captured-event mapping, and target policy. Independent string lists are not authoritative. Missing and multiply classified builtins MUST fail the consistency gate.

#### Scenario: Print is a captured event
- **WHEN** evaluation reaches `print`
- **THEN** it appends a deterministic transcript event and performs no terminal write or external host request

#### Scenario: New host-capable builtin lacks a policy
- **WHEN** a fixture registers or reclassifies an `IO`, `Filesystem`, `Network`, or later host-capable builtin without an external-request, captured-event, or unsupported classification
- **THEN** the consistency gate fails and names the builtin and declared effect

### Requirement: All external effectful evaluation is adapter-mediated
No semantic evaluator path in dependency-minimal `chelis-eval-core`, including nested functions, callbacks, transforms, tests, imported definitions, or mapped-file open, SHALL directly obtain filesystem, process, environment, network, clock, terminal, entropy, thread-scheduling, unsafe-FFI, or mutable-global capabilities. The core SHALL NOT accept a callback or trait that can expose those capabilities. A supplied outer adapter SHALL be the only component that executes external requests, and the architecture gate SHALL state the exact dependency checks and fixture classes it proves without claiming arbitrary macro/dynamic-dispatch completeness.

#### Scenario: Nested callback remains mediated
- **WHEN** an effectful builtin is invoked inside a nested function or callback
- **THEN** the outer evaluator machine yields the corresponding request

#### Scenario: Denying adapter cannot be bypassed
- **WHEN** evaluation under a deny-all policy reaches an external effect through any supported call form
- **THEN** it returns a policy-denied diagnostic and performs no host operation

### Requirement: Evaluation execution context is explicit
Semantic `EvaluationPolicy`, adapter `HostExecutionContext`, and non-semantic `EvaluationRenderPolicy` SHALL be separate. `EvaluationPolicy` SHALL contain operation authorization, lexical path-policy identity, evaluator fuel, mapped-snapshot size, and requested subprocess timeout/output bounds; the latter SHALL be copied into subprocess requests because changing them may change observations. The FCIS contract manifest SHALL pin exact v1 maxima for path bytes, file read/write bytes, directory entries/name bytes, request count, observation bytes, transcript bytes, subprocess argv entries/bytes, mapped snapshots, and total evaluator-owned protocol memory before protocol implementation. `HostExecutionContext` SHALL contain resolved relative-path base, directory handles, subprocess cwd, authorized resolved environment values, and executor safety ceilings. An executor ceiling stricter than the request SHALL return a distinct host failure and MUST NOT masquerade as a semantic timeout/output-limit observation. V1 SHALL use 10,000,000 fuel steps and a 268,435,456-byte mapped-snapshot ceiling for trusted CLI evaluation. Redaction configuration SHALL be report metadata. Semantic evaluation MUST NOT inspect process cwd or environment.

#### Scenario: Relative path uses explicit base
- **WHEN** a request contains a relative file path
- **THEN** policy and execution resolve it against the supplied base directory rather than ambient process cwd

#### Scenario: Subprocess environment is restricted
- **WHEN** policy supplies an environment allowlist that omits a process variable
- **THEN** the subprocess request executes without inheriting that variable

#### Scenario: Fuel exhaustion is deterministic
- **WHEN** evaluation consumes the v1 default budget of 10,000,000 semantic steps or a caller-supplied smaller budget
- **THEN** it returns the same structured limit error independent of wall-clock speed

#### Scenario: Redaction configuration is non-semantic
- **WHEN** only renderer redaction or terminal configuration changes
- **THEN** the request sequence, machine outcome, and semantic identity remain unchanged

### Requirement: Standard adapters have defined behavior
Chelis SHALL provide a production OS adapter, a deterministic in-memory adapter, and a deny-all adapter. The production adapter SHALL preserve specified trusted CLI behavior; the in-memory adapter SHALL access only supplied fixtures; and the deny-all adapter SHALL reject every external request.

#### Scenario: In-memory file round trip is deterministic
- **WHEN** a program writes and reads a path through an in-memory adapter initialized with fixed state and context
- **THEN** its value and final adapter state are deterministic and no host file changes

#### Scenario: Production operation fails normally
- **WHEN** the production adapter receives a valid request whose OS operation fails
- **THEN** it returns a typed host error mapped to the specified user-facing diagnostic

### Requirement: Handler errors remain structured and redacted
Not-found, permission/OS failure, policy denial, timeout, output-limit exhaustion, mapped-snapshot-limit exhaustion, malformed request, malformed response, correlation failure, oversized payload, and unsupported capability SHALL remain distinct error classes. Errors SHALL include operation identity and structured path or command context without exposing configured secrets. Sensitive payload MAY exist transiently in a live response only when marked non-persistable; it SHALL NOT enter persistent requests or observations, normalized transcripts, diagnostics, identities, or logs.

#### Scenario: Policy denial differs from missing file
- **WHEN** one read is denied and another reaches the OS for a missing file
- **THEN** their structured error classes differ

#### Scenario: Secret material is not rendered
- **WHEN** an adapter error contains configured credential material
- **THEN** the user-facing diagnostic omits or redacts that material

### Requirement: Mapped files use snapshot semantics
A successful mapped-file-open response SHALL return an evaluator-owned immutable byte snapshot no larger than the selected explicit ceiling. Trusted CLI v1 SHALL use 268,435,456 bytes. A larger file SHALL return `mapped_snapshot_limit` before evaluator-owned allocation and SHALL NOT produce a partial snapshot. Subsequent `mmap_len` and `mmap_read` operations SHALL be pure operations over the accepted snapshot and SHALL NOT reread the host file. `mmap_read` SHALL reject offsets greater than snapshot length and SHALL clamp a requested length that extends beyond the snapshot end, preserving current behavior.

#### Scenario: Host mutation after open is invisible
- **WHEN** the host file changes after a mapped-file-open response supplies its bytes
- **THEN** later mapped reads observe the original snapshot

#### Scenario: Oversized mapped file is rejected before allocation
- **WHEN** mapped-file open observes a file larger than the selected mapped-snapshot ceiling
- **THEN** it returns `mapped_snapshot_limit`, allocates no evaluator snapshot, and issues no successful partial observation

#### Scenario: Out-of-range mapped offset performs no I/O
- **WHEN** `mmap_read` requests an offset greater than the snapshot length
- **THEN** evaluation returns the existing bounds diagnostic without issuing another request

#### Scenario: Mapped length beyond end clamps
- **WHEN** `mmap_read` starts at a valid offset but its requested length extends beyond the snapshot
- **THEN** it returns bytes through the snapshot end without issuing another request

### Requirement: Filesystem policy is containment-aware
Path authorization SHALL define normalization, relative-base handling, symlink traversal, and nonexistent-write parent policy. Contained production policies SHALL authorize and access through directory-capability-relative operations rather than canonicalizing one path and reopening another. A path escaping an authorized root MUST fail closed, and a platform lacking required race-resistant containment MUST reject the contained policy.

#### Scenario: Escaping symlink is denied
- **WHEN** an authorized lexical path resolves through a symlink outside the allowed root
- **THEN** the request is denied and no external bytes are read or written

#### Scenario: Valid contained write is allowed
- **WHEN** a write path and its resolved parent satisfy the configured containment policy
- **THEN** the production adapter may perform the write

### Requirement: Subprocess execution is capability-controlled and bounded
`process_run` SHALL yield a subprocess request containing executable and argv without shell interpolation plus policy-selected requested timeout and output limits. The production adapter SHALL resolve cwd and authorized environment values from `HostExecutionContext` and SHALL NOT alter the requested limits except by returning a distinct executor-ceiling host failure. Trusted CLI policy defaults SHALL request a 300-second timeout and 64 MiB limit for each output stream. Policies SHALL be able to deny subprocesses independently from filesystem requests.

#### Scenario: Argument vector is preserved
- **WHEN** arguments contain spaces or shell metacharacters
- **THEN** the request preserves them as argument values and no shell interprets them

#### Scenario: Process-only denial is enforced
- **WHEN** policy allows filesystem requests but denies subprocess requests
- **THEN** file operations can succeed while `process_run` returns policy denial

#### Scenario: Process timeout is bounded
- **WHEN** a child exceeds the explicit timeout
- **THEN** the adapter terminates or reaps it according to policy and returns a timeout observation

#### Scenario: Process output limit is bounded
- **WHEN** stdout or stderr exceeds its explicit limit
- **THEN** the adapter returns an output-limit observation without unbounded memory growth

### Requirement: Embedding defaults are explicit and secure
Trusted CLI compatibility entry points SHALL select the production adapter and trusted-local v1 defaults explicitly. Before protocol migration, Tide SHALL pass deny-external mode into builtin dispatch so file, directory, mapped-open, and process builtins fail before any host call through every call form while captured `print`/`debug` events remain available. Tide SHALL deny filesystem and subprocess requests by default under the final protocol. Python `eval_json` SHALL deny external requests by default, `eval_json_with_policy` SHALL accept explicit capabilities, and only `eval_json_unrestricted` SHALL select trusted-local unrestricted behavior.

#### Scenario: Tide default denies file access
- **WHEN** an unconfigured Tide evaluation reaches `read_file`
- **THEN** it returns policy denial and reads no host file

#### Scenario: Tide still captures local output events
- **WHEN** unconfigured Tide evaluation reaches only `print` or `debug`
- **THEN** it returns the deterministic captured events without a terminal or external host action

#### Scenario: Explicit Tide capability enables access
- **WHEN** Tide is configured with a filesystem policy authorizing a contained path through a supported directory capability
- **THEN** evaluation may service requests for that path while other paths remain denied

#### Scenario: Ordinary Python eval denies host access
- **WHEN** `eval_json` evaluates a program that reaches `read_file` or `process_run`
- **THEN** it returns policy denial and performs no host operation

### Requirement: Directory and existence behavior is deterministic and structured
Directory observations SHALL be normalized to bytewise UTF-8 entry-name order before becoming runtime values. `file_exists` SHALL return `false` only for not-found and SHALL preserve permission or other OS failures as structured errors.

#### Scenario: Directory enumeration order is canonical
- **WHEN** equal directory entries are returned in different host enumeration orders
- **THEN** evaluation produces the same bytewise name-sorted runtime list

#### Scenario: Permission failure is not absence
- **WHEN** an existence query fails because access is denied
- **THEN** evaluation returns a permission/OS failure rather than `false`

### Requirement: Evaluation replay and persistence scope are explicit
Fresh-machine replay SHALL accept only explicitly supplied normalized, non-secret observations, validate them through the normal correlation/policy path, and perform no host action. Secret-bearing or non-persistable observations MAY be consumed for a live run but MUST NOT enter replay fixtures or persistent records. This change SHALL NOT introduce `EvaluationQueryKey`, `EvaluationOutcomeDigest`, a persistent transcript store, or final-result cache semantics.

#### Scenario: Explicit non-secret replay is deterministic
- **WHEN** a fresh machine receives equal initial inputs and the original normalized non-secret observations
- **THEN** it reproduces the same decision without host access

#### Scenario: Secret-bearing observation is ephemeral
- **WHEN** an adapter observation contains content classified non-persistable or secret-bearing
- **THEN** evaluation may consume it for the live run but stores no persistent transcript or outcome digest containing or hashing that content

#### Scenario: Persistent result cache is not implied
- **WHEN** callers use the evaluator protocol introduced by this change
- **THEN** no stable hash, persistent transcript, or final-result reuse contract exists; adding one requires a separate proposal

### Requirement: Effect registries remain consistent
An executable consistency gate SHALL jointly check builtin registration, every current or planned host-capable language effect declaration, evaluator policy classification, request/response mapping, captured-event mapping, and compiled-target policy. It SHALL remain complete when the active roadmap introduces `Filesystem` or `Network` rather than silently checking only `IO`.

#### Scenario: Complete policy mapping passes
- **WHEN** every host-capable builtin has exactly one evaluator policy and explicit target policy consistent with its declared language effect
- **THEN** the consistency gate passes

#### Scenario: Duplicate policy mapping fails
- **WHEN** a fixture classifies one builtin as both a captured event and external request
- **THEN** the consistency gate fails and names both classifications
