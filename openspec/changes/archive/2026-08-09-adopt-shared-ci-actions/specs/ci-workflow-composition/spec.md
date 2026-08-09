## ADDED Requirements

### Requirement: [CWC-001] Adopted ci surfaces use one immutable revision
Chelis SHALL pin the ci Devenv input and every adopted `Chelis-Lang/ci` action to one reviewed full commit SHA.

The adopted set SHALL include these actions:

- `setup-devenv`
- `authenticate-private-ci-input`
- `reclaim-ubuntu-runner-disk`
- `nightly-status`
- `check-authorship`
- `openspec-governance`
- `actionlint`
- `zizmor`
- `prune-actions-cache`

Adjacent action comments SHALL identify the reviewed ci change or upstream human-readable label. An adopted ci action MUST NOT use a branch, tag, or different commit.

#### Scenario: All consumer pins converge
- **WHEN** workflow and Devenv contract tests read the configured ci references
- **THEN** each reference contains the same 40-character commit SHA
- **AND** the Devenv lock resolves the configured ci input to that SHA

#### Scenario: One action retains an older commit
- **WHEN** any adopted action reference differs from the configured ci input revision
- **THEN** the workflow contract test fails and names that action

#### Scenario: A mutable action reference appears
- **WHEN** any adopted action uses a branch or tag
- **THEN** the workflow contract test fails before hosted execution

### Requirement: [CWC-002] Private-input authentication is composed once per job
Every job that evaluates the private ci Devenv input SHALL invoke `setup-devenv` before `authenticate-private-ci-input`.

The authentication step SHALL receive `vars.CI_APP_ID` as `app-client-id`. It SHALL receive `secrets.CI_APP_PRIVATE_KEY` as `app-private-key`.

Such jobs MUST NOT invoke `actions/create-github-app-token` directly. They MUST NOT write a ci token into `NIX_CONFIG` through an inline command.

Jobs that do not evaluate the private input do not need this authentication action.

#### Scenario: A Devenv job evaluates the private input
- **WHEN** a workflow job runs a project Devenv command
- **THEN** setup precedes fixed-scope authentication
- **AND** authentication precedes the first project Devenv command

#### Scenario: An inline token block returns
- **WHEN** a workflow adds direct App-token creation or token-bearing `GITHUB_ENV` text
- **THEN** the workflow contract test fails

#### Scenario: Authentication precedes setup
- **WHEN** a job reverses the two shared actions
- **THEN** the workflow contract test fails and names the invalid order

### Requirement: [CWC-003] Nightly workflows use the shared status transition
Each nightly reporter SHALL pass its upstream result to `nightly-status`. It SHALL preserve its exact issue title, body, label, and recovery comment.

Chelis SHALL retain each workflow trigger, upstream job graph, same-repository issue permission, and reporter condition.

A cancelled or skipped upstream result SHALL remain a no-op. Failure SHALL open or reuse one stable issue, and success SHALL close that issue after recovery.

#### Scenario: A nightly run fails
- **WHEN** one migrated upstream nightly job reports failure
- **THEN** the shared action opens or reuses the workflow's exact stable issue

#### Scenario: A nightly run recovers
- **WHEN** the upstream result changes from failure to success
- **THEN** the shared action posts the configured recovery comment and closes the issue

#### Scenario: A workflow changes issue identity
- **WHEN** a migration changes the title, label, or recovery text
- **THEN** the workflow contract test fails

#### Scenario: A run is cancelled
- **WHEN** the upstream result is cancelled or skipped
- **THEN** the reporter performs no issue transition

### Requirement: [CWC-004] The required authorship context preserves policy parity
The job name `No AI authorship markers` SHALL remain unchanged. The job SHALL retain pull-request and push range semantics from its existing required check.

The job SHALL invoke `check-authorship` with `profile: all-markers`. Shared parity fixtures SHALL cover every positive marker from the removed scanner.

The job MUST NOT retain an inline marker loop after shared adoption. Checkout depth and explicit base and head selection SHALL remain Chelis policy.

#### Scenario: A pull request runs the required check
- **WHEN** the workflow receives a pull-request event
- **THEN** the action scans the explicit base-exclusive and head-inclusive range
- **AND** the required context name remains unchanged

#### Scenario: A push runs the required check
- **WHEN** the workflow receives a push event
- **THEN** the action scans the prior commit through the pushed head

#### Scenario: A local positive marker lacks shared coverage
- **WHEN** the parity corpus contains a marker that the shared profile accepts
- **THEN** migration tests fail before inline scanner removal

### Requirement: [CWC-005] Workflow policy actions keep their dispositions
OpenSpec validation SHALL invoke `setup-devenv` before `openspec-governance`. It SHALL retain advisory mode, path filters, and contents-read permission.

`Lint and Unit Tests (Linux)` SHALL invoke shared actionlint in enforce mode. It SHALL invoke shared zizmor in advisory mode.

Both policy actions SHALL receive exact tool versions. Actionlint SHALL also receive its exact archive digest.

These actions MUST NOT create new required job names. Advisory findings MUST NOT fail the job, while operational failures MUST fail it.

#### Scenario: A workflow has an actionlint finding
- **WHEN** shared actionlint reports one finding in enforce mode
- **THEN** `Lint and Unit Tests (Linux)` fails

#### Scenario: Zizmor reports a security finding
- **WHEN** shared zizmor reports a finding in advisory mode
- **THEN** the job reports the finding and continues

#### Scenario: A policy action cannot execute
- **WHEN** tool verification, input parsing, or action execution fails
- **THEN** the containing job fails regardless of advisory finding mode

#### Scenario: OpenSpec reports a schema finding
- **WHEN** OpenSpec validation returns a reviewed advisory finding
- **THEN** the workflow emits a warning without activating governance

### Requirement: [CWC-006] Cache-prune policy remains repository-owned
The cache-prune workflow SHALL retain its Monday schedule, manual dispatch, apply input, permissions, and noncanceling concurrency group.

Scheduled runs SHALL use apply mode. Manual runs SHALL use dry-run mode unless the caller explicitly selects apply.

The workflow SHALL pass `refs/heads/main`, retention count `1`, and protected prefix `cvc5-prebuilt-` to `prune-actions-cache`.

The local cache-prune implementation SHALL not remain after equivalent shared tests and workflow contract tests pass.

#### Scenario: A maintainer dispatches the default command
- **WHEN** the manual apply input is false
- **THEN** the shared action runs in dry-run mode

#### Scenario: The weekly schedule runs
- **WHEN** the scheduled event starts the workflow
- **THEN** the shared action runs in apply mode with the protected cvc5 prefix

#### Scenario: The protected prefix is absent
- **WHEN** a workflow mutation removes `cvc5-prebuilt-`
- **THEN** the workflow contract test fails before hosted execution

#### Scenario: Duplicate deletion logic returns
- **WHEN** the workflow invokes a local cache selector beside the shared action
- **THEN** the workflow contract test fails

### Requirement: [CWC-007] Repository-specific policy stays in Chelis
Chelis SHALL retain local ownership of docs-only detection, cvc5 asset policy, success aggregation, release packaging, compatibility APT setup, container topology, and `scripts/gate.py`.

Shared action adoption MUST NOT change job triggers, permissions, runners, matrices, required context names, manual macOS conditions, or release publication policy.

No numbered language specification, compiler surface, runtime surface, CLI surface, backend surface, package output, or generated code SHALL change for this migration.

#### Scenario: A shared action absorbs a Chelis algorithm
- **WHEN** the migration moves one listed repository-specific policy into ci
- **THEN** design review rejects the change

#### Scenario: A required job name changes
- **WHEN** a workflow mutation changes an existing required context name
- **THEN** the workflow contract test fails

#### Scenario: Product code remains unchanged
- **WHEN** reviewers inspect the migration diff
- **THEN** no compiler, runtime, CLI, backend, package, or generated-code behavior change is present

### Requirement: [CWC-008] Evidence fails closed and preserves the hosted oracle
Workflow contract tests SHALL cover exact pins, action order, input mappings, finding modes, nightly identity, cache policy, and native Python paths.

Negative mutations SHALL remove one prerequisite or restore one forbidden inline form. Each mutation MUST fail for its intended reason.

The authoritative completion oracle SHALL be the required `Lint and Unit Tests (Linux)` check on the consumer pull request.

Local tests MUST NOT claim that they prove private action resolution, installation-token permissions, issue transitions, cache deletion, or hosted interpolation.

#### Scenario: Deterministic local evidence passes
- **WHEN** a contributor runs the full `scripts/test_*.py` discovery command inside Devenv
- **THEN** all workflow contract and mutation tests pass

#### Scenario: A direct Python path returns
- **WHEN** a workflow uses `.venv/bin/python` or `.devenv/state/venv/bin/python`
- **THEN** the nonportable marker test fails

#### Scenario: Hosted completion passes
- **WHEN** `Lint and Unit Tests (Linux)` completes successfully on the pull request
- **THEN** the migration satisfies its authoritative acceptance oracle

#### Scenario: GitHub Actions is unavailable
- **WHEN** local evidence passes but the required hosted context does not run
- **THEN** the migration remains incomplete
