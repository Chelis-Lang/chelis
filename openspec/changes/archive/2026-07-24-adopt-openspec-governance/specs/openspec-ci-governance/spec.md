## ADDED Requirements

### Requirement: Chelis owns a fixed governance checker
Chelis SHALL provide a dependency-free Python 3.11-or-newer checker as a regular non-symlink file at `scripts/check_openspec.py`. The checker SHALL accept the shared action's fixed invocation `--self-test --merge-bound --base <selected-base>`, use `OPENSPEC_BIN` and `GIT_BIN` when supplied, require OpenSpec 1.6.0 exactly, and propagate every policy or tool failure as a nonzero exit.

#### Scenario: Shared action invokes the checker
- **WHEN** the fixed path, executable environment, exact version, and arguments are valid
- **THEN** the checker SHALL run negative self-tests and merge-bound repository governance against the selected base

#### Scenario: Checker prerequisite is invalid
- **WHEN** Python, Git, OpenSpec, the checker path, an argument, or the exact OpenSpec version is missing or incompatible
- **THEN** validation SHALL fail without an ambient executable or alternate-path fallback

### Requirement: Deterministic controls fail closed
The checker and `scripts/test_check_openspec.py` SHALL cover strict artifact validation, planning-before-implementation ordering, pull-request change citations, one-record branch scope, exact maintenance exemptions, symlink rejection, built-in schema selection, active-versus-archived state, complete described tasks, archive-date validity, and replayed delta-to-baseline synchronization. Each acceptance rule SHALL have at least one planted negative fixture proving rejection.

#### Scenario: Valid governed change is evaluated
- **WHEN** one correctly ordered, complete, synchronized, and archived lifecycle is compared with its base
- **THEN** all deterministic controls SHALL pass

#### Scenario: A control is weakened
- **WHEN** a planted fixture removes, mutates, reorders, hides, or desynchronizes required governance evidence
- **THEN** the corresponding self-test SHALL fail for the intended reason

### Requirement: Event-bound comparisons use complete history
The CI workflow SHALL check out complete history with credentials persistence disabled. Pull requests SHALL compare against `pull_request.base.sha`; ordinary pushes SHALL compare against the event's nonzero `before` SHA; workflow dispatch and zero-before pushes SHALL compare against `origin/main`. Event-provided SHAs SHALL be exactly 40 hexadecimal characters, and missing or unresolvable comparison objects SHALL fail rather than silently selecting another base.

#### Scenario: Pull request runs
- **WHEN** GitHub supplies a valid pull-request base SHA and full history contains its merge base
- **THEN** the checker SHALL evaluate the complete branch diff against that base

#### Scenario: Comparison history is shallow or malformed
- **WHEN** the selected base is malformed, absent, or cannot produce a merge base
- **THEN** governance SHALL fail with a bounded diagnostic

### Requirement: CI uses an immutable shared action
Chelis SHALL invoke `Chelis-Lang/ci/actions/openspec-governance@2906e03880a2ea7d553b0959c246991b1b058990` after `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1` with `fetch-depth: 0` and `persist-credentials: false`. Chelis SHALL NOT duplicate the shared action's Node setup, npm package metadata, lockfile, event-base launcher, or exact OpenSpec installation.

#### Scenario: Immutable action is available
- **WHEN** the consumer workflow resolves the reviewed full action SHA
- **THEN** the action SHALL provision its locked OpenSpec 1.6.0 graph and invoke only the Chelis-owned fixed checker

#### Scenario: Action source or package graph is unavailable
- **WHEN** private repository access, immutable resolution, npm integrity, exact version verification, or action launch fails
- **THEN** the job SHALL fail without falling back to a mutable ref, Nix, copied npm metadata, or an ambient OpenSpec executable

### Requirement: Governance runs with least privilege
OpenSpec governance SHALL run in a dedicated workflow or job with only `contents: read`, no caller secrets, no write permission, and no arbitrary caller-selected commands, checker paths, modes, bases, or executables. Chelis SHALL retain workflow triggers, job ordering, branch-protection selection, checker policy, and all compiler and repository acceptance authority.

#### Scenario: Governance workflow is inspected
- **WHEN** reviewers inspect its permissions and action invocation
- **THEN** they SHALL find no write permission, secret forwarding, mutable action reference, or caller-controlled execution surface

#### Scenario: Existing CI has broader permission
- **WHEN** another Chelis workflow or job requires write access
- **THEN** OpenSpec governance SHALL remain isolated under its explicit read-only permission boundary

### Requirement: Pull-request routing is event-visible but noncanonical
On pull-request events the checker SHALL validate the repository-defined OpenSpec change citation from the bounded GitHub event payload and SHALL enforce that `spec/**` cannot use the documentation-only or maintenance path. Provider labels, timestamps, reviews, and body fields SHALL remain process-routing evidence only and SHALL NOT become Chelis product or future Buoy authority.

#### Scenario: Governed PR cites its lifecycle
- **WHEN** the pull-request citation exactly matches the branch lifecycle identifier
- **THEN** the checker SHALL accept the routing link subject to all repository-owned controls

#### Scenario: Provider metadata conflicts with repository evidence
- **WHEN** a label, title, body, or review claims exemption or completion that committed artifacts do not establish
- **THEN** repository-owned governance SHALL fail or remain incomplete

### Requirement: Hosted consumer evidence gates activation
Central deterministic and hosted self-tests SHALL NOT count as proof that Chelis can resolve the private action or that Chelis policy passes. Before Phase 0 activates, the complete Chelis pull-request suite SHALL execute the exact pinned action and checker from the final archived adoption branch, and the resulting governance check SHALL be eligible for branch protection. Rollback SHALL restore the prior workflow state or another separately reviewed immutable SHA without changing compiler behavior or weakening repository specifications.

#### Scenario: Adoption pointer is accepted
- **WHEN** the exact consumer pointer runs successfully with the archived adoption lifecycle and the complete Chelis hosted suite is green
- **THEN** reviewers MAY accept the merge that activates Phase 0 and subsequently require the governance check

#### Scenario: Consumer integration fails
- **WHEN** private action access, checker policy, or another required Chelis job fails during rollout
- **THEN** Phase 0 SHALL remain inactive and the branch SHALL retain a documented rollback path
