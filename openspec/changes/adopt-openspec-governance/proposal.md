## Why

Chelis already defines Phase 0 OpenSpec adoption in `spec/design/spec_provenance.md`, but that contract is not active because the repository lacks project configuration, lifecycle policy, executable validation, review routing, and a merge-bound CI gate. Activating that surface now gives agents and reviewers one addressable planning workflow without confusing OpenSpec or GitHub metadata with Chelis product authority.

## What Changes

- **BREAKING (contributor workflow):** require agent-authored feature and behavior changes to create or update one OpenSpec lifecycle before implementation and to identify affected requirements plus positive and negative scenarios.
- Establish repository-owned rules for proposal, delta-spec, design, task, completion, synchronization, archival, and implementation-citation discipline.
- Preserve a narrow nonnormative maintenance path, while requiring every `spec/**` edit to use OpenSpec even when the branch is otherwise documentation-only.
- Define a Chelis-owned, dependency-free governance checker and negative self-tests that enforce lifecycle scope, strict validation, review-queue ordering, citations, completion, and archive synchronization.
- Invoke the private shared `Chelis-Lang/ci/actions/openspec-governance` action by a reviewed full commit SHA after full-history checkout, while keeping triggers, permissions, policy, and acceptance evidence in Chelis.
- Add provider-level review routing and pull-request guidance without treating labels, reviews, or other live GitHub state as canonical Chelis or Buoy authority.
- Keep Buoy provenance work, compiler semantics, runtime behavior, atom identity, assurance, and coverage policy outside this Phase 0 adoption change.

## Capabilities

### New Capabilities

- `spec-driven-change-governance`: Classification, lifecycle, artifact, review-readiness, implementation-citation, completion, synchronization, archival, and maintenance-exemption requirements for Chelis changes.
- `openspec-ci-governance`: Chelis-owned checker behavior, strict and adversarial validation, immutable shared-action integration, event-bound comparison, least privilege, and hosted acceptance requirements.

### Modified Capabilities

None.

## Impact

- Adds the canonical `openspec/` planning tree and baseline governance specifications.
- Plans changes to `scripts/` for the checker and its tests, `.github/workflows/` for the merge-bound gate, and repository review-routing documentation or metadata.
- Uses exact OpenSpec 1.6.0 CI mechanics from `Chelis-Lang/ci`; it adds no runtime dependency to Chelis crates and changes no language, compiler, CLI, backend, numerical, package, or generated-code behavior.
- Activates the Phase 0 requirement only after the named adoption and PR-gate oracles are green; until then these artifacts describe the rollout rather than claiming enforcement.
