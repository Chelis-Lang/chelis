## Why

An OpenSpec change is prose. It has no build, no test suite, and nothing a reviewer can execute. Human approval of one is a queue, not a control: it delays the document without adding evidence.

The repository already treats OpenSpec as planning and authoring evidence rather than runtime or implementation authority. Nothing in that boundary requires a person to click approve on a document, yet every OpenSpec pull request waits for one today.

The maintainer authorized automatic acceptance for OpenSpec document paths, including normative `openspec/specs/**` text. Implementation code keeps the ordinary review and test path.

## What Changes

- Add `scripts/openspec_acceptance.py`, one trusted classifier that decides whether a change set is confined to OpenSpec document paths.
- Add `scripts/openspec_submit.py`, a one-command submission that classifies the working tree and the index, validates the tree, and opens a pull request.
- Add the `openspec-autoland` workflow on `pull_request_target`, so it runs the base branch's copy of itself: a read-only boundary job, and a merge job holding the only write token.
- Add `scripts/openspec_merge.py`, which discovers the contexts branch protection actually requires, waits for them on the exact head commit, and performs one ordinary merge bound to that commit.
- Add `openspec-autoland-validate.yml`, a read-only `pull_request` job running strict OpenSpec validation in enforce mode, published as a check the merge worker requires by name.
- Add a governance-identity check: the head must carry byte-identical `.github`, `scripts`, `openspec/config.yaml`, and CI-environment content, compared by Git object id rather than by diff.
- **Do not** request auto-merge anywhere. Auto-merge is a standing grant on a mutable branch, and any withdrawal races the merge it is trying to prevent.
- **Do not** use an administrative or bypass option, and never submit an approval.
- **Do not** change branch protection or any repository setting, and do not lower the review requirement for code. The worker must never be able to land a code change.
- Fail closed on every unclassifiable input: an empty change set, an unreadable diff record, a combined merge diff, a symlink, a submodule pointer, an executable bit, a copy record, an unknown status letter, a deletion of a capability specification, and a rename that crosses the document boundary.
- Exclude `openspec/config.yaml`, `.github/**`, `scripts/**`, and every other non-document path from automatic acceptance, so a change cannot be judged by the acceptance policy it is trying to replace.
- Exclude fork and draft pull requests from automatic acceptance.
- Record the authorization in `spec/design/spec_provenance.md` and the contributor contract in `AGENTS.md`.
- Do not weaken, disable, or bypass any branch protection rule.
- Do not change compiler, runtime, CLI, backend, package, or generated-code behavior.
- Do not make the autoland workflow a required status check.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `openspec-validation`: automatic acceptance of OpenSpec document changes, its fail-closed boundary, and the separation between structural validation and semantic correctness.

## Impact

The change affects two new scripts, their tests, one new report-only workflow, the agent contract in `AGENTS.md`, and the Phase 0 activation contract in `spec/design/spec_provenance.md`.

A submission opens an ordinary pull request, which the autoland workflow merges once the required checks pass on that exact commit. No human approval is involved and no setting is changed to allow it.

The measured hosted state, read on 2026-09-08 and unchanged by this proposal:

| Setting on `Chelis-Lang/chelis` | Value |
|---|---|
| `main` required approving reviews | `0` |
| `main` require code-owner reviews | `false` |
| `main` dismiss stale reviews | `false` |
| `main` require branches up to date (`strict`) | `false` |
| `main` enforce admins | `true` |
| `main` required status contexts | 9 CI contexts |
| Repository `allow_auto_merge` | `false` |
| Repository visibility | private |

Two consequences worth stating plainly. First, `main` has **no approval requirement today**, so the only gate is the nine required checks, and an immediate merge after those checks pass needs no new permission, credential, or setting. An earlier draft of this change recommended setting approvals to zero, which was both wrong and unnecessary. Second, `allow_auto_merge` is `false` and stays `false`: this design never asks for auto-merge, so the setting is irrelevant rather than a prerequisite.

If an approval requirement is added later, the merge API refuses and the worker reports the pull request as blocked. Nothing here works around that.
