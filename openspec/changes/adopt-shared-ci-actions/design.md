## Context

Chelis consumes private actions and a Devenv module from `Chelis-Lang/ci`. The workflows currently reference four ci revisions and repeat several shared mechanics.

The repeated mechanics include private-input authentication, nightly issue updates, authorship checks, cache pruning, disk reclamation, and workflow policy checks.

This change affects contributor and CI process only. It does not alter Chelis language semantics, compiler behavior, runtime behavior, packages, or generated code.

OpenSpec remains advisory planning evidence under `spec/design/spec_provenance.md`. Existing specifications, executable tests, and GitHub review state retain their authority.

## Goals / Non-Goals

**Goals:**

- Chelis uses reviewed shared actions for cohesive cross-repository mechanics.
- All adopted ci references use one reviewed full commit SHA.
- Devenv supplies Python and uv for native package jobs.
- Existing job names, triggers, permissions, and policy remain stable.
- Local mutation tests reject stale pins and weakened workflow wiring.
- One required hosted context is the completion oracle.

**Non-Goals:**

- The change does not alter required branch-protection contexts.
- The change does not alter macOS manual-dispatch policy.
- The change does not alter release publication policy.
- The change does not centralize Chelis-specific compiler, cvc5, release, or container logic.
- The change does not add Nix closure caching.
- The change does not activate OpenSpec governance.

## Decisions

### Central prerequisites land before consumer edits

Three shared changes must merge first:

- `add-private-ci-input-auth-action`
- `align-shared-authorship-policy`
- `add-actions-cache-prune-action`

The reviewed ci commit must also contain the existing nightly, OpenSpec, reclaim, actionlint, zizmor, and setup actions.

Chelis then pins that one commit in `devenv.yaml`, `devenv.lock`, and every adopted action call. This sequence avoids unavailable action paths.

### Shared actions own mechanics, and Chelis owns policy

The shared actions own their declared data transformations and API postconditions. Chelis retains events, permissions, runners, matrices, dependencies, and job names.

Chelis also retains nightly issue titles, labels, recovery text, cache protection prefixes, and cache apply policy. These values are data inputs, not shared policy.

Repository-specific Python remains local for docs-only detection, cvc5 assets, success aggregation, gates, release packaging, and compatibility APT setup.

### Authentication uses one fixed-scope action

Each job that evaluates the private ci input invokes `setup-devenv` and then `authenticate-private-ci-input`. Later commands inherit the updated `NIX_CONFIG`.

The workflow passes `vars.CI_APP_ID` as `app-client-id`. It passes `secrets.CI_APP_PRIVATE_KEY` as `app-private-key`.

Inline `create-github-app-token` and environment-file steps disappear from those jobs. The GitHub App and repository variables remain consumer prerequisites.

### Native package jobs use Devenv Python

Both native Nix jobs keep their Devenv package commands. Python helpers run through `devenv --profile ci shell --no-tui -- python`.

Host `setup-uv`, `uv venv`, `.venv/bin/python`, and direct Devenv-state interpreter paths disappear. Devenv remains the single Python environment owner.

The Linux job uses the shared fixed disk action before `setup-devenv`. The Darwin job keeps its manual-dispatch condition.

### Status and policy actions keep current topology

Each nightly reporter remains in its current workflow and depends on the current upstream result. The shared action receives existing issue identity data.

The `No AI authorship markers` job keeps its name and range selection. It invokes `check-authorship` with `profile: all-markers` after parity evidence passes.

OpenSpec validation keeps its path filters, read-only permissions, and advisory mode. It adds `setup-devenv` before the current shared action runtime.

Actionlint runs in enforce mode inside `Lint and Unit Tests (Linux)`. Zizmor runs there in advisory mode, so no required context changes.

The known actionlint `SC2129` finding must be fixed before enforce mode lands. Zizmor findings receive review, but advisory findings do not fail the job.

### Initial zizmor findings remain advisory

The local zizmor 1.25.2 audit reported 100 unsuppressed findings after migration.
The hosted action uses zizmor 1.28.0 and will report its final count.

The local findings have these dispositions:

- 59 `unpinned-uses` findings cover existing non-adopted external actions. A separate hardening change can pin those actions.
- 32 `artipacked` findings cover existing checkout credential policy. This migration does not change checkout behavior.
- Three `cache-poisoning` findings cover the trusted ecosystem drift workflow. That workflow does not accept pull requests.
- Two `excessive-permissions` findings cover existing workflow-level publication permissions. Existing jobs require those permissions.
- Three `template-injection` findings use values from a closed, repository-owned matrix. External input does not control those values.
- One `superfluous-actions` finding covers an existing release step. The release contract retains that step.

The migration fixed both `github-app` findings. The two direct cross-repository tokens now request explicit permissions and use an immutable action revision.

### Cache pruning retains destructive policy in Chelis

The cache workflow keeps its Monday schedule, manual apply input, permissions, and concurrency group. Scheduled runs retain apply mode.

Manual runs remain dry-run unless the caller selects apply. `cvc5-prebuilt-` remains a protected key prefix, and primary retention remains one generation.

The shared action owns enumeration, selection, and deletion mechanics. The local cache-prune implementation and its duplicate tests then leave the repository.

The hosted consumer dry-run passed in run `31330755749`.
It reported 31 caches and selected zero caches.
It reported zero selected bytes, deletions, and deleted bytes.

### Tests lock ownership boundaries

Workflow tests parse YAML and assert exact action order, pins, inputs, and disposition modes. Negative mutations remove each prerequisite or restore each forbidden inline block.

The tests also reject direct host Python paths, mixed ci revisions, changed required job names, and weakened manual macOS conditions.

The authoritative completion oracle is `Lint and Unit Tests (Linux)` on the consumer pull request. That context runs the full `scripts/test_*.py` suite.

Local script tests and actionlint provide deterministic supporting evidence. They do not prove private action resolution or GitHub API permissions.

## Risks / Trade-offs

- **A central revision breaks several consumers at once** → Chelis pins one reviewed immutable commit and keeps local contract tests.
- **The private App credentials fail** → Authentication fails before Devenv evaluation and names only the failed boundary.
- **A nightly reporter changes issue identity** → Workflow tests lock exact titles, labels, and recovery text.
- **The authorship action misses a local marker** → Central parity fixtures must pass before inline scanner removal.
- **Cache apply mode deletes an intended cache** → The workflow retains protected prefixes, dry-run dispatch, and shared fail-safe enumeration.
- **Advisory zizmor findings remain unresolved** → Review records classify findings before any later enforce-mode proposal.
- **GitHub-hosted evidence is unavailable** → The phase remains incomplete until the required Linux context passes.

## Migration Plan

1. Merge the three prerequisite ci changes.
2. Select one ci commit that contains every adopted action.
3. Add failing Chelis workflow contract tests for the target state.
4. Fix the existing actionlint finding.
5. Update the Devenv input and lock to the selected ci commit.
6. Migrate native Nix Python, disk, and authentication steps.
7. Migrate nightly reporters and OpenSpec validation.
8. Migrate authorship only after the central parity suite passes.
9. Add enforced actionlint and advisory zizmor to the existing Linux job.
10. Migrate cache pruning in dry-run mode and inspect the hosted plan.
11. Restore the existing scheduled apply expression after the dry-run passes.
12. Run the local script suite and repository gates.
13. Require `Lint and Unit Tests (Linux)` to pass on the pull request.

For rollback, restore the prior workflow blocks, Devenv lock, and immutable action pins. Do not alter branch protection during rollback.

Cache deletions cannot be reversed. Roll back cache mechanics only after the scheduled job stops.

## Open Questions

None.
