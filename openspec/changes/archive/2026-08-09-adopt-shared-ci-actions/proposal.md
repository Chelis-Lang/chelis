## Why

Chelis repeats shared authentication, status, cache, and policy logic across workflows. The copies increase drift and preserve several host-tool paths after Devenv setup.

## What Changes

- Pin the ci Devenv input and adopted ci actions to one reviewed ci commit.
- Replace all private-ci authentication blocks with the fixed-scope shared action.
- Replace three inline nightly issue reporters with `actions/nightly-status`.
- Replace the required authorship scanner after shared-policy parity passes.
- Replace the Nix runner cleanup one-liner with `actions/reclaim-ubuntu-runner-disk`.
- Replace the local cache-prune implementation with `actions/prune-actions-cache`.
- Migrate OpenSpec validation to the reviewed portable action runtime.
- Add enforced shared actionlint and advisory shared zizmor checks without new required contexts.
- Use Devenv Python in both native Nix jobs. Remove their host uv setup and root `.venv` creation.
- Preserve all job names, required contexts, triggers, permissions, manual macOS policy, and release policy.
- Keep Debian 11, cvc5, release, compiler, runtime, CLI, backend, package, and generated-code behavior unchanged.

## Capabilities

### New Capabilities

- `ci-workflow-composition`: Defines Chelis ownership boundaries, shared action adoption, pin convergence, and workflow policy evidence.

### Modified Capabilities

- `cross-platform-devenv`: Updates native package CI to use shared authentication, disk reclamation, and Devenv-managed Python from one reviewed ci revision.

## Impact

The change affects GitHub workflows, Devenv pins, workflow contract tests, OpenSpec validation tests, cache-prune scripts, documentation, and the changelog.

Implementation depends on reviewed ci revisions for private-input authentication, authorship parity, and cache pruning. Existing nightly, workflow-lint, security-audit, OpenSpec, and disk actions also move to that revision.

The authoritative completion oracle is the required `Lint and Unit Tests (Linux)` check on the consumer pull request. Local script tests provide supporting deterministic evidence.
