## 1. Satisfy Central Prerequisites

- [x] 1.1 Merge `add-private-ci-input-auth-action` in ci with green local and hosted oracles.
- [x] 1.2 Merge `align-shared-authorship-policy` in ci with complete Chelis parity fixtures.
- [x] 1.3 Merge `add-actions-cache-prune-action` in ci with a green hosted dry-run.
- [x] 1.4 Select one reviewed ci commit that contains every adopted action.
- [x] 1.5 Verify that each selected action uses the required portable or raw-host runtime.

## 2. Add Chelis Contract Tests First

- [x] 2.1 Add positive pin tests for one ci Devenv revision and all adopted action references.
- [x] 2.2 Add negative mutations for stale, mixed, short, tag, and branch references.
- [x] 2.3 Add authentication-order tests and mutations for missing setup, missing authentication, and inline token blocks.
- [x] 2.4 Add native-job tests for Devenv Python and forbidden host environment creation.
- [x] 2.5 Add nightly tests for exact issue identity, result wiring, permissions, and cancellation behavior.
- [x] 2.6 Add authorship parity fixtures and required-context name tests.
- [x] 2.7 Add actionlint enforce-mode and zizmor advisory-mode tests with operational-failure controls.
- [x] 2.8 Add cache tests for schedule, dispatch apply policy, permissions, concurrency, retention, and protected prefixes.
- [x] 2.9 Confirm that each new mutation fails for its intended reason before workflow edits.

## 3. Converge Pins and Native Jobs

- [x] 3.1 Update the ci URL in `devenv.yaml` to the selected full commit SHA.
- [x] 3.2 Refresh `devenv.lock` and verify that it resolves the same ci SHA.
- [x] 3.3 Update every adopted ci action reference to the selected SHA.
- [x] 3.4 Replace native-job inline authentication with `authenticate-private-ci-input` after `setup-devenv`.
- [x] 3.5 Replace Linux Nix cleanup with `reclaim-ubuntu-runner-disk` before setup.
- [x] 3.6 Replace native-job host uv and root `.venv` commands with Devenv Python commands.
- [x] 3.7 Preserve native package outputs, flake checks, cvc5 cache policy, and manual Darwin conditions.

## 4. Migrate Status and OpenSpec Workflows

- [x] 4.1 Replace the heavy-E2E inline reporter with `nightly-status` and exact existing issue data.
- [x] 4.2 Replace the conformance-nightly inline reporter with `nightly-status` and exact existing issue data.
- [x] 4.3 Replace the SMT full-prove inline reporter with `nightly-status` and exact existing issue data.
- [x] 4.4 Add `setup-devenv` before each portable nightly action.
- [x] 4.5 Add `setup-devenv` before the reviewed `openspec-governance` action.
- [x] 4.6 Preserve OpenSpec path filters, contents-read permission, and advisory finding mode.

## 5. Migrate Authorship and Workflow Policy

- [x] 5.1 Fix the existing actionlint `SC2129` finding without a policy exception.
- [x] 5.2 Replace the inline authorship scanner with `check-authorship` after parity passes.
- [x] 5.3 Preserve the `No AI authorship markers` name and event-specific range calculation.
- [x] 5.4 Add shared actionlint in enforce mode to `Lint and Unit Tests (Linux)`.
- [x] 5.5 Add shared zizmor in advisory mode to the same required job.
- [x] 5.6 Pin exact tool versions and the actionlint archive digest as data inputs.
- [x] 5.7 Review and classify every initial zizmor finding before merge.

## 6. Migrate Cache Pruning

- [x] 6.1 Add `setup-devenv` before `prune-actions-cache`.
- [x] 6.2 Pass `refs/heads/main`, retention `1`, and protected prefix `cvc5-prebuilt-`.
- [x] 6.3 Preserve the Monday schedule, manual apply input, permissions, and concurrency group.
- [x] 6.4 Run one hosted manual dry-run and inspect all numeric outputs.
- [x] 6.5 Restore scheduled apply mode only after the dry-run passes.
- [x] 6.6 Remove `scripts/ci_cache_prune.py` after equivalent shared and consumer tests pass.
- [x] 6.7 Remove `scripts/test_ci_cache_prune.py` after its consumer-policy cases move to workflow contract tests.

## 7. Update Documentation and Review Boundaries

- [x] 7.1 Update workflow comments and contributor documentation for shared action ownership.
- [x] 7.2 Update the changelog with pin convergence, Devenv Python, and policy-action adoption.
- [x] 7.3 Document GitHub App prerequisites without credential values.
- [x] 7.4 Keep manual macOS, manual release, required-context, and OpenSpec authority text unchanged.
- [x] 7.5 Verify that no numbered language specification or product behavior changed.

## 8. Run Deterministic Validation

- [x] 8.1 Run `devenv --profile ci shell --no-tui -- python -m unittest discover -s scripts -p 'test_*.py'`.
- [x] 8.2 Run the focused Devenv composition, workflow, release, and cvc5 contract tests.
- [x] 8.3 Run `devenv test --no-tui`.
- [x] 8.4 Run actionlint against every workflow and require zero findings.
- [x] 8.5 Run advisory zizmor and record every finding disposition.
- [x] 8.6 Run Nix formatting, focused Ruff checks, and `git diff --check`. The repository has no pre-commit configuration.
- [x] 8.7 Run `devenv --profile ci shell --no-tui -- python scripts/gate.py --local`.
- [x] 8.8 Run strict OpenSpec validation and confirm only the documented advisory schema result.

## 9. Run Adversarial and Hosted Acceptance

- [x] 9.1 Execute every workflow mutation and verify its exact failure category.
- [x] 9.2 Verify that all direct `macos-latest` jobs remain manual-only.
- [x] 9.3 Verify that branch dispatches remain artifact-only and tag publication remains manual.
- [x] 9.4 Push only after fresh approval under the repository permission policy.
- [ ] 9.5 Require `Lint and Unit Tests (Linux)` to pass as the authoritative completion oracle.
- [ ] 9.6 Require all other existing required contexts without a branch-protection change.
- [ ] 9.7 Keep the change incomplete if GitHub Actions does not produce the required oracle.
