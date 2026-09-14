# Manual shared runner canary

## Contract

The canary checks one organization runner on the existing EC2. It is not a compiler acceptance suite or a general workflow migration.

The helper requires a manual main-branch run from the fixed `Chelis-Lang/chelis` identity. A local or GitHub-hosted invocation must fail, not report a successful skip.

The host independently requires private repository metadata and admits the exact repository ID, workflow path, and approved source revision. Workflow environment values are evidence inputs, not authentication.

The canary uses group `chelis-ci-trusted` and label `chelis-ci-warm-x64`. Existing pull request and macOS workflows keep their current routes.

The canary declares no App-key input and creates no AWS credential session. Its checkout token has only repository content-read permission and does not persist in Git configuration.

The Nix probe uploads a new derivation with the member client. Readback enters a fresh file store with signature checks enabled and the reviewed public key.

The Kache probe builds an isolated Rust crate with a unique run marker. It uploads only that crate's entries and pulls them into an empty local cache.

Kache acceptance requires byte-identical restored entries and an actual local cache hit after the target directory is removed. A transfer exit code alone cannot pass.

The canary permits only the private cache endpoints. Its child environments exclude ambient App, GitHub, AWS, and OIDC credentials.

Cleanup removes only probe-owned local state. No probe deletes remote objects. Bucket lifecycle rules own remote cleanup.

A passing receipt must identify the actual run and source revision. It must also identify the existing EC2 and its current system closure. Failed commands, missing entries, altered bytes, absent signatures, or missing cache hits must fail the run.

## Preparation oracle

The local oracle is `.venv/bin/python -m unittest scripts.test_shared_runner_canary`. It includes positive and negative boundary and receipt tests.

Local fixtures do not establish live cache access. The deployed runner still needs the reviewed member client and organization admission policy.

## Future manual acceptance

1. Obtain approval for source publication and the runner cutover.
2. Pin the published main revision in the host admission policy.
3. Dispatch `shared-runner-canary.yml` on main.
4. Require the cache step to pass on `chelis-ci-warm`.
5. Retain the JSON receipt from the job log and summary.
6. Require a fresh CI policy job on the same host.

The canary uses the host's Nix Python through a fresh uv-created environment. It does not install an Ubuntu package or change the Nix daemon configuration.

## Current state

The local oracle passed all 13 tests. Ruff, actionlint, and Git whitespace checks passed.

The tests include simulated orchestration and negative readback cases. They do not prove live TLS, Nix signature verification, or actual Kache reuse.

The workflow also runs this contract suite before its cache probe. It declares the existing custom runner label in `.github/actionlint.yaml`.

The workflow inventory classifies this manual probe outside the per-PR developer gate. All 53 CI parity tests passed after that entry was added.

The PR `script-unit` job runs the portable contract suite. The ownership records bind each new control path to that job, not to live acceptance.

The pre-push gate failed at the first regeneration stage after 1106.8 seconds. `arb-sys 0.3.6` reported `Invalid MPFR directory`.

The gate did not run the remaining stages. The Cargo inputs and `chelis-prove` source match `origin/main` at `6c696c08409ea97756fe1bbbb004bbefc0ea47b2`.

The focused canary, CI parity, and ownership suites passed all 103 tests. The CI ownership plan passed with no required Rust integration targets for this change.

The source-publication approval covers commits and pushes from the preparation worktree. It does not authorize a merge, dispatch, or runner cutover.

No dispatch, runner registration, or cache write formed part of the local oracle.
