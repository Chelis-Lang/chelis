# Manual shared runner canary

## Contract

The canary checks one organization runner on the existing EC2. It is not a compiler acceptance suite or a general workflow migration.

The helper requires a manual main-branch run from the fixed `Chelis-Lang/chelis` identity. A local or GitHub-hosted invocation must fail, not report a successful skip.

The host independently requires private repository metadata and admits the exact repository ID, workflow path, and approved source revision. Workflow environment values are evidence inputs, not authentication.

The canary uses group `chelis-ci-trusted` and label `chelis-ci-warm-x64`. Existing pull request and macOS workflows keep their current routes.

The canary declares no App-key input and creates no AWS credential session. Its checkout token has only repository content-read permission and does not persist in Git configuration.

The Nix probe uploads a new derivation with the member client. Readback enters a fresh local binary cache.

The probe runs `nix store verify --sigs-needed 1` against that cache with only the reviewed public key. It disables substitute sources and private-key files for this operation.

The verifier must accept the signature before the probe emits a receipt. A successful copy or matching key-name prefix does not prove signature validity.

The Kache probe builds an isolated Rust crate with a unique run marker. It uploads only that crate's entries and pulls them into an empty local cache.

Kache acceptance requires byte-identical restored entries and an actual local cache hit after the target directory is removed. A transfer exit code alone cannot pass.

The canary permits only the private cache endpoints. Its child environments exclude ambient App, GitHub, AWS, and OIDC credentials.

Cleanup removes only probe-owned local state. No probe deletes remote objects. Bucket lifecycle rules own remote cleanup.

A passing receipt must identify the actual run and source revision. It must also identify the existing EC2 and its current system closure. Failed commands, missing entries, altered bytes, absent or invalid signatures, and missing cache hits must fail the run.

## Preparation oracle

The local oracle is `.venv/bin/python -m unittest scripts.test_shared_runner_canary`. It includes positive and negative boundary and receipt tests.

Four offline signature tests require local `nix` and `nix-store` executables. They skip explicitly when those tools are absent.

The signature oracle is `.venv/bin/python -m unittest scripts.test_shared_runner_canary.NixSignatureTests`. It must pass all four tests without a skip before signature acceptance is claimed.

That oracle generates a temporary test key and a synthetic archive outside the shared Nix store. It requires genuine signatures to pass and unsigned, forged, or wrong-key signatures to fail.

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

The repaired local oracle passed all 18 tests without a skip. Ruff and Git whitespace checks passed. The earlier actionlint result covers the unchanged workflow.

The offline signature cases used workstation Nix `2.34.8`. They first reproduced acceptance of a forged signature before the repair.

Explicit Nix verification now rejects that input before receipt creation. The tests do not prove live TLS, the pinned host's Nix behavior, or actual Kache reuse.

An off-main dispatch skips the workflow job. A skipped job has no acceptance receipt and cannot establish success.

The workflow also runs this contract suite before its cache probe. It declares the existing custom runner label in `.github/actionlint.yaml`.

The workflow inventory classifies this manual probe outside the per-PR developer gate. All 53 CI parity tests passed after that entry was added.

The PR `script-unit` job runs the portable contract suite. The ownership records bind each new control path to that job, not to live acceptance.

The first pre-push gate failed during regeneration with `Invalid MPFR directory`. The pinned Arb build omits the MPFR path.

A local build supplied that path and used the existing `ARB_SYS_CACHE` interface. Cargo inputs, compiler features, and required gate stages remain unchanged.

An extra upstream C test stalled at `arb_hypgeom_erf_bb` for exact input `7/8`, complement `0`, and precision `623`. A bounded probe reproduced the stall.

The locked dependency disables those C tests by default. This extra result is a recorded limitation, not a claimed test pass.

The repaired environment passed `cargo check -p chelis-prove --features arb --tests` and gate regeneration. The next gate exposed this document's invalid filename.

This document now uses the required snake_case name. The complete fast gate passed all four stages in 520.4 seconds.

The tripwire stage passed 76 tests and skipped one test. The gate changed no files.

The local receipt is `target/gate-reports/20260914T130407.608878Z-53548-fast.json`. It covers the renamed document and its updated test reference before commit.

The Cargo inputs and `chelis-prove` source match `origin/main` at `bdd7b961460ae7213893bb996ebc955023face55`. The local cache does not provision the future shared host.

The focused canary, CI parity, and ownership suites passed all 103 tests. The CI ownership plan passed with no required Rust integration targets for this change.

The initial source-publication approval covered commits and pushes from the preparation worktree. The later user approval permits publication to main after the required tests and review.

Source publication does not establish a deployed runner or authorize an automatic canary dispatch.

No dispatch, runner registration, or remote cache write formed part of the local oracle.
