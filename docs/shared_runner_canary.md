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

The cache endpoints, bucket, region, Nix signing key, and runner instance are deployment configuration, not source. The workflow passes them from the repository variables `SHARED_CACHE_NIX_URL`, `SHARED_CACHE_KACHE_URL`, `SHARED_CACHE_KACHE_BUCKET`, `SHARED_CACHE_KACHE_REGION`, `SHARED_CACHE_SIGNING_KEY_NAME`, `SHARED_CACHE_PUBLIC_KEY`, and `SHARED_RUNNER_INSTANCE_ID`. A missing or malformed value fails the run before any probe starts; there is no default.

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
