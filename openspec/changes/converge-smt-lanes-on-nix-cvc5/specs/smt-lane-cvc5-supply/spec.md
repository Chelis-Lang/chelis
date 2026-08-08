## ADDED Requirements

### Requirement: The SMT smoke lanes build on the Devenv toolchain
The `smt-build` and `smt-build-darwin-arm64` jobs MUST invoke the pinned shared Devenv setup and private-ci authentication before their first project command. Every Cargo command in those jobs MUST run through `devenv --profile smt shell`.

Those jobs MUST NOT install a Rust toolchain with `dtolnay/rust-toolchain`. They MUST NOT install C dependencies with apt. They MUST NOT create a host uv environment.

The job names, trigger conditions, docs-only gating, and timeout ceilings MUST keep their current identity. `SMT Feature Build (Linux)` MUST remain the required context name.

#### Scenario: The Linux smoke lane runs on a code pull request
- **WHEN** `smt-build` runs
- **THEN** the pinned Devenv setup precedes authentication
- **AND** the SMT build, the discharge verifier, and the engine smoke tests run inside `devenv --profile smt shell`

#### Scenario: A host toolchain step returns
- **WHEN** either smoke lane adds `dtolnay/rust-toolchain`, an apt install, or a host uv setup
- **THEN** the workflow contract test fails

#### Scenario: A required context is renamed
- **WHEN** the `smt-build` job name changes
- **THEN** the workflow contract test fails

### Requirement: The SMT smoke lanes link the pinned cvc5 tree
Each smoke lane MUST build Devenv `outputs.cvc5-dir` for its system. The `smt` profile MUST set that store path as `CVC5_DIR` before the SMT Cargo build.

Each lane MUST restore and save the `cvc5-dir` closure through the GitHub Actions cache, keyed on the `cvc5-dir` derivation path, with the same import/export pattern the Nix package job uses.

The smoke lanes MUST NOT fetch a harvested prebuilt cvc5 and MUST NOT build cvc5 from source through `cvc5-sys` on the per-PR path.

#### Scenario: The closure cache is warm
- **WHEN** the lane restores the cached `cvc5-dir` closure
- **THEN** `devenv build outputs.cvc5-dir` completes without rebuilding cvc5
- **AND** `cvc5-sys` links the prebuilt archives from `CVC5_DIR`

#### Scenario: The cvc5 pin moves
- **WHEN** `nix/cvc5.nix` or the `cvc5-sys` pin changes the `cvc5-dir` derivation
- **THEN** the cache key changes and the lane builds the new tree once

#### Scenario: A harvested-prebuilt fetch returns
- **WHEN** a smoke lane invokes `ci_cvc5_cache.py fetch`
- **THEN** the workflow contract test fails

### Requirement: The harvested-prebuilt producer is retired
The repository MUST NOT contain `.github/workflows/build-cvc5.yml`, `scripts/ci_publish_cvc5_release.py`, or `scripts/test_ci_publish_cvc5_release.py`.

No workflow MUST reference the deleted producer workflow or the publish script.

`smt-full-prove.yml` MAY keep the Actions-cache harvest cycle from `scripts/ci_cvc5_cache.py`. It MUST NOT fetch a durable Release asset.

#### Scenario: The producer workflow returns
- **WHEN** a workflow file publishes a `cvc5-prebuilt-*` Release asset
- **THEN** the workflow contract test fails

#### Scenario: The full-prove lane runs cold
- **WHEN** `smt-full-prove.yml` finds no cached cvc5 store
- **THEN** it builds cvc5 from source through `cvc5-sys` and harvests the result into the Actions cache

### Requirement: The full-prove lane does not mix toolchains
`smt-full-prove.yml` MUST NOT export a Nix store `CVC5_DIR` while its Cargo commands run on a host toolchain. A lane links a Nix-built `libcvc5.a` only when its Cargo commands run inside `devenv --profile smt shell`.

#### Scenario: A host cargo links the Nix archive
- **WHEN** `smt-full-prove.yml` gains a Nix store `CVC5_DIR` export without moving its Cargo commands into `devenv --profile smt shell`
- **THEN** the workflow contract test fails
