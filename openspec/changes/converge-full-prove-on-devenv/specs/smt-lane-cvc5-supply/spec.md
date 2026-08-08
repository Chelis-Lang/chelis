## MODIFIED Requirements

### Requirement: The SMT lanes build on the Devenv toolchain
The `smt-build`, `smt-build-darwin-arm64`, and `full-smt-prove` jobs MUST invoke the pinned shared Devenv setup and private-ci authentication before their first project command. Every Cargo command in those jobs MUST run through `devenv --profile smt shell`.

Those jobs MUST NOT install a Rust toolchain with `dtolnay/rust-toolchain`. They MUST NOT install C or solver dependencies with apt. They MUST NOT create a host uv environment.

The Devenv shell MUST provide the full-prove solver stack: `z3` with its header and library environment wiring, `gappa`, GMP with its compile environment wiring, `m4`, and `make`.

The job names, trigger conditions, docs-only gating, timeout ceilings, and the full-prove tracking-issue reporter MUST keep their current identity. `SMT Feature Build (Linux)` MUST remain the required context name.

#### Scenario: The Linux smoke lane runs on a code pull request
- **WHEN** `smt-build` runs
- **THEN** the pinned Devenv setup precedes authentication
- **AND** the SMT build, the discharge verifier, and the engine smoke tests run inside `devenv --profile smt shell`

#### Scenario: The full-prove lane runs its corpus
- **WHEN** `full-smt-prove` runs on schedule or dispatch
- **THEN** every solver feature build and test command runs inside `devenv --profile smt shell`
- **AND** the z3 and carcara features link the Devenv-provided solver libraries

#### Scenario: A host toolchain step returns
- **WHEN** any SMT lane adds `dtolnay/rust-toolchain`, an apt install, or a host uv setup
- **THEN** the workflow contract test fails

#### Scenario: A required context is renamed
- **WHEN** the `smt-build` job name changes
- **THEN** the workflow contract test fails

### Requirement: The SMT lanes link the pinned cvc5 tree
Each SMT lane MUST build Devenv `outputs.cvc5-dir` for its system. The `smt` profile MUST set that store path as `CVC5_DIR` before the SMT Cargo build.

Each lane MUST restore and save the `cvc5-dir` closure through the GitHub Actions cache, keyed on the `cvc5-dir` derivation path, with the same import/export pattern the Nix package job uses.

The SMT lanes MUST NOT fetch a harvested prebuilt cvc5 and MUST NOT build cvc5 from source through `cvc5-sys` in CI.

#### Scenario: The closure cache is warm
- **WHEN** a lane restores the cached `cvc5-dir` closure
- **THEN** `devenv build outputs.cvc5-dir` completes without rebuilding cvc5
- **AND** `cvc5-sys` links the prebuilt archives from `CVC5_DIR`

#### Scenario: The cvc5 pin moves
- **WHEN** `nix/cvc5.nix` or the `cvc5-sys` pin changes the `cvc5-dir` derivation
- **THEN** the cache key changes and the lane builds the new tree once

#### Scenario: A harvested-prebuilt reference returns
- **WHEN** any workflow invokes `ci_cvc5_cache.py`
- **THEN** the workflow contract test fails

### Requirement: The harvested-prebuilt machinery is retired
The repository MUST NOT contain `.github/workflows/build-cvc5.yml`, `scripts/ci_publish_cvc5_release.py`, `scripts/test_ci_publish_cvc5_release.py`, `scripts/ci_cvc5_cache.py`, or `scripts/test_ci_cvc5_cache.py`.

No workflow MUST reference the retired producer workflow, the publish script, or the cache script.

#### Scenario: The producer workflow returns
- **WHEN** a workflow file publishes a `cvc5-prebuilt-*` Release asset
- **THEN** the workflow contract test fails

#### Scenario: The cache script returns
- **WHEN** a workflow references `ci_cvc5_cache.py`
- **THEN** the workflow contract test fails

## REMOVED Requirements

### Requirement: The full-prove lane does not mix toolchains
**Reason**: The lane now runs entirely on the Devenv toolchain with the flake's pinned cvc5, so the host-cargo/Nix-archive mixing hazard the prohibition guarded cannot occur. The convergence requirements above subsume it.
