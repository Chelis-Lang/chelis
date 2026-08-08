## MODIFIED Requirements

### Requirement: Native package CI uses the reviewed portable Devenv base
Each native Nix package job MUST invoke `Chelis-Lang/ci/actions/setup-devenv@<revision>` once before its runner verification step. `<revision>` MUST be one immutable 40-character commit SHA.

That SHA MUST equal the configured and locked ci Devenv input revision. Each native job MUST invoke `authenticate-private-ci-input` at the same revision before its first project Devenv command.

Each job MUST use `devenv-ci bash --noprofile --norc -e -o pipefail {0}` as its default shell for `run` steps. The Linux job can invoke `reclaim-ubuntu-runner-disk` before setup with its explicit host shell.

The workflow MUST NOT duplicate direct Nix, Cachix, Devenv, App-token, or Nix-authentication bootstrap. The reviewed setup action supplies the pinned Nix and Devenv versions.

Each job MUST run Python helpers through `devenv --profile ci shell --no-tui -- python`. It MUST NOT install host uv, create the repository-root `.venv`, or invoke a direct Devenv-state interpreter path.

Each job MUST run `devenv test --no-tui` and `devenv build --no-tui outputs.chelis outputs.chelis-runtime outputs.chelisup`. The Devenv cache MUST NOT replace the complete native flake check.

The public Devenv cache does not contain the custom Chelis cvc5 derivation. Each job can reuse the prebuilt cvc5 toolchain closure from the repository Actions cache, keyed by the closure derivation name.

#### Scenario: Native CI checks the development shell
- **WHEN** either native Nix package job runs
- **THEN** the job invokes setup and fixed-scope authentication at the ci input revision
- **AND** each Nix and Devenv `run` step uses the portable shell
- **AND** each Python helper uses the activated Devenv interpreter
- **AND** the job runs all named Devenv tasks
- **AND** the job builds all three named Devenv package outputs
- **AND** the job runs the complete native flake check

#### Scenario: Native CI bypasses the portable base
- **WHEN** a job omits setup, authentication, the portable shell, or the shared revision
- **THEN** the native workflow contract fails

#### Scenario: Native CI recreates a Python environment
- **WHEN** a job invokes host uv, creates `.venv`, or names a direct Devenv-state interpreter
- **THEN** the native workflow contract fails

#### Scenario: The Linux runner needs disk reclamation
- **WHEN** the Linux native package job starts on a GitHub-hosted Ubuntu runner
- **THEN** the shared fixed disk action runs before `setup-devenv`

#### Scenario: The custom cvc5 output is absent from public caches
- **WHEN** the native flake check requires the custom non-GPL cvc5 derivation and the repository closure cache has no entry
- **THEN** Nix builds that derivation from source
