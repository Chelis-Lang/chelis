## MODIFIED Requirements

### Requirement: The repository provides reproducible Devenv inputs
The repository MUST track `devenv.nix`, `devenv.yaml`, `devenv.lock`, and the local modules under `devenv/`.

The lock file MUST pin all resolved input revisions.

`devenv.yaml` MUST pin the Devenv module input to release `v2.2`. `devenv.lock` MUST resolve that input to commit `ffce215a42d09c6375c3d60dd9c4110438fc4d87`.

`devenv.yaml` MUST set `require_version: true`. The Devenv CLI MUST reject a CLI version that differs from the pinned module version.

`devenv.yaml` MUST pin the shared `nixpkgs` and `rust-overlay` inputs to exact commit revisions, not floating references. `devenv.lock` MUST resolve each shared input to its configured revision.

The repository MUST ignore `.devenv/` and `.devenv.flake.nix`. These paths contain generated local state, not reproducible inputs.

The repository lint policy MUST exclude both generated paths. It MUST keep the tracked root files and local modules in the editable corpus.

#### Scenario: The repository checks the Devenv release pin
- **WHEN** the configured URL or locked revision differs from Devenv `v2.2`
- **THEN** the static Devenv version contract test fails

#### Scenario: The CLI version matches the module version
- **WHEN** a contributor runs Devenv with the version that supplies the pinned modules
- **THEN** the CLI accepts the version requirement

#### Scenario: The CLI version differs from the module version
- **WHEN** a contributor runs Devenv with a version that differs from the pinned modules
- **THEN** the CLI rejects the configuration before shell activation

#### Scenario: The CLI version requirement is absent
- **WHEN** `devenv.yaml` omits `require_version: true`
- **THEN** the static Devenv version contract test fails

#### Scenario: A shared input uses a floating reference
- **WHEN** `devenv.yaml` names a shared input by branch or tag instead of a full commit revision
- **THEN** the static Devenv version contract test fails

#### Scenario: A shared input lock drifts from its pin
- **WHEN** the locked revision of a shared input differs from the configured revision
- **THEN** the static Devenv version contract test fails

#### Scenario: A clean checkout evaluates the shell
- **WHEN** a contributor runs `devenv test` from a clean checkout
- **THEN** Devenv resolves the tracked inputs without an uncommitted input file

#### Scenario: Generated state stays untracked
- **WHEN** Devenv creates `.devenv/` or `.devenv.flake.nix`
- **THEN** Git ignores each generated path

#### Scenario: Generated state stays outside repository lint
- **WHEN** a contributor runs `chelis lint --check .` after Devenv creates local state
- **THEN** the lint excludes both generated paths and still checks the tracked Devenv inputs

### Requirement: The shell composes local configuration modules
The root `devenv.nix` MUST import these local modules:

- `./devenv/toolchains.nix`
- `./devenv/commands.nix`
- `./devenv/generated-files.nix`
- `./devenv/git-hooks.nix`
- `./devenv/smoke-tests.nix`

`devenv.yaml` MUST NOT duplicate the local imports. It MUST remain the owner of Devenv inputs and CLI options.

The root `devenv.nix` MUST contain only the module imports. Each local module MUST own one configuration concern.

#### Scenario: A local module is absent
- **WHEN** a required local import is absent or duplicated in `devenv.nix`
- **THEN** the static Devenv composition test fails

#### Scenario: YAML duplicates the module composition
- **WHEN** `devenv.yaml` declares a local imports section
- **THEN** the static Devenv composition test fails

#### Scenario: An evaluator reads the root module
- **WHEN** a Devenv evaluator reads `devenv.nix` without the root YAML imports
- **THEN** it discovers all five local modules

#### Scenario: A contributor evaluates the composed shell
- **WHEN** a contributor runs `devenv test`
- **THEN** Devenv combines all five local modules
- **AND** the composed configuration passes the shell contract
