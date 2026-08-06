## MODIFIED Requirements

### Requirement: The shell composes local configuration modules
The root `devenv.nix` MUST import these local modules:

- `./devenv/toolchains.nix`
- `./devenv/commands.nix`
- `./devenv/generated-files.nix`
- `./devenv/git-hooks.nix`
- `./devenv/smoke-tests.nix`
- `./devenv/package-outputs.nix`

`devenv.yaml` MUST NOT declare local module imports; those belong in `devenv.nix`. It MAY declare remote imports for shared configuration (the ci consumer module), and it MUST remain the owner of Devenv inputs and CLI options.

The root `devenv.nix` MUST contain only the module imports. Each local module MUST own one configuration concern.

#### Scenario: A local module is absent
- **WHEN** a required local import is absent or duplicated in `devenv.nix`
- **THEN** the static Devenv composition test fails

#### Scenario: YAML declares a local import
- **WHEN** `devenv.yaml` lists a `./`, `../`, or `.nix` import entry
- **THEN** the static Devenv composition test fails

#### Scenario: An evaluator reads the root module
- **WHEN** a Devenv evaluator reads `devenv.nix` without the root YAML imports
- **THEN** it discovers all six local modules

#### Scenario: A contributor evaluates the composed shell
- **WHEN** a contributor runs `devenv test`
- **THEN** Devenv combines all six local modules
- **AND** the composed configuration passes the shell contract

## ADDED Requirements

### Requirement: The shell composes shared ecosystem tools from ci
The shell MUST obtain crate2nix and OpenSpec from the shared ci consumer Devenv module, composed through `devenv.yaml` (a `ci` input plus the `ci/devenv/consumer` import). Neither tool may be pinned or built independently in this repository.

The shell MUST expose `config.outputs.crate2nix` and `config.outputs.openspec`, built with the repository's own `pkgs`. It MUST provide a `regenerate-crate2nix` command that refreshes or `--check`s the committed `Cargo.nix` graph using `config.outputs.crate2nix`. The Devenv smoke check MUST run the freshness `--check` and MUST verify OpenSpec reports the pinned version.

#### Scenario: The shell builds the shared tools
- **WHEN** a contributor runs `devenv build outputs.crate2nix outputs.openspec`
- **THEN** Devenv builds crate2nix and OpenSpec from the ci consumer module

#### Scenario: The repository pins a shared tool independently
- **WHEN** the shell builds or pins its own crate2nix or OpenSpec instead of composing ci
- **THEN** the static Devenv composition test fails

#### Scenario: The committed graph drifts
- **WHEN** the committed `Cargo.nix` no longer matches a fresh crate2nix generation
- **THEN** the `regenerate-crate2nix --check` smoke task fails
