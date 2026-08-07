## MODIFIED Requirements

### Requirement: The shell composes local configuration modules
The root `devenv.nix` MUST import these local modules:

- `./devenv/toolchains.nix`
- `./devenv/commands.nix`
- `./devenv/generated-files.nix`
- `./devenv/git-hooks.nix`
- `./devenv/smoke-tests.nix`
- `./devenv/package-outputs.nix`
- `./devenv/release-outputs.nix`

`devenv.yaml` MUST NOT declare local module imports. Those imports belong in `devenv.nix`.

`devenv.yaml` can declare remote imports for shared configuration. It MUST remain the owner of Devenv inputs and CLI options.

The root `devenv.nix` MUST contain only the module imports. Each local module MUST own one configuration concern.

#### Scenario: A local module is absent
- **WHEN** a required local import is absent or duplicated in `devenv.nix`
- **THEN** the static Devenv composition test fails

#### Scenario: YAML declares a local import
- **WHEN** `devenv.yaml` lists a `./`, `../`, or `.nix` import entry
- **THEN** the static Devenv composition test fails

#### Scenario: An evaluator reads the root module
- **WHEN** a Devenv evaluator reads `devenv.nix` without the root YAML imports
- **THEN** it discovers all seven local modules

#### Scenario: A contributor evaluates the composed shell
- **WHEN** a contributor runs `devenv test`
- **THEN** Devenv combines all seven local modules
- **AND** the composed configuration passes the shell contract
