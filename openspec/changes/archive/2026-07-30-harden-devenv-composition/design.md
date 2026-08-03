## Context

Chelis uses the dedicated Devenv CLI and a separate package flake. The Devenv module input uses release `v2.2`, but the configuration does not require a matching CLI version.

`devenv.yaml` imports five local modules. The root `devenv.nix` is empty, so evaluators that read only that file cannot discover the module graph.

## Goals / Non-Goals

**Goals:**

- Require parity between the Devenv CLI version and the pinned module version.
- Make `devenv.nix` the composition root for all five local modules.
- Preserve the current shell behavior and both lock files.
- Add positive and negative static contract tests before the configuration changes.

**Non-Goals:**

- Replace the dedicated Devenv CLI with flake integration.
- Add flake-parts or a `devShells` output.
- Guarantee that every root-local command works from a remote Devenv consumer.
- Change a compiler, runtime, backend, package, or public CLI contract.

## Decisions

### 1. Use the module parity form of `require_version`

Set `require_version: true` in `devenv.yaml`. Devenv then compares the CLI version with the module version.

An exact version string was considered. That option duplicates the version that the module input already owns.

### 2. Make `devenv.nix` the module composition root

Move the five local imports into the Nix `imports` option in `devenv.nix`. Remove the YAML imports to prevent two composition owners.

This structure lets tools discover the module graph from `devenv.nix`. The change does not make the root-local developer commands portable to unrelated repositories.

The alternative keeps imports in `devenv.yaml`. Devenv documents that remote projects do not evaluate imported YAML configuration.

### 3. Parse each static contract at its boundary

Change the composition parser to read Nix imports from `devenv.nix`. Add a separate parser for the YAML CLI version requirement.

Each parser returns a typed result. Positive tests use the repository files. Negative tests mutate one required field and require a boundary error.

### 4. Use the shell contract as the acceptance oracle

The authoritative completion oracle is `devenv test --no-tui`. This command proves that the relocated imports still compose the complete shell.

The Python contract tests provide negative evidence. Strict OpenSpec validation and Nix formatting provide supporting evidence.

## Risks / Trade-offs

- **[A different local CLI version stops shell entry]** → This failure enforces the selected parity contract.
- **[An import disappears during relocation]** → The composition parser requires the exact five-module list.
- **[Both files declare imports]** → The static contract rejects a YAML imports section.
- **[Remote consumers assume full command portability]** → The specification limits the contract to module discovery and composition.

## Migration Plan

1. Add positive and negative contract tests.
2. Add the CLI version requirement.
3. Move the local imports into `devenv.nix`.
4. Update the capability specification and contributor documentation.
5. Run the acceptance oracle and the supporting checks.

To roll back, restore the YAML imports, restore the empty root file, and remove `require_version`.

## Open Questions

No open question blocks this change.
