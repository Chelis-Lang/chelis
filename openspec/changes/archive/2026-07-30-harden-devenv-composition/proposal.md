## Why

The Devenv module pin does not require a matching local CLI version. Also, remote Devenv consumers cannot load the local modules because `devenv.nix` is empty.

## What Changes

- Require the local Devenv CLI version to match the pinned module version.
- Move the five local module imports from `devenv.yaml` into `devenv.nix`.
- Keep `devenv.yaml` as the owner of inputs and Devenv CLI options.
- Add positive and negative contract tests for both requirements.
- Update the contributor documentation and the Devenv capability specification.
- Keep both `devenv.lock` and `flake.lock` with their existing parity contract.
- Do not change compiler, runtime, CLI, backend, package, or generated-code behavior.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `cross-platform-devenv`: Require CLI and module version parity, and make `devenv.nix` the local module composition root.

## Impact

This change affects `devenv.yaml`, `devenv.nix`, the static Devenv contract tests, contributor documentation, and the Devenv capability specification. It does not change product artifacts or public Chelis commands.
