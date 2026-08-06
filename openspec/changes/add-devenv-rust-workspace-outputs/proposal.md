## Why

Devenv defines language imports as the standard source for package outputs. The current Chelis module instead evaluates the root flake from inside Devenv.

The Devenv Rust helper returns only `rootCrate`, but Chelis has a virtual Cargo workspace. crate2nix exposes Chelis packages through `workspaceMembers`.

## What Changes

- Add a workspace-aware Rust import extension as a local Devenv module.
- Use the crate2nix input and Rust toolchain that Devenv owns.
- Generate one crate2nix workspace graph and select `chelis-cli`, `chelis-runtime`, and `chelisup` from `workspaceMembers`.
- Reuse the cvc5 override, workspace source overrides, and artifact assembly across the flake and Devenv paths.
- Expose `chelis`, `chelis-runtime`, `chelisup`, and `default` through Devenv outputs.
- Remove the root-flake evaluation from the Devenv package module.
- Pin crate2nix in `devenv.yaml` and `devenv.lock` and enforce lock parity.
- Add positive and negative tests for workspace selection, SMT support, package layouts, and forbidden self-flake evaluation.
- Update native Nix CI and contributor documentation for the Devenv-owned package path.
- Keep the root flake as the public package API for downstream Nix users.
- Do not add a root Cargo package or change the Chelis workspace structure.
- Do not fork or replace the pinned Devenv Rust module.
- Do not change compiler, runtime, CLI, backend, language, or generated-code behavior.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `cross-platform-devenv`: Devenv will own a workspace-aware crate2nix import and package outputs without root-flake evaluation.
- `nix-package-outputs`: Devenv and the root flake will share package contracts, crate overrides, artifact assembly, and pinned package inputs.

## Impact

The change affects the local Devenv modules, Devenv inputs, shared Nix package helpers, package contract tests, native Nix CI, and contributor documentation.

The public flake package names and artifact layouts remain unchanged. The Devenv package path gains a direct crate2nix graph for the native system.
