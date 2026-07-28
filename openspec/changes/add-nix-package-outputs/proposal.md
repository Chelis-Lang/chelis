## Why

The Devenv shell supplies build tools, but the repository does not expose Chelis artifacts through `nix build`. Nix users and CI need locked outputs that match the existing release artifact boundaries.

## What Changes

- Add a tracked root Nix flake and lock file for public package outputs.
- Export a default `chelis` toolchain package for supported Linux and macOS systems.
- Export separate `chelis-runtime` and `chelisup` packages.
- Export `chelis` and `chelisup` app outputs for `nix run`.
- Build the `chelis` binary with the shipped SMT feature and without network access in the build sandbox.
- Install the runtime library and public headers with the same layout as the release toolchain.
- Add Nix checks for package contents, executable behavior, SMT activation, and lock parity with Devenv.
- Document `nix build`, `nix run`, package names, supported systems, and the relationship to `chelisup`.
- Keep internal Rust crates, the Python extension, documentation builds, release publication, and binary caches out of scope.

## Capabilities

### New Capabilities

- `nix-package-outputs`: Define locked Nix packages, apps, artifact layouts, supported systems, and executable checks for Chelis.

### Modified Capabilities

None.

## Impact

- Add `flake.nix`, `flake.lock`, and Nix package definitions.
- Add tests that compare shared Nixpkgs and Rust overlay pins with `devenv.lock`.
- Extend contributor documentation and the changelog with Nix build commands.
- Add optional Nix checks to CI without replacing the existing Cargo and release gates.
- Preserve `chelisup` as the installer and version router for release toolchains.
