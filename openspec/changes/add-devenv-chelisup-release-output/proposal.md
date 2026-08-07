## Why

The release workflow builds `chelisup` outside Devenv and duplicates the project toolchain contract. A dedicated Devenv output will make one derivation own each portable bootstrap artifact and its checksum.

## What Changes

- Add `outputs.release-chelisup` as the release build interface.
- Produce one platform-specific `chelisup` executable and one SHA-256 sidecar.
- Build a static `linux-x86_64` executable and a native `darwin-arm64` executable.
- Reject unsupported release platforms and do not advertise absent artifacts.
- Verify the executable version, architecture, runtime dependencies, inventory, and checksum during the build.
- Make the release workflow consume this output instead of a separate host Cargo build.
- Keep the portable output separate from the Nix-oriented `outputs.chelisup` package.

## Capabilities

### New Capabilities

- `portable-chelisup-release`: Defines the portable executable, platform, inventory, checksum, version, and dependency contracts.

### Modified Capabilities

- `cross-platform-devenv`: Adds the dedicated `outputs.release-chelisup` build interface on supported release hosts.

## Impact

The change affects Devenv modules, Nix release derivations, release workflow steps, and executable release tests. It does not change compiler semantics, runtime semantics, backends, generated code, or the public `chelisup` command interface.

The change adds no full Chelis toolchain release output. The current `outputs.chelisup` package remains the Nix installation output.
