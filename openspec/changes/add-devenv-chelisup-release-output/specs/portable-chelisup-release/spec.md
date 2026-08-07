## ADDED Requirements

### Requirement: Devenv exposes a dedicated portable chelisup release output
The repository MUST define `outputs.release-chelisup` on `x86_64-linux` and `aarch64-darwin`.

A contributor MUST be able to build the native output with `devenv build --no-tui outputs.release-chelisup`.

The output MUST remain separate from the Nix-oriented `outputs.chelisup` package. It MUST NOT contain the Nix launcher or require that package at runtime.

The output MUST use the filtered repository source, the committed `Cargo.nix` graph, and `config.languages.rust.toolchainPackage`. It MUST select `workspaceMembers."chelisup"` with no optional features.

The output MUST NOT evaluate the root flake, generate a crate graph during evaluation, use import-from-derivation, or run a host Cargo build.

#### Scenario: A contributor builds the native release output
- **WHEN** a contributor runs `devenv build --no-tui outputs.release-chelisup` on a supported host
- **THEN** Devenv builds `chelisup` from the committed graph with the configured Rust toolchain
- **AND** Devenv returns the platform-specific portable files

#### Scenario: The output reuses the Nix installation package
- **WHEN** the release module selects `outputs.chelisup` or the root flake package
- **THEN** the static release-output contract test fails

#### Scenario: The output creates a graph at evaluation time
- **WHEN** the release helper invokes crate2nix generation or import-from-derivation
- **THEN** the static release-output contract test fails

### Requirement: The portable output has an exact platform inventory
On `x86_64-linux`, the output MUST contain exactly these two regular files:

- `chelisup-linux-x86_64`
- `chelisup-linux-x86_64.sha256`

On `aarch64-darwin`, the output MUST contain exactly these two regular files:

- `chelisup-darwin-arm64`
- `chelisup-darwin-arm64.sha256`

The executable MUST have execute permission. The sidecar MUST use lowercase SHA-256 text with two spaces before the relative executable name.

The release output MUST NOT advertise or produce a `darwin-x86_64` artifact.

The Rust host detector and the bootstrap script MUST expose only `linux-x86_64` and `darwin-arm64`. Any other host MUST fail before an asset request.

#### Scenario: The Linux output completes
- **WHEN** the Linux derivation succeeds
- **THEN** its root contains only the Linux executable and its corresponding sidecar

#### Scenario: The Darwin output completes
- **WHEN** the Darwin derivation succeeds
- **THEN** its root contains only the Darwin executable and its corresponding sidecar

#### Scenario: The output contains an extra path
- **WHEN** the output root contains another file, directory, or symbolic link
- **THEN** the derivation fails its inventory check

#### Scenario: A file uses an unsupported platform slug
- **WHEN** the output names `darwin-x86_64` or another unsupported platform
- **THEN** the release-output contract fails before publication

#### Scenario: The bootstrap runs on Intel macOS
- **WHEN** the bootstrap detects `Darwin/x86_64`
- **THEN** it reports an unsupported host
- **AND** it does not request a release asset

### Requirement: The portable executable matches the workspace version and architecture
The executable MUST report `chelisup <workspace-version>` from `--version`. It MUST return success from `--help`.

The Linux executable MUST be an x86-64 ELF executable. The Darwin executable MUST be an arm64 Mach-O executable.

The derivation MUST read the version from `[workspace.package]` in `Cargo.toml`. It MUST NOT define another release version.

#### Scenario: The executable matches the workspace
- **WHEN** the derivation runs the executable probes
- **THEN** `--version` reports the workspace version exactly
- **AND** `--help` exits successfully
- **AND** the binary architecture matches the output platform

#### Scenario: The executable reports another version
- **WHEN** `chelisup --version` differs from the workspace version
- **THEN** the derivation fails before checksum creation

#### Scenario: The executable has another architecture
- **WHEN** the binary metadata does not match the platform slug
- **THEN** the derivation fails before publication

### Requirement: The portable executable has no Nix runtime dependency
The Linux executable MUST use the `x86_64-unknown-linux-musl` target and the musl cross package set. Its ELF metadata MUST contain no interpreter segment and no dynamic `NEEDED` entry.

The Darwin executable MUST link only paths under `/usr/lib/` or `/System/Library/Frameworks/`. Its Mach-O metadata MUST contain no `/nix/store/` load path.

The Darwin derivation MUST replace the Nix `libiconv` load path with `/usr/lib/libiconv.2.dylib` before its executable probes.

#### Scenario: The Linux binary is static
- **WHEN** the Linux derivation inspects the final ELF file
- **THEN** the file has no interpreter segment
- **AND** the file has no dynamic library requirement

#### Scenario: The Darwin binary uses Apple libraries
- **WHEN** the Darwin derivation inspects the final Mach-O file
- **THEN** every load path names an Apple system library or framework
- **AND** the rewritten executable probes succeed

#### Scenario: A runtime path names the Nix store
- **WHEN** the final binary metadata contains a `/nix/store/` load path
- **THEN** the derivation fails before checksum creation

### Requirement: The output checksum describes the final executable
The derivation MUST create the SHA-256 sidecar after all binary edits. It MUST verify the sidecar against the final executable.

The checksum file MUST name only the executable in the same output root. GitHub Actions MUST upload both files from the same successful output.

#### Scenario: The checksum matches
- **WHEN** a consumer verifies the sidecar from the output root
- **THEN** SHA-256 verification succeeds for the final executable

#### Scenario: The executable changes after checksum creation
- **WHEN** the executable bytes differ from the sidecar digest
- **THEN** the derivation or workflow verification fails

### Requirement: The release workflow publishes the Devenv output
The release workflow MUST define one `build-chelisup-release` matrix job for `linux-x86_64` and `darwin-arm64`.

The job MUST invoke the pinned shared Devenv setup before it builds the output. It MUST authenticate Nix for the private ci input before the first project Devenv command.

The job MUST run `devenv build --no-tui outputs.release-chelisup`. It MUST upload only the exact executable and sidecar for its matrix platform.

No release workflow job MUST run `cargo build` for package `chelisup`. The full toolchain jobs MUST retain their current compiler, runtime, and cvc5 evidence.

The publish job MUST depend on the release-output matrix. It MUST retain the current manual tag dispatch and publication conditions.

#### Scenario: A manual release builds portable installers
- **WHEN** the release workflow runs at a supported tag
- **THEN** the matrix builds both Devenv release outputs
- **AND** the publish job receives both executables and both sidecars

#### Scenario: A full toolchain job builds chelisup directly
- **WHEN** the release workflow contains `cargo build` for package `chelisup`
- **THEN** the release workflow contract test fails

#### Scenario: The matrix omits private input authentication
- **WHEN** the matrix reaches a project Devenv command without private ci authentication
- **THEN** the release workflow contract test fails

#### Scenario: Release publication policy changes
- **WHEN** the workflow publishes outside a manual dispatch at a `v*` tag
- **THEN** the current release workflow policy test fails
