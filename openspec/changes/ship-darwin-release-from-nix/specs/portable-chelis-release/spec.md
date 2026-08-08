## MODIFIED Requirements

### Requirement: Devenv exposes a dedicated portable toolchain release output
The repository MUST define `outputs.release-chelis` on `x86_64-linux` and `aarch64-darwin`.

A contributor MUST be able to build the native output with `devenv build --no-tui outputs.release-chelis`.

The output MUST build the compiler with the `smt` feature and the runtime static library from the committed `Cargo.nix` graph, the filtered repository source, `config.languages.rust.toolchainPackage`, and the pinned cvc5 tree.

The output MUST remain separate from the flake packages. It MUST NOT patch or replace `packages.chelis`.

The output MUST NOT evaluate the root flake, generate a crate graph during evaluation, use import-from-derivation, run a host Cargo build, or fetch a network source.

#### Scenario: A contributor builds the release output
- **WHEN** a contributor runs `devenv build --no-tui outputs.release-chelis` on a supported host
- **THEN** Devenv builds the compiler and runtime from the committed graph with the configured Rust toolchain
- **AND** Devenv returns the staged platform release tarball and its checksum sidecar

#### Scenario: The output reuses a flake package binary
- **WHEN** the release module copies its compiler from `packages.chelis` instead of the release crate build
- **THEN** the static release-output contract test fails

#### Scenario: The output generates a graph at evaluation time
- **WHEN** the release helper invokes crate2nix generation or import-from-derivation
- **THEN** the static release-output contract test fails

#### Scenario: A build attempts network access
- **WHEN** any crate or the cvc5 stack requests an undeclared network source
- **THEN** the sandboxed build fails instead of fetching the source

### Requirement: The portable tarball has an exact inventory
On `x86_64-linux`, the output root MUST contain exactly `chelis-v<workspace-version>-linux-x86_64.tar.gz` and its `.sha256` sidecar. On `aarch64-darwin`, it MUST contain exactly `chelis-v<workspace-version>-darwin-arm64.tar.gz` and its `.sha256` sidecar.

Each tarball MUST unpack to one directory named after its asset stem that contains exactly:

- `bin/chelis`
- `lib/libchelis_runtime.a`
- `include/chelis_runtime.h`
- `include/chelis_runtime_dtype.h`
- `include/chelis_blas.h`
- `include/chelis_simd.h`
- `include/chelis_math.h`
- `README.md`
- `LICENSE`

The version MUST come from `[workspace.package]` in `Cargo.toml`. The derivation MUST NOT define another release version.

Each asset name MUST match the name the `chelisup` installer constructs for its platform slug.

#### Scenario: The output completes
- **WHEN** the derivation succeeds on either platform
- **THEN** its root contains only the platform tarball and its corresponding sidecar

#### Scenario: The tarball layout drifts
- **WHEN** the unpacked tree gains, loses, or renames an entry
- **THEN** the derivation fails its inventory check

#### Scenario: The installer requests the asset
- **WHEN** `chelisup install` builds the asset name for the workspace version on either supported slug
- **THEN** the constructed name equals the published tarball name

### Requirement: The portable Darwin compiler binary has no Nix runtime dependency
The staged Darwin `bin/chelis` MUST be an arm64 Mach-O executable.

Every load command MUST name a path under `/usr/lib/` or `/System/Library/Frameworks/`. No load command may contain `/nix/store`.

The binary MUST link `libzstd` statically. The derivation MUST rewrite the `libiconv` load path to `/usr/lib/libiconv.2.dylib` before its final checks.

The derivation MUST run the behavior probes on the rewritten binary: `--version` reporting the workspace version, `--help`, the release test fixture, and the cvc5 discharge verification.

#### Scenario: The derivation inspects the final binary
- **WHEN** the derivation reads the Mach-O load commands after the rewrite
- **THEN** every load path is an Apple system library or framework
- **AND** no `/nix/store` reference remains

#### Scenario: A store dylib survives
- **WHEN** the final binary retains a `/nix/store` load path
- **THEN** the derivation fails before checksum creation

#### Scenario: The rewritten binary runs
- **WHEN** the derivation executes the rewritten binary
- **THEN** every behavior probe exits with status 0

### Requirement: The release workflow publishes the toolchain from the Devenv output
The release workflow MUST build both platform toolchain tarballs with `devenv build --no-tui outputs.release-chelis` behind the pinned shared Devenv setup and private-ci authentication, as a matrix over `linux-x86_64` and `darwin-arm64`.

The workflow MUST NOT run `cargo build` to produce any published artifact. The workflow MUST NOT build a published artifact inside a distribution container.

The workflow MUST NOT publish a `-glibc2.31` Linux asset.

At a `v*` tag, the workflow MUST verify that the tag equals `v<workspace-version>` before publication.

The Linux matrix leg MUST stage the `chelisup.sh` bootstrap script as a release asset.

Each platform MUST have an off-Nix consumption job on the exact staged tarball. The Linux job runs in a clean container without Nix with a glibc at or above the recorded floor. The Darwin job runs on a stock macOS runner and MUST compile emitted C against the shipped archive with the system clang and Accelerate.

The publish job MUST depend on the matrix and both consumption jobs. Manual tag dispatch and publication conditions MUST remain unchanged.

#### Scenario: A manual release builds both toolchains
- **WHEN** the release workflow runs at a supported tag
- **THEN** both tarballs come from the Devenv release output
- **AND** the publish job receives both tarballs, both sidecars, and `chelisup.sh`

#### Scenario: A Cargo release job returns
- **WHEN** the release workflow contains `cargo build` for a published artifact
- **THEN** the release workflow contract test fails

#### Scenario: The Darwin tarball works off the store
- **WHEN** the Darwin consumption job runs the probe sequence and the Accelerate smoke on the staged tarball
- **THEN** every step exits with status 0

#### Scenario: The off-Nix evidence is missing
- **WHEN** the publish job runs without both successful consumption jobs
- **THEN** the release workflow contract test fails
