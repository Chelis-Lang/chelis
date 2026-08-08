## ADDED Requirements

### Requirement: Devenv exposes a dedicated portable Linux toolchain release output
The repository MUST define `outputs.release-chelis` on `x86_64-linux`.

A contributor MUST be able to build the output with `devenv build --no-tui outputs.release-chelis`.

The output MUST build the compiler with the `smt` feature and the runtime static library from the committed `Cargo.nix` graph, the filtered repository source, `config.languages.rust.toolchainPackage`, and the pinned cvc5 tree.

The output MUST remain separate from the flake packages. It MUST NOT patch or replace `packages.chelis`.

The output MUST NOT evaluate the root flake, generate a crate graph during evaluation, use import-from-derivation, run a host Cargo build, or fetch a network source.

#### Scenario: A contributor builds the release output
- **WHEN** a contributor runs `devenv build --no-tui outputs.release-chelis` on `x86_64-linux`
- **THEN** Devenv builds the compiler and runtime from the committed graph with the configured Rust toolchain
- **AND** Devenv returns the staged Linux release tarball and its checksum sidecar

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
The output root MUST contain exactly these two regular files:

- `chelis-v<workspace-version>-linux-x86_64.tar.gz`
- `chelis-v<workspace-version>-linux-x86_64.tar.gz.sha256`

The tarball MUST unpack to one directory `chelis-v<workspace-version>-linux-x86_64` that contains exactly:

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

The asset name MUST match the name the `chelisup` installer constructs for the `linux-x86_64` slug.

#### Scenario: The output completes
- **WHEN** the derivation succeeds
- **THEN** its root contains only the tarball and its corresponding sidecar

#### Scenario: The tarball layout drifts
- **WHEN** the unpacked tree gains, loses, or renames an entry
- **THEN** the derivation fails its inventory check

#### Scenario: The installer requests the asset
- **WHEN** `chelisup install` builds the asset name for the workspace version on `linux-x86_64`
- **THEN** the constructed name equals the published tarball name

### Requirement: The portable compiler binary has no Nix runtime dependency
The staged `bin/chelis` MUST be a stripped x86-64 ELF executable.

Its ELF interpreter MUST be `/lib64/ld-linux-x86-64.so.2`. It MUST have no rpath and no runpath.

Its interpreter, dynamic section, and `NEEDED` entries MUST contain no `/nix/store` path.

Every `NEEDED` soname MUST be a glibc member library or `libgcc_s.so.1`. `libgcc_s.so.1` is permitted because the Rust standard library references the shared unwinder explicitly and every supported consumer system carries it in the base system.

The binary MUST link `libstdc++` and `libzstd` statically. `libstdc++.so.6` and `libzstd.so.1` MUST NOT appear in the `NEEDED` set.

The derivation MUST run its ELF portability checks after the last binary edit. The derivation MUST also prove that the rewritten binary runs against a glibc library set alone by executing it through an explicit dynamic loader.

#### Scenario: The derivation inspects the final binary
- **WHEN** the derivation reads the ELF metadata after the interpreter rewrite
- **THEN** the interpreter is `/lib64/ld-linux-x86-64.so.2`
- **AND** no `/nix/store` reference remains
- **AND** every `NEEDED` soname is a glibc member or `libgcc_s.so.1`

#### Scenario: A C++ or compression runtime dependency survives
- **WHEN** the final binary requires `libstdc++.so.6` or `libzstd.so.1`
- **THEN** the derivation fails before checksum creation

#### Scenario: The rewritten binary runs against glibc alone
- **WHEN** the derivation executes the rewritten binary through an explicit glibc dynamic loader
- **THEN** `--version` reports the workspace version

#### Scenario: A store path survives an edit
- **WHEN** the final binary metadata contains a `/nix/store` path
- **THEN** the derivation fails before checksum creation

### Requirement: The repository records the exact glibc floor of the portable binary
The repository MUST commit the expected glibc floor of the portable compiler binary as one contract value.

The derivation MUST compute the maximum `GLIBC_*` symbol version among the final binary's undefined dynamic symbols. The build MUST fail when the computed value differs from the recorded value in either direction.

The recorded value MUST NOT exceed the glibc version of the off-Nix consumption environment.

A change to the recorded value MUST land as a reviewed diff. The check MUST NOT read the value from an artifact that the build itself writes.

#### Scenario: The floor matches the record
- **WHEN** the derivation compares the computed floor with the committed value
- **THEN** the build succeeds

#### Scenario: A toolchain bump moves the floor
- **WHEN** a nixpkgs or Rust toolchain update changes the computed floor
- **THEN** the derivation fails and names both values
- **AND** the build succeeds only after a reviewed update of the recorded value

#### Scenario: The record outruns the consumption environment
- **WHEN** the recorded floor exceeds the glibc of the off-Nix consumption image
- **THEN** the release workflow contract test fails

### Requirement: The runtime static library keeps its glibc consumer contract
The staged `lib/libchelis_runtime.a` MUST be built for the `x86_64-unknown-linux-gnu` target. It MUST NOT be built for a musl target.

The archive MUST NOT define the libc allocator symbols `malloc`, `free`, `calloc`, or `realloc`.

An off-Nix glibc `gcc` MUST be able to link the shipped archive into a C program that runs.

HIP consumption of the shipped archive MUST remain covered by the documented HIP manual gate.

#### Scenario: The archive carries bundled libc objects
- **WHEN** the archive defines a libc allocator symbol
- **THEN** the archive check fails before checksum creation

#### Scenario: An off-Nix consumer links the archive
- **WHEN** the off-Nix consumption job compiles emitted C with `gcc` against the shipped archive
- **THEN** the program links and runs with exit status 0

#### Scenario: The archive target changes to musl
- **WHEN** the release output selects a musl target for the runtime crate
- **THEN** the static release-output contract test fails

### Requirement: The release artifacts pass behavior probes
Before the portability rewrite, the derivation MUST verify that the compiler reports `chelis <workspace-version>` from `--version`, returns success from `--help`, passes the release test fixture, and discharges the SMT verification fixture through cvc5.

The derivation MUST create the SHA-256 sidecar after all edits and MUST verify the sidecar against the final tarball.

The release workflow MUST run an off-Nix consumption job on the exact staged tarball: a clean Linux environment without Nix, with a glibc at or above the recorded floor, that unpacks the tarball, runs `bin/chelis --version`, verifies the SMT discharge, compiles emitted C against the shipped archive, and executes the result.

#### Scenario: The pre-rewrite probes pass
- **WHEN** the derivation runs the compiler probes before the interpreter rewrite
- **THEN** every probe exits with status 0 and the version matches the workspace manifest

#### Scenario: The shipped tarball works off the store
- **WHEN** the off-Nix consumption job runs the full probe sequence on the staged tarball
- **THEN** every step exits with status 0

#### Scenario: SMT support is absent from the shipped binary
- **WHEN** the shipped compiler does not discharge the SMT fixture through cvc5
- **THEN** the derivation or the off-Nix consumption job fails

### Requirement: The release workflow publishes the Linux toolchain from the Devenv output
The release workflow MUST build the Linux toolchain tarball with `devenv build --no-tui outputs.release-chelis` behind the pinned shared Devenv setup and private-ci authentication.

The workflow MUST NOT run `cargo build` to produce a published Linux artifact. The workflow MUST NOT build a published Linux artifact inside a distribution container.

The workflow MUST NOT publish a `-glibc2.31` Linux asset.

At a `v*` tag, the workflow MUST verify that the tag equals `v<workspace-version>` before publication.

The Linux job MUST stage the `chelisup.sh` bootstrap script as a release asset.

The publish job MUST depend on the Linux release job and the off-Nix consumption evidence. Manual tag dispatch and publication conditions MUST remain unchanged.

#### Scenario: A manual release builds the Linux toolchain
- **WHEN** the release workflow runs at a supported tag
- **THEN** the Linux tarball comes from the Devenv release output
- **AND** the publish job receives the tarball, its sidecar, and `chelisup.sh`

#### Scenario: A Cargo Linux release job returns
- **WHEN** the release workflow contains `cargo build` for a published Linux artifact
- **THEN** the release workflow contract test fails

#### Scenario: The tag disagrees with the workspace version
- **WHEN** the dispatched tag does not equal `v<workspace-version>`
- **THEN** the workflow fails before publication

#### Scenario: The off-Nix evidence is missing
- **WHEN** the publish job runs without a successful off-Nix consumption job
- **THEN** the release workflow contract test fails
