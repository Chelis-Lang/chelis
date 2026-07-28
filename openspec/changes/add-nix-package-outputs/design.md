## Context

The repository now tracks a reproducible Devenv shell, but it has no public Nix package outputs. `devenv.nix` installs development tools and does not package Chelis.

The release workflow builds three product artifact classes. These classes are the `chelis` compiler, the runtime static library with headers, and `chelisup`.

The shipped `chelis` binary enables the `smt` feature. Its `cvc5-sys` dependency normally fetches and builds cvc5 during a Cargo build.

Nix builds run in a network-isolated sandbox. The Nix build must therefore provide all Cargo and cvc5 sources before the build starts.

The first supported systems are `x86_64-linux` and `aarch64-darwin`. These systems match the native release and CI builders.

## Goals / Non-Goals

**Goals:**

- Provide locked `nix build` outputs for each product artifact class.
- Make the default package contain the complete Chelis toolchain layout.
- Keep the Nix artifact layout consistent with the release archive.
- Build an SMT-enabled compiler without network access in the build sandbox.
- Provide `nix run` apps for `chelis` and `chelisup`.
- Check each output on a native builder for every supported system.
- Detect drift between the shared Devenv and flake input pins.

**Non-Goals:**

- Package each internal Rust crate as a separate Nix output.
- Package the Python extension, documentation, or Reef source packages.
- Add a separate `chelis-std` output. The compiler bundle and Reef remain its owners.
- Replace `chelisup` as the installer and version router for release toolchains.
- Publish a binary cache or a Nix registry entry.
- Add `aarch64-linux` or an untested Darwin system in this phase.
- Change the GitHub release archive format.

## Decisions

### 1. Add a small root flake without a framework dependency

Track `flake.nix`, `flake.lock`, and focused package definitions under `nix/`. Use a small helper to generate outputs for the two supported systems.

This approach avoids a new flake framework dependency. It also keeps package evaluation separate from the standalone Devenv command surface.

The flake will export these attributes for each supported system:

- `packages.chelis`
- `packages.chelis-runtime`
- `packages.chelisup`
- `packages.default`, as an alias of `packages.chelis`
- `apps.chelis`
- `apps.chelisup`
- `apps.default`, as an alias of `apps.chelis`
- package and contract checks under `checks`

An alternative placed builds only in `devenv.nix.outputs`. That option supports `devenv build`, but it does not provide the requested public `nix build .#name` surface.

### 2. Match the existing product artifact boundaries

Build a private compiler derivation from `chelis-cli` with the `smt` feature. Build `chelis-runtime` and `chelisup` as separate derivations.

Compose `packages.chelis` from the compiler and runtime outputs. Its layout will contain:

- `bin/chelis`
- `lib/libchelis_runtime.a`
- the five public runtime headers under `include/`

`packages.chelis-runtime` will contain only the runtime library and public headers. `packages.chelisup` will contain `bin/chelisup`.

The `chelisup` package will not install a `bin/chelis` shim. This rule prevents a path collision with the real compiler package.

An alternative exposed all workspace crates. Most crates are implementation libraries, not stable product artifacts. That option creates an unsupported public package inventory.

### 3. Use the repository toolchain and Cargo lock

Use the Rust overlay compiler selected by `rust-toolchain.toml`. Use `Cargo.lock` as the complete Cargo dependency lock.

Use the Nixpkgs Rust package builder to vendor Cargo dependencies before the sandboxed build. Filter generated state and local build products from the source input.

Read the package version from the workspace manifest. Do not duplicate the Chelis version in Nix source.

### 4. Make the SMT build network-independent

Provide the cvc5 source and build tree as a separate fixed-source Nix derivation. Its version must match the `cvc5-sys` version in `Cargo.lock`.

Set `CVC5_DIR` to that derivation for the compiler build. The `cvc5-sys` build script will use its prebuilt path and must not clone a repository.

Run `.github/scripts/verify_release_smt.py` against the Nix compiler output. A compiler without active cvc5 discharge must fail the Nix check.

An alternative disabled the `smt` feature. That option produces a binary that differs from the shipped product and silently uses weaker proof behavior.

### 5. Keep flake and Devenv pins mechanically aligned

`flake.lock` will pin direct Nixpkgs and Rust overlay inputs. `devenv.lock` will remain the lock for the standalone Devenv interface.

Add a Python parity checker with positive and negative tests. It will compare the Nixpkgs source revision and Rust overlay revision in both locks.

The Nix check and the repository script suite will run this checker. A pin update in only one lock must fail with both mismatched node names.

A single lock file was considered. The standalone Devenv interface and public flake interface use different root graphs and lock file names.

### 6. Check output shape and executable behavior

Each supported system will define checks for these contracts:

- the default package equals `packages.chelis`
- every required file exists in each package
- no package contains an undeclared product executable
- `chelis --version` and `chelis --help` succeed
- the release pipe-stage fixture succeeds with the Nix compiler
- the SMT verifier confirms active cvc5 discharge
- `chelisup --help` succeeds
- each app points at the executable in its corresponding package
- the lock parity checker succeeds

CI will run the native check set on `x86_64-linux` and `aarch64-darwin`. The two native jobs form the authoritative completion oracle.

### 7. Keep installation ownership explicit

Nix is an additive source-build channel. `nix profile install` can install one Nix package, but it does not create a Chelis version store.

`chelisup` remains the installer and router for downloaded release toolchains. The Nix documentation must state this boundary.

## Risks / Trade-offs

- **[Cold cvc5 builds are expensive]** → Build cvc5 once as a separate derivation and reuse it across compiler checks.
- **[Two lock files can drift]** → Run the lock parity checker in Nix checks and the Python script suite.
- **[Source filtering can omit compile-time assets]** → Build from a clean Git source and run the compiler fixture inside the Nix check.
- **[Nix package contents can drift from releases]** → Check the release artifact paths and reuse the same runtime header list.
- **[One supported system can hide platform errors]** → Require native Linux and macOS check jobs before completion.
- **[The Nix compiler can lose SMT support]** → Run the existing SMT verifier against the final package binary.

## Migration Plan

1. Add the flake inputs and package definitions.
2. Generate and review `flake.lock`.
3. Add the lock parity checker and its tests.
4. Build each package on `x86_64-linux`.
5. Build each package on `aarch64-darwin`.
6. Add the native CI checks.
7. Document the Nix commands and installation boundary.
8. Remove the flake and CI checks to roll back this additive surface.

## Open Questions

No open question blocks this phase. A later change can add a system after a native builder proves its full check set.
