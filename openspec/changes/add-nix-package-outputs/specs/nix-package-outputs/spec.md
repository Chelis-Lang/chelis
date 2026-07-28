## ADDED Requirements

### Requirement: The repository exposes locked Nix package outputs
The repository MUST track `flake.nix` and `flake.lock`. The flake MUST export packages for `x86_64-linux` and `aarch64-darwin`.

For each supported system, the flake MUST export `chelis`, `chelis-runtime`, and `chelisup`. `packages.default` MUST resolve to the same derivation as `packages.chelis`.

The flake MUST NOT claim support for a system without the complete native check set.

#### Scenario: A user lists package outputs
- **WHEN** a user runs `nix flake show`
- **THEN** each supported system lists `chelis`, `chelis-runtime`, `chelisup`, and `default`

#### Scenario: A user builds the default package
- **WHEN** a user runs `nix build .`
- **THEN** Nix builds the same derivation as `nix build .#chelis`

#### Scenario: An unsupported system is inspected
- **WHEN** a user inspects package outputs for a system outside the supported list
- **THEN** the flake does not present that system as a supported Chelis package target

### Requirement: The default package contains the complete Chelis toolchain
`packages.chelis` MUST contain `bin/chelis`, `lib/libchelis_runtime.a`, and all public runtime headers.

The compiler MUST use the workspace version and the `smt` Cargo feature. The build MUST use `Cargo.lock` and MUST complete without network access in the build sandbox.

The package MUST preserve the artifact boundaries of the release toolchain. It MUST NOT add test-only binaries or internal workspace executables.

#### Scenario: The complete toolchain builds
- **WHEN** a native builder builds `packages.chelis`
- **THEN** the output contains the compiler, runtime static library, and five public runtime headers

#### Scenario: The compiler runs
- **WHEN** a check invokes `bin/chelis --version` and `bin/chelis --help`
- **THEN** both commands exit with status 0 and the version matches the workspace manifest

#### Scenario: SMT support is absent
- **WHEN** the final Nix compiler does not discharge the SMT verification fixture through cvc5
- **THEN** the package check fails

#### Scenario: A build attempts network access
- **WHEN** Cargo or the cvc5 build requests an undeclared network source
- **THEN** the sandboxed Nix build fails instead of fetching the source

### Requirement: The runtime package exposes the C runtime contract
`packages.chelis-runtime` MUST contain `lib/libchelis_runtime.a` and these headers under `include/`:

- `chelis_runtime.h`
- `chelis_runtime_dtype.h`
- `chelis_blas.h`
- `chelis_simd.h`
- `chelis_math.h`

The package MUST NOT contain the `chelis` or `chelisup` executable.

#### Scenario: A C consumer uses the runtime package
- **WHEN** a native check compiles and links a minimal C program against `packages.chelis-runtime`
- **THEN** the program links with the static library and includes the public headers

#### Scenario: A runtime artifact is missing
- **WHEN** the runtime library or one required header is absent
- **THEN** the runtime package check fails and names the missing path

#### Scenario: The runtime package gains a product executable
- **WHEN** `packages.chelis-runtime` contains `bin/chelis` or `bin/chelisup`
- **THEN** the runtime package check fails

### Requirement: The chelisup package exposes only the installer
`packages.chelisup` MUST contain `bin/chelisup`. It MUST NOT install a `bin/chelis` shim.

The package MUST use the workspace version and `Cargo.lock`.

#### Scenario: A user builds chelisup
- **WHEN** a user runs `nix build .#chelisup`
- **THEN** the result contains a runnable `bin/chelisup`

#### Scenario: The installer package collides with the compiler
- **WHEN** the `chelisup` output contains `bin/chelis`
- **THEN** the package check fails

### Requirement: The flake exposes runnable applications
For each supported system, the flake MUST export `apps.chelis` and `apps.chelisup`. `apps.default` MUST resolve to `apps.chelis`.

Each app MUST use the executable from its corresponding package output. The app MUST NOT invoke a host command outside the Nix store.

#### Scenario: A user runs the compiler app
- **WHEN** a user runs `nix run .#chelis -- --version`
- **THEN** Nix executes `packages.chelis/bin/chelis`

#### Scenario: A user runs the installer app
- **WHEN** a user runs `nix run .#chelisup -- --help`
- **THEN** Nix executes `packages.chelisup/bin/chelisup`

#### Scenario: An app points outside its package
- **WHEN** an app program path does not belong to its corresponding package
- **THEN** flake evaluation or the app contract check fails

### Requirement: Nix and Devenv share input revisions
`flake.lock` and `devenv.lock` MUST pin the same Nixpkgs source revision and Rust overlay revision.

The repository MUST provide a parity checker with positive and negative tests. A mismatch MUST fail before a package build can count as accepted.

#### Scenario: Shared pins match
- **WHEN** the parity checker reads both committed lock files
- **THEN** it exits with status 0 and reports matching shared input revisions

#### Scenario: One Nixpkgs pin changes
- **WHEN** a fixture changes the Nixpkgs revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the Nixpkgs nodes

#### Scenario: One Rust overlay pin changes
- **WHEN** a fixture changes the Rust overlay revision in only one lock file
- **THEN** the checker exits with a nonzero status and names the Rust overlay nodes

### Requirement: Native Nix checks protect every supported system
Each supported system MUST define checks for package construction, package contents, executable behavior, app paths, SMT activation, and lock parity.

CI MUST run the complete check set on native `x86_64-linux` and `aarch64-darwin` builders. A system MUST NOT count as supported from evaluation-only evidence.

The authoritative completion oracle MUST be the two successful native CI check jobs.

#### Scenario: Linux checks pass
- **WHEN** CI runs the complete flake check set on `x86_64-linux`
- **THEN** every package and contract check exits with status 0

#### Scenario: macOS checks pass
- **WHEN** CI runs the complete flake check set on `aarch64-darwin`
- **THEN** every package and contract check exits with status 0

#### Scenario: One package check fails
- **WHEN** any required package or contract check fails on a supported system
- **THEN** the native system job fails and the change does not satisfy the completion oracle

### Requirement: Documentation explains the Nix channel boundary
Contributor documentation MUST show the package names and commands for `nix build`, `nix run`, and flake checks.

The documentation MUST list the supported systems. It MUST state that Nix is an additive source-build channel and that `chelisup` remains the release installer.

The documentation MUST NOT claim separate Nix packages for internal crates, the Python extension, or `chelis-std`.

#### Scenario: A Nix user reads the build instructions
- **WHEN** the user reads the source-build documentation
- **THEN** the documentation identifies every public package, app, supported system, and check command

#### Scenario: A user needs version routing
- **WHEN** the user reads the Nix installation boundary
- **THEN** the documentation directs release toolchain routing to `chelisup`
